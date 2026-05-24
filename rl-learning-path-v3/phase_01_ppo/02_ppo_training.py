"""
Phase 1, Script 2: PPO Alignment Training
==========================================

This script runs PPO (Proximal Policy Optimization) to align a language model
using the reward model trained in 01_train_reward_model.py.

The pipeline:
    1. Load the policy model (SFT or pretrained) and reference model
    2. Load the trained reward model
    3. Run PPO: generate -> score -> compute advantage -> update policy
    4. Generate sample outputs to inspect quality

Run:
    python 02_ppo_training.py \\
        --reward_model_path ./outputs/reward_model \\
        --max_samples 500

What to observe:
    - Mean reward should increase over training steps
    - KL divergence should stay moderate (not explode)
    - Sample outputs at the end: are they more helpful than before?
"""

import sys
import argparse
from pathlib import Path

# ── Add project root to path so we can import shared utilities ──────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import yaml
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
from transformers import AutoTokenizer, pipeline

from shared.model_utils import DEFAULT_MODEL, get_lora_config
from shared.data_utils import load_hh_rlhf
from shared.eval_utils import generate_responses, compare_outputs

# ╭──────────────────────────────────────────────────────────────────────────╮
# │ CONFIG — Adjust these values to experiment                              │
# ╰──────────────────────────────────────────────────────────────────────────╯
CONFIG = {
    "model_name": DEFAULT_MODEL,               # Policy model
    "max_length": 256,                         # Max prompt length
    "max_new_tokens": 128,                     # Max tokens to generate per response
    "ppo_epochs": 4,                           # PPO optimization epochs per batch
    "batch_size": 16,                          # Rollout batch size
    "mini_batch_size": 4,                      # Mini-batch for PPO updates
    "learning_rate": 1e-5,                     # Policy learning rate
    "kl_penalty": "kl",                        # KL penalty type
    "init_kl_coef": 0.05,                      # KL coefficient (beta)
    "target_kl": 6.0,                          # Target KL for adaptive control
    "gamma": 1.0,                              # Discount factor
    "lam": 0.95,                               # GAE lambda
    "temperature": 0.7,                        # Sampling temperature
    "num_sample_prompts": 5,                   # Number of prompts for final demo
}

# Prompts for comparing outputs before/after PPO
EVAL_PROMPTS = [
    "\n\nHuman: How can I improve my writing skills?\n\nAssistant:",
    "\n\nHuman: What are some healthy breakfast ideas?\n\nAssistant:",
    "\n\nHuman: Can you explain what machine learning is in simple terms?\n\nAssistant:",
    "\n\nHuman: How do I deal with stress at work?\n\nAssistant:",
    "\n\nHuman: What is the best way to learn a new language?\n\nAssistant:",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Run PPO alignment training")
    parser.add_argument(
        "--reward_model_path", type=str, default="./outputs/reward_model",
        help="Path to the trained reward model (default: ./outputs/reward_model)"
    )
    parser.add_argument(
        "--max_samples", type=int, default=500,
        help="Maximum number of training prompts (default: 500)"
    )
    parser.add_argument(
        "--output_dir", type=str, default="./outputs/ppo_model",
        help="Directory to save the PPO-trained model (default: ./outputs/ppo_model)"
    )
    return parser.parse_args()


def extract_prompts_from_hh_rlhf(dataset, max_length, tokenizer):
    """Extract the prompt (Human turn) from HH-RLHF examples."""
    prompts = []
    for example in dataset:
        text = example["chosen"]
        # Extract up to the last Assistant turn
        idx = text.rfind("\n\nAssistant:")
        if idx != -1:
            prompt = text[:idx + len("\n\nAssistant:")]
        else:
            prompt = text
        prompts.append(prompt.strip())
    return prompts


def build_prompt_dataset(prompts, tokenizer, max_length):
    """Tokenize prompts and return as list of tensors for PPO."""
    tokenized = []
    for prompt in prompts:
        encoded = tokenizer(
            prompt,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        tokenized.append(encoded["input_ids"].squeeze(0))
    return tokenized


def main():
    args = parse_args()
    print(f"\n{'='*60}")
    print(f"  PPO Alignment Training")
    print(f"  Policy Model:  {CONFIG['model_name']}")
    print(f"  Reward Model:  {args.reward_model_path}")
    print(f"  Samples:       {args.max_samples}")
    print(f"  KL coef (β):   {CONFIG['init_kl_coef']}")
    print(f"  Output:        {args.output_dir}")
    print(f"{'='*60}\n")

    # ── Step 1: Load dataset and extract prompts ────────────────────────────
    print("[1/5] Loading dataset and extracting prompts...")
    dataset = load_hh_rlhf(split="train", max_samples=args.max_samples)

    tokenizer = AutoTokenizer.from_pretrained(
        CONFIG["model_name"], trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    prompts = extract_prompts_from_hh_rlhf(dataset, CONFIG["max_length"], tokenizer)
    print(f"  Extracted {len(prompts)} prompts")

    # ── Step 2: Load policy model with value head ───────────────────────────
    print("\n[2/5] Loading policy model with value head...")
    lora_config = get_lora_config(r=16, lora_alpha=32)

    model = AutoModelForCausalLMWithValueHead.from_pretrained(
        CONFIG["model_name"],
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        peft_config=lora_config,
    )

    # ── Step 3: Load reward model as a scoring pipeline ─────────────────────
    print("\n[3/5] Loading reward model for scoring...")
    reward_pipe = pipeline(
        "text-classification",
        model=args.reward_model_path,
        tokenizer=tokenizer,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        function_to_apply="none",  # Return raw logits as scores
        truncation=True,
        max_length=CONFIG["max_length"],
    )

    # ── Step 4: Configure and run PPO ───────────────────────────────────────
    print("\n[4/5] Running PPO training loop...")
    ppo_config = PPOConfig(
        learning_rate=CONFIG["learning_rate"],
        batch_size=CONFIG["batch_size"],
        mini_batch_size=CONFIG["mini_batch_size"],
        ppo_epochs=CONFIG["ppo_epochs"],
        init_kl_coef=CONFIG["init_kl_coef"],
        target_kl=CONFIG["target_kl"],
        gamma=CONFIG["gamma"],
        lam=CONFIG["lam"],
        log_with=None,  # Set to "wandb" for W&B logging
    )

    ppo_trainer = PPOTrainer(
        config=ppo_config,
        model=model,
        tokenizer=tokenizer,
    )

    # Tokenize prompts
    prompt_tensors = build_prompt_dataset(prompts, tokenizer, CONFIG["max_length"])

    # Store baseline outputs before training
    print("  Generating baseline outputs (before PPO)...")
    baseline_model = ppo_trainer.model
    baseline_responses = generate_responses(
        baseline_model.pretrained_model,
        tokenizer,
        EVAL_PROMPTS[:CONFIG["num_sample_prompts"]],
        max_new_tokens=CONFIG["max_new_tokens"],
        temperature=CONFIG["temperature"],
    )

    # Training loop
    generation_kwargs = {
        "max_new_tokens": CONFIG["max_new_tokens"],
        "temperature": CONFIG["temperature"],
        "do_sample": True,
        "pad_token_id": tokenizer.pad_token_id,
    }

    all_rewards = []
    num_batches = len(prompt_tensors) // CONFIG["batch_size"]

    for batch_idx in range(num_batches):
        start = batch_idx * CONFIG["batch_size"]
        end = start + CONFIG["batch_size"]
        batch_prompts = prompt_tensors[start:end]

        # Generate responses
        response_tensors = ppo_trainer.generate(
            batch_prompts,
            return_prompt=False,
            **generation_kwargs,
        )

        # Decode for reward scoring
        batch_texts = []
        for prompt_t, response_t in zip(batch_prompts, response_tensors):
            full_text = tokenizer.decode(
                torch.cat([prompt_t, response_t]),
                skip_special_tokens=True,
            )
            batch_texts.append(full_text)

        # Score with reward model
        pipe_outputs = reward_pipe(batch_texts)
        rewards = [torch.tensor(output["score"], dtype=torch.float32) for output in pipe_outputs]

        # Run PPO step
        stats = ppo_trainer.step(batch_prompts, response_tensors, rewards)

        # Log statistics
        batch_mean_reward = sum(r.item() for r in rewards) / len(rewards)
        all_rewards.append(batch_mean_reward)
        kl_val = stats.get("objective/kl", 0.0)

        print(
            f"  Batch {batch_idx + 1}/{num_batches} | "
            f"Mean Reward: {batch_mean_reward:.4f} | "
            f"KL: {kl_val:.4f}"
        )

    # ── Step 5: Generate and compare outputs ────────────────────────────────
    print(f"\n[5/5] Generating post-PPO outputs and comparing...")

    ppo_responses = generate_responses(
        ppo_trainer.model.pretrained_model,
        tokenizer,
        EVAL_PROMPTS[:CONFIG["num_sample_prompts"]],
        max_new_tokens=CONFIG["max_new_tokens"],
        temperature=CONFIG["temperature"],
    )

    compare_outputs(
        EVAL_PROMPTS[:CONFIG["num_sample_prompts"]],
        baseline_responses,
        ppo_responses,
        label_a="Before PPO",
        label_b="After PPO",
    )

    # Print reward trajectory
    print(f"\n{'='*60}")
    print(f"  Reward Trajectory")
    print(f"{'='*60}")
    for i, r in enumerate(all_rewards):
        bar = "█" * int(max(0, (r + 2) * 10))  # Simple ASCII bar
        print(f"  Batch {i + 1:3d}: {r:+.4f} {bar}")

    if len(all_rewards) >= 2:
        improvement = all_rewards[-1] - all_rewards[0]
        print(f"\n  Total reward change: {improvement:+.4f}")
        print(f"  First batch avg:     {all_rewards[0]:.4f}")
        print(f"  Last batch avg:      {all_rewards[-1]:.4f}")

    # Save the model
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    ppo_trainer.save_pretrained(str(output_path))
    tokenizer.save_pretrained(str(output_path))
    print(f"\n  Model saved to: {args.output_dir}")
    print()


if __name__ == "__main__":
    main()
