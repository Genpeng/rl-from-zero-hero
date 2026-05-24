"""
Phase 1, Script 3: KL Coefficient Ablation Experiment
=====================================================

This script runs PPO with different KL penalty coefficients (beta) and compares
the results. It demonstrates the fundamental tradeoff in RLHF:

    Higher beta -> stays close to pretrained model -> lower reward
    Lower beta  -> drifts freely -> higher reward but risk of reward hacking

The experiment runs three configurations:
    - beta = 0.0:  No KL constraint (expect reward hacking)
    - beta = 0.05: Light constraint (balanced)
    - beta = 0.2:  Strong constraint (conservative)

Run:
    python 03_ppo_experiment.py \\
        --reward_model_path ./outputs/reward_model \\
        --max_samples 200

What to observe:
    - Compare mean rewards across beta values
    - Look at sample outputs: does beta=0.0 produce degenerate text?
    - Is beta=0.2 too conservative (reward barely increases)?
"""

import sys
import argparse
from pathlib import Path

# ── Add project root to path so we can import shared utilities ──────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
from transformers import AutoTokenizer, pipeline

from shared.model_utils import DEFAULT_MODEL, get_lora_config
from shared.data_utils import load_hh_rlhf
from shared.eval_utils import generate_responses, compare_outputs

# ╭──────────────────────────────────────────────────────────────────────────╮
# │ CONFIG — Adjust these values to experiment                              │
# ╰──────────────────────────────────────────────────────────────────────────╯
CONFIG = {
    "model_name": DEFAULT_MODEL,
    "max_length": 256,
    "max_new_tokens": 128,
    "ppo_epochs": 4,
    "batch_size": 8,
    "mini_batch_size": 4,
    "learning_rate": 1e-5,
    "gamma": 1.0,
    "lam": 0.95,
    "temperature": 0.7,
    # KL coefficients to compare
    "kl_coefs": [0.0, 0.05, 0.2],
}

# Fixed prompts for qualitative comparison across runs
EVAL_PROMPTS = [
    "\n\nHuman: How can I be more productive at work?\n\nAssistant:",
    "\n\nHuman: What should I know about investing money?\n\nAssistant:",
    "\n\nHuman: Can you explain climate change simply?\n\nAssistant:",
]


def parse_args():
    parser = argparse.ArgumentParser(description="PPO KL ablation experiment")
    parser.add_argument(
        "--reward_model_path", type=str, default="./outputs/reward_model",
        help="Path to the trained reward model (default: ./outputs/reward_model)"
    )
    parser.add_argument(
        "--max_samples", type=int, default=200,
        help="Maximum number of training prompts per run (default: 200)"
    )
    return parser.parse_args()


def extract_prompts(dataset, tokenizer, max_length):
    """Extract prompts from HH-RLHF dataset."""
    prompts = []
    for example in dataset:
        text = example["chosen"]
        idx = text.rfind("\n\nAssistant:")
        if idx != -1:
            prompt = text[:idx + len("\n\nAssistant:")]
        else:
            prompt = text
        prompts.append(prompt.strip())
    return prompts


def tokenize_prompts(prompts, tokenizer, max_length):
    """Tokenize prompts into tensors for PPO."""
    tokenized = []
    for prompt in prompts:
        encoded = tokenizer(
            prompt, truncation=True, max_length=max_length, return_tensors="pt"
        )
        tokenized.append(encoded["input_ids"].squeeze(0))
    return tokenized


def run_ppo_with_kl(kl_coef, prompt_tensors, tokenizer, reward_pipe, config):
    """Run a single PPO training run with a given KL coefficient."""
    print(f"\n{'─'*60}")
    print(f"  Running PPO with beta = {kl_coef}")
    print(f"{'─'*60}")

    # Fresh model for each run (so comparisons are fair)
    lora_config = get_lora_config(r=16, lora_alpha=32)
    model = AutoModelForCausalLMWithValueHead.from_pretrained(
        config["model_name"],
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        peft_config=lora_config,
    )

    ppo_config = PPOConfig(
        learning_rate=config["learning_rate"],
        batch_size=config["batch_size"],
        mini_batch_size=config["mini_batch_size"],
        ppo_epochs=config["ppo_epochs"],
        init_kl_coef=kl_coef,
        target_kl=None if kl_coef == 0.0 else 6.0,
        gamma=config["gamma"],
        lam=config["lam"],
        log_with=None,
    )

    ppo_trainer = PPOTrainer(
        config=ppo_config,
        model=model,
        tokenizer=tokenizer,
    )

    generation_kwargs = {
        "max_new_tokens": config["max_new_tokens"],
        "temperature": config["temperature"],
        "do_sample": True,
        "pad_token_id": tokenizer.pad_token_id,
    }

    all_rewards = []
    all_kls = []
    num_batches = len(prompt_tensors) // config["batch_size"]

    for batch_idx in range(num_batches):
        start = batch_idx * config["batch_size"]
        end = start + config["batch_size"]
        batch_prompts = prompt_tensors[start:end]

        # Generate responses
        response_tensors = ppo_trainer.generate(
            batch_prompts,
            return_prompt=False,
            **generation_kwargs,
        )

        # Decode and score
        batch_texts = []
        for prompt_t, response_t in zip(batch_prompts, response_tensors):
            full_text = tokenizer.decode(
                torch.cat([prompt_t, response_t]),
                skip_special_tokens=True,
            )
            batch_texts.append(full_text)

        pipe_outputs = reward_pipe(batch_texts)
        rewards = [torch.tensor(o["score"], dtype=torch.float32) for o in pipe_outputs]

        # PPO step
        stats = ppo_trainer.step(batch_prompts, response_tensors, rewards)

        batch_reward = sum(r.item() for r in rewards) / len(rewards)
        batch_kl = stats.get("objective/kl", 0.0)
        all_rewards.append(batch_reward)
        all_kls.append(batch_kl)

        print(
            f"    Batch {batch_idx + 1}/{num_batches} | "
            f"Reward: {batch_reward:.4f} | KL: {batch_kl:.4f}"
        )

    # Generate sample outputs with trained model
    sample_responses = generate_responses(
        ppo_trainer.model.pretrained_model,
        tokenizer,
        EVAL_PROMPTS,
        max_new_tokens=config["max_new_tokens"],
        temperature=config["temperature"],
    )

    # Collect results
    result = {
        "kl_coef": kl_coef,
        "mean_reward": sum(all_rewards) / len(all_rewards) if all_rewards else 0.0,
        "final_reward": all_rewards[-1] if all_rewards else 0.0,
        "mean_kl": sum(all_kls) / len(all_kls) if all_kls else 0.0,
        "final_kl": all_kls[-1] if all_kls else 0.0,
        "reward_trajectory": all_rewards,
        "sample_responses": sample_responses,
    }

    # Clean up to free GPU memory for next run
    del ppo_trainer, model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result


def print_comparison_table(results):
    """Print a formatted comparison table of all runs."""
    print(f"\n{'='*80}")
    print(f"  KL Ablation — Results Comparison")
    print(f"{'='*80}")
    print(f"  {'Beta':>8} | {'Mean Reward':>12} | {'Final Reward':>13} | {'Mean KL':>10} | {'Final KL':>10}")
    print(f"  {'─'*8}-+-{'─'*12}-+-{'─'*13}-+-{'─'*10}-+-{'─'*10}")

    for r in results:
        print(
            f"  {r['kl_coef']:>8.2f} | "
            f"{r['mean_reward']:>12.4f} | "
            f"{r['final_reward']:>13.4f} | "
            f"{r['mean_kl']:>10.4f} | "
            f"{r['final_kl']:>10.4f}"
        )

    # Highlight key observations
    rewards = [r["mean_reward"] for r in results]
    kls = [r["mean_kl"] for r in results]
    best_reward_idx = rewards.index(max(rewards))
    lowest_kl_idx = kls.index(min(kls))

    print(f"\n  Highest reward: beta = {results[best_reward_idx]['kl_coef']} "
          f"(mean reward = {results[best_reward_idx]['mean_reward']:.4f})")
    print(f"  Lowest KL:      beta = {results[lowest_kl_idx]['kl_coef']} "
          f"(mean KL = {results[lowest_kl_idx]['mean_kl']:.4f})")


def print_sample_comparison(results):
    """Print sample outputs from each beta configuration."""
    print(f"\n{'='*80}")
    print(f"  Sample Outputs — Qualitative Comparison")
    print(f"{'='*80}")

    for i, prompt in enumerate(EVAL_PROMPTS):
        print(f"\n  PROMPT: {prompt.strip()[:80]}...")
        for r in results:
            print(f"\n    [beta={r['kl_coef']}]:")
            response = r["sample_responses"][i]
            # Show first 200 chars to keep output manageable
            truncated = response[:200] + ("..." if len(response) > 200 else "")
            for line in truncated.split("\n"):
                print(f"      {line}")
        print(f"  {'─'*70}")


def main():
    args = parse_args()
    print(f"\n{'='*60}")
    print(f"  PPO KL Ablation Experiment")
    print(f"  Model:       {CONFIG['model_name']}")
    print(f"  Reward:      {args.reward_model_path}")
    print(f"  Samples:     {args.max_samples}")
    print(f"  KL values:   {CONFIG['kl_coefs']}")
    print(f"{'='*60}")

    # ── Load dataset and reward model (shared across runs) ──────────────────
    print("\n[1/3] Loading dataset and reward model...")
    dataset = load_hh_rlhf(split="train", max_samples=args.max_samples)

    tokenizer = AutoTokenizer.from_pretrained(
        CONFIG["model_name"], trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    prompts = extract_prompts(dataset, tokenizer, CONFIG["max_length"])
    prompt_tensors = tokenize_prompts(prompts, tokenizer, CONFIG["max_length"])
    print(f"  Prepared {len(prompt_tensors)} prompts")

    reward_pipe = pipeline(
        "text-classification",
        model=args.reward_model_path,
        tokenizer=tokenizer,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        function_to_apply="none",
        truncation=True,
        max_length=CONFIG["max_length"],
    )

    # ── Run PPO with each KL coefficient ────────────────────────────────────
    print("\n[2/3] Running PPO with different KL coefficients...")
    results = []
    for kl_coef in CONFIG["kl_coefs"]:
        result = run_ppo_with_kl(
            kl_coef=kl_coef,
            prompt_tensors=prompt_tensors,
            tokenizer=tokenizer,
            reward_pipe=reward_pipe,
            config=CONFIG,
        )
        results.append(result)

    # ── Print comparison ────────────────────────────────────────────────────
    print("\n[3/3] Comparing results...")
    print_comparison_table(results)
    print_sample_comparison(results)

    # ── Summary ─────────────────────────────────────────────────────────────
    print(f"\n{'='*80}")
    print(f"  Experiment Summary")
    print(f"{'='*80}")
    print(f"""
  Key takeaways to look for:

  1. beta=0.0 likely has the highest reward but inspect the outputs carefully.
     Are they repetitive? Do they exploit patterns the reward model favors?

  2. beta=0.05 should give a reasonable balance — decent rewards while
     maintaining output quality close to the pretrained model.

  3. beta=0.2 is conservative. Rewards may not improve much, but the outputs
     should stay very close to the original model's style.

  If beta=0.0 outputs look normal, your reward model might be robust to hacking,
  or training was too short for reward hacking to emerge. Try increasing
  --max_samples to see if degenerate patterns appear with longer training.
""")


if __name__ == "__main__":
    main()
