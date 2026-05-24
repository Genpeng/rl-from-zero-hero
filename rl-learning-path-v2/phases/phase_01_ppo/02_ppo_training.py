"""
Phase 1, Script 2: PPO Training Loop
======================================
This is the core RLHF script. It uses 4 models:
  1. Policy Model     — the LLM being aligned (Qwen2.5-1.5B)
  2. Reference Model  — frozen copy of the SFT model (for KL penalty)
  3. Reward Model     — trained in Script 1 (scores responses)
  4. Value Model      — estimates expected reward (critic, initialized from policy)

What you'll learn:
- How the PPO training loop works step by step
- What each metric means (reward, KL, policy loss)
- How KL coefficient affects training dynamics

Usage:
    python phases/phase_01_ppo/02_ppo_training.py
"""

import sys
import os
import yaml
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_dataset
from trl import PPOConfig, PPOTrainer, AutoModelForCausalLMWithValueHead

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from shared.model_utils import MODEL_NAME, print_model_info


def load_config():
    config_path = os.path.join(os.path.dirname(__file__), "configs", "ppo_config.yaml")
    with open(config_path) as f:
        return yaml.safe_load(f)


def prepare_prompts(config):
    ds = load_dataset(config["dataset"]["name"], split="train")
    ds = ds.shuffle(seed=42).select(range(min(config["dataset"]["max_samples"], len(ds))))

    prompts = []
    for example in ds:
        prompt = example["prompt"]
        if len(prompt) > 0:
            prompts.append(prompt)
    return prompts


def main():
    config = load_config()
    ppo_cfg = config["ppo"]

    # --- Step 1: Load the policy model (with value head for PPO) ---
    print("\n📦 Step 1/4: Loading POLICY model (with value head)...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLMWithValueHead.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    print_model_info(model.pretrained_model, "Policy Model (Qwen2.5-1.5B)")

    # --- Step 2: The reference model is handled internally by PPOTrainer ---
    print("📦 Step 2/4: Reference model will be created automatically by PPOTrainer.")
    print("   (It's a frozen copy of the policy, used to compute KL penalty)\n")

    # --- Step 3: Load the reward model ---
    print("📦 Step 3/4: Loading REWARD model...")
    rm_path = os.path.join(config["training"]["output_dir"], "reward_model")
    if not os.path.exists(rm_path):
        print(f"  ⚠️  No trained reward model found at {rm_path}")
        print(f"  Using pretrained: {config['reward_model']['name']}")
        rm_path = config["reward_model"]["name"]

    reward_model = AutoModelForSequenceClassification.from_pretrained(
        rm_path,
        num_labels=1,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    reward_tokenizer = AutoTokenizer.from_pretrained(rm_path)
    if reward_tokenizer.pad_token is None:
        reward_tokenizer.pad_token = reward_tokenizer.eos_token
    print_model_info(reward_model, "Reward Model")

    # --- Step 4: Configure PPO ---
    print("📦 Step 4/4: Configuring PPO trainer...\n")

    ppo_config = PPOConfig(
        learning_rate=ppo_cfg["learning_rate"],
        batch_size=ppo_cfg["batch_size"],
        mini_batch_size=ppo_cfg["mini_batch_size"],
        ppo_epochs=ppo_cfg["ppo_epochs"],
        init_kl_coef=ppo_cfg["init_kl_coef"],
        kl_penalty=ppo_cfg["kl_penalty"],
        clip_range=ppo_cfg["clip_range"],
        gamma=ppo_cfg["gamma"],
        lam=ppo_cfg["lam"],
        log_with="tensorboard",
        project_kwargs={"logging_dir": os.path.join(config["training"]["output_dir"], "logs")},
    )

    prompts = prepare_prompts(config)
    print(f"Prepared {len(prompts)} training prompts\n")

    trainer = PPOTrainer(
        config=ppo_config,
        model=model,
        tokenizer=tokenizer,
        dataset=None,
    )

    # --- Training Loop ---
    print("🚀 Starting PPO training loop...")
    print("=" * 60)
    print(f"  KL coefficient:  {ppo_cfg['init_kl_coef']}")
    print(f"  Clip range:      {ppo_cfg['clip_range']}")
    print(f"  Batch size:      {ppo_cfg['batch_size']}")
    print(f"  Learning rate:   {ppo_cfg['learning_rate']}")
    print("=" * 60)
    print("\nMetrics to watch:")
    print("  reward/mean  ↑  (should increase — model learning to generate preferred outputs)")
    print("  objective/kl ↔  (should stay moderate — too high means reward hacking)")
    print("  ppo/loss     ↓  (should decrease — policy improving)")
    print()

    gen_kwargs = {
        "max_new_tokens": ppo_cfg["max_new_tokens"],
        "temperature": config["generation"]["temperature"],
        "top_p": config["generation"]["top_p"],
        "do_sample": config["generation"]["do_sample"],
        "pad_token_id": tokenizer.pad_token_id,
    }

    for step in range(0, len(prompts), ppo_cfg["batch_size"]):
        batch_prompts = prompts[step:step + ppo_cfg["batch_size"]]
        if len(batch_prompts) < ppo_cfg["batch_size"]:
            break

        # Tokenize prompts
        query_tensors = [
            tokenizer(p, return_tensors="pt", truncation=True,
                      max_length=config["dataset"]["max_prompt_length"])["input_ids"].squeeze()
            for p in batch_prompts
        ]

        # Generate responses
        response_tensors = trainer.generate(query_tensors, **gen_kwargs)

        # Decode responses for reward scoring
        responses = [tokenizer.decode(r.squeeze(), skip_special_tokens=True) for r in response_tensors]

        # Score with reward model
        rewards = []
        for prompt, response in zip(batch_prompts, responses):
            inputs = reward_tokenizer(
                prompt + response,
                return_tensors="pt",
                truncation=True,
                max_length=512,
            ).to(reward_model.device)
            with torch.no_grad():
                score = reward_model(**inputs).logits.squeeze().cpu()
            rewards.append(score)

        # PPO update step
        stats = trainer.step(query_tensors, response_tensors, rewards)

        # Log metrics
        batch_num = step // ppo_cfg["batch_size"]
        if batch_num % config["training"]["logging_steps"] == 0:
            print(f"\nStep {batch_num}:")
            for key in ["ppo/loss/total", "ppo/mean_scores", "objective/kl"]:
                if key in stats:
                    print(f"  {key}: {stats[key]:.4f}")

    # Save the aligned model
    output_path = os.path.join(config["training"]["output_dir"], "ppo_model")
    print(f"\n💾 Saving PPO-aligned model to {output_path}")
    trainer.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)

    print("\n✅ PPO training complete!")
    print("Next: run 03_ppo_analysis.py to analyze the results.\n")


if __name__ == "__main__":
    main()
