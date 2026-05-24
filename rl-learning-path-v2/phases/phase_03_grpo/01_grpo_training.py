"""
Phase 3, Script 1: GRPO Training
==================================
Group Relative Policy Optimization — online RL without a value model (critic).
Generates multiple responses per prompt, normalizes rewards within the group.

What you'll learn:
- How group-based advantage estimation works in practice
- The effect of group size on training dynamics
- How GRPO compares to PPO in compute efficiency

Usage:
    python phases/phase_03_grpo/01_grpo_training.py
"""

import sys
import os
import yaml
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSequenceClassification
from trl import GRPOConfig, GRPOTrainer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from shared.model_utils import MODEL_NAME, print_model_info


def load_config():
    config_path = os.path.join(os.path.dirname(__file__), "configs", "grpo_config.yaml")
    with open(config_path) as f:
        return yaml.safe_load(f)


def prepare_grpo_dataset(config):
    ds = load_dataset(config["dataset"]["name"], split="train")
    ds = ds.shuffle(seed=42).select(range(min(config["dataset"]["max_samples"], len(ds))))

    def format_example(example):
        return {"prompt": example["prompt"]}

    ds = ds.map(format_example, remove_columns=[
        c for c in ds.column_names if c != "prompt"
    ])
    return ds


def build_reward_fn(config):
    rm_name = config["reward_model"]["name"]
    rm_tokenizer = AutoTokenizer.from_pretrained(rm_name)
    if rm_tokenizer.pad_token is None:
        rm_tokenizer.pad_token = rm_tokenizer.eos_token

    rm_model = AutoModelForSequenceClassification.from_pretrained(
        rm_name,
        num_labels=1,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    rm_model.eval()
    print_model_info(rm_model, f"Reward Model: {rm_name}")

    def reward_fn(completions, **kwargs):
        rewards = []
        for completion in completions:
            text = completion if isinstance(completion, str) else completion[0]["content"]
            inputs = rm_tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True,
            ).to(rm_model.device)
            with torch.no_grad():
                score = rm_model(**inputs).logits.squeeze().cpu().float().item()
            rewards.append(score)
        return rewards

    return reward_fn


def main():
    config = load_config()
    grpo_cfg = config["grpo"]

    print("\n📦 Loading model and tokenizer...")
    print(f"Group size (G): {grpo_cfg['num_generations']} responses per prompt\n")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    print_model_info(model, "Policy Model (Qwen2.5-1.5B)")

    print("📊 Preparing dataset...")
    train_ds = prepare_grpo_dataset(config)
    print(f"  Train prompts: {len(train_ds)}\n")

    print("📦 Loading reward model for scoring...")
    reward_fn = build_reward_fn(config)

    output_dir = os.path.join(config["training"]["output_dir"], "grpo_model")

    training_args = GRPOConfig(
        output_dir=output_dir,
        num_generations=grpo_cfg["num_generations"],
        learning_rate=grpo_cfg["learning_rate"],
        per_device_train_batch_size=grpo_cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=grpo_cfg["gradient_accumulation_steps"],
        num_train_epochs=grpo_cfg["num_train_epochs"],
        warmup_ratio=grpo_cfg["warmup_ratio"],
        logging_steps=grpo_cfg["logging_steps"],
        save_strategy="steps",
        save_steps=grpo_cfg["save_steps"],
        bf16=grpo_cfg["bf16"],
        max_completion_length=grpo_cfg["max_completion_length"],
        remove_unused_columns=False,
    )

    trainer = GRPOTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        processing_class=tokenizer,
        reward_funcs=reward_fn,
    )

    print("🚀 Starting GRPO training...")
    print("=" * 60)
    print(f"  Group size (G):  {grpo_cfg['num_generations']}")
    print(f"  KL coefficient:  {grpo_cfg['kl_coef']}")
    print(f"  Temperature:     {grpo_cfg['temperature']}")
    print(f"  Learning rate:   {grpo_cfg['learning_rate']}")
    print("=" * 60)
    print("\nMetrics to watch:")
    print("  reward/mean        ↑  (average reward across all groups)")
    print("  reward/std         ↔  (within-group variance — too low means all responses are similar)")
    print("  grpo/advantages    ↔  (should have both positive and negative values)")
    print("  kl                 ↔  (KL from reference — same as PPO)")
    print()

    trainer.train()

    print(f"\n💾 Saving GRPO-aligned model to {output_dir}")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    print("\n✅ GRPO training complete!")
    print("Next: run 02_grpo_analysis.py for the three-way comparison.\n")


if __name__ == "__main__":
    main()
