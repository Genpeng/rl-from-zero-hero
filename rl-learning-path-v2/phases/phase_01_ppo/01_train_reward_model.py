"""
Phase 1, Script 1: Train a Reward Model
========================================
This script trains a reward model on preference data.
The reward model scores responses — it's the "judge" that PPO optimizes against.

What you'll learn:
- How preference data (chosen vs rejected) becomes training signal
- How the Bradley-Terry model works in practice
- Why reward model quality matters

Usage:
    python phases/phase_01_ppo/01_train_reward_model.py
"""

import sys
import os
import yaml
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_dataset
from trl import RewardTrainer, RewardConfig

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from shared.model_utils import print_model_info


def load_config():
    config_path = os.path.join(os.path.dirname(__file__), "configs", "ppo_config.yaml")
    with open(config_path) as f:
        return yaml.safe_load(f)


def prepare_reward_dataset(config):
    ds = load_dataset(
        config["dataset"]["name"],
        split="train",
    )
    ds = ds.shuffle(seed=42).select(range(min(config["dataset"]["max_samples"], len(ds))))

    def preprocess(example):
        return {
            "chosen": example["chosen"],
            "rejected": example["rejected"],
        }

    ds = ds.map(preprocess, remove_columns=[c for c in ds.column_names if c not in ["chosen", "rejected"]])
    split = ds.train_test_split(test_size=0.1, seed=42)
    return split["train"], split["test"]


def main():
    config = load_config()

    print("\n📦 Loading reward model and tokenizer...")
    rm_name = config["reward_model"]["name"]
    tokenizer = AutoTokenizer.from_pretrained(rm_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForSequenceClassification.from_pretrained(
        rm_name,
        num_labels=1,
        torch_dtype=torch.bfloat16,
    )
    print_model_info(model, f"Reward Model: {rm_name}")

    print("\n📊 Preparing preference dataset...")
    train_ds, eval_ds = prepare_reward_dataset(config)
    print(f"  Train samples: {len(train_ds)}")
    print(f"  Eval samples:  {len(eval_ds)}")

    output_dir = os.path.join(config["training"]["output_dir"], "reward_model")

    training_args = RewardConfig(
        output_dir=output_dir,
        num_train_epochs=1,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        learning_rate=2e-5,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=50,
        save_strategy="steps",
        save_steps=100,
        bf16=True,
        remove_unused_columns=False,
        max_length=512,
    )

    trainer = RewardTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
    )

    print("\n🚀 Training reward model...")
    print("Watch the eval accuracy — it should be >60% for a useful reward model.")
    print("If it's near 50%, the model isn't learning preferences.\n")

    trainer.train()

    print(f"\n💾 Saving reward model to {output_dir}")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    print("\n✅ Reward model training complete!")
    print("Next: run 02_ppo_training.py to use this reward model for PPO.\n")


if __name__ == "__main__":
    main()
