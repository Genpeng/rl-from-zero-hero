"""
Phase 2, Script 1: DPO Training
=================================
Direct Preference Optimization — align the model using preference pairs directly.
No reward model needed. Just 2 models: Policy + Reference.

Compare how much simpler this is vs the PPO pipeline in Phase 1!

What you'll learn:
- How DPO training works in practice
- The role of beta (β) in controlling preference strength
- How to monitor DPO-specific metrics

Usage:
    python phases/phase_02_dpo/01_dpo_training.py
"""

import sys
import os
import yaml
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from shared.model_utils import MODEL_NAME, print_model_info


def load_config():
    config_path = os.path.join(os.path.dirname(__file__), "configs", "dpo_config.yaml")
    with open(config_path) as f:
        return yaml.safe_load(f)


def prepare_dpo_dataset(config):
    ds = load_dataset(config["dataset"]["name"], split="train")
    ds = ds.shuffle(seed=42).select(range(min(config["dataset"]["max_samples"], len(ds))))

    def format_example(example):
        return {
            "prompt": example["prompt"],
            "chosen": example["chosen"],
            "rejected": example["rejected"],
        }

    ds = ds.map(format_example, remove_columns=[
        c for c in ds.column_names if c not in ["prompt", "chosen", "rejected"]
    ])
    split = ds.train_test_split(test_size=0.1, seed=42)
    return split["train"], split["test"]


def main():
    config = load_config()
    dpo_cfg = config["dpo"]

    print("\n📦 Loading model and tokenizer...")
    print("Notice: only 2 models needed (Policy + Reference). No reward model!\n")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    print_model_info(model, "Policy Model (Qwen2.5-1.5B)")

    ref_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    print("Reference model loaded (frozen copy for KL computation)\n")

    print("📊 Preparing preference dataset...")
    train_ds, eval_ds = prepare_dpo_dataset(config)
    print(f"  Train: {len(train_ds)} preference pairs")
    print(f"  Eval:  {len(eval_ds)} preference pairs\n")

    output_dir = os.path.join(config["training"]["output_dir"], "dpo_model")

    training_args = DPOConfig(
        output_dir=output_dir,
        beta=dpo_cfg["beta"],
        learning_rate=dpo_cfg["learning_rate"],
        per_device_train_batch_size=dpo_cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=dpo_cfg["gradient_accumulation_steps"],
        num_train_epochs=dpo_cfg["num_train_epochs"],
        warmup_ratio=dpo_cfg["warmup_ratio"],
        logging_steps=dpo_cfg["logging_steps"],
        eval_strategy="steps",
        eval_steps=dpo_cfg["eval_steps"],
        save_strategy="steps",
        save_steps=dpo_cfg["save_steps"],
        bf16=dpo_cfg["bf16"],
        max_length=dpo_cfg["max_length"],
        max_prompt_length=dpo_cfg["max_prompt_length"],
        remove_unused_columns=False,
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
    )

    print("🚀 Starting DPO training...")
    print("=" * 60)
    print(f"  Beta (β):        {dpo_cfg['beta']}")
    print(f"  Learning rate:   {dpo_cfg['learning_rate']}")
    print(f"  Batch size:      {dpo_cfg['per_device_train_batch_size']} x {dpo_cfg['gradient_accumulation_steps']} grad accum")
    print("=" * 60)
    print("\nMetrics to watch:")
    print("  train/loss        ↓  (should decrease)")
    print("  rewards/chosen    ↑  (log-ratio for preferred responses)")
    print("  rewards/rejected  ↓  (log-ratio for rejected responses)")
    print("  rewards/margins   ↑  (gap between chosen and rejected)")
    print("  rewards/accuracies ↑ (fraction where chosen > rejected)")
    print()

    trainer.train()

    print(f"\n💾 Saving DPO-aligned model to {output_dir}")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    print("\n✅ DPO training complete!")
    print("Next: run 02_dpo_analysis.py to compare with PPO results.\n")


if __name__ == "__main__":
    main()
