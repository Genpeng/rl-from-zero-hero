"""
Phase 0 — SFT Warm-Up Script
=============================

This script fine-tunes Qwen2.5-0.5B on the "chosen" responses from the Anthropic
HH-RLHF dataset using TRL's SFTTrainer with LoRA.

The result is an SFT base model that will serve as the starting point for all
later alignment experiments (PPO, DPO, GRPO).

Usage:
    python 01_sft_warmup.py
    python 01_sft_warmup.py --max_samples 500 --output_dir ./outputs/sft_test

What this script does:
    1. Loads Qwen2.5-0.5B + applies LoRA adapters
    2. Loads a subset of Anthropic HH-RLHF (chosen responses only)
    3. Runs supervised fine-tuning with SFTTrainer
    4. Prints before/after comparisons on test prompts
    5. Saves the fine-tuned model
"""

import sys
import os
import argparse

# ---------------------------------------------------------------------------
# Add parent directory to path so we can import shared utilities.
# This is needed because the script lives in phase_00_foundation/ but the
# shared/ module lives at the project root.
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from datasets import load_dataset
from transformers import TrainingArguments
from trl import SFTTrainer, SFTConfig

from shared.model_utils import load_model_and_tokenizer, get_lora_config, apply_lora
from shared.eval_utils import generate_responses, compare_outputs


# ============================================================================
# CONFIG — All tweakable parameters in one place
# ============================================================================

CONFIG = {
    # Model
    "model_name": "Qwen/Qwen2.5-0.5B",

    # LoRA
    "lora_r": 16,                     # LoRA rank — lower = fewer params, higher = more capacity
    "lora_alpha": 32,                 # LoRA alpha — scaling factor, typically 2x rank
    "lora_dropout": 0.05,             # Dropout in LoRA layers

    # Dataset
    "dataset_name": "Anthropic/hh-rlhf",
    "max_seq_length": 512,            # Max tokens per training example

    # Training
    "num_train_epochs": 1,            # 1 epoch is enough for a warm-up
    "per_device_train_batch_size": 4,  # Adjust based on GPU memory
    "gradient_accumulation_steps": 4,  # Effective batch size = 4 * 4 = 16
    "learning_rate": 2e-4,            # Standard LoRA learning rate
    "warmup_ratio": 0.05,             # Fraction of steps for LR warmup
    "weight_decay": 0.01,
    "logging_steps": 10,              # Log metrics every N steps
    "save_strategy": "epoch",
    "bf16": True,                     # Use bfloat16 (set False if GPU doesn't support it)

    # Evaluation
    "max_new_tokens": 256,            # Max tokens to generate during evaluation
}

# Test prompts for before/after comparison
TEST_PROMPTS = [
    "\n\nHuman: What are the health benefits of regular exercise?\n\nAssistant:",
    "\n\nHuman: Can you explain what machine learning is in simple terms?\n\nAssistant:",
    "\n\nHuman: I'm feeling really stressed about work. Any advice?\n\nAssistant:",
]


# ============================================================================
# DATA PREPARATION
# ============================================================================

def extract_chosen_text(example):
    """Extract the full chosen conversation from HH-RLHF.

    The HH-RLHF dataset has 'chosen' and 'rejected' fields. For SFT, we only
    use the 'chosen' responses — these are the human-preferred completions
    that we want the model to learn to imitate.

    The data format is:
        "\\n\\nHuman: <question>\\n\\nAssistant: <answer>"

    We use the full chosen text (prompt + response) as the training target,
    so the model learns both the conversational format and good responses.
    """
    return {"text": example["chosen"]}


def prepare_dataset(dataset_name, max_samples):
    """Load and prepare the HH-RLHF dataset for SFT.

    Steps:
        1. Load the training split from HH-RLHF
        2. Take a subset (for fast iteration)
        3. Extract just the 'chosen' responses
        4. Shuffle for good training dynamics
    """
    print(f"Loading dataset: {dataset_name}")
    dataset = load_dataset(dataset_name, split="train")

    # Take a subset for manageable training time
    num_samples = min(max_samples, len(dataset))
    dataset = dataset.select(range(num_samples))
    print(f"Using {num_samples} samples (out of {len(dataset)} available)")

    # Extract the chosen (human-preferred) conversations
    dataset = dataset.map(extract_chosen_text, remove_columns=dataset.column_names)

    # Shuffle so we don't train on data in its original order
    dataset = dataset.shuffle(seed=42)

    print(f"Dataset prepared: {len(dataset)} examples")
    print(f"Sample text (first 200 chars): {dataset[0]['text'][:200]}...")
    return dataset


# ============================================================================
# TRAINING
# ============================================================================

def train(args):
    """Main training function.

    This follows the standard SFT recipe:
        1. Load base model + tokenizer
        2. Apply LoRA adapters (so we only train ~0.5% of parameters)
        3. Prepare the dataset
        4. Generate "before" responses (to compare later)
        5. Train with SFTTrainer
        6. Generate "after" responses and compare
        7. Save the model
    """

    # ------------------------------------------------------------------
    # Step 1: Load model and tokenizer
    # ------------------------------------------------------------------
    print("=" * 60)
    print("Step 1: Loading model and tokenizer")
    print("=" * 60)

    model, tokenizer = load_model_and_tokenizer(CONFIG["model_name"])
    print(f"Model: {CONFIG['model_name']}")
    print(f"Parameters: {model.num_parameters():,}")

    # ------------------------------------------------------------------
    # Step 2: Apply LoRA
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Step 2: Applying LoRA adapters")
    print("=" * 60)

    lora_config = get_lora_config(
        r=CONFIG["lora_r"],
        lora_alpha=CONFIG["lora_alpha"],
    )
    model = apply_lora(model, lora_config)
    # apply_lora already calls print_trainable_parameters()

    # ------------------------------------------------------------------
    # Step 3: Prepare dataset
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Step 3: Preparing dataset")
    print("=" * 60)

    dataset = prepare_dataset(CONFIG["dataset_name"], args.max_samples)

    # ------------------------------------------------------------------
    # Step 4: Generate "before" responses
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Step 4: Generating BEFORE responses (base model)")
    print("=" * 60)

    before_responses = generate_responses(
        model,
        tokenizer,
        TEST_PROMPTS,
        max_new_tokens=CONFIG["max_new_tokens"],
    )

    for i, (prompt, response) in enumerate(zip(TEST_PROMPTS, before_responses)):
        print(f"\n--- Prompt {i+1} ---")
        print(f"Prompt: {prompt.strip()}")
        print(f"Response: {response[:300]}")

    # ------------------------------------------------------------------
    # Step 5: Train with SFTTrainer
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Step 5: Starting SFT training")
    print("=" * 60)

    # SFTConfig extends TrainingArguments with SFT-specific options
    training_args = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=CONFIG["num_train_epochs"],
        per_device_train_batch_size=CONFIG["per_device_train_batch_size"],
        gradient_accumulation_steps=CONFIG["gradient_accumulation_steps"],
        learning_rate=CONFIG["learning_rate"],
        warmup_ratio=CONFIG["warmup_ratio"],
        weight_decay=CONFIG["weight_decay"],
        logging_steps=CONFIG["logging_steps"],
        save_strategy=CONFIG["save_strategy"],
        bf16=CONFIG["bf16"],
        max_seq_length=CONFIG["max_seq_length"],
        # Disable wandb unless explicitly configured
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    print(f"Training for {CONFIG['num_train_epochs']} epoch(s)...")
    print(f"Effective batch size: {CONFIG['per_device_train_batch_size'] * CONFIG['gradient_accumulation_steps']}")
    print(f"Total optimization steps: ~{len(dataset) // (CONFIG['per_device_train_batch_size'] * CONFIG['gradient_accumulation_steps'])}")

    trainer.train()

    # ------------------------------------------------------------------
    # Step 6: Generate "after" responses and compare
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Step 6: Generating AFTER responses (fine-tuned model)")
    print("=" * 60)

    after_responses = generate_responses(
        model,
        tokenizer,
        TEST_PROMPTS,
        max_new_tokens=CONFIG["max_new_tokens"],
    )

    print("\n" + "=" * 60)
    print("BEFORE vs AFTER Comparison")
    print("=" * 60)

    compare_outputs(
        TEST_PROMPTS,
        before_responses,
        after_responses,
        label_a="Before SFT (base model)",
        label_b="After SFT (fine-tuned)",
    )

    # ------------------------------------------------------------------
    # Step 7: Save the model
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Step 7: Saving model")
    print("=" * 60)

    # Save LoRA adapters (small, ~5-20 MB)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Model saved to: {args.output_dir}")
    print(f"  - LoRA adapters: {args.output_dir}/adapter_model.safetensors")
    print(f"  - Tokenizer: {args.output_dir}/tokenizer.json")

    print("\n" + "=" * 60)
    print("SFT warm-up complete!")
    print("=" * 60)
    print(f"\nThis SFT model will be the starting point for:")
    print(f"  - Phase 1: PPO training")
    print(f"  - Phase 2: DPO training")
    print(f"  - Phase 3: GRPO training")
    print(f"\nNext step: Read phase_01_ppo/README.md")


# ============================================================================
# MAIN
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="SFT warm-up: fine-tune Qwen2.5-0.5B on Anthropic HH-RLHF chosen responses"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=1000,
        help="Maximum number of training samples to use (default: 1000)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./outputs/sft_warmup",
        help="Directory to save the fine-tuned model (default: ./outputs/sft_warmup)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
