"""
Phase 1, Script 1: Train a Reward Model on Anthropic HH-RLHF
=============================================================

This script trains a reward model that learns to distinguish "good" from "bad"
responses. Given a pair (chosen, rejected), the model learns to assign a higher
score to the chosen response.

The trained reward model is used in the next script (02_ppo_training.py) to
provide the reward signal for PPO.

Run:
    python 01_train_reward_model.py --max_samples 2000 --num_epochs 1

What to observe:
    - Training loss should decrease steadily
    - Evaluation accuracy should be above 50% (random baseline)
"""

import sys
import argparse
from pathlib import Path

# ── Add project root to path so we can import shared utilities ──────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from transformers import TrainingArguments
from trl import RewardTrainer
from peft import TaskType

from shared.model_utils import load_reward_model, get_lora_config
from shared.data_utils import load_hh_rlhf

# ╭──────────────────────────────────────────────────────────────────────────╮
# │ CONFIG — Adjust these values to experiment                              │
# ╰──────────────────────────────────────────────────────────────────────────╯
CONFIG = {
    "model_name": "Qwen/Qwen2.5-0.5B",       # Base model for reward head
    "max_length": 512,                         # Max sequence length
    "learning_rate": 1e-4,                     # Learning rate for LoRA layers
    "per_device_batch_size": 4,                # Batch size per GPU
    "gradient_accumulation_steps": 4,          # Effective batch = 4 * 4 = 16
    "lora_r": 16,                              # LoRA rank
    "lora_alpha": 32,                          # LoRA alpha
    "eval_split_ratio": 0.1,                   # Fraction of data for evaluation
    "logging_steps": 10,                       # Log every N steps
}


def parse_args():
    parser = argparse.ArgumentParser(description="Train a reward model on HH-RLHF")
    parser.add_argument(
        "--max_samples", type=int, default=2000,
        help="Maximum number of training samples (default: 2000)"
    )
    parser.add_argument(
        "--output_dir", type=str, default="./outputs/reward_model",
        help="Directory to save the trained reward model (default: ./outputs/reward_model)"
    )
    parser.add_argument(
        "--num_epochs", type=int, default=1,
        help="Number of training epochs (default: 1)"
    )
    return parser.parse_args()


def format_for_reward_trainer(example):
    """Format HH-RLHF example into chosen/rejected text pairs.

    The RewardTrainer expects each example to have 'chosen' and 'rejected'
    fields containing the full text (prompt + response).
    """
    return {
        "chosen": example["chosen"],
        "rejected": example["rejected"],
    }


def main():
    args = parse_args()
    print(f"\n{'='*60}")
    print(f"  Reward Model Training")
    print(f"  Model:       {CONFIG['model_name']}")
    print(f"  Samples:     {args.max_samples}")
    print(f"  Epochs:      {args.num_epochs}")
    print(f"  Output:      {args.output_dir}")
    print(f"{'='*60}\n")

    # ── Step 1: Load and prepare data ───────────────────────────────────────
    print("[1/4] Loading Anthropic HH-RLHF dataset...")
    dataset = load_hh_rlhf(split="train", max_samples=args.max_samples)
    dataset = dataset.map(format_for_reward_trainer)

    # Split into train and eval
    split = dataset.train_test_split(test_size=CONFIG["eval_split_ratio"], seed=42)
    train_dataset = split["train"]
    eval_dataset = split["test"]
    print(f"  Train: {len(train_dataset)} examples")
    print(f"  Eval:  {len(eval_dataset)} examples")

    # ── Step 2: Load model with reward head ─────────────────────────────────
    print("\n[2/4] Loading reward model...")
    model, tokenizer = load_reward_model(
        model_name=CONFIG["model_name"],
        num_labels=1,
        device_map="auto",
    )

    # Apply LoRA for parameter-efficient training
    lora_config = get_lora_config(
        r=CONFIG["lora_r"],
        lora_alpha=CONFIG["lora_alpha"],
        task_type=TaskType.SEQ_CLS,
    )

    # ── Step 3: Set up training ─────────────────────────────────────────────
    print("\n[3/4] Starting training...")
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=CONFIG["per_device_batch_size"],
        per_device_eval_batch_size=CONFIG["per_device_batch_size"],
        gradient_accumulation_steps=CONFIG["gradient_accumulation_steps"],
        learning_rate=CONFIG["learning_rate"],
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        logging_steps=CONFIG["logging_steps"],
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        report_to="none",  # Set to "wandb" if you want W&B logging
        bf16=torch.cuda.is_available(),
        remove_unused_columns=False,
        max_grad_norm=1.0,
    )

    trainer = RewardTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        peft_config=lora_config,
        max_length=CONFIG["max_length"],
    )

    trainer.train()

    # ── Step 4: Evaluate and save ───────────────────────────────────────────
    print("\n[4/4] Evaluating trained reward model...")
    eval_results = trainer.evaluate()

    print(f"\n{'='*60}")
    print(f"  Evaluation Results")
    print(f"{'='*60}")
    for key, value in eval_results.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")

    # Save the final model
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"\n  Model saved to: {args.output_dir}")

    # ── Quick sanity check: score a chosen vs rejected pair ─────────────────
    print(f"\n{'='*60}")
    print(f"  Sanity Check: Scoring a sample pair")
    print(f"{'='*60}")

    sample = eval_dataset[0]
    model.eval()
    with torch.no_grad():
        chosen_inputs = tokenizer(
            sample["chosen"], return_tensors="pt",
            truncation=True, max_length=CONFIG["max_length"],
        ).to(model.device)
        rejected_inputs = tokenizer(
            sample["rejected"], return_tensors="pt",
            truncation=True, max_length=CONFIG["max_length"],
        ).to(model.device)

        chosen_score = model(**chosen_inputs).logits.item()
        rejected_score = model(**rejected_inputs).logits.item()

    print(f"  Chosen score:   {chosen_score:.4f}")
    print(f"  Rejected score: {rejected_score:.4f}")
    print(f"  Correct order:  {'Yes' if chosen_score > rejected_score else 'No'}")
    print()


if __name__ == "__main__":
    main()
