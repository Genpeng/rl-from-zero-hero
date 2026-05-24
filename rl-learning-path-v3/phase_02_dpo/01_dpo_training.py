"""
Phase 2, Script 1: DPO Alignment Training
==========================================

This script trains a language model using Direct Preference Optimization (DPO)
on Anthropic HH-RLHF preference data. Unlike PPO, DPO skips the reward model
entirely — it learns directly from preference pairs.

The pipeline:
    1. Load the base model and tokenizer
    2. Apply LoRA for parameter-efficient training
    3. Load HH-RLHF formatted as (prompt, chosen, rejected) triples
    4. Train with TRL's DPOTrainer using the DPO loss
    5. Generate sample outputs to inspect quality

Run:
    python 01_dpo_training.py \\
        --max_samples 2000 \\
        --beta 0.1 \\
        --num_epochs 1

What to observe:
    - Training loss should decrease over steps
    - Lower loss means the model is better at separating chosen from rejected
    - Compare sample outputs to the base model — are they more helpful?
    - Try different --beta values to see the effect on output style
"""

import sys
import argparse
from pathlib import Path

# ── Add project root to path so we can import shared utilities ──────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from trl import DPOTrainer, DPOConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

from shared.model_utils import DEFAULT_MODEL, get_lora_config, get_device
from shared.data_utils import load_hh_rlhf_for_dpo
from shared.eval_utils import generate_responses

# ╭──────────────────────────────────────────────────────────────────────────╮
# │ CONFIG — Adjust these values to experiment                              │
# ╰──────────────────────────────────────────────────────────────────────────╯
CONFIG = {
    "model_name": DEFAULT_MODEL,               # Base model (policy + reference)
    "max_length": 512,                         # Max sequence length for DPO
    "max_prompt_length": 256,                  # Max prompt token length
    "max_new_tokens": 128,                     # Max tokens to generate for demos
    "per_device_batch_size": 4,                # Batch size per device
    "gradient_accumulation_steps": 4,          # Effective batch = batch * grad_accum
    "learning_rate": 5e-5,                     # Learning rate for LoRA params
    "warmup_ratio": 0.1,                       # Warmup fraction of total steps
    "temperature": 0.7,                        # Sampling temperature for generation
    "num_sample_prompts": 5,                   # Number of demo prompts at the end
    "logging_steps": 10,                       # Log metrics every N steps
}

# Prompts for inspecting outputs after training
EVAL_PROMPTS = [
    "\n\nHuman: How can I improve my writing skills?\n\nAssistant:",
    "\n\nHuman: What are some healthy breakfast ideas?\n\nAssistant:",
    "\n\nHuman: Can you explain what machine learning is in simple terms?\n\nAssistant:",
    "\n\nHuman: How do I deal with stress at work?\n\nAssistant:",
    "\n\nHuman: What is the best way to learn a new language?\n\nAssistant:",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Run DPO alignment training")
    parser.add_argument(
        "--max_samples", type=int, default=2000,
        help="Maximum number of preference pairs to train on (default: 2000)"
    )
    parser.add_argument(
        "--output_dir", type=str, default="./outputs/dpo_model",
        help="Directory to save the DPO-trained model (default: ./outputs/dpo_model)"
    )
    parser.add_argument(
        "--beta", type=float, default=0.1,
        help="DPO beta parameter — KL constraint strength (default: 0.1)"
    )
    parser.add_argument(
        "--num_epochs", type=int, default=1,
        help="Number of training epochs (default: 1)"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    print(f"\n{'='*60}")
    print(f"  DPO Alignment Training")
    print(f"  Model:       {CONFIG['model_name']}")
    print(f"  Samples:     {args.max_samples}")
    print(f"  Beta (β):    {args.beta}")
    print(f"  Epochs:      {args.num_epochs}")
    print(f"  Output:      {args.output_dir}")
    print(f"{'='*60}\n")

    # ── Step 1: Load the dataset ───────────────────────────────────────────
    print("[1/5] Loading and formatting HH-RLHF preference data...")
    dataset = load_hh_rlhf_for_dpo(split="train", max_samples=args.max_samples)
    print(f"  Loaded {len(dataset)} preference pairs")
    print(f"  Example prompt:   {dataset[0]['prompt'][:80]}...")
    print(f"  Example chosen:   {dataset[0]['chosen'][:80]}...")
    print(f"  Example rejected: {dataset[0]['rejected'][:80]}...")

    # ── Step 2: Load model and tokenizer ───────────────────────────────────
    print("\n[2/5] Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        CONFIG["model_name"], trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        CONFIG["model_name"],
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    # ── Step 3: Set up LoRA ────────────────────────────────────────────────
    print("\n[3/5] Applying LoRA for parameter-efficient training...")
    lora_config = get_lora_config(r=16, lora_alpha=32)
    # DPOTrainer applies LoRA internally when we pass peft_config

    # ── Step 4: Store baseline outputs before training ─────────────────────
    print("\n[4/5] Generating baseline outputs (before DPO)...")
    baseline_responses = generate_responses(
        model,
        tokenizer,
        EVAL_PROMPTS[:CONFIG["num_sample_prompts"]],
        max_new_tokens=CONFIG["max_new_tokens"],
        temperature=CONFIG["temperature"],
    )

    # ── Step 5: Configure and run DPO training ─────────────────────────────
    print("\n[5/5] Running DPO training...")
    training_args = DPOConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=CONFIG["per_device_batch_size"],
        gradient_accumulation_steps=CONFIG["gradient_accumulation_steps"],
        learning_rate=CONFIG["learning_rate"],
        warmup_ratio=CONFIG["warmup_ratio"],
        beta=args.beta,
        max_length=CONFIG["max_length"],
        max_prompt_length=CONFIG["max_prompt_length"],
        logging_steps=CONFIG["logging_steps"],
        save_strategy="epoch",
        remove_unused_columns=False,
        bf16=torch.cuda.is_available(),
        report_to="none",
    )

    trainer = DPOTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=lora_config,
    )

    train_result = trainer.train()

    # Print training summary
    print(f"\n{'─'*60}")
    print(f"  Training Summary")
    print(f"{'─'*60}")
    metrics = train_result.metrics
    for key, value in sorted(metrics.items()):
        print(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")

    # ── Generate and compare outputs ───────────────────────────────────────
    print(f"\n{'─'*60}")
    print(f"  Generating post-DPO outputs...")
    print(f"{'─'*60}")

    dpo_responses = generate_responses(
        trainer.model,
        tokenizer,
        EVAL_PROMPTS[:CONFIG["num_sample_prompts"]],
        max_new_tokens=CONFIG["max_new_tokens"],
        temperature=CONFIG["temperature"],
    )

    # Print comparison
    for i, prompt in enumerate(EVAL_PROMPTS[:CONFIG["num_sample_prompts"]]):
        print(f"\n{'='*80}")
        print(f"PROMPT {i+1}: {prompt.strip()[:80]}...")
        print(f"\n--- Before DPO ---")
        print(baseline_responses[i][:300])
        print(f"\n--- After DPO ---")
        print(dpo_responses[i][:300])
    print(f"\n{'='*80}")

    # ── Save the model ─────────────────────────────────────────────────────
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(output_path))
    tokenizer.save_pretrained(str(output_path))
    print(f"\n  Model saved to: {args.output_dir}")

    print(f"\n{'='*60}")
    print(f"  DPO Training Complete!")
    print(f"{'='*60}")
    print(f"""
  Next steps:
    1. Run 02_compare_ppo_dpo.py to compare this model with your PPO model
    2. Run 03_dpo_experiment.py to see how beta affects training
    3. Try re-running with different --beta values (0.01, 0.5) and compare
""")


if __name__ == "__main__":
    main()
