"""
Phase 3, Script 2: Evaluate GRPO Model on Math
================================================

This script evaluates a GRPO-trained model on GSM8K test problems and compares
its accuracy to the base (untrained) model. This is the "before vs after" check
that shows whether GRPO training actually improved math reasoning.

The pipeline:
    1. Load the GSM8K test set (formatted with prompts and gold answers)
    2. Load the base model and measure its accuracy
    3. Load the GRPO-trained model and measure its accuracy
    4. Print a comparison and show sample outputs

Run:
    python 02_grpo_evaluation.py \\
        --model_path ./outputs/grpo_model \\
        --max_samples 100

What to observe:
    - Base model accuracy vs GRPO model accuracy
    - Sample outputs: does the GRPO model show step-by-step reasoning?
    - Even modest improvements are meaningful with a 0.5B parameter model
"""

import sys
import argparse
from pathlib import Path

# ── Add project root to path so we can import shared utilities ──────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from peft import PeftModel

from shared.model_utils import DEFAULT_MODEL, load_model_and_tokenizer
from shared.data_utils import load_gsm8k_formatted
from shared.eval_utils import (
    evaluate_math_accuracy,
    generate_responses,
    compare_outputs,
)

# ╭──────────────────────────────────────────────────────────────────────────╮
# │ CONFIG — Adjust these values to experiment                              │
# ╰──────────────────────────────────────────────────────────────────────────╯
CONFIG = {
    "model_name": DEFAULT_MODEL,               # Base model name
    "max_samples": 100,                        # Number of test problems to evaluate
    "max_new_tokens": 512,                     # Max tokens to generate per response
    "temperature": 0.0,                        # Use greedy decoding for evaluation
    "num_sample_outputs": 5,                   # Number of examples to display
}


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate GRPO model on GSM8K")
    parser.add_argument(
        "--model_path", type=str, default="./outputs/grpo_model",
        help="Path to the GRPO-trained model (default: ./outputs/grpo_model)"
    )
    parser.add_argument(
        "--max_samples", type=int, default=CONFIG["max_samples"],
        help=f"Number of test problems to evaluate (default: {CONFIG['max_samples']})"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    print(f"\n{'='*60}")
    print(f"  GRPO Model Evaluation on GSM8K")
    print(f"  Base Model:    {CONFIG['model_name']}")
    print(f"  GRPO Model:    {args.model_path}")
    print(f"  Test Samples:  {args.max_samples}")
    print(f"{'='*60}\n")

    # ── Step 1: Load test dataset ──────────────────────────────────────────
    print("[1/4] Loading GSM8K test set...")
    test_dataset = load_gsm8k_formatted(split="test", max_samples=args.max_samples)
    print(f"  Loaded {len(test_dataset)} test problems")

    # ── Step 2: Evaluate base model ────────────────────────────────────────
    print("\n[2/4] Evaluating base model...")
    base_model, tokenizer = load_model_and_tokenizer(
        CONFIG["model_name"], device_map="auto"
    )

    print(f"  Running inference on {len(test_dataset)} problems (greedy decoding)...")
    base_accuracy = evaluate_math_accuracy(
        base_model, tokenizer, test_dataset, max_samples=args.max_samples
    )

    # Generate sample outputs from base model for comparison
    sample_prompts = [test_dataset[i]["prompt"] for i in range(CONFIG["num_sample_outputs"])]
    base_responses = generate_responses(
        base_model, tokenizer, sample_prompts,
        max_new_tokens=CONFIG["max_new_tokens"],
        temperature=CONFIG["temperature"],
    )

    # Free base model memory
    del base_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ── Step 3: Evaluate GRPO model ────────────────────────────────────────
    print(f"\n[3/4] Evaluating GRPO model from {args.model_path}...")

    # Load the base model again, then apply the LoRA adapter
    grpo_model, tokenizer = load_model_and_tokenizer(
        CONFIG["model_name"], device_map="auto"
    )
    grpo_model = PeftModel.from_pretrained(grpo_model, args.model_path)
    grpo_model = grpo_model.merge_and_unload()

    print(f"  Running inference on {len(test_dataset)} problems (greedy decoding)...")
    grpo_accuracy = evaluate_math_accuracy(
        grpo_model, tokenizer, test_dataset, max_samples=args.max_samples
    )

    # Generate sample outputs from GRPO model for comparison
    grpo_responses = generate_responses(
        grpo_model, tokenizer, sample_prompts,
        max_new_tokens=CONFIG["max_new_tokens"],
        temperature=CONFIG["temperature"],
    )

    # ── Step 4: Compare results ────────────────────────────────────────────
    print(f"\n[4/4] Comparison Results")
    print(f"\n{'='*60}")
    print(f"  Accuracy Comparison")
    print(f"{'='*60}")
    print(f"  Base Model:  {base_accuracy:.1%}")
    print(f"  GRPO Model:  {grpo_accuracy:.1%}")

    improvement = grpo_accuracy - base_accuracy
    if improvement > 0:
        print(f"  Improvement: +{improvement:.1%}")
    elif improvement < 0:
        print(f"  Change:      {improvement:.1%} (decrease)")
    else:
        print(f"  Change:      No change")

    # Show sample outputs side by side
    print(f"\n{'='*60}")
    print(f"  Sample Outputs (first {CONFIG['num_sample_outputs']} problems)")
    print(f"{'='*60}")

    gold_answers = [test_dataset[i]["answer"] for i in range(CONFIG["num_sample_outputs"])]

    compare_outputs(
        sample_prompts,
        base_responses,
        grpo_responses,
        label_a="Base Model",
        label_b="GRPO Model",
    )

    # Print gold answers for reference
    print(f"\n  Gold Answers:")
    for i, answer in enumerate(gold_answers):
        print(f"    Problem {i+1}: {answer}")

    # ── Summary ────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Evaluation Summary")
    print(f"{'='*60}")
    print(f"""
  Base model accuracy:  {base_accuracy:.1%}
  GRPO model accuracy:  {grpo_accuracy:.1%}

  Things to look for in the sample outputs:
  - Does the GRPO model show more structured reasoning?
  - Does it attempt step-by-step solutions?
  - Are the final answers formatted more consistently?

  Note: With a 0.5B model and limited training, improvements may be small.
  The goal is to verify that GRPO training is working directionally, not
  to achieve state-of-the-art math performance.

  Next step: Run 03_grpo_experiment.py for group size ablation.
""")


if __name__ == "__main__":
    main()
