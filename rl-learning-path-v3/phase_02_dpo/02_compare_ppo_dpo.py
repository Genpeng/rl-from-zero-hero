"""
Phase 2, Script 2: PPO vs DPO Output Comparison
================================================

This script loads both a PPO-trained model (from Phase 1) and a DPO-trained
model (from Script 1 of this phase) and generates responses from both on the
same prompts. It provides a side-by-side qualitative comparison.

The pipeline:
    1. Load the PPO model and DPO model
    2. Generate responses from both on identical test prompts
    3. Display side-by-side comparison using shared eval_utils

Run:
    python 02_compare_ppo_dpo.py \\
        --ppo_model_path ./outputs/ppo_model \\
        --dpo_model_path ./outputs/dpo_model

What to observe:
    - Do PPO and DPO models produce similar or different styles?
    - Which outputs feel more natural and helpful?
    - Does one method produce longer or shorter responses?
    - PPO was trained with a reward model; DPO learned from preference pairs.
      Can you see this difference in the outputs?
"""

import sys
import argparse
from pathlib import Path

# ── Add project root to path so we can import shared utilities ──────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from shared.model_utils import DEFAULT_MODEL, get_device
from shared.eval_utils import generate_responses, compare_outputs

# ╭──────────────────────────────────────────────────────────────────────────╮
# │ CONFIG — Adjust these values to experiment                              │
# ╰──────────────────────────────────────────────────────────────────────────╯
CONFIG = {
    "base_model_name": DEFAULT_MODEL,          # Base model for reference
    "max_new_tokens": 128,                     # Max tokens per response
    "temperature": 0.7,                        # Sampling temperature
}

# Test prompts for comparison
TEST_PROMPTS = [
    "\n\nHuman: How can I improve my writing skills?\n\nAssistant:",
    "\n\nHuman: What are some healthy breakfast ideas?\n\nAssistant:",
    "\n\nHuman: Can you explain what machine learning is in simple terms?\n\nAssistant:",
    "\n\nHuman: How do I deal with stress at work?\n\nAssistant:",
    "\n\nHuman: What is the best way to learn a new language?\n\nAssistant:",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Compare PPO and DPO model outputs")
    parser.add_argument(
        "--ppo_model_path", type=str, default="./outputs/ppo_model",
        help="Path to the PPO-trained model (default: ./outputs/ppo_model)"
    )
    parser.add_argument(
        "--dpo_model_path", type=str, default="./outputs/dpo_model",
        help="Path to the DPO-trained model (default: ./outputs/dpo_model)"
    )
    return parser.parse_args()


def load_aligned_model(model_path, base_model_name):
    """Load an aligned model (either PPO or DPO) with its LoRA adapter.

    Tries to load as a PEFT (LoRA) model first. If that fails, falls back to
    loading as a full model (some save formats differ between PPO and DPO).
    """
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    try:
        # Try loading as a LoRA adapter on top of the base model
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        model = PeftModel.from_pretrained(base_model, model_path)
        model = model.merge_and_unload()
        print(f"  Loaded as LoRA adapter from: {model_path}")
    except Exception:
        # Fall back to loading as a full model
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        print(f"  Loaded as full model from: {model_path}")

    return model, tokenizer


def main():
    args = parse_args()
    print(f"\n{'='*60}")
    print(f"  PPO vs DPO — Side-by-Side Comparison")
    print(f"  PPO Model:  {args.ppo_model_path}")
    print(f"  DPO Model:  {args.dpo_model_path}")
    print(f"{'='*60}\n")

    # ── Check that model paths exist ───────────────────────────────────────
    ppo_path = Path(args.ppo_model_path)
    dpo_path = Path(args.dpo_model_path)

    if not ppo_path.exists():
        print(f"  WARNING: PPO model not found at {args.ppo_model_path}")
        print(f"  Run phase_01_ppo/02_ppo_training.py first to train a PPO model.")
        print(f"  Will load base model as fallback for PPO.\n")
    if not dpo_path.exists():
        print(f"  WARNING: DPO model not found at {args.dpo_model_path}")
        print(f"  Run 01_dpo_training.py first to train a DPO model.")
        print(f"  Will load base model as fallback for DPO.\n")

    # ── Step 1: Load base model for reference ──────────────────────────────
    print("[1/4] Loading base model (for reference)...")
    base_model = AutoModelForCausalLM.from_pretrained(
        CONFIG["base_model_name"],
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    base_tokenizer = AutoTokenizer.from_pretrained(
        CONFIG["base_model_name"], trust_remote_code=True
    )
    if base_tokenizer.pad_token is None:
        base_tokenizer.pad_token = base_tokenizer.eos_token

    print("  Generating base model responses...")
    base_responses = generate_responses(
        base_model,
        base_tokenizer,
        TEST_PROMPTS,
        max_new_tokens=CONFIG["max_new_tokens"],
        temperature=CONFIG["temperature"],
    )

    # Free base model memory
    del base_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ── Step 2: Load PPO model ─────────────────────────────────────────────
    print("\n[2/4] Loading PPO model...")
    if ppo_path.exists():
        ppo_model, ppo_tokenizer = load_aligned_model(
            args.ppo_model_path, CONFIG["base_model_name"]
        )
    else:
        print("  Using base model as PPO fallback.")
        ppo_model = AutoModelForCausalLM.from_pretrained(
            CONFIG["base_model_name"],
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        ppo_tokenizer = base_tokenizer

    print("  Generating PPO responses...")
    ppo_responses = generate_responses(
        ppo_model,
        ppo_tokenizer,
        TEST_PROMPTS,
        max_new_tokens=CONFIG["max_new_tokens"],
        temperature=CONFIG["temperature"],
    )

    # Free PPO model memory
    del ppo_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ── Step 3: Load DPO model ─────────────────────────────────────────────
    print("\n[3/4] Loading DPO model...")
    if dpo_path.exists():
        dpo_model, dpo_tokenizer = load_aligned_model(
            args.dpo_model_path, CONFIG["base_model_name"]
        )
    else:
        print("  Using base model as DPO fallback.")
        dpo_model = AutoModelForCausalLM.from_pretrained(
            CONFIG["base_model_name"],
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        dpo_tokenizer = base_tokenizer

    print("  Generating DPO responses...")
    dpo_responses = generate_responses(
        dpo_model,
        dpo_tokenizer,
        TEST_PROMPTS,
        max_new_tokens=CONFIG["max_new_tokens"],
        temperature=CONFIG["temperature"],
    )

    del dpo_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ── Step 4: Display comparisons ────────────────────────────────────────
    print(f"\n[4/4] Comparing outputs...")

    # Compare base vs PPO
    print(f"\n{'='*80}")
    print(f"  COMPARISON 1: Base Model vs PPO")
    print(f"{'='*80}")
    compare_outputs(
        TEST_PROMPTS,
        base_responses,
        ppo_responses,
        label_a="Base Model",
        label_b="PPO Model",
    )

    # Compare base vs DPO
    print(f"\n{'='*80}")
    print(f"  COMPARISON 2: Base Model vs DPO")
    print(f"{'='*80}")
    compare_outputs(
        TEST_PROMPTS,
        base_responses,
        dpo_responses,
        label_a="Base Model",
        label_b="DPO Model",
    )

    # Compare PPO vs DPO
    print(f"\n{'='*80}")
    print(f"  COMPARISON 3: PPO vs DPO (the main event)")
    print(f"{'='*80}")
    compare_outputs(
        TEST_PROMPTS,
        ppo_responses,
        dpo_responses,
        label_a="PPO Model",
        label_b="DPO Model",
    )

    # ── Summary ────────────────────────────────────────────────────────────
    print(f"\n{'='*80}")
    print(f"  Comparison Summary")
    print(f"{'='*80}")

    # Compute average response lengths
    avg_base = sum(len(r) for r in base_responses) / len(base_responses)
    avg_ppo = sum(len(r) for r in ppo_responses) / len(ppo_responses)
    avg_dpo = sum(len(r) for r in dpo_responses) / len(dpo_responses)

    print(f"\n  Average response length (characters):")
    print(f"    Base Model: {avg_base:.0f}")
    print(f"    PPO Model:  {avg_ppo:.0f}")
    print(f"    DPO Model:  {avg_dpo:.0f}")

    print(f"""
  Things to reflect on:
    1. Which alignment method produced more helpful responses?
    2. Did either method produce noticeably longer or shorter outputs?
    3. PPO used a trained reward model; DPO used preference pairs directly.
       Can you detect any difference in output style?
    4. Would you trust one method's outputs more than the other?
""")


if __name__ == "__main__":
    main()
