"""
Phase 3, Script 2: GRPO Analysis — Three-Way Comparison
=========================================================
Compare all three approaches: SFT baseline vs PPO vs DPO vs GRPO.
This is the dress rehearsal for the capstone.

Usage:
    python phases/phase_03_grpo/02_grpo_analysis.py
"""

import sys
import os

import torch
from transformers import AutoModelForCausalLM

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from shared.model_utils import MODEL_NAME, load_model_and_tokenizer
from shared.eval_utils import generate_samples, compare_outputs


EVAL_PROMPTS = [
    "Explain machine learning to a 10-year-old.",
    "What are the pros and cons of remote work?",
    "Write a short poem about coding.",
    "How do I deal with a difficult coworker?",
    "Explain why the sky is blue in simple terms.",
    "What is the most important thing in life?",
    "How should I prepare for a job interview?",
    "Summarize the concept of climate change in 3 sentences.",
]


MODEL_PATHS = {
    "PPO Aligned": "outputs/phase_01_ppo/ppo_model",
    "DPO Aligned": "outputs/phase_02_dpo/dpo_model",
    "GRPO Aligned": "outputs/phase_03_grpo/grpo_model",
}


def try_load_model(path, label):
    if os.path.exists(path):
        print(f"  Loading {label} from {path}...")
        return AutoModelForCausalLM.from_pretrained(
            path, torch_dtype=torch.bfloat16, device_map="auto",
        )
    print(f"  ⚠️  {label} not found at {path} — skipping")
    return None


def main():
    print("\n📊 Phase 3 Analysis: SFT vs PPO vs DPO vs GRPO\n")

    print("Loading models...")
    sft_model, tokenizer = load_model_and_tokenizer(MODEL_NAME)

    models = {}
    for label, path in MODEL_PATHS.items():
        m = try_load_model(path, label)
        if m is not None:
            models[label] = m

    print("\nGenerating responses...\n")
    model_outputs = {"SFT Baseline": generate_samples(sft_model, tokenizer, EVAL_PROMPTS)}

    for label, model in models.items():
        model_outputs[label] = generate_samples(model, tokenizer, EVAL_PROMPTS)

    compare_outputs(
        EVAL_PROMPTS,
        model_outputs,
        save_path="outputs/phase_03_grpo/three_way_comparison.json",
    )

    print("\n" + "=" * 60)
    print("🔍 THREE-WAY COMPARISON QUESTIONS:")
    print("=" * 60)
    print("""
Compare all three alignment methods across these dimensions:

1. OUTPUT QUALITY
   - Which produces the most helpful, relevant responses?
   - Do any show signs of reward hacking (exploiting patterns)?

2. TRAINING COMPLEXITY
   - PPO: 4 models, complex loop, many hyperparameters
   - DPO: 2 models, simple loss, few hyperparameters
   - GRPO: 3 models, group generation, moderate complexity
   Which felt most manageable?

3. COMPUTE COST
   - Wall-clock training time
   - Peak GPU memory usage
   - Total GPU-hours

4. STABILITY
   - Which training was most stable (smoothest loss curve)?
   - Which required the most hyperparameter tuning?

5. DATA REQUIREMENTS
   - PPO: Reward model + unlabeled prompts
   - DPO: Direct preference pairs
   - GRPO: Reward function + unlabeled prompts
   Which data format is easiest to obtain for your work?

📝 Write down your comparison notes — you'll formalize this in the capstone.

🔬 EXPERIMENT: Try GRPO with a rule-based reward function:
   - Reward = -abs(len(response) - 100)  (penalize responses far from 100 chars)
   - This demonstrates GRPO without a learned reward model
""")


if __name__ == "__main__":
    main()
