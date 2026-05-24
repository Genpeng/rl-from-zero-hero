"""
Phase 1, Script 3: PPO Analysis
=================================
Compare the SFT baseline against the PPO-aligned model.
This script helps you build intuition for what PPO actually changes.

What you'll learn:
- How to compare model outputs qualitatively
- What KL coefficient changes look like in practice
- How to spot reward hacking

Usage:
    python phases/phase_01_ppo/03_ppo_analysis.py
"""

import sys
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

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


def main():
    print("\n📊 Phase 1 Analysis: SFT vs PPO\n")

    # Load SFT baseline
    print("Loading SFT baseline model...")
    sft_model, tokenizer = load_model_and_tokenizer(MODEL_NAME)

    # Load PPO-aligned model
    ppo_path = "outputs/phase_01_ppo/ppo_model"
    if os.path.exists(ppo_path):
        print(f"Loading PPO model from {ppo_path}...")
        ppo_model = AutoModelForCausalLM.from_pretrained(
            ppo_path, torch_dtype=torch.bfloat16, device_map="auto",
        )
    else:
        print(f"⚠️  PPO model not found at {ppo_path}. Run 02_ppo_training.py first.")
        print("Showing SFT baseline only.\n")
        ppo_model = None

    # Generate and compare
    print("\nGenerating responses...\n")
    sft_outputs = generate_samples(sft_model, tokenizer, EVAL_PROMPTS)

    model_outputs = {"SFT Baseline": sft_outputs}
    if ppo_model is not None:
        ppo_outputs = generate_samples(ppo_model, tokenizer, EVAL_PROMPTS)
        model_outputs["PPO Aligned"] = ppo_outputs

    results = compare_outputs(
        EVAL_PROMPTS,
        model_outputs,
        save_path="outputs/phase_01_ppo/comparison_results.json",
    )

    print("\n" + "=" * 60)
    print("🔍 THINGS TO LOOK FOR:")
    print("=" * 60)
    print("""
1. HELPFULNESS: Are PPO responses more directly helpful?
   (They should be — the reward model was trained on helpfulness preferences)

2. VERBOSITY: Are PPO responses longer or shorter?
   (Often longer — models learn that longer ≈ higher reward, a form of reward hacking)

3. SAFETY: Does the PPO model handle sensitive topics differently?
   (It should be more careful if the reward model penalizes harmful content)

4. STYLE: Has the model's "personality" changed?
   (Slight changes are normal; drastic changes suggest KL was too low)

📝 Write down your observations — you'll compare these with DPO/GRPO results later.
""")


if __name__ == "__main__":
    main()
