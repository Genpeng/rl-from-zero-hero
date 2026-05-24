"""
Phase 2, Script 2: DPO Analysis
=================================
Compare SFT baseline, PPO-aligned, and DPO-aligned models side by side.

What you'll learn:
- How DPO outputs differ from PPO outputs
- Trade-offs between the two approaches in practice
- Whether simpler (DPO) produces comparable quality to complex (PPO)

Usage:
    python phases/phase_02_dpo/02_dpo_analysis.py
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


def try_load_model(path, label):
    if os.path.exists(path):
        print(f"  Loading {label} from {path}...")
        return AutoModelForCausalLM.from_pretrained(
            path, torch_dtype=torch.bfloat16, device_map="auto",
        )
    print(f"  ⚠️  {label} not found at {path} — skipping")
    return None


def main():
    print("\n📊 Phase 2 Analysis: SFT vs PPO vs DPO\n")

    print("Loading models...")
    sft_model, tokenizer = load_model_and_tokenizer(MODEL_NAME)
    ppo_model = try_load_model("outputs/phase_01_ppo/ppo_model", "PPO model")
    dpo_model = try_load_model("outputs/phase_02_dpo/dpo_model", "DPO model")

    print("\nGenerating responses...\n")
    model_outputs = {"SFT Baseline": generate_samples(sft_model, tokenizer, EVAL_PROMPTS)}

    if ppo_model is not None:
        model_outputs["PPO Aligned"] = generate_samples(ppo_model, tokenizer, EVAL_PROMPTS)
    if dpo_model is not None:
        model_outputs["DPO Aligned"] = generate_samples(dpo_model, tokenizer, EVAL_PROMPTS)

    compare_outputs(
        EVAL_PROMPTS,
        model_outputs,
        save_path="outputs/phase_02_dpo/comparison_results.json",
    )

    print("\n" + "=" * 60)
    print("🔍 COMPARISON QUESTIONS:")
    print("=" * 60)
    print("""
1. QUALITY: Which produces more helpful responses — PPO or DPO?
   (They're often surprisingly similar despite DPO being much simpler)

2. STYLE: Does DPO have different verbosity/style tendencies than PPO?
   (DPO often shows less reward hacking since there's no reward model to exploit)

3. TRAINING EFFORT: Compare wall-clock time and GPU memory usage.
   (DPO should be significantly faster and use less memory)

4. STABILITY: Was DPO training smoother than PPO?
   (DPO is typically much more stable — fewer hyperparameters to get wrong)

📝 Document your observations for the capstone comparison.

🔬 EXPERIMENT: Try changing β in dpo_config.yaml:
   - β=0.01 (aggressive) — does the model overfit to preferences?
   - β=0.5  (conservative) — does the model barely change?
   Compare with KL coefficient experiments from PPO.
""")


if __name__ == "__main__":
    main()
