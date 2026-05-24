"""
Phase 4, Script 2: Evaluate and Compare
==========================================
Generate the comparison table across all three algorithms.
This is the core deliverable of the capstone.

Usage:
    python phases/phase_04_capstone/02_evaluate_and_compare.py
"""

import sys
import os
import json
import time

import torch
import pandas as pd
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from shared.model_utils import MODEL_NAME, load_model_and_tokenizer
from shared.eval_utils import generate_samples, compare_outputs


DEFAULT_EVAL_PROMPTS = [
    "Explain machine learning to a 10-year-old.",
    "What are the pros and cons of remote work?",
    "Write a short poem about coding.",
    "How do I deal with a difficult coworker?",
    "Explain why the sky is blue in simple terms.",
    "What is the most important thing in life?",
    "How should I prepare for a job interview?",
    "Summarize the concept of climate change in 3 sentences.",
    "What makes a good leader?",
    "Explain quantum computing in simple terms.",
    "How can I improve my public speaking skills?",
    "What are the ethical concerns around AI?",
    "Write a motivational message for someone learning to code.",
    "How does blockchain technology work?",
    "What habits lead to a productive morning routine?",
    "Explain the difference between empathy and sympathy.",
    "How should I approach learning a new programming language?",
    "What are the benefits of reading books regularly?",
    "How can teams collaborate more effectively remotely?",
    "Explain the concept of compound interest simply.",
]


def load_config():
    config_path = os.path.join(os.path.dirname(__file__), "configs", "capstone_config.yaml")
    with open(config_path) as f:
        return yaml.safe_load(f)


def score_responses(responses, rm_model, rm_tokenizer):
    scores = []
    for resp in responses:
        inputs = rm_tokenizer(
            resp, return_tensors="pt", truncation=True, max_length=512, padding=True,
        ).to(rm_model.device)
        with torch.no_grad():
            score = rm_model(**inputs).logits.squeeze().cpu().float().item()
        scores.append(score)
    return scores


def main():
    config = load_config()
    output_dir = config["evaluation"]["output_dir"]
    eval_cfg = config["evaluation"]

    print("\n📊 Capstone Evaluation: PPO vs DPO vs GRPO\n")

    # Load eval prompts
    if eval_cfg.get("eval_prompts_file") and os.path.exists(eval_cfg["eval_prompts_file"]):
        with open(eval_cfg["eval_prompts_file"]) as f:
            eval_prompts = [line.strip() for line in f if line.strip()]
    else:
        eval_prompts = DEFAULT_EVAL_PROMPTS[:eval_cfg.get("num_eval_prompts", 20)]

    print(f"Using {len(eval_prompts)} evaluation prompts\n")

    # Load models
    print("Loading models...")
    sft_model, tokenizer = load_model_and_tokenizer(MODEL_NAME)

    model_configs = {
        "PPO": os.path.join(output_dir, "ppo_model"),
        "DPO": os.path.join(output_dir, "dpo_model"),
        "GRPO": os.path.join(output_dir, "grpo_model"),
    }

    aligned_models = {}
    for name, path in model_configs.items():
        if os.path.exists(path):
            print(f"  Loading {name} from {path}...")
            aligned_models[name] = AutoModelForCausalLM.from_pretrained(
                path, torch_dtype=torch.bfloat16, device_map="auto",
            )
        else:
            print(f"  ⚠️  {name} not found at {path} — skipping")

    # Load reward model for scoring
    print("\nLoading reward model for scoring...")
    rm_name = config["reward_model"]["name"]
    rm_tokenizer = AutoTokenizer.from_pretrained(rm_name)
    if rm_tokenizer.pad_token is None:
        rm_tokenizer.pad_token = rm_tokenizer.eos_token
    rm_model = AutoModelForSequenceClassification.from_pretrained(
        rm_name, num_labels=1, torch_dtype=torch.bfloat16, device_map="auto",
    )
    rm_model.eval()

    # Generate and score
    print("\nGenerating and scoring responses...\n")

    results = {"SFT": {}}
    sft_outputs = generate_samples(sft_model, tokenizer, eval_prompts,
                                   max_new_tokens=eval_cfg.get("max_new_tokens", 256))
    sft_scores = score_responses(sft_outputs, rm_model, rm_tokenizer)
    results["SFT"] = {
        "outputs": sft_outputs,
        "scores": sft_scores,
        "avg_score": sum(sft_scores) / len(sft_scores),
        "avg_length": sum(len(o) for o in sft_outputs) / len(sft_outputs),
    }

    for name, model in aligned_models.items():
        outputs = generate_samples(model, tokenizer, eval_prompts,
                                   max_new_tokens=eval_cfg.get("max_new_tokens", 256))
        scores = score_responses(outputs, rm_model, rm_tokenizer)
        results[name] = {
            "outputs": outputs,
            "scores": scores,
            "avg_score": sum(scores) / len(scores),
            "avg_length": sum(len(o) for o in outputs) / len(outputs),
        }

    # Load timing results if available
    timing_path = os.path.join(output_dir, "training_results.json")
    timing = {}
    if os.path.exists(timing_path):
        with open(timing_path) as f:
            for entry in json.load(f):
                timing[entry["algorithm"]] = entry["training_time_seconds"]

    # Build comparison table
    print("\n" + "=" * 80)
    print("  COMPARISON TABLE")
    print("=" * 80)

    table_data = []
    for name in ["SFT", "PPO", "DPO", "GRPO"]:
        if name not in results:
            continue
        r = results[name]
        row = {
            "Algorithm": name,
            "Avg Reward Score": f"{r['avg_score']:.3f}",
            "Avg Response Length": f"{r['avg_length']:.0f}",
            "Training Time (s)": f"{timing.get(name, 'N/A')}",
            "Models in Memory": {"SFT": "1", "PPO": "4", "DPO": "2", "GRPO": "3"}.get(name, "?"),
            "Online/Offline": {"SFT": "N/A", "PPO": "Online", "DPO": "Offline", "GRPO": "Online"}.get(name, "?"),
        }
        table_data.append(row)

    df = pd.DataFrame(table_data)
    print(df.to_string(index=False))

    # Save detailed results
    detailed_path = os.path.join(output_dir, "evaluation_results.json")
    save_results = {name: {k: v for k, v in data.items() if k != "outputs"}
                    for name, data in results.items()}
    with open(detailed_path, "w") as f:
        json.dump(save_results, f, indent=2)

    # Save comparison outputs
    model_outputs = {name: data["outputs"] for name, data in results.items()}
    compare_outputs(
        eval_prompts,
        model_outputs,
        save_path=os.path.join(output_dir, "all_outputs_comparison.json"),
    )

    print(f"\n📊 Detailed results saved to {detailed_path}")
    print("Next: run 03_write_report.py to generate the report skeleton.\n")


if __name__ == "__main__":
    main()
