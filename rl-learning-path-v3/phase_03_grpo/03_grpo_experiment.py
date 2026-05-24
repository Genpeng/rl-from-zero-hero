"""
Phase 3, Script 3: Group Size and Temperature Ablation
======================================================

This script runs GRPO with different group sizes (2, 4, 8) and optionally
different temperatures to study how these hyperparameters affect training.

Larger groups give more stable advantage estimates but cost more inference
compute. Temperature controls exploration: higher temperature produces more
diverse completions, which may help the model discover correct solutions
but also adds noise.

The experiment trains a fresh model for each configuration and compares
accuracy and training dynamics in a summary table.

Run:
    python 03_grpo_experiment.py --max_samples 500

What to observe:
    - Group size 2: noisiest signal, fastest per step
    - Group size 4: good balance of signal and cost
    - Group size 8: most stable advantages, highest cost per step
    - Temperature effects: higher temp may help or hurt depending on task
"""

import sys
import re
import argparse
from pathlib import Path

# ── Add project root to path so we can import shared utilities ──────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from trl import GRPOTrainer, GRPOConfig
from peft import LoraConfig, PeftModel, TaskType

from shared.model_utils import DEFAULT_MODEL, load_model_and_tokenizer
from shared.data_utils import load_gsm8k_formatted
from shared.eval_utils import evaluate_math_accuracy

# ╭──────────────────────────────────────────────────────────────────────────╮
# │ CONFIG — Adjust these values to experiment                              │
# ╰──────────────────────────────────────────────────────────────────────────╯
CONFIG = {
    "model_name": DEFAULT_MODEL,
    "max_length": 512,                         # Max prompt length (tokens)
    "max_new_tokens": 512,                     # Max tokens to generate per completion
    "max_samples": 500,                        # Training problems per run
    "eval_samples": 50,                        # Test problems for evaluation
    "num_epochs": 1,                           # Epochs per run
    "learning_rate": 5e-6,                     # Learning rate
    "per_device_batch_size": 4,                # Prompts per device per step
    "gradient_accumulation_steps": 2,          # Effective batch = batch_size * grad_accum
    "beta": 0.04,                              # KL penalty coefficient
    "logging_steps": 10,                       # Log every N steps
    # LoRA settings
    "lora_r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    # Ablation settings
    "group_sizes": [2, 4, 8],                  # Group sizes to compare
    "temperatures": [0.7],                     # Temperatures to test
    # Set to [0.5, 0.7, 1.0] to also ablate temperature
}


def parse_args():
    parser = argparse.ArgumentParser(description="GRPO group size ablation experiment")
    parser.add_argument(
        "--max_samples", type=int, default=CONFIG["max_samples"],
        help=f"Training problems per run (default: {CONFIG['max_samples']})"
    )
    parser.add_argument(
        "--eval_samples", type=int, default=CONFIG["eval_samples"],
        help=f"Test problems for evaluation (default: {CONFIG['eval_samples']})"
    )
    parser.add_argument(
        "--ablate_temperature", action="store_true",
        help="Also ablate temperature values [0.5, 0.7, 1.0]"
    )
    return parser.parse_args()


def extract_number_from_response(text):
    """Extract the final numeric answer from model-generated text."""
    if "####" in text:
        after = text.split("####")[-1].strip()
        numbers = re.findall(r"-?\d[\d,]*\.?\d*", after)
        if numbers:
            return numbers[0].replace(",", "")

    answer_match = re.search(
        r"(?:the answer is|answer:)\s*(-?\d[\d,]*\.?\d*)", text, re.IGNORECASE
    )
    if answer_match:
        return answer_match.group(1).replace(",", "")

    numbers = re.findall(r"-?\d[\d,]*\.?\d*", text.replace(",", ""))
    if numbers:
        return numbers[-1]
    return None


def make_reward_function(dataset):
    """Create a reward function for math answer correctness."""
    prompt_to_answer = {}
    for example in dataset:
        prompt_to_answer[example["prompt"]] = example["answer"]

    def reward_func(completions, **kwargs):
        prompts = kwargs.get("prompts", [])
        rewards = []
        for prompt, completion in zip(prompts, completions):
            gold = prompt_to_answer.get(prompt)
            if gold is None:
                rewards.append(0.0)
                continue
            predicted = extract_number_from_response(completion)
            if predicted is not None and predicted == gold:
                rewards.append(1.0)
            else:
                rewards.append(0.0)
        return rewards

    return reward_func


def run_grpo_experiment(
    group_size, temperature, train_dataset, test_dataset, config, run_output_dir
):
    """Run a single GRPO training experiment with the given hyperparameters.

    Returns a dict with the experiment results.
    """
    print(f"\n{'─'*60}")
    print(f"  Running GRPO: G={group_size}, temp={temperature}")
    print(f"{'─'*60}")

    # Create reward function
    reward_func = make_reward_function(train_dataset)

    # LoRA configuration
    lora_config = LoraConfig(
        r=config["lora_r"],
        lora_alpha=config["lora_alpha"],
        lora_dropout=config["lora_dropout"],
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "v_proj"],
    )

    # GRPO configuration for this run
    training_config = GRPOConfig(
        output_dir=run_output_dir,
        num_train_epochs=config["num_epochs"],
        per_device_train_batch_size=config["per_device_batch_size"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        learning_rate=config["learning_rate"],
        num_generations=group_size,
        max_completion_length=config["max_new_tokens"],
        max_prompt_length=config["max_length"],
        temperature=temperature,
        beta=config["beta"],
        logging_steps=config["logging_steps"],
        save_steps=9999,  # Only save at the end
        bf16=True,
        report_to="none",
        remove_unused_columns=False,
    )

    # Train
    trainer = GRPOTrainer(
        model=config["model_name"],
        args=training_config,
        train_dataset=train_dataset,
        reward_funcs=reward_func,
        peft_config=lora_config,
    )

    train_result = trainer.train()
    train_metrics = train_result.metrics

    # Save model for evaluation
    trainer.save_model(run_output_dir)

    # Evaluate: load base model + trained adapter
    print(f"  Evaluating on {len(test_dataset)} test problems...")
    eval_model, tokenizer = load_model_and_tokenizer(
        config["model_name"], device_map="auto"
    )
    eval_model = PeftModel.from_pretrained(eval_model, run_output_dir)
    eval_model = eval_model.merge_and_unload()

    accuracy = evaluate_math_accuracy(
        eval_model, tokenizer, test_dataset, max_samples=len(test_dataset)
    )

    # Clean up GPU memory
    del trainer, eval_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    result = {
        "group_size": group_size,
        "temperature": temperature,
        "accuracy": accuracy,
        "train_loss": train_metrics.get("train_loss", float("nan")),
        "train_runtime": train_metrics.get("train_runtime", 0),
        "train_samples_per_second": train_metrics.get("train_samples_per_second", 0),
    }

    print(f"  Result: accuracy={accuracy:.1%}, loss={result['train_loss']:.4f}")
    return result


def print_comparison_table(results):
    """Print a formatted comparison table of all experiment runs."""
    print(f"\n{'='*80}")
    print(f"  GRPO Ablation — Results Comparison")
    print(f"{'='*80}")
    print(
        f"  {'G':>3} | {'Temp':>5} | {'Accuracy':>9} | "
        f"{'Train Loss':>11} | {'Runtime (s)':>12} | {'Samples/s':>10}"
    )
    print(f"  {'─'*3}-+-{'─'*5}-+-{'─'*9}-+-{'─'*11}-+-{'─'*12}-+-{'─'*10}")

    for r in results:
        print(
            f"  {r['group_size']:>3} | "
            f"{r['temperature']:>5.1f} | "
            f"{r['accuracy']:>8.1%} | "
            f"{r['train_loss']:>11.4f} | "
            f"{r['train_runtime']:>12.1f} | "
            f"{r['train_samples_per_second']:>10.1f}"
        )

    # Highlight best accuracy
    best = max(results, key=lambda r: r["accuracy"])
    print(
        f"\n  Best accuracy: G={best['group_size']}, temp={best['temperature']} "
        f"({best['accuracy']:.1%})"
    )

    # Compute cost-efficiency: accuracy per samples/s
    print(f"\n  Cost-Efficiency Analysis:")
    for r in results:
        efficiency = r["accuracy"] / max(r["train_runtime"], 1) * 100
        print(
            f"    G={r['group_size']:>2}, temp={r['temperature']:.1f}: "
            f"accuracy={r['accuracy']:.1%}, "
            f"runtime={r['train_runtime']:.0f}s"
        )


def main():
    args = parse_args()

    # Determine temperature values
    temperatures = CONFIG["temperatures"]
    if args.ablate_temperature:
        temperatures = [0.5, 0.7, 1.0]

    total_runs = len(CONFIG["group_sizes"]) * len(temperatures)

    print(f"\n{'='*60}")
    print(f"  GRPO Ablation Experiment")
    print(f"  Model:        {CONFIG['model_name']}")
    print(f"  Train samples: {args.max_samples} per run")
    print(f"  Eval samples:  {args.eval_samples}")
    print(f"  Group sizes:   {CONFIG['group_sizes']}")
    print(f"  Temperatures:  {temperatures}")
    print(f"  Total runs:    {total_runs}")
    print(f"{'='*60}")

    # ── Load datasets (shared across all runs) ─────────────────────────────
    print("\n[1/3] Loading datasets...")
    train_dataset = load_gsm8k_formatted(split="train", max_samples=args.max_samples)
    test_dataset = load_gsm8k_formatted(split="test", max_samples=args.eval_samples)
    print(f"  Train: {len(train_dataset)} problems")
    print(f"  Test:  {len(test_dataset)} problems")

    # ── Also evaluate base model for reference ─────────────────────────────
    print("\n[2/3] Evaluating base model (reference)...")
    base_model, tokenizer = load_model_and_tokenizer(
        CONFIG["model_name"], device_map="auto"
    )
    base_accuracy = evaluate_math_accuracy(
        base_model, tokenizer, test_dataset, max_samples=len(test_dataset)
    )
    del base_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ── Run experiments ────────────────────────────────────────────────────
    print(f"\n[3/3] Running {total_runs} experiment(s)...")
    results = []
    run_idx = 0

    for group_size in CONFIG["group_sizes"]:
        for temperature in temperatures:
            run_idx += 1
            print(f"\n  === Run {run_idx}/{total_runs} ===")

            run_output_dir = f"./outputs/grpo_ablation/G{group_size}_T{temperature}"

            result = run_grpo_experiment(
                group_size=group_size,
                temperature=temperature,
                train_dataset=train_dataset,
                test_dataset=test_dataset,
                config=CONFIG,
                run_output_dir=run_output_dir,
            )
            results.append(result)

    # ── Print comparison ───────────────────────────────────────────────────
    print_comparison_table(results)

    # ── Summary ────────────────────────────────────────────────────────────
    print(f"\n{'='*80}")
    print(f"  Experiment Summary")
    print(f"{'='*80}")
    print(f"""
  Base model accuracy:  {base_accuracy:.1%}

  Key observations to look for:

  1. Group size effect:
     - G=2 is the cheapest but noisiest. With only 2 samples per prompt,
       advantages are binary: one is above average, one below (or both
       are the same and contribute zero gradient).
     - G=4 is a common sweet spot. You get enough variety to compute
       meaningful group statistics.
     - G=8 gives the most stable advantages but costs 4x the inference
       of G=2 per prompt. Look at the runtime column.

  2. Temperature effect (if --ablate_temperature was used):
     - Lower temp (0.5): less diverse completions. The group may have
       many duplicates, reducing the effective group size.
     - Higher temp (1.0): more diverse completions, but also more
       random noise. Can help discover correct solutions the model
       would not find with greedy decoding.

  3. The best configuration depends on your compute budget.
     If inference is cheap, prefer larger groups. If it is expensive,
     G=4 with moderate temperature is a safe choice.
""")


if __name__ == "__main__":
    main()
