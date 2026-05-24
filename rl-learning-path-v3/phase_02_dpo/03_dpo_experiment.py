"""
Phase 2, Script 3: DPO Beta Ablation Experiment
================================================

This script runs DPO with different beta values and compares the results.
It demonstrates the same tradeoff you explored with KL coefficients in PPO
(Phase 1), but now through the DPO lens:

    Low beta  (0.01) -> aggressive optimization, large policy changes
    Mid beta  (0.1)  -> balanced, standard default
    High beta (0.5)  -> conservative, policy barely changes

The experiment trains a fresh model for each beta value, tracks the training
loss, and generates sample outputs for qualitative comparison.

Run:
    python 03_dpo_experiment.py --max_samples 1000

What to observe:
    - Compare training loss across beta values
    - Inspect sample outputs: does low beta produce unusual text?
    - Does high beta keep outputs nearly identical to the base model?
"""

import sys
import argparse
from pathlib import Path

# ── Add project root to path so we can import shared utilities ──────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from trl import DPOTrainer, DPOConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

from shared.model_utils import DEFAULT_MODEL, get_lora_config
from shared.data_utils import load_hh_rlhf_for_dpo
from shared.eval_utils import generate_responses

# ╭──────────────────────────────────────────────────────────────────────────╮
# │ CONFIG — Adjust these values to experiment                              │
# ╰──────────────────────────────────────────────────────────────────────────╯
CONFIG = {
    "model_name": DEFAULT_MODEL,
    "max_length": 512,                         # Max sequence length for DPO
    "max_prompt_length": 256,                  # Max prompt token length
    "max_new_tokens": 128,                     # Max tokens to generate for demos
    "per_device_batch_size": 4,                # Batch size per device
    "gradient_accumulation_steps": 4,          # Effective batch = batch * grad_accum
    "learning_rate": 5e-5,                     # Learning rate for LoRA params
    "warmup_ratio": 0.1,                       # Warmup fraction of total steps
    "num_epochs": 1,                           # Training epochs per run
    "temperature": 0.7,                        # Sampling temperature for generation
    "logging_steps": 10,                       # Log metrics every N steps
    # Beta values to compare
    "beta_values": [0.01, 0.1, 0.5],
}

# Fixed prompts for qualitative comparison across runs
EVAL_PROMPTS = [
    "\n\nHuman: How can I be more productive at work?\n\nAssistant:",
    "\n\nHuman: What should I know about investing money?\n\nAssistant:",
    "\n\nHuman: Can you explain climate change simply?\n\nAssistant:",
]


def parse_args():
    parser = argparse.ArgumentParser(description="DPO beta ablation experiment")
    parser.add_argument(
        "--max_samples", type=int, default=1000,
        help="Maximum number of preference pairs per run (default: 1000)"
    )
    return parser.parse_args()


def run_dpo_with_beta(beta, dataset, config):
    """Run a single DPO training run with a given beta value.

    Returns a dict with training metrics and sample outputs.
    """
    print(f"\n{'─'*60}")
    print(f"  Running DPO with beta = {beta}")
    print(f"{'─'*60}")

    # Fresh model for each run (so comparisons are fair)
    tokenizer = AutoTokenizer.from_pretrained(
        config["model_name"], trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        config["model_name"],
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    lora_config = get_lora_config(r=16, lora_alpha=32)

    # Configure DPO training
    output_dir = f"./outputs/dpo_beta_{beta}"
    training_args = DPOConfig(
        output_dir=output_dir,
        num_train_epochs=config["num_epochs"],
        per_device_train_batch_size=config["per_device_batch_size"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        learning_rate=config["learning_rate"],
        warmup_ratio=config["warmup_ratio"],
        beta=beta,
        max_length=config["max_length"],
        max_prompt_length=config["max_prompt_length"],
        logging_steps=config["logging_steps"],
        save_strategy="no",  # Don't save intermediate checkpoints
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

    # Train
    train_result = trainer.train()
    metrics = train_result.metrics

    # Extract log history for loss tracking
    log_history = trainer.state.log_history
    train_losses = [
        entry["loss"] for entry in log_history
        if "loss" in entry
    ]

    # Generate sample outputs
    sample_responses = generate_responses(
        trainer.model,
        tokenizer,
        EVAL_PROMPTS,
        max_new_tokens=config["max_new_tokens"],
        temperature=config["temperature"],
    )

    # Collect results
    result = {
        "beta": beta,
        "train_loss": metrics.get("train_loss", float("nan")),
        "train_runtime": metrics.get("train_runtime", 0.0),
        "train_samples_per_second": metrics.get("train_samples_per_second", 0.0),
        "loss_history": train_losses,
        "sample_responses": sample_responses,
    }

    print(f"  Final loss: {result['train_loss']:.4f}")
    print(f"  Runtime:    {result['train_runtime']:.1f}s")

    # Clean up to free memory for next run
    del trainer, model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result


def print_comparison_table(results):
    """Print a formatted comparison table of all runs."""
    print(f"\n{'='*80}")
    print(f"  DPO Beta Ablation — Results Comparison")
    print(f"{'='*80}")
    print(f"  {'Beta':>8} | {'Train Loss':>12} | {'Runtime (s)':>12} | {'Samples/sec':>12}")
    print(f"  {'─'*8}-+-{'─'*12}-+-{'─'*12}-+-{'─'*12}")

    for r in results:
        print(
            f"  {r['beta']:>8.2f} | "
            f"{r['train_loss']:>12.4f} | "
            f"{r['train_runtime']:>12.1f} | "
            f"{r['train_samples_per_second']:>12.2f}"
        )

    # Highlight key observations
    losses = [r["train_loss"] for r in results]
    lowest_loss_idx = losses.index(min(losses))
    highest_loss_idx = losses.index(max(losses))

    print(f"\n  Lowest loss:  beta = {results[lowest_loss_idx]['beta']} "
          f"(loss = {results[lowest_loss_idx]['train_loss']:.4f})")
    print(f"  Highest loss: beta = {results[highest_loss_idx]['beta']} "
          f"(loss = {results[highest_loss_idx]['train_loss']:.4f})")


def print_loss_trajectories(results):
    """Print ASCII visualization of loss trajectories."""
    print(f"\n{'='*80}")
    print(f"  Loss Trajectories")
    print(f"{'='*80}")

    for r in results:
        losses = r["loss_history"]
        if not losses:
            print(f"\n  beta={r['beta']}: no loss data recorded")
            continue

        print(f"\n  beta={r['beta']} ({len(losses)} logged steps):")
        print(f"    Start: {losses[0]:.4f}  ->  End: {losses[-1]:.4f}")

        # Simple ASCII sparkline
        if len(losses) >= 2:
            min_l = min(losses)
            max_l = max(losses)
            range_l = max_l - min_l if max_l > min_l else 1.0
            # Show at most 50 data points
            step = max(1, len(losses) // 50)
            sampled = losses[::step]
            sparkline = ""
            for l in sampled:
                normalized = (l - min_l) / range_l
                bar_chars = " ▁▂▃▄▅▆▇█"
                idx = min(int(normalized * (len(bar_chars) - 1)), len(bar_chars) - 1)
                sparkline += bar_chars[idx]
            print(f"    Loss:  [{sparkline}]")


def print_sample_comparison(results):
    """Print sample outputs from each beta configuration."""
    print(f"\n{'='*80}")
    print(f"  Sample Outputs — Qualitative Comparison")
    print(f"{'='*80}")

    for i, prompt in enumerate(EVAL_PROMPTS):
        print(f"\n  PROMPT: {prompt.strip()[:80]}...")
        for r in results:
            print(f"\n    [beta={r['beta']}]:")
            response = r["sample_responses"][i]
            # Show first 200 chars to keep output manageable
            truncated = response[:200] + ("..." if len(response) > 200 else "")
            for line in truncated.split("\n"):
                print(f"      {line}")
        print(f"  {'─'*70}")


def main():
    args = parse_args()
    print(f"\n{'='*60}")
    print(f"  DPO Beta Ablation Experiment")
    print(f"  Model:       {CONFIG['model_name']}")
    print(f"  Samples:     {args.max_samples}")
    print(f"  Beta values: {CONFIG['beta_values']}")
    print(f"{'='*60}")

    # ── Load dataset (shared across runs) ──────────────────────────────────
    print("\n[1/3] Loading and formatting preference data...")
    dataset = load_hh_rlhf_for_dpo(split="train", max_samples=args.max_samples)
    print(f"  Loaded {len(dataset)} preference pairs")

    # ── Run DPO with each beta value ───────────────────────────────────────
    print("\n[2/3] Running DPO with different beta values...")
    results = []
    for beta in CONFIG["beta_values"]:
        result = run_dpo_with_beta(
            beta=beta,
            dataset=dataset,
            config=CONFIG,
        )
        results.append(result)

    # ── Print comparisons ──────────────────────────────────────────────────
    print("\n[3/3] Comparing results...")
    print_comparison_table(results)
    print_loss_trajectories(results)
    print_sample_comparison(results)

    # ── Summary ────────────────────────────────────────────────────────────
    print(f"\n{'='*80}")
    print(f"  Experiment Summary")
    print(f"{'='*80}")
    print(f"""
  Key takeaways to look for:

  1. beta=0.01 (aggressive): The model has maximum freedom to deviate from the
     reference. Training loss is typically lowest because the model can push
     log-probability margins as far as it wants. But inspect the outputs — are
     they coherent? Has the model overfitted to the preference signal?

  2. beta=0.1 (balanced): The standard default. The model learns meaningful
     preferences while staying reasonably close to the reference. Outputs
     should be improved but still natural.

  3. beta=0.5 (conservative): The model barely moves from the reference.
     Training loss may be higher because the sigmoid saturates quickly with a
     large beta, but the actual policy changes are minimal. Outputs should look
     very similar to the base model.

  Compare this to your PPO KL ablation from Phase 1:
  - PPO's beta=0.0 (no KL)  is analogous to DPO's beta=0.01 (very low)
  - PPO's beta=0.05         is analogous to DPO's beta=0.1
  - PPO's beta=0.2          is analogous to DPO's beta=0.5

  The tradeoff is the same: more freedom leads to higher optimization signal
  but risks degenerate outputs. More constraint is safer but slower to learn.
""")


if __name__ == "__main__":
    main()
