"""
Phase 3, Script 1: GRPO Training on GSM8K
==========================================

This script trains a language model on GSM8K math problems using Group Relative
Policy Optimization (GRPO). For each math problem, the model generates a group
of candidate solutions. A reward function checks each solution's correctness
(does the final number match the gold answer?). GRPO computes advantages
relative to the group and updates the policy to favor correct solutions.

The pipeline:
    1. Load and format GSM8K math problems
    2. Load the base model with LoRA for efficient training
    3. Define a reward function: correct answer -> 1.0, incorrect -> 0.0
    4. Run GRPO training via TRL's GRPOTrainer
    5. Save the trained model

Run:
    python 01_grpo_training.py \\
        --max_samples 1000 \\
        --num_generations 4 \\
        --output_dir ./outputs/grpo_model

What to observe:
    - Training accuracy (proportion of correct answers) should increase
    - The reward is binary: 1.0 or 0.0 per completion
    - With a small model (0.5B), absolute accuracy will be modest, but
      you should see improvement from the baseline
"""

import sys
import re
import argparse
from pathlib import Path

# ── Add project root to path so we can import shared utilities ──────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from trl import GRPOTrainer, GRPOConfig
from peft import LoraConfig, TaskType

from shared.model_utils import DEFAULT_MODEL, get_device
from shared.data_utils import load_gsm8k_formatted, extract_gsm8k_answer

# ╭──────────────────────────────────────────────────────────────────────────╮
# │ CONFIG — Adjust these values to experiment                              │
# ╰──────────────────────────────────────────────────────────────────────────╯
CONFIG = {
    "model_name": DEFAULT_MODEL,               # Base model to fine-tune
    "max_length": 512,                         # Max prompt length (tokens)
    "max_new_tokens": 512,                     # Max tokens to generate per completion
    "max_samples": 1000,                       # Number of training problems
    "num_generations": 4,                      # Group size G (completions per prompt)
    "num_epochs": 1,                           # Training epochs
    "learning_rate": 5e-6,                     # Learning rate for LoRA parameters
    "per_device_batch_size": 4,                # Prompts per device per step
    "gradient_accumulation_steps": 2,          # Effective batch = batch_size * grad_accum
    "temperature": 0.7,                        # Sampling temperature for generation
    "beta": 0.04,                              # KL penalty coefficient
    "logging_steps": 10,                       # Log metrics every N steps
    "save_steps": 200,                         # Save checkpoint every N steps
    # LoRA settings
    "lora_r": 16,                              # LoRA rank
    "lora_alpha": 32,                          # LoRA alpha (scaling factor)
    "lora_dropout": 0.05,                      # LoRA dropout
}


def parse_args():
    parser = argparse.ArgumentParser(description="GRPO training on GSM8K math problems")
    parser.add_argument(
        "--max_samples", type=int, default=CONFIG["max_samples"],
        help=f"Maximum number of training problems (default: {CONFIG['max_samples']})"
    )
    parser.add_argument(
        "--output_dir", type=str, default="./outputs/grpo_model",
        help="Directory to save the trained model (default: ./outputs/grpo_model)"
    )
    parser.add_argument(
        "--num_generations", type=int, default=CONFIG["num_generations"],
        help=f"Group size: completions per prompt (default: {CONFIG['num_generations']})"
    )
    parser.add_argument(
        "--num_epochs", type=int, default=CONFIG["num_epochs"],
        help=f"Number of training epochs (default: {CONFIG['num_epochs']})"
    )
    return parser.parse_args()


def extract_number_from_response(text):
    """Extract the final numeric answer from model-generated text.

    Tries several patterns:
        1. '#### <number>' (GSM8K-style)
        2. 'the answer is <number>'
        3. Last number in the text
    """
    # Pattern 1: GSM8K-style delimiter
    if "####" in text:
        after = text.split("####")[-1].strip()
        numbers = re.findall(r"-?\d[\d,]*\.?\d*", after)
        if numbers:
            return numbers[0].replace(",", "")

    # Pattern 2: Natural language answer
    answer_match = re.search(r"(?:the answer is|answer:)\s*(-?\d[\d,]*\.?\d*)", text, re.IGNORECASE)
    if answer_match:
        return answer_match.group(1).replace(",", "")

    # Pattern 3: Last number in text
    numbers = re.findall(r"-?\d[\d,]*\.?\d*", text.replace(",", ""))
    if numbers:
        return numbers[-1]

    return None


def make_reward_function(dataset):
    """Create a reward function that checks math answer correctness.

    Returns a function compatible with TRL's GRPOTrainer reward_func interface.
    The function receives a list of prompt-completion pairs and returns a list
    of reward floats.
    """
    # Build a lookup from prompt text to gold answer
    prompt_to_answer = {}
    for example in dataset:
        prompt_to_answer[example["prompt"]] = example["answer"]

    def reward_func(completions, **kwargs):
        """Score completions by checking if the extracted answer matches gold.

        Args:
            completions: List of generated completion strings.
            **kwargs: Additional keyword arguments from GRPOTrainer, including
                      'prompts' (list of prompt strings).

        Returns:
            List of float rewards (1.0 for correct, 0.0 for incorrect).
        """
        prompts = kwargs.get("prompts", [])
        rewards = []

        for prompt, completion in zip(prompts, completions):
            # Look up the gold answer for this prompt
            gold = prompt_to_answer.get(prompt)
            if gold is None:
                rewards.append(0.0)
                continue

            # Extract the predicted answer from the completion
            predicted = extract_number_from_response(completion)

            if predicted is not None and predicted == gold:
                rewards.append(1.0)
            else:
                rewards.append(0.0)

        return rewards

    return reward_func


def main():
    args = parse_args()
    print(f"\n{'='*60}")
    print(f"  GRPO Training on GSM8K")
    print(f"  Model:           {CONFIG['model_name']}")
    print(f"  Samples:         {args.max_samples}")
    print(f"  Group size (G):  {args.num_generations}")
    print(f"  Epochs:          {args.num_epochs}")
    print(f"  KL beta:         {CONFIG['beta']}")
    print(f"  Output:          {args.output_dir}")
    print(f"{'='*60}\n")

    # ── Step 1: Load and format dataset ────────────────────────────────────
    print("[1/4] Loading GSM8K dataset...")
    dataset = load_gsm8k_formatted(split="train", max_samples=args.max_samples)
    print(f"  Loaded {len(dataset)} math problems")
    print(f"  Example prompt:  {dataset[0]['prompt'][:80]}...")
    print(f"  Example answer:  {dataset[0]['answer']}")

    # ── Step 2: Create the reward function ─────────────────────────────────
    print("\n[2/4] Setting up reward function...")
    reward_func = make_reward_function(dataset)
    print("  Reward function: 1.0 if correct answer, 0.0 otherwise")

    # ── Step 3: Configure GRPO training ────────────────────────────────────
    print("\n[3/4] Configuring GRPO trainer...")

    # LoRA configuration for parameter-efficient training
    lora_config = LoraConfig(
        r=CONFIG["lora_r"],
        lora_alpha=CONFIG["lora_alpha"],
        lora_dropout=CONFIG["lora_dropout"],
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "v_proj"],
    )

    # GRPO training configuration
    training_config = GRPOConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=CONFIG["gradient_accumulation_steps"],
        learning_rate=CONFIG["learning_rate"],
        num_generations=args.num_generations,
        max_completion_length=CONFIG["max_new_tokens"],
        max_prompt_length=CONFIG["max_length"],
        temperature=CONFIG["temperature"],
        beta=CONFIG["beta"],
        logging_steps=CONFIG["logging_steps"],
        save_steps=CONFIG["save_steps"],
        bf16=True,
        report_to="none",
        remove_unused_columns=False,
        log_completions=True,
    )

    # ── Step 4: Train ──────────────────────────────────────────────────────
    print("\n[4/4] Starting GRPO training...")
    print(f"  This will generate {args.num_generations} completions per prompt.")
    print(f"  The reward function scores each completion independently.")
    print(f"  Advantages are computed relative to the group.\n")

    trainer = GRPOTrainer(
        model=CONFIG["model_name"],
        args=training_config,
        train_dataset=dataset,
        reward_funcs=reward_func,
        peft_config=lora_config,
    )

    train_result = trainer.train()

    # ── Save the model ─────────────────────────────────────────────────────
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(output_path))
    print(f"\n  Model saved to: {args.output_dir}")

    # ── Print training summary ─────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Training Summary")
    print(f"{'='*60}")
    metrics = train_result.metrics
    for key, value in sorted(metrics.items()):
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")

    print(f"\n  Next step: Run 02_grpo_evaluation.py to compare base vs GRPO model.")
    print()


if __name__ == "__main__":
    main()
