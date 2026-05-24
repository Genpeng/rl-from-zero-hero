"""
Phase 4, Script 1: Run All Experiments
========================================
Train PPO, DPO, and GRPO under matched conditions for fair comparison.
This is the unified training script for the capstone.

Usage:
    python phases/phase_04_capstone/01_run_all_experiments.py [--algo ppo|dpo|grpo|all]
"""

import argparse
import sys
import os
import time
import json
import yaml
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)
from trl import (
    DPOConfig, DPOTrainer,
    GRPOConfig, GRPOTrainer,
)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from shared.model_utils import MODEL_NAME, print_model_info


def load_config():
    config_path = os.path.join(os.path.dirname(__file__), "configs", "capstone_config.yaml")
    with open(config_path) as f:
        return yaml.safe_load(f)


def prepare_dataset(config):
    ds = load_dataset(config["dataset"]["name"], split="train")
    ds = ds.shuffle(seed=42).select(range(min(config["dataset"]["max_samples"], len(ds))))
    return ds


def run_dpo(config, dataset):
    print("\n" + "=" * 60)
    print("  Running DPO Experiment")
    print("=" * 60)

    dpo_cfg = config["dpo"]
    common = config["common"]
    output_dir = os.path.join(config["evaluation"]["output_dir"], "dpo_model")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.bfloat16, device_map="auto",
    )
    ref_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.bfloat16, device_map="auto",
    )

    def format_dpo(example):
        return {
            "prompt": example["prompt"],
            "chosen": example["chosen"],
            "rejected": example["rejected"],
        }

    ds = dataset.map(format_dpo, remove_columns=[
        c for c in dataset.column_names if c not in ["prompt", "chosen", "rejected"]
    ])
    split = ds.train_test_split(test_size=0.1, seed=42)

    training_args = DPOConfig(
        output_dir=output_dir,
        beta=dpo_cfg["beta"],
        learning_rate=common["learning_rate"],
        per_device_train_batch_size=common["per_device_train_batch_size"],
        gradient_accumulation_steps=common["gradient_accumulation_steps"],
        num_train_epochs=common["num_train_epochs"],
        warmup_ratio=common["warmup_ratio"],
        logging_steps=common["logging_steps"],
        save_strategy="steps",
        save_steps=common["save_steps"],
        bf16=common["bf16"],
        max_length=dpo_cfg["max_length"],
        max_prompt_length=dpo_cfg["max_prompt_length"],
        remove_unused_columns=False,
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=training_args,
        train_dataset=split["train"],
        eval_dataset=split["test"],
        processing_class=tokenizer,
    )

    start_time = time.time()
    trainer.train()
    elapsed = time.time() - start_time

    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    return {"algorithm": "DPO", "training_time_seconds": elapsed, "output_dir": output_dir}


def run_grpo(config, dataset):
    print("\n" + "=" * 60)
    print("  Running GRPO Experiment")
    print("=" * 60)

    grpo_cfg = config["grpo"]
    common = config["common"]
    output_dir = os.path.join(config["evaluation"]["output_dir"], "grpo_model")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.bfloat16, device_map="auto",
    )

    rm_name = config["reward_model"]["name"]
    rm_tokenizer = AutoTokenizer.from_pretrained(rm_name)
    if rm_tokenizer.pad_token is None:
        rm_tokenizer.pad_token = rm_tokenizer.eos_token
    rm_model = AutoModelForSequenceClassification.from_pretrained(
        rm_name, num_labels=1, torch_dtype=torch.bfloat16, device_map="auto",
    )
    rm_model.eval()

    def reward_fn(completions, **kwargs):
        rewards = []
        for completion in completions:
            text = completion if isinstance(completion, str) else completion[0]["content"]
            inputs = rm_tokenizer(
                text, return_tensors="pt", truncation=True, max_length=512, padding=True,
            ).to(rm_model.device)
            with torch.no_grad():
                score = rm_model(**inputs).logits.squeeze().cpu().float().item()
            rewards.append(score)
        return rewards

    def format_grpo(example):
        return {"prompt": example["prompt"]}

    ds = dataset.map(format_grpo, remove_columns=[
        c for c in dataset.column_names if c != "prompt"
    ])

    training_args = GRPOConfig(
        output_dir=output_dir,
        num_generations=grpo_cfg["num_generations"],
        learning_rate=common["learning_rate"],
        per_device_train_batch_size=common["per_device_train_batch_size"] // 2,
        gradient_accumulation_steps=common["gradient_accumulation_steps"] * 2,
        num_train_epochs=common["num_train_epochs"],
        warmup_ratio=common["warmup_ratio"],
        logging_steps=common["logging_steps"],
        save_strategy="steps",
        save_steps=common["save_steps"],
        bf16=common["bf16"],
        max_completion_length=grpo_cfg["max_completion_length"],
        remove_unused_columns=False,
    )

    trainer = GRPOTrainer(
        model=model,
        args=training_args,
        train_dataset=ds,
        processing_class=tokenizer,
        reward_funcs=reward_fn,
    )

    start_time = time.time()
    trainer.train()
    elapsed = time.time() - start_time

    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    return {"algorithm": "GRPO", "training_time_seconds": elapsed, "output_dir": output_dir}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", choices=["ppo", "dpo", "grpo", "all"], default="all")
    args = parser.parse_args()

    config = load_config()
    dataset = prepare_dataset(config)

    results = []

    if args.algo in ("dpo", "all"):
        results.append(run_dpo(config, dataset))

    if args.algo in ("grpo", "all"):
        results.append(run_grpo(config, dataset))

    if args.algo in ("ppo", "all"):
        print("\n⚠️  PPO capstone training requires the Phase 1 PPO pipeline.")
        print("Copy your trained PPO model from outputs/phase_01_ppo/ppo_model")
        print("to outputs/phase_04_capstone/ppo_model for comparison.\n")

    # Save timing results
    results_path = os.path.join(config["evaluation"]["output_dir"], "training_results.json")
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n📊 Training results saved to {results_path}")
    print("Next: run 02_evaluate_and_compare.py\n")


if __name__ == "__main__":
    main()
