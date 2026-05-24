"""Shared dataset loading and formatting utilities."""

from datasets import load_dataset


def load_hh_rlhf(split="train", max_samples=None):
    """Load Anthropic HH-RLHF dataset for PPO/DPO training."""
    dataset = load_dataset("Anthropic/hh-rlhf", split=split)
    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
    return dataset


def format_hh_rlhf_for_dpo(example):
    """Format HH-RLHF into prompt/chosen/rejected for DPO."""
    chosen = example["chosen"]
    rejected = example["rejected"]

    # HH-RLHF format: "\n\nHuman: ...\n\nAssistant: ..."
    # Split at last Assistant turn to get prompt + response
    prompt = chosen[: chosen.rfind("\n\nAssistant: ") + len("\n\nAssistant: ")]
    chosen_response = chosen[chosen.rfind("\n\nAssistant: ") + len("\n\nAssistant: "):]
    rejected_response = rejected[rejected.rfind("\n\nAssistant: ") + len("\n\nAssistant: "):]

    return {
        "prompt": prompt.strip(),
        "chosen": chosen_response.strip(),
        "rejected": rejected_response.strip(),
    }


def load_hh_rlhf_for_dpo(split="train", max_samples=None):
    """Load and format HH-RLHF for DPO training."""
    dataset = load_hh_rlhf(split=split, max_samples=max_samples)
    dataset = dataset.map(format_hh_rlhf_for_dpo)
    return dataset


def load_gsm8k(split="train", max_samples=None):
    """Load GSM8K math reasoning dataset for GRPO."""
    dataset = load_dataset("openai/gsm8k", "main", split=split)
    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
    return dataset


def extract_gsm8k_answer(answer_text):
    """Extract the final numeric answer from GSM8K answer text.

    GSM8K answers end with '#### <number>'.
    """
    if "####" in answer_text:
        return answer_text.split("####")[-1].strip().replace(",", "")
    return answer_text.strip()


def format_gsm8k_prompt(example):
    """Format a GSM8K example into a prompt for the model."""
    return {
        "prompt": f"Solve this math problem step by step.\n\nQuestion: {example['question']}\n\nAnswer:",
        "answer": extract_gsm8k_answer(example["answer"]),
    }


def load_gsm8k_formatted(split="train", max_samples=None):
    """Load and format GSM8K for GRPO training."""
    dataset = load_gsm8k(split=split, max_samples=max_samples)
    dataset = dataset.map(format_gsm8k_prompt)
    return dataset
