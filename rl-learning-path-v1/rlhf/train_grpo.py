import re
import torch
from datasets import load_dataset
from transformers import AutoTokenizer
from trl import GRPOTrainer, GRPOConfig

model_name = "Qwen/Qwen2-1.5B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

dataset = load_dataset("openai/gsm8k", "main", split="train[:2000]")

def format_prompt(row):
    return {
        "prompt": f"Solve this math problem step by step.\n\nQuestion: {row['question']}\n\nAnswer:",
        "answer": row["answer"].split("####")[-1].strip(),
    }

dataset = dataset.map(format_prompt)


def extract_answer(text: str) -> str:
    """Extract the last number from model output."""
    numbers = re.findall(r"\d+(?:\.\d+)?", text.replace(",", ""))
    return numbers[-1] if numbers else ""


def accuracy_reward(completions, prompts, answer, **kwargs) -> list[float]:
    """Reward = 1.0 if model's final number matches ground truth, else 0.0."""
    rewards = []
    for completion, gt in zip(completions, answer):
        text = completion[0]["content"] if isinstance(completion[0], dict) else completion
        pred = extract_answer(text)
        rewards.append(1.0 if pred == gt else 0.0)
    return rewards


def format_reward(completions, **kwargs) -> list[float]:
    """Small reward for using chain-of-thought structure."""
    rewards = []
    for completion in completions:
        text = completion[0]["content"] if isinstance(completion[0], dict) else completion
        has_steps = any(word in text.lower() for word in ["step", "first", "then", "therefore"])
        rewards.append(0.1 if has_steps else 0.0)
    return rewards


def combined_reward(completions, prompts, answer, **kwargs) -> list[float]:
    acc = accuracy_reward(completions, prompts, answer, **kwargs)
    fmt = format_reward(completions, **kwargs)
    return [a + f for a, f in zip(acc, fmt)]


config = GRPOConfig(
    output_dir="./grpo_model_output",
    per_device_train_batch_size=2,
    num_train_epochs=1,
    learning_rate=5e-6,
    num_generations=8,
    max_new_tokens=512,
    logging_steps=10,
    save_steps=200,
    fp16=True,
    report_to="none",
    kl_coef=0.04,
)

trainer = GRPOTrainer(
    model=model_name,
    reward_funcs=combined_reward,
    args=config,
    train_dataset=dataset,
    tokenizer=tokenizer,
)

trainer.train()
trainer.save_model("./grpo_aligned_model")
print("GRPO model saved.")
