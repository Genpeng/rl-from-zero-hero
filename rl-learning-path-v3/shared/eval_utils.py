"""Shared evaluation and generation utilities."""

import re
import torch
import matplotlib.pyplot as plt
from pathlib import Path


def generate_responses(model, tokenizer, prompts, max_new_tokens=256, temperature=0.7):
    """Generate responses for a list of prompts."""
    model.eval()
    responses = []
    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
                pad_token_id=tokenizer.pad_token_id,
            )

        generated = outputs[0][inputs["input_ids"].shape[1]:]
        response = tokenizer.decode(generated, skip_special_tokens=True)
        responses.append(response)

    return responses


def compare_outputs(prompts, responses_a, responses_b, label_a="Model A", label_b="Model B"):
    """Print side-by-side comparison of two models' outputs."""
    for i, prompt in enumerate(prompts):
        print(f"\n{'='*80}")
        print(f"PROMPT {i+1}: {prompt[:100]}...")
        print(f"\n--- {label_a} ---")
        print(responses_a[i][:300])
        print(f"\n--- {label_b} ---")
        print(responses_b[i][:300])
    print(f"\n{'='*80}")


def plot_reward_distribution(rewards, title="Reward Distribution", save_path=None):
    """Plot histogram of reward scores."""
    plt.figure(figsize=(8, 5))
    plt.hist(rewards, bins=30, edgecolor="black", alpha=0.7)
    plt.xlabel("Reward")
    plt.ylabel("Count")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved plot to {save_path}")
    plt.show()


def plot_training_curve(values, title="Training Curve", xlabel="Step", ylabel="Value", save_path=None):
    """Plot a training metric over steps."""
    plt.figure(figsize=(10, 5))
    plt.plot(values)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved plot to {save_path}")
    plt.show()


def extract_number_from_text(text):
    """Extract the last number from generated text (for math evaluation)."""
    numbers = re.findall(r"-?\d+\.?\d*", text.replace(",", ""))
    if numbers:
        return numbers[-1]
    return None


def evaluate_math_accuracy(model, tokenizer, dataset, max_samples=100):
    """Evaluate model accuracy on math problems (GSM8K)."""
    correct = 0
    total = min(max_samples, len(dataset))

    prompts = [dataset[i]["prompt"] for i in range(total)]
    gold_answers = [dataset[i]["answer"] for i in range(total)]

    responses = generate_responses(model, tokenizer, prompts, max_new_tokens=512, temperature=0.0)

    for response, gold in zip(responses, gold_answers):
        predicted = extract_number_from_text(response)
        if predicted and predicted == gold:
            correct += 1

    accuracy = correct / total
    print(f"Math Accuracy: {correct}/{total} = {accuracy:.1%}")
    return accuracy
