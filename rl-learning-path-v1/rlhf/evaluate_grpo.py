import re
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


def extract_answer(text: str) -> str:
    numbers = re.findall(r"\d+(?:\.\d+)?", text.replace(",", ""))
    return numbers[-1] if numbers else ""


test_data = load_dataset("openai/gsm8k", "main", split="test[:200]")

for model_name, path in [
    ("Base (no training)", "Qwen/Qwen2-1.5B-Instruct"),
    ("GRPO-aligned", "./grpo_aligned_model"),
]:
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2-1.5B-Instruct")
    model = AutoModelForCausalLM.from_pretrained(path)
    model.eval()

    correct = 0
    for row in test_data:
        prompt = f"Solve this math problem step by step.\n\nQuestion: {row['question']}\n\nAnswer:"
        gt = row["answer"].split("####")[-1].strip()

        inputs = tok(prompt, return_tensors="pt")
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=256, do_sample=False)
        response = tok.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        pred = extract_answer(response)
        if pred == gt:
            correct += 1

    print(f"{model_name}: {correct}/200 = {correct/2:.1f}% accuracy on GSM8K")
