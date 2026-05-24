import json
import os
from datetime import datetime

import torch


def generate_samples(
    model,
    tokenizer,
    prompts: list[str],
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    num_return_sequences: int = 1,
) -> list[str]:
    model.eval()
    outputs = []
    for prompt in prompts:
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            generated = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                num_return_sequences=num_return_sequences,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
            )
        decoded = tokenizer.decode(generated[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        outputs.append(decoded)
    return outputs


def compare_outputs(
    prompts: list[str],
    model_outputs: dict[str, list[str]],
    save_path: str | None = None,
):
    results = []
    for i, prompt in enumerate(prompts):
        entry = {"prompt": prompt}
        for model_name, outputs in model_outputs.items():
            entry[model_name] = outputs[i]
        results.append(entry)

        print(f"\n{'='*60}")
        print(f"Prompt: {prompt[:100]}...")
        for model_name, outputs in model_outputs.items():
            print(f"\n--- {model_name} ---")
            print(outputs[i][:300])

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to {save_path}")

    return results


def log_training_metrics(metrics: dict, phase: str, step: int):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] Phase: {phase} | Step: {step}")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")
