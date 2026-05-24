from transformers import AutoModelForCausalLM, AutoTokenizer

prompts = [
    "Human: Explain quantum entanglement simply.\n\nAssistant:",
    "Human: How do I deal with a difficult coworker?\n\nAssistant:",
    "Human: What are the pros and cons of vegetarianism?\n\nAssistant:",
]

tok = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B-Instruct")

models = {
    "Base": "Qwen/Qwen2-0.5B-Instruct",
    "PPO": "./ppo_aligned_model",
    "DPO": "./dpo_aligned_model",
}

for prompt in prompts:
    print(f"\n{'='*60}")
    print(f"PROMPT: {prompt}\n")
    for name, path in models.items():
        m = AutoModelForCausalLM.from_pretrained(path)
        inputs = tok(prompt, return_tensors="pt")
        out = m.generate(**inputs, max_new_tokens=150, do_sample=True, temperature=0.7)
        response = tok.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        print(f"--- {name} ---")
        print(response)
