from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
from datasets import load_dataset
import torch

model_name = "Qwen/Qwen2-0.5B-Instruct"
reward_model_path = "./reward_model_final"

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

policy = AutoModelForCausalLMWithValueHead.from_pretrained(model_name)
ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(model_name)
reward_model = AutoModelForSequenceClassification.from_pretrained(reward_model_path)
reward_model.eval()

config = PPOConfig(
    model_name=model_name,
    learning_rate=1e-5,
    batch_size=16,
    mini_batch_size=4,
    gradient_accumulation_steps=1,
    ppo_epochs=4,
    kl_penalty="kl",
    init_kl_coef=0.2,
    target_kl=6.0,
    log_with=None,
)

dataset = load_dataset("Anthropic/hh-rlhf", split="train[:1000]")
prompts = [row["chosen"].split("\n\nAssistant:")[0] + "\n\nAssistant:" for row in dataset]

ppo_trainer = PPOTrainer(config, policy, ref_model, tokenizer)

generation_kwargs = {
    "max_new_tokens": 128,
    "do_sample": True,
    "temperature": 0.7,
    "pad_token_id": tokenizer.eos_token_id,
}

for step, batch_prompts in enumerate(
    [prompts[i:i+16] for i in range(0, min(len(prompts), 160), 16)]
):
    query_tensors = [
        tokenizer(p, return_tensors="pt").input_ids[0] for p in batch_prompts
    ]
    response_tensors = ppo_trainer.generate(query_tensors, **generation_kwargs)

    responses = [tokenizer.decode(r, skip_special_tokens=True) for r in response_tensors]

    with torch.no_grad():
        inputs = tokenizer(responses, return_tensors="pt", truncation=True,
                           max_length=512, padding=True)
        rewards = reward_model(**inputs).logits.squeeze(-1)
        reward_list = [r for r in rewards]

    stats = ppo_trainer.step(query_tensors, response_tensors, reward_list)

    if step % 2 == 0:
        mean_reward = sum(r.item() for r in reward_list) / len(reward_list)
        kl = stats.get("objective/kl", 0)
        print(f"Step {step} | mean_reward={mean_reward:.3f} | kl={kl:.3f}")

ppo_trainer.save_pretrained("./ppo_aligned_model")
print("PPO-aligned model saved.")
