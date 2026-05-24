from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOTrainer, DPOConfig

model_name = "Qwen/Qwen2-0.5B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(model_name)

dataset = load_dataset("Anthropic/hh-rlhf", split="train[:2000]")

def format_dataset(row):
    parts = row["chosen"].split("\n\nHuman: ")
    if len(parts) > 1:
        last_human = parts[-1].split("\n\nAssistant:")[0]
        prompt = f"Human: {last_human}\n\nAssistant:"
    else:
        prompt = ""
    chosen = row["chosen"].split("\n\nAssistant:")[-1].strip()
    rejected = row["rejected"].split("\n\nAssistant:")[-1].strip()
    return {"prompt": prompt, "chosen": chosen, "rejected": rejected}

dataset = dataset.map(format_dataset)
dataset = dataset.filter(lambda x: len(x["prompt"]) > 10 and len(x["chosen"]) > 5)

training_args = DPOConfig(
    output_dir="./dpo_model_output",
    per_device_train_batch_size=4,
    num_train_epochs=1,
    learning_rate=5e-6,
    beta=0.1,
    logging_steps=50,
    save_steps=500,
    fp16=True,
    report_to="none",
    max_length=512,
    max_prompt_length=256,
)

trainer = DPOTrainer(
    model=model,
    ref_model=None,
    args=training_args,
    train_dataset=dataset,
    tokenizer=tokenizer,
)

trainer.train()
trainer.save_model("./dpo_aligned_model")
print("DPO model saved.")
