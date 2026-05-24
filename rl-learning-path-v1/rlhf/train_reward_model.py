from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from trl import RewardTrainer, RewardConfig

model_name = "Qwen/Qwen2-0.5B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(
    model_name, num_labels=1
)

dataset = load_dataset("Anthropic/hh-rlhf", split="train[:5000]")

def preprocess(batch):
    chosen = tokenizer(batch["chosen"], truncation=True, max_length=512)
    rejected = tokenizer(batch["rejected"], truncation=True, max_length=512)
    return {
        "input_ids_chosen": chosen["input_ids"],
        "attention_mask_chosen": chosen["attention_mask"],
        "input_ids_rejected": rejected["input_ids"],
        "attention_mask_rejected": rejected["attention_mask"],
    }

dataset = dataset.map(preprocess, batched=True)

training_args = RewardConfig(
    output_dir="./reward_model_output",
    per_device_train_batch_size=4,
    num_train_epochs=1,
    learning_rate=2e-5,
    logging_steps=50,
    save_steps=500,
    fp16=True,
    report_to="none",
)

trainer = RewardTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    tokenizer=tokenizer,
)

trainer.train()
trainer.save_model("./reward_model_final")
print("Reward model saved.")
