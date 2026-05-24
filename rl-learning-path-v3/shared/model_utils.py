"""Shared model loading and device utilities."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSequenceClassification
from peft import LoraConfig, get_peft_model, TaskType


DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B"


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_model_and_tokenizer(model_name=DEFAULT_MODEL, device_map="auto"):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map=device_map,
        trust_remote_code=True,
    )
    return model, tokenizer


def load_reward_model(model_name=DEFAULT_MODEL, num_labels=1, device_map="auto"):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        torch_dtype=torch.bfloat16,
        device_map=device_map,
        trust_remote_code=True,
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    return model, tokenizer


def get_lora_config(r=16, lora_alpha=32, task_type=TaskType.CAUSAL_LM):
    return LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=0.05,
        bias="none",
        task_type=task_type,
        target_modules=["q_proj", "v_proj"],
    )


def apply_lora(model, lora_config=None):
    if lora_config is None:
        lora_config = get_lora_config()
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model
