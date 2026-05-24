import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"


def load_model_and_tokenizer(
    model_name: str = MODEL_NAME,
    device_map: str = "auto",
    torch_dtype=torch.bfloat16,
):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map=device_map,
        torch_dtype=torch_dtype,
    )
    return model, tokenizer


def print_model_info(model, label: str = "Model"):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n{'='*50}")
    print(f"{label}")
    print(f"  Total parameters:     {total:>15,}")
    print(f"  Trainable parameters: {trainable:>15,}")
    print(f"  Frozen parameters:    {total - trainable:>15,}")
    print(f"{'='*50}\n")
