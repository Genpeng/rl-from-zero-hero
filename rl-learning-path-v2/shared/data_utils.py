from datasets import load_dataset


PREFERENCE_DATASET = "argilla/ultrafeedback-binarized-preferences-cleaned"


def load_preference_dataset(
    dataset_name: str = PREFERENCE_DATASET,
    split: str = "train",
    max_samples: int = 2000,
):
    ds = load_dataset(dataset_name, split=split)
    if max_samples and len(ds) > max_samples:
        ds = ds.shuffle(seed=42).select(range(max_samples))
    return ds


def format_for_dpo(example):
    return {
        "prompt": example["prompt"],
        "chosen": example["chosen"],
        "rejected": example["rejected"],
    }


def format_for_ppo(example):
    return {
        "query": example["prompt"],
    }


def format_for_grpo(example):
    return {
        "prompt": example["prompt"],
    }
