# RL Learning Path Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a complete, self-contained learning path with theory guides, runnable training scripts, and a capstone comparison framework — everything a learner needs to go from zero RL knowledge to applying PPO, DPO, and GRPO for LLM alignment at work.

**Architecture:** A `phases/` directory with one subdirectory per phase (00–04). Each phase contains a `README.md` theory guide and Python scripts/configs for hands-on labs. A shared `shared/` directory holds common utilities (model loading, data helpers, metrics). A root `requirements.txt` pins all dependencies. The capstone (Phase 4) pulls from all prior phases into a unified comparison notebook.

**Tech Stack:** Python 3.10+, PyTorch, TRL (Transformer Reinforcement Learning), Hugging Face Transformers/Datasets/Accelerate, Weights & Biases (optional), Qwen2.5-1.5B as base model.

---

## File Structure

```
rl-learning-path/
├── requirements.txt                        # All dependencies pinned
├── setup_env.sh                            # One-command environment setup
├── shared/
│   ├── __init__.py
│   ├── model_utils.py                      # Load model/tokenizer, print param counts
│   ├── data_utils.py                       # Load & format datasets for each trainer
│   └── eval_utils.py                       # Generate samples, compare outputs, log metrics
├── phases/
│   ├── phase_00_foundation/
│   │   └── README.md                       # Theory guide: RLHF pipeline, RL concepts, reward models
│   ├── phase_01_ppo/
│   │   ├── README.md                       # Theory guide: policy gradient → PPO → PPO for LLMs
│   │   ├── 01_train_reward_model.py        # Train a reward model on preference data
│   │   ├── 02_ppo_training.py              # Full PPO training loop with TRL
│   │   ├── 03_ppo_analysis.py              # Compare SFT vs PPO outputs, KL experiments
│   │   └── configs/
│   │       └── ppo_config.yaml             # Hyperparameters with comments explaining each
│   ├── phase_02_dpo/
│   │   ├── README.md                       # Theory guide: DPO insight, loss, trade-offs
│   │   ├── 01_dpo_training.py              # DPO training loop with TRL
│   │   ├── 02_dpo_analysis.py              # Compare DPO vs PPO vs SFT
│   │   └── configs/
│   │       └── dpo_config.yaml             # Hyperparameters with comments
│   ├── phase_03_grpo/
│   │   ├── README.md                       # Theory guide: GRPO mechanism, algorithm landscape
│   │   ├── 01_grpo_training.py             # GRPO training loop with TRL
│   │   ├── 02_grpo_analysis.py             # Three-way comparison
│   │   └── configs/
│   │       └── grpo_config.yaml            # Hyperparameters with comments
│   └── phase_04_capstone/
│       ├── README.md                       # Capstone instructions and experiment design
│       ├── 01_run_all_experiments.py        # Train PPO/DPO/GRPO with unified config
│       ├── 02_evaluate_and_compare.py       # Generate comparison table, charts
│       ├── 03_write_report.py              # Auto-generate report skeleton from results
│       └── configs/
│           └── capstone_config.yaml        # Unified experiment config
```

---

### Task 1: Project Setup & Shared Utilities

**Files:**
- Create: `requirements.txt`
- Create: `setup_env.sh`
- Create: `shared/__init__.py`
- Create: `shared/model_utils.py`
- Create: `shared/data_utils.py`
- Create: `shared/eval_utils.py`

- [ ] **Step 1: Create `requirements.txt`**

```txt
torch>=2.1.0
transformers>=4.46.0
datasets>=3.0.0
trl>=0.12.0
accelerate>=1.0.0
peft>=0.13.0
bitsandbytes>=0.44.0
wandb>=0.18.0
pyyaml>=6.0
pandas>=2.0.0
matplotlib>=3.8.0
```

- [ ] **Step 2: Create `setup_env.sh`**

```bash
#!/usr/bin/env bash
set -euo pipefail

echo "=== RL Learning Path: Environment Setup ==="

if ! command -v python3 &> /dev/null; then
    echo "ERROR: python3 not found. Install Python 3.10+."
    exit 1
fi

python3 -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt

echo ""
echo "=== Setup complete ==="
echo "Activate with: source .venv/bin/activate"
echo ""
echo "Verify GPU:"
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'Device: {torch.cuda.get_device_name(0)}') if torch.cuda.is_available() else None"
```

- [ ] **Step 3: Create `shared/__init__.py`**

```python
from .model_utils import load_model_and_tokenizer, print_model_info
from .data_utils import load_preference_dataset, format_for_dpo, format_for_ppo
from .eval_utils import generate_samples, compare_outputs, log_training_metrics
```

- [ ] **Step 4: Create `shared/model_utils.py`**

```python
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
```

- [ ] **Step 5: Create `shared/data_utils.py`**

```python
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
```

- [ ] **Step 6: Create `shared/eval_utils.py`**

```python
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
```

- [ ] **Step 7: Verify setup**

Run:
```bash
chmod +x setup_env.sh
bash setup_env.sh
source .venv/bin/activate
python -c "from shared import load_model_and_tokenizer; print('Shared utils OK')"
```

Expected: No import errors, GPU detected.

- [ ] **Step 8: Commit**

```bash
git add requirements.txt setup_env.sh shared/
git commit -m "feat: add project setup and shared utilities"
```

---

### Task 2: Phase 0 — Foundation Theory Guide

**Files:**
- Create: `phases/phase_00_foundation/README.md`

- [ ] **Step 1: Create the Phase 0 theory guide**

```markdown
# Phase 0: Foundation — RLHF Big Picture & RL Concepts

**Duration:** 2-3 days (~8 hours total)
**Goal:** Understand why RLHF exists and learn the minimal RL concepts needed to follow PPO.

---

## Day 1: The Alignment Problem (~2 hours)

### Why SFT Alone Isn't Enough

Supervised Fine-Tuning (SFT) teaches a model to mimic human-written examples. But imitation has limits:

- **The "helpful but harmful" problem**: An SFT model trained on helpful answers will also helpfully explain how to do dangerous things — it learned the pattern "be helpful" without learning boundaries.
- **Distribution mismatch**: Training data is curated, but users ask anything. SFT models can behave unpredictably on out-of-distribution inputs.
- **No preference signal**: SFT treats all training examples as equally good. But some answers are better than others — SFT can't learn "this answer is good, that one is bad."

### The RLHF Pipeline

RLHF (Reinforcement Learning from Human Feedback) adds a feedback loop:

```
Step 1: SFT          →  Base capable model
Step 2: Reward Model  →  Learns what humans prefer
Step 3: RL (PPO)      →  Optimizes the model to produce preferred outputs
```

The key insight: instead of showing the model what to say (SFT), we show it what humans *prefer* and let it figure out how to produce more of that.

### 📖 Reading

- **InstructGPT paper** (Ouyang et al., 2022): Read Sections 1-3
  - Focus on: Figure 2 (the pipeline diagram), the comparison between SFT and RLHF results
  - Skip: the mathematical formulations, ablation studies
  - Paper: "Training language models to follow instructions with human feedback"

---

## Day 2: Just-in-Time RL Concepts (~4 hours)

You do NOT need to learn general RL (no Atari, no robotics, no Q-learning). Only these concepts, reframed for LLMs:

### Policy (π)

In RL: an agent's strategy for choosing actions.
**In LLM terms:** The language model itself. Given an input prompt, it produces a probability distribution over possible next tokens. The full model IS the policy.

### Action

In RL: a choice the agent makes.
**In LLM terms:** Generating a single token. A full response is a sequence of actions.

### Reward (r)

In RL: a score that says how good an action was.
**In LLM terms:** A score from a reward model (or a rule-based function) that rates the quality of a complete response. Higher reward = better response.

### Trajectory / Episode

In RL: a sequence of actions from start to finish.
**In LLM terms:** Generating a complete response to a prompt. The "episode" starts when the model sees the prompt and ends when it produces the end-of-sequence token.

### Policy Gradient

The core optimization idea: if a response got a high reward, increase the probability of the tokens in that response. If it got a low reward, decrease them.

Mathematically: `∇J(θ) = E[∇log π(a|s) · R]`

In plain English: "nudge the model weights so that high-reward responses become more likely."

**The problem:** This is noisy and unstable. One bad update can ruin the model. This is why we need PPO (Phase 1).

### KL Divergence

A measure of how different two probability distributions are. In RLHF:

- We compare the RL-tuned model (π) against the original SFT model (π_ref)
- If KL divergence gets too high, the model has drifted too far → "reward hacking"
- We add a KL penalty to the reward: `reward_adjusted = reward - β · KL(π || π_ref)`
- β controls the trade-off: higher β = stay closer to SFT, lower β = more freedom to optimize

**Intuition:** KL divergence is the "leash" that keeps the RL model from going wild. Without it, the model learns to exploit reward model weaknesses instead of genuinely improving.

### 📖 Reading

- **Lilian Weng's blog post** "What is RLHF" — covers all the above with diagrams
- Skim: [Hugging Face RLHF blog post](https://huggingface.co/blog/rlhf) — shorter, more practical

---

## Day 3: Reward Model Basics (~2 hours)

### How Preference Data Works

Humans compare two model outputs for the same prompt and pick the better one:

```
Prompt: "Explain photosynthesis simply"
Response A: "Plants use sunlight to make food from CO2 and water..."  ← Chosen
Response B: "Photosynthesis is a biochemical process involving..."    ← Rejected
```

This creates preference pairs: (prompt, chosen, rejected).

### The Bradley-Terry Model

The math behind preference modeling is surprisingly simple — it's logistic regression:

```
P(y_w > y_l) = σ(r(x, y_w) - r(x, y_l))
```

- `σ` = sigmoid function
- `r(x, y)` = reward model's score for response y given prompt x
- The model learns to assign higher scores to preferred responses

**Loss:** `L = -log σ(r(x, y_w) - r(x, y_l))`

This is just binary cross-entropy — "increase the score gap between chosen and rejected."

### Why Reward Model Quality Is the Bottleneck

- Garbage in, garbage out: if human preferences are noisy or inconsistent, the reward model learns noise
- The RL stage amplifies reward model errors — the policy will exploit any weakness
- This is why DPO (Phase 2) tries to remove the reward model entirely

### 📖 Reading

- Revisit InstructGPT paper Section 3.2 (reward model training)
- Optional: "Learning to summarize with human feedback" (Stiennon et al., 2020) — earlier work that established the reward model approach

---

## ✅ Self-Check: Am I Ready for Phase 1?

Before moving on, make sure you can answer these:

1. **Pipeline:** "What are the three steps of RLHF, and what does each step produce?"
   - Expected: SFT → capable base model; Reward Model training → preference scorer; RL fine-tuning → aligned model

2. **KL Divergence:** "Why do we need a KL penalty in RLHF?"
   - Expected: To prevent the model from drifting too far from the SFT baseline and exploiting reward model weaknesses

3. **Reward Model:** "How is a reward model trained?"
   - Expected: From human preference pairs using Bradley-Terry (logistic regression on score differences)

If you can explain these to a colleague clearly, you're ready for Phase 1.
```

- [ ] **Step 2: Commit**

```bash
git add phases/phase_00_foundation/
git commit -m "docs: add Phase 0 foundation theory guide"
```

---

### Task 3: Phase 1 — PPO Theory Guide

**Files:**
- Create: `phases/phase_01_ppo/README.md`

- [ ] **Step 1: Create the Phase 1 theory guide**

```markdown
# Phase 1: PPO in the RLHF Pipeline

**Duration:** 4-5 days (~15 hours: 6 theory + 9 hands-on)
**Goal:** Understand how PPO works in LLM alignment and run a full RLHF training loop.

---

## Theory: From Policy Gradient to PPO (Day 4-5)

### The Problem with Vanilla Policy Gradient

Recall from Phase 0: policy gradient says "increase the probability of high-reward actions."

The problem: how MUCH should we increase/decrease probabilities?

- **Too large an update** → the model's behavior changes dramatically → performance collapses → can't recover
- **Too small an update** → training is impossibly slow
- There's no good way to pick the right step size because the relationship between parameter changes and behavior changes is nonlinear

This is the "catastrophic update" problem. One bad gradient step can destroy a model that took days to train.

### PPO: The Clipped Objective

PPO (Proximal Policy Optimization) solves this with a simple trick: **clip the update so the policy can't change too much in one step**.

The objective function:

```
L_CLIP = E[ min( r(θ) · A,  clip(r(θ), 1-ε, 1+ε) · A ) ]
```

Breaking this down:

| Symbol | Meaning | LLM Context |
|--------|---------|-------------|
| `r(θ)` | Probability ratio: π_new(a\|s) / π_old(a\|s) | How much more/less likely is this token under the new policy vs old? |
| `A` | Advantage: how much better was this action than average | Was this token's contribution to the response above or below average reward? |
| `ε` | Clip range (typically 0.2) | The policy can change by at most 20% per update |
| `clip(r, 1-ε, 1+ε)` | Clamp the ratio to [0.8, 1.2] | Even if the gradient wants a bigger change, cap it |
| `min(...)` | Take the more pessimistic estimate | Prevent overly optimistic updates in both directions |

**Intuition in one sentence:** "If a response was good, make it more likely — but not more than 20% more likely per step."

### Why `min` and Not Just Clipping?

The `min` creates an asymmetric effect:

- When advantage is **positive** (good action): the ratio is clipped from above → prevents making a good action TOO much more likely
- When advantage is **negative** (bad action): the ratio is clipped from below → prevents making a bad action TOO much less likely

This conservatism in both directions is what makes PPO stable.

### PPO Adapted for LLMs: The 4-Model Setup

In classic RL, PPO uses 2 networks (actor + critic). For LLM alignment, you need 4 models:

```
┌─────────────────────────────────────────────────────────┐
│                    RLHF with PPO                        │
│                                                         │
│  1. Policy Model (π_θ)      ← Being trained             │
│     The LLM we're aligning                              │
│                                                         │
│  2. Reference Model (π_ref) ← Frozen copy of SFT model  │
│     Used to compute KL penalty                          │
│                                                         │
│  3. Reward Model (r_φ)      ← Trained on preferences    │
│     Scores generated responses                          │
│                                                         │
│  4. Value Model (V_ψ)       ← The critic                │
│     Estimates expected reward for a given state          │
│     Used to compute advantages (A = R - V)              │
└─────────────────────────────────────────────────────────┘
```

**Why 4 models?** Each serves a distinct role:
- Policy: the model being improved
- Reference: the anchor preventing drift (for KL)
- Reward: the scoring function (from human preferences)
- Value: the baseline for advantage estimation (reduces variance in gradient updates)

**Why this is expensive:** All 4 models must be in GPU memory. For a 7B model, that's ~56GB just for model weights in fp16. This is a major motivation for DPO (Phase 2) and GRPO (Phase 3).

### One PPO Iteration

```
1. GENERATE: Policy produces responses to a batch of prompts
2. SCORE:    Reward model scores each response
3. COMPUTE:  Calculate advantages using value model: A = R - V(s)
4. UPDATE:   Update policy using clipped objective
             Update value model to better predict rewards
5. PENALTY:  Add KL penalty: R_adjusted = R - β · KL(π_θ || π_ref)
```

### Key Hyperparameters

| Parameter | Typical Value | Effect |
|-----------|---------------|--------|
| `learning_rate` | 1e-5 to 5e-6 | Standard LLM fine-tuning range |
| `kl_penalty` (β) | 0.01 - 0.2 | Higher = more conservative (stays close to SFT) |
| `clip_range` (ε) | 0.2 | How far the policy can move per step |
| `batch_size` | 64-256 | Larger = more stable but more memory |
| `ppo_epochs` | 2-4 | How many times to reuse each batch |
| `gamma` | 1.0 | Discount factor (usually 1.0 for LLMs since episodes are short) |

### 📖 Reading

- **PPO paper** (Schulman et al., 2017): Read Sections 1-3 (skip Atari/MuJoCo experiments)
  - Paper: "Proximal Policy Optimization Algorithms"
- **TRL PPOTrainer docs**: Read the API overview and example scripts
- Optional: "Secrets of RLHF in Large Language Models Part I" — practical tips

---

## Hands-on Lab (Day 6-8)

See the Python scripts in this directory:

1. `01_train_reward_model.py` — Train a reward model on preference data
2. `02_ppo_training.py` — Run the full PPO training loop
3. `03_ppo_analysis.py` — Analyze results and experiment with KL coefficient

Run them in order. Each script has inline comments explaining what's happening at each step.

### What to Watch For

- **Reward mean increasing** → the model is learning to produce higher-reward responses ✓
- **KL divergence increasing** → the model is drifting from SFT (watch for runaway growth) ⚠️
- **Reward increasing but outputs degrading** → reward hacking (KL penalty too low) ✗
- **Reward flat** → KL penalty too high or learning rate too low ✗

---

## ✅ Self-Check: Am I Ready for Phase 2?

1. "What does the clipping in PPO prevent?"
   - Expected: Prevents the policy from changing too much in a single update, avoiding catastrophic collapse

2. "Why does PPO for LLMs need 4 models? What does each one do?"
   - Expected: Policy (being trained), Reference (KL anchor), Reward (scoring), Value (advantage estimation)

3. "What happens if you set the KL coefficient too low? Too high?"
   - Expected: Too low → reward hacking; Too high → model barely learns, stays too close to SFT
```

- [ ] **Step 2: Commit**

```bash
git add phases/phase_01_ppo/README.md
git commit -m "docs: add Phase 1 PPO theory guide"
```

---

### Task 4: Phase 1 — PPO Training Scripts

**Files:**
- Create: `phases/phase_01_ppo/configs/ppo_config.yaml`
- Create: `phases/phase_01_ppo/01_train_reward_model.py`
- Create: `phases/phase_01_ppo/02_ppo_training.py`
- Create: `phases/phase_01_ppo/03_ppo_analysis.py`

- [ ] **Step 1: Create `phases/phase_01_ppo/configs/ppo_config.yaml`**

```yaml
# PPO Training Configuration
# Each parameter has a comment explaining its role and how to tune it.

model:
  name: "Qwen/Qwen2.5-1.5B-Instruct"
  torch_dtype: "bfloat16"

reward_model:
  name: "Ray2333/GRM-Gemma-2B-sftreg"
  torch_dtype: "bfloat16"

dataset:
  name: "argilla/ultrafeedback-binarized-preferences-cleaned"
  max_samples: 1000
  max_prompt_length: 256

ppo:
  learning_rate: 1.0e-5
  batch_size: 16
  mini_batch_size: 4
  ppo_epochs: 2
  max_new_tokens: 128
  kl_penalty: "kl"
  init_kl_coef: 0.05
  clip_range: 0.2
  gamma: 1.0
  lam: 0.95

generation:
  temperature: 0.7
  top_p: 0.9
  do_sample: true

training:
  num_train_epochs: 1
  save_steps: 50
  logging_steps: 10
  output_dir: "outputs/phase_01_ppo"

# Experiments: try these alternative KL coefficients
kl_experiments:
  - 0.01    # Very loose — watch for reward hacking
  - 0.05    # Default — balanced
  - 0.2     # Tight — model may barely learn
```

- [ ] **Step 2: Create `phases/phase_01_ppo/01_train_reward_model.py`**

```python
"""
Phase 1, Script 1: Train a Reward Model
========================================
This script trains a reward model on preference data.
The reward model scores responses — it's the "judge" that PPO optimizes against.

What you'll learn:
- How preference data (chosen vs rejected) becomes training signal
- How the Bradley-Terry model works in practice
- Why reward model quality matters

Usage:
    python phases/phase_01_ppo/01_train_reward_model.py
"""

import sys
import os
import yaml
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_dataset
from trl import RewardTrainer, RewardConfig

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from shared.model_utils import print_model_info


def load_config():
    config_path = os.path.join(os.path.dirname(__file__), "configs", "ppo_config.yaml")
    with open(config_path) as f:
        return yaml.safe_load(f)


def prepare_reward_dataset(config):
    ds = load_dataset(
        config["dataset"]["name"],
        split="train",
    )
    ds = ds.shuffle(seed=42).select(range(min(config["dataset"]["max_samples"], len(ds))))

    def preprocess(example):
        return {
            "chosen": example["chosen"],
            "rejected": example["rejected"],
        }

    ds = ds.map(preprocess, remove_columns=[c for c in ds.column_names if c not in ["chosen", "rejected"]])
    split = ds.train_test_split(test_size=0.1, seed=42)
    return split["train"], split["test"]


def main():
    config = load_config()

    print("\n📦 Loading reward model and tokenizer...")
    rm_name = config["reward_model"]["name"]
    tokenizer = AutoTokenizer.from_pretrained(rm_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForSequenceClassification.from_pretrained(
        rm_name,
        num_labels=1,
        torch_dtype=torch.bfloat16,
    )
    print_model_info(model, f"Reward Model: {rm_name}")

    print("\n📊 Preparing preference dataset...")
    train_ds, eval_ds = prepare_reward_dataset(config)
    print(f"  Train samples: {len(train_ds)}")
    print(f"  Eval samples:  {len(eval_ds)}")

    output_dir = os.path.join(config["training"]["output_dir"], "reward_model")

    training_args = RewardConfig(
        output_dir=output_dir,
        num_train_epochs=1,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        learning_rate=2e-5,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=50,
        save_strategy="steps",
        save_steps=100,
        bf16=True,
        remove_unused_columns=False,
        max_length=512,
    )

    trainer = RewardTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
    )

    print("\n🚀 Training reward model...")
    print("Watch the eval accuracy — it should be >60% for a useful reward model.")
    print("If it's near 50%, the model isn't learning preferences.\n")

    trainer.train()

    print(f"\n💾 Saving reward model to {output_dir}")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    print("\n✅ Reward model training complete!")
    print("Next: run 02_ppo_training.py to use this reward model for PPO.\n")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Create `phases/phase_01_ppo/02_ppo_training.py`**

```python
"""
Phase 1, Script 2: PPO Training Loop
======================================
This is the core RLHF script. It uses 4 models:
  1. Policy Model     — the LLM being aligned (Qwen2.5-1.5B)
  2. Reference Model  — frozen copy of the SFT model (for KL penalty)
  3. Reward Model     — trained in Script 1 (scores responses)
  4. Value Model      — estimates expected reward (critic, initialized from policy)

What you'll learn:
- How the PPO training loop works step by step
- What each metric means (reward, KL, policy loss)
- How KL coefficient affects training dynamics

Usage:
    python phases/phase_01_ppo/02_ppo_training.py
"""

import sys
import os
import yaml
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_dataset
from trl import PPOConfig, PPOTrainer, AutoModelForCausalLMWithValueHead

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from shared.model_utils import MODEL_NAME, print_model_info


def load_config():
    config_path = os.path.join(os.path.dirname(__file__), "configs", "ppo_config.yaml")
    with open(config_path) as f:
        return yaml.safe_load(f)


def prepare_prompts(config):
    ds = load_dataset(config["dataset"]["name"], split="train")
    ds = ds.shuffle(seed=42).select(range(min(config["dataset"]["max_samples"], len(ds))))

    prompts = []
    for example in ds:
        prompt = example["prompt"]
        if len(prompt) > 0:
            prompts.append(prompt)
    return prompts


def main():
    config = load_config()
    ppo_cfg = config["ppo"]

    # --- Step 1: Load the policy model (with value head for PPO) ---
    print("\n📦 Step 1/4: Loading POLICY model (with value head)...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLMWithValueHead.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    print_model_info(model.pretrained_model, "Policy Model (Qwen2.5-1.5B)")

    # --- Step 2: The reference model is handled internally by PPOTrainer ---
    print("📦 Step 2/4: Reference model will be created automatically by PPOTrainer.")
    print("   (It's a frozen copy of the policy, used to compute KL penalty)\n")

    # --- Step 3: Load the reward model ---
    print("📦 Step 3/4: Loading REWARD model...")
    rm_path = os.path.join(config["training"]["output_dir"], "reward_model")
    if not os.path.exists(rm_path):
        print(f"  ⚠️  No trained reward model found at {rm_path}")
        print(f"  Using pretrained: {config['reward_model']['name']}")
        rm_path = config["reward_model"]["name"]

    reward_model = AutoModelForSequenceClassification.from_pretrained(
        rm_path,
        num_labels=1,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    reward_tokenizer = AutoTokenizer.from_pretrained(rm_path)
    if reward_tokenizer.pad_token is None:
        reward_tokenizer.pad_token = reward_tokenizer.eos_token
    print_model_info(reward_model, "Reward Model")

    # --- Step 4: Configure PPO ---
    print("📦 Step 4/4: Configuring PPO trainer...\n")

    ppo_config = PPOConfig(
        learning_rate=ppo_cfg["learning_rate"],
        batch_size=ppo_cfg["batch_size"],
        mini_batch_size=ppo_cfg["mini_batch_size"],
        ppo_epochs=ppo_cfg["ppo_epochs"],
        init_kl_coef=ppo_cfg["init_kl_coef"],
        kl_penalty=ppo_cfg["kl_penalty"],
        clip_range=ppo_cfg["clip_range"],
        gamma=ppo_cfg["gamma"],
        lam=ppo_cfg["lam"],
        log_with="tensorboard",
        project_kwargs={"logging_dir": os.path.join(config["training"]["output_dir"], "logs")},
    )

    prompts = prepare_prompts(config)
    print(f"Prepared {len(prompts)} training prompts\n")

    trainer = PPOTrainer(
        config=ppo_config,
        model=model,
        tokenizer=tokenizer,
        dataset=None,
    )

    # --- Training Loop ---
    print("🚀 Starting PPO training loop...")
    print("=" * 60)
    print(f"  KL coefficient:  {ppo_cfg['init_kl_coef']}")
    print(f"  Clip range:      {ppo_cfg['clip_range']}")
    print(f"  Batch size:      {ppo_cfg['batch_size']}")
    print(f"  Learning rate:   {ppo_cfg['learning_rate']}")
    print("=" * 60)
    print("\nMetrics to watch:")
    print("  reward/mean  ↑  (should increase — model learning to generate preferred outputs)")
    print("  objective/kl ↔  (should stay moderate — too high means reward hacking)")
    print("  ppo/loss     ↓  (should decrease — policy improving)")
    print()

    gen_kwargs = {
        "max_new_tokens": ppo_cfg["max_new_tokens"],
        "temperature": config["generation"]["temperature"],
        "top_p": config["generation"]["top_p"],
        "do_sample": config["generation"]["do_sample"],
        "pad_token_id": tokenizer.pad_token_id,
    }

    for step in range(0, len(prompts), ppo_cfg["batch_size"]):
        batch_prompts = prompts[step:step + ppo_cfg["batch_size"]]
        if len(batch_prompts) < ppo_cfg["batch_size"]:
            break

        # Tokenize prompts
        query_tensors = [
            tokenizer(p, return_tensors="pt", truncation=True,
                      max_length=config["dataset"]["max_prompt_length"])["input_ids"].squeeze()
            for p in batch_prompts
        ]

        # Generate responses
        response_tensors = trainer.generate(query_tensors, **gen_kwargs)

        # Decode responses for reward scoring
        responses = [tokenizer.decode(r.squeeze(), skip_special_tokens=True) for r in response_tensors]

        # Score with reward model
        rewards = []
        for prompt, response in zip(batch_prompts, responses):
            inputs = reward_tokenizer(
                prompt + response,
                return_tensors="pt",
                truncation=True,
                max_length=512,
            ).to(reward_model.device)
            with torch.no_grad():
                score = reward_model(**inputs).logits.squeeze().cpu()
            rewards.append(score)

        # PPO update step
        stats = trainer.step(query_tensors, response_tensors, rewards)

        # Log metrics
        batch_num = step // ppo_cfg["batch_size"]
        if batch_num % config["training"]["logging_steps"] == 0:
            print(f"\nStep {batch_num}:")
            for key in ["ppo/loss/total", "ppo/mean_scores", "objective/kl"]:
                if key in stats:
                    print(f"  {key}: {stats[key]:.4f}")

    # Save the aligned model
    output_path = os.path.join(config["training"]["output_dir"], "ppo_model")
    print(f"\n💾 Saving PPO-aligned model to {output_path}")
    trainer.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)

    print("\n✅ PPO training complete!")
    print("Next: run 03_ppo_analysis.py to analyze the results.\n")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Create `phases/phase_01_ppo/03_ppo_analysis.py`**

```python
"""
Phase 1, Script 3: PPO Analysis
=================================
Compare the SFT baseline against the PPO-aligned model.
This script helps you build intuition for what PPO actually changes.

What you'll learn:
- How to compare model outputs qualitatively
- What KL coefficient changes look like in practice
- How to spot reward hacking

Usage:
    python phases/phase_01_ppo/03_ppo_analysis.py
"""

import sys
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from shared.model_utils import MODEL_NAME, load_model_and_tokenizer
from shared.eval_utils import generate_samples, compare_outputs

EVAL_PROMPTS = [
    "Explain machine learning to a 10-year-old.",
    "What are the pros and cons of remote work?",
    "Write a short poem about coding.",
    "How do I deal with a difficult coworker?",
    "Explain why the sky is blue in simple terms.",
    "What is the most important thing in life?",
    "How should I prepare for a job interview?",
    "Summarize the concept of climate change in 3 sentences.",
]


def main():
    print("\n📊 Phase 1 Analysis: SFT vs PPO\n")

    # Load SFT baseline
    print("Loading SFT baseline model...")
    sft_model, tokenizer = load_model_and_tokenizer(MODEL_NAME)

    # Load PPO-aligned model
    ppo_path = "outputs/phase_01_ppo/ppo_model"
    if os.path.exists(ppo_path):
        print(f"Loading PPO model from {ppo_path}...")
        ppo_model = AutoModelForCausalLM.from_pretrained(
            ppo_path, torch_dtype=torch.bfloat16, device_map="auto",
        )
    else:
        print(f"⚠️  PPO model not found at {ppo_path}. Run 02_ppo_training.py first.")
        print("Showing SFT baseline only.\n")
        ppo_model = None

    # Generate and compare
    print("\nGenerating responses...\n")
    sft_outputs = generate_samples(sft_model, tokenizer, EVAL_PROMPTS)

    model_outputs = {"SFT Baseline": sft_outputs}
    if ppo_model is not None:
        ppo_outputs = generate_samples(ppo_model, tokenizer, EVAL_PROMPTS)
        model_outputs["PPO Aligned"] = ppo_outputs

    results = compare_outputs(
        EVAL_PROMPTS,
        model_outputs,
        save_path="outputs/phase_01_ppo/comparison_results.json",
    )

    print("\n" + "=" * 60)
    print("🔍 THINGS TO LOOK FOR:")
    print("=" * 60)
    print("""
1. HELPFULNESS: Are PPO responses more directly helpful?
   (They should be — the reward model was trained on helpfulness preferences)

2. VERBOSITY: Are PPO responses longer or shorter?
   (Often longer — models learn that longer ≈ higher reward, a form of reward hacking)

3. SAFETY: Does the PPO model handle sensitive topics differently?
   (It should be more careful if the reward model penalizes harmful content)

4. STYLE: Has the model's "personality" changed?
   (Slight changes are normal; drastic changes suggest KL was too low)

📝 Write down your observations — you'll compare these with DPO/GRPO results later.
""")


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Commit**

```bash
git add phases/phase_01_ppo/
git commit -m "feat: add Phase 1 PPO training scripts and config"
```

---

### Task 5: Phase 2 — DPO Theory Guide and Training Scripts

**Files:**
- Create: `phases/phase_02_dpo/README.md`
- Create: `phases/phase_02_dpo/configs/dpo_config.yaml`
- Create: `phases/phase_02_dpo/01_dpo_training.py`
- Create: `phases/phase_02_dpo/02_dpo_analysis.py`

- [ ] **Step 1: Create `phases/phase_02_dpo/README.md`**

```markdown
# Phase 2: DPO — Removing the Reward Model

**Duration:** 3-4 days (~11 hours: 5 theory + 6 hands-on)
**Goal:** Understand how DPO simplifies RLHF by eliminating the reward model, and compare directly with PPO.

---

## Theory: The DPO Insight (Day 9-10)

### The Problem with PPO-Based RLHF

By now you've experienced PPO's pain points firsthand:

- **4 models in memory** — Policy, Reference, Reward, Value (critic)
- **Hyperparameter sensitivity** — KL coefficient, clip range, learning rate all interact
- **Reward hacking** — the policy exploits reward model weaknesses
- **Engineering complexity** — generation loop, reward scoring, advantage computation, PPO update

What if we could get the same result with just a simple loss function on preference data?

### The DPO Breakthrough

DPO (Direct Preference Optimization) makes a mathematical observation:

1. The RLHF objective is: maximize reward while staying close to the reference policy
2. This optimization problem has a **closed-form solution**: `π*(y|x) ∝ π_ref(y|x) · exp(r(x,y) / β)`
3. We can rearrange this to express the reward in terms of the policy:
   `r(x,y) = β · log(π(y|x) / π_ref(y|x)) + const`
4. Plug this into the Bradley-Terry preference model → the reward model disappears

**Result:** A loss function that trains the policy directly on preference pairs.

### The DPO Loss

```
L_DPO = -E[ log σ( β · ( log π(y_w|x)/π_ref(y_w|x) - log π(y_l|x)/π_ref(y_l|x) ) ) ]
```

In plain English:
- For each preference pair (prompt, chosen, rejected):
- Compute how much MORE likely the chosen response is vs the reference (log ratio for chosen)
- Compute how much MORE likely the rejected response is vs the reference (log ratio for rejected)
- The loss pushes these two ratios APART — make chosen more likely, rejected less likely
- β controls how aggressively to push (same role as KL coefficient in PPO)

### The Simplification

| Aspect | PPO | DPO |
|--------|-----|-----|
| Models in memory | 4 (Policy, Ref, RM, Value) | 2 (Policy, Ref) |
| Training data | Reward model + prompts | Preference pairs directly |
| Training loop | Generate → Score → Compute advantages → Update | Forward pass → Compute loss → Update |
| Hyperparameters | KL coef, clip range, GAE λ, PPO epochs... | β, learning rate |
| Online/Offline | Online (generates new responses) | Offline (fixed dataset) |

### Trade-offs

**DPO advantages:**
- Much simpler to implement and debug
- 2x less GPU memory
- More stable training
- Fewer hyperparameters to tune

**DPO limitations:**
- **Data quality dependency**: No reward model to generalize — if your preference pairs are noisy, DPO amplifies that noise
- **Offline only**: Optimizes against a fixed dataset. Can't discover new good responses through exploration
- **Verbosity bias**: Tends to learn surface-level patterns in preferred responses (e.g., "longer is better")
- **β sensitivity**: Too low → overfits to preference data (memorization); Too high → barely changes from SFT

### When to Use Which

- **Use DPO when:** You have high-quality preference data, limited compute, want simplicity
- **Use PPO when:** Data is noisy, you need online exploration, you have large compute budget

### 📖 Reading

- **DPO paper** (Rafailov et al., 2023): Sections 1-4
  - Paper: "Direct Preference Optimization: Your Language Model is Secretly a Reward Model"
- **"Is DPO Superior to PPO?"** — Comparison analyses

---

## Hands-on Lab (Day 11-12)

See the Python scripts in this directory:

1. `01_dpo_training.py` — Run DPO training on the same model/task as PPO
2. `02_dpo_analysis.py` — Direct comparison: SFT vs PPO vs DPO

### What to Watch For

- **Training loss decreasing** → model is learning preferences ✓
- **Chosen rewards > Rejected rewards** → correct preference direction ✓
- **Rewards margin increasing** → model is separating good from bad more confidently ✓
- **Loss drops to near zero quickly** → possible overfitting (try higher β or more data) ⚠️

---

## ✅ Self-Check: Am I Ready for Phase 3?

1. "What does DPO eliminate compared to PPO, and how?"
   - Expected: Eliminates the reward model by deriving the optimal policy directly from preference data

2. "When would you choose DPO over PPO at work?"
   - Expected: When you have clean preference data, limited GPU budget, and want a simpler pipeline

3. "What does β control in DPO?"
   - Expected: How aggressively the model deviates from the reference — analogous to KL coefficient in PPO
```

- [ ] **Step 2: Create `phases/phase_02_dpo/configs/dpo_config.yaml`**

```yaml
# DPO Training Configuration

model:
  name: "Qwen/Qwen2.5-1.5B-Instruct"
  torch_dtype: "bfloat16"

dataset:
  name: "argilla/ultrafeedback-binarized-preferences-cleaned"
  max_samples: 1000
  max_length: 512

dpo:
  beta: 0.1
  learning_rate: 5.0e-6
  per_device_train_batch_size: 4
  gradient_accumulation_steps: 4
  num_train_epochs: 1
  warmup_ratio: 0.1
  logging_steps: 10
  eval_steps: 50
  save_steps: 100
  bf16: true
  max_length: 512
  max_prompt_length: 256

training:
  output_dir: "outputs/phase_02_dpo"

# Experiments: try these alternative beta values
beta_experiments:
  - 0.01   # Very aggressive — risk of overfitting to preferences
  - 0.1    # Default — balanced
  - 0.5    # Very conservative — model barely changes from SFT
```

- [ ] **Step 3: Create `phases/phase_02_dpo/01_dpo_training.py`**

```python
"""
Phase 2, Script 1: DPO Training
=================================
Direct Preference Optimization — align the model using preference pairs directly.
No reward model needed. Just 2 models: Policy + Reference.

Compare how much simpler this is vs the PPO pipeline in Phase 1!

What you'll learn:
- How DPO training works in practice
- The role of beta (β) in controlling preference strength
- How to monitor DPO-specific metrics

Usage:
    python phases/phase_02_dpo/01_dpo_training.py
"""

import sys
import os
import yaml
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from shared.model_utils import MODEL_NAME, print_model_info


def load_config():
    config_path = os.path.join(os.path.dirname(__file__), "configs", "dpo_config.yaml")
    with open(config_path) as f:
        return yaml.safe_load(f)


def prepare_dpo_dataset(config):
    ds = load_dataset(config["dataset"]["name"], split="train")
    ds = ds.shuffle(seed=42).select(range(min(config["dataset"]["max_samples"], len(ds))))

    def format_example(example):
        return {
            "prompt": example["prompt"],
            "chosen": example["chosen"],
            "rejected": example["rejected"],
        }

    ds = ds.map(format_example, remove_columns=[
        c for c in ds.column_names if c not in ["prompt", "chosen", "rejected"]
    ])
    split = ds.train_test_split(test_size=0.1, seed=42)
    return split["train"], split["test"]


def main():
    config = load_config()
    dpo_cfg = config["dpo"]

    print("\n📦 Loading model and tokenizer...")
    print("Notice: only 2 models needed (Policy + Reference). No reward model!\n")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    print_model_info(model, "Policy Model (Qwen2.5-1.5B)")

    ref_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    print("Reference model loaded (frozen copy for KL computation)\n")

    print("📊 Preparing preference dataset...")
    train_ds, eval_ds = prepare_dpo_dataset(config)
    print(f"  Train: {len(train_ds)} preference pairs")
    print(f"  Eval:  {len(eval_ds)} preference pairs\n")

    output_dir = os.path.join(config["training"]["output_dir"], "dpo_model")

    training_args = DPOConfig(
        output_dir=output_dir,
        beta=dpo_cfg["beta"],
        learning_rate=dpo_cfg["learning_rate"],
        per_device_train_batch_size=dpo_cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=dpo_cfg["gradient_accumulation_steps"],
        num_train_epochs=dpo_cfg["num_train_epochs"],
        warmup_ratio=dpo_cfg["warmup_ratio"],
        logging_steps=dpo_cfg["logging_steps"],
        eval_strategy="steps",
        eval_steps=dpo_cfg["eval_steps"],
        save_strategy="steps",
        save_steps=dpo_cfg["save_steps"],
        bf16=dpo_cfg["bf16"],
        max_length=dpo_cfg["max_length"],
        max_prompt_length=dpo_cfg["max_prompt_length"],
        remove_unused_columns=False,
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
    )

    print("🚀 Starting DPO training...")
    print("=" * 60)
    print(f"  Beta (β):        {dpo_cfg['beta']}")
    print(f"  Learning rate:   {dpo_cfg['learning_rate']}")
    print(f"  Batch size:      {dpo_cfg['per_device_train_batch_size']} x {dpo_cfg['gradient_accumulation_steps']} grad accum")
    print("=" * 60)
    print("\nMetrics to watch:")
    print("  train/loss        ↓  (should decrease)")
    print("  rewards/chosen    ↑  (log-ratio for preferred responses)")
    print("  rewards/rejected  ↓  (log-ratio for rejected responses)")
    print("  rewards/margins   ↑  (gap between chosen and rejected)")
    print("  rewards/accuracies ↑ (fraction where chosen > rejected)")
    print()

    trainer.train()

    print(f"\n💾 Saving DPO-aligned model to {output_dir}")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    print("\n✅ DPO training complete!")
    print("Next: run 02_dpo_analysis.py to compare with PPO results.\n")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Create `phases/phase_02_dpo/02_dpo_analysis.py`**

```python
"""
Phase 2, Script 2: DPO Analysis
=================================
Compare SFT baseline, PPO-aligned, and DPO-aligned models side by side.

What you'll learn:
- How DPO outputs differ from PPO outputs
- Trade-offs between the two approaches in practice
- Whether simpler (DPO) produces comparable quality to complex (PPO)

Usage:
    python phases/phase_02_dpo/02_dpo_analysis.py
"""

import sys
import os

import torch
from transformers import AutoModelForCausalLM

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from shared.model_utils import MODEL_NAME, load_model_and_tokenizer
from shared.eval_utils import generate_samples, compare_outputs


EVAL_PROMPTS = [
    "Explain machine learning to a 10-year-old.",
    "What are the pros and cons of remote work?",
    "Write a short poem about coding.",
    "How do I deal with a difficult coworker?",
    "Explain why the sky is blue in simple terms.",
    "What is the most important thing in life?",
    "How should I prepare for a job interview?",
    "Summarize the concept of climate change in 3 sentences.",
]


def try_load_model(path, label):
    if os.path.exists(path):
        print(f"  Loading {label} from {path}...")
        return AutoModelForCausalLM.from_pretrained(
            path, torch_dtype=torch.bfloat16, device_map="auto",
        )
    print(f"  ⚠️  {label} not found at {path} — skipping")
    return None


def main():
    print("\n📊 Phase 2 Analysis: SFT vs PPO vs DPO\n")

    print("Loading models...")
    sft_model, tokenizer = load_model_and_tokenizer(MODEL_NAME)
    ppo_model = try_load_model("outputs/phase_01_ppo/ppo_model", "PPO model")
    dpo_model = try_load_model("outputs/phase_02_dpo/dpo_model", "DPO model")

    print("\nGenerating responses...\n")
    model_outputs = {"SFT Baseline": generate_samples(sft_model, tokenizer, EVAL_PROMPTS)}

    if ppo_model is not None:
        model_outputs["PPO Aligned"] = generate_samples(ppo_model, tokenizer, EVAL_PROMPTS)
    if dpo_model is not None:
        model_outputs["DPO Aligned"] = generate_samples(dpo_model, tokenizer, EVAL_PROMPTS)

    compare_outputs(
        EVAL_PROMPTS,
        model_outputs,
        save_path="outputs/phase_02_dpo/comparison_results.json",
    )

    print("\n" + "=" * 60)
    print("🔍 COMPARISON QUESTIONS:")
    print("=" * 60)
    print("""
1. QUALITY: Which produces more helpful responses — PPO or DPO?
   (They're often surprisingly similar despite DPO being much simpler)

2. STYLE: Does DPO have different verbosity/style tendencies than PPO?
   (DPO often shows less reward hacking since there's no reward model to exploit)

3. TRAINING EFFORT: Compare wall-clock time and GPU memory usage.
   (DPO should be significantly faster and use less memory)

4. STABILITY: Was DPO training smoother than PPO?
   (DPO is typically much more stable — fewer hyperparameters to get wrong)

📝 Document your observations for the capstone comparison.

🔬 EXPERIMENT: Try changing β in dpo_config.yaml:
   - β=0.01 (aggressive) — does the model overfit to preferences?
   - β=0.5  (conservative) — does the model barely change?
   Compare with KL coefficient experiments from PPO.
""")


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Commit**

```bash
git add phases/phase_02_dpo/
git commit -m "feat: add Phase 2 DPO theory guide and training scripts"
```

---

### Task 6: Phase 3 — GRPO Theory Guide and Training Scripts

**Files:**
- Create: `phases/phase_03_grpo/README.md`
- Create: `phases/phase_03_grpo/configs/grpo_config.yaml`
- Create: `phases/phase_03_grpo/01_grpo_training.py`
- Create: `phases/phase_03_grpo/02_grpo_analysis.py`

- [ ] **Step 1: Create `phases/phase_03_grpo/README.md`**

```markdown
# Phase 3: GRPO — Group-Based Optimization

**Duration:** 3-4 days (~11 hours: 5 theory + 6 hands-on)
**Goal:** Understand how GRPO eliminates the value model from PPO while keeping online RL benefits.

---

## Theory: Group Relative Policy Optimization (Day 13-14)

### The Problem GRPO Solves

Recall PPO's 4-model setup. Two of those models exist only for advantage estimation:

- **Value Model (critic)**: estimates expected reward → used to compute `A = R - V(s)`
- **Reference Model**: provides KL penalty

The value model is expensive to train and introduces its own instability. What if we could estimate advantages WITHOUT a learned critic?

### GRPO's Key Idea: Grade on a Curve

Instead of asking "how good was this response compared to the average?" (which requires a value model), GRPO asks "how good was this response compared to its peers?"

**Algorithm:**

```
For each prompt x:
  1. Generate G responses: {y_1, y_2, ..., y_G}  (the "group")
  2. Score each with reward model: {r_1, r_2, ..., r_G}
  3. Normalize within the group:
     A_i = (r_i - mean(r)) / std(r)
  4. Update policy using PPO-style clipped objective with these advantages
  5. Add KL penalty against reference model
```

**Why this works:**
- A response that scored above the group average gets a positive advantage → reinforced
- A response that scored below average gets a negative advantage → suppressed
- No value model needed — the group itself serves as the baseline

### GRPO vs PPO vs DPO

| Aspect | PPO | DPO | GRPO |
|--------|-----|-----|------|
| Models needed | 4 (Policy, Ref, RM, Critic) | 2 (Policy, Ref) | 3 (Policy, Ref, RM) |
| Advantage estimation | Value model (learned) | N/A (no RL loop) | Group normalization |
| Online/Offline | Online | Offline | Online |
| Needs reward model | Yes | No | Yes (or rule-based) |
| Generation per step | 1 response/prompt | None (uses dataset) | G responses/prompt |
| Key strength | Full RL with exploration | Simplicity | Online RL without critic complexity |

### When GRPO Shines

GRPO is especially powerful when you have **verifiable rewards** — rewards that can be computed automatically without a learned model:

- **Math reasoning**: check if the answer is correct
- **Code generation**: run the code, check if tests pass
- **Format compliance**: check if output follows a template
- **Length constraints**: penalize outputs that are too long/short

This is how DeepSeek-R1 was trained: GRPO + rule-based rewards for mathematical reasoning.

### Key Hyperparameters

| Parameter | Typical Value | Effect |
|-----------|---------------|--------|
| `num_generations` (G) | 4-16 | Group size. Larger = better advantage estimates but more compute per step |
| `kl_coef` | 0.01-0.1 | KL penalty against reference (same role as in PPO) |
| `clip_range` | 0.2 | PPO-style clipping (same as PPO) |
| `temperature` | 0.7-1.0 | For generation diversity within the group |

### 📖 Reading

- **DeepSeekMath paper** (Shao et al., 2024): Section 3 on GRPO
  - Paper: "DeepSeekMath: Pushing the Limits of Mathematical Reasoning"
- **DeepSeek-R1 technical report**: How GRPO was used at scale
- **TRL GRPOTrainer docs**: API reference and examples

---

## Hands-on Lab (Day 15-16)

1. `01_grpo_training.py` — Run GRPO training with different group sizes
2. `02_grpo_analysis.py` — Full three-way comparison: PPO vs DPO vs GRPO

### What to Watch For

- **Group reward variance** → if all G responses score the same, advantages are near zero (model isn't learning). Increase temperature or group size.
- **Training cost vs group size** → G=16 gives better advantage estimates but costs 16x more generation compute than G=1
- **Rule-based vs learned rewards** → try both and compare training stability

---

## ✅ Self-Check: Am I Ready for the Capstone?

1. "How does GRPO estimate advantages without a value model?"
   - Expected: Generates multiple responses per prompt, normalizes their rewards within the group

2. "Where does GRPO sit between PPO and DPO?"
   - Expected: Online like PPO (generates new responses), simpler like DPO (no critic), but still needs a reward model

3. "When is GRPO the best choice?"
   - Expected: When you have verifiable/rule-based rewards (math, code) and want online RL without critic complexity
```

- [ ] **Step 2: Create `phases/phase_03_grpo/configs/grpo_config.yaml`**

```yaml
# GRPO Training Configuration

model:
  name: "Qwen/Qwen2.5-1.5B-Instruct"
  torch_dtype: "bfloat16"

dataset:
  name: "argilla/ultrafeedback-binarized-preferences-cleaned"
  max_samples: 500
  max_prompt_length: 256

grpo:
  num_generations: 8
  learning_rate: 5.0e-6
  per_device_train_batch_size: 2
  gradient_accumulation_steps: 4
  num_train_epochs: 1
  warmup_ratio: 0.1
  logging_steps: 5
  save_steps: 50
  bf16: true
  max_completion_length: 256
  kl_coef: 0.05
  temperature: 0.8

reward_model:
  name: "Ray2333/GRM-Gemma-2B-sftreg"
  torch_dtype: "bfloat16"

training:
  output_dir: "outputs/phase_03_grpo"

# Experiments: try different group sizes
group_size_experiments:
  - 4    # Smaller group — faster but noisier advantage estimates
  - 8    # Default — balanced
  - 16   # Larger group — better estimates but more compute
```

- [ ] **Step 3: Create `phases/phase_03_grpo/01_grpo_training.py`**

```python
"""
Phase 3, Script 1: GRPO Training
==================================
Group Relative Policy Optimization — online RL without a value model (critic).
Generates multiple responses per prompt, normalizes rewards within the group.

What you'll learn:
- How group-based advantage estimation works in practice
- The effect of group size on training dynamics
- How GRPO compares to PPO in compute efficiency

Usage:
    python phases/phase_03_grpo/01_grpo_training.py
"""

import sys
import os
import yaml
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSequenceClassification
from trl import GRPOConfig, GRPOTrainer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from shared.model_utils import MODEL_NAME, print_model_info


def load_config():
    config_path = os.path.join(os.path.dirname(__file__), "configs", "grpo_config.yaml")
    with open(config_path) as f:
        return yaml.safe_load(f)


def prepare_grpo_dataset(config):
    ds = load_dataset(config["dataset"]["name"], split="train")
    ds = ds.shuffle(seed=42).select(range(min(config["dataset"]["max_samples"], len(ds))))

    def format_example(example):
        return {"prompt": example["prompt"]}

    ds = ds.map(format_example, remove_columns=[
        c for c in ds.column_names if c != "prompt"
    ])
    return ds


def build_reward_fn(config):
    rm_name = config["reward_model"]["name"]
    rm_tokenizer = AutoTokenizer.from_pretrained(rm_name)
    if rm_tokenizer.pad_token is None:
        rm_tokenizer.pad_token = rm_tokenizer.eos_token

    rm_model = AutoModelForSequenceClassification.from_pretrained(
        rm_name,
        num_labels=1,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    rm_model.eval()
    print_model_info(rm_model, f"Reward Model: {rm_name}")

    def reward_fn(completions, **kwargs):
        rewards = []
        for completion in completions:
            text = completion if isinstance(completion, str) else completion[0]["content"]
            inputs = rm_tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True,
            ).to(rm_model.device)
            with torch.no_grad():
                score = rm_model(**inputs).logits.squeeze().cpu().float().item()
            rewards.append(score)
        return rewards

    return reward_fn


def main():
    config = load_config()
    grpo_cfg = config["grpo"]

    print("\n📦 Loading model and tokenizer...")
    print(f"Group size (G): {grpo_cfg['num_generations']} responses per prompt\n")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    print_model_info(model, "Policy Model (Qwen2.5-1.5B)")

    print("📊 Preparing dataset...")
    train_ds = prepare_grpo_dataset(config)
    print(f"  Train prompts: {len(train_ds)}\n")

    print("📦 Loading reward model for scoring...")
    reward_fn = build_reward_fn(config)

    output_dir = os.path.join(config["training"]["output_dir"], "grpo_model")

    training_args = GRPOConfig(
        output_dir=output_dir,
        num_generations=grpo_cfg["num_generations"],
        learning_rate=grpo_cfg["learning_rate"],
        per_device_train_batch_size=grpo_cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=grpo_cfg["gradient_accumulation_steps"],
        num_train_epochs=grpo_cfg["num_train_epochs"],
        warmup_ratio=grpo_cfg["warmup_ratio"],
        logging_steps=grpo_cfg["logging_steps"],
        save_strategy="steps",
        save_steps=grpo_cfg["save_steps"],
        bf16=grpo_cfg["bf16"],
        max_completion_length=grpo_cfg["max_completion_length"],
        remove_unused_columns=False,
    )

    trainer = GRPOTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        processing_class=tokenizer,
        reward_funcs=reward_fn,
    )

    print("🚀 Starting GRPO training...")
    print("=" * 60)
    print(f"  Group size (G):  {grpo_cfg['num_generations']}")
    print(f"  KL coefficient:  {grpo_cfg['kl_coef']}")
    print(f"  Temperature:     {grpo_cfg['temperature']}")
    print(f"  Learning rate:   {grpo_cfg['learning_rate']}")
    print("=" * 60)
    print("\nMetrics to watch:")
    print("  reward/mean        ↑  (average reward across all groups)")
    print("  reward/std         ↔  (within-group variance — too low means all responses are similar)")
    print("  grpo/advantages    ↔  (should have both positive and negative values)")
    print("  kl                 ↔  (KL from reference — same as PPO)")
    print()

    trainer.train()

    print(f"\n💾 Saving GRPO-aligned model to {output_dir}")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    print("\n✅ GRPO training complete!")
    print("Next: run 02_grpo_analysis.py for the three-way comparison.\n")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Create `phases/phase_03_grpo/02_grpo_analysis.py`**

```python
"""
Phase 3, Script 2: GRPO Analysis — Three-Way Comparison
=========================================================
Compare all three approaches: SFT baseline vs PPO vs DPO vs GRPO.
This is the dress rehearsal for the capstone.

Usage:
    python phases/phase_03_grpo/02_grpo_analysis.py
"""

import sys
import os

import torch
from transformers import AutoModelForCausalLM

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from shared.model_utils import MODEL_NAME, load_model_and_tokenizer
from shared.eval_utils import generate_samples, compare_outputs


EVAL_PROMPTS = [
    "Explain machine learning to a 10-year-old.",
    "What are the pros and cons of remote work?",
    "Write a short poem about coding.",
    "How do I deal with a difficult coworker?",
    "Explain why the sky is blue in simple terms.",
    "What is the most important thing in life?",
    "How should I prepare for a job interview?",
    "Summarize the concept of climate change in 3 sentences.",
]


MODEL_PATHS = {
    "PPO Aligned": "outputs/phase_01_ppo/ppo_model",
    "DPO Aligned": "outputs/phase_02_dpo/dpo_model",
    "GRPO Aligned": "outputs/phase_03_grpo/grpo_model",
}


def try_load_model(path, label):
    if os.path.exists(path):
        print(f"  Loading {label} from {path}...")
        return AutoModelForCausalLM.from_pretrained(
            path, torch_dtype=torch.bfloat16, device_map="auto",
        )
    print(f"  ⚠️  {label} not found at {path} — skipping")
    return None


def main():
    print("\n📊 Phase 3 Analysis: SFT vs PPO vs DPO vs GRPO\n")

    print("Loading models...")
    sft_model, tokenizer = load_model_and_tokenizer(MODEL_NAME)

    models = {}
    for label, path in MODEL_PATHS.items():
        m = try_load_model(path, label)
        if m is not None:
            models[label] = m

    print("\nGenerating responses...\n")
    model_outputs = {"SFT Baseline": generate_samples(sft_model, tokenizer, EVAL_PROMPTS)}

    for label, model in models.items():
        model_outputs[label] = generate_samples(model, tokenizer, EVAL_PROMPTS)

    compare_outputs(
        EVAL_PROMPTS,
        model_outputs,
        save_path="outputs/phase_03_grpo/three_way_comparison.json",
    )

    print("\n" + "=" * 60)
    print("🔍 THREE-WAY COMPARISON QUESTIONS:")
    print("=" * 60)
    print("""
Compare all three alignment methods across these dimensions:

1. OUTPUT QUALITY
   - Which produces the most helpful, relevant responses?
   - Do any show signs of reward hacking (exploiting patterns)?

2. TRAINING COMPLEXITY
   - PPO: 4 models, complex loop, many hyperparameters
   - DPO: 2 models, simple loss, few hyperparameters
   - GRPO: 3 models, group generation, moderate complexity
   Which felt most manageable?

3. COMPUTE COST
   - Wall-clock training time
   - Peak GPU memory usage
   - Total GPU-hours

4. STABILITY
   - Which training was most stable (smoothest loss curve)?
   - Which required the most hyperparameter tuning?

5. DATA REQUIREMENTS
   - PPO: Reward model + unlabeled prompts
   - DPO: Direct preference pairs
   - GRPO: Reward function + unlabeled prompts
   Which data format is easiest to obtain for your work?

📝 Write down your comparison notes — you'll formalize this in the capstone.

🔬 EXPERIMENT: Try GRPO with a rule-based reward function:
   - Reward = -abs(len(response) - 100)  (penalize responses far from 100 chars)
   - This demonstrates GRPO without a learned reward model
""")


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Commit**

```bash
git add phases/phase_03_grpo/
git commit -m "feat: add Phase 3 GRPO theory guide and training scripts"
```

---

### Task 7: Phase 4 — Capstone Comparison Framework

**Files:**
- Create: `phases/phase_04_capstone/README.md`
- Create: `phases/phase_04_capstone/configs/capstone_config.yaml`
- Create: `phases/phase_04_capstone/01_run_all_experiments.py`
- Create: `phases/phase_04_capstone/02_evaluate_and_compare.py`
- Create: `phases/phase_04_capstone/03_write_report.py`

- [ ] **Step 1: Create `phases/phase_04_capstone/README.md`**

```markdown
# Phase 4: Capstone — Comparative Experiment

**Duration:** 3-4 days
**Goal:** Run a controlled comparison of PPO, DPO, and GRPO on a work-relevant task. Produce a decision framework and reusable scripts.

---

## Instructions

### Day 17: Experiment Design

1. **Choose your task.** Pick something relevant to your work:
   - Instruction following (does the model follow formatting requirements?)
   - Summarization quality (concise, accurate, complete?)
   - Safety alignment (refuses harmful requests appropriately?)
   - Helpfulness (detailed, actionable answers?)

2. **Prepare your data.** You need:
   - A set of prompts (at least 200-500 for meaningful comparison)
   - For DPO: preference pairs (prompt, chosen, rejected)
   - For PPO/GRPO: a reward function (learned model or rule-based)

3. **Define evaluation metrics:**
   - Automated: reward model scores, response length, format compliance
   - Human: pick 50 responses to manually rate on a 1-5 scale

### Day 18-19: Run Experiments

Use `01_run_all_experiments.py` to train all three algorithms with matched settings.
Use `02_evaluate_and_compare.py` to generate the comparison table.

### Day 20: Write Report

Use `03_write_report.py` to auto-generate a report skeleton, then fill in your analysis.

The report should answer: **"Given task X with constraints Y, I would choose algorithm Z because..."**

---

## Deliverables

1. **Comparison table** — filled with your experimental results
2. **Decision framework** — when to use PPO vs DPO vs GRPO
3. **Reusable training scripts** — adapted from earlier phases for your team
4. **Lessons learned** — hyperparameter gotchas, failure modes, practical tips
```

- [ ] **Step 2: Create `phases/phase_04_capstone/configs/capstone_config.yaml`**

```yaml
# Capstone Experiment Configuration
# Unified config for running all three algorithms under matched conditions.

model:
  name: "Qwen/Qwen2.5-1.5B-Instruct"
  torch_dtype: "bfloat16"

dataset:
  name: "argilla/ultrafeedback-binarized-preferences-cleaned"
  max_samples: 1000
  max_prompt_length: 256
  max_length: 512

reward_model:
  name: "Ray2333/GRM-Gemma-2B-sftreg"
  torch_dtype: "bfloat16"

# Matched training settings across algorithms
common:
  learning_rate: 5.0e-6
  num_train_epochs: 1
  bf16: true
  warmup_ratio: 0.1
  logging_steps: 10
  save_steps: 100
  gradient_accumulation_steps: 4
  per_device_train_batch_size: 4

ppo:
  init_kl_coef: 0.05
  clip_range: 0.2
  ppo_epochs: 2
  max_new_tokens: 128

dpo:
  beta: 0.1
  max_length: 512
  max_prompt_length: 256

grpo:
  num_generations: 8
  max_completion_length: 256
  kl_coef: 0.05
  temperature: 0.8

evaluation:
  num_eval_prompts: 100
  max_new_tokens: 256
  temperature: 0.7
  eval_prompts_file: null  # Set to a file path to use custom prompts
  output_dir: "outputs/phase_04_capstone"
```

- [ ] **Step 3: Create `phases/phase_04_capstone/01_run_all_experiments.py`**

```python
"""
Phase 4, Script 1: Run All Experiments
========================================
Train PPO, DPO, and GRPO under matched conditions for fair comparison.
This is the unified training script for the capstone.

Usage:
    python phases/phase_04_capstone/01_run_all_experiments.py [--algo ppo|dpo|grpo|all]
"""

import argparse
import sys
import os
import time
import json
import yaml
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)
from trl import (
    DPOConfig, DPOTrainer,
    GRPOConfig, GRPOTrainer,
)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from shared.model_utils import MODEL_NAME, print_model_info


def load_config():
    config_path = os.path.join(os.path.dirname(__file__), "configs", "capstone_config.yaml")
    with open(config_path) as f:
        return yaml.safe_load(f)


def prepare_dataset(config):
    ds = load_dataset(config["dataset"]["name"], split="train")
    ds = ds.shuffle(seed=42).select(range(min(config["dataset"]["max_samples"], len(ds))))
    return ds


def run_dpo(config, dataset):
    print("\n" + "=" * 60)
    print("  Running DPO Experiment")
    print("=" * 60)

    dpo_cfg = config["dpo"]
    common = config["common"]
    output_dir = os.path.join(config["evaluation"]["output_dir"], "dpo_model")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.bfloat16, device_map="auto",
    )
    ref_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.bfloat16, device_map="auto",
    )

    def format_dpo(example):
        return {
            "prompt": example["prompt"],
            "chosen": example["chosen"],
            "rejected": example["rejected"],
        }

    ds = dataset.map(format_dpo, remove_columns=[
        c for c in dataset.column_names if c not in ["prompt", "chosen", "rejected"]
    ])
    split = ds.train_test_split(test_size=0.1, seed=42)

    training_args = DPOConfig(
        output_dir=output_dir,
        beta=dpo_cfg["beta"],
        learning_rate=common["learning_rate"],
        per_device_train_batch_size=common["per_device_train_batch_size"],
        gradient_accumulation_steps=common["gradient_accumulation_steps"],
        num_train_epochs=common["num_train_epochs"],
        warmup_ratio=common["warmup_ratio"],
        logging_steps=common["logging_steps"],
        save_strategy="steps",
        save_steps=common["save_steps"],
        bf16=common["bf16"],
        max_length=dpo_cfg["max_length"],
        max_prompt_length=dpo_cfg["max_prompt_length"],
        remove_unused_columns=False,
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=training_args,
        train_dataset=split["train"],
        eval_dataset=split["test"],
        processing_class=tokenizer,
    )

    start_time = time.time()
    trainer.train()
    elapsed = time.time() - start_time

    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    return {"algorithm": "DPO", "training_time_seconds": elapsed, "output_dir": output_dir}


def run_grpo(config, dataset):
    print("\n" + "=" * 60)
    print("  Running GRPO Experiment")
    print("=" * 60)

    grpo_cfg = config["grpo"]
    common = config["common"]
    output_dir = os.path.join(config["evaluation"]["output_dir"], "grpo_model")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.bfloat16, device_map="auto",
    )

    rm_name = config["reward_model"]["name"]
    rm_tokenizer = AutoTokenizer.from_pretrained(rm_name)
    if rm_tokenizer.pad_token is None:
        rm_tokenizer.pad_token = rm_tokenizer.eos_token
    rm_model = AutoModelForSequenceClassification.from_pretrained(
        rm_name, num_labels=1, torch_dtype=torch.bfloat16, device_map="auto",
    )
    rm_model.eval()

    def reward_fn(completions, **kwargs):
        rewards = []
        for completion in completions:
            text = completion if isinstance(completion, str) else completion[0]["content"]
            inputs = rm_tokenizer(
                text, return_tensors="pt", truncation=True, max_length=512, padding=True,
            ).to(rm_model.device)
            with torch.no_grad():
                score = rm_model(**inputs).logits.squeeze().cpu().float().item()
            rewards.append(score)
        return rewards

    def format_grpo(example):
        return {"prompt": example["prompt"]}

    ds = dataset.map(format_grpo, remove_columns=[
        c for c in dataset.column_names if c != "prompt"
    ])

    training_args = GRPOConfig(
        output_dir=output_dir,
        num_generations=grpo_cfg["num_generations"],
        learning_rate=common["learning_rate"],
        per_device_train_batch_size=common["per_device_train_batch_size"] // 2,
        gradient_accumulation_steps=common["gradient_accumulation_steps"] * 2,
        num_train_epochs=common["num_train_epochs"],
        warmup_ratio=common["warmup_ratio"],
        logging_steps=common["logging_steps"],
        save_strategy="steps",
        save_steps=common["save_steps"],
        bf16=common["bf16"],
        max_completion_length=grpo_cfg["max_completion_length"],
        remove_unused_columns=False,
    )

    trainer = GRPOTrainer(
        model=model,
        args=training_args,
        train_dataset=ds,
        processing_class=tokenizer,
        reward_funcs=reward_fn,
    )

    start_time = time.time()
    trainer.train()
    elapsed = time.time() - start_time

    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    return {"algorithm": "GRPO", "training_time_seconds": elapsed, "output_dir": output_dir}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", choices=["ppo", "dpo", "grpo", "all"], default="all")
    args = parser.parse_args()

    config = load_config()
    dataset = prepare_dataset(config)

    results = []

    if args.algo in ("dpo", "all"):
        results.append(run_dpo(config, dataset))

    if args.algo in ("grpo", "all"):
        results.append(run_grpo(config, dataset))

    if args.algo in ("ppo", "all"):
        print("\n⚠️  PPO capstone training requires the Phase 1 PPO pipeline.")
        print("Copy your trained PPO model from outputs/phase_01_ppo/ppo_model")
        print("to outputs/phase_04_capstone/ppo_model for comparison.\n")

    # Save timing results
    results_path = os.path.join(config["evaluation"]["output_dir"], "training_results.json")
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n📊 Training results saved to {results_path}")
    print("Next: run 02_evaluate_and_compare.py\n")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Create `phases/phase_04_capstone/02_evaluate_and_compare.py`**

```python
"""
Phase 4, Script 2: Evaluate and Compare
==========================================
Generate the comparison table across all three algorithms.
This is the core deliverable of the capstone.

Usage:
    python phases/phase_04_capstone/02_evaluate_and_compare.py
"""

import sys
import os
import json
import time

import torch
import pandas as pd
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from shared.model_utils import MODEL_NAME, load_model_and_tokenizer
from shared.eval_utils import generate_samples, compare_outputs


DEFAULT_EVAL_PROMPTS = [
    "Explain machine learning to a 10-year-old.",
    "What are the pros and cons of remote work?",
    "Write a short poem about coding.",
    "How do I deal with a difficult coworker?",
    "Explain why the sky is blue in simple terms.",
    "What is the most important thing in life?",
    "How should I prepare for a job interview?",
    "Summarize the concept of climate change in 3 sentences.",
    "What makes a good leader?",
    "Explain quantum computing in simple terms.",
    "How can I improve my public speaking skills?",
    "What are the ethical concerns around AI?",
    "Write a motivational message for someone learning to code.",
    "How does blockchain technology work?",
    "What habits lead to a productive morning routine?",
    "Explain the difference between empathy and sympathy.",
    "How should I approach learning a new programming language?",
    "What are the benefits of reading books regularly?",
    "How can teams collaborate more effectively remotely?",
    "Explain the concept of compound interest simply.",
]


def load_config():
    config_path = os.path.join(os.path.dirname(__file__), "configs", "capstone_config.yaml")
    with open(config_path) as f:
        return yaml.safe_load(f)


def score_responses(responses, rm_model, rm_tokenizer):
    scores = []
    for resp in responses:
        inputs = rm_tokenizer(
            resp, return_tensors="pt", truncation=True, max_length=512, padding=True,
        ).to(rm_model.device)
        with torch.no_grad():
            score = rm_model(**inputs).logits.squeeze().cpu().float().item()
        scores.append(score)
    return scores


def main():
    config = load_config()
    output_dir = config["evaluation"]["output_dir"]
    eval_cfg = config["evaluation"]

    print("\n📊 Capstone Evaluation: PPO vs DPO vs GRPO\n")

    # Load eval prompts
    if eval_cfg.get("eval_prompts_file") and os.path.exists(eval_cfg["eval_prompts_file"]):
        with open(eval_cfg["eval_prompts_file"]) as f:
            eval_prompts = [line.strip() for line in f if line.strip()]
    else:
        eval_prompts = DEFAULT_EVAL_PROMPTS[:eval_cfg.get("num_eval_prompts", 20)]

    print(f"Using {len(eval_prompts)} evaluation prompts\n")

    # Load models
    print("Loading models...")
    sft_model, tokenizer = load_model_and_tokenizer(MODEL_NAME)

    model_configs = {
        "PPO": os.path.join(output_dir, "ppo_model"),
        "DPO": os.path.join(output_dir, "dpo_model"),
        "GRPO": os.path.join(output_dir, "grpo_model"),
    }

    aligned_models = {}
    for name, path in model_configs.items():
        if os.path.exists(path):
            print(f"  Loading {name} from {path}...")
            aligned_models[name] = AutoModelForCausalLM.from_pretrained(
                path, torch_dtype=torch.bfloat16, device_map="auto",
            )
        else:
            print(f"  ⚠️  {name} not found at {path} — skipping")

    # Load reward model for scoring
    print("\nLoading reward model for scoring...")
    rm_name = config["reward_model"]["name"]
    rm_tokenizer = AutoTokenizer.from_pretrained(rm_name)
    if rm_tokenizer.pad_token is None:
        rm_tokenizer.pad_token = rm_tokenizer.eos_token
    rm_model = AutoModelForSequenceClassification.from_pretrained(
        rm_name, num_labels=1, torch_dtype=torch.bfloat16, device_map="auto",
    )
    rm_model.eval()

    # Generate and score
    print("\nGenerating and scoring responses...\n")

    results = {"SFT": {}}
    sft_outputs = generate_samples(sft_model, tokenizer, eval_prompts,
                                   max_new_tokens=eval_cfg.get("max_new_tokens", 256))
    sft_scores = score_responses(sft_outputs, rm_model, rm_tokenizer)
    results["SFT"] = {
        "outputs": sft_outputs,
        "scores": sft_scores,
        "avg_score": sum(sft_scores) / len(sft_scores),
        "avg_length": sum(len(o) for o in sft_outputs) / len(sft_outputs),
    }

    for name, model in aligned_models.items():
        outputs = generate_samples(model, tokenizer, eval_prompts,
                                   max_new_tokens=eval_cfg.get("max_new_tokens", 256))
        scores = score_responses(outputs, rm_model, rm_tokenizer)
        results[name] = {
            "outputs": outputs,
            "scores": scores,
            "avg_score": sum(scores) / len(scores),
            "avg_length": sum(len(o) for o in outputs) / len(outputs),
        }

    # Load timing results if available
    timing_path = os.path.join(output_dir, "training_results.json")
    timing = {}
    if os.path.exists(timing_path):
        with open(timing_path) as f:
            for entry in json.load(f):
                timing[entry["algorithm"]] = entry["training_time_seconds"]

    # Build comparison table
    print("\n" + "=" * 80)
    print("  COMPARISON TABLE")
    print("=" * 80)

    table_data = []
    for name in ["SFT", "PPO", "DPO", "GRPO"]:
        if name not in results:
            continue
        r = results[name]
        row = {
            "Algorithm": name,
            "Avg Reward Score": f"{r['avg_score']:.3f}",
            "Avg Response Length": f"{r['avg_length']:.0f}",
            "Training Time (s)": f"{timing.get(name, 'N/A')}",
            "Models in Memory": {"SFT": "1", "PPO": "4", "DPO": "2", "GRPO": "3"}.get(name, "?"),
            "Online/Offline": {"SFT": "N/A", "PPO": "Online", "DPO": "Offline", "GRPO": "Online"}.get(name, "?"),
        }
        table_data.append(row)

    df = pd.DataFrame(table_data)
    print(df.to_string(index=False))

    # Save detailed results
    detailed_path = os.path.join(output_dir, "evaluation_results.json")
    save_results = {name: {k: v for k, v in data.items() if k != "outputs"}
                    for name, data in results.items()}
    with open(detailed_path, "w") as f:
        json.dump(save_results, f, indent=2)

    # Save comparison outputs
    model_outputs = {name: data["outputs"] for name, data in results.items()}
    compare_outputs(
        eval_prompts,
        model_outputs,
        save_path=os.path.join(output_dir, "all_outputs_comparison.json"),
    )

    print(f"\n📊 Detailed results saved to {detailed_path}")
    print("Next: run 03_write_report.py to generate the report skeleton.\n")


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Create `phases/phase_04_capstone/03_write_report.py`**

```python
"""
Phase 4, Script 3: Generate Report Skeleton
=============================================
Auto-generates a comparison report from your experiment results.
Fill in the analysis sections with your observations.

Usage:
    python phases/phase_04_capstone/03_write_report.py
"""

import os
import json
from datetime import datetime


def load_results(output_dir):
    eval_path = os.path.join(output_dir, "evaluation_results.json")
    timing_path = os.path.join(output_dir, "training_results.json")

    eval_results = {}
    timing_results = {}

    if os.path.exists(eval_path):
        with open(eval_path) as f:
            eval_results = json.load(f)

    if os.path.exists(timing_path):
        with open(timing_path) as f:
            for entry in json.load(f):
                timing_results[entry["algorithm"]] = entry

    return eval_results, timing_results


def generate_report(eval_results, timing_results, output_dir):
    date = datetime.now().strftime("%Y-%m-%d")

    report = f"""# LLM Alignment Algorithm Comparison Report

**Date:** {date}
**Base Model:** Qwen2.5-1.5B-Instruct
**Algorithms Compared:** PPO, DPO, GRPO

---

## 1. Experiment Setup

- **Task:** [FILL IN: What task did you train on?]
- **Dataset:** [FILL IN: Dataset name and size]
- **Hardware:** [FILL IN: GPU type and count]
- **Training duration:** 1 epoch each, matched learning rate and batch size

---

## 2. Quantitative Results

| Metric | SFT (Baseline) | PPO | DPO | GRPO |
|--------|----------------|-----|-----|------|
"""

    for metric in ["avg_score", "avg_length"]:
        label = "Avg Reward Score" if metric == "avg_score" else "Avg Response Length"
        row = f"| {label} |"
        for algo in ["SFT", "PPO", "DPO", "GRPO"]:
            if algo in eval_results and metric in eval_results[algo]:
                val = eval_results[algo][metric]
                row += f" {val:.3f} |" if metric == "avg_score" else f" {val:.0f} |"
            else:
                row += " N/A |"
        report += row + "\n"

    row = "| Training Time (s) |"
    for algo in ["SFT", "PPO", "DPO", "GRPO"]:
        if algo in timing_results:
            row += f" {timing_results[algo]['training_time_seconds']:.0f} |"
        else:
            row += " N/A |"
    report += row + "\n"

    report += """
---

## 3. Qualitative Observations

### PPO
- **Output quality:** [FILL IN: Your observations]
- **Training stability:** [FILL IN: Was it stable? Any reward hacking?]
- **Difficulty:** [FILL IN: How hard was it to get working?]

### DPO
- **Output quality:** [FILL IN: Your observations]
- **Training stability:** [FILL IN: Was it stable? Any overfitting?]
- **Difficulty:** [FILL IN: How hard was it to get working?]

### GRPO
- **Output quality:** [FILL IN: Your observations]
- **Training stability:** [FILL IN: Was it stable? Group size effects?]
- **Difficulty:** [FILL IN: How hard was it to get working?]

---

## 4. Decision Framework

**Use PPO when:**
- [FILL IN based on your experience]

**Use DPO when:**
- [FILL IN based on your experience]

**Use GRPO when:**
- [FILL IN based on your experience]

---

## 5. Practical Tips & Gotchas

### Hyperparameter Sensitivity
- [FILL IN: Which hyperparameters mattered most for each algorithm?]

### Common Failure Modes
- [FILL IN: What went wrong during training and how did you fix it?]

### Recommended Defaults
- [FILL IN: Your recommended starting hyperparameters for each algorithm]

---

## 6. Recommended Next Steps

- [FILL IN: What would you explore further?]
- [FILL IN: How would you apply this to your team's specific use case?]
"""

    report_path = os.path.join(output_dir, "comparison_report.md")
    with open(report_path, "w") as f:
        f.write(report)

    return report_path


def main():
    output_dir = "outputs/phase_04_capstone"
    os.makedirs(output_dir, exist_ok=True)

    eval_results, timing_results = load_results(output_dir)

    report_path = generate_report(eval_results, timing_results, output_dir)

    print(f"\n📝 Report skeleton generated at: {report_path}")
    print("\nThe report has pre-filled quantitative results from your experiments.")
    print("Fill in the [FILL IN] sections with your qualitative observations.")
    print("\nThis report is your capstone deliverable — present it to your team!\n")


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Commit**

```bash
git add phases/phase_04_capstone/
git commit -m "feat: add Phase 4 capstone comparison framework"
```

---

### Task 8: Root README and Final Verification

**Files:**
- Create: `README.md`

- [ ] **Step 1: Create root `README.md`**

```markdown
# RL Learning Path: PPO, DPO, GRPO for LLM Alignment

A self-contained learning path to master three key reinforcement learning algorithms for LLM alignment. Go from zero RL knowledge to running and comparing PPO, DPO, and GRPO in 3-4 weeks.

## Quick Start

```bash
# 1. Set up environment
chmod +x setup_env.sh && bash setup_env.sh
source .venv/bin/activate

# 2. Start with Phase 0 theory
open phases/phase_00_foundation/README.md
```

## Learning Path

| Phase | Topic | What You'll Do |
|-------|-------|---------------|
| 0 | [Foundation](phases/phase_00_foundation/README.md) | Read theory: RLHF pipeline, RL concepts, reward models |
| 1 | [PPO](phases/phase_01_ppo/README.md) | Read theory + run: train reward model, run PPO, analyze results |
| 2 | [DPO](phases/phase_02_dpo/README.md) | Read theory + run: train DPO, compare with PPO |
| 3 | [GRPO](phases/phase_03_grpo/README.md) | Read theory + run: train GRPO, three-way comparison |
| 4 | [Capstone](phases/phase_04_capstone/README.md) | Run all three on a work task, write comparison report |

## Requirements

- Python 3.10+
- CUDA-capable GPU (A100/H100 recommended)
- ~20GB disk space for model weights and outputs

## Structure

- `phases/` — One directory per learning phase, each with a README (theory) and Python scripts (hands-on)
- `shared/` — Common utilities for model loading, data preparation, and evaluation
- `outputs/` — Created during training, stores models and results (git-ignored)
```

- [ ] **Step 2: Create `.gitignore`**

```
.venv/
outputs/
__pycache__/
*.pyc
.DS_Store
wandb/
*.egg-info/
dist/
build/
```

- [ ] **Step 3: Verify all files exist**

Run:
```bash
find phases/ shared/ -type f | sort
```

Expected output:
```
phases/phase_00_foundation/README.md
phases/phase_01_ppo/01_train_reward_model.py
phases/phase_01_ppo/02_ppo_training.py
phases/phase_01_ppo/03_ppo_analysis.py
phases/phase_01_ppo/README.md
phases/phase_01_ppo/configs/ppo_config.yaml
phases/phase_02_dpo/01_dpo_training.py
phases/phase_02_dpo/02_dpo_analysis.py
phases/phase_02_dpo/README.md
phases/phase_02_dpo/configs/dpo_config.yaml
phases/phase_03_grpo/01_grpo_training.py
phases/phase_03_grpo/02_grpo_analysis.py
phases/phase_03_grpo/README.md
phases/phase_03_grpo/configs/grpo_config.yaml
phases/phase_04_capstone/01_run_all_experiments.py
phases/phase_04_capstone/02_evaluate_and_compare.py
phases/phase_04_capstone/03_write_report.py
phases/phase_04_capstone/README.md
phases/phase_04_capstone/configs/capstone_config.yaml
shared/__init__.py
shared/data_utils.py
shared/eval_utils.py
shared/model_utils.py
```

- [ ] **Step 4: Final commit**

```bash
git add README.md .gitignore
git commit -m "docs: add root README and gitignore"
```

---

## Self-Review

**Spec coverage check:**
- Phase 0 (Foundation theory): Covered in Task 2 ✓
- Phase 1 (PPO theory + hands-on): Covered in Tasks 3-4 ✓
- Phase 2 (DPO theory + hands-on): Covered in Task 5 ✓
- Phase 3 (GRPO theory + hands-on): Covered in Task 6 ✓
- Phase 4 (Capstone comparison): Covered in Task 7 ✓
- Shared utilities: Covered in Task 1 ✓
- Key resources/reading: Embedded in each phase README ✓

**Placeholder scan:** No TBD/TODO in code. The capstone report template has intentional `[FILL IN]` markers (these are for the learner, not unfinished plan items).

**Type consistency:** All scripts use `MODEL_NAME` from `shared.model_utils`, consistent config loading pattern, consistent dataset preparation pattern.
