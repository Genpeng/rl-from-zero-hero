# PPO / DPO / GRPO Learning Path — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Complete a focused 6-week learning path that gives hands-on mastery of PPO, DPO, and GRPO for alignment work in an NLP/ML role.

**Architecture:** Just-in-time RL foundation (Week 1) → PPO for LLMs (Weeks 2–3) → DPO as the leaner alternative (Week 4) → GRPO for reasoning tasks (Weeks 5–6). Each module pairs theory reading with hands-on TRL training runs, culminating in a personal decision framework.

**Tech Stack:** Python · PyTorch · Hugging Face TRL (`RewardTrainer`, `PPOTrainer`, `DPOTrainer`, `GRPOTrainer`) · `transformers` · `peft` · `datasets` · CleanRL · Cloud GPU (A100/H100) · Qwen model family

---

## Task 0: File Setup

**Files:**
- Commit: `Lukas's Notes/LLM/docs/superpowers/specs/2026-04-21-ppo-dpo-grpo-learning-path-design.md`
- Commit: `Lukas's Notes/LLM/2026-04-21-rlhf-learning-path-design-v1-archived.md`

- [ ] **Step 1: Verify the spec file exists**

```bash
ls "Lukas's Notes/LLM/docs/superpowers/specs/"
# Expected: 2026-04-21-ppo-dpo-grpo-learning-path-design.md
```

- [ ] **Step 2: Stage and commit both files**

```bash
git add "Lukas's Notes/LLM/docs/superpowers/specs/2026-04-21-ppo-dpo-grpo-learning-path-design.md"
git add "Lukas's Notes/LLM/2026-04-21-rlhf-learning-path-design-v1-archived.md"
git commit -m "add 6-week PPO/DPO/GRPO learning path spec, archive old 10-week RLHF plan"
```

---

## Task 1: Week 1 — RL Essentials (6–8 hrs)

**Goal:** Build just enough RL intuition to understand PPO when applied to LLMs.

**Files:**
- Create: `Lukas's Notes/LLM/PPO.md` (notes stub, to fill as you read)
- Read: CleanRL PPO implementation (see Step 5)

- [ ] **Step 1: Read OpenAI Spinning Up — "Key Concepts in RL" (~1 hr)**

Focus on: MDP (Markov Decision Process), state, action, reward, policy, return, discount factor γ.

Stop after the "Key Concepts" page. Do NOT continue to the algorithms section yet.

Write a 5-line summary in `PPO.md`:
```markdown
## RL Fundamentals
- **State (s)**: the environment's current situation (in LLMs: prompt + generated tokens so far)
- **Action (a)**: what the agent does (in LLMs: next token to generate)
- **Reward (r)**: signal measuring how good the action was
- **Policy (π)**: the function mapping states to actions (in LLMs: the LLM itself)
- **Return (G)**: sum of future discounted rewards from a given state
```

- [ ] **Step 2: Read Lilian Weng — "Policy Gradient Algorithms" (~1.5 hrs)**

Focus on: REINFORCE section only. Key idea: the log-probability trick — why `∇E[R] = E[R · ∇log π(a|s)]`.

Stop before the "Actor-Critic" section. Add to `PPO.md`:
```markdown
## Policy Gradient / REINFORCE
- **Core idea**: we can't differentiate through the sampling step, but we can use the log-prob trick
- **REINFORCE objective**: maximize E[R(τ) · log π(a|s)], where τ is a trajectory
- **Problem**: high variance — reward signal is noisy and updates can be catastrophic
```

- [ ] **Step 3: Read PPO paper (Schulman et al. 2017) — Section 3 only (~1.5 hrs)**

Download: https://arxiv.org/abs/1707.06347

Read only Section 3: "Clipped Surrogate Objective". Key formulas:
- `r_t(θ) = π_θ(a|s) / π_θ_old(a|s)` — the policy ratio
- `L_CLIP = E[min(r_t · Â_t, clip(r_t, 1-ε, 1+ε) · Â_t)]` — the clipped objective

Add to `PPO.md`:
```markdown
## PPO
- **Problem with vanilla policy gradient**: unconstrained updates can collapse the policy
- **Trust region**: don't let the new policy stray too far from the old one
- **Clipping trick**: clip the ratio r_t to [1-ε, 1+ε] — equivalent to a trust region but simpler
- **Advantage (Â)**: tells PPO whether an action was better (+) or worse (-) than expected
- **GAE**: Generalized Advantage Estimation — smoothed advantage using a λ parameter
- **ε = 0.2** is the standard clip range
```

- [ ] **Step 4: Run the CleanRL PPO on CartPole (~2 hrs)**

Install CleanRL:
```bash
pip install cleanrl
# Or clone the repo for code inspection
git clone https://github.com/vwxyzjn/cleanrl.git
```

Read the PPO implementation file (don't run yet):
```bash
cat cleanrl/cleanrl/ppo.py | head -200
```
Key things to find in the code:
1. Where is the ratio `r_t` computed? (look for `logratio`)
2. Where is clipping applied? (look for `torch.clamp`)
3. Where is the advantage computed? (look for `advantages`)

Run CartPole:
```bash
python cleanrl/cleanrl/ppo.py --env-id CartPole-v1 --total-timesteps 50000
```

Expected output: episodic reward starting around 20–50, climbing to 300–500 by the end.

- [ ] **Step 5: Write Week 1 milestone note**

Add to `PPO.md`:
```markdown
## Week 1 Milestone
"PPO prevents catastrophic policy updates by clipping the ratio between new and old policy
probabilities to [1-ε, 1+ε]. The advantage function tells it which actions to reinforce more
strongly (positive advantage) or discourage (negative advantage)."
```

- [ ] **Step 6: Commit Week 1 notes**

```bash
git add "Lukas's Notes/LLM/PPO.md"
git commit -m "add Week 1 notes: RL essentials, PPO theory"
```

---

## Task 2: Week 2 — Reward Model Training (6–7 hrs)

**Goal:** Understand the reward model component of RLHF and train one.

**Files:**
- Modify: `Lukas's Notes/LLM/PPO.md` (add RLHF section)
- Create: `rlhf/train_reward_model.py`

- [ ] **Step 1: Read InstructGPT paper sections 3.2–3.4 (~2 hrs)**

Download: https://arxiv.org/abs/2203.02155

Read only Sections 3.2 (reward model) and 3.3 (reinforcement learning). Key ideas:
- Reward model architecture: same LLM backbone, replace the final LM head with a scalar head
- Bradley-Terry model: P(y_w > y_l | x) = σ(r(x, y_w) − r(x, y_l)) — the loss that compares pairs
- The dataset: "chosen" response (human preferred) vs "rejected" response (human dispreferred)

Add to `PPO.md`:
```markdown
## RLHF — Reward Model
- **Why?** We can't write a loss function for "be helpful and harmless" — humans must signal preferences
- **Bradley-Terry loss**: for chosen y_w and rejected y_l: loss = -log σ(r(y_w) - r(y_l))
- **Architecture**: LLM backbone + linear head outputing a single scalar (the reward)
- **Dataset format**: (prompt, chosen_response, rejected_response) triples
- **HH-RLHF**: Anthropic's public preference dataset — harmlessness + helpfulness annotations
```

- [ ] **Step 2: Set up the training environment**

```bash
pip install trl transformers datasets accelerate peft
```

- [ ] **Step 3: Write the reward model training script**

Create `rlhf/train_reward_model.py`:

```python
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
```

- [ ] **Step 4: Run the reward model training**

```bash
python rlhf/train_reward_model.py
```

Expected: training loss should decrease from ~0.69 (random) to ~0.55–0.60 within epoch 1.

- [ ] **Step 5: Verify — do chosen responses score higher than rejected?**

Add verification to the end of `train_reward_model.py` and run it:

```python
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

model = AutoModelForSequenceClassification.from_pretrained("./reward_model_final")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B-Instruct")

# Test on 5 examples from the dataset
from datasets import load_dataset
test = load_dataset("Anthropic/hh-rlhf", split="test[:5]")

model.eval()
for row in test:
    with torch.no_grad():
        c = tokenizer(row["chosen"], return_tensors="pt", truncation=True, max_length=512)
        r = tokenizer(row["rejected"], return_tensors="pt", truncation=True, max_length=512)
        chosen_score = model(**c).logits.item()
        rejected_score = model(**r).logits.item()
    print(f"Chosen: {chosen_score:.3f} | Rejected: {rejected_score:.3f} | {'✓' if chosen_score > rejected_score else '✗'}")
```

Expected: chosen score > rejected score in most (>60%) examples.

- [ ] **Step 6: Commit**

```bash
git add "Lukas's Notes/LLM/PPO.md" rlhf/train_reward_model.py
git commit -m "add reward model training script and Week 2 notes"
```

---

## Task 3: Week 3 — PPO Alignment + Reward Hacking Experiment (6–7 hrs)

**Goal:** Run full PPO-RLHF alignment loop, then observe reward hacking when KL penalty is removed.

**Files:**
- Create: `rlhf/train_ppo.py`
- Create: `rlhf/train_ppo_no_kl.py` (reward hacking experiment)
- Modify: `Lukas's Notes/LLM/PPO.md`

- [ ] **Step 1: Read "Secrets of RLHF" (Zheng et al. 2023) — §1–3 (~2 hrs)**

Download: https://arxiv.org/abs/2307.04964

Focus on: the PPO training loop diagram (Figure 1), the role of KL penalty, common failure modes.

Add to `PPO.md`:
```markdown
## PPO for LLMs — Training Loop
1. **Sample**: use the current policy (LLM) to generate responses for a batch of prompts
2. **Score**: pass each (prompt, response) through the reward model → scalar reward
3. **KL penalty**: subtract β · KL(π_new || π_ref) from reward — penalizes straying from SFT model
4. **PPO update**: run PPO update steps using the adjusted reward and clipped objective
5. **Repeat**: iterate until reward plateaus or KL budget is exhausted

**4 models in memory during PPO:**
- π_new: the model being trained (policy)
- π_ref: the frozen SFT model (reference, for KL)
- Reward model: scores responses
- Value model (critic): estimates expected future reward (often initialized from π_ref)

**Reward hacking**: without KL penalty, the model learns to produce outputs the RM scores
highly that are actually low quality — e.g., repetitive text, sycophancy
```

- [ ] **Step 2: Write the PPO training script**

Create `rlhf/train_ppo.py`:

```python
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
    kl_penalty="kl",        # <- KL constraint active
    init_kl_coef=0.2,       # β coefficient
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
```

- [ ] **Step 3: Run PPO training**

```bash
python rlhf/train_ppo.py
```

Expected: reward should gradually increase (e.g., from -0.5 to +0.5 over 10 steps). KL should stay below 10.

- [ ] **Step 4: Run the reward hacking experiment**

Create `rlhf/train_ppo_no_kl.py` — identical to `train_ppo.py` but change these two lines:

```python
# Change in PPOConfig:
kl_penalty="none",   # <- remove KL penalty
init_kl_coef=0.0,
```

```bash
python rlhf/train_ppo_no_kl.py
```

Expected: reward inflates much faster (the model learns to game the reward model), but qualitative outputs will degrade — repetitive or sycophantic responses.

- [ ] **Step 5: Compare outputs from both models**

Add to the end of `train_ppo.py` (or a separate `compare.py`):

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

test_prompt = "Human: What is the capital of France?\n\nAssistant:"
tok = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B-Instruct")

for name, path in [("Base", "Qwen/Qwen2-0.5B-Instruct"),
                    ("PPO (with KL)", "./ppo_aligned_model"),
                    ("PPO (no KL — hacked)", "./ppo_nkl_model")]:
    m = AutoModelForCausalLM.from_pretrained(path)
    inputs = tok(test_prompt, return_tensors="pt")
    out = m.generate(**inputs, max_new_tokens=100, do_sample=False)
    print(f"\n=== {name} ===")
    print(tok.decode(out[0], skip_special_tokens=True))
```

Observe: the no-KL model likely produces unusual repetitive or overly-positive outputs.

- [ ] **Step 6: Add milestone note and commit**

Add to `PPO.md`:
```markdown
## Week 3 Milestone
"Has run the complete RLHF pipeline end-to-end (reward model → PPO alignment) and
observed reward hacking first-hand: without KL constraint, the model games the reward
model rather than becoming genuinely more helpful."
```

```bash
git add "Lukas's Notes/LLM/PPO.md" rlhf/train_ppo.py rlhf/train_ppo_no_kl.py
git commit -m "add PPO training scripts, reward hacking experiment, Week 3 notes"
```

---

## Task 4: Week 4 — DPO (8–10 hrs)

**Goal:** Understand DPO's closed-form insight, train a model with DPO, compare it to PPO.

**Files:**
- Create: `Lukas's Notes/LLM/DPO.md`
- Create: `rlhf/train_dpo.py`

- [ ] **Step 1: Read DPO paper (Rafailov et al. 2023) — full (~3 hrs)**

Download: https://arxiv.org/abs/2305.18290

Focus on Sections 4 and 5. The key derivation to understand:

1. RLHF solves: `max_π E[r(x,y)] - β · KL(π || π_ref)`
2. The optimal solution is: `π*(y|x) ∝ π_ref(y|x) · exp(r(x,y)/β)`
3. Rearranging: `r(x,y) = β · log(π*(y|x)/π_ref(y|x)) + β · log Z(x)`
4. Substituting into the Bradley-Terry loss gives the DPO objective:

```
L_DPO = -E[ log σ( β·log(π(y_w|x)/π_ref(y_w|x)) - β·log(π(y_l|x)/π_ref(y_l|x)) ) ]
```

This is just binary cross-entropy on the log-ratio difference. No reward model. No RL loop.

Create `DPO.md`:
```markdown
# DPO (Direct Preference Optimization)

## The Core Insight
The optimal policy under KL-constrained RLHF has a closed form:
`π*(y|x) ∝ π_ref(y|x) · exp(r(x,y)/β)`

This means the reward can be expressed as a function of the policy itself:
`r(x,y) = β · log(π(y|x)/π_ref(y|x)) + const`

So we can substitute this into the Bradley-Terry preference loss and get a loss
directly on the policy — no reward model needed.

## DPO Objective
`L = -E[ log σ( β · (log π(y_w|x)/π_ref(y_w|x)) - β · (log π(y_l|x)/π_ref(y_l|x)) ) ]`

Where:
- y_w = chosen (winning) response
- y_l = rejected (losing) response
- β = temperature controlling deviation from reference policy (typically 0.1–0.5)
- π_ref = frozen SFT model (reference)

## DPO vs PPO
| | PPO | DPO |
|-|-----|-----|
| Reward model needed? | Yes | No |
| Models in memory | 4 | 2 (policy + reference) |
| Stability | Can be unstable | Generally stable |
| Online learning | Yes (explores) | No (offline only) |
| Compute | High | Low |
| When to prefer | Complex rewards, online setting | Preference data available, offline |

## Variants (awareness level)
- **IPO**: replaces σ with identity to avoid overfit to hard preferences
- **KTO**: uses thumbs-up/thumbs-down data instead of pairs
- **SimPO**: removes reference model dependency entirely
```

- [ ] **Step 2: Read Sebastian Raschka's DPO explainer (~30 min)**

Find it at: https://magazine.sebastianraschka.com/p/llm-training-rlhf-and-its-alternatives

Focus on the diagrams comparing PPO and DPO pipelines.

- [ ] **Step 3: Write the DPO training script**

Create `rlhf/train_dpo.py`:

```python
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOTrainer, DPOConfig

model_name = "Qwen/Qwen2-0.5B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(model_name)

# Load and format preference dataset
dataset = load_dataset("Anthropic/hh-rlhf", split="train[:2000]")

def format_dataset(row):
    # Extract the last human turn as the prompt
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
    beta=0.1,               # <- the β temperature parameter
    logging_steps=50,
    save_steps=500,
    fp16=True,
    report_to="none",
    max_length=512,
    max_prompt_length=256,
)

trainer = DPOTrainer(
    model=model,
    ref_model=None,    # None = auto-creates frozen copy of model
    args=training_args,
    train_dataset=dataset,
    tokenizer=tokenizer,
)

trainer.train()
trainer.save_model("./dpo_aligned_model")
print("DPO model saved.")
```

- [ ] **Step 4: Run DPO training**

```bash
python rlhf/train_dpo.py
```

Expected: training loss starts around 0.69 (log 2), decreases smoothly. DPO training should be noticeably more stable (less variance in loss) than PPO.

- [ ] **Step 5: Compare DPO vs PPO outputs using LLM-as-Judge**

Create `rlhf/compare_dpo_ppo.py`:

```python
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
```

Run and qualitatively compare. Then (optional, requires GPT-4o API):

```python
# Use GPT-4o to judge which response is better
from openai import OpenAI
client = OpenAI()

def judge(prompt, response_a, response_b, name_a, name_b):
    judge_prompt = f"""Which response is better for this prompt?

Prompt: {prompt}

Response A ({name_a}): {response_a}

Response B ({name_b}): {response_b}

Reply with only "A" or "B" and one sentence of reasoning."""
    
    result = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": judge_prompt}]
    )
    return result.choices[0].message.content
```

- [ ] **Step 6: Add milestone note and commit**

Add to `DPO.md`:
```markdown
## Week 4 Milestone
"Can explain the closed-form trick behind DPO: by rearranging the optimal RLHF policy
equation, the reward disappears and the loss becomes BCE on log-ratios. DPO trains stably
with 2 models instead of 4, but cannot do online learning."
```

```bash
git add "Lukas's Notes/LLM/DPO.md" rlhf/train_dpo.py rlhf/compare_dpo_ppo.py
git commit -m "add DPO training script, comparison, Week 4 notes"
```

---

## Task 5: Week 5 — GRPO Theory + Setup (6–7 hrs)

**Goal:** Understand GRPO's mechanism and set up the training environment.

**Files:**
- Create: `Lukas's Notes/LLM/GRPO.md`
- Create: `rlhf/train_grpo.py`

- [ ] **Step 1: Read DeepSeekMath paper (Shao et al. 2024) — Section 3 only (~2 hrs)**

Download: https://arxiv.org/abs/2402.03300

Read Section 3: "Reinforcement Learning from Code and Math Feedback". Key idea: GRPO samples G completions per question, uses group-relative normalization instead of a value model.

Create `GRPO.md`:
```markdown
# GRPO (Group Relative Policy Optimization)

## Why GRPO Exists
PPO and DPO both assign a single scalar reward to a complete response. For reasoning tasks
(math, code), the correctness reward is:
- **Sparse**: only 0 or 1 (correct / incorrect)
- **Delayed**: the model doesn't know until the end of the response

This makes the value function (PPO's critic) hard to train accurately. GRPO removes it.

## The Mechanism
For each prompt x:
1. Sample G responses: {y_1, y_2, ..., y_G} (e.g., G = 8)
2. Compute reward for each: {r_1, r_2, ..., r_G}
3. Normalize within the group:
   `advantage_i = (r_i - mean(r)) / std(r)`
4. Use these group-relative advantages as the PPO advantage signal
5. No value function / critic needed

## GRPO Objective
Same as PPO's clipped objective, but with group-relative advantage:
`L_GRPO = E[ min(r_t · Â_group, clip(r_t, 1-ε, 1+ε) · Â_group) ]`

Where `Â_group = (r_i - mean({r_j})) / std({r_j})`

## Why This Works
- Responses in the same group share the same prompt context
- The group mean is a natural baseline (similar to a value function estimate)
- High variance responses (some correct, some not) provide the strongest learning signal

## Reward Function Design
For math reasoning:
- **Accuracy reward**: +1 if final answer matches ground truth, 0 otherwise
- **Format reward**: small +0.1 if the response contains <think>...</think> structure
  (encourages chain-of-thought reasoning)

## GRPO vs PPO vs DPO
| | PPO | DPO | GRPO |
|-|-----|-----|------|
| Value model needed | Yes | No | No |
| Reward model needed | Yes | No | No (rule-based) |
| Online learning | Yes | No | Yes |
| Best for | General RLHF | Preference alignment | Reasoning tasks |
| Memory cost | Highest (4 models) | Low (2 models) | Medium (2 models) |
```

- [ ] **Step 2: Read DeepSeek-R1 paper — Section 2 (~1.5 hrs)**

Download: https://arxiv.org/abs/2501.12948

Read Section 2: "DeepSeek-R1-Zero". Key insight: using GRPO with only accuracy and format rewards (no human preference data) to develop long chain-of-thought reasoning.

Add to `GRPO.md`:
```markdown
## Connection to DeepSeek-R1
DeepSeek-R1-Zero used GRPO with only two rewards:
1. Accuracy: is the final answer correct? (verified against the math dataset)
2. Format: does the response use <think>...</think> tags?

No human annotations. No reward model. Just rule-based verification.

The model spontaneously developed:
- Self-reflection ("wait, let me reconsider...")
- Multi-step reasoning chains
- Error correction mid-response

This shows that GRPO + sparse rewards can teach *process-level* reasoning.
```

- [ ] **Step 3: Write the GRPO training script**

Create `rlhf/train_grpo.py`:

```python
import re
import torch
from datasets import load_dataset
from transformers import AutoTokenizer
from trl import GRPOTrainer, GRPOConfig
from trl.models import AutoModelForCausalLMWithValueHead

model_name = "Qwen/Qwen2-1.5B-Instruct"  # larger model for reasoning

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# Load GSM8K — grade-school math problems
dataset = load_dataset("openai/gsm8k", "main", split="train[:2000]")

def format_prompt(row):
    return {
        "prompt": f"Solve this math problem step by step.\n\nQuestion: {row['question']}\n\nAnswer:",
        "answer": row["answer"].split("####")[-1].strip()  # extract final numeric answer
    }

dataset = dataset.map(format_prompt)

def extract_answer(text: str) -> str:
    """Extract the last number from model output."""
    numbers = re.findall(r"\d+(?:\.\d+)?", text.replace(",", ""))
    return numbers[-1] if numbers else ""

def accuracy_reward(completions, prompts, answer, **kwargs) -> list[float]:
    """Reward = 1.0 if model's final number matches ground truth, else 0.0."""
    rewards = []
    for completion, gt in zip(completions, answer):
        pred = extract_answer(completion[0]["content"] if isinstance(completion[0], dict) 
                              else completion)
        rewards.append(1.0 if pred == gt else 0.0)
    return rewards

def format_reward(completions, **kwargs) -> list[float]:
    """Small reward for using chain-of-thought structure."""
    rewards = []
    for completion in completions:
        text = completion[0]["content"] if isinstance(completion[0], dict) else completion
        has_steps = any(word in text.lower() for word in ["step", "first", "then", "therefore"])
        rewards.append(0.1 if has_steps else 0.0)
    return rewards

def combined_reward(completions, prompts, answer, **kwargs) -> list[float]:
    acc = accuracy_reward(completions, prompts, answer, **kwargs)
    fmt = format_reward(completions, **kwargs)
    return [a + f for a, f in zip(acc, fmt)]

config = GRPOConfig(
    output_dir="./grpo_model_output",
    per_device_train_batch_size=2,
    num_train_epochs=1,
    learning_rate=5e-6,
    num_generations=8,        # G = 8 responses per prompt
    max_new_tokens=512,
    logging_steps=10,
    save_steps=200,
    fp16=True,
    report_to="none",
    kl_coef=0.04,
)

trainer = GRPOTrainer(
    model=model_name,
    reward_funcs=combined_reward,
    args=config,
    train_dataset=dataset,
    tokenizer=tokenizer,
)

trainer.train()
trainer.save_model("./grpo_aligned_model")
print("GRPO model saved.")
```

- [ ] **Step 4: Run GRPO training**

```bash
python rlhf/train_grpo.py
```

Expected: accuracy reward starts near 0.0–0.1 (random guessing), should improve to 0.2–0.4 within first epoch on GSM8K. Training will be slower than DPO because it generates G=8 completions per step.

- [ ] **Step 5: Commit**

```bash
git add "Lukas's Notes/LLM/GRPO.md" rlhf/train_grpo.py
git commit -m "add GRPO training script and Week 5 theory notes"
```

---

## Task 6: Week 6 — GRPO Evaluation + Decision Framework (6–7 hrs)

**Goal:** Evaluate GRPO, compare all three algorithms, write your personal decision framework.

**Files:**
- Create: `rlhf/evaluate_grpo.py`
- Modify: `Lukas's Notes/LLM/GRPO.md`
- Create: `Lukas's Notes/LLM/AlgorithmDecisionFramework.md`

- [ ] **Step 1: Evaluate GRPO accuracy on GSM8K test set**

Create `rlhf/evaluate_grpo.py`:

```python
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
```

```bash
python rlhf/evaluate_grpo.py
```

Expected: base model ~10–20% accuracy, GRPO model ~25–45% (significant improvement with only 1 epoch).

- [ ] **Step 2: Write the personal decision framework**

Create `AlgorithmDecisionFramework.md`:

```markdown
# My Algorithm Decision Framework: PPO vs DPO vs GRPO

## Quick Decision Guide

| My task looks like... | Use | Because |
|-----------------------|-----|---------|
| Instruction following, chat, customer support | **DPO** | Preference pairs available, stable, no RM needed, offline |
| I want the model to explore and improve over time | **PPO** | Online learning, reward model can improve iteratively |
| Math reasoning, code generation, multi-step problems | **GRPO** | Group-relative comparison works with sparse correctness signals |
| I have thumbs-up/thumbs-down data (not pairs) | **KTO** (DPO variant) | Doesn't need paired comparisons |
| I need to avoid a reference model entirely | **SimPO** | Reference-free DPO variant |

## Key Trade-offs I Learned

### PPO
**When it's the right call:**
- Online RLHF where the model generates new data during training
- Complex rewards that require a learned reward model (not just rule-based)
- Research settings where you want full control over the training loop

**Hidden costs I now understand:**
- 4 models in GPU memory (policy, reference, reward model, value model)
- Reward hacking is real — always monitor KL divergence and qualitative outputs
- Much harder to tune than DPO (many hyperparameters)

### DPO
**When it's the right call:**
- You have a good preference dataset and want stable, cheap training
- Offline setting — no need to generate new data
- Time/compute constraints

**Limitations I now understand:**
- Cannot do online learning (model doesn't explore)
- Sensitive to data quality — bad preference pairs mislead the loss
- β needs tuning — too low = ignores reference, too high = barely trains

### GRPO
**When it's the right call:**
- Reasoning tasks where you can verify correctness programmatically (math, code)
- No human preference data available
- Want to develop chain-of-thought behavior

**Key insight I internalized:**
- The group-relative advantage is an elegant substitute for the value function
- Works because responses within a group share the same prompt context
- G=8 is a good default; too small → high variance, too large → expensive

## Red Flags I Watch For

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| PPO reward skyrockets, quality drops | Reward hacking (KL too low) | Increase init_kl_coef or β |
| DPO loss goes negative | β too high, model diverging | Reduce learning rate or β |
| GRPO all rewards = 0 in early steps | Reward function never triggered | Check reward fn, lower max_new_tokens |
| Model collapses to repetitive output | Overtraining or mode collapse | Reduce epochs, add temperature |
```

- [ ] **Step 3: Final week milestone note**

Add to `GRPO.md`:
```markdown
## Week 6 Milestone
"Can train a reasoning model with GRPO, explain group-relative advantage estimation,
evaluate accuracy on GSM8K, and compare all three algorithms. Personal decision
framework written and validated through hands-on experience with all three methods."
```

- [ ] **Step 4: Final commit**

```bash
git add "Lukas's Notes/LLM/GRPO.md" \
        "Lukas's Notes/LLM/AlgorithmDecisionFramework.md" \
        rlhf/evaluate_grpo.py
git commit -m "add GRPO evaluation, personal decision framework — Week 6 complete"
```

---

## End-to-End Verification Checklist

| Week | Checkpoint | How to verify |
|------|-----------|---------------|
| 1 | RL fundamentals understood | Can explain PPO clipping in own words |
| 1 | CleanRL runs | CartPole reward reaches 400+ |
| 2 | Reward model trained | Chosen > rejected score on >60% of test examples |
| 3 | PPO alignment runs | Reward increases; KL stays < target |
| 3 | Reward hacking observed | No-KL run shows inflated reward, degraded quality |
| 4 | DPO training runs | Loss decreases smoothly from ~0.69 |
| 4 | PPO vs DPO compared | Qualitative comparison completed |
| 5 | GRPO script runs | Accuracy reward improves over base |
| 6 | GSM8K accuracy measured | GRPO > base model by at least 10 pp |
| 6 | Decision framework written | Framework covers all 3 algorithms with personal reasoning |
