# PPO / DPO / GRPO Minimum Learning Path — Spec

**Date**: 2026-04-21
**Target audience**: NLP practitioner with solid LLM theory, basic ML knowledge, no RL background
**Goal**: Theory + hands-on mastery of PPO, DPO, and GRPO for alignment tasks at work
**Time budget**: 6–10 hours/week · 6 weeks total (~45 hours)
**Compute**: Cloud GPU (A100/H100)

> **Replaces**: `../../2026-04-21-rlhf-learning-path-design-v1-archived.md` (the previous 10-week broad RLHF plan)

---

## Design Principle: Just-in-Time RL

RL fundamentals are introduced only as needed — not as a standalone RL course. The three algorithms follow a natural **problem → solution** arc:

| Algorithm | Problem it solves | Key insight |
|-----------|------------------|-------------|
| PPO | How to optimize LLMs from human feedback? | Clip the policy ratio so updates can't be catastrophically large |
| DPO | PPO requires a reward model + 4 models in GPU memory — too expensive | The optimal policy has a closed form → reparametrize reward, skip the RM entirely |
| GRPO | PPO/DPO fail at reasoning (sparse rewards, no process feedback) | Normalize rewards within a group of sampled responses → no value function needed |

---

## Module 1: RL Essentials (Week 1, 6–8 hrs)

**Goal**: Build just enough RL intuition to understand PPO when applied to LLMs.

### Theory (4–5 hrs)

1. **MDP fundamentals** (1 hr): state, action, reward, policy, return, discount factor
   - Resource: OpenAI Spinning Up — "Key Concepts in RL" only
2. **Policy gradient / REINFORCE** (1.5 hrs): log-probability trick, why we can differentiate through sampling
   - Resource: Lilian Weng — "Policy Gradient Algorithms"
3. **PPO** (1.5 hrs): trust region motivation → clipping → PPO objective → GAE advantage estimation
   - Resource: PPO paper (Schulman et al. 2017) — Section 3 only

### Hands-on (2–3 hrs)

- Read CleanRL's PPO implementation (~300 lines) — don't implement, just understand the structure
- Run it on CartPole, observe clipping behavior in training logs

### Milestone

*"PPO prevents catastrophic policy updates by clipping the ratio between new and old policy probabilities. The advantage function tells it which actions to reinforce more strongly."*

---

## Module 2: PPO for LLMs / RLHF (Weeks 2–3, 12–14 hrs)

**Goal**: Understand the complete RLHF pipeline and run it end-to-end on a small model.

### 2A: Reward Model Training (Week 2, 6–7 hrs)

**Theory (3–4 hrs)**:
1. LLM–RL mapping: prompt = state, token = action, LLM = policy, reward model = reward function
2. Reward model: Bradley-Terry preference model, pairwise comparison loss, LLM + scalar head architecture
3. KL constraint: why necessary, what reward hacking looks like when removed

Resources:
- InstructGPT paper (Ouyang et al. 2022) — Sections 3.2–3.4
- "Secrets of RLHF in Large Language Models" (Zheng et al. 2023)

**Hands-on (3–4 hrs)**:
- Train a reward model on Anthropic HH-RLHF dataset using TRL `RewardTrainer` (Qwen-0.5B backbone)
- Verify: do chosen responses consistently score higher than rejected ones?

### 2B: PPO Training Loop (Week 3, 6–7 hrs)

**Theory (2–3 hrs)**:
- PPO-RLHF loop: generate → score → PPO update → repeat
- KL budget as a control knob; reward overoptimization

**Hands-on (4–5 hrs)**:
- Run PPO alignment using TRL `PPOTrainer` on the same Qwen-0.5B model
- **Key experiment**: disable KL penalty, observe reward hacking (score inflates, output quality degrades)
- Compare aligned vs. unaligned outputs

### Milestone

*"Has run the complete RLHF pipeline end-to-end (reward model → PPO alignment) and observed reward hacking first-hand."*

---

## Module 3: DPO (Week 4, 8–10 hrs)

**Goal**: Understand why DPO emerged, how it works, and when to choose it over PPO.

### Theory (4–5 hrs)

1. **The closed-form insight**: the optimal policy under KL-constrained RLHF satisfies `π*(y|x) ∝ π_ref(y|x) · exp(r(x,y)/β)` — so you can express the reward as a function of the policy itself
2. **DPO objective**: preference data → BCE loss on log-ratios of (policy / reference) — no reward model, no RL loop
3. **DPO vs. PPO trade-offs**: DPO is stable and memory-efficient; PPO supports online learning and distributional exploration
4. **Variant awareness** (10 min each): IPO, KTO, SimPO

Resources:
- DPO paper (Rafailov et al. 2023) — full read
- Sebastian Raschka — "DPO explained"

### Hands-on (4–5 hrs)

- Fine-tune the same Qwen model using TRL `DPOTrainer` on the same HH-RLHF preference data
- Compare PPO vs. DPO: training curves, output quality, GPU-hours
- Evaluate with LLM-as-Judge (GPT-4o or similar) side-by-side *(requires API access)*

### Milestone

*"Can explain the closed-form trick behind DPO and knows when to prefer DPO over PPO."*

---

## Module 4: GRPO (Weeks 5–6, 12–14 hrs)

**Goal**: Understand GRPO's mechanism, train a reasoning model, and build a decision framework across all three algorithms.

### Theory (5–6 hrs)

1. **Why PPO/DPO struggle with reasoning**: both assign a single scalar to a complete output — breaks down when correctness is binary and rare (math, code)
2. **GRPO mechanism**:
   - Sample G responses per prompt (e.g., G = 8)
   - Compute reward for each response
   - Normalize within the group: `advantage_i = (r_i − mean(r)) / std(r)`
   - No value function / critic — removes the largest memory cost vs. PPO
3. **Reward function design**: accuracy reward (is the answer correct?) + format reward (is chain-of-thought structured?)
4. **Connection to DeepSeek-R1 / DeepSeekMath**: how GRPO taught LLMs to reason through RL on process

Resources:
- DeepSeekMath paper (Shao et al. 2024) — Section 3
- DeepSeek-R1 paper (DeepSeek-AI 2025) — Section 2
- TRL GRPO documentation

### Hands-on (7–8 hrs)

- Train GRPO on Qwen-1.5B or 3B using TRL `GRPOTrainer` on GSM8K subset
- Define accuracy reward + format reward functions
- Compare GRPO vs. DPO on reasoning task quality
- **Week 6 wrap-up**: write your personal decision framework (see below)

### Milestone

*"Can train a reasoning model with GRPO, explain group-relative advantage estimation, and justify choosing PPO/DPO/GRPO for a given task to a teammate."*

---

## Timeline Summary

| Module | Weeks | Hours | Focus |
|--------|-------|-------|-------|
| 1: RL Essentials | 1 | 6–8 | Policy gradient, PPO intuition |
| 2: PPO for LLMs | 2–3 | 12–14 | Reward model, RLHF loop, reward hacking |
| 3: DPO | 4 | 8–10 | Closed-form trick, PPO vs. DPO comparison |
| 4: GRPO | 5–6 | 12–14 | Group-relative advantage, reasoning tasks |
| **Total** | **6 weeks** | **~38–46 hrs** | **6–10 hrs/week** |

---

## Decision Framework (Week 6 deliverable)

| Task type | Algorithm | Reason |
|-----------|-----------|--------|
| Instruction following, chat, general alignment | DPO | No reward model needed, stable training, preference data is sufficient |
| Online learning, complex or learned rewards | PPO | Explores the distribution, reward model can improve iteratively |
| Math reasoning, code, multi-step tasks | GRPO | Group-relative rewards handle sparse correctness signals; cheaper than PPO (no value model) |

---

## Key Resources

### Papers
- Schulman et al. (2017) — "Proximal Policy Optimization Algorithms" (§3)
- Ouyang et al. (2022) — InstructGPT (§3.2–3.4)
- Zheng et al. (2023) — "Secrets of RLHF in Large Language Models"
- Rafailov et al. (2023) — "Direct Preference Optimization" (full)
- Shao et al. (2024) — "DeepSeekMath" (§3)
- DeepSeek-AI (2025) — "DeepSeek-R1" (§2)

### Blogs / Tutorials
- OpenAI Spinning Up — "Key Concepts in RL"
- Lilian Weng — "Policy Gradient Algorithms"
- Sebastian Raschka — "DPO explained"
- Hugging Face TRL documentation

### Tools
- `trl`: `RewardTrainer`, `PPOTrainer`, `DPOTrainer`, `GRPOTrainer`
- `transformers` + `peft` (LoRA for memory efficiency)
- `datasets`: HH-RLHF, GSM8K
- CleanRL: minimal PPO reference implementation
- Models: Qwen-0.5B → 1.5B → 3B (progressive scale)

---

## Success Criteria

By end of Week 6:
1. Can explain PPO's clipping objective and why it prevents training instability
2. Has run the full RLHF pipeline and observed reward hacking experimentally
3. Can derive (at concept level) why DPO works — the closed-form optimal policy trick
4. Can choose between PPO and DPO for a given task with justified reasoning
5. Has trained a reasoning model with GRPO and can explain group-relative advantage
6. Can read new alignment papers (ORPO, SimPO, online DPO) independently

---

## End-to-End Verification

| Week | Checkpoint |
|------|-----------|
| 1 | CleanRL PPO runs on CartPole; reward improves without diverging |
| 3 | Reward model scores chosen > rejected; PPO converges; reward hacking demo replicates |
| 4 | DPO training curve is smoother than PPO; LLM-as-Judge shows comparable quality at lower compute |
| 6 | GRPO improves GSM8K accuracy over base model; personal decision framework is written |
