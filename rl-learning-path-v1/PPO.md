# PPO (Proximal Policy Optimization)

## RL Fundamentals
- **State (s)**: the environment's current situation (in LLMs: prompt + generated tokens so far)
- **Action (a)**: what the agent does (in LLMs: next token to generate)
- **Reward (r)**: signal measuring how good the action was
- **Policy (π)**: the function mapping states to actions (in LLMs: the LLM itself)
- **Return (G)**: sum of future discounted rewards from a given state
- **Discount factor (γ)**: how much we value future rewards vs immediate ones (0 < γ ≤ 1)

### Key Resources
- OpenAI Spinning Up — "Key Concepts in RL"

---

## Policy Gradient / REINFORCE
- **Core idea**: we can't differentiate through the sampling step, but we can use the log-prob trick
- **REINFORCE objective**: maximize `E[R(τ) · log π(a|s)]`, where τ is a trajectory
- **The log-probability trick**: `∇E[R] = E[R · ∇log π(a|s)]` — allows gradient-based optimization even through discrete sampling
- **Problem**: high variance — reward signal is noisy and updates can be catastrophically large

### Key Resources
- Lilian Weng — "Policy Gradient Algorithms"

---

## PPO
- **Problem with vanilla policy gradient**: unconstrained updates can collapse the policy — one bad batch can ruin training
- **Trust region**: don't let the new policy stray too far from the old one
- **Policy ratio**: `r_t(θ) = π_θ(a|s) / π_θ_old(a|s)` — how much the new policy differs from the old
- **Clipping trick**: clip the ratio `r_t` to `[1-ε, 1+ε]` — equivalent to a trust region but simpler to implement
- **Clipped objective**: `L_CLIP = E[min(r_t · Â_t, clip(r_t, 1-ε, 1+ε) · Â_t)]`
- **Advantage (Â)**: tells PPO whether an action was better (+) or worse (-) than expected
- **GAE (Generalized Advantage Estimation)**: smoothed advantage using a λ parameter — balances bias vs variance
- **ε = 0.2** is the standard clip range

### Key Resources
- PPO paper (Schulman et al. 2017) — Section 3

---

## Hands-on: CleanRL PPO on CartPole

Things to look for in CleanRL's PPO implementation (~300 lines):
1. Where is the ratio `r_t` computed? (look for `logratio`)
2. Where is clipping applied? (look for `torch.clamp`)
3. Where is the advantage computed? (look for `advantages`)

Run command:
```bash
pip install cleanrl
python cleanrl/cleanrl/ppo.py --env-id CartPole-v1 --total-timesteps 50000
```

Expected: episodic reward starts ~20-50, climbs to 300-500.

---

## Week 1 Milestone

> "PPO prevents catastrophic policy updates by clipping the ratio between new and old policy probabilities to [1-ε, 1+ε]. The advantage function tells it which actions to reinforce more strongly (positive advantage) or discourage (negative advantage)."

---

## RLHF — Reward Model
- **Why?** We can't write a loss function for "be helpful and harmless" — humans must signal preferences
- **Bradley-Terry loss**: for chosen y_w and rejected y_l: `loss = -log σ(r(y_w) - r(y_l))`
- **Architecture**: LLM backbone + linear head outputting a single scalar (the reward)
- **Dataset format**: `(prompt, chosen_response, rejected_response)` triples
- **HH-RLHF**: Anthropic's public preference dataset — harmlessness + helpfulness annotations
- **Training**: ~5000 samples, 1 epoch, expect loss to drop from ~0.69 (random) to ~0.55–0.60

### Key Resources
- InstructGPT paper (Ouyang et al. 2022) — Sections 3.2–3.4
- "Secrets of RLHF in Large Language Models" (Zheng et al. 2023)

---

## PPO for LLMs — Training Loop
1. **Sample**: use the current policy (LLM) to generate responses for a batch of prompts
2. **Score**: pass each (prompt, response) through the reward model → scalar reward
3. **KL penalty**: subtract `β · KL(π_new || π_ref)` from reward — penalizes straying from SFT model
4. **PPO update**: run PPO update steps using the adjusted reward and clipped objective
5. **Repeat**: iterate until reward plateaus or KL budget is exhausted

### 4 Models in Memory During PPO
- **π_new**: the model being trained (policy)
- **π_ref**: the frozen SFT model (reference, for KL)
- **Reward model**: scores responses
- **Value model (critic)**: estimates expected future reward (often initialized from π_ref)

### Reward Hacking
Without KL penalty, the model learns to produce outputs the RM scores highly that are actually low quality — e.g., repetitive text, sycophancy. Experiment: `train_ppo_no_kl.py` demonstrates this.

## Week 3 Milestone

> "Has run the complete RLHF pipeline end-to-end (reward model → PPO alignment) and observed reward hacking first-hand: without KL constraint, the model games the reward model rather than becoming genuinely more helpful."
