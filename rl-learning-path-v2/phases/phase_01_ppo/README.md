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
