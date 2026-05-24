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
- Skim: Hugging Face RLHF blog post — shorter, more practical

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
