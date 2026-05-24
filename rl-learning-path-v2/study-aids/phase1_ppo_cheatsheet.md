# Phase 1 Cheat Sheet: PPO for LLM Alignment

```
+------------------------------------------------------------------+
|                       PPO IN 30 SECONDS                          |
|                                                                  |
|  PPO trains an LLM to produce better responses by trial and      |
|  error, but with a speed limit: each update can change the       |
|  policy by at most ~20%. This prevents catastrophic collapse      |
|  while still learning from reward signals. For LLMs, we run      |
|  4 models in a loop: generate -> score -> compute advantage      |
|  -> update (clipped). A KL penalty keeps the model close to      |
|  its original SFT behavior so it doesn't game the reward.        |
+------------------------------------------------------------------+
```

## The Problem PPO Solves

Vanilla policy gradient computes "make good actions more likely," but gives
**no control over update size**:

- Too large --> policy collapses (forgets language, outputs garbage)
- Too small --> no learning (wastes compute)

PPO adds a **clipped objective** that caps each update, making training stable.

## PPO Clipped Objective

```
L_CLIP = E[ min( r(theta) * A,  clip(r(theta), 1-e, 1+e) * A ) ]
```

| Symbol | Meaning |
|--------|---------|
| `r(theta)` | Probability ratio `pi_new(a) / pi_old(a)` -- how much the policy changed |
| `A` | Advantage: how much better this action was than average (`R - V(s)`) |
| `e` | Clip range, typically **0.2** (max 20% change per step) |
| `min(...)` | Takes the **pessimistic** (lower) estimate, blocking overly optimistic updates |

**Intuition:** "If an action is good, make it more likely -- but not more than 20% per step.
If it is bad, make it less likely -- but again, not more than 20% per step."

## 4-Model Setup for LLMs

```
  +-------------------+       +-------------------+
  |  Policy Model     |       |  Reference Model  |
  |  (pi_theta)       |       |  (pi_ref)         |
  |  Being trained    |       |  Frozen SFT copy  |
  |                   |       |  For KL penalty   |
  +--------+----------+       +---------+---------+
           |                            |
           |    generates responses     |  provides baseline probs
           v                            v
  +-------------------+       +-------------------+
  |  Reward Model     |       |  Value Model      |
  |  (r_phi)          |       |  (V_psi)          |
  |  Scores responses |       |  Critic: estimates|
  |  (from human      |       |  expected reward  |
  |   preferences)    |       |  A = R - V(s)     |
  +-------------------+       +-------------------+

  Memory cost: ~56 GB for four 7B models in fp16
  --> This is a major motivation for DPO and GRPO (fewer models needed)
```

## One PPO Iteration

```
  [Prompt Batch]
       |
       v
  1. GENERATE -----> Policy produces responses
       |
       v
  2. SCORE --------> Reward model rates each response
       |
       v
  3. ADVANTAGES ---> Value model estimates baseline; A = R - V(s)
       |
       v
  4. UPDATE -------> Clipped objective updates policy weights
       |
       v
  5. KL PENALTY ---> R_adj = R - beta * KL(pi_theta || pi_ref)
       |
       v
  [Next Iteration]
```

## Hyperparameter Tuning Guide

| Parameter | Typical Range | Too Low | Too High |
|-----------|--------------|---------|----------|
| `learning_rate` | 1e-5 to 5e-6 | Reward flat, slow learning | Policy collapse |
| `kl_penalty (beta)` | 0.01 -- 0.2 | Reward hacking, drifts from SFT | Barely learns, too conservative |
| `clip_range (e)` | 0.2 | Updates too timid | Large swings, instability |
| `batch_size` | 64 -- 256 | Noisy gradients | Slow iteration, high memory |
| `ppo_epochs` | 2 -- 4 | Underfitting per batch | Overfitting per batch |

## What Can Go Wrong

| Signal | Diagnosis | Fix |
|--------|-----------|-----|
| Reward up, output quality down | **Reward hacking** -- model exploits reward model quirks | Increase `beta`, improve reward model, add human eval |
| Reward flat | KL penalty too high or LR too low | Lower `beta` or raise `learning_rate` |
| KL divergence spikes | Policy drifting too fast from reference | Raise `beta`, lower `learning_rate` |
| Outputs collapse to repetition | Update steps too aggressive | Lower `learning_rate`, reduce `ppo_epochs` |
| Training loss oscillates wildly | Batch size too small or LR too high | Increase `batch_size`, lower `learning_rate` |

## Healthy Training Checklist

- Reward mean: increasing gradually
- KL divergence: stable or slowly rising (not spiking)
- Output diversity: maintained (not collapsing to one pattern)
- Reward variance: decreasing over time

---
*Prereq: Phase 0 (RLHF pipeline, basic RL concepts, reward models) | Next: Phase 2*
