# Phase 3 Cheat Sheet: GRPO (Group Relative Policy Optimization)

## GRPO in 30 Seconds

GRPO replaces PPO's expensive value model (critic) with a simple trick: generate multiple
responses per prompt, score them all, then **grade on a curve**. The group mean becomes the
baseline. High-scoring responses get reinforced; low-scoring ones get suppressed. No critic
needed -- the group **is** the baseline.

---

## The Problem GRPO Solves

PPO estimates advantages as `A = R - V(s)`, where `V(s)` comes from a learned value model.

| Pain Point | Detail |
|---|---|
| Memory cost | Value model is the same size as the policy -- doubles GPU memory |
| Training instability | Poorly trained critic produces noisy advantages, destabilizing policy updates |
| Extra complexity | Separate training loop, learning rate, loss function for the critic |

**Core question:** Can we estimate advantages WITHOUT a learned critic?

GRPO's answer: **Yes -- use the group statistics instead.**

---

## GRPO Algorithm: "Grade on a Curve"

```
                          Prompt x
                             |
              +--------------+--------------+
              |              |              |
           y_1            y_2    ...     y_G        <-- 1. Sample G responses
              |              |              |
           r_1            r_2    ...     r_G        <-- 2. Score with reward model
              |              |              |
              +-------> mean(r), std(r) <---+       <-- 3. Compute group statistics
              |              |              |
           A_1            A_2    ...     A_G        <-- 4. Normalize: A_i = (r_i - mean) / std
              |              |              |
              +--------> Clipped PPO Loss <-+       <-- 5. Update policy (clipped objective)
                             |
                        + KL(pi || pi_ref)          <-- 6. KL penalty against reference model
```

**Step-by-step:**

1. **Generate** G completions `{y_1, ..., y_G}` from the current policy for prompt `x`
2. **Score** each completion: `{r_1, ..., r_G}` via reward model (or rule-based check)
3. **Normalize** within group: `A_i = (r_i - mean(r)) / std(r)`
4. **Update** policy using PPO-style clipped surrogate objective with these advantages
5. **Regularize** with KL divergence penalty against the reference model

**Key insight:** The group mean replaces the value model. Zero extra parameters.

---

## 3-Way Comparison

| Aspect | PPO | DPO | GRPO |
|---|---|---|---|
| Models at train time | 4 (Policy, Ref, RM, Critic) | 2 (Policy, Ref) | 3 (Policy, Ref, RM) |
| Advantage estimation | Learned value model | N/A (implicit in preference loss) | Group normalization |
| Online / Offline | Online | Offline | Online |
| Needs reward model | Yes | No (preferences baked into loss) | Yes (or rule-based) |
| Generation per step | 1 response/prompt | None (uses static dataset) | G responses/prompt |
| Memory footprint | Highest | Lowest | Medium |
| Reward signal | Scalar from RM | Pairwise preference | Scalar from RM or rules |

---

## When GRPO Shines: Verifiable Rewards

GRPO is strongest when you can **verify correctness programmatically** -- no learned RM needed.

| Domain | Verification Method | Why It Works |
|---|---|---|
| Math reasoning | Check final answer against ground truth | Binary correct/incorrect, no ambiguity |
| Code generation | Execute code, run test suite | Pass/fail is deterministic |
| Format compliance | Regex or template matching | Structural rules are easy to check |
| Length constraints | Token count thresholds | Simple scalar penalty |
| Instruction following | Rule-based checklist | Each constraint is verifiable |

**Landmark:** DeepSeek-R1 was trained with GRPO + rule-based rewards (no learned RM),
demonstrating that verifiable rewards can scale to frontier-level reasoning.

---

## Key Hyperparameters

| Parameter | Typical Range | Notes |
|---|---|---|
| `num_generations` (G) | 4 -- 16 | Larger = better advantage estimates but linear cost increase |
| `kl_coef` | 0.01 -- 0.1 | Too high = slow learning; too low = reward hacking |
| `clip_range` | 0.2 | Same as PPO; limits policy update magnitude |
| `temperature` | 0.7 -- 1.0 | Must be high enough for diversity within each group |

---

## What to Watch During Training

| Signal | What It Means | Action |
|---|---|---|
| Group reward variance near 0 | All G responses score the same, so all advantages are ~0 | Raise temperature or increase G |
| KL divergence spiking | Policy drifting too far from reference | Increase `kl_coef` |
| Reward mean plateaus early | Policy stuck in local optimum | Check reward function granularity |
| Cost scaling with G | Each step generates G times more tokens | Balance G vs. batch size for your GPU budget |
| Rule-based vs. learned RM | Rule-based is more stable, less prone to reward hacking | Prefer rules when the task is verifiable |

---

## The 3-Way Decision Guide

```
Start Here: "I want to align / fine-tune my LLM"
  |
  |-- Do I have a reward model or verifiable reward function?
  |     |
  |     |-- YES, and rewards are verifiable (math, code, format)
  |     |     --> Use GRPO with rule-based rewards (cheapest, most stable)
  |     |
  |     |-- YES, learned RM only
  |     |     |
  |     |     |-- Can I afford 4 models in memory?
  |     |     |     --> YES: PPO (gold standard, most flexible)
  |     |     |     --> NO:  GRPO (drops the critic, 3 models)
  |     |
  |     |-- NO reward model
  |           |
  |           |-- Do I have human preference pairs?
  |                 --> YES: DPO (simplest, offline, 2 models)
  |                 --> NO:  Collect preferences first, then DPO
```

---

## Quick Reference: GRPO vs PPO vs DPO -- When to Use What

- **Choose GRPO** when you have verifiable rewards (math, code) or want online RL without the
  memory cost of a critic. Best for tasks where "correct vs. incorrect" is clear.
- **Choose PPO** when you need maximum control, have a strong learned reward model, and can
  afford 4 models in memory. Gold standard for general RLHF.
- **Choose DPO** when you have a good preference dataset and want the simplest training setup.
  No generation at training time, no reward model, just supervised-style optimization.

---

*Prerequisites: Phase 1 (PPO) and Phase 2 (DPO). Next: Phase 4 -- Reward Modeling Deep Dive.*
