# GRPO (Group Relative Policy Optimization)

## Why GRPO Exists

PPO and DPO both assign a single scalar reward to a complete response. For reasoning tasks (math, code), the correctness reward is:
- **Sparse**: only 0 or 1 (correct / incorrect)
- **Delayed**: the model doesn't know until the end of the response

This makes the value function (PPO's critic) hard to train accurately. GRPO removes it entirely.

---

## The Mechanism

For each prompt x:
1. Sample G responses: `{y_1, y_2, ..., y_G}` (e.g., G = 8)
2. Compute reward for each: `{r_1, r_2, ..., r_G}`
3. Normalize within the group: `advantage_i = (r_i - mean(r)) / std(r)`
4. Use these group-relative advantages as the PPO advantage signal
5. No value function / critic needed

## GRPO Objective

Same as PPO's clipped objective, but with group-relative advantage:

`L_GRPO = E[ min(r_t * A_group, clip(r_t, 1-e, 1+e) * A_group) ]`

Where `A_group = (r_i - mean({r_j})) / std({r_j})`

## Why This Works
- Responses in the same group share the same prompt context
- The group mean is a natural baseline (similar to a value function estimate)
- High variance responses (some correct, some not) provide the strongest learning signal
- G=8 is a good default; too small -> high variance, too large -> expensive

---

## Reward Function Design

For math reasoning:
- **Accuracy reward**: +1 if final answer matches ground truth, 0 otherwise
- **Format reward**: small +0.1 if the response contains structured chain-of-thought (e.g., "Step", "First", "Therefore")

---

## Connection to DeepSeek-R1

DeepSeek-R1-Zero used GRPO with only two rewards:
1. **Accuracy**: is the final answer correct? (verified against the math dataset)
2. **Format**: does the response use `<think>...</think>` tags?

No human annotations. No reward model. Just rule-based verification.

The model spontaneously developed:
- Self-reflection ("wait, let me reconsider...")
- Multi-step reasoning chains
- Error correction mid-response

This shows that GRPO + sparse rewards can teach *process-level* reasoning.

---

## GRPO vs PPO vs DPO

| | PPO | DPO | GRPO |
|-|-----|-----|------|
| Value model needed | Yes | No | No |
| Reward model needed | Yes | No | No (rule-based) |
| Online learning | Yes | No | Yes |
| Best for | General RLHF | Preference alignment | Reasoning tasks |
| Memory cost | Highest (4 models) | Low (2 models) | Medium (2 models) |

### Key Resources
- DeepSeekMath paper (Shao et al. 2024) -- Section 3
- DeepSeek-R1 paper (DeepSeek-AI 2025) -- Section 2
- TRL GRPO documentation

---

## Week 6 Milestone

> "Can train a reasoning model with GRPO, explain group-relative advantage estimation, evaluate accuracy on GSM8K, and compare all three algorithms. Personal decision framework written and validated through hands-on experience with all three methods."
