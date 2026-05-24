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
