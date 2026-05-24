# DPO (Direct Preference Optimization)

## The Core Insight

The optimal policy under KL-constrained RLHF has a closed form:

`π*(y|x) ∝ π_ref(y|x) · exp(r(x,y)/β)`

This means the reward can be expressed as a function of the policy itself:

`r(x,y) = β · log(π(y|x)/π_ref(y|x)) + const`

So we can substitute this into the Bradley-Terry preference loss and get a loss directly on the policy — no reward model needed.

## DPO Objective

`L = -E[ log σ( β · (log π(y_w|x)/π_ref(y_w|x)) - β · (log π(y_l|x)/π_ref(y_l|x)) ) ]`

Where:
- `y_w` = chosen (winning) response
- `y_l` = rejected (losing) response
- `β` = temperature controlling deviation from reference policy (typically 0.1–0.5)
- `π_ref` = frozen SFT model (reference)

This is just binary cross-entropy on the log-ratio difference.

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
- **IPO**: replaces σ with identity to avoid overfitting to hard preferences
- **KTO**: uses thumbs-up/thumbs-down data instead of pairs
- **SimPO**: removes reference model dependency entirely

### Key Resources
- DPO paper (Rafailov et al. 2023) — full read
- Sebastian Raschka — "DPO explained"

---

## Week 4 Milestone

> "Can explain the closed-form trick behind DPO: by rearranging the optimal RLHF policy equation, the reward disappears and the loss becomes BCE on log-ratios. DPO trains stably with 2 models instead of 4, but cannot do online learning."
