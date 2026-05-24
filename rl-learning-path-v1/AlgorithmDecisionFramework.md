# My Algorithm Decision Framework: PPO vs DPO vs GRPO

## Quick Decision Guide

| My task looks like... | Use | Because |
|-----------------------|-----|---------|
| Instruction following, chat, customer support | **DPO** | Preference pairs available, stable, no RM needed, offline |
| I want the model to explore and improve over time | **PPO** | Online learning, reward model can improve iteratively |
| Math reasoning, code generation, multi-step problems | **GRPO** | Group-relative comparison works with sparse correctness signals |
| I have thumbs-up/thumbs-down data (not pairs) | **KTO** (DPO variant) | Doesn't need paired comparisons |
| I need to avoid a reference model entirely | **SimPO** | Reference-free DPO variant |

---

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

---

## Red Flags I Watch For

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| PPO reward skyrockets, quality drops | Reward hacking (KL too low) | Increase `init_kl_coef` or β |
| DPO loss goes negative | β too high, model diverging | Reduce learning rate or β |
| GRPO all rewards = 0 in early steps | Reward function never triggered | Check reward fn, lower `max_new_tokens` |
| Model collapses to repetitive output | Overtraining or mode collapse | Reduce epochs, add temperature |
