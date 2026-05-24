# Phase 2: DPO — Removing the Reward Model

**Duration:** 3-4 days (~11 hours: 5 theory + 6 hands-on)
**Goal:** Understand how DPO simplifies RLHF by eliminating the reward model, and compare directly with PPO.

---

## Theory: The DPO Insight (Day 9-10)

### The Problem with PPO-Based RLHF

By now you've experienced PPO's pain points firsthand:

- **4 models in memory** — Policy, Reference, Reward, Value (critic)
- **Hyperparameter sensitivity** — KL coefficient, clip range, learning rate all interact
- **Reward hacking** — the policy exploits reward model weaknesses
- **Engineering complexity** — generation loop, reward scoring, advantage computation, PPO update

What if we could get the same result with just a simple loss function on preference data?

### The DPO Breakthrough

DPO (Direct Preference Optimization) makes a mathematical observation:

1. The RLHF objective is: maximize reward while staying close to the reference policy
2. This optimization problem has a **closed-form solution**: `π*(y|x) ∝ π_ref(y|x) · exp(r(x,y) / β)`
3. We can rearrange this to express the reward in terms of the policy:
   `r(x,y) = β · log(π(y|x) / π_ref(y|x)) + const`
4. Plug this into the Bradley-Terry preference model → the reward model disappears

**Result:** A loss function that trains the policy directly on preference pairs.

### The DPO Loss

```
L_DPO = -E[ log σ( β · ( log π(y_w|x)/π_ref(y_w|x) - log π(y_l|x)/π_ref(y_l|x) ) ) ]
```

In plain English:
- For each preference pair (prompt, chosen, rejected):
- Compute how much MORE likely the chosen response is vs the reference (log ratio for chosen)
- Compute how much MORE likely the rejected response is vs the reference (log ratio for rejected)
- The loss pushes these two ratios APART — make chosen more likely, rejected less likely
- β controls how aggressively to push (same role as KL coefficient in PPO)

### The Simplification

| Aspect | PPO | DPO |
|--------|-----|-----|
| Models in memory | 4 (Policy, Ref, RM, Value) | 2 (Policy, Ref) |
| Training data | Reward model + prompts | Preference pairs directly |
| Training loop | Generate → Score → Compute advantages → Update | Forward pass → Compute loss → Update |
| Hyperparameters | KL coef, clip range, GAE λ, PPO epochs... | β, learning rate |
| Online/Offline | Online (generates new responses) | Offline (fixed dataset) |

### Trade-offs

**DPO advantages:**
- Much simpler to implement and debug
- 2x less GPU memory
- More stable training
- Fewer hyperparameters to tune

**DPO limitations:**
- **Data quality dependency**: No reward model to generalize — if your preference pairs are noisy, DPO amplifies that noise
- **Offline only**: Optimizes against a fixed dataset. Can't discover new good responses through exploration
- **Verbosity bias**: Tends to learn surface-level patterns in preferred responses (e.g., "longer is better")
- **β sensitivity**: Too low → overfits to preference data (memorization); Too high → barely changes from SFT

### When to Use Which

- **Use DPO when:** You have high-quality preference data, limited compute, want simplicity
- **Use PPO when:** Data is noisy, you need online exploration, you have large compute budget

### 📖 Reading

- **DPO paper** (Rafailov et al., 2023): Sections 1-4
  - Paper: "Direct Preference Optimization: Your Language Model is Secretly a Reward Model"
- **"Is DPO Superior to PPO?"** — Comparison analyses

---

## Hands-on Lab (Day 11-12)

See the Python scripts in this directory:

1. `01_dpo_training.py` — Run DPO training on the same model/task as PPO
2. `02_dpo_analysis.py` — Direct comparison: SFT vs PPO vs DPO

### What to Watch For

- **Training loss decreasing** → model is learning preferences ✓
- **Chosen rewards > Rejected rewards** → correct preference direction ✓
- **Rewards margin increasing** → model is separating good from bad more confidently ✓
- **Loss drops to near zero quickly** → possible overfitting (try higher β or more data) ⚠️

---

## ✅ Self-Check: Am I Ready for Phase 3?

1. "What does DPO eliminate compared to PPO, and how?"
   - Expected: Eliminates the reward model by deriving the optimal policy directly from preference data

2. "When would you choose DPO over PPO at work?"
   - Expected: When you have clean preference data, limited GPU budget, and want a simpler pipeline

3. "What does β control in DPO?"
   - Expected: How aggressively the model deviates from the reference — analogous to KL coefficient in PPO
