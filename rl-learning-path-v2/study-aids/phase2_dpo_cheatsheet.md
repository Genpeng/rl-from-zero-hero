# Phase 2 Cheat Sheet: Direct Preference Optimization (DPO)

## DPO in 30 Seconds

You survived PPO -- the 4-model circus, the hyperparameter minefield, the reward hacking.
DPO asks: **what if we skip the reward model entirely?** It bakes the reward into the policy
itself, turning alignment into a single supervised-learning step on preference pairs. Done.

---

## The Derivation in 4 Steps

```
Step 1                        Step 2
RLHF Objective                Closed-Form Optimal Policy
max E[r(x,y)]                 pi*(y|x) = pi_ref(y|x) * exp(r(x,y)/beta)
 - beta * KL(pi || pi_ref)                     / Z(x)
        |                              |
        v                              v
Step 3                        Step 4
Solve for Reward              Plug into Bradley-Terry
r(x,y) = beta * log           P(y_w > y_l) = sigma(r_w - r_l)
  pi(y|x)/pi_ref(y|x)         Substitute r from Step 3...
  + beta * log Z(x)            Reward model disappears!
                                Z(x) cancels out!
```

**Step 1.** Standard RLHF: maximize reward minus a KL penalty to stay near the reference.
**Step 2.** This KL-constrained problem has an analytical solution -- the optimal policy.
**Step 3.** Rearrange Step 2 to express reward as a function of the policy ratio.
**Step 4.** Substitute into Bradley-Terry preference model. The partition function Z cancels,
and the explicit reward model is gone. Preferences supervise the policy directly.

---

## The DPO Loss

```
L_DPO = -E[ log sigma( beta * ( log pi(y_w|x)/pi_ref(y_w|x)
                               - log pi(y_l|x)/pi_ref(y_l|x) ) ) ]
```

**Plain English:** For each preference pair, compute how much the policy prefers the
chosen response over the reference (log-ratio for y_w), do the same for the rejected
response (log-ratio for y_l), then push those two apart. The sigmoid + log makes this
a soft classification loss -- "chosen should win by a margin."

**beta** controls aggressiveness -- same role as the KL coefficient in PPO.

---

## PPO Pipeline vs DPO Pipeline

```
PPO (4 models)                        DPO (2 models)
==============================        ==============================
Prompt --> [Policy] --> Response       Prompt + (y_w, y_l) pair
              |                               |
       [Reward Model] --> Score        [Policy]     [Ref Policy]
              |                          |               |
       [Critic/Value] --> Advantage    log pi(y|x)   log pi_ref(y|x)
              |                               |
       [Ref Policy] --> KL penalty     Compute log-ratios, apply loss
              |                               |
       PPO Clipped Update              Gradient step
```

## PPO vs DPO Comparison

| Aspect            | PPO                                  | DPO                         |
|--------------------|--------------------------------------|------------------------------|
| Models at train    | 4 (policy, ref, reward, critic)      | 2 (policy + frozen ref)      |
| Training data      | Reward model + prompt pool           | Preference pairs directly    |
| Training loop      | Generate -> Score -> Advantage -> Update | Forward -> Loss -> Update |
| Hyperparameters    | Many (KL, clip, GAE lambda, epochs)  | Few (beta, learning rate)    |
| Online / Offline   | Online (generates new responses)     | Offline (fixed dataset)      |
| GPU memory         | ~4x model size                       | ~2x model size               |
| Reward hacking     | Real risk                            | Lower (no explicit reward)   |

---

## DPO Advantages

- **Simpler**: one loss function, no RL machinery, no rollouts
- **2x less GPU**: only two models in memory
- **More stable**: no moving reward target, no value function bootstrap
- **Fewer hyperparams**: beta and LR are nearly all you tune

## DPO Limitations

- **Data quality dependency**: garbage pairs in, garbage policy out (no online correction)
- **Offline only**: cannot explore beyond the dataset
- **Verbosity bias**: tends to prefer longer responses if they dominate "chosen" set
- **beta sensitivity**: still requires tuning (see below)
- **Distribution shift**: policy drifts from data distribution over training

---

## beta Tuning Guide

| beta  | Behavior                              | Symptom                          |
|-------|---------------------------------------|----------------------------------|
| 0.01  | Too aggressive                        | Overfits, memorizes preferences  |
| 0.1   | Default / balanced                    | Good starting point              |
| 0.5   | Too conservative                      | Barely changes from SFT          |

Rule of thumb: start at 0.1, lower if the model doesn't differentiate chosen/rejected,
raise if training loss drops too fast or eval quality degrades.

---

## Decision Flowchart: DPO or PPO?

```
START: Do you have clean, high-quality preference pairs?
  |
  +-- YES --> Is compute limited?
  |             |
  |             +-- YES --> Use DPO
  |             +-- NO  --> Do you need online exploration / self-play?
  |                           |
  |                           +-- NO  --> Use DPO (simpler wins)
  |                           +-- YES --> Use PPO
  |
  +-- NO --> Can you build a reliable reward model?
               |
               +-- YES --> Use PPO (online data compensates for noise)
               +-- NO  --> Collect better data, then use DPO
```

---

## Quick Reference: When to Use Which

| Scenario                                  | Pick  |
|-------------------------------------------|-------|
| Clean preference data, limited compute    | DPO   |
| Want fast iteration and simplicity        | DPO   |
| Noisy/sparse labels, need exploration     | PPO   |
| Large compute budget, want best ceiling   | PPO   |
| First alignment attempt on a new model    | DPO   |

---

## Key Takeaway

DPO is not "PPO but worse" -- it is the **same objective**, solved analytically instead of
via RL. You trade online exploration for simplicity. When your data is solid, that trade wins.

> Next up -- **Phase 3: Variants and Extensions** (IPO, KTO, ORPO, online DPO)
