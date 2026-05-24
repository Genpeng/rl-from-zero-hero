# Phase 0 -- Foundation Cheat Sheet

> For NLP practitioners with SFT experience and zero RL background.

---

## Day 1: Why SFT Is Not Enough

**The "helpful but harmful" problem** -- SFT trains a model to imitate demonstrations.
It cannot distinguish a *good* answer from a *merely plausible* one.

| Limitation of SFT          | What goes wrong                                    |
|-----------------------------|----------------------------------------------------|
| No preference signal        | Model learns *what to say*, not *what to prefer*   |
| Distribution mismatch       | Train on gold text, deploy on own generations      |
| Harmful fluency             | Toxic/wrong answers can be perfectly fluent         |

**Core insight:** we need to show the model what humans *prefer*, not just what to *say*.

### The RLHF Pipeline

```
                        preference
  Human demos            pairs            reward signal
      |                   |                     |
      v                   v                     v
  +-------+    +----------------+    +----------------+
  |  SFT  | -> | Reward Model   | -> |  RL fine-tune  |
  | (warm  |    | (learns human  |    |  (PPO adjusts  |
  |  start)|    |  preferences)  |    |   policy)      |
  +-------+    +----------------+    +----------------+
```

Think of it as three courses in a meal:
1. **SFT** = teach the model English grammar and style (imitation)
2. **Reward Model** = teach a judge what "good" looks like (preferences)
3. **RL (PPO)** = let the model practice and improve under that judge

---

## Day 2: RL Concepts Mapped to LLMs

| RL Term              | LLM Equivalent                          | Analogy                              |
|----------------------|-----------------------------------------|--------------------------------------|
| Policy pi(a\|s)      | The language model itself (weights)     | The student writing an essay         |
| State s              | Tokens generated so far + prompt        | What's on the page already           |
| Action a             | Generating the next token               | Writing the next word                |
| Trajectory / Episode | Generating one complete response        | Finishing an entire essay            |
| Reward r             | Score from the reward model             | Teacher's grade on the essay         |
| Return R             | Total reward for the full response      | Final mark                           |

### The Policy Gradient (one equation to remember)

```
grad J(theta) = E[ grad log pi(a|s) * R ]
```

Plain English: **"nudge the weights so that high-reward responses become more likely."**

- `log pi(a|s)` = log-probability the model assigned to the action it took
- `R` = how good the full response was
- If R is high, increase that log-prob. If R is low, decrease it.

### KL Divergence -- the "leash"

KL divergence measures how far the RL-tuned policy has drifted from the original SFT model.

```
R_adjusted = R_reward - beta * KL( pi || pi_ref )
```

**Analogy:** KL penalty is a rubber band tying the student to their textbook.
Explore freely, but snap back if you wander too far from what SFT taught you.

- `beta` high -> short leash, conservative updates, less reward hacking
- `beta` low  -> long leash, more freedom, risk of reward hacking

---

## Day 3: Reward Modelling Basics

### Preference Data Format

Each training example is a triplet:

```
(prompt, chosen_response, rejected_response)
```

The model never sees an absolute score -- only *which response is better*.

### Bradley-Terry Model

Converts pairwise preferences into a probability:

```
P(y_chosen > y_rejected) = sigmoid( r(x, y_chosen) - r(x, y_rejected) )
```

Only the **gap** between scores matters, not the absolute values.

### Training Loss

```
L = -log sigmoid( r(x, y_chosen) - r(x, y_rejected) )
```

This is binary cross-entropy on the score gap.
Minimizing L pushes `r(chosen)` above `r(rejected)`.

### Why Reward Quality Is the Bottleneck

```
  Noisy RM           RL amplifies
  preferences  --->  wrong signal  --->  broken policy
```

RL is an amplifier: a good reward model makes the policy better;
a bad reward model makes it confidently worse ("reward hacking").

---

## Common Misconceptions

| Misconception | Reality |
|---|---|
| "RLHF replaces SFT" | SFT is the starting point; RL refines it. Skip SFT and RL has nothing to work with. |
| "The reward model outputs THE correct score" | It outputs *relative* preferences. Only the gap between two scores is trained. |
| "More RL training = better model" | Without KL penalty, the model over-optimizes the RM and quality degrades. |
| "One token = one reward" | Reward is given once for the *complete* response, then credit is spread back. |
| "Policy gradient is a new optimizer" | It is a *loss formulation*. You still use Adam underneath. |
| "KL penalty hurts performance" | It prevents reward hacking -- the model gaming the RM instead of being helpful. |
