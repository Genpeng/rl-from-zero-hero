# Phase 2: DPO (Direct Preference Optimization)

**Days 6-7 | ~15-20 min read**

In Phase 1, you built the full PPO pipeline: a policy model, a reference model, a reward
model, and a value model — four models working together to align an LLM with human
preferences. It works, but it is complex and expensive.

DPO asks a provocative question: **what if we could skip the reward model entirely?**

It turns out that the optimal policy under the reward-maximization-with-KL-constraint
objective has a closed-form solution. Instead of training a reward model and then running
RL against it, we can directly optimize the policy using preference data. Same data, same
goal, half the infrastructure.

---

## 1. The Key Insight: Skipping the Reward Model

Recall the PPO setup from Phase 1. At training time, you need all four models loaded
simultaneously:

```
PPO pipeline (4 models in memory):
  1. Policy model     — the model being trained
  2. Reference model  — frozen copy for KL constraint
  3. Reward model     — scores generated responses
  4. Value model      — estimates expected future reward
```

For a 7B parameter model in half-precision, each copy takes roughly 14 GB of GPU memory.
Four models means ~56 GB just for weights — before accounting for activations, gradients,
and optimizer states. This makes PPO impractical for many teams.

Rafailov et al. (2023) noticed something elegant. The standard RLHF objective is:

```
maximize  E[reward(x, y)]  -  β · KL(π_θ || π_ref)
```

This says: generate responses that score well with the reward model, but don't drift too
far from the reference model. The key mathematical insight is that this optimization
problem has a **closed-form solution** — you can write down exactly what the optimal
policy looks like:

```
π*(y|x) = (1/Z(x)) · π_ref(y|x) · exp(reward(x, y) / β)
```

If you rearrange this formula, you can express the reward entirely in terms of the policy
and reference model:

```
reward(x, y) = β · log(π_θ(y|x) / π_ref(y|x)) + β · log Z(x)
```

This means: **the reward is implicitly encoded in the ratio of the policy's probabilities
to the reference model's probabilities.** You do not need a separate reward model to
extract it — the policy *is* the reward model.

DPO exploits this by substituting this implicit reward directly into the preference
learning objective. The result is a training procedure that:

- Takes preference pairs (chosen vs. rejected) as input
- Uses only the policy model and a frozen reference model (2 models)
- Optimizes a simple classification-like loss
- Produces the same optimal policy as RLHF with PPO (in theory)

```
DPO pipeline (2 models in memory):
  1. Policy model     — the model being trained
  2. Reference model  — frozen copy for KL constraint
  (No reward model, no value model)
```

---

## 2. DPO Loss Function

The DPO loss is surprisingly compact. Here it is in full:

```
L_DPO = -E[ log sigma( beta * ( log pi_theta(y_w|x) / pi_ref(y_w|x)
                               - log pi_theta(y_l|x) / pi_ref(y_l|x) ) ) ]
```

Or in more mathematical notation:

```
L_DPO = -E[ log σ( β · ( log π_θ(y_w|x)/π_ref(y_w|x) - log π_θ(y_l|x)/π_ref(y_l|x) ) ) ]
```

Where:
- **x** is the prompt
- **y_w** is the preferred (chosen/winning) response
- **y_l** is the dispreferred (rejected/losing) response
- **π_θ** is the policy model (being trained)
- **π_ref** is the reference model (frozen)
- **β** is the temperature parameter controlling KL strength
- **σ** is the sigmoid function

Let's unpack what each piece means intuitively.

### The log-probability ratios

```
log π_θ(y_w|x) / π_ref(y_w|x)
```

This measures how much *more* likely the policy model makes the preferred response
compared to the reference model. If this is positive, the policy has learned to favor
the preferred response. If it is negative, the policy still considers this response less
likely than the reference does.

```
log π_θ(y_l|x) / π_ref(y_l|x)
```

Same thing for the rejected response. If this is positive, the policy has (unfortunately)
also increased the likelihood of the rejected response.

### The difference

```
log π_θ(y_w|x)/π_ref(y_w|x) - log π_θ(y_l|x)/π_ref(y_l|x)
```

This is the **margin**: how much more the policy has boosted the preferred response
relative to the rejected response. A larger margin means the policy has done a better job
of separating good from bad.

### The sigmoid

```
σ( β · margin )
```

The sigmoid squashes the margin into a probability between 0 and 1. When the margin is
large and positive, σ approaches 1. When the margin is near zero or negative, σ
approaches 0.

### Negative log (the loss)

```
-log σ( β · margin )
```

Taking the negative log makes this a loss to minimize. When the margin is large (policy
correctly separates chosen from rejected), the loss is small. When the margin is small or
negative (policy fails to distinguish them), the loss is large.

### The intuition in one sentence

**Make the preferred response more likely and the rejected response less likely, relative
to what the reference model would do, and penalize the model when it fails to separate
the two.**

If you have ever trained a binary classifier, DPO should feel familiar. It is essentially
a binary cross-entropy loss where the "correct class" is "prefer y_w over y_l." The
twist is that the log-probability ratios (relative to the reference model) act as the
logits, and β controls how aggressively the model separates the two.

---

## 3. The Beta Parameter

Beta (β) plays the same role in DPO as the KL coefficient in PPO: it controls how much
the policy is allowed to deviate from the reference model. But the mechanism is different.

In PPO, β appears as an explicit penalty term in the reward:

```
reward_total = reward_RM - β · KL(π_θ || π_ref)
```

In DPO, β appears inside the loss function as a scaling factor on the margin:

```
L = -log σ( β · margin )
```

The effect is analogous:

### Low β (e.g., 0.01)

- The margin is multiplied by a small number, so the sigmoid saturates slowly.
- The model needs a very large margin to drive the loss toward zero.
- In practice, this means the model is allowed to deviate significantly from the
  reference — it aggressively pushes preferred outputs up and rejected outputs down.
- Risk: the model may overfit to the preference data or produce outputs that are very
  different from the pretrained model's style.

### High β (e.g., 0.5 or higher)

- The margin is amplified, so even a small margin produces a sigmoid close to 1.
- The model does not need to change much to satisfy the loss.
- In practice, the policy stays very close to the reference model — conservative updates.
- Risk: the model may not learn enough from the preference data. The alignment signal is
  too weak.

### The sweet spot (typically 0.1 to 0.2)

Most DPO implementations use β = 0.1 as the default. This provides a reasonable balance:
the model learns to distinguish preferred from rejected responses without drifting too far
from the reference.

```
β value   | Behavior
----------|----------------------------------------------------------
0.01      | Very aggressive — large policy changes, risk of overfitting
0.05      | Moderately aggressive — noticeable deviation from reference
0.1       | Standard default — balanced optimization
0.2       | Conservative — small policy changes
0.5+      | Very conservative — almost no change from reference
```

In `03_dpo_experiment.py`, you will run DPO with different β values and observe the
tradeoff directly — similar to the KL ablation you did in Phase 1.

---

## 4. PPO vs DPO Comparison

Now that you understand both algorithms, here is a side-by-side comparison:

| Aspect | PPO | DPO |
|---|---|---|
| **Models in memory** | 4 (policy, reference, reward, value) | 2 (policy, reference) |
| **Training data** | Prompts only (generates responses on-the-fly) | Preference pairs (prompt + chosen + rejected) |
| **Training loop** | Complex: generate → score → advantage → update | Simple: forward pass on preference pairs |
| **Stability** | Sensitive to hyperparameters (KL coef, clip range, GAE params) | Generally more stable (fewer moving parts) |
| **Memory usage** | Very high (4 models + generation buffers) | Moderate (2 models + preference batch) |
| **Compute** | High (generation is expensive) | Lower (no generation during training) |
| **Reward hacking** | Can occur (policy exploits reward model) | Less common (no separate reward model to exploit) |
| **Data requirements** | Needs a trained reward model + prompts | Needs preference pairs directly |
| **Online vs offline** | Online (generates fresh data each step) | Offline (trains on fixed preference dataset) |
| **Flexibility** | Can optimize any reward signal | Limited to pairwise preferences |
| **Iterative improvement** | Natural: update policy → generate → score → repeat | Harder: need to re-collect preferences for updated policy |

### When each is stronger

**PPO is stronger when:**
- You have a robust reward model and want to do online/iterative optimization
- Your reward signal is not easily expressed as pairwise preferences (e.g., a rule-based
  reward function, a verifier, or a composite score)
- You need the model to explore and discover responses not present in your training data
- You are training at very large scale with dedicated infrastructure

**DPO is stronger when:**
- You have a fixed dataset of human preferences
- You want simpler training with fewer hyperparameters to tune
- GPU memory is limited (you can only fit 2 models)
- You want faster iteration speed (no generation during training)
- Stability matters more than squeezing out maximum performance

---

## 5. When to Choose DPO

Here is a practical decision framework:

**Choose DPO when:**
1. You have preference data but no trained reward model (and don't want to train one)
2. You are working with limited GPU resources
3. You want a quick alignment pass and plan to iterate on the data, not the algorithm
4. Your preference data is high quality and representative

**Choose PPO when:**
1. You have a reliable reward model or reward function (e.g., code execution, math
   verification)
2. You want the model to explore novel responses beyond what is in your training data
3. You are doing iterative RLHF where you collect new preferences after each round
4. You need to combine multiple reward signals

**A common pattern in practice:**
Many teams start with DPO for initial alignment (it is fast and easy to set up), then
switch to PPO or other online methods for refinement when they have a good reward model
and want to push further. The two approaches are complementary, not competing.

---

## 6. Hands-on Plan

### Script 1: `01_dpo_training.py` (Day 6)

**What it does:** Trains a model using DPO on Anthropic HH-RLHF preference data. Uses
TRL's DPOTrainer for the heavy lifting and LoRA for memory efficiency.

**Run:**
```bash
python 01_dpo_training.py --max_samples 2000 --beta 0.1 --num_epochs 1
```

**What to observe:**
- Training loss should decrease over steps
- The loss value represents how well the model separates chosen from rejected responses
- Sample outputs at the end: compare them to the base model
- Try running with `--beta 0.01` and `--beta 0.5` to see the effect

### Script 2: `02_compare_ppo_dpo.py` (Day 6-7)

**What it does:** Loads your PPO model (from Phase 1) and DPO model side by side, then
generates responses from both on the same prompts. This is where you see the practical
differences between the two alignment approaches.

**Run:**
```bash
python 02_compare_ppo_dpo.py \
    --ppo_model_path ./outputs/ppo_model \
    --dpo_model_path ./outputs/dpo_model
```

**What to observe:**
- Are PPO and DPO outputs similar or different in style?
- Does one produce longer or shorter responses?
- Which feels more "aligned" — helpful and harmless?
- Remember: the PPO model was trained with a reward model you trained, while the DPO
  model learned directly from preference data

### Script 3: `03_dpo_experiment.py` (Day 7)

**What it does:** Runs DPO training with three different β values (0.01, 0.1, 0.5) and
compares the resulting models. This is the DPO equivalent of the KL ablation you did in
Phase 1.

**Run:**
```bash
python 03_dpo_experiment.py --max_samples 1000
```

**What to observe:**
- β = 0.01: aggressive optimization — does the model overfit or produce unusual outputs?
- β = 0.1: the standard default — should be a reasonable balance
- β = 0.5: very conservative — outputs should be nearly identical to the base model
- Compare training loss across the three runs: lower β typically produces lower loss
  (the model is allowed to separate preferences more aggressively)

---

## Check Your Understanding

Test yourself before moving to Phase 3 (GRPO). Click each question to reveal the answer.

<details>
<summary><strong>1. DPO eliminates the reward model. Where did the reward signal go?</strong></summary>

The reward is implicitly represented in the log-probability ratio between the policy and
reference model. The DPO paper shows that the optimal policy under the
reward-maximization-with-KL-constraint objective can be rearranged so the reward equals
`β * log(π_θ(y|x) / π_ref(y|x))` plus a constant. So the "reward" is baked into how
much the policy's probabilities differ from the reference model's probabilities. There is
no separate reward model because the policy *itself* encodes the reward function.

</details>

<details>
<summary><strong>2. Why does DPO need a reference model if there is no KL penalty term?</strong></summary>

The KL constraint is still there — it is just folded into the loss function. The
log-probability ratios `log π_θ(y|x) / π_ref(y|x)` are measuring how far the policy has
moved from the reference. If you removed the reference model (equivalent to setting all
reference log-probs to zero), the loss would just maximize the probability of chosen
responses and minimize rejected responses without any anchor, leading to overfitting and
degenerate outputs. The reference model provides the same grounding effect as the KL
penalty in PPO.

</details>

<details>
<summary><strong>3. If you increase β, what happens to the training loss and why?</strong></summary>

The training loss typically becomes easier to minimize (lower values, faster convergence)
because the sigmoid saturates more quickly — the model needs only a small margin to
produce a confident prediction. However, the actual policy changes very little because
even small margins are sufficient. Conversely, low β makes the loss harder to minimize
(the model must achieve large margins), but allows larger policy updates. You will
observe this directly in `03_dpo_experiment.py`.

</details>

<details>
<summary><strong>4. PPO is online (generates fresh data); DPO is offline (uses a fixed dataset). Why does this matter?</strong></summary>

In online RL, the model generates its own data, which means it can explore and discover
responses not present in the training set. This is powerful but expensive. In offline DPO,
the model can only learn from the preference pairs you provide. If those pairs do not
cover important scenarios, the model has no way to learn about them. This is why DPO
sometimes underperforms PPO on tasks requiring creative or novel responses — DPO is
limited to the distribution of the training data. A practical consequence: the quality
of your preference dataset matters much more for DPO than for PPO.

</details>

<details>
<summary><strong>5. You trained a DPO model but the outputs are nearly identical to the base model. What went wrong and what would you try?</strong></summary>

This usually means β is too high (the model is too constrained to change). Try reducing β
from 0.1 to 0.05 or 0.01. Other possibilities: the learning rate is too low, the number
of training steps is too few, or the preference data does not have a strong enough signal
(chosen and rejected responses are too similar). Check the training loss — if it barely
decreased, the model did not learn. If it decreased but outputs are still similar, the
β-induced constraint is probably the issue.

</details>

---

## References

- **Rafailov et al. (2023)** — *Direct Preference Optimization: Your Language Model is
  Secretly a Reward Model*
  [arXiv:2305.18290](https://arxiv.org/abs/2305.18290). The original DPO paper. Section 4
  derives the loss function from the RLHF objective; Section 5 has the experiments.

- **TRL DPOTrainer Documentation** — [https://huggingface.co/docs/trl/dpo_trainer](https://huggingface.co/docs/trl/dpo_trainer).
  API reference for the trainer used in the scripts.

- **Tunstall et al. (2023)** — *Zephyr: Direct Distillation of LM Alignment and
  Back-Translation*
  [arXiv:2310.16944](https://arxiv.org/abs/2310.16944). Demonstrates DPO at scale on the
  Zephyr model — a practical case study of DPO in production.
