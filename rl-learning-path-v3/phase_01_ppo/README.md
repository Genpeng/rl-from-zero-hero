# Phase 1: PPO (Proximal Policy Optimization)

**Days 3-5 | ~15-20 min read**

PPO is the algorithm that powered the original ChatGPT. It takes a pretrained language model
and fine-tunes it using human feedback — turning a text-completion engine into an assistant
that follows instructions, avoids harmful outputs, and generally behaves the way users want.

If you completed Phase 0, you already know *why* we need alignment and you have an SFT-warmed
model. Now we dive into *how* PPO actually does the alignment.

---

## 1. The PPO Setup in RLHF: Four Models Working Together

PPO-based RLHF requires **four separate models** in memory at training time. This is
why PPO training is expensive — and why alternatives like DPO (Phase 2) were invented.

Here is what each model does:

### Policy Model (the student)

This is the model we are training. It starts as your SFT model and gets updated at every
PPO step. Its job is to generate responses to prompts. We are trying to make it generate
*better* responses, where "better" is defined by the reward model.

Think of it as a student writing essays. Each PPO step is like getting feedback on an essay
and revising your writing approach.

### Reference Model (the anchor)

This is a **frozen copy** of the policy model from before PPO training started. It never
gets updated. Its only purpose is to measure how far the policy has drifted from its
starting point.

Why do we need this? Because without an anchor, the policy model can drift into bizarre
territory — producing outputs that score high with the reward model but are actually
gibberish or exploit loopholes. The reference model keeps the policy "grounded."

### Reward Model (the teacher)

This is a separate model trained on human preference data. Given a prompt and a response,
it outputs a single number: how good this response is.

The reward model was trained in a prior step (you will do this in `01_train_reward_model.py`).
It learned from thousands of examples where humans said "response A is better than response B."

Think of it as a teacher grading essays. It does not write essays itself — it just assigns
scores.

### Value Model (the estimator)

This model predicts **how much reward the policy will get** from a given state (partial
response). It is used to compute the "advantage" — whether an action turned out better or
worse than expected.

If the reward model is the teacher grading the final essay, the value model is the student's
own sense of "I think this essay is going pretty well" or "this is not going well" while
still writing it.

In practice, the value model is often initialized from the reward model or the policy
model, and it gets updated during PPO training.

```
┌─────────────────────────────────────────────────────────┐
│                  PPO Training Setup                     │
│                                                         │
│  ┌──────────────┐         ┌──────────────┐             │
│  │ Policy Model │ ──────> │   Generate   │             │
│  │  (updated)   │         │   Response   │             │
│  └──────────────┘         └──────┬───────┘             │
│         │                        │                      │
│         │ compare        ┌───────▼───────┐             │
│  ┌──────▼──────────┐     │ Reward Model  │             │
│  │ Reference Model │     │   (frozen)    │             │
│  │    (frozen)     │     │  scores the   │             │
│  │  "how far did   │     │   response    │             │
│  │   we drift?"    │     └───────┬───────┘             │
│  └─────────────────┘             │                      │
│                          ┌───────▼───────┐             │
│                          │  Value Model  │             │
│                          │  (updated)    │             │
│                          │  estimates    │             │
│                          │  advantage    │             │
│                          └───────────────┘             │
└─────────────────────────────────────────────────────────┘
```

---

## 2. The Clipped Surrogate Objective: "Take Small, Safe Steps"

PPO's core innovation is a simple idea: **don't change the policy too much in one step.**

In older policy gradient methods (like TRPO), enforcing this "small step" constraint
required complex second-order optimization. PPO replaces all of that with a clever clipping
trick.

### The Formula

```
L_CLIP = E[ min( r(θ) · A,  clip(r(θ), 1-ε, 1+ε) · A ) ]
```

Let's break down each piece:

**r(θ) — the probability ratio**

```
r(θ) = π_θ(a|s) / π_θ_old(a|s)
```

This asks: "how much more (or less) likely is this action under the new policy compared
to the old policy?" If r(θ) = 1.0, the new and old policy agree perfectly. If r(θ) = 1.5,
the new policy is 50% more likely to take this action.

**A — the advantage**

The advantage tells us whether this action was better or worse than expected. Positive
advantage means "this action was surprisingly good." Negative advantage means "this action
was worse than expected."

**clip(r(θ), 1-ε, 1+ε) — the safety guard**

Epsilon (ε) is typically 0.2. This clips the ratio to the range [0.8, 1.2], preventing
any single update from changing the policy too drastically.

**min(...) — the pessimistic choice**

We take the *minimum* of the clipped and unclipped objectives. This creates a "floor" —
the optimization can never exploit a single example by making a huge update.

### Intuition: Why Does This Work?

Imagine you are tuning a recipe. You tried adding more salt, and people liked it (positive
advantage). PPO says: "Great, add a bit more salt next time — but not five times as much.
Cap your change."

Now imagine you tried adding more salt and people hated it (negative advantage). PPO says:
"Reduce the salt, but again, don't swing wildly in the other direction."

The clipping ensures that regardless of how extreme the advantage signal is, the actual
policy change stays moderate. This prevents the "one bad batch ruins everything" problem
that plagues other RL methods.

### Typical Value of ε

The standard value is ε = 0.2, meaning the ratio can range from 0.8 to 1.2. Some
implementations use ε = 0.1 for more conservative updates or ε = 0.3 for faster but
riskier learning.

---

## 3. KL Penalty: "Don't Forget What You Learned in Pretraining"

Even with clipping, there is a risk: the policy can slowly drift away from the pretrained
model over many small steps. Each step is safe, but the *accumulated* drift can be large.

This is where the **KL divergence penalty** comes in.

### What Is KL Divergence?

KL divergence measures how different two probability distributions are. In our case, it
measures how different the policy model's outputs have become from the reference model's
outputs.

```
KL(π_θ || π_ref) = E[ log(π_θ(a|s)) - log(π_ref(a|s)) ]
```

When KL = 0, the models produce identical distributions. As KL grows, the models diverge.

### The Modified Reward

PPO-RLHF adds a KL penalty directly to the reward:

```
reward_total = reward_from_RM - β · KL(π_θ || π_ref)
```

Where β (beta) is the KL penalty coefficient. Higher β means "stay closer to the original
model." Lower β means "focus more on maximizing the reward."

### Why We Need This: Reward Hacking

Without the KL penalty, the policy can find **degenerate shortcuts** to maximize the reward
model's score. This is called **reward hacking**.

**Concrete example:** Suppose the reward model was trained on data where longer, more
detailed responses were preferred. Without a KL penalty, the policy might learn to produce
extremely long, repetitive outputs — repeating the same helpful-sounding phrases over and
over — because the reward model gives high scores to long text. The response looks like:

```
"That's a great question! I'd be happy to help. Let me explain in detail.
 That's a great question! I'd be happy to help. Let me explain in detail.
 That's a great question! I'd be happy to help. Let me explain in detail. ..."
```

The reward model gives this a high score (it sees "helpful" phrases), but the output is
useless. The KL penalty prevents this by penalizing the policy whenever it produces text
that the pretrained model would consider unlikely.

Another common form of reward hacking: the model learns to produce outputs in a very
specific format or style that the reward model scores highly, even though that format is
not actually what users want. The KL penalty keeps the model's outputs within the space
of "normal" text.

### Choosing the KL Coefficient (β)

| β Value | Behavior |
|---------|----------|
| 0.0 | No constraint — maximum reward hacking risk |
| 0.01-0.05 | Light constraint — policy can explore freely |
| 0.1-0.2 | Standard range — good balance |
| 0.5+ | Strong constraint — very conservative updates |

In `03_ppo_experiment.py`, you will run PPO with different β values and see this tradeoff
yourself.

---

## 4. The PPO Training Loop: One Iteration Step by Step

Here is exactly what happens in one PPO training iteration:

### Step 1: Generate Responses

The policy model receives a batch of prompts and generates responses. This is the same
as normal text generation — sampling tokens one at a time.

```
prompts = ["Human: How do I learn Python?\n\nAssistant:",
           "Human: What is machine learning?\n\nAssistant:", ...]

responses = policy_model.generate(prompts)
```

### Step 2: Score with Reward Model

Each (prompt, response) pair is fed to the reward model, which outputs a scalar score.

```
rewards = reward_model.score(prompts, responses)
# e.g., [0.82, 0.45, 0.91, ...]
```

### Step 3: Compute KL Penalty

For each generated token, we compute the log-probability under both the policy model and
the reference model. The difference gives us the per-token KL divergence.

```
kl_penalty = β * (log_probs_policy - log_probs_reference)
adjusted_rewards = rewards - kl_penalty
```

### Step 4: Compute Advantages (GAE)

The value model predicts expected returns, and we compute advantages using **Generalized
Advantage Estimation (GAE)**. This tells us, for each action: "was this better or worse
than expected?"

```
advantages = compute_gae(adjusted_rewards, values, gamma=0.99, lam=0.95)
```

You do not need to implement GAE yourself — TRL's PPOTrainer handles it.

### Step 5: Update Policy and Value Models

Using the advantages and the clipped surrogate objective, we update the policy model.
The value model is also updated to improve its predictions.

This step typically runs for multiple "mini-batch epochs" (usually 4) over the same batch
of data, which is one of PPO's efficiencies.

```
for epoch in range(ppo_epochs):
    for mini_batch in split_into_mini_batches(batch):
        policy_loss = compute_clip_loss(mini_batch)
        value_loss = compute_value_loss(mini_batch)
        total_loss = policy_loss + vf_coef * value_loss
        total_loss.backward()
        optimizer.step()
```

### Step 6: Log and Repeat

Log the mean reward, KL divergence, policy loss, and value loss. Then go back to Step 1
with the next batch of prompts.

```
┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐
│ Generate │───>│  Score   │───>│ Compute  │───>│ Compute  │───>│  Update  │
│ Response │    │  Reward  │    │ KL + Adj │    │Advantage │    │  Policy  │
└──────────┘    └──────────┘    └──────────┘    └──────────┘    └──────────┘
      ^                                                               │
      └───────────────────────── repeat ──────────────────────────────┘
```

---

## 5. Hands-on Plan

### Script 1: `01_train_reward_model.py` (Day 3)

**What it does:** Trains a reward model on Anthropic HH-RLHF preference data. Given a
pair of (chosen, rejected) responses, the model learns to assign a higher score to the
chosen response.

**Run:**
```bash
python 01_train_reward_model.py --max_samples 2000 --num_epochs 1
```

**What to observe:**
- Training loss should decrease steadily
- Evaluation accuracy: the reward model should prefer chosen over rejected responses
  more than 50% of the time (random baseline). Aim for 60-70% with this small setup.

### Script 2: `02_ppo_training.py` (Day 4)

**What it does:** Runs PPO alignment using the reward model from Script 1. The policy model
learns to generate responses that score higher with the reward model while staying close
to the reference model.

**Run:**
```bash
python 02_ppo_training.py --reward_model_path ./outputs/reward_model --max_samples 500
```

**What to observe:**
- Mean reward should increase over training steps
- KL divergence should stay moderate (not explode)
- Sample outputs at the end: compare them to what the base model would produce

### Script 3: `03_ppo_experiment.py` (Day 5)

**What it does:** Runs PPO with different KL coefficients (β = 0.0, 0.05, 0.2) and
compares the results. This is where you see the reward-vs-KL tradeoff in action.

**Run:**
```bash
python 03_ppo_experiment.py --reward_model_path ./outputs/reward_model --max_samples 200
```

**What to observe:**
- β = 0.0: highest rewards, but check the outputs — are they sensible?
- β = 0.05: good balance — decent rewards, coherent outputs
- β = 0.2: lower rewards, but outputs stay close to the pretrained model's style

---

## Check Your Understanding

Before moving to Phase 2 (DPO), make sure you can answer these:

**1. Why does PPO need four models, and what happens if you remove one?**

Think about what would break. Without the reference model, you lose the KL anchor. Without
the value model, you cannot compute advantages (you would need a different advantage
estimator). Without the reward model, you have no training signal at all.

**2. In the clipped objective, why do we take the minimum and not the maximum?**

Hint: think about what the maximum would do. If we took the max, the optimizer could
exploit outlier examples by making arbitrarily large updates whenever the advantage is
extreme.

**3. What happens when β (KL coefficient) is set to 0?**

You observed this in `03_ppo_experiment.py`. Without KL constraint, the policy is free to
drift arbitrarily far from the reference model. In practice, this often leads to reward
hacking — the policy finds degenerate outputs that score high with the reward model but
are not actually useful.

**4. Why is PPO considered "expensive" for LLM alignment?**

Consider the memory requirements. You need four full-sized models in GPU memory at once.
For a 7B parameter model, each copy takes ~14 GB in half-precision, so you need ~56 GB
just for the models — before accounting for activations, gradients, and optimizer states.
This is the main motivation for DPO (Phase 2), which only needs two models.

**5. If your mean reward is increasing but output quality is getting worse, what might be happening?**

This is reward hacking in action. The policy found a way to exploit the reward model.
Check the KL divergence — if it is very high, increase β. Also inspect the actual outputs:
are they repetitive? Unusually long? Using specific phrases repeatedly?

---

## References

- **Schulman et al. (2017)** — *Proximal Policy Optimization Algorithms*
  [arXiv:1707.06347](https://arxiv.org/abs/1707.06347). The original PPO paper. Sections
  1-3 cover the clipped objective; the rest is about continuous control (less relevant
  for LLM alignment).

- **TRL PPOTrainer Documentation** — [https://huggingface.co/docs/trl/ppo_trainer](https://huggingface.co/docs/trl/ppo_trainer).
  API reference for the trainer you will use in the scripts.

- **Ouyang et al. (2022)** — *Training language models to follow instructions with human
  feedback* [arXiv:2203.02155](https://arxiv.org/abs/2203.02155). The InstructGPT paper
  that describes how PPO is applied to LLMs specifically.
