# Phase 0: Foundation — Why Alignment Matters and How It Works

> **Time estimate:** 15-20 min read  
> **Goal:** Build the intuition you need before touching PPO, DPO, or GRPO code.

---

## 1. Why Alignment Matters

### The Gap Between "Next Token" and "Helpful"

A base language model is trained on one objective: **predict the next token**. That's it. The model learns to mimic the statistical patterns of its training data — Wikipedia articles, Reddit threads, code repositories, books, everything. This makes it shockingly capable at continuing text, but it does not make it *helpful*, *safe*, or *honest*.

Here are three concrete examples of the gap:

**Example 1: The Harmful Completion**
```
User: How do I pick a lock?

Base model: "First, you'll need a tension wrench and a rake pick. Insert the
tension wrench into the bottom of the keyhole..."

Aligned model: "I can explain how locks work from a general knowledge perspective.
Lock mechanisms use pins of varying lengths..."
```

The base model has seen lockpicking instructions in its training data, so it happily reproduces them. It has no concept of "should I answer this?" — it only knows "what text is likely to come next?"

**Example 2: The Sycophantic Answer**
```
User: I think the Earth is flat. Am I right?

Base model (sycophantic): "Yes, many people share your view that the Earth is flat.
There are several compelling arguments..."

Aligned model: "The Earth is not flat — it's an oblate spheroid. This is supported
by satellite imagery, physics of gravity, and centuries of scientific observation..."
```

Because the training data contains text where people agree with each other, a base model often learns to be a yes-man. Alignment teaches it to prioritize truthfulness over agreeableness.

**Example 3: The Rambling Non-Answer**
```
User: What's the capital of France?

Base model: "The capital of France is a topic that has been discussed extensively
in European history. France, a country in Western Europe, has a rich history
dating back to the Gauls. The Franks, under Clovis I, unified..."

Aligned model: "The capital of France is Paris."
```

The base model produces *plausible text*, but it doesn't optimize for *being useful*. An aligned model learns that concise, direct answers are what users actually want.

### The Core Problem

Pre-training gives the model *knowledge*. Alignment gives it *behavior*.

Without alignment, you have a text-completion engine. With alignment, you have an assistant. The entire field of RLHF (Reinforcement Learning from Human Feedback) exists to bridge this gap.

The landmark paper that demonstrated this at scale was **InstructGPT** (Ouyang et al., 2022), which showed that a 1.3B parameter model fine-tuned with RLHF could be preferred by human evaluators over the 175B parameter base GPT-3. Smaller model, better behavior — because behavior was being explicitly optimized.

---

## 2. The RLHF Pipeline

The standard RLHF pipeline has three stages. Each stage addresses a specific limitation of the previous one.

```
┌─────────────────────────────────────────────────────────────┐
│                    THE RLHF PIPELINE                        │
│                                                             │
│   Stage 1          Stage 2            Stage 3               │
│  ┌───────┐      ┌───────────┐      ┌──────────────┐        │
│  │  SFT  │ ──▶  │  Reward   │ ──▶  │     RL       │        │
│  │       │      │  Model    │      │ Optimization │        │
│  └───────┘      └───────────┘      └──────────────┘        │
│  "Learn the      "Learn what       "Get better at          │
│   format"         humans like"      what humans like"       │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Stage 1: Supervised Fine-Tuning (SFT)

**What it does:** Takes the base model and fine-tunes it on high-quality (prompt, response) pairs written or curated by humans.

**Why it's needed:** The base model can complete any text, but it doesn't know the *format* of a helpful assistant. SFT teaches it the pattern:

```
Human asks question → Assistant gives helpful answer
```

**What it doesn't do:** SFT only teaches the model to imitate the demonstration data. If the demonstrations don't cover a topic, or if they contain subtle quality differences, SFT can't distinguish "good" from "great." It learns to produce *plausible* assistant-style responses, not necessarily *optimal* ones.

**Analogy:** SFT is like a new employee watching training videos. They learn the general format of how to do the job, but they haven't yet received real feedback on their own work.

### Stage 2: Reward Model Training

**What it does:** Trains a separate model to predict how much a human would prefer one response over another. The input is a (prompt, response) pair; the output is a scalar score.

**How it's trained:** Human annotators are shown a prompt and two candidate responses. They pick which one they prefer. These preference pairs become training data:

```
Prompt: "Explain photosynthesis"
Response A: [clear, concise explanation]     ← preferred (higher reward)
Response B: [vague, rambling explanation]     ← rejected (lower reward)
```

The reward model learns to assign higher scores to responses that humans tend to prefer.

**Why it's needed:** We can't run RL directly on human feedback — humans are slow and expensive. The reward model is a *learned proxy* for human judgment. Once trained, it can score millions of responses instantly.

**Analogy:** The reward model is like a senior manager who has internalized company standards. Instead of the CEO reviewing every piece of work, the manager can quickly assess quality.

### Stage 3: RL Optimization

**What it does:** Uses reinforcement learning (specifically PPO in the original InstructGPT) to update the SFT model so that it generates responses that score higher according to the reward model.

**The loop:**
1. The model generates a response to a prompt
2. The reward model scores that response
3. The RL algorithm updates the model to make high-scoring responses more likely
4. Repeat thousands of times

**The critical constraint:** The RL-optimized model must not drift too far from the SFT model. If it does, it will find degenerate ways to "hack" the reward model — producing outputs that get high scores but are nonsensical to humans. This is called **reward hacking**. The tool we use to prevent this is **KL divergence** (more on this below).

**Analogy:** RL optimization is like the employee doing their job and getting performance reviews. They adjust their approach based on feedback, but they can't deviate so much that they're unrecognizable from what they were trained to do.

---

## 3. Just-in-Time RL Concepts

You don't need a full RL course to understand PPO, DPO, and GRPO. You need exactly five concepts. Here they are, explained in the context of language models.

### Policy (= the LLM itself)

In RL terminology, the **policy** is the decision-maker — the thing that observes a situation and chooses an action. In LLM alignment:

- **The policy is the language model.**
- The "situation" (state) is the prompt plus all tokens generated so far.
- The "action" is choosing the next token.
- The policy is a probability distribution over the vocabulary at each step.

When papers say "update the policy," they mean "update the model weights." When they say "the policy generates a response," they mean "the model runs autoregressive decoding."

You'll see two policies mentioned frequently:
- **pi_theta** (or just "the policy"): the model being trained
- **pi_ref** (the "reference policy"): a frozen copy of the model before RL training, used to keep the trained model from drifting too far

### Reward (= a score for how good the output is)

The **reward** is a scalar number that tells the RL algorithm how good a complete generated response is. Higher reward = the response is more aligned with what humans want.

Where does the reward come from?
- **In PPO:** From the trained reward model (Stage 2 of the pipeline)
- **In DPO:** There's no explicit reward — it's implicit in the preference data
- **In GRPO:** From a rule-based function (e.g., "did the model get the math answer correct?")

The reward is always assigned to the *complete* response, not to individual tokens. This creates a credit assignment problem: if a 200-token response gets a high reward, which tokens deserve the credit? That's where the next two concepts come in.

### Value Function (= predicting expected future reward)

The **value function** V(s) answers the question: "Given the tokens generated so far, how much total reward do we *expect* to get by the time the response is complete?"

Think of it as the model's prediction of its own future performance. At each token position, the value function looks at the prompt + tokens so far and estimates the final reward score.

- **In PPO:** A separate value head (a small neural network) is attached to the model to learn this estimate. It's trained alongside the policy.
- **In DPO:** No value function needed — it's mathematically eliminated.
- **In GRPO:** No value function needed — replaced by group-level comparisons.

The value function exists to make the advantage calculation possible (see below). It is one of the things that makes PPO more complex than DPO and GRPO.

### Advantage (= how much better this action was than average)

The **advantage** A(s, a) answers: "How much better was *this specific token choice* compared to what we'd *typically* expect at this position?"

```
Advantage = Actual reward received - Expected reward (from value function)
          = Q(s,a) - V(s)
```

If the advantage is positive, the chosen token led to a better-than-expected outcome, so we should make that token more likely in the future. If negative, the token led to a worse outcome, so we should make it less likely.

**Concrete example:**
```
Prompt: "What is 2+2?"
Generated: "The answer is 4."

If the reward model gives this a high score (say +2.0),
and the value function predicted a reward of +1.0,
then the advantage is +1.0 (better than expected).

The model will increase the probability of generating
"The answer is 4." in similar situations.
```

How advantage is computed differs across algorithms:
- **PPO:** Uses GAE (Generalized Advantage Estimation) combining per-token value function predictions with actual rewards. Requires a value model.
- **GRPO:** Generates multiple responses to the same prompt, scores them all, then uses the *group statistics* (mean and standard deviation) to normalize. Responses above the group average get positive advantage; those below get negative. No value model needed.

### KL Divergence (= how far the model has drifted)

**KL divergence** (Kullback-Leibler divergence) measures how different two probability distributions are. In alignment, it measures how far the policy (the model being trained) has drifted from the reference policy (the original SFT model).

**Why it matters:** Without a KL penalty, the RL algorithm will find degenerate solutions. The model might learn to repeat a single "magic phrase" that the reward model scores highly, or generate gibberish that exploits some quirk in the reward model. This is **reward hacking**.

The KL penalty is added to the reward to create a modified objective:

```
Objective = Reward(response) - beta * KL(policy || reference_policy)
```

- **beta** is a hyperparameter that controls the trade-off. Higher beta = the model stays closer to the reference (safer but less improvement). Lower beta = the model has more freedom (more improvement but risk of reward hacking).

How KL divergence is used:
- **PPO:** KL penalty is added as part of the reward calculation, penalizing per-token deviations from the reference model.
- **DPO:** KL constraint is built into the math of the loss function itself (controlled by the beta parameter).
- **GRPO:** KL divergence is added as an explicit penalty term to the loss.

**Intuition:** KL divergence is the training wheels on RL fine-tuning. It lets the model improve while preventing it from falling off a cliff. Finding the right beta is one of the most important practical decisions in alignment.

---

## 4. What's Next: Preview of the Three Algorithms

Now that you have the foundation, here's a roadmap of what we'll learn and how each phase builds on the previous.

### Phase 1: PPO (Proximal Policy Optimization) — Days 3-5

PPO is the **original approach** used by InstructGPT and early ChatGPT. It follows the full RLHF pipeline:

```
SFT Model → Reward Model → RL Training (PPO)
```

PPO requires the most moving parts: a policy model, a reference model, a reward model, and a value function. It's complex but powerful, and understanding it gives you the vocabulary to understand everything else.

**Key idea:** "Update the policy to get higher rewards, but don't change it too much in any single step."

### Phase 2: DPO (Direct Preference Optimization) — Days 6-8

DPO asks: "What if we could skip the reward model entirely?"

The insight is that you can mathematically derive the *optimal policy* directly from preference data, without ever training an explicit reward model or running RL. DPO reformulates the RLHF objective into a simple classification loss.

```
SFT Model → DPO Training (using preference pairs directly)
```

DPO eliminates two of PPO's components (reward model and value function), making it much simpler to implement and train. We'll compare its results directly with PPO on the same data.

**Key idea:** "Why learn a reward model and then optimize against it, when you can go straight from preferences to the optimal policy?"

### Phase 3: GRPO (Group Relative Policy Optimization) — Days 9-11

GRPO is designed for tasks where you can *verify* the answer — like math, coding, or fact-checking. Instead of training a reward model, you use a rule-based verifier (e.g., "did the model produce the correct answer?").

The twist: GRPO eliminates the value function by generating *multiple responses* to each prompt and using the group's statistics to estimate advantage. Responses that are above average for the group get reinforced; those below average get suppressed.

```
SFT Model → GRPO Training (using verifiable rewards + group normalization)
```

We'll apply GRPO to math reasoning (GSM8K) and see how it teaches the model to improve at step-by-step problem solving.

**Key idea:** "Generate a batch of responses, score them all, reinforce the ones that are better than average."

### Phase 4: Capstone — Days 12-14

Bring it all together. Compare all three algorithms on the same base model and build your personal decision framework for when to use each one.

### How the Phases Connect

```
Phase 0 (you are here)
  └── SFT warm-up (produces the base for all later experiments)
        │
        ├──▶ Phase 1: PPO (full RLHF pipeline)
        │
        ├──▶ Phase 2: DPO (simplified pipeline, same preference data)
        │
        ├──▶ Phase 3: GRPO (verifiable tasks, no reward model)
        │
        └──▶ Phase 4: Compare all three
```

---

## Check Your Understanding

Before moving on to the SFT warm-up script (`01_sft_warmup.py`), test yourself on these questions. Don't look up the answers — just check whether you have an intuition.

**Q1.** A base language model gives a confident, detailed answer to a dangerous question. Is this a failure of the model's *knowledge* or its *behavior*? Why?

<details>
<summary>Answer</summary>

It's a failure of **behavior**, not knowledge. The model *knows* the information (it's in the training data), but it hasn't learned the behavioral policy of *when not to answer*. Alignment addresses this gap — it's about shaping behavior, not adding or removing knowledge.
</details>

**Q2.** You're running PPO training and notice the model starts producing repetitive, formulaic responses that all get high reward scores but sound unnatural. What's likely happening, and which hyperparameter would you adjust?

<details>
<summary>Answer</summary>

This is **reward hacking** — the model has found a pattern that exploits the reward model. The model has drifted too far from its original behavior. You would **increase beta** (the KL divergence coefficient) to penalize deviation from the reference model more heavily, pulling the model back toward natural language.
</details>

**Q3.** DPO eliminates the reward model and value function. What does it still require that you must prepare as training data?

<details>
<summary>Answer</summary>

DPO requires **preference pairs** — triples of (prompt, chosen_response, rejected_response) where a human has indicated which response is better. The reward model is mathematically absorbed into the loss function, but the *data* that would have trained the reward model is still needed.
</details>

**Q4.** In GRPO, the model generates 8 responses to the same prompt. 2 of them get the correct answer and 6 are wrong. Which responses get positive advantage, and why doesn't GRPO need a value function?

<details>
<summary>Answer</summary>

The 2 correct responses get **positive advantage** (above the group average), and the 6 wrong ones get **negative advantage** (below average). GRPO doesn't need a value function because it uses the **group's mean and standard deviation** as the baseline instead. The group statistics serve the same purpose — estimating "what's average" — without requiring a separate neural network to learn it.
</details>

**Q5.** Why does every alignment method (PPO, DPO, GRPO) include some form of KL constraint? What would happen without it?

<details>
<summary>Answer</summary>

Without a KL constraint, the model would **over-optimize** against its training signal and forget general language capabilities. In PPO, it would hack the reward model. In DPO, it would overfit to the preference pairs. In GRPO, it would collapse to always producing one "safe" answer pattern. The KL constraint ensures the model improves *relative to a sensible baseline* rather than degenerating.
</details>

---

## References

- **InstructGPT:** Ouyang, L., et al. (2022). "Training language models to follow instructions with human feedback." *NeurIPS 2022*. [arXiv:2203.02155](https://arxiv.org/abs/2203.02155)
- **Lilian Weng's RLHF Blog Post:** Weng, L. (2023). "Reinforcement Learning from Human Feedback." [lilianweng.github.io](https://lilianweng.github.io/posts/2023-03-15-rlhf/) — Excellent visual walkthrough of the full pipeline.
- **PPO:** Schulman, J., et al. (2017). "Proximal Policy Optimization Algorithms." [arXiv:1707.06347](https://arxiv.org/abs/1707.06347)
- **DPO:** Rafailov, R., et al. (2023). "Direct Preference Optimization: Your Language Model is Secretly a Reward Model." *NeurIPS 2023*. [arXiv:2305.18290](https://arxiv.org/abs/2305.18290)
- **GRPO:** Shao, Z., et al. (2024). "DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models." [arXiv:2402.03300](https://arxiv.org/abs/2402.03300)

---

**Next step:** Run `01_sft_warmup.py` to fine-tune your base model and create the starting point for all later experiments.
