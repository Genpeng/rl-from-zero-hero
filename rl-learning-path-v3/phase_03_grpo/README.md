# Phase 3: GRPO (Group Relative Policy Optimization)

**Days 8-10 | ~15-20 min read**

You have now seen both sides of the RLHF coin: PPO (Phase 1), which trains a policy
using a learned reward model and a value function, and DPO (Phase 2), which skips the
reward model entirely by learning directly from preference pairs.

Both methods were designed for a world where "good" and "bad" are matters of opinion --
human preference data tells the model what users like. But what about tasks where there
is a *right answer*? Math problems have correct solutions. Code either passes tests or
it does not. Logical puzzles have verifiable conclusions.

GRPO was built for exactly this setting. It is the algorithm behind DeepSeek-R1, the
model that learned to reason step-by-step *without any human preference data at all*.

---

## 1. The Problem with PPO and DPO for Reasoning

### Sparse and Delayed Rewards

Consider teaching a model to solve a multi-step math problem. The final answer is either
right or wrong -- that is a single bit of feedback at the very end of a long chain of
reasoning. PPO's reward model was trained on human preferences about *style* and
*helpfulness*, not on mathematical correctness. And the value function, which estimates
"how well are we doing so far?" at every token, has almost nothing useful to learn from
a single end-of-sequence reward.

This is the **sparse reward problem**. The model produces 200 tokens of work, and the
only signal is a thumbs-up or thumbs-down at the end. PPO's machinery -- the value
model, the GAE advantage estimation, the per-token KL penalties -- was designed for
richer, denser feedback.

### Verifiable Correctness Changes the Game

Math and code tasks have a special property: you can **automatically verify** whether the
answer is correct. You do not need a human to judge the output. You do not need a neural
network to approximate human judgment. You just check: did the model produce the right
number?

This means:
- You do not need to collect expensive human preference data
- You do not need to train a reward model (which can be wrong or hackable)
- You have a *perfect* reward signal -- binary, noise-free, and cheap to compute

The question is: how do you use this simple binary signal to train a policy effectively?

### Why Not Just Use PPO with a Binary Reward?

You could, in principle, use PPO with a reward function that returns 1.0 for correct
answers and 0.0 for incorrect ones. But PPO's value model would struggle badly. With a
binary reward that only arrives at the end of generation, the value model cannot learn
useful per-token predictions. It would add a lot of parameters and computational cost
without contributing much signal.

And you certainly cannot use DPO -- it requires paired preferences, and a binary
correctness signal does not naturally produce (chosen, rejected) pairs.

What we need is a method that:
1. Works with a simple verifiable reward function
2. Does not require a value model
3. Does not require paired preference data
4. Can still compute meaningful advantages for policy updates

Enter GRPO.

---

## 2. GRPO's Innovation: Let the Group Be the Baseline

GRPO stands for **Group Relative Policy Optimization**. The core idea is deceptively
simple:

> Instead of training a value model to estimate "how good is this state?", sample a
> *group* of outputs for the same prompt and compare them against each other. The group
> average becomes the baseline.

### The Intuition

Imagine you are a teacher grading math exams. You have no answer key (no reward model),
and you do not know how hard the problems are (no value model). But you have a way to
check whether each answer is correct.

For a given problem, you ask the student to solve it multiple times (say, 4 attempts).
If 3 out of 4 attempts are correct, then a correct answer is *average* -- nothing
special. But if only 1 out of 4 is correct, that one correct attempt was *exceptionally
good* relative to the student's current ability, and you should reinforce whatever
reasoning led to it.

This is exactly what GRPO does:
- For each prompt, generate a **group** of completions (say, G = 4 or 8)
- Score each completion with a verifiable reward function
- Compute advantages *relative to the group* -- above-average completions get positive
  advantage, below-average get negative advantage
- Update the policy to make above-average completions more likely

No value model needed. No reward model needed. The group itself provides the baseline.

### Why This Works

The group-relative baseline solves the sparse reward problem in a clever way. Even when
the reward is binary (correct or incorrect), the *proportion* of correct answers in the
group carries rich information:

- If all G completions are wrong: no useful gradient (the model cannot solve this yet)
- If all G completions are correct: no useful gradient (the model already solved this)
- If *some* are correct and some are wrong: this is the sweet spot. GRPO can compare
  what the correct ones did differently and reinforce those patterns.

This is why the group size matters. A larger group gives a more stable estimate of the
model's current ability on each prompt and a more informative advantage signal.

---

## 3. The GRPO Algorithm

Let's walk through the algorithm step by step.

### Step 1: Sample a Group of Outputs

For each prompt `x` in the batch, sample `G` completions from the current policy:

```
o_1, o_2, ..., o_G  ~  pi_theta(· | x)
```

For example, with G = 4, you generate 4 different solutions to each math problem.

### Step 2: Score Each Output

Apply the reward function to each completion:

```
r_1, r_2, ..., r_G  =  R(o_1), R(o_2), ..., R(o_G)
```

For math problems, R(o) = 1.0 if the final answer is correct, 0.0 otherwise. But R can
be any function -- partial credit, test case pass rate for code, or even a learned reward
model if you want.

### Step 3: Compute Group-Relative Advantages

This is the key formula. For each output `i` in the group:

```
A_i = (r_i - mean(r)) / std(r)
```

Where `mean(r)` and `std(r)` are computed over the group `{r_1, ..., r_G}`.

**What this means:**
- If `r_i` is above the group average, `A_i > 0` (reinforce this output)
- If `r_i` is below the group average, `A_i < 0` (discourage this output)
- The division by `std(r)` normalizes the scale so the advantage is comparable across
  prompts with different difficulty levels

**Example with G = 4, binary reward:**

| Output | Correct? | r_i | mean(r) | std(r) | A_i    |
|--------|----------|-----|---------|--------|--------|
| o_1    | Yes      | 1.0 | 0.5     | 0.577  | +0.87  |
| o_2    | No       | 0.0 | 0.5     | 0.577  | -0.87  |
| o_3    | Yes      | 1.0 | 0.5     | 0.577  | +0.87  |
| o_4    | No       | 0.0 | 0.5     | 0.577  | -0.87  |

Here, 2 out of 4 are correct. The correct ones get a positive advantage of +0.87; the
incorrect ones get -0.87. The policy update will make the correct solutions more likely
and the incorrect ones less likely.

Now compare what happens if 3 out of 4 are correct:

| Output | Correct? | r_i | mean(r) | std(r) | A_i    |
|--------|----------|-----|---------|--------|--------|
| o_1    | Yes      | 1.0 | 0.75    | 0.50   | +0.50  |
| o_2    | Yes      | 1.0 | 0.75    | 0.50   | +0.50  |
| o_3    | Yes      | 1.0 | 0.75    | 0.50   | +0.50  |
| o_4    | No       | 0.0 | 0.75    | 0.50   | -1.50  |

The single incorrect output gets a strongly negative advantage (-1.50), while each
correct output gets a modest positive advantage (+0.50). This makes sense: the model
mostly knows how to solve this problem, so the one failure is the important learning
signal.

### Step 4: Policy Update with Clipping and KL

GRPO uses the same clipped surrogate objective as PPO, but with the group-relative
advantage:

```
L_GRPO = E[ min( r(theta) * A_i,  clip(r(theta), 1-eps, 1+eps) * A_i ) ]
         - beta * KL(pi_theta || pi_ref)
```

Where:
- `r(theta) = pi_theta(o_i | x) / pi_old(o_i | x)` is the probability ratio
- `eps` is the clipping range (typically 0.2, same as PPO)
- `beta` is the KL penalty coefficient
- `pi_ref` is the reference (initial) model

The clipping prevents overly large policy updates (same as PPO), and the KL penalty
keeps the model from drifting too far from its pretrained distribution.

### The Full Picture

```
┌─────────────────────────────────────────────────────────────┐
│                    GRPO Training Loop                       │
│                                                             │
│  For each prompt x:                                         │
│                                                             │
│  ┌──────────────┐    ┌──────────────────┐                  │
│  │ Policy Model │───>│ Sample G outputs │                  │
│  │  pi_theta    │    │ o_1, o_2, ..o_G  │                  │
│  └──────────────┘    └────────┬─────────┘                  │
│                               │                             │
│                      ┌────────▼─────────┐                  │
│                      │  Reward Function  │                  │
│                      │  r_i = R(o_i)     │                  │
│                      │  (e.g., correct?) │                  │
│                      └────────┬─────────┘                  │
│                               │                             │
│                      ┌────────▼──────────────┐             │
│                      │  Group Advantages     │             │
│                      │  A_i = (r_i - mean)   │             │
│                      │         / std          │             │
│                      └────────┬──────────────┘             │
│                               │                             │
│                      ┌────────▼──────────────┐             │
│                      │  Clipped PPO Update   │             │
│                      │  + KL penalty         │             │
│                      └───────────────────────┘             │
│                                                             │
│  No value model. No reward model training.                  │
│  Just: sample, score, normalize, update.                    │
└─────────────────────────────────────────────────────────────┘
```

---

## 4. Connection to DeepSeek-R1

GRPO is not just a theoretical curiosity. It is the algorithm that enabled one of the
most impressive demonstrations of emergent reasoning in large language models.

### DeepSeek-R1-Zero: Reasoning Without Human Data

In January 2025, DeepSeek released **DeepSeek-R1**, a model that performs chain-of-thought
reasoning comparable to OpenAI's o1. The remarkable part is how it was trained.

**DeepSeek-R1-Zero** was the proof-of-concept: a base language model trained *purely* with
GRPO on math and coding problems, with no supervised fine-tuning, no human preference
data, and no reward model. The only training signal was a simple rule-based reward:

- **Accuracy reward:** Does the final answer match the ground truth? (1.0 or 0.0)
- **Format reward:** Does the output include `<think>...</think>` tags for reasoning
  followed by a final answer? (small bonus)

That is it. No human annotators labeling "this reasoning chain is good." No reward model
learning from preference pairs. Just binary correctness feedback and GRPO.

### What Emerged

The results were striking. Without being explicitly taught to reason step-by-step, the
model spontaneously developed:

1. **Extended chain-of-thought reasoning**: The model learned to "think out loud" before
   answering, breaking problems into substeps.
2. **Self-verification**: The model began checking its own work, sometimes catching and
   correcting errors mid-solution.
3. **Exploration of multiple approaches**: When stuck, the model would try different
   solution strategies.

These behaviors *emerged* from GRPO training -- they were not present in the base model
and were not taught through supervised examples. GRPO's group sampling naturally favors
outputs that reason carefully, because those are more likely to arrive at correct answers.

### From R1-Zero to R1

The full DeepSeek-R1 added a few more steps on top:
1. Start with GRPO on the base model (like R1-Zero) to develop reasoning
2. Curate high-quality reasoning examples from the GRPO-trained model
3. Run supervised fine-tuning on these curated examples
4. Apply another round of GRPO + preference-based RL for polish

But the foundational insight -- that GRPO with verifiable rewards can teach a model to
reason -- came from the R1-Zero experiment.

### Why GRPO Was the Right Tool

PPO would have required training a value model to estimate expected reward at every token
of a long reasoning chain. With sparse, binary rewards, this value model would learn
very slowly and add significant overhead.

DPO would have required paired preference data: "this solution is better than that
solution." Creating such data for math problems is possible but expensive and unnatural.

GRPO needed only the math problems and their correct answers -- a verifiable reward
function. The group sampling provided all the "comparison" signal needed.

---

## 5. PPO vs DPO vs GRPO: Comparison

| Aspect                  | PPO                              | DPO                              | GRPO                              |
|-------------------------|----------------------------------|----------------------------------|------------------------------------|
| **Reward signal**       | Learned reward model             | Implicit (from preference pairs) | Any reward function (verifiable)   |
| **Models needed**       | Policy + Reference + Reward + Value (4) | Policy + Reference (2)     | Policy + Reference (2)             |
| **Training data**       | Prompts + reward model           | Paired preferences (chosen/rejected) | Prompts + reward function      |
| **Best for**            | General alignment (helpfulness, safety) | Preference alignment when data exists | Tasks with verifiable correctness |
| **Key advantage**       | Flexible; works with any reward signal | Simple; no RL infrastructure | No value model; works without human data |
| **Key limitation**      | Expensive (4 models in memory); value model struggles with sparse rewards | Needs paired preference data; not natural for verification tasks | Needs multiple samples per prompt (higher inference cost); reward must be computable |
| **KL constraint**       | KL penalty in reward             | Built into the DPO objective     | KL penalty in objective            |
| **Advantage estimation**| GAE with learned value function  | N/A (no explicit advantage)      | Group-relative normalization       |
| **Notable use**         | ChatGPT (InstructGPT)            | Llama 2, Zephyr                  | DeepSeek-R1                        |

**When to use which:**
- **PPO:** You have a reward model (or can train one) and want maximum flexibility.
  Best when reward signals are dense or the task is too subjective for verification.
- **DPO:** You have high-quality preference pairs and want a simpler training pipeline.
  Best for stylistic/behavioral alignment where humans have clear preferences.
- **GRPO:** You have tasks with verifiable correct answers (math, code, logic) and want
  the model to develop reasoning capabilities. Best when human data is expensive or
  unavailable.

---

## 6. Hands-on Plan

### Script 1: `01_grpo_training.py` (Day 8)

**What it does:** Trains a language model on GSM8K math problems using GRPO. For each
problem, the model generates a group of solutions, and a reward function checks whether
each solution arrives at the correct answer. GRPO reinforces the correct solutions
relative to the group.

**Run:**
```bash
python 01_grpo_training.py --max_samples 1000 --num_generations 4
```

**What to observe:**
- Training accuracy should increase over time as the model learns to solve more problems
- The reward function is binary: 1.0 for correct, 0.0 for incorrect
- With a small model (0.5B), do not expect high absolute accuracy, but you should see
  improvement from the starting point
- Check the training logs for the proportion of correct answers in each batch

### Script 2: `02_grpo_evaluation.py` (Day 9)

**What it does:** Evaluates the GRPO-trained model on GSM8K test problems and compares
it to the base model. This is the "before vs after" check.

**Run:**
```bash
python 02_grpo_evaluation.py --model_path ./outputs/grpo_model --max_samples 100
```

**What to observe:**
- Base model accuracy vs GRPO model accuracy -- is there a clear improvement?
- Look at the actual generated solutions: does the GRPO model show any signs of
  step-by-step reasoning?
- Even small improvements are meaningful with a 0.5B model

### Script 3: `03_grpo_experiment.py` (Day 10)

**What it does:** Ablation study comparing GRPO with different group sizes (2, 4, 8).
Larger groups provide more stable advantage estimates but cost more inference compute.

**Run:**
```bash
python 03_grpo_experiment.py --max_samples 500
```

**What to observe:**
- Group size 2: noisiest signal (each prompt has only 2 samples to compare)
- Group size 4: reasonable balance of signal quality and cost
- Group size 8: most stable advantages, but 2x the generation cost of G=4
- The comparison table shows how accuracy and training dynamics differ

---

## Check Your Understanding

Before moving to Phase 4, make sure you can answer these:

<details>
<summary><strong>1. Why can't DPO be used directly for math reasoning tasks?</strong></summary>

DPO requires paired preference data: for each prompt, you need a "chosen" (better)
response and a "rejected" (worse) response. For math problems, the natural signal is
binary correctness -- an answer is right or wrong. You could artificially construct pairs
by pairing a correct solution with an incorrect one, but this is not what DPO was
designed for. DPO's objective optimizes for *relative preference* between two responses,
not for absolute correctness. GRPO is more natural because it works directly with the
binary reward signal and uses group sampling to derive the relative comparisons
internally.
</details>

<details>
<summary><strong>2. What happens when all outputs in a GRPO group are correct (or all are incorrect)?</strong></summary>

When all outputs have the same reward, `std(r) = 0`, which means the advantage formula
`A_i = (r_i - mean(r)) / std(r)` is undefined (division by zero). In practice,
implementations handle this by setting all advantages to zero for that prompt --
effectively skipping it. This makes intuitive sense: if the model always gets the answer
right, there is nothing to learn from this problem. If it always gets it wrong, there is
no correct solution to reinforce. The most useful learning happens when there is a
*mixture* of correct and incorrect outputs in the group.
</details>

<details>
<summary><strong>3. Why does GRPO not need a value model, while PPO does?</strong></summary>

PPO uses a value model to estimate the expected reward at each step, so it can compute
per-token advantages ("was this token better or worse than expected?"). GRPO avoids this
by computing advantages at the *sequence level*: each complete output gets a single
advantage score based on how its reward compares to the group. This is much simpler but
means GRPO cannot assign credit to individual tokens -- it treats the entire output as a
unit. For tasks with end-of-sequence rewards (like math correctness), this is fine
because the per-token value estimates would have been noisy anyway.
</details>

<details>
<summary><strong>4. How does group size (G) affect GRPO training?</strong></summary>

Larger groups give more stable advantage estimates because the mean and standard
deviation are computed over more samples. With G=2, you have extreme variance: each
prompt produces either (correct, incorrect) -- useful but noisy -- or (same, same) --
useless. With G=8, you get a richer picture of the model's ability on each prompt.
However, larger groups cost more in inference: G=8 generates 4x as many tokens as G=2.
There is also a memory tradeoff: more completions per prompt means more data to store
during the update step. In practice, G=4 to G=8 is a common sweet spot.
</details>

<details>
<summary><strong>5. DeepSeek-R1-Zero developed chain-of-thought reasoning without being trained on reasoning examples. How is this possible?</strong></summary>

GRPO selects *for outputs that get the right answer*. Among the random outputs a base
model generates, those that happen to include some intermediate reasoning steps are more
likely to arrive at the correct answer by chance. GRPO reinforces these outputs, making
the model more likely to produce reasoning steps in the future. Over many training
iterations, this creates a positive feedback loop: more reasoning leads to more correct
answers, which leads to more reinforcement of reasoning. The chain-of-thought behavior
was not explicitly taught -- it emerged because it was *useful* for maximizing the
verifiable reward. This is an example of emergent behavior from reward optimization.
</details>

---

## References

- **Shao et al. (2024)** -- *DeepSeekMath: Pushing the Limits of Mathematical Reasoning
  in Open Language Models* [arXiv:2402.03300](https://arxiv.org/abs/2402.03300). The
  paper that introduced GRPO, showing it outperforms PPO for mathematical reasoning
  while being simpler and more efficient.

- **DeepSeek-AI (2025)** -- *DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via
  Reinforcement Learning* [arXiv:2501.12948](https://arxiv.org/abs/2501.12948). The R1
  paper describing how GRPO with verifiable rewards produced emergent chain-of-thought
  reasoning in DeepSeek-R1-Zero.

- **TRL GRPOTrainer Documentation** --
  [https://huggingface.co/docs/trl/grpo_trainer](https://huggingface.co/docs/trl/grpo_trainer).
  API reference for the trainer used in this phase's scripts.
