# Algorithm Decision Framework

Your personal cheat sheet for choosing the right alignment algorithm.

Fill in the sections marked with `[YOUR OBSERVATION]` based on your experiments.

---

## 1. Summary Comparison Table

| Dimension | PPO | DPO | GRPO |
|-----------|-----|-----|------|
| **Full name** | Proximal Policy Optimization | Direct Preference Optimization | Group Relative Policy Optimization |
| **Core idea** | Train a reward model, then use RL to maximize reward while staying close to reference | Skip the reward model; optimize the policy directly from preference pairs | Generate multiple outputs, score them with a verifiable reward, use group-relative advantages |
| **Models in memory** | 4 (policy, reference, reward, value) | 2 (policy, reference) | 1 (policy only; reference-free variant) |
| **Data required** | Human preference pairs (for reward model) + prompts (for RL) | Human preference pairs (chosen/rejected) | Prompts + a verifiable reward function (no human labels needed) |
| **Reward signal** | Learned reward model (neural network) | Implicit (derived from preference data) | Explicit, rule-based (e.g., correct/incorrect answer) |
| **Training stability** | Sensitive to hyperparameters (KL coeff, learning rate, clipping) | Generally stable; fewer moving parts | Stable with enough group size; sensitive to group size choice |
| **Compute cost** | High (4 models + RL sampling loop) | Low (standard supervised training loop) | Medium (multiple generations per prompt, but only 1 model) |
| **Key hyperparameters** | KL coefficient (beta), clip epsilon, learning rate, batch size | Beta (temperature), learning rate | Group size (G), learning rate |
| **Original paper** | Schulman et al., 2017 | Rafailov et al., 2023 | Shao et al., 2024 |
| **Famous use case** | ChatGPT (original) | Zephyr, many open-source models | DeepSeek-R1 |

### Your Observations from Experiments

**Output quality on helpfulness prompts (PPO vs DPO vs GRPO vs Base):**

`[YOUR OBSERVATION]`

**Output quality on safety prompts:**

`[YOUR OBSERVATION]`

**Output quality on math prompts:**

`[YOUR OBSERVATION]`

**Which algorithm changed the model the most from the base?**

`[YOUR OBSERVATION]`

**Training stability -- which was easiest to train?**

`[YOUR OBSERVATION]`

---

## 2. Decision Flowchart

Use this flowchart when you face a new alignment task. Start at the top and follow
the branches.

```
START: What kind of task are you aligning for?
│
├─── A) VERIFIABLE CORRECTNESS (math, code, logic, factual QA)
│    │
│    │   The output can be automatically checked as right or wrong.
│    │
│    ├── Do you have a reliable automated verifier?
│    │   (test suite, answer checker, formal proof system)
│    │
│    │   ├── YES → Use GRPO
│    │   │   • No human labels needed
│    │   │   • No reward model to train
│    │   │   • Scales with compute (more generations = better signal)
│    │   │   • Proven on math (DeepSeek-R1) and code
│    │   │
│    │   └── NO, but I have preference pairs → Use DPO
│    │       • Humans ranked solutions by quality
│    │       • Works when "better" is not binary but graded
│    │       • Simpler than PPO for the same data
│    │
├─── B) SUBJECTIVE QUALITY (chat, creative writing, open-ended)
│    │
│    │   There is no single "right" answer. Quality is a human judgment.
│    │
│    ├── What data do you have?
│    │
│    │   ├── Preference pairs (chosen/rejected responses)
│    │   │   │
│    │   │   ├── Large GPU budget → Consider PPO
│    │   │   │   • More flexible than DPO
│    │   │   │   • Can iterate on the reward model separately
│    │   │   │   • Better when preference data is noisy
│    │   │   │
│    │   │   └── Limited GPU budget → Use DPO
│    │   │       • Same data as PPO, half the compute
│    │   │       • No reward model to train or maintain
│    │   │       • Easier to debug and iterate
│    │   │
│    │   └── No preference pairs, only prompts
│    │       │
│    │       ├── Can you build a reward model or use an API judge?
│    │       │   • YES → Use PPO with your reward model
│    │       │   • NO  → Collect preference data first, then use DPO
│    │       │
│    │       └── Can you define a rule-based quality metric?
│    │           • YES → Consider GRPO with your metric as reward
│    │           • NO  → You need human feedback. Start collecting data.
│    │
└─── C) MIXED TASKS (reasoning + conversation + safety)
     │
     │   Real products often need multiple capabilities.
     │
     └── Consider a multi-stage pipeline:
         1. SFT to establish instruction-following baseline
         2. DPO or PPO for general helpfulness/safety alignment
         3. GRPO for task-specific reasoning (math, code)

         [YOUR NOTES ON WHICH COMBINATION WOULD WORK BEST]
```

---

## 3. Quick Reference Card

One-sentence description of when to use each algorithm. Fill in or revise based on
your experiments.

### PPO -- When to Use

> Use PPO when you have human preference data, sufficient GPU budget for 4 models,
> and need maximum control over the alignment process (e.g., you want to iterate on
> the reward model independently from policy training).

**Your version:**

`[YOUR ONE-SENTENCE DESCRIPTION]`

### DPO -- When to Use

> Use DPO when you have human preference data and want the simplest, most
> compute-efficient path to alignment -- same data as PPO but with a standard
> supervised training loop instead of RL.

**Your version:**

`[YOUR ONE-SENTENCE DESCRIPTION]`

### GRPO -- When to Use

> Use GRPO when your task has automatically verifiable correctness (math, code, logic)
> and you want to improve reasoning ability without collecting any human preference
> data.

**Your version:**

`[YOUR ONE-SENTENCE DESCRIPTION]`

---

## 4. Common Pitfalls

### PPO Pitfalls

| Pitfall | Symptom | Fix |
|---------|---------|-----|
| **Reward hacking** | High reward scores but gibberish or repetitive output | Increase KL coefficient (beta); inspect outputs regularly |
| **KL divergence explosion** | KL grows unboundedly; model outputs become unrecognizable | Lower learning rate; increase beta; check for bugs in reference model loading |
| **Reward model quality** | Policy optimizes for reward model quirks, not actual quality | Evaluate reward model accuracy first; use more diverse preference data |
| **Memory pressure** | OOM errors during training | Use LoRA; reduce batch size; use gradient checkpointing |
| **Unstable value function** | Advantage estimates are noisy; training does not converge | Increase batch size; reduce value function learning rate |

**Pitfalls you encountered:**

`[YOUR NOTES]`

### DPO Pitfalls

| Pitfall | Symptom | Fix |
|---------|---------|-----|
| **Beta too low** | Model overfits to preference data; outputs become unnatural | Increase beta (try 0.1 to 0.5) |
| **Beta too high** | Model barely changes from reference; alignment has no effect | Decrease beta (try 0.05) |
| **Low-quality preference data** | Model learns noise or inconsistent preferences | Curate data; filter out ambiguous pairs |
| **Distribution mismatch** | Preference data was generated by a different model | Use on-policy data generation; or accept some distribution shift |
| **Catastrophic forgetting** | Model loses general capabilities after DPO training | Use LoRA; reduce number of training steps; increase beta |

**Pitfalls you encountered:**

`[YOUR NOTES]`

### GRPO Pitfalls

| Pitfall | Symptom | Fix |
|---------|---------|-----|
| **Group size too small** | Noisy advantage estimates; training is unstable | Increase group size (try 4 to 8) |
| **Group size too large** | Slow training; diminishing returns on signal quality | Reduce group size; profile compute time per step |
| **Sparse reward signal** | Most or all outputs in a group get the same reward | Make the reward function more granular; use partial credit |
| **All-correct or all-wrong groups** | Zero advantage for all samples; no learning signal | Adjust problem difficulty; mix easy and hard problems |
| **Reward function too simple** | Model finds shortcuts (e.g., always outputs the same number) | Add format checks; verify the reward function is correct |

**Pitfalls you encountered:**

`[YOUR NOTES]`

---

## 5. Scenario Worksheet

For each scenario below, write which algorithm you would choose and why.

### Scenario A: Customer Support Chatbot

You are building a chatbot that answers customer questions about a SaaS product.
You have 10,000 conversations with human-rated quality scores.

- **Algorithm choice:** `[YOUR CHOICE]`
- **Reasoning:** `[WHY]`
- **What data preparation would you need?** `[YOUR NOTES]`

### Scenario B: Math Tutoring Assistant

You want a model that solves K-12 math problems step by step. You have access to
a dataset of problems with verified solutions.

- **Algorithm choice:** `[YOUR CHOICE]`
- **Reasoning:** `[WHY]`
- **What reward function would you use?** `[YOUR NOTES]`

### Scenario C: Code Generation Tool

You need a model that generates Python functions from docstrings. You have a test
suite that can verify whether the generated code is correct.

- **Algorithm choice:** `[YOUR CHOICE]`
- **Reasoning:** `[WHY]`
- **How would you handle partial correctness?** `[YOUR NOTES]`

### Scenario D: Creative Writing Assistant

You want a model that writes engaging short stories. There is no "correct" answer,
but you have a team of editors who can rank outputs.

- **Algorithm choice:** `[YOUR CHOICE]`
- **Reasoning:** `[WHY]`
- **How would you collect preference data?** `[YOUR NOTES]`

### Scenario E: Multi-Capability Production Model

You need a model that is helpful, safe, AND good at reasoning. This is the hardest
real-world scenario.

- **Algorithm choice(s):** `[YOUR CHOICE -- likely a combination]`
- **Training pipeline:** `[DESCRIBE THE STAGES]`
- **How would you evaluate the final model?** `[YOUR NOTES]`

---

## 6. Your Personal Notes

Use this space for anything else you want to remember.

### What surprised me most:

`[YOUR NOTES]`

### What I would do differently next time:

`[YOUR NOTES]`

### What I want to explore next:

`[YOUR NOTES]`

### Key takeaway in one sentence:

`[YOUR SENTENCE]`
