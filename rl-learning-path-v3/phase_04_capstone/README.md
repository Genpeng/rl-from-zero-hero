# Phase 4: Capstone -- Unified Comparison and Decision Framework

**Days 12-14 | ~10-15 min read**

You have now trained, evaluated, and experimented with all three alignment algorithms:
PPO (Phase 1), DPO (Phase 2), and GRPO (Phase 3). Each one solved the same fundamental
problem -- making a language model behave the way we want -- but with very different
approaches, tradeoffs, and assumptions.

This final phase ties everything together. You will run all three trained models
side-by-side on the same prompts, analyze the results, and build a personal decision
framework that you can use in real projects.

---

## Day 12: Run the Unified Comparison

### Goal

Generate responses from all three trained models (plus the base model) on identical
prompts and compare them directly. This is the "bake-off" that shows how each algorithm
affected the model's behavior.

### What You Need

Before running the comparison, make sure you have trained models from the earlier phases:

| Model | Expected Path | Trained In |
|-------|---------------|------------|
| PPO model | `./outputs/ppo_model` | Phase 1, Script 2 |
| DPO model | `./outputs/dpo_model` | Phase 2, Script 1 |
| GRPO model | `./outputs/grpo_model` | Phase 3, Script 1 |

If any model is missing, the script will skip it gracefully and note which ones were
unavailable.

### Run

```bash
python 01_unified_comparison.py
```

Or with custom model paths:

```bash
python 01_unified_comparison.py \
    --ppo_path ./outputs/ppo_model \
    --dpo_path ./outputs/dpo_model \
    --grpo_path ./outputs/grpo_model \
    --base_model Qwen/Qwen2.5-0.5B
```

### What to Observe

The script tests three categories of prompts:

1. **Helpfulness prompts** -- general questions like "How can I improve my writing?"
   - Which algorithm produces the most helpful, on-topic response?
   - Do any models produce noticeably longer or shorter answers?
   - PPO and DPO were both trained on helpfulness data. Does GRPO (trained on math)
     perform differently here?

2. **Safety prompts** -- requests that should be handled carefully
   - Which models are more cautious? Which are more direct?
   - PPO and DPO saw preference data that included safety examples. GRPO did not.
     Can you see this difference?

3. **Math/reasoning prompts** -- problems with verifiable correct answers
   - GRPO was specifically trained on math. Does it outperform PPO and DPO here?
   - Does the base model get any math problems right by chance?
   - Look at the reasoning process: does GRPO show more structured step-by-step work?

### Key Output

The script prints:
- A 4-way comparison (base, PPO, DPO, GRPO) for each prompt
- Average response lengths per model
- Math accuracy scores for models that were evaluated
- A summary table showing which models were loaded and their basic stats

---

## Day 13: Analyze Results and Build Your Comparison Table

### Goal

Turn your observations from Day 12 into a structured comparison. This is where you move
from "I ran the experiments" to "I understand the tradeoffs."

### Step 1: Fill in the Comparison Table

Open `algorithm_decision_framework.md` and fill in the comparison table. The key
dimensions are:

| Dimension | PPO | DPO | GRPO |
|-----------|-----|-----|------|
| **Training complexity** | How many models in memory? How many moving parts? |
| **Compute cost** | GPU hours, memory requirements for the scale you tested |
| **Data requirements** | What kind of data? How much? |
| **Output quality** | Your subjective assessment from the comparison |
| **Training stability** | Did loss curves converge smoothly? Any instabilities? |
| **Best use case** | Where does this algorithm shine? |

### Step 2: Review Your Training Logs

Go back to each phase and review:
- Training curves: which algorithm converged fastest?
- Final metrics: which achieved the best scores on its target task?
- Failure modes: did any algorithm produce obviously broken outputs?

### Step 3: Cross-Task Analysis

This is the most important observation: **each algorithm was trained on different data
for different objectives.** Think about what this means:

- PPO and DPO were both trained on Anthropic HH-RLHF data (human preferences about
  helpfulness and harmlessness). They should perform similarly on conversational tasks.
- GRPO was trained on GSM8K math problems with verifiable answers. It should excel at
  math but may not help with general conversation.
- The base model had no alignment training at all. It is the control group.

Ask yourself: if you could only choose *one* algorithm for a real product, which would
it be? The answer depends entirely on the product.

---

## Day 14: Write Your Algorithm Decision Framework

### Goal

Create a personal cheat sheet -- a decision framework you can reference whenever you
face a new alignment task. This is the deliverable that proves you understood the
material, not just ran the code.

### Step 1: Complete the Decision Flowchart

Open `algorithm_decision_framework.md` and work through the decision flowchart. The
template provides the structure; you fill in the recommendations based on your
experiments.

The flowchart follows this logic:

```
What kind of task?
  |
  ├── Verifiable correctness (math, code, logic)
  |     └── What data do you have?
  |           ├── Labeled correct answers → ...
  |           └── Preference pairs → ...
  |
  └── Subjective quality (chat, creative, open-ended)
        └── What compute do you have?
              ├── Large GPU budget → ...
              └── Limited GPU budget → ...
```

### Step 2: Write Your Quick Reference Card

For each algorithm, write one sentence describing when you would use it. Be specific
and practical. "Use PPO when..." should reference concrete scenarios, not abstract
properties.

### Step 3: Document Common Pitfalls

List the mistakes that tripped you up during each phase. These are invaluable in
real projects -- knowing what *not* to do is often more useful than knowing what to do.

---

## Reflection Questions

After completing the capstone, spend 10 minutes on these questions. Write your answers
in the framework document or in a separate notebook.

### Understanding the Tradeoffs

1. **What surprised you most?** Was there an algorithm that performed better or worse
   than you expected? Did the base model do surprisingly well on any task?

2. **Training complexity vs output quality:** PPO is the most complex to set up. Did it
   produce noticeably better outputs than the simpler DPO? Was the extra complexity
   worth it for the scale you tested?

3. **Data requirements:** PPO and DPO both need preference data (chosen/rejected pairs).
   GRPO needs verifiable answers. Which kind of data is easier to obtain in practice?

### Applying to Real Scenarios

4. **Chatbot for customer support:** You need a model that is helpful, safe, and
   conversational. Which algorithm would you choose? Why?

5. **Math tutoring assistant:** You need a model that reliably solves math problems
   step by step. Which algorithm would you choose? Would you combine approaches?

6. **Code generation tool:** You want a model that writes correct, runnable code.
   The signal is binary -- the code either passes tests or it does not. Which algorithm
   fits best?

7. **Creative writing assistant:** You want a model that produces engaging, varied
   creative writing. There is no "correct" answer. Which algorithm and what kind of
   preference data would you use?

### Looking Ahead

8. **Combining algorithms:** Could you use multiple algorithms in sequence? For example,
   DPO for initial alignment followed by GRPO for reasoning tasks? What would the
   pipeline look like?

9. **Scaling up:** You tested on a 0.5B parameter model. How would your conclusions
   change for a 7B or 70B model? Would the relative tradeoffs between algorithms shift?

10. **What would you study next?** Based on what you learned, what is the next topic
    you would explore? Constitutional AI? RLHF with AI feedback? Multi-turn training?

---

## Summary: What You Built

Over 14 days, you went from zero to a working understanding of three major alignment
algorithms:

```
Phase 0: Foundation
  └── SFT warmup -- understanding the starting point

Phase 1: PPO
  └── Reward model → RL training → KL ablation
  └── The original RLHF pipeline, complex but powerful

Phase 2: DPO
  └── Direct preference optimization → PPO/DPO comparison
  └── Same goal as PPO, half the infrastructure

Phase 3: GRPO
  └── Group-relative optimization → math evaluation → group size ablation
  └── RL without a reward model, using verifiable correctness

Phase 4: Capstone (you are here)
  └── Unified comparison → analysis → personal decision framework
  └── Synthesizing everything into practical knowledge
```

You did not just read about these algorithms -- you trained models with each one,
observed the tradeoffs firsthand, and built a framework for making real decisions.
That hands-on understanding is something no paper or blog post can give you.

---

## References

- **Schulman et al. (2017)** -- *Proximal Policy Optimization Algorithms*
  [arXiv:1707.06347](https://arxiv.org/abs/1707.06347)

- **Rafailov et al. (2023)** -- *Direct Preference Optimization: Your Language Model
  is Secretly a Reward Model*
  [arXiv:2305.18290](https://arxiv.org/abs/2305.18290)

- **Shao et al. (2024)** -- *DeepSeekMath: Pushing the Limits of Mathematical Reasoning
  in Open Language Models*
  [arXiv:2402.03300](https://arxiv.org/abs/2402.03300)

- **DeepSeek-AI (2025)** -- *DeepSeek-R1: Incentivizing Reasoning Capability in LLMs
  via Reinforcement Learning*
  [arXiv:2501.12948](https://arxiv.org/abs/2501.12948)
