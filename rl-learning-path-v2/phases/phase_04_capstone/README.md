# Phase 4: Capstone — Comparative Experiment

**Duration:** 3-4 days
**Goal:** Run a controlled comparison of PPO, DPO, and GRPO on a work-relevant task. Produce a decision framework and reusable scripts.

---

## Instructions

### Day 17: Experiment Design

1. **Choose your task.** Pick something relevant to your work:
   - Instruction following (does the model follow formatting requirements?)
   - Summarization quality (concise, accurate, complete?)
   - Safety alignment (refuses harmful requests appropriately?)
   - Helpfulness (detailed, actionable answers?)

2. **Prepare your data.** You need:
   - A set of prompts (at least 200-500 for meaningful comparison)
   - For DPO: preference pairs (prompt, chosen, rejected)
   - For PPO/GRPO: a reward function (learned model or rule-based)

3. **Define evaluation metrics:**
   - Automated: reward model scores, response length, format compliance
   - Human: pick 50 responses to manually rate on a 1-5 scale

### Day 18-19: Run Experiments

Use `01_run_all_experiments.py` to train all three algorithms with matched settings.
Use `02_evaluate_and_compare.py` to generate the comparison table.

### Day 20: Write Report

Use `03_write_report.py` to auto-generate a report skeleton from results.

The report should answer: **"Given task X with constraints Y, I would choose algorithm Z because..."**

---

## Deliverables

1. **Comparison table** — filled with your experimental results
2. **Decision framework** — when to use PPO vs DPO vs GRPO
3. **Reusable training scripts** — adapted from earlier phases for your team
4. **Lessons learned** — hyperparameter gotchas, failure modes, practical tips
