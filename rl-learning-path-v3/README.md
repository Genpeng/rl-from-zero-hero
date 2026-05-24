# PPO / DPO / GRPO — Minimal Learning Path

A 14-day, hands-on learning path for mastering three key LLM alignment algorithms.

**Who is this for?** NLP practitioners who use LLMs but haven't trained or fine-tuned one. No reinforcement learning background required.

**What you'll learn:**
- PPO (Proximal Policy Optimization) — the original RLHF approach
- DPO (Direct Preference Optimization) — alignment without a reward model
- GRPO (Group Relative Policy Optimization) — alignment for reasoning tasks

## Quick Start

```bash
pip install -r requirements.txt
```

Base model: **Qwen2.5-0.5B** (tiny, fast iterations on cloud GPU)

## 14-Day Schedule

| Phase | Days | Topic | What You'll Do |
|-------|------|-------|----------------|
| 0 | 1-2 | Foundation | Understand RLHF motivation + run SFT warm-up |
| 1 | 3-5 | PPO | Train reward model + run PPO + experiment with KL |
| 2 | 6-8 | DPO | Run DPO + compare with PPO + tweak beta |
| 3 | 9-11 | GRPO | Run GRPO on math reasoning + ablation experiments |
| 4 | 12-14 | Capstone | Compare all 3 + build your decision framework |

## Project Structure

```
shared/                    Reusable utilities (model loading, data prep, evaluation)
phase_00_foundation/       RLHF big picture + SFT warm-up
phase_01_ppo/              PPO theory + training scripts + experiments
phase_02_dpo/              DPO theory + training scripts + experiments
phase_03_grpo/             GRPO theory + training scripts + experiments
phase_04_capstone/         Comparative analysis + decision framework
```

## How to Use

Each phase has:
1. **README.md** — Theory notes (~15 min read). Read this first.
2. **Training scripts** — Annotated Python scripts. Each has a CONFIG section at the top for easy tweaking.
3. **Config files** — YAML hyperparameter configs.

Follow phases in order. Each algorithm builds on understanding from the previous one.

## Prerequisites

- Python 3.10+
- Cloud GPU with 16+ GB VRAM (A100/H100 recommended)
- Hugging Face account (for model/dataset access)
