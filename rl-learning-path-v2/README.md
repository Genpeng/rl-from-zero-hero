# RL Learning Path: PPO, DPO, GRPO for LLM Alignment

A self-contained learning path to master three key reinforcement learning algorithms for LLM alignment. Go from zero RL knowledge to running and comparing PPO, DPO, and GRPO in 3-4 weeks.

## Quick Start

```bash
# 1. Set up environment
chmod +x setup_env.sh && bash setup_env.sh
source .venv/bin/activate

# 2. Start with Phase 0 theory
open phases/phase_00_foundation/README.md
```

## Learning Path

| Phase | Topic | What You'll Do |
|-------|-------|---------------|
| 0 | [Foundation](phases/phase_00_foundation/README.md) | Read theory: RLHF pipeline, RL concepts, reward models |
| 1 | [PPO](phases/phase_01_ppo/README.md) | Read theory + run: train reward model, run PPO, analyze results |
| 2 | [DPO](phases/phase_02_dpo/README.md) | Read theory + run: train DPO, compare with PPO |
| 3 | [GRPO](phases/phase_03_grpo/README.md) | Read theory + run: train GRPO, three-way comparison |
| 4 | [Capstone](phases/phase_04_capstone/README.md) | Run all three on a work task, write comparison report |

## Requirements

- Python 3.10+
- CUDA-capable GPU (A100/H100 recommended)
- ~20GB disk space for model weights and outputs

## Structure

- `phases/` — One directory per learning phase, each with a README (theory) and Python scripts (hands-on)
- `shared/` — Common utilities for model loading, data preparation, and evaluation
- `outputs/` — Created during training, stores models and results (git-ignored)
