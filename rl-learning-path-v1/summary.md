# PPO / DPO / GRPO Learning Path — Summary

**Learning Notes (Obsidian):**
- `PPO.md` — RL fundamentals, policy gradient, PPO theory, RLHF pipeline, reward hacking
- `DPO.md` — Closed-form insight, DPO objective, PPO vs DPO comparison
- `GRPO.md` — Group-relative advantage, DeepSeek-R1 connection, 3-way comparison
- `AlgorithmDecisionFramework.md` — Personal decision guide with trade-offs and red flags

**Training Scripts (`rlhf/`):**
- `train_reward_model.py` — TRL RewardTrainer on HH-RLHF (Week 2)
- `train_ppo.py` — PPO with KL penalty (Week 3)
- `train_ppo_no_kl.py` — PPO without KL, for reward hacking demo (Week 3)
- `train_dpo.py` — DPO on same dataset (Week 4)
- `compare_dpo_ppo.py` — Side-by-side output comparison (Week 4)
- `train_grpo.py` — GRPO on GSM8K with accuracy + format rewards (Week 5)
- `evaluate_grpo.py` — GSM8K accuracy evaluation (Week 6)

**Spec + Plan:**
- `docs/superpowers/specs/2026-04-21-ppo-dpo-grpo-learning-path-design.md`
- `docs/superpowers/plans/2026-04-21-ppo-dpo-grpo-learning-path.md`

**Next steps:** Follow the plan week by week. The notes are your study companions — fill them in as you read the papers. The scripts are ready to run on your cloud GPU. Start with Week 1 (Spinning Up + CleanRL CartPole), then work forward through the RLHF → DPO → GRPO progression.
