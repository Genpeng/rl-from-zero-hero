# RL Learning Path: PPO/DPO/GRPO Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Complete a 4-week learning journey through V2 (with optional V4 CartPole supplement) to achieve production-ready mastery of PPO, DPO, and GRPO for LLM alignment.

**Architecture:** Primary path is `rl-learning-path-v2/` — a 4-phase structured curriculum (Foundation → PPO → DPO → GRPO → Capstone). Each phase has a theory README and hands-on Python scripts using TRL on `Qwen/Qwen2.5-1.5B-Instruct`. Optional CartPole supplement from `rl-learning-path-v4/phase1_foundations/` provides from-scratch RL intuition before the LLM-specific phases.

**Tech Stack:** Python 3.10+, PyTorch, Hugging Face Transformers, TRL, PEFT, Accelerate, WandB, Gymnasium (for CartPole supplement)

---

### Task 1: Environment Setup

**Files:**
- Run: `rl-learning-path-v2/setup_env.sh`
- Check: `rl-learning-path-v2/requirements.txt`

- [ ] **Step 1: Navigate to the V2 learning path directory**

```bash
cd rl-learning-path-v2
```

- [ ] **Step 2: Run the environment setup script**

```bash
chmod +x setup_env.sh && bash setup_env.sh
```

Expected: Virtual environment created at `.venv/`, all packages installed (torch, transformers, trl, datasets, peft, accelerate, bitsandbytes, wandb, pyyaml, pandas, matplotlib).

- [ ] **Step 3: Activate the environment and verify GPU**

```bash
source .venv/bin/activate
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'Device: {torch.cuda.get_device_name(0)}') if torch.cuda.is_available() else None"
```

Expected: `CUDA available: True` and your GPU name printed. If False, you need to install CUDA-compatible PyTorch for your GPU driver.

- [ ] **Step 4: Verify TRL is installed and importable**

```bash
python3 -c "from trl import PPOTrainer, DPOTrainer; print('TRL OK')"
```

Expected: `TRL OK` with no import errors.

- [ ] **Step 5: (Optional) Set up WandB for training monitoring**

```bash
wandb login
```

Enter your WandB API key when prompted. This enables live training dashboards during Phases 1-4. If you don't have a WandB account, training will still work — logs go to stdout.

- [ ] **Step 6: Install Gymnasium for CartPole supplement (optional)**

Only if you plan to use the V4 CartPole notebooks:

```bash
pip install gymnasium[classic-control] jupyter
```

Expected: `gymnasium` and `jupyter` installed alongside the existing environment.

---

### Task 2: Phase 0 — Foundation Theory (Week 1, Days 1-5)

**Files:**
- Read: `rl-learning-path-v2/phases/phase_00_foundation/README.md`

- [ ] **Step 1: Read Day 1 — The Alignment Problem (~2 hours)**

Read: `phases/phase_00_foundation/README.md` — "Day 1: The Alignment Problem" section.

Key concepts to internalize:
- Why SFT alone isn't enough (the "helpful but harmful" problem, distribution mismatch, no preference signal)
- The 3-step RLHF pipeline: SFT → Reward Model → RL (PPO)
- The core insight: showing the model what humans *prefer* rather than what to *say*

Reading assignment: InstructGPT paper (Ouyang et al., 2022), Sections 1-3. Focus on Figure 2 (pipeline diagram) and the SFT vs RLHF results comparison. Skip the math formulations and ablation studies.

Paper: "Training language models to follow instructions with human feedback"

- [ ] **Step 2: Write a 3-sentence summary of Day 1**

In your own words, write down:
1. What problem does RLHF solve that SFT cannot?
2. What are the 3 steps of the RLHF pipeline?
3. What does the reward model learn?

Save this as `notes/day1_summary.md` in the V2 directory. Creating notes forces you to synthesize, which is the fastest way to internalize concepts.

```bash
mkdir -p notes
```

- [ ] **Step 3: Read Day 2 — Just-in-Time RL Concepts (~4 hours)**

Read: `phases/phase_00_foundation/README.md` — "Day 2: Just-in-Time RL Concepts" section.

Key concepts — map each RL term to its LLM equivalent:

| RL Term | LLM Equivalent |
|---------|---------------|
| Policy (π) | The language model itself |
| Action | Generating a single token |
| Reward (r) | Score from reward model for a complete response |
| Trajectory/Episode | Generating a complete response to a prompt |
| Policy Gradient | Nudge weights so high-reward responses become more likely: `∇J(θ) = E[∇log π(a|s) · R]` |
| KL Divergence | The "leash" — measures drift from SFT model; penalty: `R_adj = R - β · KL(π ‖ π_ref)` |

Reading assignments:
- Lilian Weng's blog post "What is RLHF" — comprehensive with diagrams
- Hugging Face RLHF blog post — shorter, more practical

- [ ] **Step 4: Read Day 3 — Reward Model Basics (~2 hours)**

Read: `phases/phase_00_foundation/README.md` — "Day 3: Reward Model Basics" section.

Key concepts:
- Preference data format: (prompt, chosen response, rejected response)
- Bradley-Terry model: `P(y_w > y_l) = σ(r(x, y_w) - r(x, y_l))` — it's logistic regression on score differences
- Loss: `L = -log σ(r(x, y_w) - r(x, y_l))` — binary cross-entropy pushing the score gap apart
- Why reward model quality is the bottleneck: RL amplifies its errors

Reading assignment: Revisit InstructGPT Section 3.2 (reward model training).

- [ ] **Step 5: Complete the Phase 0 Self-Check**

Answer these 3 questions out loud or in writing. If you can't answer clearly, re-read the relevant section.

1. **Pipeline:** "What are the three steps of RLHF, and what does each step produce?"
   - Target: SFT → capable base model; Reward Model → preference scorer; RL fine-tuning → aligned model

2. **KL Divergence:** "Why do we need a KL penalty in RLHF?"
   - Target: Prevents model from drifting too far from SFT and exploiting reward model weaknesses

3. **Reward Model:** "How is a reward model trained?"
   - Target: From human preference pairs using Bradley-Terry (logistic regression on score differences)

Save your answers to `notes/phase0_selfcheck.md`.

---

### Task 3: (Optional) CartPole Supplement — Build RL Intuition (Days 6-8)

**Files:**
- Run: `rl-learning-path-v4/phase1_foundations/01_reinforce_cartpole.ipynb`
- Run: `rl-learning-path-v4/phase1_foundations/02_gae_cartpole.ipynb`

**When to do this:** If the RL concepts from Phase 0 (policy gradient, advantage, KL divergence) still feel abstract, spend 2-3 days on these CartPole notebooks. They let you see RL working on a simple, visual problem before applying it to LLMs. If Phase 0 felt solid, skip this and go to Task 4.

- [ ] **Step 1: Open and run the REINFORCE notebook (~2-3 hours)**

```bash
cd ../rl-learning-path-v4/phase1_foundations
jupyter notebook 01_reinforce_cartpole.ipynb
```

Run each cell and observe:
- `PolicyNetwork`: a simple neural network that outputs action probabilities — this IS the "policy"
- `collect_trajectory`: the agent plays one complete episode — this IS the "trajectory"
- `compute_returns`: calculates discounted reward-to-go — this IS the return signal
- `reinforce_update`: `-log_prob * G` — this IS the policy gradient formula `∇J(θ) = E[∇log π(a|s) · R]`

Watch the reward plot: it should climb from ~20 to ~475 over ~500-1000 episodes. The noise and instability you see is exactly why PPO adds clipping.

- [ ] **Step 2: Open and run the GAE notebook (~2-3 hours)**

```bash
jupyter notebook 02_gae_cartpole.ipynb
```

Key observations:
- `ActorCritic`: now has TWO heads — actor (policy) and critic (value function). This is the precursor to PPO's 4-model setup.
- `compute_gae`: the GAE formula `δ_t = r_t + γV(s_{t+1}) - V(s_t)`, then `A_t = Σ (γλ)^l · δ_{t+l}`. λ controls bias-variance tradeoff.
- The λ comparison plot: watch how λ=0 (high bias, low variance) vs λ=1 (low bias, high variance) affect training stability.

- [ ] **Step 3: Write down your observations**

Save to `notes/cartpole_observations.md`:
1. What did you observe about training stability in REINFORCE vs GAE?
2. How did different λ values affect the learning curve?
3. How does the actor-critic setup relate to the 4-model PPO setup in LLMs?

```bash
cd ../../rl-learning-path-v2
```

Return to the V2 directory for the rest of the path.

---

### Task 4: Phase 1 — PPO Theory (Week 2, Days 6-8 or 9-11)

**Files:**
- Read: `rl-learning-path-v2/phases/phase_01_ppo/README.md`

- [ ] **Step 1: Read PPO Theory — The Clipping Mechanism (~3 hours)**

Read: `phases/phase_01_ppo/README.md` — "Theory: From Policy Gradient to PPO" section.

Key concepts to internalize:

1. **The problem PPO solves:** Vanilla policy gradient has no way to control update size → too large = collapse, too small = no learning

2. **The clipped objective:**
   ```
   L_CLIP = E[ min( r(θ) · A,  clip(r(θ), 1-ε, 1+ε) · A ) ]
   ```
   - `r(θ)` = probability ratio (new vs old policy)
   - `A` = advantage (how much better than average)
   - `ε` = clip range (typically 0.2 → policy changes at most 20% per step)
   - `min(...)` = pessimistic — prevents overly optimistic updates in both directions

3. **The 4-model setup for LLMs:**
   - Policy Model (π_θ): being trained
   - Reference Model (π_ref): frozen SFT copy for KL penalty
   - Reward Model (r_φ): scores responses
   - Value Model (V_ψ): critic for advantage estimation

4. **One PPO iteration:**
   GENERATE → SCORE → COMPUTE advantages → UPDATE policy → PENALTY (KL)

- [ ] **Step 2: Study the PPO config file**

Read: `phases/phase_01_ppo/configs/ppo_config.yaml`

Understand each parameter:
- Model: `Qwen/Qwen2.5-1.5B-Instruct` (small enough to train on a single GPU)
- Reward model: `Ray2333/GRM-Gemma-2B-sftreg`
- Key PPO params: `init_kl_coef: 0.05`, `clip_range: 0.2`, `ppo_epochs: 2`
- Dataset: `argilla/ultrafeedback-binarized-preferences-cleaned` (1000 samples)

- [ ] **Step 3: Read the PPO paper (targeted reading, ~1.5 hours)**

Read: Schulman et al. (2017), "Proximal Policy Optimization Algorithms", Sections 1-3.
Skip: Atari/MuJoCo experiments.
Focus on: the clipping formula, the comparison with TRPO, and the "why min?" explanation.

- [ ] **Step 4: Complete the Phase 1 Theory Self-Check**

Answer these 3 questions:

1. "What does the clipping in PPO prevent?"
   - Target: Prevents the policy from changing too much in a single update, avoiding catastrophic collapse

2. "Why does PPO for LLMs need 4 models? What does each one do?"
   - Target: Policy (being trained), Reference (KL anchor), Reward (scoring), Value (advantage estimation)

3. "What happens if you set the KL coefficient too low? Too high?"
   - Target: Too low → reward hacking; Too high → model barely learns

---

### Task 5: Phase 1 — PPO Hands-on Lab (Week 2, Days 9-12 or 12-15)

**Files:**
- Run: `rl-learning-path-v2/phases/phase_01_ppo/01_train_reward_model.py`
- Run: `rl-learning-path-v2/phases/phase_01_ppo/02_ppo_training.py`
- Run: `rl-learning-path-v2/phases/phase_01_ppo/03_ppo_analysis.py`
- Config: `rl-learning-path-v2/phases/phase_01_ppo/configs/ppo_config.yaml`

- [ ] **Step 1: Read the reward model training script**

```bash
cat phases/phase_01_ppo/01_train_reward_model.py
```

Read through the code. Identify:
- How preference data is loaded from `argilla/ultrafeedback-binarized-preferences-cleaned`
- How the reward model (`Ray2333/GRM-Gemma-2B-sftreg`) is used
- The Bradley-Terry loss function from Phase 0 theory

- [ ] **Step 2: Run the reward model training**

```bash
source .venv/bin/activate
python phases/phase_01_ppo/01_train_reward_model.py
```

Expected output: Training logs showing decreasing loss, reward accuracy improving. The reward model should converge within the configured training steps. Output saved to `outputs/phase_01_ppo/`.

- [ ] **Step 3: Read the PPO training script**

```bash
cat phases/phase_01_ppo/02_ppo_training.py
```

Map each section of the code to the PPO theory:
- Model loading → 4-model setup (policy, reference, reward, value)
- Generation loop → "GENERATE" step
- Reward scoring → "SCORE" step
- PPO update → clipped objective with advantages

- [ ] **Step 4: Run PPO training with default config**

```bash
python phases/phase_01_ppo/02_ppo_training.py
```

**What to watch during training:**
- ✓ Reward mean increasing → model learning to produce higher-reward responses
- ⚠️ KL divergence growing steadily → model drifting from SFT (watch for runaway)
- ✗ Reward up but output quality down → reward hacking (KL penalty too low)
- ✗ Reward flat → KL penalty too high or learning rate too low

- [ ] **Step 5: Run PPO analysis**

```bash
python phases/phase_01_ppo/03_ppo_analysis.py
```

Review the output: reward curves, KL divergence progression, sample generated outputs before/after training.

- [ ] **Step 6: Experiment with KL coefficients**

The config file lists 3 values to try: `0.01` (very loose), `0.05` (default), `0.2` (tight).

Edit `phases/phase_01_ppo/configs/ppo_config.yaml`, change `init_kl_coef` to `0.01`, and re-run:

```bash
python phases/phase_01_ppo/02_ppo_training.py
```

Then change to `0.2` and re-run. Compare:
- How fast does reward increase for each?
- Does KL divergence explode for `0.01`?
- Does the model barely learn with `0.2`?

Save observations to `notes/ppo_kl_experiment.md`.

- [ ] **Step 7: Reset config to default**

After experiments, set `init_kl_coef` back to `0.05` in the config file.

---

### Task 6: Phase 2 — DPO Theory (Week 3, Days 13-15)

**Files:**
- Read: `rl-learning-path-v2/phases/phase_02_dpo/README.md`

- [ ] **Step 1: Read DPO Theory — The Reward Model Elimination (~3 hours)**

Read: `phases/phase_02_dpo/README.md` — "Theory: The DPO Insight" section.

Key concepts:

1. **The DPO mathematical insight:**
   - RLHF objective has a closed-form solution: `π*(y|x) ∝ π_ref(y|x) · exp(r(x,y) / β)`
   - Rearrange to express reward in terms of policy: `r(x,y) = β · log(π(y|x) / π_ref(y|x)) + const`
   - Plug into Bradley-Terry → reward model disappears

2. **The DPO loss:**
   ```
   L_DPO = -E[ log σ( β · ( log π(y_w|x)/π_ref(y_w|x) - log π(y_l|x)/π_ref(y_l|x) ) ) ]
   ```
   Push log-ratios apart: make chosen more likely, rejected less likely.

3. **The simplification table:**
   - PPO: 4 models, online, complex training loop
   - DPO: 2 models, offline, simple forward pass → loss → update

4. **Trade-offs:**
   - DPO advantages: simpler, 2x less memory, more stable, fewer hyperparameters
   - DPO limitations: data quality dependency, offline only, verbosity bias, β sensitivity

- [ ] **Step 2: Read the DPO paper (targeted, ~1.5 hours)**

Read: Rafailov et al. (2023), "Direct Preference Optimization: Your Language Model is Secretly a Reward Model", Sections 1-4.
Focus on: the derivation chain (how the reward model gets eliminated) and the experimental comparison with PPO.

- [ ] **Step 3: Study the DPO config**

Read: `phases/phase_02_dpo/configs/dpo_config.yaml`

Note the differences from PPO config:
- Same model (`Qwen/Qwen2.5-1.5B-Instruct`) and dataset
- NO reward model needed
- Key param: `beta: 0.1` (analogous to KL coefficient)
- Much simpler config: just standard training params + β

---

### Task 7: Phase 2 — DPO Hands-on Lab (Week 3, Days 16-19)

**Files:**
- Run: `rl-learning-path-v2/phases/phase_02_dpo/01_dpo_training.py`
- Run: `rl-learning-path-v2/phases/phase_02_dpo/02_dpo_analysis.py`
- Config: `rl-learning-path-v2/phases/phase_02_dpo/configs/dpo_config.yaml`

- [ ] **Step 1: Read the DPO training script**

```bash
cat phases/phase_02_dpo/01_dpo_training.py
```

Compare with the PPO script. Notice:
- Only 2 models loaded (policy + reference) — no reward model, no value model
- No generation loop — trains directly on the preference dataset
- The DPO loss function: simple forward pass on chosen + rejected, compute log-ratio difference

- [ ] **Step 2: Run DPO training with default config**

```bash
python phases/phase_02_dpo/01_dpo_training.py
```

**What to watch:**
- ✓ Training loss decreasing → model learning preferences
- ✓ Chosen rewards > Rejected rewards → correct preference direction
- ✓ Rewards margin increasing → model separating good from bad more confidently
- ⚠️ Loss drops to near zero quickly → possible overfitting (try higher β)

- [ ] **Step 3: Run the DPO vs PPO comparison analysis**

```bash
python phases/phase_02_dpo/02_dpo_analysis.py
```

This script compares SFT baseline, PPO-trained, and DPO-trained models on the same evaluation prompts. Review:
- Quality comparison across the three models
- Training time and resource usage differences
- Sample outputs side-by-side

- [ ] **Step 4: Experiment with β values**

The config lists 3 values: `0.01` (aggressive), `0.1` (default), `0.5` (conservative).

Test `0.01` and `0.5` by editing `phases/phase_02_dpo/configs/dpo_config.yaml` and re-running training. Observe:
- Does `0.01` overfit to preferences?
- Does `0.5` barely change the model from SFT?

Save observations to `notes/dpo_beta_experiment.md`.

- [ ] **Step 5: Reset config and complete self-check**

Reset `beta` to `0.1`. Answer these questions:

1. "What does DPO eliminate compared to PPO, and how?"
   - Target: Eliminates the reward model by deriving optimal policy directly from preference data

2. "When would you choose DPO over PPO at work?"
   - Target: Clean preference data, limited GPU budget, simpler pipeline

3. "What does β control in DPO?"
   - Target: How aggressively the model deviates from reference — analogous to KL coefficient in PPO

Save answers to `notes/phase2_selfcheck.md`.

---

### Task 8: Phase 3 — GRPO Theory (Week 4, Days 20-22)

**Files:**
- Read: `rl-learning-path-v2/phases/phase_03_grpo/README.md`

- [ ] **Step 1: Read GRPO Theory — Group Relative Policy Optimization (~3 hours)**

Read: `phases/phase_03_grpo/README.md` — full theory section.

Key concepts:

1. **The problem GRPO solves:** PPO's value model (critic) is expensive and introduces instability. Can we estimate advantages without a learned critic?

2. **GRPO's "grade on a curve" approach:**
   ```
   For each prompt x:
     1. Generate G responses: {y_1, ..., y_G} (the "group")
     2. Score each with reward model: {r_1, ..., r_G}
     3. Normalize: A_i = (r_i - mean(r)) / std(r)
     4. Update with PPO-style clipped objective
     5. Add KL penalty against reference
   ```
   The group itself serves as the baseline — no value model needed.

3. **The 3-way comparison:**
   - PPO: 4 models, online, full RL
   - DPO: 2 models, offline, no RL loop
   - GRPO: 3 models (Policy, Ref, RM — no critic), online, group-based advantages

4. **When GRPO shines:** Verifiable rewards — math (check answer), code (run tests), format compliance (check template).

5. **DeepSeek-R1 connection:** Trained with GRPO + rule-based rewards for mathematical reasoning.

- [ ] **Step 2: Read the DeepSeekMath paper (targeted, ~1 hour)**

Read: Shao et al. (2024), "DeepSeekMath: Pushing the Limits of Mathematical Reasoning", Section 3 on GRPO.
Also skim the DeepSeek-R1 technical report for how GRPO was used at scale.

- [ ] **Step 3: Study the GRPO config**

Read: `phases/phase_03_grpo/configs/grpo_config.yaml`

Key differences from PPO/DPO:
- `num_generations: 8` — group size (G), the defining GRPO parameter
- `temperature: 0.8` — needs diversity within the group for useful advantages
- Still uses a reward model (`Ray2333/GRM-Gemma-2B-sftreg`)
- `max_samples: 500` — fewer samples because each prompt generates G responses

---

### Task 9: Phase 3 — GRPO Hands-on Lab (Week 4, Days 23-25)

**Files:**
- Run: `rl-learning-path-v2/phases/phase_03_grpo/01_grpo_training.py`
- Run: `rl-learning-path-v2/phases/phase_03_grpo/02_grpo_analysis.py`
- Config: `rl-learning-path-v2/phases/phase_03_grpo/configs/grpo_config.yaml`

- [ ] **Step 1: Read the GRPO training script**

```bash
cat phases/phase_03_grpo/01_grpo_training.py
```

Identify:
- How the group generation works (G responses per prompt)
- How advantages are computed via group normalization (mean/std)
- How the PPO-style clipped objective is applied with these advantages

- [ ] **Step 2: Run GRPO training with default config (G=8)**

```bash
python phases/phase_03_grpo/01_grpo_training.py
```

**What to watch:**
- Group reward variance → if all G responses score the same, advantages ≈ 0 (no learning signal). Increase temperature or group size.
- Training cost → G=8 means 8x more generation compute per prompt than PPO
- Compare training speed/stability with PPO (Task 5)

- [ ] **Step 3: Experiment with group sizes**

The config lists: `4` (faster, noisier), `8` (default), `16` (better estimates, more compute).

Test G=4 and G=16 by editing `num_generations` in `phases/phase_03_grpo/configs/grpo_config.yaml` and re-running. Observe:
- How does group size affect advantage estimate quality?
- What's the compute tradeoff?
- Is G=16 noticeably better than G=8?

Save observations to `notes/grpo_group_size_experiment.md`.

- [ ] **Step 4: Run the 3-way comparison analysis**

```bash
python phases/phase_03_grpo/02_grpo_analysis.py
```

This compares PPO vs DPO vs GRPO results. Review:
- Quality comparison on evaluation prompts
- Training cost comparison (time, memory, compute)
- Sample outputs from each algorithm

- [ ] **Step 5: Reset config and complete self-check**

Reset `num_generations` to `8`. Answer these questions:

1. "How does GRPO estimate advantages without a value model?"
   - Target: Generates multiple responses per prompt, normalizes rewards within the group

2. "Where does GRPO sit between PPO and DPO?"
   - Target: Online like PPO, simpler like DPO (no critic), but still needs a reward model

3. "When is GRPO the best choice?"
   - Target: Verifiable/rule-based rewards (math, code) and want online RL without critic complexity

Save answers to `notes/phase3_selfcheck.md`.

---

### Task 10: Phase 4 — Capstone Experiment (Week 4, Days 25-28)

**Files:**
- Run: `rl-learning-path-v2/phases/phase_04_capstone/01_run_all_experiments.py`
- Run: `rl-learning-path-v2/phases/phase_04_capstone/02_evaluate_and_compare.py`
- Run: `rl-learning-path-v2/phases/phase_04_capstone/03_write_report.py`
- Config: `rl-learning-path-v2/phases/phase_04_capstone/configs/capstone_config.yaml`

- [ ] **Step 1: Review the capstone config**

Read: `phases/phase_04_capstone/configs/capstone_config.yaml`

This config uses **matched settings** across all 3 algorithms:
- Same model: `Qwen/Qwen2.5-1.5B-Instruct`
- Same dataset: `argilla/ultrafeedback-binarized-preferences-cleaned` (1000 samples)
- Same reward model: `Ray2333/GRM-Gemma-2B-sftreg`
- Matched hyperparameters: `learning_rate: 5e-6`, `num_train_epochs: 1`, `gradient_accumulation_steps: 4`

This ensures a fair comparison — only the algorithm differs.

- [ ] **Step 2: Run all three experiments under matched conditions**

```bash
python phases/phase_04_capstone/01_run_all_experiments.py
```

This trains PPO, DPO, and GRPO sequentially with the matched config. Expected: takes significant time (potentially hours depending on GPU). Outputs saved to `outputs/phase_04_capstone/`.

- [ ] **Step 3: Run evaluation and comparison**

```bash
python phases/phase_04_capstone/02_evaluate_and_compare.py
```

This generates comparison tables and evaluation results. Review:
- Reward model scores across all 3 algorithms
- Response length distributions
- Sample outputs for the same prompts across all 3

- [ ] **Step 4: Generate the report skeleton**

```bash
python phases/phase_04_capstone/03_write_report.py
```

This auto-generates a report skeleton from results in `outputs/phase_04_capstone/`.

- [ ] **Step 5: Write your personal decision framework**

Create `notes/algorithm_decision_framework.md` with this structure:

```markdown
# My Algorithm Decision Framework

## When to use PPO
- [Your findings from experiments]
- [Specific scenarios where PPO won]

## When to use DPO
- [Your findings from experiments]
- [Specific scenarios where DPO won]

## When to use GRPO
- [Your findings from experiments]
- [Specific scenarios where GRPO won]

## Quick Decision Guide
Given task X with constraints Y, I would choose algorithm Z because...
```

Fill in based on your experimental results and Phase 1-3 theory. This is your main deliverable — the document you'll reference when choosing an algorithm at work.

- [ ] **Step 6: Compile lessons learned**

Add a `notes/lessons_learned.md` documenting:
- Hyperparameter gotchas you encountered (KL coefficient, β, group size)
- Failure modes observed (reward hacking, overfitting, flat rewards)
- Practical tips for each algorithm
- Resource requirements comparison

---

### Task 11: Final Mastery Validation

- [ ] **Step 1: Verify all deliverables exist**

Check that you have:

```bash
ls notes/
```

Expected files:
- `day1_summary.md` — Foundation Day 1 synthesis
- `phase0_selfcheck.md` — Foundation self-check answers
- `cartpole_observations.md` — (optional) CartPole supplement observations
- `ppo_kl_experiment.md` — PPO KL coefficient experiments
- `dpo_beta_experiment.md` — DPO β experiments
- `grpo_group_size_experiment.md` — GRPO group size experiments
- `phase2_selfcheck.md` — DPO self-check answers
- `phase3_selfcheck.md` — GRPO self-check answers
- `algorithm_decision_framework.md` — Personal decision guide
- `lessons_learned.md` — Practical tips and gotchas

```bash
ls outputs/
```

Expected: `phase_01_ppo/`, `phase_02_dpo/`, `phase_03_grpo/`, `phase_04_capstone/` with trained model outputs and evaluation results.

- [ ] **Step 2: Final self-assessment**

Answer these 5 questions that cover the complete learning path:

1. "Explain the RLHF pipeline to a colleague who only knows SFT." (Phase 0)
2. "What makes PPO stable but expensive?" (Phase 1)
3. "How does DPO eliminate the reward model, and what's the tradeoff?" (Phase 2)
4. "When would you pick GRPO over DPO and PPO?" (Phase 3)
5. "A colleague asks you to align a 7B model for instruction following. You have clean preference data and 2x A100s. Which algorithm do you recommend and why?" (Capstone)

If you can answer all 5 confidently with specifics from your experiments, you've achieved the mastery goal defined in the spec.
