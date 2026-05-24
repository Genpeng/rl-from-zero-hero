# Minimal Learning Path: PPO, DPO, GRPO for LLM Alignment

## Context

- **Learner profile**: NLP practitioner, experienced with Hugging Face Transformers fine-tuning, basic linear algebra & calculus, no RL background
- **Goal**: Apply RLHF/DPO/GRPO to align LLMs at work
- **Timeline**: 3-4 weeks part-time (~3 hours/day)
- **Resources**: Cloud A100/H100 GPUs available
- **Approach**: Top-down, LLM-centric — learn RL concepts just-in-time in the LLM context, not as general RL theory

## Overall Structure

| Phase | Topic | Duration | Focus |
|-------|-------|----------|-------|
| 0 | Foundation: RLHF Big Picture + Just-in-Time RL Theory | 2-3 days | Theory |
| 1 | PPO in the RLHF Pipeline | 4-5 days | Theory + Hands-on |
| 2 | DPO — Removing the Reward Model | 3-4 days | Theory + Hands-on |
| 3 | GRPO — Group-Based Optimization | 3-4 days | Theory + Hands-on |
| 4 | Capstone: Comparative Experiment | 3-4 days | Pure Hands-on |

Consistent model across all phases: Qwen2.5-1.5B (small enough to iterate fast, large enough to show real behavior). All hands-on labs use TRL (Hugging Face's Transformer Reinforcement Learning library).

---

## Phase 0: Foundation (Days 1-3)

**Goal**: Understand why RLHF exists and learn the minimal RL concepts needed to follow PPO.

### Theory

#### The Alignment Problem (~2 hours)

- Why SFT alone isn't enough: the "helpful but harmful" failure mode
- The RLHF pipeline at a high level: SFT → Reward Model → RL Fine-tuning
- Key reading: InstructGPT paper (Ouyang et al., 2022) — Sections 1-3 (skip the math, read the pipeline diagrams and results)

#### Just-in-Time RL Concepts (~4 hours)

Only these concepts, explained in LLM terms:

- **Policy**: the LLM itself (maps input → output distribution)
- **Action**: generating a token
- **Reward**: score from a reward model (or human preference)
- **Trajectory/Episode**: generating a full response
- **Policy Gradient**: increase probability of actions that got high reward
- **KL Divergence**: a "distance" between two distributions — keeps RL-tuned model close to SFT model (prevents reward hacking)

Key reading: Lilian Weng's blog post "What is RLHF"

#### Reward Model Basics (~2 hours)

- How preference data (chosen vs rejected) trains a reward model
- Bradley-Terry model: logistic regression on preference pairs
- Why reward model quality is the bottleneck of RLHF

### Success Criteria

- Can explain the full RLHF pipeline to a colleague in 2 minutes
- Understands KL divergence intuitively (not the formula derivation)
- Knows what a reward model does and how it's trained

---

## Phase 1: PPO in the RLHF Pipeline (Days 4-8)

**Goal**: Understand how PPO works in LLM alignment and run a full RLHF training loop.

### Theory (Day 4-5, ~6 hours)

#### From Policy Gradient to PPO (~3 hours)

- Vanilla policy gradient problem: updates are unstable, one bad step can destroy the model
- The core PPO idea: clip the update so the policy can't change too much in one step
- The clipped objective: `L = min(r(θ)·A, clip(r(θ), 1-ε, 1+ε)·A)`
  - `r(θ)` = probability ratio (new policy / old policy)
  - `A` = advantage (how much better was this action than average)
  - `ε` = clip range (typically 0.2)
- Intuition: "if an action was good, increase its probability, but not by too much"
- Key reading: PPO paper (Schulman et al., 2017) — Sections 1-3

#### PPO Adapted for LLMs (~3 hours)

- The 4-model RLHF setup: Policy (being trained), Reference Policy (frozen SFT copy), Reward Model, Value Model (critic)
- Generation phase → Scoring phase → Update phase (one PPO iteration)
- KL penalty added to reward to prevent drift from reference
- Why PPO for LLMs is expensive: 4 models in memory, generation is slow
- Key reading: TRL library documentation on `PPOTrainer`

### Hands-on Lab (Day 6-8, ~9 hours)

**Project**: Use TRL's `PPOTrainer` to align Qwen2.5-1.5B for a specific task (e.g., controlled sentiment generation or helpfulness).

- **Day 6**: Environment setup + data preparation
  - Install TRL, prepare preference dataset, train or load a reward model

- **Day 7**: Run PPO training
  - Configure hyperparameters: learning rate, KL coefficient, clip range, batch size
  - Monitor: reward mean, KL divergence, policy loss
  - Observe: what happens when KL penalty is too low (reward hacking) vs too high (no learning)

- **Day 8**: Analysis + debugging
  - Compare outputs: SFT model vs PPO-aligned model
  - Experiment: change KL coefficient and observe behavior
  - Document lessons learned

### Success Criteria

- Can explain PPO clipping and why it matters in plain language
- Can explain why RLHF needs 4 models and what each one does
- Has run a complete PPO training loop and can interpret training metrics
- Understands KL coefficient tuning in practice

---

## Phase 2: DPO — Removing the Reward Model (Days 9-12)

**Goal**: Understand how DPO simplifies RLHF by eliminating the reward model, and run a DPO training loop.

### Theory (Day 9-10, ~5 hours)

#### The Key Insight of DPO (~2.5 hours)

- The problem with PPO-based RLHF: expensive (4 models), unstable (reward hacking, hyperparameter-sensitive), complex engineering
- DPO's breakthrough: derive the optimal policy directly from preference data — no reward model needed
- The derivation intuition (no need to reproduce the math):
  - RLHF objective has a closed-form solution: `π*(y|x) ∝ π_ref(y|x) · exp(r(x,y) / β)`
  - Rearrange to express reward in terms of the policy
  - Plug back into Bradley-Terry → reward model disappears
- The DPO loss: `L = -log σ(β · (log π(y_w|x)/π_ref(y_w|x) - log π(y_l|x)/π_ref(y_l|x)))`
  - `y_w` = preferred response, `y_l` = rejected response
  - Intuition: increase the gap between preferred and rejected response log-ratios
- Key reading: DPO paper (Rafailov et al., 2023) — Sections 1-4

#### DPO vs PPO: Trade-offs (~2.5 hours)

- **DPO advantages**: only 2 models (policy + reference), no reward model training, simpler, more stable
- **DPO limitations**:
  - Relies entirely on data quality — no reward model to generalize
  - Offline algorithm: optimizes against fixed dataset, can't explore
  - `β` parameter is critical: too low → overfits, too high → barely learns
  - Can suffer from verbosity bias
- When to use DPO: good preference data, limited compute, want simplicity
- When PPO still wins: noisy data, need online exploration, have compute budget
- Key reading: comparison studies on DPO vs PPO for LLM alignment

### Hands-on Lab (Day 11-12, ~6 hours)

**Project**: Use TRL's `DPOTrainer` to align the same Qwen2.5-1.5B on the same task as Phase 1.

- **Day 11**: DPO training
  - Prepare data in (prompt, chosen, rejected) format
  - Configure `DPOTrainer`: `β`, learning rate, batch size
  - Monitor: loss curve, chosen/rejected reward margins, preference accuracy

- **Day 12**: Comparison with PPO
  - Compare outputs: PPO-aligned vs DPO-aligned vs SFT baseline
  - Compare: wall-clock time, GPU memory, stability
  - Experiment: vary `β` (parallel to KL coefficient in PPO)

### Success Criteria

- Can explain DPO in one sentence: "DPO skips the reward model by expressing the optimal policy directly as a function of preference data"
- Can articulate when to choose DPO vs PPO for a real project
- Has trained a DPO model and compared against PPO
- Understands `β` and has experimented with it

---

## Phase 3: GRPO — Group-Based Optimization (Days 13-16)

**Goal**: Understand how GRPO eliminates the value model from PPO while keeping online RL benefits, and run a GRPO training loop.

### Theory (Day 13-14, ~5 hours)

#### The Problem GRPO Solves (~2 hours)

- Recall PPO's 4-model burden — the value model (critic) is expensive and adds instability
- GRPO's idea (from DeepSeek): estimate advantages from the group, not from a critic
- Key paper: DeepSeekMath (Shao et al., 2024) — Section 3 on GRPO

#### How GRPO Works (~3 hours)

- For each prompt, generate **G responses** (a group, e.g., G=8 or 16)
- Score all G responses with a reward model (or rule-based reward)
- Compute advantage by normalizing within the group: `A_i = (r_i - mean(r)) / std(r)`
  - Above group average → positive advantage → reinforce
  - Below average → negative advantage → suppress
- Apply PPO-style clipped update using group-normalized advantages
- Still uses KL penalty against reference policy
- Key difference from PPO: no value model, advantages come from peer comparison
- Key difference from DPO: still online, still uses a reward model
- Intuition: "grade on a curve"

#### GRPO in the Algorithm Landscape

- Sits between PPO and DPO:
  - Like PPO: online, uses reward model, explores
  - Like DPO: simpler than full PPO (no critic)
- Particularly well-suited for tasks with verifiable rewards (math, code)
- This is how DeepSeek-R1 was trained — GRPO + rule-based rewards

### Hands-on Lab (Day 15-16, ~6 hours)

**Project**: Use TRL's `GRPOTrainer` to align the same Qwen2.5-1.5B, completing the three-way comparison.

- **Day 15**: GRPO training
  - Configure: group size G, reward function, KL coefficient, clip range
  - Monitor: reward distribution across groups, advantage statistics, KL divergence
  - Experiment with group size: G=4 vs G=8 vs G=16

- **Day 16**: Three-way analysis
  - Compare PPO vs DPO vs GRPO on same model/task
  - Dimensions: output quality, training time, memory usage, stability
  - Try GRPO with rule-based reward (e.g., length constraint, format compliance)

### Success Criteria

- Can explain GRPO's advantage estimation: "generate a group, normalize rewards within the group, no critic needed"
- Understands where GRPO sits relative to PPO and DPO
- Has run GRPO training and experimented with group size
- Has all three algorithms' results ready for comparison

---

## Phase 4: Capstone — Comparative Experiment (Days 17-20)

**Goal**: Consolidate understanding through a structured comparison presentable at work.

### Project Design (~1 day)

- Pick a work-relevant task (e.g., instruction following, summarization quality, safety alignment)
- Design controlled experiment: same base model, same dataset, comparable compute budget, clear evaluation metrics

### Execution (~2 days)

Train all three algorithms and build a comparison table:

| Dimension | PPO | DPO | GRPO |
|-----------|-----|-----|------|
| Models in memory | 4 | 2 | 3 |
| Needs reward model | Yes | No | Yes (or rule-based) |
| Online / Offline | Online | Offline | Online |
| Training stability | Sensitive | Stable | Moderate |
| Data requirement | RM + prompts | Preference pairs | Reward fn + prompts |
| Best suited for | Complex alignment with exploration | Clean preference data, limited compute | Verifiable rewards (math, code) |

### Deliverable (~1 day)

Write a short internal report or notebook:

- Decision framework: given task X with constraints Y, choose algorithm Z
- Practical tips and hyperparameter gotchas
- Recommended defaults for team use cases
- Reusable training scripts

### Success Criteria

- Has a decision framework for algorithm selection
- Can present findings to teammates with confidence
- Has reusable training scripts for all three algorithms

---

## Key Resources

### Papers (Read in This Order)

1. InstructGPT (Ouyang et al., 2022) — the RLHF pipeline origin
2. PPO (Schulman et al., 2017) — the RL algorithm
3. DPO (Rafailov et al., 2023) — the simplification
4. DeepSeekMath / GRPO (Shao et al., 2024) — the latest evolution

### Blog Posts & Tutorials

- Lilian Weng: "What is RLHF" — best accessible overview
- Hugging Face blog: "RLHF with TRL" — practical guide
- TRL documentation: `PPOTrainer`, `DPOTrainer`, `GRPOTrainer` API references

### Tools

- **TRL** (Transformer Reinforcement Learning): primary training framework, integrates with HF ecosystem
- **Weights & Biases** or **TensorBoard**: training metric visualization
- **Base model**: Qwen2.5-1.5B (consistent across all phases)
