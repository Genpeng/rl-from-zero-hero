# Design Spec: RL Learning Path Selection for PPO/DPO/GRPO

**Date:** 2026-04-23  
**Status:** Approved

---

## 1. Learner Profile

| Attribute | Value |
|-----------|-------|
| NLP experience | Beginner |
| RL background | None |
| Math level | Basic calculus + linear algebra; can follow formulas if explained step-by-step |
| Code level | PyTorch + Hugging Face for inference/SFT; no RL training experience |
| Time budget | 1-2 hours/day, 4 weeks (~28-56 total hours) |
| End goal | Production-ready skills — fine-tune LLMs at work using TRL and similar frameworks |

## 2. Problem Statement

Five pre-built learning paths (V1–V5) exist in the project directory. Each covers PPO, DPO, and GRPO but differs in structure, depth, timeline, and pedagogical approach. The learner needs a clear recommendation on which path to follow, with a concrete weekly plan.

## 3. Path Evaluation Summary

| Path | Timeline | Theory Depth | Hands-on Style | Fit |
|------|----------|-------------|----------------|-----|
| V1 | 6 weeks | High (markdown notes) | TRL scripts, unstructured | Poor — too long, no phase structure |
| **V2** | 3-4 weeks | Medium-High (READMEs) | TRL + comparison experiments | **Best fit** |
| V3 | 14 days | Low (code-is-textbook) | Code-primary, dense | Poor — too dense for RL beginners |
| V4 | 24 days | Medium (notebooks) | Jupyter + production pipeline | Good alternative |
| V5 | 21 days | Very High (full derivations) | From-scratch + TRL | Poor — math exceeds comfort zone |

### Why V2 Wins

1. **Production-aligned:** TRL-first approach matches the work goal directly
2. **Right theory level:** READMEs build intuition without requiring full mathematical derivations
3. **Beginner-friendly:** Phase 0 provides just-in-time RL concepts for people with zero RL background
4. **Timeline match:** 3-4 weeks fits the 4-week budget with buffer
5. **SFT bridge:** Assumes SFT experience and builds from there

### Why Not the Others

- **V1:** 6-week timeline exceeds budget; no formal phase structure makes self-pacing difficult
- **V3:** Code-as-textbook approach assumes prior RL familiarity; theory is too minimal for a beginner to build intuition
- **V4:** Strong alternative but notebook-heavy format is less production-ready; 24 days at 1-2 hrs/day may feel rushed
- **V5:** Full mathematical derivations (Days 1-2 are pure theory) will slow down a learner with basic math background

## 4. Selected Approach: V2 as Primary + V4 CartPole Supplement

### Core Path: V2 (rl-learning-path-v2/)

Structured 4-phase progression with clear success criteria at each stage.

### Optional Supplement: V4 Phase 1 CartPole Notebooks

V2 lacks classic RL examples. If the jump from zero RL to LLM PPO is too steep, V4's Phase 1 CartPole notebooks (REINFORCE + GAE) serve as a 2-3 day warm-up.

## 5. Weekly Plan

### Week 1: Foundation (V2 Phase 0) — Days 1-5

**Goal:** Build RL intuition from NLP/SFT background

**Content:**
- RLHF motivation: why SFT alone isn't enough
- Core RL concepts: policy, action, reward, value, advantage
- Bradley-Terry preference model
- KL divergence intuition (why it matters for alignment)
- RLHF pipeline overview (reward model → PPO → aligned model)

**Activities:**
- Read Phase 0 README thoroughly
- Run any provided demo scripts
- Answer self-check questions at the end of the phase

**Success Criteria:**
- Can explain the RLHF pipeline end-to-end
- Can articulate why KL divergence is needed
- Understands what a reward model does and why

**Time:** ~1.5 hrs/day × 5 days = ~7.5 hours

### Week 2: PPO (V2 Phase 1) — Days 6-12

**Goal:** Understand and run PPO training on a small LLM

**Prerequisite Check:** If the RL concepts from Week 1 feel shaky, insert V4 Phase 1 CartPole notebooks (2-3 days) here before proceeding. This will push the schedule by ~3 days but build much stronger intuition.

**Content:**
- PPO clipping mechanism (why and how it stabilizes training)
- 4-model setup: policy model, reference model, reward model, value model
- KL penalty in practice
- Advantage estimation and GAE
- Training loop walkthrough

**Activities:**
- Read Phase 1 README
- Run PPO training scripts with provided configs
- Modify hyperparameters (KL coefficient, clip range) and observe effects
- Answer self-check questions

**Success Criteria:**
- Successfully run PPO training on a small model
- Can explain each step in the PPO training loop
- Understands the role of each of the 4 models
- Can describe what happens when KL penalty is removed (reward hacking)

**Time:** ~1.5 hrs/day × 7 days = ~10.5 hours

### Week 3: DPO (V2 Phase 2) — Days 13-19

**Goal:** Understand DPO as a simpler alternative, run DPO vs PPO comparison

**Content:**
- DPO derivation (intuitive level): how it eliminates the reward model
- The closed-form solution insight
- Beta (β) parameter and its role
- DPO vs PPO: when to use which
- Training on preference data directly

**Activities:**
- Read Phase 2 README
- Run DPO training scripts
- Run the DPO vs PPO comparison experiment
- Analyze results: quality, training time, resource usage

**Success Criteria:**
- Successfully run DPO training
- Can explain how DPO eliminates the reward model
- Can articulate trade-offs between DPO and PPO
- Has run a direct comparison and can interpret the results

**Time:** ~1.5 hrs/day × 7 days = ~10.5 hours

### Week 4: GRPO + Capstone (V2 Phases 3-4) — Days 20-28

**Goal:** Learn GRPO, then run a 3-way comparison and build a decision framework

**Content (Phase 3 — Days 20-24):**
- GRPO group normalization: how it simplifies PPO
- Verifiable rewards vs learned rewards
- DeepSeek-R1 connection
- GRPO vs PPO vs DPO comparison

**Content (Phase 4 — Days 25-28):**
- Capstone: run all 3 algorithms on the same model and dataset
- Build a personal algorithm decision framework
- Document findings and trade-offs

**Activities:**
- Read Phase 3 README, run GRPO training scripts
- Run 3-way comparison experiment (capstone)
- Write personal decision framework: "Given task X, use algorithm Y because Z"

**Success Criteria:**
- Successfully run GRPO training
- Completed 3-way comparison experiment with documented results
- Has a written decision framework for choosing PPO vs DPO vs GRPO
- Can recommend and justify an algorithm choice for a given real-world scenario

**Time:** ~1.5 hrs/day × 9 days = ~13.5 hours

## 6. Total Time Estimate

| Week | Phase | Days | Hours |
|------|-------|------|-------|
| 1 | Foundation | 5 | ~7.5 |
| 2 | PPO | 7 | ~10.5 |
| 3 | DPO | 7 | ~10.5 |
| 4 | GRPO + Capstone | 9 | ~13.5 |
| **Total** | | **28** | **~42 hours** |

42 hours fits comfortably within the 28-56 hour budget (1-2 hrs/day × 28 days).

If the CartPole supplement is added in Week 2, add ~4-5 hours and extend the timeline by 2-3 days.

## 7. Resources Required

- **GPU access:** A100 or equivalent for Phases 1-4 (PPO/DPO/GRPO training)
- **Software:** Python 3.10+, PyTorch, Hugging Face Transformers, TRL, datasets
- **Environment setup:** V2 provides `setup_env.sh` and `requirements.txt`
- **Models:** Small LLMs (e.g., GPT-2 or similar) for training experiments

## 8. Risk Mitigation

| Risk | Mitigation |
|------|------------|
| Week 2 PPO feels too abstract without RL basics | Insert V4 Phase 1 CartPole notebooks as warm-up |
| Training scripts fail due to GPU/environment issues | V2 provides setup_env.sh; run environment setup on Day 1 |
| Math in READMEs feels too hard | V2 is already pitched at intuitive level; skip derivation details and focus on the "what it means" summaries |
| 4-week timeline too tight | Capstone (Phase 4) can extend into Week 5 if needed; the core algorithms are covered by end of Week 4 Day 24 |

## 9. Definition of Done

The learner has achieved "mastery" when all of the following are true:

1. **Theory:** Can explain PPO, DPO, and GRPO at an intuitive level — what each does, how they differ, when to use which
2. **Hands-on:** Has successfully run training scripts for all 3 algorithms using TRL
3. **Comparison:** Has completed a 3-way comparison experiment with documented results
4. **Decision-making:** Has a written decision framework for algorithm selection
5. **Production readiness:** Can configure and launch a TRL training run for a new task at work
