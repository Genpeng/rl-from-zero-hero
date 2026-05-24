# Master Comparison Card: PPO vs DPO vs GRPO

## 1. Algorithm Summary Table

| Aspect                  | PPO                          | DPO                           | GRPO                          |
|-------------------------|------------------------------|-------------------------------|-------------------------------|
| Models needed           | 4 (policy, ref, reward, critic) | 2 (policy, ref)            | 2 (policy, ref)               |
| Memory cost             | High (4x model weights)     | Low (2x model weights)       | Medium (2x + G rollouts)     |
| Training data           | Prompts + reward fn         | Offline preference pairs     | Prompts + verifiable reward  |
| Online / Offline        | Online                      | Offline                      | Online                       |
| Advantage estimation    | GAE from critic network     | Implicit (log-ratio gap)     | Group normalization of rewards|
| Key hyperparams         | eps, KL coeff, GAE lambda   | beta                         | G (group size), eps          |
| Training stability      | Moderate (many moving parts)| High (supervised-style)      | High (no critic drift)       |
| Implementation complexity| Hard                        | Easy                         | Medium                       |
| Best use case           | General RLHF at scale       | Limited GPU, clean pref data | Math/code with reward signal |

## 2. Decision Flowchart

```
START
  |
  v
Do you have verifiable rewards (math/code)?
  |YES                        |NO
  v                           v
 GRPO                  Do you have clean preference data?
                         |YES                    |NO
                         v                       v
                  Limited GPU budget?         PPO
                   |YES       |NO         (reward model
                   v          v            generalizes)
                  DPO    Need online
                         exploration?
                          |YES    |NO
                          v       v
                         PPO    Is data noisy?
                                 |YES    |NO
                                 v       v
                                PPO     DPO
                                      (simplest)
```

## 3. Key Formulas Side-by-Side

**PPO:**  `L = E[ min( r(t)*A,  clip(r(t), 1-e, 1+e)*A ) ]`  where `r(t) = pi_new / pi_old`

**DPO:**  `L = -E[ log s( b * ( log pi(yw|x)/pi_ref(yw|x) - log pi(yl|x)/pi_ref(yl|x) ) ) ]`

**GRPO:** Same clipped objective as PPO, but `A_i = (r_i - mean(r)) / std(r)` from a group of G sampled responses. No critic network.

## 4. Common Failure Modes

| Algorithm | Failure              | Cause                  | Fix                            |
|-----------|----------------------|------------------------|--------------------------------|
| PPO       | Reward hacking       | KL penalty too low     | Raise KL coeff                 |
| PPO       | Training collapse    | LR too high            | Lower LR, warm up longer       |
| PPO       | OOM                  | 4 models in memory     | Offload ref/reward, use LoRA   |
| DPO       | Overfitting          | beta too low            | Raise beta (0.1-0.5)          |
| DPO       | No improvement       | beta too high           | Lower beta                    |
| DPO       | Verbosity bias       | Longer = preferred     | Length-normalized reward       |
| GRPO      | Zero advantages      | Group variance too low | Increase G, raise temperature  |
| GRPO      | Compute explosion    | G too large            | Reduce G, use vLLM batching   |

## 5. Hyperparameter Quick Reference

| Param             | Algorithm | Default    | Too low             | Too high             |
|-------------------|-----------|------------|----------------------|----------------------|
| epsilon (clip)    | PPO/GRPO  | 0.2        | Slow learning        | Unstable updates     |
| KL coeff          | PPO       | 0.02       | Reward hacking       | Underfitting         |
| GAE lambda        | PPO       | 0.95       | High bias            | High variance        |
| beta              | DPO       | 0.1        | Overfitting          | No learning signal   |
| G (group size)    | GRPO      | 8-16       | Noisy advantages     | Compute blowup       |
| Temperature       | GRPO      | 0.7-1.0    | Low diversity        | Noisy samples        |
| Learning rate     | All       | 1e-6-5e-7  | No progress          | Training collapse    |

## 6. If I Only Remember One Thing

| PPO  | Stable RL with clipping, but expensive (4 models).                  |
|------|---------------------------------------------------------------------|
| DPO  | Direct training on preferences, no reward model needed.             |
| GRPO | Grade on a curve -- group responses replace the critic.             |
