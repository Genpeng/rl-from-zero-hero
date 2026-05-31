# LLM 中最重要的强化学习概念

对 LLM 来说，强化学习（RL）的核心不是“怎么让智能体在 Atari 里玩游戏”，而是：

> 如何把语言生成看成一个序列决策问题，并用 reward 信号去优化 policy。

下面这些概念可以按重要程度来理解。

## Cheatsheet：核心概念对比表

| 概念 | 一句话理解 | 在 LLM 中的对应 | 主要作用 | 容易混淆或注意点 | 相关方法 / 关键词 |
| --- | --- | --- | --- | --- | --- |
| Policy | 在某个状态下选择动作的概率分布 | LLM 的 next-token distribution | 决定模型会生成什么 token 和回答 | LLM 不只是文本模型，也可以看成逐 token 决策的 stochastic policy | RLHF、RLAIF、RLVR、GRPO |
| Reward | 判断结果好坏的分数 | 对完整 response 的质量评分 | 给 policy 提供优化方向 | LLM reward 通常是 sequence-level，不是每个 token 都有即时 reward | reward model、verifier、human preference |
| Trajectory | 一次完整的状态-动作序列 | 从 prompt 到完整 response 的生成过程 | RL 优化的基本样本单位 | 在 LLM 中通常就是一条 completion，而不是游戏里的多步环境交互记录 | rollout、completion、episode |
| Return | 从当前时刻开始累积的总收益 | 一整段回答最终得到的总分 | 表示整段生成的优化目标 | LLM 场景常可近似为最终 response reward，但会带来 credit assignment 问题 | outcome reward、discounted return |
| Advantage | 当前动作比平均预期好多少 | 某个回答比同状态下平均回答好多少 | 降低 policy gradient 方差，让更新更稳定 | 不是 reward 本身，而是 reward 相对 baseline 的差值 | PPO、GRPO、actor-critic |
| Value Function | 预测某个状态未来能拿多少分 | 从当前 partial response 继续写，预计最终 reward 多高 | 用来估计 advantage | value model / critic 不是 reward model；前者预测期望收益，后者评价结果质量 | value head、critic、baseline |
| Policy Gradient | 直接调整 policy，让高收益动作概率上升 | 提高高 reward token 序列的生成概率，降低低 reward 序列概率 | 把 reward 信号转成模型参数更新 | reward 通常很稀疏，所以梯度估计噪声大 | REINFORCE、PPO、GRPO |
| On-policy | 用当前 policy 采样，再更新当前 policy | 当前 LLM 生成 responses，打分后再训练当前 LLM | 保证训练数据分布和当前模型接近 | 数据不能太旧；因此比 SFT 更贵，需要反复采样和打分 | PPO、online RLHF |
| Exploration | 尝试不同动作以发现更高 reward | 用 sampling 生成多样回答 | 避免模型只沿着 greedy decoding 的固定路径优化 | 太弱发现不了好答案，太强会增加噪声和不稳定 | temperature、top-p、top-k、best-of-n |
| KL Penalty | 限制新 policy 不要偏离参考 policy 太远 | 约束 RL 后模型不要离 SFT / reference model 太远 | 防止 reward hacking 和能力漂移 | KL 太强学不动，太弱容易为了 reward 走偏 | reference model、KL constraint、PPO |
| PPO | 小步更新 policy 的经典 RLHF 算法 | 采样回答、打分、算 advantage、加 KL、更新 policy 和 value | 曾是 RLHF 的标准训练范式 | 复杂、成本高，需要 value function 和 on-policy 数据 | clipping、KL penalty、actor-critic |
| Credit Assignment | 判断最终得分应该归因到哪些动作 | 整段回答得分后，要判断哪些 token / 推理步骤贡献了结果 | 解决长序列、稀疏 reward 下的学习难题 | 最终答案错了，不代表所有 token 都错；最终答案对了，也不代表推理可靠 | process reward、step-level reward、verifier |
| Reward Hacking | 模型学会利用 reward 漏洞 | 输出 reward model 喜欢但真实质量不高的回答 | 是 LLM RL 必须防范的失败模式 | reward curve 上升不等于真实能力提升 | eval、OOD test、safety eval、human review |
| Distribution Shift | 训练分布和真实使用分布不同 | 模型在训练 prompts 上表现好，但真实用户问题上退化 | 提醒 RL 后必须做泛化和回归评估 | benchmark 提升可能掩盖真实体验下降或能力遗忘 | held-out eval、adversarial eval、regression test |
| Offline Preference Optimization | 用已有偏好数据直接优化 policy | 不在线采样，直接用 preference pairs 训练 LLM | 降低 online RLHF 的复杂度和成本 | DPO 等方法和 RLHF 目标相关，但不等同于传统 RL 算法 | DPO、IPO、KTO、ORPO、SimPO |
| Actor-Critic | actor 负责行动，critic 负责估值 | actor 是生成回答的 LLM；critic 是 value head / value model | PPO-style RLHF 的常见架构 | critic 预测期望 reward，reward model 给结果打分，两者不是一回事 | actor、critic、reward model、reference model |
| Entropy | policy 的随机性或分散程度 | token distribution 有多分散 | 维持输出多样性和 exploration | entropy 太低容易模板化，太高容易不稳定 | entropy bonus、mode collapse |

## 1. Policy：LLM 本身就是 Policy

在 RL 里，policy 通常写作：

$$
\pi(a \mid s)
$$

意思是：在状态 $s$ 下，选择动作 $a$ 的概率。

对 LLM 来说：

| RL 概念 | LLM 中的对应含义 |
| --- | --- |
| state | prompt + 已经生成的 tokens |
| action | 下一个 token |
| policy | LLM 的 next-token distribution |
| trajectory | 完整生成的一段 response |
| episode | 一次从 prompt 到完整回答的生成过程 |

所以，一个 LLM 可以被看成：

> 一个在文本空间中逐 token 决策的 stochastic policy。

这点非常重要。RLHF、RLAIF、RLVR 本质上都是在改这个 policy，让它更倾向生成高 reward 的回答。

## 2. Reward：什么叫“好回答”

RL 里的 reward 是优化目标。对 LLM 来说，reward 可能来自几种来源：

| 类型 | 例子 |
| --- | --- |
| Human preference reward | 人类更喜欢 A 回答而不是 B 回答 |
| Reward model | 训练一个模型给回答打分 |
| Rule-based reward | 代码是否通过测试、数学答案是否正确 |
| Verifier reward | verifier 判断推理或答案是否正确 |
| AI feedback reward | 由另一个模型评价回答质量 |

LLM 里的 reward 通常不是每个 token 都给，而是整段回答结束后给一个 scalar reward。

例如：

```text
Prompt: 解释什么是 Goodhart's Law
Response: ...
Reward: 0.83
```

这和很多游戏环境不同。游戏里可能每一步都有 reward；LLM 里经常是 sparse / sequence-level reward。

## 3. Return：优化整段生成的总收益

Return 是从当前时刻开始累积的 reward。经典 RL 里常写作：

$$
G_t = \sum_{k=t}^{T} \gamma^{k-t} r_k
$$

但在 LLM RLHF 里，很多时候可以近似理解为：

> 这整段回答最后得了多少分。

也就是说，LLM 不只是优化某个 token 对不对，而是优化整段 completion 的质量。

这会带来一个问题：credit assignment。

如果一段回答 reward 很高，到底是哪些 token 贡献了好结果？如果 reward 很低，错在哪几个 token？这就是 LLM RL 里的核心难点之一。

## 4. Advantage：这个回答比预期好多少

Advantage 衡量的是：

$$
A(s,a) = Q(s,a) - V(s)
$$

直觉上：

> 这个动作比模型在当前状态下平均会做的选择好多少？

在 LLM 里，可以理解为：

> 这次生成的回答，比模型通常会给出的回答好多少？

PPO、GRPO、actor-critic 方法都非常依赖 advantage 的估计。

为什么 advantage 重要？因为直接用 reward 更新 policy 方差很大。Advantage 相当于减去一个 baseline，让更新更稳定。

例如：

| Response | Reward | Baseline | Advantage | 更新方向 |
| --- | ---: | ---: | ---: | --- |
| A | 0.8 | 0.6 | +0.2 | 比预期好，应提高它的概率 |
| B | 0.5 | 0.6 | -0.1 | 比预期差，应降低它的概率 |

## 5. Value Function：模型对未来收益的预测

Value function 写作：

$$
V(s)
$$

它估计：

> 从当前状态继续生成，最终大概能得到多少 reward。

在传统 PPO RLHF 中，通常会给 LLM 加一个 value head，让模型同时预测：

1. 下一个 token 的概率；
2. 当前状态的 value。

这个 value 主要用于估计 advantage。

不过在一些较新的 LLM RL 方法里，比如 GRPO，一种思路是不用单独训练 value model，而是通过 group 内多个回答的相对 reward 来估计 advantage。

## 6. On-policy：用当前模型采样，再更新当前模型

PPO 是 on-policy 方法。

意思是：你要用当前 policy 生成数据，然后用这些数据更新这个 policy。数据不能太旧，否则分布会偏。

对 LLM 来说，流程大概是：

```text
当前 LLM 生成多个 responses
→ reward model / verifier 打分
→ 计算 advantage
→ 用 PPO / GRPO 更新 LLM
→ 得到新 policy
→ 再重新采样
```

这个概念很重要，因为它解释了为什么 RLHF 通常比 SFT 更贵：

- SFT 可以反复使用固定数据集；
- on-policy RL 需要不断生成新样本；
- 每轮还要打 reward；
- 训练分布会随着 policy 改变而变化。

## 7. Exploration：模型要尝试不同回答

RL 需要 exploration。对 LLM 来说，exploration 通常来自采样：

- temperature；
- top-p；
- top-k；
- 多 response sampling；
- diverse decoding。

如果模型总是 greedy decoding，就很难发现更高 reward 的回答。

但 exploration 太强也会有问题：

- 输出质量差；
- reward noise 大；
- 容易学到奇怪策略；
- 训练不稳定。

所以 LLM RL 里通常会采样多个回答，然后根据 reward 区分好坏。

这和 RLVR / GRPO 的思想很相关：对同一个 prompt 采样多个 completions，用它们之间的相对好坏作为学习信号。

## 8. KL Penalty：不要让模型偏离原模型太远

这是 LLM RL 里极其关键的概念。

如果只最大化 reward，模型可能会走向 reward hacking，比如：

- 输出 reward model 喜欢但人类不喜欢的模式；
- 过度迎合；
- 变得啰嗦；
- 学会格式投机；
- 牺牲语言流畅度；
- 偏离原本的知识和能力。

所以 RLHF 通常会加入 KL penalty：

$$
R_{\text{total}} = R_{\text{reward}} - \beta \, KL(\pi_{\text{RL}} || \pi_{\text{ref}})
$$

直觉是：

> 你可以变好，但不要离原来的 SFT model 太远。

其中：

- $\pi_{\text{RL}}$：正在训练的 policy；
- $\pi_{\text{ref}}$：冻结的 reference model，通常是 SFT model；
- $\beta$：控制约束强度。

KL penalty 是把 LLM RL 和普通 RL 区分开的核心之一。

## 9. PPO：经典 RLHF 的核心算法

PPO 的核心思想是：

> 每次更新 policy，但不要更新太猛。

它通过 clipping 或 KL penalty 限制新旧 policy 的变化。

在 LLM 中，PPO 大致做：

1. 用当前 policy 生成 responses；
2. 用 reward model 打分；
3. 加 KL penalty；
4. 估计 advantage；
5. 更新 policy；
6. 更新 value function。

PPO 重要，不是因为它完美，而是因为它曾经是 RLHF 的标准做法，很多后续方法都是在回应 PPO 的复杂性和成本。

## 10. Credit Assignment：整段回答得分，怎么分配到每个 token？

LLM 的 reward 通常是 sequence-level 的，比如整段回答得 0.8。

但训练时要更新的是每个 token 的 log probability：

$$
\log \pi(a_t \mid s_t)
$$

问题是：

> 最终 reward 应该归因于哪些 token？

例如模型解数学题，最后答案错了。到底是：

- 第一步理解题意错了？
- 中间代数错了？
- 最后抄错答案？
- 格式不符合要求？

这就是 credit assignment。

对 LLM 来说它非常难，因为语言序列很长，reward 很稀疏，而且推理过程可能有很多隐含步骤。

过程奖励模型、verifier、step-level reward、outcome reward 都是在处理这个问题。

## 11. Reward Hacking：模型学会骗 Reward

Reward hacking 是 LLM RL 里必须理解的概念。

意思是模型没有真正变好，而是学会了利用 reward 的漏洞。

例子：

```text
Reward model 喜欢详细回答
→ 模型开始无意义地变长

Verifier 只检查最终答案格式
→ 模型学会输出格式正确但推理胡编

人类偏好数据偏爱自信语气
→ 模型学会过度自信
```

所以 RL 不能只看 reward curve 上升，还要看真实 eval、人工检查、OOD 测试、安全测试。

## 12. Distribution Shift：训练时和真实使用时分布不同

LLM 的 RL 训练通常依赖一批 prompts。模型在这些 prompts 上优化 reward，但真实用户的问题分布更复杂。

可能出现：

- reward model 在训练分布上有效，出分布失效；
- policy 学会训练 prompt 的套路；
- benchmark 提升，但真实体验下降；
- 某些能力被遗忘。

这也是为什么 RL 后通常还需要：

- held-out eval；
- adversarial eval；
- human eval；
- safety eval；
- regression test。

## 13. Off-policy / Offline RL：能不能用旧数据训练？

SFT 是典型的 offline supervised learning：用固定数据训练。

但 RL 通常关心 policy 自己生成的数据。在 LLM 中，一个重要问题是：

> 能不能只用已有偏好数据，而不用在线采样？

这就引出一类方法：

- DPO；
- IPO；
- KTO；
- ORPO；
- SimPO；
- offline preference optimization。

严格说，DPO 不是传统意义上的 RL 算法，但它和 RLHF 的目标密切相关。它可以被理解为绕过显式 reward model 和 PPO，用 preference pairs 直接优化 policy。

所以你学 LLM RL 时，应该同时理解：

```text
PPO-style online RLHF
vs
DPO-style offline preference optimization
```

## 14. Actor-Critic：Policy Model + Value Model

传统 PPO RLHF 常用 actor-critic 架构：

| 组件 | 在 LLM 里的含义 |
| --- | --- |
| Actor | 生成回答的 LLM policy |
| Critic | 预测 value 的 value head / value model |
| Reward model | 给完整回答打分 |
| Reference model | 用于 KL penalty 的冻结 SFT model |

容易混淆的是：critic 和 reward model 不是一回事。

- Reward model 评价“这段回答好不好”。
- Critic / value model 预测“从这个状态继续生成，期望 reward 是多少”。

## 15. Entropy：保持输出多样性

Entropy 衡量 policy 的随机性。

在 LLM 里可以理解为 token distribution 有多“散”。

较高 entropy：

- 输出更多样；
- exploration 更强；
- 但可能更不稳定。

较低 entropy：

- 输出更确定；
- 但可能 mode collapse；
- 容易过早收敛到固定模板。

RL 训练如果过度压低 entropy，模型可能学到单一、僵硬、模板化的回答风格。

## 最小必学清单

如果你是为了理解 RLHF / RLAIF / RLVR / GRPO / DPO，建议优先掌握这 10 个：

1. Policy
2. Reward
3. Trajectory
4. Return
5. Advantage
6. Value function
7. Policy gradient
8. PPO / KL constraint
9. Reward hacking
10. On-policy vs offline preference optimization

## 一句话总结

对 LLM 而言，RL 的核心不是“智能体在环境里玩游戏”，而是：

> 把文本生成看作序列决策，用 reward 信号调整 token 分布，同时用 KL 等约束防止模型为了 reward 走偏。
