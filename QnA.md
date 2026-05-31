# LLM RL Q&A

本文档用于整理学习 LLM 强化学习过程中遇到的问题和回答，方便后续回顾。

相关主文档：[LLM 中最重要的强化学习概念](./llm-rl-core-concepts.md)

## Q1：Policy 可以理解成模型吗？

可以粗略这样理解。

在 LLM RL 里，常见的直觉是：

> policy ≈ 负责生成回答的 LLM

但更精确地说：

> policy 是模型产生的“在某个状态下选择动作的概率分布”。

对 LLM 来说：

```text
state  = prompt + 已经生成的 tokens
action = 下一个 token
policy = 下一个 token 的概率分布
```

比如模型看到：

```text
Prompt: Explain PPO
Generated so far: PPO is
```

此时 policy 可以理解成类似这样的分布：

```text
P("a" | context)    = 0.18
P("an" | context)   = 0.07
P("used" | context) = 0.04
...
```

模型参数会产生这个分布，而这个分布就是 policy。

需要注意的区别：

- **LLM policy model**：正在被训练、负责生成回答的模型；
- **policy**：这个模型表现出来的行为，即 $\pi(a \mid s)$；
- **reward model**：也是模型，但它负责给回答打分，不负责选择下一个 token；
- **value model / critic**：也是模型，但它负责预测未来期望 reward，不直接生成回答。

所以，在日常讨论 LLM RL 时，把 “policy” 理解成“模型”通常没问题；但在更严谨的表达里，应该说：

> LLM parameterizes the policy.

也就是：

> LLM 用它的参数表示并产生 policy。
