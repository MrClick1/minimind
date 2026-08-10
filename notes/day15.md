# Day 15：RLHF 流程与原理（PPO 对齐）

- 对应源码：`trainer/train_ppo.py`、`trainer/rollout_engine.py`（MiniMind 的 PPO 实现）

## 为什么需要 RLHF
- 预训练 / SFT 让模型"会说话"，但不一定符合人类偏好（有用、诚实、无害）。
- RLHF：用人类偏好信号把模型行为对齐到人类期望。

## 三阶段流程
```text
阶段1 SFT：人类示范数据监督微调（Day 10）
阶段2 奖励模型：偏好对 + Bradley-Terry 损失，学一个"打分器" r(x,y)
阶段3 PPO：策略模型采样回答 → 奖励打分 + KL 约束 → PPO 更新
```

## 奖励模型（RM）
- 输入 prompt + response，输出一个标量分数。
- 训练损失（BT）：`L = -log σ(r(x,y_w) - r(x,y_l))`（Day 14 已讲）。

## PPO 环节的四件套
| 模型 | 角色 |
|---|---|
| actor（策略） | 正在对齐的模型，输出分布 π_θ |
| ref（参考） | 冻结的 SFT 模型，用于 KL 约束 |
| reward model | 给回答打分 |
| critic（价值） | 估计期望收益，用于算优势 |

## 关键机制
- **rollout**：从当前策略采样回答。采样是离散 token、奖励不可导 → 需要强化学习而非直接梯度下降。
- **奖励**：MiniMind 里是 `RM 分数 + 规则奖励`（长度、思考格式、重复惩罚等）。
- **KL 约束**：惩罚策略偏离参考模型太远 → 防止 reward hacking（刷高分但语无伦次）。
- **优势估计**：用 critic 的价值估计 + 实际奖励算优势，决定哪些 token 该加/减概率。
- **clipped 目标**：限制单步更新幅度，保证稳定。
- **重要性比**：用 rollout 时保存的旧 logp 做 correction，同一批采样可复用多次更新。

## 为什么 PPO 而不是直接监督
- 采样出的回答是离散 token，奖励无法反向传播；
- 策略梯度思路：按优势大小"放大高概率的好 token、压低坏 token"。

## 与 DPO 的关系
- DPO 是 RLHF 的简化替代：从 BT 模型 + KL 约束推出闭式解，免去 RM、critic 和在线采样（Day 14）。

## 面试表述
"RLHF 分三步：SFT 学基本对话能力，用偏好对按 Bradley-Terry 损失训练奖励模型，最后用 PPO 让策略模型在采样回答上最大化奖励并加 KL 约束防止偏离参考模型；PPO 需要 actor/ref/reward/critic 四个模型，复杂度高，DPO 通过闭式解省去了 RM 和 critic。"
