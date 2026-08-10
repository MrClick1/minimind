# Day 14：DPO（直接偏好优化）

- 对应源码：`trainer/train_dpo.py`、`dataset/lm_dataset.py` → `DPODataset`

## 背景：从 RLHF 到 DPO
- RLHF（PPO）需要奖励模型 + 策略模型 + 价值模型 + 参考模型，训练复杂、不稳定。
- DPO（2023）把"偏好对齐"变成二分类问题：**不需要奖励模型和价值模型**，只需策略模型 + 冻结的参考模型。

## 数据格式：偏好对
- 每行 `{chosen: [对话], rejected: [对话]}`：同一个 prompt 的两个回答，chosen 更好。
- 各自应用聊天模板 + loss mask（只有 assistant 回复参与，复用 Day 10 的逻辑）。

## 核心公式
```text
L = -log σ( β · ( log π_θ(y_w|x) - log π_ref(y_w|x)
                - log π_θ(y_l|x) + log π_ref(y_l|x) ) )
```
- π_θ：策略模型（可训练）；π_ref：参考模型（冻结）；y_w = chosen，y_l = rejected；β 控制敏感度。
- 直觉：让 chosen 的概率相对参考模型**升高**，rejected **降低**；差距越大 loss 越小。

## 源码实现（train_dpo.py）
- `logits_to_log_probs`：`log_softmax` 后按 labels `gather` 出每个 token 的 log prob → [B, T]。
- `dpo_loss`：
  ```python
  ref_log_probs = (ref_log_probs * mask).sum(dim=1)       # 掩码后按序列求和
  pi_logratios = chosen_policy - reject_policy
  ref_logratios = chosen_ref - reject_ref
  logits = pi_logratios - ref_logratios
  loss = -F.logsigmoid(beta * logits)
  ```
  - batch 前一半是 chosen、后一半是 rejected（`x = cat([x_chosen, x_rejected])`）。
- 训练循环差异：
  - 两个模型：`model`（策略，可训练）+ `ref_model`（冻结：`requires_grad_(False)`、eval、no_grad 前向）；
  - 学习率极小：默认 4e-8（建议 ≤5e-8，防止遗忘）；
  - beta 默认 0.15；batch_size 4（两个模型显存翻倍）。

## Bradley-Terry 偏好模型（BT 模型）

- 统计学里建模"两两比较"的经典模型（1952）：给每个物品一个强度/分数 s，则 i 胜过 j 的概率：
  ```text
  P(i > j) = s_i / (s_i + s_j) = σ(s_i - s_j)
  ```
  其中 σ 是 sigmoid 函数：分数差越大，胜率越接近 1；分数相等时胜率 0.5。
- 数字感受（sigmoid）：
  ```text
  分数差 -3 → 0.047    0 → 0.500    1 → 0.731    3 → 0.953
  ```
- 应用：体育排名、国际象棋 Elo 评分（Elo 本质就是 BT + 逻辑斯蒂尺度）、RLHF 的奖励模型。
- RLHF 里的用法：把回答 y 的奖励 r(x,y) 当作"分数"，人类偏好 chosen 胜过 rejected 的概率建模为：
  ```text
  P(y_w > y_l | x) = σ( r(x, y_w) - r(x, y_l) )
  ```
- 对 DPO 的意义：DPO 正是从这个概率模型出发，结合 RL 目标反解出闭式解，把奖励用策略/参考模型的 log 概率比代替，所以 DPO 损失里那个 sigmoid 就是 BT 概率。
## 为什么不需要奖励模型
- 从 Bradley-Terry 偏好模型 + RL 目标可推出闭式解：最优策略 `π* ∝ π_ref · exp(r(x,y)/β)`。
- 重参数化：把奖励 r 用"策略/参考模型的 log 概率比"表示 → 直接监督优化策略模型，RL 退化为二分类。

## DPO vs PPO（简表）
| | PPO | DPO |
|---|---|---|
| 需要的模型 | reward + critic + actor + ref | policy + ref |
| 数据 | 在线采样 | 离线静态偏好对 |
| 稳定性 | 复杂、调参多 | 简单、稳定 |
| 适用 | 需要在线探索 | 静态偏好数据充足时 |

## 面试表述
"DPO 从 Bradley-Terry 偏好模型推导出闭式解，用策略模型与冻结参考模型的 log 概率比代替奖励，把 RLHF 的偏好对齐简化为二分类：提升 chosen 回答的相对概率、压低 rejected 回答，损失为 -log σ(β·log-ratio)；只需两个模型、训练稳定，学习率通常要很小以防遗忘。"

