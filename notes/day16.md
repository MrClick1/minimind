# Day 16：MoE（混合专家）

- 对应源码：`model/model_minimind.py` → `MOEFeedForward`；`MiniMindBlock` 在 `use_moe=True` 时用它替换 MLP。
- 配置：num_experts=4，num_experts_per_tok=1（top-1），norm_topk_prob=True，router_aux_loss_coef=5e-4。

## 核心思想：稀疏激活
- 把 MLP 复制成 N 份（专家），每个 token 只激活其中 top-k 个。
- 效果：参数量可以很大（容量大），但每个 token 的计算量只取决于 k（稀疏）。
- MiniMind-3-moe = **198M-A64M**：总参数 198M，每个 token 只激活约 64M（≈1/3）。

## 组成
- **路由器 gate**：`Linear(hidden_size, num_experts)`，输出每个专家的分数 → softmax 成概率。
- **N 个专家**：每个都是完整 FeedForward（gate_proj / up_proj / down_proj + SiLU）。
- **top-k 选择**：每个 token 只挑分数最高的 k 个专家。

## 前向流程（源码逐行）
```python
x_flat = x.view(-1, hidden_dim)                 # [B,T,C] -> [B*T, C]，token 摊平
scores = F.softmax(self.gate(x_flat), dim=-1)   # 路由概率
topk_weight, topk_idx = torch.topk(scores, k=num_experts_per_tok)  # 挑 k 个专家
if norm_topk_prob: topk_weight = topk_weight / (topk_weight.sum(-1, keepdim=True) + 1e-20)
y = zeros_like(x_flat)
for i, expert in enumerate(self.experts):       # 每个专家处理分给它的 token
    mask = (topk_idx == i)
    if mask.any():
        y.index_add_(0, token_idx, expert(x_flat[token_idx]) * weight)
return y.view(batch_size, seq_len, hidden_dim)  # 形状不变 [B,T,C]
```

## 辅助损失（负载均衡）
- 问题：路由器可能偏心，总把 token 丢给 1~2 个专家，其他专家学不到东西。
- 公式：
  ```python
  load = F.one_hot(topk_idx, num_experts).float().mean(0)   # 每个专家被选中的频率
  aux_loss = (load * scores.mean(0)).sum() * num_experts * router_aux_loss_coef
  ```
- 直觉：被选中多的专家（load 大）× 平均路由分数高 → 被惩罚；鼓励均匀使用。
- 训练时加入总 loss：`loss = res.loss + res.aux_loss`（train_pretrain/train_sft 里那个 aux_loss 的来由）。

## 训练/推理差异
- 训练：所有专家反向传播；aux_loss 保证均衡。
- 推理：每个 token 只过 top-1 专家 → 计算量小。
- Attention 不变（MoE 只替换 MLP），KV Cache 逻辑不变。

## 实测演示（Tiny 配置，4 专家，top-1）
- 输入 [2,6,128] → 输出 [2,6,128]（形状不变）。
- 初始路由分布 [5,2,2,3]（略不均衡），aux_loss ≈ 0.0005；训练中 aux loss 会把它拉向均衡。

## 面试表述
"MoE 把 FFN 复制成多个专家，用路由器给每个 token 选 top-k 个专家计算，实现参数量大但计算量稀疏：如 MiniMind-3-moe 总参数 198M、每 token 只激活 64M；配合负载均衡辅助损失防止路由塌缩，训练时加入总 loss，推理时每个 token 只走少数专家。"
