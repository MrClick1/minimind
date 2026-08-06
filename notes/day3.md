# Day 3：Attention 内部结构（Q/K/V、GQA、Causal Mask）

- 对应分支：`learn/day3-attention`
- 对应实验：`experiments/03_trace_attention_shapes.py`
- 目标：理解 Attention 内部每一步的形状变化与含义。该实验已成功运行。

## 要点

### 1. Q/K/V 投影层（Tiny 配置：hidden=128，Q 头=4，KV 头=2，head_dim=32）

```text
q_proj = Linear(128, 128)    # 4 头 × 32
k_proj = Linear(128, 64)     # 2 头 × 32（GQA）
v_proj = Linear(128, 64)     # 2 头 × 32
o_proj = Linear(128, 128)
```

PyTorch 中 `Linear(in, out)` 的 `weight.shape = [out, in]`，所以：

```text
q_proj.weight [128, 128]
k_proj.weight [64, 128]
v_proj.weight [64, 128]
o_proj.weight [128, 128]
```

权重初始是接近 0 的随机数，训练时通过反向传播更新，最终"学出"不同的投影功能。

### 2. 四维张量怎么理解

`[B, T, H, D]` 不要想象成四维几何体，而是四级嵌套结构：

```text
文本 → token → 注意力头 → 头内向量
```

定位一个数字：`x[batch, token, head, dimension]`

### 3. `transpose(1, 2)`：`[B,T,H,D] → [B,H,T,D]`

含义：原来"按 token 组织多个头"，转置后"按头组织整条序列的 token"。

元素对应：`y[b, h, t, d] == x[b, t, h, d]`

注意：当 `T = H` 时（如 T=4、H=4），转置前后形状数字相同（`[2,4,4,32]`），容易误以为没变化——但维度含义已经交换。实验用 `T=6` 就能看到 `[2,6,4,32] → [2,4,6,32]`。

### 4. GQA 与 `repeat_kv`

```text
Query 头 = 4，KV 头 = 2
Q0、Q1 共享一组 K/V；Q2、Q3 共享另一组 K/V
```

`repeat_kv` 把 K/V 从 `[B,T,2,32]` 复制扩展为 `[B,T,4,32]`，方便与 4 个 Query 头批量计算。作用：减少参数和 KV Cache 开销。

### 5. Attention Score 为什么是 `[B,H,T,T]`

```text
Q [B,H,T,D] × Kᵀ [B,H,D,T] = [B,H,T,T]
```

两个 T 含义不同：

```text
第一个 T：Query token 数量
第二个 T：Key token 数量
```

准确索引：`scores[batch, head, query_token, key_token]`

### 6. Causal Mask

因果掩码防止当前 token 看到未来 token，未来位置设为 `-inf`，Softmax 后概率约为 0：

```text
          K0 K1 K2 K3
Q0         ✓  ×  ×  ×
Q1         ✓  ✓  ×  ×
Q2         ✓  ✓  ✓  ×
Q3         ✓  ✓  ✓  ✓
```

## 遇到的问题与答案

### 四维张量想象不出来

- **问题**：`[2,6,4,32]` 这种四维形状很难用"几何空间"理解。
- **答案**：不要想四维空间，把它当四级索引/文件夹：batch 下有哪些 token，每个 token 下有哪些 head，每个 head 下是 32 维向量。

### `transpose` 前后形状数字相同导致混淆

- **问题**：T=4、H=4 时 `[2,4,4,32] → [2,4,4,32]`，看起来"没变"。
- **答案**：改用 `T=6`，看到 `[2,6,4,32] → [2,4,6,32]`，确认维度确实交换了；并记住"形状数字相同 ≠ 语义没变"。

### 实验输出（实测）

```text
input_ids: [2, 6]
embedding x: [2, 6, 128]
raw Q: [2, 6, 128]   raw K: [2, 6, 64]   raw V: [2, 6, 64]
拆头后 Q: [2, 6, 4, 32]   K/V: [2, 6, 2, 32]
repeat_kv 后 K/V: [2, 6, 4, 32]
transpose 后 Q/K/V: [2, 4, 6, 32]
attention scores: [2, 4, 6, 6]
```

## 验收

- 能解释 GQA 为什么省参数、`repeat_kv` 做了什么；
- 能说出 `scores` 两个 T 分别代表什么；
- 能画出 Causal Mask 矩阵并解释 `-inf` 的作用。
