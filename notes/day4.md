# Day 4：完整 Attention（手工串联 + 概念验收）

- 对应分支：`learn/day4-attention-complete`
- 对应实验：`experiments/04_manual_attention.py`（AI 生成，作为参考答案）；`experiments/04_my_manual_attention.py`（已跟随参考流程补齐到 `o_proj`，尚待独立修正、运行和脱稿复现）
- 状态：参考脚本可运行、形状全部符合预期；目前已大致理解完整流程，但**独立复现和变化测试暂缓，后续回来完成后才算最终验收**。

## 要点

### 1. 完整 Attention 流程

```text
hidden_states
→ q_proj / k_proj / v_proj
→ 拆分多个 Head（view）
→ repeat_kv（GQA 扩展 K/V）
→ transpose(1,2)  → [B,H,T,D]
→ QKᵀ  → [B,H,T,T]
→ 除以 sqrt(head_dim)
→ 加 causal mask（未来位置 -inf）
→ softmax(dim=-1)
→ Attention Weight × V → context [B,H,T,D]
→ transpose 转回 [B,T,H,D]
→ 合并 Head → [B,T,H*D]
→ o_proj → [B,T,C]
```

核心公式：

```text
Attention(Q,K,V) = softmax(QKᵀ / √D + mask) · V
```

### 2. 实测形状（T=6, H=4, D=32）

```text
hidden_states:     [2, 6, 128]
Q:                 [2, 6, 128]   K/V: [2, 6, 64]
拆头后 Q:           [2, 6, 4, 32]   K/V: [2, 6, 2, 32]
repeat_kv 后 K/V:  [2, 6, 4, 32]
transpose 后:      [2, 4, 6, 32]
scores:            [2, 4, 6, 6]
attention_weights: [2, 4, 6, 6]
context:           [2, 4, 6, 32]
转回后:            [2, 6, 4, 32]
合并 Head 后:      [2, 6, 128]
o_proj 后:         [2, 6, 128]
```

### 3. 重要观察：随机初始化时权重是均匀分布

实测第一个 Head 的权重：

```text
[1.0000, 0, 0, 0, 0, 0]
[0.5000, 0.5000, 0, 0, 0, 0]
[0.3333, 0.3333, 0.3333, 0, 0, 0]
...
```

原因：模型未训练，投影权重接近 0 → QKᵀ 分数接近 0 → Softmax 退化成"在可见位置均匀分配"。**"关注谁"是训练学出来的，不是结构自带的。**

### 4. 与真实源码的差异

`04_manual_attention.py` 是核心流程的手工版；真实 `MiniMindAttention.forward` 还包含：

| 真实实现 | 说明 | 学习时机 |
|---|---|---|
| `q_norm` / `k_norm` | 对 Q/K 做 RMSNorm（QK-Norm） | 以后 |
| `apply_rotary_pos_emb` | RoPE 位置编码 | Day 5 |
| KV Cache（`past_key_value`） | 推理加速 | 以后 |
| Flash Attention | `scaled_dot_product_attention` | 以后 |
| Dropout | 训练时随机丢弃 | 以后 |

## 遇到的问题与答案

### 1. 为什么除以 `sqrt(head_dim)`？（数学推导）

目的（先记住）：让 Attention Score 的尺度不随 head_dim 变化，防止 Softmax 饱和 → 梯度消失。

假设 Q、K 的每个维度独立、均值 0、方差 σ²（随机初始化时成立）。点积是 D 个乘积项之和：

```text
s = q₁k₁ + q₂k₂ + ... + q_D·k_D
```

第一步，看单个乘积项 `qᵢkᵢ`：

```text
E[qᵢkᵢ] = E[qᵢ]·E[kᵢ] = 0          （独立 + 零均值）
Var(qᵢkᵢ) = E[(qᵢkᵢ)²] = E[qᵢ²]·E[kᵢ²] = σ²·σ² = σ⁴
```

每个维度贡献的方差是常数 σ⁴，与 D 无关。

第二步，独立随机变量之和的方差可加：

```text
Var(s) = Var(q₁k₁) + Var(q₂k₂) + ... + Var(q_D·k_D) = D·σ⁴
```

所以：

```text
标准差 std(s) = √D·σ²   （随 √D 增长）
```

第三步，比较两种缩放（注意 Var(cX) = c²·Var(X)）：

```text
除以 √D：Var(s/√D) = D·σ⁴ / (√D)² = σ⁴        → 常数，尺度不随 D 变
除以 D：  Var(s/D)  = D·σ⁴ / D² = σ⁴/D        → 趋近 0，分数全挤在 0 附近
```

除以 D 的后果：Softmax 输入几乎全相等 → 输出退化为均匀分布 → 注意力失去区分能力。除以 D 的常见错误是少算了一个平方（把 Var(cX)=c²Var(X) 当成了 cVar(X)）。

直觉（中心极限定理）：D 个独立零均值项的和，典型波动量级是 √D，所以"和"的自然尺度是 √D；除以 √D 是把波动量级归一化到 O(1)。

实测（q、k ~ N(0,1)，σ⁴=1）：D=4 → var 4.0；D=64 → var 64.1；D=256 → var 256.6；var/D ≈ 1，std ≈ √D。
### 2. 为什么 Softmax 用 `dim=-1`？

先确认形状：`scores` 是 QKᵀ 之后的结果，形状是 **`[B,H,T,T]`**，不是 `[B,T,H,D]`（那是 Q/K/V 拆头后的形状）。最后两个 T 分别是 Query 数和 Key 数。

`dim=-1` 就是最后一个维度 = **Key 维度**。沿它归一化的含义：对每个 Query token（固定 batch、head、query 位置），在它能看到的所有 Key 之间分配权重：

```text
p[q, k] = exp(s[q, k]) / Σ_{k'} exp(s[q, k'])
```

保证每个 Query 的权重行和为 1。

为什么不能沿其他维度：

```text
dim=0（batch）：不同样本之间不该竞争权重
dim=1（head）：每个头独立学习各自的关注模式
dim=2（query）：每个 query 独立分配权重，不该跨 query 归一化
dim=3（key）：唯一语义正确
```

数字例子：一行分数 `[1.0, 2.0, -inf, -inf]`，沿 dim=-1 做 Softmax ≈ `[0.269, 0.731, 0, 0]`，和为 1；被 Causal Mask 置为 `-inf` 的未来位置恰好变成 0。

### 3. 为什么 Attention Weight 要乘 V？

V 是 token 承载的内容信息。权重乘 V 是对所有可见 token 的 V 做加权求和，得到 **context**：当前 token 从序列中聚合到的上下文信息，形状 `[B,H,T,D]`。

### 4. 为什么 `[T,T] @ [T,D] = [T,D]`？

中间的 T（Key 数）是矩阵乘法中被"求和消掉"的维度：

```text
out[q] = Σ_k W[q,k]·V[k]
```

每个 Query 对 T 个 Key 的 V 向量做加权求和，得到一个 D 维向量；T 个 Query 就是 T 行。

### 5. 为什么需要 `o_proj`？（曾答错，重点）

`o_proj` **不是**输出词表概率的层（那是 `lm_head`，输出 `[B,T,V]`）。`o_proj` 的作用：

1. 把拼接的多头结果 `[B,T,H*D]` 投影回 `hidden_size` → `[B,T,C]`，才能与残差相加；
2. 学习如何混合不同头关注到的不同信息。

验证：Day 4 脚本里 `o_proj 后: [2,6,128]`，而不是 `[2,6,6400]`。

### 6. 环境问题：打印中文报 `UnicodeEncodeError`

- **现象**：默认控制台编码（cp1252）下，脚本打印中文直接崩溃。
- **答案**：运行前设置 `$env:PYTHONIOENCODING='utf-8'`，或在脚本开头加：

```python
import sys
sys.stdout.reconfigure(encoding='utf-8')
```

### 7. 代码是 AI 生成的，怎么学？

- 跑通 AI 生成的代码 ≠ 自己会写。检验标准：**不看任何参考，从头写出完整流程，并解释每一步形状为什么是这样**。
- 建议：`04_my_manual_attention.py` 自己从零写（可参考模型源码，不看 AI 版），每步先预测形状再运行；再做两个改动测试（`seq_len=5` 看 transpose、`num_key_value_heads=1` 看 repeat_kv）。

### 8. 梯度消失与梯度爆炸是什么？

- **梯度**：loss 对参数的变化率，决定参数更新方向和步长。
- 深层网络的梯度是各层导数的**乘积**（链式法则）：每层导数都 <1，连乘趋近 0（消失）；都 >1，连乘爆炸。
- 例：0.5²⁰ ≈ 1e-6；1.5²⁰ ≈ 3300。
- Softmax 饱和时导数 `p(1-p) → 0`，是注意力里梯度消失的典型来源。
- 缓解：残差连接、归一化、`÷√head_dim`、梯度裁剪、ReLU 等。


### 9. Softmax 输出对输入的导数为什么是 p(1−p)？

Softmax 的输入是向量 `z = [z₁, ..., z_K]`，输出是向量 `p = [p₁, ..., p_K]`，所以导数是一张 K×K 的矩阵（Jacobian），不是单个数字：

```text
对角线（j = i）：     ∂p_i/∂z_i = p_i(1 − p_i)
非对角线（j ≠ i）：   ∂p_i/∂z_j = −p_i·p_j
```

推导（商法则）：记 `p_i = e^{z_i} / S`，`S = Σ_j e^{z_j}`，且 `dS/dz_i = e^{z_i} = p_i·S`：

```text
∂p_i/∂z_i = (e^{z_i}·S − e^{z_i}·e^{z_i}) / S²
           = p_i − p_i²
           = p_i(1 − p_i)
```

p(1−p) 在 p=0.5 时最大（0.25），p→1 或 p→0 都趋近 0。直觉：某个类别已经 99% 确定时，再推高它的分数，概率几乎不变，对自身输入不敏感；在 0.5 的"犹豫区"最敏感。这就是"Softmax 饱和 → 梯度消失"的机制。

实测验证（z = [3.0, 0.5, 0.5]，p ≈ [0.859, 0.071, 0.071]）：

```text
解析值  p1(1−p1) = 0.1211，−p1·p2 = −0.0606
数值差分 0.1210，−0.0608（吻合）
```

两个细节：

1. 同一行导数之和为 0：`Σ_j ∂p_i/∂z_j = 0`。因为所有 logit 同时加同一个常数，Softmax 输出不变（平移不变性）。
2. `p(1−p)` 是 **Softmax 单独一层**的导数；实际中 Softmax 后面通常接交叉熵，两者合成后梯度是 `p − y`，不会消失。"饱和导致梯度消失"要放在梯度穿过 Softmax 中间量的语境下理解（如 Attention 权重乘 V 那一步）。
## 验收清单（全部满足才算 Day 4 完成）

1. 自己能从头写出完整 Attention（`04_my_manual_attention.py` 非空、可运行）；
2. 所有形状符合预期；
3. `attention_weights` 是 `[B,H,T,T]`；
4. 每行权重之和接近 1；
5. 未来 token 位置权重为 0；
6. `context` 由 `[B,H,T,T] @ [B,H,T,D]` 得到 `[B,H,T,D]`；
7. 合并 Head 后回到 `[B,T,128]`，`o_proj` 后仍为 `[B,T,128]`；
8. 能独立回答上面 5 个概念问题（含修正后的 o_proj）。

## 下一步

用户决定先进入 Day 5，在学习 RMSNorm 与 RoPE 的过程中保留以下 Day 4 回看任务：

1. 先修正并独立运行 `04_my_manual_attention.py` 中遗留的类名、配置项和 K Head 形状问题；
2. 不看实现，只写出完整 Attention 的 12 步注释骨架；
3. 独立补出 Score、缩放、Causal Mask、Softmax 和 `Attention Weight @ V`；
4. 每一步先预测张量形状，再运行验证；
5. 将 `seq_len` 改为 `5`，观察所有与 `T` 有关的形状；
6. 将 `num_key_value_heads` 改为 `1`，预测 `n_rep` 和 `repeat_kv` 前后的形状；
7. 重新回答 5 个概念问题，重点修正 `softmax(dim=-1)` 与 `o_proj` 的表述。

这些任务不阻塞 Day 5 的开始，但应在 Attention 阶段最终验收前回来完成。




