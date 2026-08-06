# Day 4：完整 Attention（手工串联 + 概念验收）

- 对应分支：`learn/day4-attention-complete`
- 对应实验：`experiments/04_manual_attention.py`（AI 生成，作为参考答案）；`experiments/04_my_manual_attention.py`（待自己手写，当前为空文件）
- 状态：脚本可运行、形状全部符合预期；**自己手写一版并独立回答 5 个问题后才算验收完成**。

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

### 1. 为什么除以 `sqrt(head_dim)`？

点积是 D 个独立零均值项的和，方差可加：

```text
Var(dot) = D·σ⁴  →  标准差 ∝ √D
```

所以分数尺度随 D 增大。除以 `√D` 让分数标准差不随 D 变化（保持 O(1)），防止 Softmax 过尖锐 → 饱和 → 梯度消失。

注意不要除以 D：`Var(dot/D) = σ⁴/D`，D 越大分数越挤在 0 附近，Softmax 输出趋近均匀分布，注意力失去区分能力。

### 2. 为什么 Softmax 用 `dim=-1`？

`scores` 的形状是 **`[B,H,T,T]`**（QKᵀ 之后），不是 `[B,T,H,D]`！最后一维是 Key 的维度。每个 Query token 需要在所有可见 Key 之间分配权重，所以沿最后一维归一化，保证每行和为 1。

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

Day 4 验收后 → 创建 `learn/day5-rmsnorm-rope` 分支，学习 RMSNorm（为什么归一化、与 LayerNorm 区别、`rms_norm_eps` 作用）和 RoPE（为什么需要、作用对象、旋转平面、`cos/sin` 形状）。
