# Day 11：KV Cache 与推理生成

- 对应源码：`model/model_minimind.py` → `Attention.forward` / `MiniMindModel.forward` / `MiniMindForCausalLM.generate`
- 默认配置：hidden=768，q_heads=8，kv_heads=4，head_dim=96，max_position_embeddings=32768

## 问题：朴素推理的重复计算
- 自回归每步只生成 1 个新 token，但 Attention 需要和全部历史 token 交互。
- 若不缓存：第 t 步要把整个序列重新 forward，第 t 步注意力代价 O(t²)，生成 T 步累计 O(T³)；其中前面 token 的 K/V 每步都被重复计算。

## 缓存什么：只缓存 K/V，不缓存 Q
- K/V 是"被查询的内容"，未来每个新 token 都要和全部历史 K 算 QKᵀ、对 V 加权 → 缓存复用。
- Q 只属于当前 token（当前 query 查历史），用完即弃。

## 源码实现
1. `Attention.forward`：
   ```python
   if past_key_value is not None:
       xk = torch.cat([past_key_value[0], xk], dim=1)   # 沿序列维拼接
       xv = torch.cat([past_key_value[1], xv], dim=1)
   past_kv = (xk, xv) if use_cache else None
   ```
2. `MiniMindModel.forward`：
   ```python
   start_pos = past_key_values[0][0].shape[1] if past_key_values[0] is not None else 0
   position_embeddings = (freqs_cos[start_pos:start_pos+seq_length], ...)
   ```
   新 token 从 start_pos 取位置编码 → RoPE 位置连续正确。
3. `generate` 循环：
   ```python
   past_len = past_key_values[0][0].shape[1] if past_key_values else 0
   outputs = self.forward(input_ids[:, past_len:], ..., past_key_values, use_cache=use_cache)
   logits = outputs.logits[:, -1, :] / temperature   # 只看最后一个 token
   ... top_k / top_p / repetition_penalty / 采样 ...
   input_ids = torch.cat([input_ids, next_token], dim=-1)
   past_key_values = outputs.past_key_values if use_cache else None
   ```

## 为什么历史 token 的 K/V 每步重算结果完全相同

两个前提：
1. 推理时模型权重冻结，且 eval 模式关闭 dropout → 同一输入必得同一输出（确定性）。
2. Causal Mask：位置 i 的 hidden state 只能看到 0..i，**看不到未来的 token**。

推导：
- 位置 i 的表示只依赖 tokens 0..i；
- 每步新增的 token 都排在 i 后面，前缀 0..i 与上一步完全一样；
- 所以 hidden_i 不变 → K_i、V_i 不变。

例子：
```text
step 1：输入 [A, B]        → 算出 K_A、K_B、V_A、V_B
step 2：输入 [A, B, C]     → A 仍只看 A；B 仍只看 A、B
                             → K_A、V_A、K_B、V_B 和 step 1 完全相同
                             只有 C 的 K/V 是新计算的
```

反向思考：如果没有 causal mask（双向注意力），前面的 token 会看到新增的 C，hidden 会改变，K/V 就不能复用——这正是 KV Cache 只适用于自回归 Decoder-Only 模型的原因。
## 复杂度对比
```text
无缓存：第 t 步注意力 O(t²)，生成 T 步累计 O(T³)
有缓存：第 t 步只算新 token 的 QKᵀ（O(t)），累计 O(T²)；模型其他层每步只跑 1 个 token
```

## 推理时新 token 的完整计算流程（确认理解）

推理 = 用训练/微调后**冻结的权重**做前向（`torch.inference_mode()`，eval 模式，无反向、无 dropout）。

生成第 t 个新 token 时：
```text
① 取刚生成的那个 token（只喂这一个）
② Embedding → hidden state
③ 每一层：用固定权重算它的 Q、K、V
④ 新 K/V 拼接到该层缓存末尾
⑤ Attention：新 Q × 缓存里的全部历史 K → 权重 × 历史 V → context
⑥ 逐层向上 → 最终 logits → 采样下一个 token
```

历史 token：K/V 直接读缓存，**不重新计算**；它们当年的 Q 在当年用完后即弃（Q 不复用）。
## 时间复杂度为什么是 O(T³) / O(T²)

注意力单步代价来源：第 t 步序列有 t 个 token，QKᵀ 是 `[t,d]×[d,t]=[t,t]`，权重×V 也是 `[t,t]×[t,d]`——两个矩阵乘法都是 O(t²)（d 固定）。

```text
无缓存：第 t 步重算整个 [t,t] 注意力矩阵 → O(t²)
        累计 Σ_{t=1}^{T} t² = T(T+1)(2T+1)/6 ≈ T³/3 → O(T³)

有缓存：每步只算新 token 的一行 [1,t] → O(t)
        累计 Σ_{t=1}^{T} t = T(T+1)/2 ≈ T²/2 → O(T²)
```

其余层（Embedding、RMSNorm、MLP、投影）只处理新 token，每步 O(1)，不随序列长度增长。

数字例子（忽略常数，只看加法/乘法次数量级）：

```text
T=100     无缓存 338,350    有缓存 5,050    加速比 67x
T=1000    无缓存 333,833,500  有缓存 500,500  加速比 667x
T=10000   无缓存 333,383,335,000 有缓存 50,005,000  加速比 6,667x
```

直觉：无缓存 = 每步把整张注意力表重画一遍；有缓存 = 每步只画新加的一行。
## 矩阵乘法的时间复杂度

规则：`[m×n] × [n×p] → [m×p]`，输出有 m×p 个元素，每个元素是"一行 × 一列"的点积（n 次乘法 + n-1 次加法）→ 乘法次数 = m·n·p，量级 **O(m·n·p)**。

常用 FLOPs 口径 = 2·m·n·p（乘法 + 加法各一次）。

套到 Attention 形状（d 固定为常数）：
```text
[t,d] × [d,t] → O(t·d·t) = O(t²)
[t,t] × [t,d] → O(t·t·d) = O(t²)
[1,d] × [d,t] → O(1·d·t) = O(t)
[1,t] × [t,d] → O(1·t·d) = O(t)
```

数字例：
```text
[2×3]×[3×4] → 2·3·4 = 24 次乘法
[100×96]×[96×100] → 960,000 次乘法（t²d 量级）
[1×96]×[96×100] → 9,600 次乘法（td 量级）
```

注意：并行（GPU Tensor Core）只是同时算多个元素，**运算次数不变**；Strassen 等更快算法理论存在，但实际矩阵运算仍按 m·n·p 估计。
## 显存开销（KV cache 大小）
```text
公式：layers × kv_heads × head_dim × 2(K,V) × seq_len × 字节
MiniMind-3：8 × 4 × 96 × 2 × 2B = 12288 B ≈ 12KB/token
32K 上下文：≈ 384MB
```
GQA（kv_heads=4 而不是 8）让缓存直接减半——Day 3 学的 GQA 在这里兑现价值。

## generate 采样组件（简要）
- `temperature`：`logits / T` 控制分布锐利度（T 小更确定）
- `top_k` / `top_p`：截断候选集
- `repetition_penalty`：抑制重复
- `multinomial` 采样 vs `argmax` 贪心
- `eos_token_id`：生成终止条件

## 面试表述
"推理时自回归每步只生成一个新 token，K/V 会被未来所有 token 复用，因此缓存每层 K/V，每步只计算新 token 的 QKᵀ 并与缓存拼接，生成复杂度从 O(T³) 降到 O(T²)，代价是显存线性增长；GQA 通过减少 KV 头数进一步降低缓存占用。"




