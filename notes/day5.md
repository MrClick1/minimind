# Day 5：RMSNorm 与 RoPE

- 对应分支：`learn/day5-rmsnorm-rope`
- 状态：RMSNorm 已完成手写与源码对比；RoPE 已理解核心概念，代码实践暂缓
- Day 4 延后验收：见 `notes/day4.md` 的“下一步”清单，后续回来完成，不在此处丢失。

## 学习目标

### 1. RMSNorm

1. 为什么 Transformer 需要归一化；
2. RMSNorm 的计算公式和每一步张量形状；
3. RMSNorm 与 LayerNorm 的区别；
4. `rms_norm_eps` 为什么存在；
5. MiniMind 为什么采用 Pre-Norm；
6. 手写简化 RMSNorm，并与源码输出比较。

### 2. RoPE

1. 为什么 Attention 需要位置信息；
2. RoPE 作用于 Q/K 而不是 V 的原因；
3. 二维旋转、偶数维与奇数维配对；
4. `cos` / `sin` 缓存的形状与广播；
5. 手工追踪一次 RoPE 前后的张量形状；
6. 对照 MiniMind 源码理解 `apply_rotary_pos_emb`。

## 本日顺序

先学习 RMSNorm，再进入 RoPE。第一项任务是阅读 `model/model_minimind.py` 中的 `RMSNorm` 实现，逐行解释公式和形状，不直接背代码。

## 阶段结果

### RMSNorm

- 已理解 RMSNorm 用于稳定每个 token 隐藏向量的数值尺度；
- 已理解 `mean(x²)`、`eps`、`rsqrt`、广播和可学习 `weight`；
- 已理解 `x.float()` 与恢复原 dtype 的原因；
- 已理解 RMSNorm 与 LayerNorm、Pre-Norm 的主要区别；
- 已完成 `experiments/05_manual_rmsnorm.py`；
- 已将手写 `MyRMSNorm` 与 MiniMind 源码实现进行对比。

### RoPE

- 已理解 Attention 为什么需要位置信息；
- 已理解 RoPE 旋转 Q/K 而不旋转 V 的原因；
- 已理解二维旋转公式和向量长度保持不变；
- 已理解每个 Head 的 `head_dim` 被组成二维旋转对；
- 已理解 token 位置决定角度倍数、不同维度组使用不同频率；
- 已理解 MiniMind 当前通过前半维与后半维配对，例如 `head_dim=32` 时配对 `(x0,x16)` 到 `(x15,x31)`。

## 延后实践任务

用户暂时不进行 RoPE 代码实践，先进入 Day 6。后续回来完成：

1. 新建 `experiments/05_manual_rope.py`；
2. 验证 `[1,0]` 旋转 90° 后得到接近 `[0,1]`；
3. 扩展到 `head_dim=8`，手工实现前半维与后半维配对；
4. 追踪 Q/K、`cos`、`sin` 的形状和广播过程；
5. 与 MiniMind 的 `apply_rotary_pos_emb` 做逐元素比较；
6. 完成以上任务后再将 RoPE 标记为最终验收通过。
