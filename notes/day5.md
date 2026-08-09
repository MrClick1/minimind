# Day 5：RMSNorm 与 RoPE

- 对应分支：`learn/day5-rmsnorm-rope`
- 状态：已开始
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
