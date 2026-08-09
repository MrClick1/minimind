# Day 6：MLP 与 SwiGLU

- 对应分支：`learn/day6-mlp-swiglu`
- 状态：已开始，当前先学习概念与源码结构
- 延后实践：Day4 独立 Attention 任务见 `notes/day4.md`；Day5 RoPE 实践见 `notes/day5.md`

## 学习目标

1. 理解 Transformer 已经有 Attention，为什么还需要 MLP。
2. 理解 MLP 对每个 token 分别计算，不负责 token 之间的信息交换。
3. 理解 `hidden_size` 和 `intermediate_size` 的区别。
4. 理解 `gate_proj`、`up_proj`、`down_proj` 的作用和张量形状。
5. 理解 SiLU 激活函数的作用。
6. 理解 SwiGLU 的核心计算：

   ```python
   down_proj(silu(gate_proj(x)) * up_proj(x))
   ```

7. 对照 MiniMind 的 `FeedForward` 源码串起完整前向流程。
8. 等准备好实践时，再手写一个最小版 SwiGLU 并核对输出形状。

## 学习顺序

1. 先区分 Attention 和 MLP 的职责。
2. 再沿着源码观察三次线性投影和中间张量形状。
3. 接着理解 SiLU 和门控乘法。
4. 最后再决定是否进行代码实践。
