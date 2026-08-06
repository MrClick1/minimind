# Day 2：完整前向流程与张量形状

- 对应分支：`learn/day2-forward-flow`
- 对应实验：`experiments/02_trace_forward_shapes.py`
- 目标：理解从 token 到 logits 的完整形状主线，以及 Transformer Block 的基本结构。

## 要点

### 1. 完整前向主线

```text
input_ids → Embedding → 多个 MiniMindBlock → Final RMSNorm → lm_head → logits → loss
```

### 2. 张量符号

```text
B = batch size（几条序列）
T = sequence length（每条序列几个 token）
C = hidden size（隐藏维度）
V = vocab size（词表大小）
H = attention heads（头数）
D = head dimension（每头维度）
```

### 3. 形状变化（实测）

```text
input_ids     [2, 16]
embedding     [2, 16, 128]
hidden_states [2, 16, 128]
logits        [2, 16, 6400]
loss          ≈ 8.66
```

`[2, 16]` 表示 2 条序列、每条 16 个 token ID；Embedding 把每个 token ID 变成 128 维向量，所以变成 `[2, 16, 128]`。

### 4. Transformer Block 基本结构

```text
x
├─ RMSNorm → Attention ─┐
└────────────────────── +
                         │
├─ RMSNorm → MLP ───────┐
└────────────────────── +
```

即 **Pre-Norm + Residual Connection**（先归一化再做子层，输出与输入相加）。

### 5. `lm_head` 与 `o_proj` 的区别（提前铺垫，Day 4 详解）

- `lm_head`：`Linear(hidden_size, vocab_size)`，把 `[B,T,C]` 变成 `[B,T,V]`，产生词表概率；
- `o_proj`：Attention 内部的输出投影，输出仍是 `[B,T,C]`，不是词表层。

## 遇到的问题与答案

### `ModuleNotFoundError: No module named 'model'`

- **现象**：在 `experiments` 目录下运行 `02_trace_forward_shapes.py` 报错。
- **原因**：脚本运行目录 `experiments/` 被加入 `sys.path`，而 `model/` 在项目根目录。
- **答案**：脚本开头加入：

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
```

含义：当前文件 → `experiments` 目录 → 向上两级得到项目根目录 → 插入模块搜索路径。

## 验收

- 能说出 `[2,16] → [2,16,128] → [2,16,6400]` 每一步由哪个模块完成；
- 能画出 Pre-Norm + Residual 的结构；
- 能区分 `lm_head` 和 `o_proj` 的职责（各自输出什么形状）。
