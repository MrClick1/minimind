# Day 1：Tiny 模型与最小前向传播

- 对应分支：`learn/day1-model-overview`
- 对应实验：`experiments/01_tiny_forward.py`
- 目标：跑通最小模型 forward，理解配置项、参数量统计、next-token prediction 的 loss 对齐方式。

## 要点

### 1. `MiniMindConfig` 关键字段

| 字段 | 含义 |
|---|---|
| `hidden_size` | 隐藏层维度（Tiny 配置为 128） |
| `num_hidden_layers` | Transformer Block 层数（2） |
| `vocab_size` | 词表大小（6400） |
| `num_attention_heads` | Query 头数（4） |
| `num_key_value_heads` | K/V 头数（GQA，2） |
| `max_position_embeddings` | 最大序列长度（256） |
| `use_moe` | 是否使用 MoE（Tiny 实验关闭） |
| `flash_attn` | 是否使用 Flash Attention |

默认计算：`head_dim = hidden_size // num_attention_heads = 128 // 4 = 32`

### 2. `*args` 与 `**kwargs`

- 在函数**定义**中：`*args` 收集位置参数为 tuple，`**kwargs` 收集关键字参数为 dict。
- 在函数**调用**中：`*` 表示解包序列，`**` 表示解包 dict。

### 3. Next-token prediction 的 loss 对齐（shift）

自回归目标：

```text
输入“我”          → 预测“喜”
输入“我 喜”       → 预测“欢”
输入“我 喜 欢”    → 预测“编”
```

Loss 对齐代码：

```python
x = logits[..., :-1, :]
y = labels[..., 1:]
```

原因：最后一个 logits 没有"下一个位置"的标签；第一个 label 没有对应的历史输入，所以 logits 与 labels 错开一位比较。

### 4. 最小前向流程

```text
input_ids → model → logits → shift labels → cross entropy loss
```

参数量统计：`sum(p.numel() for p in model.parameters())`

## 遇到的问题与答案

### `ModuleNotFoundError: No module named 'model'`（01 脚本目前仍存在）

- **现象**：在项目根目录执行 `python experiments/01_tiny_forward.py`，`import model.model_minimind` 报错。
- **原因**：Python 直接运行脚本时，主要把脚本所在目录 `experiments/` 加入模块搜索路径；`model/` 在项目根目录，与 `experiments/` 同级，不会自动进入 `sys.path`。
- **答案**：在 `import model` 之前插入：

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
```

- **现状**：`02`、`03`、`04` 已内置该修复；`01_tiny_forward.py` 尚未加，需要补上才能直接运行。

### 随机初始化时 loss 为什么约等于 ln(V)

- 同配置下 `02_trace_forward_shapes.py` 实测 loss ≈ 8.66，而 `ln(6400) ≈ 8.76`。
- 原因：模型未训练时 logits 近似均匀分布，每个 token 的交叉熵 ≈ `ln(vocab_size)`。

## 验收

- 能不看资料说出 Tiny 配置里每个字段的含义；
- 能解释为什么要 `logits[..., :-1, :]` 与 `labels[..., 1:]` 对齐；
- 补上 `sys.path` 修复后，01 脚本可以运行并打印 logits/loss/参数量。
