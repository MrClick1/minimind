# MiniMind 学习项目交接文档

> 最后更新：2026-08-06  
> 用途：供另一台主机上的本地大模型、Coding Agent 或新的对话会话快速接手当前学习进度。  
> 使用语言：中文。

---

## 1. 用户背景与学习目标

用户是研究生，目标岗位为大模型 / Agent 开发，目前正在通过 MiniMind 项目系统学习：

- Decoder-Only Transformer 的模型结构；
- PyTorch 中张量形状的变化；
- 自回归语言模型的 next-token prediction；
- Multi-Head Attention、GQA、Causal Mask；
- 后续的 RMSNorm、RoPE、SwiGLU、MoE；
- Pretrain、SFT 等训练流程；
- 与学习过程配套的 Git 分支管理。

用户对 Python 和深度学习仍处于逐步建立体系的阶段。讲解时应：

1. 使用中文；
2. 从直观含义讲到代码和形状；
3. 对重要代码逐行说明；
4. 对 `[B,T,C]`、`view`、`transpose` 等张量操作给出具体数字示例；
5. 不要一次跳到过多高级概念；
6. 每个阶段最好配一个可运行的小实验；
7. 不要重复已经掌握的内容，除非用户主动追问。

---

## 2. 仓库信息

### 2.1 仓库

用户 Fork：

```text
https://github.com/MrClick1/minimind
```

原始上游：

```text
https://github.com/jingyaogong/minimind
```

本地常用路径：

```text
C:\Users\BHJ4SZH\Desktop\Study\minimind
```

### 2.2 Git Remote 约定

通常应为：

```text
origin   → https://github.com/MrClick1/minimind.git
upstream → https://github.com/jingyaogong/minimind.git
```

检查命令：

```bash
git remote -v
```

`master` 用于跟踪上游代码，不建议直接提交个人学习实验或交接文档。

同步上游的典型命令：

```bash
git switch master
git fetch upstream
git merge upstream/master
git push origin master
```

---

## 3. 当前远程分支及作用

目前已确认的远程学习分支：

```text
master
└── learn/base
    └── learn/day1-model-overview
        └── learn/day2-forward-flow
            └── learn/day3-attention
                └── learn/day4-attention-complete
```

### 分支定位

| 分支 | 主要内容 | 说明 |
|---|---|---|
| `master` | MiniMind 上游源码 | 保持尽量干净，用于同步 upstream |
| `learn/base` | 公共学习文档、`handoff.md` | 目前与 Day 4 已经发生分叉，不能假设它包含最新实验 |
| `learn/day1-model-overview` | `experiments/01_tiny_forward.py` | Tiny 模型最小前向传播 |
| `learn/day2-forward-flow` | `experiments/02_trace_forward_shapes.py` | 模型完整前向主线与形状追踪 |
| `learn/day3-attention` | `experiments/03_trace_attention_shapes.py` | Q/K/V、GQA、transpose、Attention Score、Causal Mask |
| `learn/day4-attention-complete` | `experiments/04_manual_attention.py`、同步后的 `handoff.md` | 当前包含最完整学习代码的分支 |

### 重要提醒

`learn/base` 后来单独增加了 `handoff.md` 提交，而 Day 4 分支也有自己独立的实验提交，因此两个分支出现了分叉。

执行：

```bash
git merge --ff-only learn/base
```

曾出现：

```text
Diverging branches can't be fast-forwarded
```

这是正常现象，表示两个分支都有对方没有的提交，不能仅移动分支指针完成快进合并。

为了只同步 `handoff.md`，用户已经采用了：

```bash
git switch learn/day4-attention-complete
git restore --source learn/base -- handoff.md
git add handoff.md
git commit -m "docs: 同步项目进度交接文档"
git push
```

因此，当前需要接续学习时，优先从：

```text
learn/day4-attention-complete
```

开始，而不是从旧状态的 `learn/base` 开始。

新一天建议创建：

```bash
git switch learn/day4-attention-complete
git pull --ff-only
git switch -c learn/day5-rmsnorm-rope
git push -u origin learn/day5-rmsnorm-rope
```

后续可以考虑新建一个长期稳定分支：

```text
learn/current
```

让它始终指向最新完整进度，并在该分支维护唯一的 `handoff.md`。在没有整理前，不要擅自重写或强制移动现有分支历史。

---

## 4. 本地环境

### 4.1 系统与工具

```text
系统：Windows PowerShell
虚拟环境提示符：(minimind)
环境管理：uv
主要框架：PyTorch
```

用户已经能够成功运行 PyTorch 模型实验，因此旧交接文档中“尚未安装 PyTorch”的信息已经过时。

验证环境：

```powershell
uv run python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

运行实验时通常在项目根目录：

```powershell
cd C:\Users\BHJ4SZH\Desktop\Study\minimind
```

然后执行：

```powershell
uv run python .\experiments\03_trace_attention_shapes.py
```

---

## 5. 已解决的环境问题

### 5.1 `ModuleNotFoundError: No module named 'model'`

曾在 `experiments` 目录中运行：

```powershell
uv run .\02_trace_forward_shapes.py
```

报错：

```text
ModuleNotFoundError: No module named 'model'
```

原因：

- Python 直接运行脚本时，主要把脚本所在目录 `experiments/` 加入模块搜索路径；
- `model/` 位于项目根目录，与 `experiments/` 同级；
- 项目根目录没有自动进入 `sys.path`。

当前实验脚本采用的解决方案：

```python
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
```

含义：

```text
当前文件
→ experiments 目录
→ 再向上一级得到 minimind 项目根目录
→ 插入 Python 模块搜索路径
```

该方案适合当前学习实验。更规范的长期方案可以是把项目配置为可编辑安装包，或通过 `python -m` 运行模块，但现在不需要优先改造。

---

## 6. 已完成学习内容

## Day 1：Tiny 模型与最小前向传播

对应分支：

```text
learn/day1-model-overview
```

对应文件：

```text
experiments/01_tiny_forward.py
```

### 已理解内容

#### 1. `MiniMindConfig`

用户已阅读 `MiniMindConfig`，理解了：

- `hidden_size`
- `num_hidden_layers`
- `vocab_size`
- `num_attention_heads`
- `num_key_value_heads`
- `head_dim`
- `intermediate_size`
- `max_position_embeddings`
- `rms_norm_eps`
- `rope_theta`
- `use_moe`
- `num_experts`
- `num_experts_per_tok`
- `router_aux_loss_coef`
- `**kwargs`
- `super().__init__(**kwargs)`

Tiny 配置示例：

```python
MiniMindConfig(
    hidden_size=128,
    num_hidden_layers=2,
    vocab_size=6400,
    num_attention_heads=4,
    num_key_value_heads=2,
    max_position_embeddings=256,
    use_moe=False,
    flash_attn=False,
)
```

默认：

```text
head_dim = hidden_size // num_attention_heads
         = 128 // 4
         = 32
```

#### 2. `*args` 与 `**kwargs`

用户已理解：

- `*args` 收集位置参数，得到 tuple；
- `**kwargs` 收集关键字参数，得到 dict；
- 在函数定义中表示收集；
- 在函数调用中表示解包。

#### 3. Next-token prediction

用户已理解自回归语言模型的目标：

```text
输入“我”          → 预测“喜”
输入“我 喜”       → 预测“欢”
输入“我 喜 欢”    → 预测“编”
输入“我 喜 欢 编” → 预测“程”
```

Loss 对齐代码：

```python
x = logits[..., :-1, :]
y = labels[..., 1:]
```

原因：

- 最后一个 logits 没有当前序列中的下一个标签；
- 第一个 label 没有对应的历史输入；
- 因此 logits 与 labels 错开一位比较。

#### 4. 最小前向过程

用户已跑通或基本理解：

```text
input_ids
→ model
→ logits
→ shift labels
→ cross entropy loss
```

---

## Day 2：完整前向流程与张量形状

对应分支：

```text
learn/day2-forward-flow
```

对应文件：

```text
experiments/02_trace_forward_shapes.py
```

### 已理解内容

模型主线：

```text
input_ids
→ Embedding
→ 多个 MiniMindBlock
→ Final RMSNorm
→ LM Head
→ logits
→ loss
```

### 张量符号

```text
B = batch size
T = sequence length
C = hidden size
V = vocabulary size
H = attention heads
D = head dimension
```

### 形状变化

```text
input_ids     [B,T]
embedding     [B,T,C]
hidden_states [B,T,C]
logits        [B,T,V]
```

用户已经理解：

```text
[B,T]
= [多少条 token 序列, 每条序列多少个 token]
```

例如：

```text
[2,16]
= 2 条序列，每条 16 个 token ID
```

经过 Embedding：

```text
[2,16] → [2,16,128]
```

表示每个 token ID 被转换成一个 128 维向量。

### Transformer Block 基本结构

用户已经接触：

```text
x
├─ RMSNorm → Attention ─┐
└────────────────────── +
                         │
├─ RMSNorm → MLP ───────┐
└────────────────────── +
```

即 Pre-Norm + Residual Connection。

---

## Day 3：Attention 内部结构

对应分支：

```text
learn/day3-attention
```

对应文件：

```text
experiments/03_trace_attention_shapes.py
```

该实验已经成功运行。

### 实际输出

```text
input_ids: torch.Size([2, 4])
embedding x: torch.Size([2, 4, 128])

投影后：
raw Q: torch.Size([2, 4, 128])
raw K: torch.Size([2, 4, 64])
raw V: torch.Size([2, 4, 64])

拆分注意力头后：
Q: torch.Size([2, 4, 4, 32])
K: torch.Size([2, 4, 2, 32])
V: torch.Size([2, 4, 2, 32])

repeat_kv 后：
K: torch.Size([2, 4, 4, 32])
V: torch.Size([2, 4, 4, 32])

transpose 后：
Q: torch.Size([2, 4, 4, 32])
K: torch.Size([2, 4, 4, 32])
V: torch.Size([2, 4, 4, 32])

attention scores: torch.Size([2, 4, 4, 4])

causal mask:
tensor([
    [0., -inf, -inf, -inf],
    [0., 0., -inf, -inf],
    [0., 0., 0., -inf],
    [0., 0., 0., 0.]
])
```

### 已理解概念

#### 1. Q、K、V 投影层

源码：

```python
self.q_proj = nn.Linear(
    config.hidden_size,
    config.num_attention_heads * self.head_dim,
    bias=False,
)

self.k_proj = nn.Linear(
    config.hidden_size,
    self.num_key_value_heads * self.head_dim,
    bias=False,
)

self.v_proj = nn.Linear(
    config.hidden_size,
    self.num_key_value_heads * self.head_dim,
    bias=False,
)

self.o_proj = nn.Linear(
    config.num_attention_heads * self.head_dim,
    config.hidden_size,
    bias=False,
)
```

Tiny 配置下：

```text
hidden_size = 128
Query heads = 4
KV heads = 2
head_dim = 32
```

所以：

```text
q_proj = Linear(128,128)
k_proj = Linear(128,64)
v_proj = Linear(128,64)
o_proj = Linear(128,128)
```

PyTorch 中：

```text
Linear(in_features, out_features)
weight.shape = [out_features, in_features]
```

因此：

```text
q_proj.weight [128,128]
k_proj.weight [64,128]
v_proj.weight [64,128]
o_proj.weight [128,128]
```

权重初始是接近 0 的随机浮点数，训练时通过反向传播更新，最终学习出不同投影功能。

#### 2. 四维 Tensor

用户已建立以下理解：

```text
[B,T,H,D]
= 文本 → token → 注意力头 → 头内向量
```

定位一个数字：

```python
x[batch, token, head, dimension]
```

四维 Tensor 不需要强行想象为四维几何图形，应理解为四级嵌套分类或文件夹结构。

#### 3. `transpose(1,2)`

```text
[B,T,H,D] → [B,H,T,D]
```

含义：

```text
原来：按 token 组织多个 head
后来：按 head 组织整条序列的 token
```

元素对应：

```python
y[b, h, t, d] == x[b, t, h, d]
```

之前 `T=4`、`H=4`，所以形状数字在 transpose 前后看起来相同：

```text
[2,4,4,32] → [2,4,4,32]
```

但维度含义已交换。为避免混淆，后续实验建议使用：

```python
seq_len = 6
```

此时可以看到：

```text
[2,6,4,32] → [2,4,6,32]
```

#### 4. GQA 与 `repeat_kv`

```text
Query heads = 4
KV heads = 2
```

GQA 让多个 Query Head 共享较少的 K/V Head，从而减少参数和 KV Cache。

```text
Q0、Q1 使用同一组 K/V
Q2、Q3 使用另一组 K/V
```

`repeat_kv` 将：

```text
K/V [B,T,2,32]
```

扩展为：

```text
K/V [B,T,4,32]
```

方便与 4 个 Query Head 进行批量计算。

#### 5. Attention Score 为什么是 `[B,H,T,T]`

Q 和 K：

```text
Q  [B,H,T,D]
K  [B,H,T,D]
Kᵀ [B,H,D,T]
```

矩阵乘法：

```text
[B,H,T,D] × [B,H,D,T]
= [B,H,T,T]
```

两个 T 分别表示：

```text
第一个 T：Query token 数量
第二个 T：Key token 数量
```

准确索引：

```python
scores[batch, head, query_token, key_token]
```

#### 6. Causal Mask

因果掩码防止当前 token 看到未来 token：

```text
          K0 K1 K2 K3
Q0         ✓  ×  ×  ×
Q1         ✓  ✓  ×  ×
Q2         ✓  ✓  ✓  ×
Q3         ✓  ✓  ✓  ✓
```

未来位置被设为 `-inf`，Softmax 后对应概率约为 0。

---

## 7. Day 4 当前状态：完整 Attention

对应分支：

```text
learn/day4-attention-complete
```

对应文件：

```text
experiments/04_manual_attention.py
```

### 当前状态判断

- Day 4 分支和实验脚本已经创建并推送；
- 代码覆盖了完整 Attention 手工流程；
- 用户尚未明确反馈已经完整运行、逐项验收并学完；
- 因此接手模型应将 Day 4 视为“正在进行 / 待运行验证”，不要直接跳过。

### Day 4 要串起来的流程

```text
hidden_states
→ q_proj / k_proj / v_proj
→ 拆分多个 Head
→ repeat_kv
→ transpose
→ QKᵀ
→ 除以 sqrt(head_dim)
→ 加 causal mask
→ softmax
→ Attention Weight × V
→ 转回 [B,T,H,D]
→ 合并 Head
→ o_proj
→ [B,T,C]
```

核心公式：

```text
Attention(Q,K,V)
= softmax(QKᵀ / sqrt(D) + mask) V
```

### 接手后的第一项任务

在项目根目录运行：

```powershell
git switch learn/day4-attention-complete
git pull --ff-only
uv run python .\experiments\04_manual_attention.py
```

需要检查：

1. 所有形状是否符合预期；
2. `attention_weights` 是否为 `[B,H,T,T]`；
3. 每一行 Attention Weight 的和是否接近 1；
4. 未来 token 对应位置是否为 0；
5. `context` 是否由 `[B,H,T,T] @ [B,H,T,D]` 得到 `[B,H,T,D]`；
6. 合并 Head 后是否回到 `[B,T,128]`；
7. `o_proj` 后是否仍为 `[B,T,128]`。

预期形状（建议 `B=2,T=6,H=4,D=32`）：

```text
hidden_states:     [2,6,128]

投影后：
Q:                 [2,6,128]
K:                 [2,6,64]
V:                 [2,6,64]

拆头后：
Q:                 [2,6,4,32]
K:                 [2,6,2,32]
V:                 [2,6,2,32]

repeat_kv 后：
K:                 [2,6,4,32]
V:                 [2,6,4,32]

transpose 后：
Q/K/V:             [2,4,6,32]

scores:            [2,4,6,6]
attention_weights: [2,4,6,6]
context:           [2,4,6,32]
转回后:            [2,6,4,32]
合并 Head 后:      [2,6,128]
o_proj 后:         [2,6,128]
```

---

## 8. Day 4 需要重点讲清的问题

接手模型应逐个帮助用户回答：

### 1. 为什么除以 `sqrt(head_dim)`？

Q 和 K 的点积会随维度增大而增大。若分数绝对值过大，Softmax 会过于尖锐，容易导致梯度变小、训练不稳定。

### 2. 为什么 Softmax 使用 `dim=-1`？

最后一维代表所有 Key token。每个 Query token 都需要在所有可见 Key 之间分配权重，因此沿最后一维归一化。

### 3. 为什么 Attention Weight 要乘 V？

Q 和 K 决定“关注谁”，V 承载“需要读取的内容”。权重乘 V 是对所有 Value 向量做加权求和。

### 4. 为什么 `[T,T] @ [T,D] = [T,D]`？

每个 Query token 对 T 个 Value 分配权重，最终汇总成一个 D 维向量，所以 Key 的 T 维被求和消去。

### 5. 为什么需要 `o_proj`？

多个 Head 合并后只是拼接，`o_proj` 用于混合各个头的信息，并保证输出回到 `hidden_size`，以便残差相加。

---

## 9. 下一阶段计划

Day 4 完成后进入：

```text
Day 5：RMSNorm 与 RoPE
```

建议创建：

```bash
git switch learn/day4-attention-complete
git pull --ff-only
git switch -c learn/day5-rmsnorm-rope
git push -u origin learn/day5-rmsnorm-rope
```

### Day 5 学习顺序

先学 RMSNorm：

1. 为什么需要归一化；
2. RMSNorm 与 LayerNorm 的区别；
3. `rms_norm_eps` 的作用；
4. 为什么 MiniMind 使用 Pre-Norm；
5. 手写简化 RMSNorm，并与源码输出比较。

再学 RoPE：

1. Attention 本身为什么无法感知顺序；
2. RoPE 为什么只作用于 Q 和 K；
3. 相邻两个维度如何组成旋转平面；
4. `cos`、`sin` 张量的形状；
5. `apply_rotary_pos_emb` 的输入输出形状；
6. 先重视代码与直觉，不要求一开始完整推导复数公式。

Day 5 暂时不要进入：

- MoE；
- Flash Attention 源码；
- KV Cache 深层实现；
- Pretrain 训练脚本；
- SFT 数据格式。

这些内容应在 RMSNorm、RoPE、MLP/SwiGLU 学完后再继续。

---

## 10. 建议的后续路线

```text
Day 4  完整 Attention
Day 5  RMSNorm + RoPE
Day 6  MLP / SwiGLU + 完整 Transformer Block
Day 7  参数量计算 + 模型结构总结
Day 8  Tokenizer 与 Dataset
Day 9  Pretrain 数据流与训练循环
Day 10 SFT 数据格式与训练流程
之后   KV Cache、推理生成、LoRA、DPO、MoE
```

不要机械按天数推进。若用户仍不理解某一处 Tensor 形状，应优先补齐理解。

---

## 11. Git 使用认知

用户已了解：

### 创建分支

```bash
git switch -c learn/dayX-topic
```

### 推送并设置跟踪

```bash
git push -u origin learn/dayX-topic
```

`-u` 表示设置本地分支与远程分支的跟踪关系，以后可以直接：

```bash
git push
git pull
```

### 查看分支图

```bash
git log --oneline --graph --decorate --all
```

进入分页器后按：

```text
q
```

退出。

不进入分页器：

```bash
git --no-pager log --oneline --graph --decorate --all
```

### 只从另一个分支取一个文件

```bash
git restore --source learn/base -- handoff.md
```

该命令不会合并整个分支，只将目标文件恢复到当前工作区。

### 查看分支差异

```bash
git log --oneline HEAD..learn/base
git log --oneline learn/base..HEAD
git diff --stat learn/base...HEAD
```

### 本机 Git 代理说明

用户级 Git 配置中保留了指向 `http://127.0.0.1:7890` 的 HTTP/HTTPS 代理，不要因为本项目的连接问题删除或修改全局代理配置。

如果本项目执行 `clone`、`fetch`、`pull` 或 `push` 时出现无法连接 `127.0.0.1:7890`，只对当前 Git 命令临时禁用代理，例如：

```bash
git -c http.proxy= -c https.proxy= pull --ff-only
git -c http.proxy= -c https.proxy= push
```

该方式只影响本次命令，不会修改用户的全局 Git 配置。

### 安全原则

- 执行切换、merge、rebase 前先运行 `git status`；
- 不要在不理解时使用 `git reset --hard`；
- 不要随意使用 `git push --force`；
- 需要改写远程历史时，至少使用 `--force-with-lease`，且应先解释风险；
- 用户当前更适合普通 merge 或创建新分支，不优先使用 rebase 改写历史。

---

## 12. 关键文件

| 文件 | 作用 |
|---|---|
| `model/model_minimind.py` | Config、RMSNorm、RoPE、Attention、MLP、MoE、完整模型 |
| `experiments/01_tiny_forward.py` | Tiny 模型前向传播与 loss |
| `experiments/02_trace_forward_shapes.py` | 主模型各阶段张量形状 |
| `experiments/03_trace_attention_shapes.py` | Q/K/V、GQA、transpose、scores、mask |
| `experiments/04_manual_attention.py` | 手工串联完整 Attention 计算 |
| `requirements.txt` | 项目依赖 |
| `handoff.md` | 跨设备、跨模型交接文档 |

---

## 13. 接手模型的操作清单

接手后按以下顺序执行：

1. 查看当前分支和工作区：

   ```bash
   git status
   git branch --show-current
   ```

2. 获取远程状态：

   ```bash
   git fetch origin
   git branch -vv
   ```

3. 切换到当前最新学习分支：

   ```bash
   git switch learn/day4-attention-complete
   git pull --ff-only
   ```

4. 查看关键实验：

   ```text
   experiments/03_trace_attention_shapes.py
   experiments/04_manual_attention.py
   ```

5. 运行 Day 4：

   ```powershell
   uv run python .\experiments\04_manual_attention.py
   ```

6. 根据输出逐步解释完整 Attention，不要一次只给结论。

7. 帮助用户完成一份简短笔记：

   ```text
   notes/04_complete_attention.md
   ```

8. 验收用户是否能独立回答第 8 节中的五个问题。

9. 用户决定先进入 Day 5；Day 4 的独立复现、变化测试和概念复答已记录在 `notes/day4.md`，后续需要回来完成最终验收。

---

## 14. 交互偏好

- 用户经常从一行代码继续追问底层原因；
- 用户需要详细解释，但应保持主线清晰；
- 解释 Tensor 时优先使用具体形状，例如 `[2,6,4,32]`；
- 解释四维 Tensor 时使用“多级索引 / 文件夹结构”，不要只说“四维空间”；
- 解释矩阵乘法时写出最后两维：

  ```text
  [T,D] × [D,T] = [T,T]
  ```

- 解释 Git 时明确：

  ```text
  当前分支是谁
  来源分支是谁
  合并方向是什么
  是否会创建提交
  是否会改写历史
  ```

- 用户希望最终能将 MiniMind 学习成果用于大模型 / Agent 开发岗位的面试和项目表达，因此每个模块学完后可补充一段面试表述，但不要替代源码理解。

---

## 15. 当前一句话状态

用户已经完成 MiniMind 的 Tiny 前向传播、整体形状追踪和 Attention 主流程学习；Day 4 的独立复现任务保存在 `notes/day4.md`。Day 5 已完成 RMSNorm 的手写和源码对比，并理解 RoPE 的位置旋转、Q/K、二维配对与多频率等核心概念；RoPE 代码实践保存在 `notes/day5.md` 的延后任务中。当前开始 Day 6：MLP 与 SwiGLU，之后再回看 Day 4/Day 5 的实践验收。
