# Day 13：LoRA（参数高效微调）

- 对应源码：`model/model_lora.py`、`trainer/train_lora.py`

## 核心思想
- 全量微调更新所有参数；LoRA 冻结原权重 W，只训练两个低秩矩阵 A、B。
- 假设：微调的权重变化 ΔW 是**低秩**的，用 `ΔW ≈ B·A` 近似（A: d×r，B: r×d，r 远小于 d）。
- 前向：`y = Wx + BAx`（原始输出 + 低秩增量）。
- 初始 `B=0` → 增量 `BA=0` → 输出与基座模型完全一致；A 高斯初始化（std=0.02）。
- 每个投影的 LoRA 参数量 = `d×r + r×d = 2dr`。

## 源码实现（model_lora.py）
- `LoRA` 类：`A = Linear(in, rank)`，`B = Linear(rank, out)`，`forward = B(A(x))`。
- `apply_lora`：遍历所有 `nn.Linear` 且 `in_features == out_features` 的模块（MiniMind 里就是 q_proj / o_proj，8 层共 16 个），挂 `lora` 子模块并 monkey-patch forward：`layer1(x) + layer2(x)`（闭包默认参数绑定，避免循环变量捕获的经典 Python 坑）。
- `save_lora` / `load_lora`：只保存/读取 A、B 权重。
- `merge_lora`：把 BA 合并进 W（`W' = W + BA`），推理时无需 LoRA 结构、无额外开销。

## 实测参数统计（rank=16，dense 768/8 层）
- 总参数 64.305M；LoRA 0.3932M；占比 **0.611%**。
- 挂载模块：8 层 ×（q_proj + o_proj）。
- 说明：MiniMind 简化版只处理方阵（in==out）；生产环境（peft）会把 LoRA 应用到 k/v、MLP 等非方阵投影。

## 训练差异（train_lora.py）
- `apply_lora` 后：非 lora 参数 `requires_grad=False`，`optimizer = AdamW(lora_params)`，`clip_grad_norm_(lora_params)` 只针对 LoRA 参数。
- `from_weight` 默认 `full_sft`（在 SFT 基座上做垂直微调，如医疗/法律）；学习率 1e-4。
- 保存：`save_lora` 只存 LoRA 权重（体积小），resume 存完整训练状态。
- monkey-patch 与 `torch.compile` 不兼容 → `use_compile` 自动关闭。

## 为什么省显存
- 冻结参数不求梯度、不进优化器 → 省掉它们的梯度和 AdamW 两个矩的显存。
- 例：64M 模型全量 AdamW 状态约 512MB；LoRA 只需 0.39M 参数对应的状态（约 3MB）。

## 面试表述
"LoRA 假设微调权重变化是低秩的，冻结原权重，只训练 A、B 两个低秩矩阵，前向为 Wx + BAx，初始 B=0 保证起点与基座一致；训练时只更新约 0.6% 的参数，大幅节省梯度与优化器显存；推理前可把 BA 合并回 W，不增加任何推理开销。"
