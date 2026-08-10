# Day 9：Pretrain 训练循环（一）—— 梯度累积

- 对应分支：`learn/day6-mlp-swiglu`（当前接续分支）
- 对应源码：`trainer/train_pretrain.py` → `train_epoch()`
- 默认配置：`accumulation_steps=8`、`batch_size=32`、`learning_rate=5e-4`、`grad_clip=1.0`

## 已完成
- 前向 → loss → `backward()` → `optimizer.step()` → `zero_grad()` 基本循环

## 本次要点：梯度累积（gradient accumulation）

### 为什么需要
- 显存限制：真正的目标 batch size（32×8=256）一次放不下，所以拆成 8 个 micro-batch 依次跑。
- 梯度是线性的：`Σ ∇L_i = ∇(Σ L_i)`，分次 `backward()` 累加的梯度 = 一次性用大 batch 算出的梯度。
- 等效 batch size = `batch_size × accumulation_steps`。

### 核心代码
```python
loss = res.loss + res.aux_loss
loss = loss / args.accumulation_steps      # 除 N，防止梯度被放大 N 倍
scaler.scale(loss).backward()              # 每个 micro-batch 都 backward（只累加梯度）
if step % args.accumulation_steps == 0:    # 每 N 步才更新一次
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
    scaler.step(optimizer)                 # 用累积的梯度更新参数
    scaler.update()
    optimizer.zero_grad(set_to_none=True)  # 清空梯度，开始新一轮累积
```

### 为什么 loss 要除以 N
- 若 8 个 batch 的 loss 直接相加再 backward：梯度 = `Σ∇L_i` = 8 × 平均梯度，等于把学习率放大了 8 倍，学习率语义就变了。
- 除以 N 后：每个 batch 贡献 `∇L_i/N`，累积 N 次 = `(1/N)Σ∇L_i` = 平均梯度。
- 等价于以"N 个 batch 的平均 loss"为优化目标，学习率保持原语义。

### 逐步走一遍（N=8）
```text
step 1: loss=L1/8 → backward → 梯度 = ∇L1/8
step 2: loss=L2/8 → backward → 梯度 = (∇L1+∇L2)/8
...
step 8: 梯度 = (∇L1+...+∇L8)/8；step%8==0 → clip → step → zero_grad
step 9~16：再来一轮
```

### 循环末尾的收尾逻辑
```python
if last_step > start_step and last_step % args.accumulation_steps != 0:
    scaler.unscale_(optimizer)
    clip_grad_norm_
    scaler.step(optimizer); scaler.update()
    optimizer.zero_grad(set_to_none=True)
```
作用：训练结束时如果步数不是 N 的整数倍（如 step=20，20%8=4），把剩余 4 步累积的梯度也更新一次，不浪费。

### 日志里为什么乘回来
```python
current_loss = loss.item() * args.accumulation_steps
```
因为 loss 已被除以 N，打印时乘回 N，显示"真实 batch 平均 loss"，便于与不累积的配置对比。

### 面试表述
"显存不足时可用梯度累积扩大等效 batch：每个 micro-batch 都 `backward()` 累加梯度，攒够 N 步再 `optimizer.step()`；`loss` 需除以 N，保证梯度尺度不变、学习率语义不变。"

## 混合精度（autocast + GradScaler）

### 为什么需要
- 速度和显存：FP16/BF16 计算更快（Tensor Core）、显存占用减半。
- FP16 的问题：数值范围窄（最大 ~65504，最小正常值 ~6e-5），梯度经常很小，直接存 FP16 会下溢成 0。

### 三个组件
1. `autocast`：前向时自动选择精度——矩阵乘法等用 FP16，Softmax/LayerNorm/损失等敏感运算仍用 FP32（所以叫"混合"精度）。
2. `GradScaler`：把 loss 放大（默认 2^16），让梯度在 FP16 反向传播时不至于下溢；更新前再除回去。
3. `unscale → clip → step → update` 的顺序不能乱：先还原梯度，再裁剪，再更新，最后根据是否溢出调整缩放系数。

### 源码（train_pretrain.py）
```python
dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
autocast_ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast(dtype=dtype)
scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype == 'float16'))

with autocast_ctx:
    res = model(input_ids, labels=labels)
    loss = res.loss + res.aux_loss
    loss = loss / args.accumulation_steps

scaler.scale(loss).backward()                  # 放大 loss 后反向
if step % args.accumulation_steps == 0:
    scaler.unscale_(optimizer)                 # 梯度除回缩放系数（还原真实大小）
    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)  # 裁剪用真实梯度
    scaler.step(optimizer)                     # 有 inf 则跳过本次更新
    scaler.update()                            # 根据是否溢出调整缩放系数
    optimizer.zero_grad(set_to_none=True)
```

### 为什么 BF16 不需要 GradScaler
- BF16：8 位指数 + 7 位尾数，范围与 FP32 相同（只是精度低），梯度不会下溢 → `enabled=False`，scaler 全部变成空操作。
- FP16：5 位指数 + 10 位尾数，范围窄但精度高 → 需要 loss scaling。
- 项目默认 `--dtype bfloat16`，所以默认 GradScaler 不生效。

### 数字例子
- 真实梯度 1e-6，小于 FP16 最小正常值 ~6e-5，直接存会变 0。
- loss × 2^16 = 65536 后：1e-6 × 65536 ≈ 0.065，可表示；反向后再 ÷65536 还原。
- 若某步梯度溢出成 inf：`scaler.step` 跳过参数更新，`scaler.update` 把系数减半（如 65536 → 32768），避免持续溢出。

### 注意
- `scaler.scale(loss)` 不修改原 loss，日志里 `loss.item() * accumulation_steps` 仍是真实平均 loss。
- CPU 上 `autocast_ctx = nullcontext()`，不做混合精度。
## 梯度裁剪（clip_grad_norm_）

### 为什么需要
- 深层网络梯度是连乘，可能爆炸（梯度范数巨大）→ 参数一步跳飞 → loss 变 NaN。
- 梯度裁剪是安全网：不阻止梯度变大，但限制"更新步长"的上限。

### 数学原理
```text
total_norm = sqrt( Σ_p ||g_p||² )      # 所有参数梯度的全局 L2 范数
if total_norm > max_norm:              # 超阈值 → 统一等比缩小
    scale = max_norm / total_norm
    所有参数梯度 ×= scale              # 方向不变，长度变短
else:
    不动
```
- 保证 `||g||₂ ≤ max_norm`。
- 所有参数用同一个缩放系数 → 梯度方向（参数间的相对比例）完全保留。

### 源码位置（train_pretrain.py）
```python
scaler.unscale_(optimizer)                                  # 先还原梯度
torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)  # max_norm=1.0
scaler.step(optimizer)
```
- 必须在 `unscale_` 之后：缩放前的梯度是放大 65536 倍的状态，直接算范数会错。
- 默认 `norm_type=2`（L2），MiniMind 只传 `max_norm=1.0`。
- 循环末尾收尾块里也有一份同样的裁剪。

### 演示数字（max_norm=1.0）
```text
裁剪前：p1.grad=[3,4]（范数5），p2.grad=[0,12]（范数12），全局范数=13
裁剪后：所有梯度 ×1/13 → [0.2308,0.3077] 和 [0,0.9231]，全局范数=1.0
比例 p1[0]/p2[1] = 0.25 裁剪前后不变（方向保留）
范数未超阈值（如 0.707 < 1.0）时原样保留
```

### 面试表述
"梯度裁剪在参数更新前计算所有梯度的全局 L2 范数，超过阈值就统一等比缩放，保证方向不变但步长受限，防止梯度爆炸导致训练崩溃。"
## 补充：什么是范数（norm）

- 范数 = 衡量一个向量"长度 / 大小 / 幅度"的函数。
- L2 范数（最常见）：`||x||₂ = sqrt(x₁² + x₂² + ... + x_n²)`，就是几何里的"直线距离"（勾股定理推广到任意维）。
  - 例：向量 `(3, 4)` → `sqrt(9+16) = 5`；向量 `(1, 2, 2)` → `sqrt(1+4+4) = 3`。
- L1 范数：`|x₁| + |x₂| + ... + |x_n|`，像曼哈顿街区只能横竖走的路程。
- L∞ 范数：`max(|x₁|, ..., |x_n|)`，只看最大的那个分量。
- 梯度裁剪里的"全局范数"：把模型**所有参数**的梯度首尾拼接成一个巨大的向量，再算 L2 长度。
  - 例：梯度 `[3,4]` 和 `[0,12]` 拼接成 `(3,4,0,12)` → 范数 = `sqrt(9+16+0+144) = 13`。
- 机器学习里的常见用途：梯度裁剪（衡量梯度大小）、L1/L2 正则化（惩罚参数大小）、向量相似度（两个向量差的长短）。
## AdamW、学习率调度与 checkpoint

### AdamW
- 源码：`optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)`（默认 lr=5e-4）。
- Adam：每个参数有自适应学习率——维护一阶矩（梯度均值，相当于动量）和二阶矩（梯度平方均值，控制缩放），并做偏置校正。
- W（Weight Decay）：权重衰减与自适应学习率**解耦**——Adam 里 weight decay 混在梯度中会被二阶矩缩放；AdamW 直接对参数本身做衰减。
- 训练 Transformer 的事实标准。

### 学习率调度（余弦衰减）
- 源码 `get_lr`：
  ```python
  return lr * (0.1 + 0.45 * (1 + math.cos(math.pi * current_step / total_steps)))
  ```
- 起点 step=0：cos=1 → 1.0×lr（满学习率）
- 中点：cos=0 → 0.55×lr
- 终点：cos=-1 → 0.1×lr（衰减到 10%，不为 0）
- 循环里每个 step 重新计算并写进 `optimizer.param_groups[...]['lr']`。
- 为什么：训练后期用小学习率精细收敛、更稳定。

### checkpoint 保存（两类文件）
- `{weight}_{hidden_size}.pth`：**纯权重**（state_dict，half + cpu），用于推理和加载模型。
- `{weight}_{hidden_size}_resume.pth`：**完整续训数据**（model + optimizer + scaler + epoch + step + world_size + wandb_id）。
- 原子写入：先写 `.tmp` 再 `os.replace`，防止保存中途损坏。
- 保存前 `model.eval()`，保存后切回 `model.train()`。
- DDP / torch.compile 包装时取 `raw_model`（`.module` / `_orig_mod`）。

### 恢复训练
- `--from_resume 1`：加载 resume 文件 → `model/optimizer/scaler.load_state_dict(...)` + `start_epoch/start_step`。
- world_size（GPU 数）变化时 step 自动换算：`step * saved_ws // current_ws`。
- 数据顺序确定：`setup_seed(42 + epoch)` 固定每轮 shuffle → `SkipBatchSampler` 跳过前 `start_step` 个 batch → 从断点继续。
- 入口：`train_epoch(epoch, loader, len(loader) + skip, start_step, wandb)`。
## 完整串起 train_pretrain.py

### 运行方式（README 官方）
```powershell
cd trainer
uv run python train_pretrain.py            # 从头训练；加 --from_resume 1 续训
```
相对路径以 `trainer` 目录为基准：数据 `../dataset/pretrain_t2t_mini.jsonl`，输出 `../out` 与 `../checkpoints`。

### 启动到结束的 9 段流水线
```text
1. 解析命令行参数（epochs / batch_size / lr / accumulation_steps / grad_clip / dtype / max_seq_len / from_weight / from_resume ...）
2. 初始化分布式环境 + 随机种子 setup_seed(42 + rank)
3. 建目录；MiniMindConfig(hidden=768, layers=8, use_moe)；from_resume 时检查 checkpoint
4. 混合精度：autocast_ctx（GPU → autocast(dtype)；CPU → nullcontext）
5. wandb（可选）
6. init_model（from_weight 加载或随机初始化）→ PretrainDataset → DataLoader → GradScaler（仅 fp16 生效）→ AdamW
7. 断点恢复：model / optimizer / scaler.load_state_dict + start_epoch / start_step
8. torch.compile / DDP 包装
9. for epoch：固定种子 randperm shuffle → SkipBatchSampler → DataLoader → train_epoch
```

### train_epoch 一个 step 的完整流程
```text
① 计算当前 lr（余弦调度）→ 写入 optimizer.param_groups
② autocast 前向 → loss = res.loss + aux_loss；loss /= accumulation_steps
③ scaler.scale(loss).backward()             # 累加梯度
④ 每 accumulation_steps 步：unscale_ → clip_grad_norm_ → step → update → zero_grad
⑤ 日志：loss.item() × accumulation_steps 显示真实平均 loss
⑥ 每 save_interval 步：保存纯权重 pth + resume pth
⑦ 循环末尾：残余未更新的梯度也 step 一次
```

### 断点续训的原理链
```text
resume 文件保存 optimizer / scaler 状态 → 学习率调度与 Adam 动量不丢失
setup_seed(42 + epoch) 固定每轮 shuffle → SkipBatchSampler 跳过前 start_step 个 batch
→ 数据顺序和 step 时间轴都从中断处接上
```
## 设备量级与成本（官方 + 实测）

### 官方参考（README）
- 作者完整复现环境：8× RTX 3090 (24GB) + 128GB RAM。
- 单卡 3090（租卡约 1.3￥/h）即可复现：
  - `minimind-3`（64M）：pretrain_t2t_mini ≈ 1.21h ≈ 1.57￥；sft_t2t_mini ≈ 1.10h ≈ 1.43￥；合计 ≈ 2.3h ≈ 3.0￥。
  - `minimind-3-moe`（198M-A64M）：pretrain ≈ 1.69h ≈ 2.20￥；sft ≈ 1.54h ≈ 2.00￥。
  - 8× H100 可压缩到分钟级。
- 数据：`pretrain_t2t_mini.jsonl` 约 1.2GB。

### 64M dense 默认配置的显存估算
- 参数 64M：fp32 权重 ~256MB，bf16 计算副本 ~128MB，AdamW 两个 fp32 矩 ~512MB，梯度 ~128~256MB。
- 激活值（batch=32, seq=340, 8 层）约 1~3GB（主要变量）。
- 合计约 2~5GB → 8GB 显存的卡就能跑默认 mini 训练；24GB（3090/4090）很舒适。
- bf16 需要 Ampere 及以上（RTX 30 系以后）；老卡用 `--dtype float16`（自动启用 GradScaler）。

### 公司电脑实测（2026-08-10）
- PyTorch `2.13.0+cpu`，`torch.cuda.is_available() = False`，内存约 16GB。
- 结论：当前这台机器**不能 GPU 训练**，只能跑 CPU 小实验（fp32，很慢）。
- 选项：a) 装 CUDA 版 PyTorch + NVIDIA 显卡；b) 租云 GPU（3090 ~1.3￥/h）；c) 学习阶段用小配置在 CPU 上验证训练循环。
## 训练计划决定（2026-08-10 公司电脑）

- 不在公司电脑上做预训练（当前为 CPU 环境，无 GPU）。
- 回家后租云服务器（参考：单卡 3090 ~1.3￥/h）进行实际训练。
- 公司电脑继续以理论学习 + 小规模 CPU 验证为主。
## 下一步
- `autocast` 与 `GradScaler`（混合精度）
- `clip_grad_norm_`（梯度裁剪）
- AdamW、学习率调度、checkpoint 保存与恢复
- 完整串起 `train_pretrain.py`







