# 简历项目经历：MiniMind 64M LLM 源码剖析与复现

## 项目名称

**MiniMind 64M LLM 源码剖析与复现**

## 项目简介

基于开源 MiniMind（Decoder-Only 架构，约 64M 参数）进行系统性源码剖析与核心模块手写复现，完整覆盖 LLM 从模型结构、预训练、指令微调、偏好对齐到推理生成的整条链路；以 Git 分支管理学习进度，沉淀 16 篇结构化笔记与 5 个可运行实验脚本，形成可复用的个人学习体系。

## 主要工作

1. **模型结构理解与手写复现**：逐行解读 `MiniMindConfig`、Embedding、RMSNorm、多头自注意力（GQA、`repeat_kv`、causal mask）、SwiGLU MLP、MoE 路由器与负载均衡辅助损失；独立手写 RMSNorm 与完整 Attention 计算流程，并与官方源码逐层对比输出一致；通过张量形状追踪脚本（`[B,T,C] → [B,T,H,D] → [B,H,T,T]`）验证每个模块的输入输出变化，建立对四维张量与矩阵运算的直观理解。

2. **预训练与微调全流程**：通读预训练训练循环，掌握梯度累积（`loss / accumulation_steps` 保持梯度尺度）、混合精度（`autocast` + `GradScaler`）、梯度裁剪、AdamW、余弦学习率调度与 checkpoint 断点续训机制；理解 SFT 数据格式与 label 掩码（仅 assistant 回复参与 loss，其余置 `-100`）；对比全量微调与 LoRA 参数高效微调（冻结主干、只训练低秩矩阵 A/B，实测仅训练 0.6% 参数，推理前可合并回原权重）。

3. **偏好对齐与推理优化**：理解 RLHF 三阶段（SFT → 奖励模型 → PPO）及 PPO 四件套（actor/ref/reward/critic）与 KL 约束防奖励黑客机制；掌握 DPO 从 Bradley-Terry 偏好模型推导闭式解的思路（`-log σ(β·log-ratio)`）；实现推理侧 KV Cache 理解（生成复杂度由 O(T³) 降至 O(T²)，GQA 使缓存减半）与采样策略（temperature / top_k / top_p / repetition penalty）。

4. **工程与学习方法**：使用 `master/upstream` 同步上游 + 主题分支管理学习代码；产出 16 天体系化笔记与跨设备交接文档；独立排查环境问题（模块搜索路径、终端编码、依赖编译失败）。

## 项目成果

- 产出 `notes/day1~16` 共 16 篇结构化学习笔记及 5 个可运行实验脚本，覆盖模型结构、训练、微调、对齐、推理全链路；
- 手写 RMSNorm / Attention 并与官方实现对比验证；独立完成参数量与显存估算（64M 模型、LoRA 参数占比 0.611%、KV Cache 约 12KB/token）；
- 规划并在云端 GPU（单卡 3090，成本约 3 元、耗时约 2.3 小时）完成 pretrain + SFT 实际训练复现（进行中）。

## 技术栈

Python、PyTorch、Transformers、HuggingFace Tokenizers、uv、Git；LLM 相关：Decoder-Only Transformer、GQA、RoPE、RMSNorm、SwiGLU、MoE、BPE、Pretrain/SFT、LoRA、RLHF/DPO、KV Cache。

---

## 一句话亮点（简历空间有限时）

> 基于 MiniMind（64M 参数 LLM）逐行剖析并手写复现 RMSNorm、GQA 注意力、SwiGLU、MoE 等核心模块，完整掌握预训练/SFT/LoRA/DPO 训练链路与 KV Cache 推理优化，产出 16 篇结构化笔记。

## 面试展开建议

- 手写 Attention 时实际纠正过的两个理解：`scores` 是 `[B,H,T,T]` 而非 `[B,T,H,D]`；`o_proj` 不是词表层（`lm_head` 才是），它负责混合多头信息并投影回 hidden_size 供残差相加。
- 讲梯度累积时能说清"为什么每个 batch 都 backward、但攒够 N 步才 step"，以及 `loss / N` 保持梯度尺度与学习率语义的原因。
- LoRA 可以讲实测数字：64M 模型只训练 0.3932M 参数（0.611%），推理前 `W + BA` 合并回原权重。
- 云端训练是"进行中"状态，跑完后把成果里的表述改成具体数据（如 loss 收敛值、训练耗时）。
