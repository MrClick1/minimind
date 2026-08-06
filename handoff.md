# 会话交接 (Handoff)

## 1. 背景

- **项目**: [MiniMind](https://github.com/jingyaogong/minimind) —— 一个面向 LLM 入门与复现的超小参数中文大语言模型项目（主线 MiniMind-3 约 64M 参数）。
- **当前分支**: `master`（git status 干净）。
- **目标**: 在本地 Windows 环境中用 `uv` 管理 Python 环境，并逐步跑通 `experiments/` 下的学习脚本，理解模型结构与训练链路。

## 2. 已完成工作

### 2.1 项目概览
- 浏览了项目结构、README、核心模型与数据集代码。
- 关键认知：
  - `model/model_minimind.py`: MiniMind-3 使用 Transformer Decoder-Only + Pre-Norm + RMSNorm + RoPE + GQA + SwiGLU，可选 MoE。
  - `dataset/lm_dataset.py`: 定义了 Pretrain / SFT / DPO / RLAIF / AgentRL 等多种 Dataset。
  - `requirements.txt`: 项目依赖，`torch` 被注释，需要按 CUDA/CPU 手动安装；`ujson==5.1.0` 在 Windows 下会触发源码编译失败。

### 2.2 环境搭建（进行中）
- 使用 `uv venv` 创建了 `.venv/` 虚拟环境。
- 将 `requirements.txt` 中 `ujson==5.1.0` 升级为 `ujson>=5.1.0`（或已手动安装 `ujson==5.13.0`），绕开了缺少 Microsoft C++ Build Tools 导致的编译错误。
- 尚未安装 `torch` / `torchvision`，这是下一步。

### 2.3 实验脚本运行与问题排查
- 已查看 `experiments/01_tiny_forward.py`、`02_trace_forward_shapes.py`、`03_trace_attention_shapes.py`。
- 解决了 `ModuleNotFoundError: No module named 'model'`：
  - 原因：`uv run python experiments/xx.py` 时，Python 的 `sys.path` 不包含项目根目录，只包含脚本所在目录 `experiments/`。
  - 解决方案：在脚本开头插入 `sys.path.insert(0, str(Path(__file__).resolve().parent.parent))`，把项目根目录加入模块搜索路径。
  - `03_trace_attention_shapes.py` 已经内置了该修复。

## 3. 未完成任务

- [ ] 安装 PyTorch（根据本地 CUDA/CPU 选择对应版本）。
- [ ] 完整跑通 `requirements.txt` 中剩余依赖。
- [ ] 实际运行并验证：
  - `experiments/01_tiny_forward.py`
  - `experiments/02_trace_forward_shapes.py`
  - `experiments/03_trace_attention_shapes.py`
- [ ] 若运行中出现新依赖/环境问题，继续排查。

## 4. 关键文件

| 文件 | 说明 |
|---|---|
| `README.md` | 项目总览、训练流程、数据说明、部署文档 |
| `model/model_minimind.py` | MiniMind 模型结构定义（Config / Attention / GQA / RoPE / MoE） |
| `dataset/lm_dataset.py` | 各阶段 Dataset 实现（Pretrain / SFT / DPO / RLAIF / AgentRL） |
| `requirements.txt` | 依赖列表，已修改 `ujson>=5.1.0` |
| `experiments/01_tiny_forward.py` | 最小化完整 forward 测试 |
| `experiments/02_trace_forward_shapes.py` | 模型前向各层输出形状追踪 |
| `experiments/03_trace_attention_shapes.py` | 注意力内部 Q/K/V 形状变化与 causal mask 可视化 |
| `handoff.md` | 本交接文件 |

## 5. 下一步计划

1. **安装 PyTorch**
   ```powershell
   # 有 CUDA 12.x 的推荐命令
   uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
   ```
   若为 CPU 或无 NVIDIA GPU，改用：
   ```powershell
   uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
   ```
2. **验证环境**
   ```powershell
   uv run python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
   ```
3. **跑通实验脚本**
   ```powershell
   uv run python experiments/01_tiny_forward.py
   uv run python experiments/02_trace_forward_shapes.py
   uv run python experiments/03_trace_attention_shapes.py
   ```
4. **继续学习**
   - 理解 `model_minimind.py` 中的 RoPE、`repeat_kv`、GQA、MoE 细节。
   - 进入 `trainer/` 训练脚本，理解 Pretrain / SFT 流程。
