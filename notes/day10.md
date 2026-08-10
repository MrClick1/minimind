# Day 10：SFT 数据格式与训练流程

- 对应源码：`dataset/lm_dataset.py` → `SFTDataset`；`trainer/train_full_sft.py`
- 目标：理解监督微调的数据格式、label 掩码，以及与预训练的差异。

## SFT 是什么
- 用人工标注的（指令, 回答）对话数据对预训练模型做监督微调，让模型学会对话 / 指令跟随。

## 数据格式
- JSONL，每行一个 `conversations` 列表，支持多轮 system / user / assistant，可选 `reasoning_content` / `tools` / `tool_calls`。
- `apply_chat_template` 把 messages 渲染成 ChatML 文本：`<|im_start|>role\n内容<|im_end|>`。

## label 掩码（核心）
- `labels` 初始全为 `-100`（cross entropy 的 ignore_index，不参与 loss）。
- `generate_labels` 用 `bos_id = tokenizer("<bos>assistant\n")` 定位回答起点，`eos_id = tokenizer("<eos>\n")` 定位终点。
- 只有 assistant 回复区间（含 `<think>`、`<|im_end|>`）的 token 被赋真实 label。
- 效果：模型只学习"如何回答"，不学习预测用户指令。

## 与预训练训练循环的差异（train_full_sft.py）
| 项目 | pretrain | full_sft |
|---|---|---|
| 数据 | PretrainDataset（纯文本） | SFTDataset（对话） |
| 起始权重 | 随机（`from_weight=none`） | pretrain 权重（默认 `from_weight=pretrain`） |
| 学习率 | 5e-4 | 1e-5（低一个数量级，保护预训练知识） |
| batch_size | 32 | 16 |
| max_seq_len | 340 | 768 |
| accumulation_steps | 8 | 1 |
| 训练循环 | 相同 | 相同（loss/acc → backward → 累积 → clip → step → 保存） |

## 实测演示（单条对话）
- 模板：`<|im_start|>system\n...<|im_end|> <|im_start|>user\n...<|im_end|> <|im_start|>assistant\n<think>\n</think>\n1+1等于2。<|im_end|>`
- `labels`：system / user 区域全为 -100；assistant 区域为真实 token（含 `<think>`、`</think>`、答案、`<|im_end|>`）。

## 面试表述
"SFT 用对话数据微调预训练模型，核心是 label 掩码：只有 assistant 回复部分参与交叉熵，用户指令用 -100 忽略；训练循环与预训练一致，但学习率低一个数量级，从预训练权重继续。"
