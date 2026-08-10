# Day 12：推理采样策略（temperature / top_k / top_p / 重复惩罚）

- 对应源码：`model/model_minimind.py` → `MiniMindForCausalLM.generate`
- 默认参数：temperature=0.85、top_k=50、top_p=0.85、repetition_penalty=1.0、do_sample=True

## 每步生成流程
```text
logits（最后一个 token 的 [V] 分数）
→ / temperature（温度缩放）
→ repetition_penalty（惩罚已出现 token）
→ top_k / top_p（截断候选集，被淘汰的置 -inf）
→ softmax → multinomial 采样（do_sample=True）或 argmax（do_sample=False）
→ eos 判断终止
```

## temperature：控制分布的锐利程度
- `softmax(logits / T)`：T 越大分布越平（更随机），T 越小越尖（更确定），T→0 趋近 argmax。
- 实测（logits=[0.1,2,1,0.5,3]）：
```text
T=0.1  → [0, 0, 0, 0, 1.0]        几乎 one-hot
T=0.85 → [0.022, 0.207, 0.064, 0.036, 0.672]
T=3.0  → [0.125, 0.235, 0.169, 0.143, 0.328]  接近均匀
```

## top_k：只保留概率前 k 个
```python
logits[logits < torch.topk(logits, top_k)[0][..., -1, None]] = -float('inf')
```
低于第 k 名的分数全部置 -inf，softmax 后概率为 0。

## top_p（nucleus，核采样）：累计概率到 p 的最小集合
```python
sorted_logits, sorted_indices = torch.sort(logits, descending=True)
mask = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1) > top_p
mask[..., 1:], mask[..., 0] = mask[..., :-1].clone(), 0   # 保留累计概率刚好超过 p 的最小集合
logits[mask.scatter(1, sorted_indices, mask)] = -float('inf')
```
按分数从高到低累加概率，超过 p 后剩下的全部淘汰。

## repetition_penalty：抑制重复
```python
score = logits[i, seen]
logits[i, seen] = torch.where(score > 0, score / penalty, score * penalty)
```
已出现过的 token：正分数被除以 penalty（压低），负分数被乘 penalty（拉高，绝对值变小）。

## repetition_penalty 详解（为什么能压制重复）

问题：生成时模型容易陷入重复循环（如"今天天气很好，今天天气很好，今天天气很好..."），贪心解码尤其明显。

机制：每步生成前，先找出**当前序列里已经出现过的 token**（`seen = torch.unique(input_ids[i])`），把它们的分数压下去：
```python
logits[i, seen] = torch.where(score > 0, score / penalty, score * penalty)
```
- 正分数 → 除以 penalty（变小）
- 负分数 → 乘以 penalty（变更负）

为什么分正负两个分支：softmax 的分数有正有负。如果统一"除以 penalty"，负分会变成"更接近 0"（反而变大）；统一"乘 penalty"，正分会变大——都会弄巧成拙。两个分支保证**无论正负，出现过 token 的相对概率都下降**。

数字演示（penalty=1.3，token 1、3 已出现过）：
```text
token  原分数  出现过  惩罚后   惩罚前概率  惩罚后概率
  0     5.0     否     5.000    0.9440    0.9613   ↑
  1     2.0     是     1.538    0.0470    0.0302   ↓
  2     0.0     否     0.000    0.0064    0.0065
  3    -1.0     是    -1.300    0.0023    0.0018   ↓
  4    -3.0     否    -3.000    0.0003    0.0003
```
已出现 token 概率下降，未出现 token 相对概率上升 → 模型倾向说"新话"，重复被抑制。
- penalty=1.0：不惩罚（默认）；越大抑制越强；过大容易让输出变得不连贯（刻意避开常见词）。
## 采样 vs 贪心
- `do_sample=True`：softmax 后 `multinomial` 按概率随机抽 → 输出多样；
- `do_sample=False`：`argmax` 每步选概率最大 → 确定性，但容易重复、呆板。

## 面试表述
"生成时对最后一个 token 的 logits 做 temperature 缩放控制随机性，top_k/top_p 截断低概率候选减少乱编，repetition_penalty 压低重复 token 的分数，最后 softmax 采样；贪心解码确定性高但易重复，采样多样性好但需要这些约束控制质量。"

