1. dimension 较大的情况下，乘出来的 attention score 会偏大，导致 softmax 之后概率会集中在较大值，导致反向传播时出现梯度消失或者梯度爆炸的问题。除 sqrt(head_dim) 可以帮助缩小 attention score 的大小。

2. 因为 scores 是 [B, T, H, D], 最后一个维度才是 attention score

3. V 是代表每个 token 可以给出的信息，attention weight 是每个 token（这里对未来 token 进行了 mask，看不见） 对当前 token 的重要性，然后加起来就得到（这里我不太会说）。

4. [T, T] @ [T, D] = [T, D] 是线性代数的基础，我也不太会说，如果要理解的话，那就是每行是每个 token 的权重，然后对后面每一行（也就是每个 token 的 dim 下的投影向量）进行加权，得到当前计算的这个 token 的结果向量？（我不清除这里怎么描述），一共计算 T 个 token，每个 token 向量的维度是 D

5. 因为需要把加权的 token 向量转换为对 token 词表的概率分布，用来选择预测的下一个 token