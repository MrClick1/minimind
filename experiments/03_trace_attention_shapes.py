import sys
from pathlib import Path

import torch

sys.path.insert(
    0,
    str(Path(__file__).resolve().parent.parent),
)

from model.model_minimind import (
    MiniMindConfig,
    MiniMindForCausalLM,
    repeat_kv,
)


def main() -> None:
    config = MiniMindConfig(
        hidden_size=128,
        num_hidden_layers=2,
        vocab_size=6400,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=256,
        flash_attn=False,
        use_moe=False,
    )

    model = MiniMindForCausalLM(config)
    model.eval()

    batch_size = 2
    seq_len = 6

    input_ids = torch.randint(
        low=0,
        high=config.vocab_size,
        size=(batch_size, seq_len),
        dtype=torch.long,
    )

    # [B, T] -> [B, T, C]
    x = model.model.embed_tokens(input_ids)

    attention = model.model.layers[0].self_attn

    # 线性投影后的形状
    xq = attention.q_proj(x)
    xk = attention.k_proj(x)
    xv = attention.v_proj(x)

    print("input_ids:", input_ids.shape)
    print("embedding x:", x.shape)

    print("\n投影后：")
    print("raw Q:", xq.shape)
    print("raw K:", xk.shape)
    print("raw V:", xv.shape)

    # [B, T, H * D] -> [B, T, H, D]
    xq = xq.view(
        batch_size,
        seq_len,
        attention.n_local_heads,
        attention.head_dim,
    )

    xk = xk.view(
        batch_size,
        seq_len,
        attention.n_local_kv_heads,
        attention.head_dim,
    )

    xv = xv.view(
        batch_size,
        seq_len,
        attention.n_local_kv_heads,
        attention.head_dim,
    )

    print("\n拆分注意力头后：")
    print("Q:", xq.shape)
    print("K:", xk.shape)
    print("V:", xv.shape)

    # GQA：复制 K/V 头
    xk = repeat_kv(xk, attention.n_rep)
    xv = repeat_kv(xv, attention.n_rep)

    print("\nrepeat_kv 后：")
    print("K:", xk.shape)
    print("V:", xv.shape)

    # [B, T, H, D] -> [B, H, T, D]
    xq = xq.transpose(1, 2)
    xk = xk.transpose(1, 2)
    xv = xv.transpose(1, 2)

    print("\ntranspose 后：")
    print("Q:", xq.shape)
    print("K:", xk.shape)
    print("V:", xv.shape)

    # Attention 分数
    scores = (
        xq @ xk.transpose(-2, -1)
    ) / (attention.head_dim ** 0.5)

    print("\nattention scores:", scores.shape)

    causal_mask = torch.full(
        (seq_len, seq_len),
        float("-inf"),
    ).triu(1)

    print("\ncausal mask:")
    print(causal_mask)


if __name__ == "__main__":
    main()