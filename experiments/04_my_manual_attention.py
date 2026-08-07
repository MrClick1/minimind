import math
import sys

from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from model.model_minimind import (
    MiniMindConfig,
    MiniMindForCasualLM,
    repeat_kv,
)


def main(): -> None:
    torch.manual_seed(42)

    config = MiniMindConfig(
        hidden_size=128,
        num_hidden_layer=2,
        vocab_size=6400,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=256,
        flash_attn=False,
        use_mode=False,
    )

    model = MiniMindForCasualLM(config)
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
    hidden_states = model.model.embed_tokens(input_ids)

    attention = model.model.layers[0].self_attn

    batch_size, seq_len, _ = hidden_states.shape

    print("hidden_states:", hidden_states.shape)

    # 1. 生成 Q、K、V
    query = attention.q_proj(hidden_states)
    key = attention.k_proj(hidden_states)
    value = attention.v_proj(hidden_states)

    print("\n投影后：")
    print("query:", query.shape)
    print("key:", key.shape)
    print("value:", value.shape)

    # 2. 拆分注意力头
    query = query.view(
        batch_size,
        seq_len,
        attention.n_local_heads,
        attention.head_dim,
    )

    key = key.view(
        batch_size,
        seq_len,
        attention.n_local_heads,
        attention.head_dim,
    )

    value = value.view(
        batch_size,
        seq_len,
        attention.n_local_kv_heads,
        attention.head_dim,
    )

    print("\n拆分 Head 后：")
    print("query:", query.shape)
    print("key:", key.shape)
    print("value:", value.shape)

    # 3. GQA: 扩展 K/V Head
    key = repeat_kv(key, attention.n_rep)
    value = repeat_kv(value, attention.n_rep)

    print("\n 扩展 K/V Head 后：")
    print("key:", key.shape)
    print("value:", value.shape)

    # 4. [B, T, H, D] -> [B, H, T, D]
    query = query.transpose(1, 2)
    key = key.transpose(1, 2)
    value = value.transpose(1, 2)

    print("\n转置后：")
    print("query:", query.shape)
    print("key:", key.shape)
    print("value:", value.shape)

    # 5. 计算注意力分数
    scores = query @ key.transpose(-2, -1)

    print("\n原始注意力分数：", scores.shape)

    # 6.缩放
    scores = scores / math.sqrt(attention.head_dim)
    
    print("\n缩放后的注意力分数", scores.shape)

    # 7. 创建因果掩码
    
