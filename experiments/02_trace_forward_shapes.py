import torch
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from model.model_minimind import (
    MiniMindConfig,
    MiniMindForCausalLM,
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

    input_ids = torch.randint(
        low=0,
        high=config.vocab_size,
        size=(2, 16),
        dtype=torch.long,
    )

    print("1. input_ids:", input_ids.shape)

    with torch.no_grad():
        embeddings = model.model.embed_tokens(input_ids)

    print("2. embeddings:", embeddings.shape)

    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            labels=input_ids,
        )

    print("3. hidden_states:", outputs.hidden_states.shape)
    print("4. logits:", outputs.logits.shape)
    print("5. loss:", outputs.loss.item())


if __name__ == "__main__":
    main()