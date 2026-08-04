import torch

from model.model_minimind import MiniMindConfig, MiniMindForCausalLM


def main() -> None:
    config = MiniMindConfig(
        hidden_size=128,
        num_hidden_layers=2,
        vocab_size=6400,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=256,
        use_moe=False,
    )

    model = MiniMindForCausalLM(config)
    model.eval()

    batch_size = 2
    seq_len = 16

    input_ids = torch.randint(
        low=0,
        high=config.vocab_size,
        size=(batch_size, seq_len),
        dtype=torch.long,
    )

    labels = input_ids.clone()

    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            labels=labels,
        )

    parameter_count = sum(
        parameter.numel() for parameter in model.parameters()
    )

    print("input_ids shape:", input_ids.shape)
    print("logits shape:", outputs.logits.shape)
    print("loss:", outputs.loss.item())
    print("parameter count:", parameter_count)
    print("parameter count (M):", parameter_count / 1_000_000)


if __name__ == "__main__":
    main()