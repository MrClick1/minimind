import torch


def main() -> None:
    torch.manual_seed(42)

    # B=2，T=3，C=4
    x = torch.randn(2, 3, 4)

    print("x:")
    print(x)
    print("x.shape:", x.shape)
    print("x.dtype:", x.dtype)

    eps = 1e-5

    # 1. 计算每个 token 隐藏向量的平方
    squared = x.pow(2)

    # 2. 对 hidden_size 维度求平均
    mean_square = squared.mean(
        dim=-1,
        keepdim=True,
    )

    # 3. 计算 RMS 的倒数
    reciprocal_rms = mean_square.add(eps).rsqrt()

    # 4. 完成归一化
    normalized = x * reciprocal_rms

    print("\nmean_square.shape:", mean_square.shape)
    print("reciprocal_rms.shape:", reciprocal_rms.shape)
    print("normalized.shape:", normalized.shape)


if __name__ == "__main__":
    main()