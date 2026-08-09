import torch


class MyRMSNorm(torch.nn.Module):
    def __init__(
        self,
        dim: int,
        eps: float = 1e-5,
    ) -> None:
        super().__init__()

        self.eps = eps
        self.weight = torch.nn.Parameter(
            torch.ones(dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. 临时转换为 float32
        x_float = x.float()

        # 2. 计算平方的平均值
        mean_square = x_float.square().mean(
            dim=-1,
            keepdim=True,
        )

        # 3. 计算 RMS 的倒数
        reciprocal_rms = mean_square.add(self.eps).rsqrt()

        # 4. 完成归一化
        normalized = x_float * reciprocal_rms

        # 5. 乘以可学习 weight
        output = normalized * self.weight

        # 6. 转回原始 x 的类型
        return output.to(x.dtype)


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

    # 5. 重新计算归一化结果的 RMS
    normalized_rms = (
        normalized
        .square()
        .mean(dim=-1, keepdim=True)
        .sqrt()
    )

    print("\n归一化后的 RMS：")
    print(normalized_rms)
    print("normalized_rms.shape:", normalized_rms.shape)

    # 6. 创建每个隐藏维度的可学习缩放参数
    weight = torch.nn.Parameter(
        torch.ones(x.shape[-1])
    )

    # 7. 应用 weight
    output = normalized * weight

    print("\nweight:")
    print(weight)
    print("weight.shape:", weight.shape)
    print("weight.requires_grad:", weight.requires_grad)

    print("\n最终输出：")
    print(output)
    print("output.shape:", output.shape)

    print(
        "初始输出是否等于 normalized:",
        torch.allclose(output, normalized)
    )


    my_rms_norm = MyRMSNorm(
        dim=x.shape[-1],
        eps=eps,
    )

    my_output = my_rms_norm(x)

    print("\nMyRMSNorm 输出：")
    print(my_output)
    print("my_output.shape:", my_output.shape)
    print("my_output.dtype:", my_output.dtype)


if __name__ == "__main__":
    main()