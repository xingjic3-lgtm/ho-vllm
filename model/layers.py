"""任务 02：模型基础层。"""

import torch
from torch import nn
from torch.nn import functional as F


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x32 = x.float()
        mean_square = x32.square().mean(dim=-1, keepdim=True)
        normalized = x32 * torch.rsqrt(mean_square + self.eps)
        return normalized.to(x.dtype) * self.weight


class MLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        hidden = F.silu(gate) * up
        return self.down_proj(hidden)


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(1, 3, 4)
    norm = RMSNorm(hidden_size=4)
    mlp = MLP(hidden_size=4, intermediate_size=8)

    with torch.inference_mode():
        normalized = norm(x)  # 填空 2：调用 norm 处理 x。
        output = x + mlp(normalized)  # 填空 3：normalized 经过 mlp，结果与原始 x 相加。

    print("输入形状：", x.shape)
    print("输出形状：", output.shape)
    print("输出：", output)
