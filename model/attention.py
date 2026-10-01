"""任务 02：逐步实现注意力，当前先返回分头后的 Q、K、V。"""

import torch
from torch import nn
import math
from torch.nn import functional as F

from .layers import RMSNorm


class RotaryEmbedding(nn.Module):
    """前后半段配对的 RoPE：输入和输出均为 [B, heads, T, D]。"""

    def __init__(self, head_dim: int, rope_theta: float = 1_000_000.0):
        super().__init__()
        self.head_dim = head_dim
        self.rope_theta = rope_theta

    def forward(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        # positions：[T]，允许从非零位置开始。
        positions = positions.to(device=x.device, dtype=torch.float32)
        pair_indices = torch.arange(
            self.head_dim // 2, device=x.device, dtype=torch.float32
        )
        inv_freq = self.rope_theta ** (-2 * pair_indices / self.head_dim)

        # 空 1：每个位置乘上每个频率，得到 [T, D/2] 的角度表。
        angles = positions[:,None]*inv_freq[None,:]   # [positions, pairindex]

        # 前两维长度为 1，让所有请求、所有头共享同一张角度表。
        cos = angles.cos()[None, None, :, :]    # 每个数cos
        sin = angles.sin()[None, None, :, :]    # 每个数sin

        # 前后半段对应元素成对旋转，计算暂用 float32。
        half = self.head_dim // 2
        a = x[..., :half].float()
        b = x[..., half:].float()

        # 空 2、3：根据二维旋转公式，求新的前半段和后半段。
        rotated_a = a * cos - b * sin
        rotated_b = a * sin + b * cos

        # 空 4：按前半段、后半段的顺序拼回完整的最后一维。
        rotated = torch.cat([rotated_a, rotated_b], dim=-1)
        return rotated.to(x.dtype)


class Attention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.q_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads*head_dim, hidden_size, bias=False)
        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)
        self.rope = RotaryEmbedding(self.head_dim)

    def forward(self, x: torch.Tensor, positions: torch.Tensor):
        """输入 x [B,T,H] 和 positions [T]，返回 [B,T,H]。"""
        q = self.q_proj(x)   # [B,T,head_dim*num_heads]
        k = self.k_proj(x)   # [B,T,head_dim*num_kv_heads]
        v = self.v_proj(x)   # [B,T,head_dim*num_kv_heads]

        B,T,_ = x.shape

        q_view = q.view(B,T,self.num_heads,self.head_dim)
        k_view = k.view(B,T,self.num_kv_heads, self.head_dim)
        v_view = v.view(B,T,self.num_kv_heads, self.head_dim)

        Q = q_view.transpose(1,2)  # Q [B,Hq,T,D]
        K = k_view.transpose(1,2)  # K/V [B,Hkv,T,D]
        V = v_view.transpose(1,2)
        Q = self.q_norm(Q)
        K = self.k_norm(K)
        Q = self.rope(Q, positions)
        K = self.rope(K, positions)


        repeat = self.num_heads // self.num_kv_heads
        K = K.repeat_interleave(repeat,dim=1)
        V = V.repeat_interleave(repeat,dim=1)
        # softmax(（Q @ Kt）/ 厂d + M) V   这里M掩码对于需要掩住的地方是-∞不掩住的是0   why：-∞+qk结果=-∞   e的-∞ == 0
        M = torch.triu(
            torch.full((T, T), float("-inf"), device=Q.device, dtype=Q.dtype),
            diagonal=1,
        )
        d_sqrt = math.sqrt(self.head_dim)     
        res = F.softmax(
            (Q @ K.transpose(2,3))/d_sqrt + M,  
            dim=-1
            )@V

        res = res.transpose(1,2).reshape(B,T,self.num_heads * self.head_dim)
        return self.o_proj(res)


if __name__ == "__main__":
    torch.manual_seed(0)
    attention = Attention(hidden_size=12, num_heads=4, num_kv_heads=2, head_dim=4)
    x = torch.randn(2, 3, 12)
    positions = torch.arange(x.shape[1])

    with torch.inference_mode():
        original_output = attention(x, positions)

        # 只改变最后一个 token，保留原输入和同一份模型权重。
        changed_x = x.clone()
        changed_x[:, -1, :] += 10.0
        changed_output = attention(changed_x, positions)

    # 前两个位置不能看到最后一个 token，因此输出应保持不变。
    earlier_difference = (
        original_output[:, :-1, :] - changed_output[:, :-1, :]
    ).abs().max().item()
    last_difference = (
        original_output[:, -1, :] - changed_output[:, -1, :]
    ).abs().max().item()

    print("输出形状：", original_output.shape)
    print("前两个位置的最大差异（预期为 0）：", earlier_difference)
    print("最后一个位置的最大差异（允许变化）：", last_difference)
