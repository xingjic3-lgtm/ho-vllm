"""任务 02：逐步实现注意力，当前先返回分头后的 Q、K、V。"""

import torch
from torch import nn
import math
from torch.nn import functional as F

from .layers import RMSNorm
from ..forward_context import get_forward_context
from .kernels.paged_gather_gqa import paged_gather_gqa

class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, rope_theta: float = 1_000_000.0):
        super().__init__()
        self.head_dim = head_dim
        self.rope_theta = rope_theta

    @torch.cuda.nvtx.range("build_cos_sin")
    def build_cos_sin(self, positions: torch.Tensor):
        positions = positions.to(device=positions.device, dtype=torch.float32)
        pair_indices = torch.arange(self.head_dim // 2, device=positions.device, dtype=torch.float32)
        inv_freq = self.rope_theta ** (-2 * pair_indices / self.head_dim)

        angles = positions[:, None] * inv_freq[None, :]
        cos = angles.cos()[:, None, :]
        sin = angles.sin()[:, None, :]
        return cos, sin

    @torch.cuda.nvtx.range("apply_RoPE")
    def apply_rotary(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
        
        half = self.head_dim // 2
        a = x[..., :half].float()
        b = x[..., half:].float()

        rotated_a = a * cos - b * sin
        rotated_b = a * sin + b * cos

        return torch.cat([rotated_a, rotated_b], dim=-1).to(x.dtype)

class Attention(nn.Module):
    def __init__(self, config: dict, layer_id: int):
        super().__init__()

        hidden_size = config["hidden_size"]

        self.layer_id = layer_id
        self.num_heads = config["num_attention_heads"]
        self.num_kv_heads = config["num_key_value_heads"]
        self.head_dim = config["head_dim"]

        attention_bias = config["attention_bias"]

        self.q_proj = nn.Linear(hidden_size, self.num_heads * self.head_dim, bias=attention_bias)
        self.k_proj = nn.Linear(hidden_size, self.num_kv_heads * self.head_dim, bias=attention_bias)
        self.v_proj = nn.Linear(hidden_size, self.num_kv_heads * self.head_dim, bias=attention_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, hidden_size, bias=attention_bias)

        self.q_norm = RMSNorm(self.head_dim, eps=config["rms_norm_eps"])
        self.k_norm = RMSNorm(self.head_dim, eps=config["rms_norm_eps"])
        self.rope = RotaryEmbedding(self.head_dim, rope_theta=config["rope_theta"])

        self.kv_cache = None

    def bind_kv_cache(self, kv_cache: torch.Tensor) -> None:
            self.kv_cache = kv_cache

    @torch.cuda.nvtx.range("Attention")
    def forward(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        assert self.kv_cache is not None
        attn_metadata = get_forward_context().attn_metadata
        query_start_loc = attn_metadata.query_start_loc
        seq_lens = attn_metadata.seq_lens
        block_table = attn_metadata.block_table
        slot_mapping = attn_metadata.slot_mapping

        N = x.shape[0]

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        Q = q.view(N, self.num_heads, self.head_dim)
        K = k.view(N, self.num_kv_heads, self.head_dim)
        V = v.view(N, self.num_kv_heads, self.head_dim)

        Q = self.q_norm(Q)
        K = self.k_norm(K)

        if attn_metadata.rope_cos is None:
            rope_cos, rope_sin = self.rope.build_cos_sin(positions)
            attn_metadata.rope_cos = rope_cos
            attn_metadata.rope_sin = rope_sin
        cos = attn_metadata.rope_cos
        sin = attn_metadata.rope_sin
        Q = self.rope.apply_rotary(Q, cos, sin)
        K = self.rope.apply_rotary(K, cos, sin)

        block_size = self.kv_cache.shape[2]

        physical_block_ids = slot_mapping // block_size
        block_offsets = slot_mapping % block_size

        self.kv_cache[0, physical_block_ids, block_offsets] = K
        self.kv_cache[1, physical_block_ids, block_offsets] = V

        output = torch.empty((N, self.num_heads * self.head_dim), dtype=Q.dtype, device=Q.device)

        num_requests = len(seq_lens)

        for i in range(num_requests):
            query_start = query_start_loc[i]
            query_end = query_start_loc[i + 1]
            seq_len = seq_lens[i]

            Q_i = Q[query_start:query_end]
            query_positions = positions[query_start:query_end]

            kv_positions = torch.arange(seq_len, device=Q.device)

            K_cache, V_cache = paged_gather_gqa(
                    self.kv_cache,
                    block_table[i],  # Only this request's logical-to-physical mapping.
                    seq_len,
                    self.num_heads,
                    self.num_kv_heads,
                    self.head_dim,
                    block_size,
                )

            Q_i = Q_i.transpose(0, 1)    # [head, query_len, head_dim]
            K_cache = K_cache.transpose(0, 1)   # [head, seq_len, head_dim]
            V_cache = V_cache.transpose(0, 1)

            scores = Q_i @ K_cache.transpose(1, 2)    # [head, head_dim, seq_len]
            scores = scores / math.sqrt(self.head_dim)     # [head, query_len, seq_len]

            causal_mask = kv_positions[None, :] <= query_positions[:, None]
            scores = scores.masked_fill(~causal_mask[None, :, :], float("-inf"))

            probs = F.softmax(scores.float(), dim=-1).to(Q_i.dtype)
            result = probs @ V_cache

            result = result.transpose(0, 1).reshape(query_end - query_start, self.num_heads * self.head_dim)
            output[query_start:query_end] = result
        return self.o_proj(output)
