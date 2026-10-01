import torch
from .layers import RMSNorm, MLP
from .attention import Attention
from torch import nn


class DecoderLayer(nn.Module):
    def __init__(self,hidden_size: int, num_heads: int, num_kv_heads: int, head_dim: int, intermediate_size: int):
        super().__init__()
        self.input_layernorm = RMSNorm(hidden_size)
        self.self_attn = Attention(hidden_size, num_heads, num_kv_heads, head_dim)
        self.post_attention_layernorm = RMSNorm(hidden_size)
        self.mlp = MLP( hidden_size, intermediate_size )

    def forward(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        residual = x
        normalized = self.input_layernorm(x)    
        attn_output = self.self_attn(normalized, positions)
        x = attn_output + residual

        residual = x
        normalized = self.post_attention_layernorm(x)
        mlp_output = self.mlp(normalized)
        x = residual + mlp_output
        return x

class Qwen3Model(nn.Module):
    """模型主体：token IDs → embedding → 解码器层 → norm。"""
    def __init__(self,hidden_size:int, vocab_size:int, layers:int,num_heads: int, num_kv_heads: int, head_dim: int, intermediate_size: int):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        self.layers = nn.ModuleList([
                    DecoderLayer(
                        hidden_size=hidden_size,
                        num_heads=num_heads,
                        num_kv_heads=num_kv_heads,
                        head_dim=head_dim,
                        intermediate_size=intermediate_size,
                    )
                    for _ in range(layers)
                ])
        self.norm = RMSNorm(hidden_size)

    def forward(self, token_ids: torch.Tensor, positions:torch.Tensor):  
        x = self.embed_tokens(token_ids)
        for layer in self.layers:
            x = layer(x, positions)
        return self.norm(x)


class Qwen3(nn.Module):
    """模型主体加词表投影；沿用当前练习直接返回 logits 的接口。"""
    def __init__(self,hidden_size:int, vocab_size:int, layers:int,num_heads: int, num_kv_heads: int, head_dim: int, intermediate_size: int):
        super().__init__()
        self.model = Qwen3Model(
            hidden_size=hidden_size,
            vocab_size=vocab_size,
            layers=layers,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            intermediate_size=intermediate_size,
        )
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        self.lm_head.weight = self.model.embed_tokens.weight

    def forward(self, token_ids: torch.Tensor, positions: torch.Tensor):
        return self.lm_head(self.model(token_ids, positions))


if __name__ == "__main__":
    torch.manual_seed(0)
    model = Qwen3(
        hidden_size=12,
        vocab_size=32,
        layers=2,
        num_heads=4,
        num_kv_heads=2,
        head_dim=4,
        intermediate_size=24,
    ).eval()
    token_ids = torch.tensor([[5, 8, 10], [2, 17, 6]])
    positions = torch.arange(token_ids.shape[1])

    with torch.inference_mode():
        logits = model(token_ids, positions)

        # 只替换最后一个 token，检查完整模型是否仍遵守因果限制。
        changed_ids = token_ids.clone()
        changed_ids[:, -1] = (changed_ids[:, -1] + 1) % 32
        changed_logits = model(changed_ids, positions)

    print("logits 形状（预期 [2,3,32]）：", logits.shape)
    print("所有分数均为有限值（预期 True）：", torch.isfinite(logits).all().item())
    print(
        "前两个位置最大差异（预期 0）：",
        (logits[:, :-1, :] - changed_logits[:, :-1, :]).abs().max().item(),
    )
    print(
        "最后一个位置最大差异（允许变化）：",
        (logits[:, -1, :] - changed_logits[:, -1, :]).abs().max().item(),
    )
    print(
        "embedding 与词表投影共享参数（预期 True）：",
        model.model.embed_tokens.weight is model.lm_head.weight,
    )
    print(
        "两层注意力的 Q 投影参数独立（预期 True）：",
        model.model.layers[0].self_attn.q_proj.weight is not model.model.layers[1].self_attn.q_proj.weight,
    )
