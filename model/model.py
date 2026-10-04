import torch
from .layers import RMSNorm, MLP
from .attention import Attention
from torch import nn


class DecoderLayer(nn.Module):
    def __init__(self, config: dict, layer_id:int):
        super().__init__()
        self.input_layernorm = RMSNorm(config["hidden_size"], eps=config["rms_norm_eps"])
        self.self_attn = Attention(config, layer_id)
        self.post_attention_layernorm = RMSNorm(config["hidden_size"], eps=config["rms_norm_eps"])
        self.mlp = MLP(config["hidden_size"], config["intermediate_size"])

    def forward(self, x: torch.Tensor, positions: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        residual = x
        normalized = self.input_layernorm(x)    
        attn_output = self.self_attn(normalized, positions, mask)
        x = attn_output + residual

        residual = x
        normalized = self.post_attention_layernorm(x)
        mlp_output = self.mlp(normalized)
        x = residual + mlp_output
        return x

class Qwen3Model(nn.Module):
    """模型主体：token IDs → embedding → 解码器层 → norm。"""
    def __init__(self, config: dict):
        super().__init__()
        self.embed_tokens = nn.Embedding(config["vocab_size"], config["hidden_size"])
        self.layers = nn.ModuleList([
            DecoderLayer(config, layer_id) for layer_id in range(config["num_hidden_layers"])
        ])
        self.norm = RMSNorm(config["hidden_size"], eps=config["rms_norm_eps"])

    def forward(self, token_ids: torch.Tensor, positions:torch.Tensor, mask: torch.Tensor):  
        x = self.embed_tokens(token_ids)
        for layer in self.layers:
            x = layer(x, positions, mask)
        return self.norm(x)


class Qwen3(nn.Module):
    """模型主体加词表投影；沿用当前练习直接返回 logits 的接口。"""
    def __init__(self, config: dict):
        super().__init__()
        self.model = Qwen3Model(config)
        self.lm_head = nn.Linear(config["hidden_size"], config["vocab_size"], bias=False)
        if config["tie_word_embeddings"]:
            self.lm_head.weight = self.model.embed_tokens.weight

    def forward(self, token_ids: torch.Tensor, positions: torch.Tensor, mask: torch.Tensor):
        return self.lm_head(self.model(token_ids, positions, mask))