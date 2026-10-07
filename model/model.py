import torch
from torch import nn

from .layers import RMSNorm, MLP
from .attention import Attention


class DecoderLayer(nn.Module):
    def __init__(self, config: dict, layer_id: int):
        super().__init__()
        self.input_layernorm = RMSNorm(config["hidden_size"], eps=config["rms_norm_eps"])
        self.self_attn = Attention(config, layer_id)
        self.post_attention_layernorm = RMSNorm(config["hidden_size"], eps=config["rms_norm_eps"])
        self.mlp = MLP(config["hidden_size"], config["intermediate_size"])

    def forward(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.input_layernorm(x)
        x = self.self_attn(x, positions)
        x = x + residual

        residual = x
        x = self.post_attention_layernorm(x)
        x = self.mlp(x)
        x = x + residual

        return x


class Qwen3Model(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        self.embed_tokens = nn.Embedding(config["vocab_size"], config["hidden_size"])
        self.layers = nn.ModuleList([DecoderLayer(config, layer_id) for layer_id in range(config["num_hidden_layers"])])
        self.norm = RMSNorm(config["hidden_size"], eps=config["rms_norm_eps"])

    def forward(self, token_ids: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        x = self.embed_tokens(token_ids)

        for layer in self.layers:
            x = layer(x, positions)

        return self.norm(x)


class Qwen3(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        self.model = Qwen3Model(config)
        self.lm_head = nn.Linear(config["hidden_size"], config["vocab_size"], bias=False)

        if config["tie_word_embeddings"]:
            self.lm_head.weight = self.model.embed_tokens.weight

    def forward(self, token_ids: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        return self.model(token_ids, positions)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.lm_head(hidden_states)