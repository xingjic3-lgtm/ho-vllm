import json
from pathlib import Path

from safetensors.torch import load_file
from .model import Qwen3


def load_weight():
    model_path = Path("/root/huggingface/Qwen3-0.6B")
    with (model_path / "config.json").open() as f:
        config = json.load(f)

    model = Qwen3(
        hidden_size=config["hidden_size"],
        vocab_size=config["vocab_size"],
        layers=config["num_hidden_layers"],
        num_heads=config["num_attention_heads"],
        num_kv_heads=config["num_key_value_heads"],
        head_dim=config["head_dim"],
        intermediate_size=config["intermediate_size"],
    )

    state_dict = load_file(str(model_path / "model.safetensors"), device="cpu")

    model.load_state_dict(state_dict)

    model.to("cuda")

    model.eval()

    return model
