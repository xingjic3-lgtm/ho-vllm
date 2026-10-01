from pathlib import Path

import torch
from safetensors.torch import load_file
from .model import Qwen3


def load_weight(
    model_path,
    config,
    device="cuda",
    dtype=None,
):
    model_path = Path(model_path).expanduser()
    if dtype is None:
        dtype = getattr(torch, config.get("dtype") or config["torch_dtype"])

    model = Qwen3(config)

    state_dict = load_file(str(model_path / "model.safetensors"), device="cpu")

    model.load_state_dict(state_dict)

    model.to(device=device, dtype=dtype)

    model.eval()

    return model
