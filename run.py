import torch
import os
import sys
from pathlib import Path
if __package__ in (None, ""):
    package_dir = Path(__file__).resolve().parent
    sys.path.insert(0, str(package_dir.parent))
    __package__ = package_dir.name
from .config import load_config
from .engine import Engine
from .sampler import SamplingParams


def main():
    torch.manual_seed(0)

    model_path = os.environ.get("QWEN3_MODEL_PATH", "/home/chen/projects/dzyy-vllm-local/models/Qwen3-0.6B")
    config = load_config(model_path)

    params = SamplingParams(
        temperature=0.0,
        top_k=50,
        top_p=0.8,
        max_tokens=30,
    )

    engine = Engine(model_path, config, device="cuda")

    request1 = engine.add_request("你好，请介绍一下自己。", params)
    request2 = engine.add_request("2026年10月新闻", params)

    engine.run()

    print(engine.tokenizer.decode(request1.all_token_ids, skip_special_tokens=True))
    print(engine.tokenizer.decode(request2.all_token_ids, skip_special_tokens=True))


if __name__ == "__main__":
    main()