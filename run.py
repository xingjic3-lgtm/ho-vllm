"""入口只负责设置参数、创建引擎和显示输出。

从仓库根目录运行：python -m learn_vllm.run
"""
import torch
import os
import sys
from pathlib import Path

# Allow both ``python -m learn-vllm.run`` and ``python /path/to/run.py``.
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
    text = "你好，请介绍一下自己。"

    engine = Engine(model_path, config, device="cuda")
    for result in engine.generate(text, params):
        print(result)


if __name__ == "__main__":
    main()
