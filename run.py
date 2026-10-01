"""入口只负责设置参数、创建引擎和显示输出。

从仓库根目录运行：python -m learn_vllm.run
"""
import torch

from .config import load_config
from .engine import Engine
from .sampler import SamplingParams


def main():
    torch.manual_seed(0)
    model_path = "/root/huggingface/Qwen3-0.6B"
    config = load_config(model_path)
    params = SamplingParams(
        temperature=0.0,
        top_k=50,
        top_p=0.8,
        max_tokens=10,
    )
    text = "你好，请介绍一下自己。"

    engine = Engine(model_path, config, device="cuda")
    for result in engine.generate(text, params):
        print(result)


if __name__ == "__main__":
    main()
