from learn_vllm.engine import Engine
from learn_vllm.config import load_config
from learn_vllm.sampler import SamplingParams
import os
import torch

model_path = os.environ.get("QWEN3_MODEL_PATH", "/home/chen/projects/dzyy-vllm-local/models/Qwen3-0.6B")
config = load_config(model_path)

engine = Engine(model_path, config, device="cuda")

params = SamplingParams(temperature=0.7, top_k=40, top_p=0.9, max_tokens=30)
seed = 0
torch.manual_seed(seed)
prompts = [
    "教我python",
    "请介绍一下人工智能",
    "什么是GPU",
    "什么是操作系统",
    "解释一下CUDA",
    "什么是KV Cache",
    "什么是Continuous Batching"
]

requests = []
for prompt in prompts:
    requests.append(engine.add_request(prompt, params))
engine.run()

for req in requests:
    text = engine.tokenizer.decode(req.token_ids, skip_special_tokens=True)
    print(text)