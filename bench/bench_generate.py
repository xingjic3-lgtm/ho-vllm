
from learn_vllm.engine import Engine

from learn_vllm.config import load_config
from learn_vllm.engine import Engine
from learn_vllm.sampler import SamplingParams
import torch
import time
import os

model_path = os.environ.get("QWEN3_MODEL_PATH", "/home/chen/projects/dzyy-vllm-local/models/Qwen3-0.6B")
config = load_config(model_path)
params = SamplingParams(
    temperature=0.0,
    top_k=0,
    top_p=1.0,
    max_tokens=30,
)
text = "你好，请介绍一下自己。"

engine = Engine(model_path, config, device="cuda")

# 预热
for _ in range(3):
    generated_tokens = 0
    for _ in engine.generate(text, params):
        generated_tokens += 1
print("generated_tokens=",generated_tokens)

# total time
times = []
for _ in range(5):
    torch.cuda.synchronize()
    start = time.perf_counter()
    for result in engine.generate(text, params):
        pass
    torch.cuda.synchronize()
    end = time.perf_counter()
    times.append(end-start)
print("times=",times)

# time to first token
ttft_times = []
for _ in range(5):
    generator = engine.generate(text, params)
    torch.cuda.synchronize()
    start = time.perf_counter()
    next(generator)
    torch.cuda.synchronize()
    end = time.perf_counter()
    ttft_times.append(end-start)

print("ttft_times=", ttft_times)

# 测tpot   time per output token
tpot_times = []
for _ in range(5):
    generator = engine.generate(text, params)
    # 第一个 token 不计入 TPOT
    next(generator)
    # 从第一个 token 已经返回之后开始计时
    torch.cuda.synchronize()
    start = time.perf_counter()
    # 连续生成剩余 token，中间不插同步
    for _ in generator:
        pass
    torch.cuda.synchronize()
    end = time.perf_counter()
    tpot = (end - start) / (generated_tokens - 1)
    tpot_times.append(tpot)
print("tpot_times =", tpot_times)


# 测peak显存    pytorch分配给tensor的显存peak
torch.cuda.synchronize()
torch.cuda.reset_peak_memory_stats()
for _ in engine.generate(text, params):
    pass
peak_memory = torch.cuda.max_memory_allocated()
print("peak_memory=", peak_memory/1024**2, "MB")
print("peak reserved =", torch.cuda.max_memory_reserved() / 1024**2, "MB")

# 统计    中位数不易被异常值带偏
import statistics
total_median = statistics.median(times)
ttft_median = statistics.median(ttft_times)
tpot_median = statistics.median(tpot_times)
print(f"Total: {total_median:.4f} s")
print(f"TTFT:  {ttft_median * 1000:.2f} ms")
print(f"TPOT:  {tpot_median * 1000:.2f} ms/token")

# times= [2.4924637519943644, 2.0398501220042817, 2.929943937000644, 2.181643569994776, 1.8748837639941485]
# ttft_times= [0.04430561399931321, 0.05250034900382161, 0.07841149299929384, 0.02612183499877574, 0.10670454999490175]
# tpot_times = [0.07621730658615104, 0.08614336572419892, 0.07176092575863205, 0.061135843068768736, 0.055193076758486906]
# peak_memory= 1165.08203125 MB
# peak reserved = 1200.0 MB
# Total: 2.1816 s
# TTFT:  52.50 ms
# TPOT:  71.76 ms/token
