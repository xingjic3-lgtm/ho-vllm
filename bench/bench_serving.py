import time
import random
import statistics
import torch

import sys
from pathlib import Path

if __package__ in (None, ""):
    package_root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(package_root.parent))
    __package__ = f"{package_root.name}.bench"

from ..config import load_config
from ..engine import Engine
from ..sampler import SamplingParams

MODEL_PATH = "/home/chen/projects/dzyy-vllm-local/models/Qwen3-0.6B"

WARMUP = 1
REPEAT = 5


def make_prompt(tokenizer, target_len: int) -> str:
    text = (
        "请详细介绍大语言模型推理优化，包括注意力机制、"
        "KV Cache、Continuous Batching、GPU计算和显存管理。"
    )

    text = text * (target_len // 10 + 10)

    token_ids = tokenizer.encode(
        text,
        add_special_tokens=False,
    )

    token_ids = token_ids[:target_len]

    return tokenizer.decode(
        token_ids,
        skip_special_tokens=True,
    )


def make_lengths(
    num_requests: int,
    mean_len: int,
    range_ratio: float,
    seed: int,
):
    rng = random.Random(seed)

    low = int(mean_len * (1 - range_ratio))
    high = int(mean_len * (1 + range_ratio))

    return [
        rng.randint(low, high)
        for _ in range(num_requests)
    ]


def run_once(
    engine,
    input_lengths,
    output_len,
):
    requests = []

    params = SamplingParams(
        temperature=0.0,
        top_k=50,
        top_p=0.8,
        max_tokens=output_len,
    )

    for input_len in input_lengths:
        prompt = make_prompt(
            engine.tokenizer,
            input_len,
        )

        request = engine.add_request(
            prompt,
            params,
        )

        requests.append(request)

    total_input_tokens = sum(
        request.num_prompt_tokens
        for request in requests
    )

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    start = time.perf_counter()

    engine.run()

    torch.cuda.synchronize()

    elapsed = time.perf_counter() - start

    total_output_tokens = sum(
        request.num_output_tokens
        for request in requests
    )

    peak_memory = (
        torch.cuda.max_memory_allocated()
        / 1024**2
    )

    return {
        "time": elapsed,
        "input_tokens": total_input_tokens,
        "output_tokens": total_output_tokens,
        "total_tokens": total_input_tokens + total_output_tokens,
        "requests": len(requests),
        "peak_memory": peak_memory,
    }


def benchmark(
    engine,
    name,
    num_requests,
    input_len,
    output_len,
    range_ratio=0.0,
):
    print()
    print("=" * 70)
    print(name)
    print(
        f"requests={num_requests}, "
        f"input≈{input_len}, "
        f"output={output_len}, "
        f"range_ratio={range_ratio}"
    )
    print("=" * 70)

    input_lengths = make_lengths(
        num_requests=num_requests,
        mean_len=input_len,
        range_ratio=range_ratio,
        seed=0,
    )

    times = []

    for i in range(REPEAT):
        result = run_once(
            engine,
            input_lengths,
            output_len,
        )

        times.append(result["time"])

        request_throughput = (
            result["requests"]
            / result["time"]
        )

        input_throughput = (
            result["input_tokens"]
            / result["time"]
        )

        output_throughput = (
            result["output_tokens"]
            / result["time"]
        )

        total_throughput = (
            result["total_tokens"]
            / result["time"]
        )

        print(
            f"[{i + 1}] "
            f"time={result['time']:.4f}s | "
            f"req/s={request_throughput:.2f} | "
            f"input tok/s={input_throughput:.2f} | "
            f"output tok/s={output_throughput:.2f} | "
            f"total tok/s={total_throughput:.2f} | "
            f"memory={result['peak_memory']:.2f} MB"
        )

    print()
    print(
        f"median = {statistics.median(times):.4f}s"
    )
    print(
        f"mean   = {statistics.mean(times):.4f}s"
    )
    print(
        f"std    = {statistics.stdev(times):.4f}s"
        if len(times) > 1
        else "std    = 0"
    )


def main():
    config = load_config(MODEL_PATH)

    engine = Engine(
        MODEL_PATH,
        config,
        device="cuda",
    )

    # ---------- warmup ----------
    print("warmup...")

    for _ in range(WARMUP):
        run_once(
            engine,
            input_lengths=[128, 128],
            output_len=4,
        )

    # 1. Prefill-heavy
    # 最适合观察 selective lm_head
    benchmark(
        engine,
        name="Prefill-heavy",
        num_requests=32,
        input_len=1024,
        output_len=4,
        range_ratio=0.5,
    )

    # 2. Mixed workload
    benchmark(
        engine,
        name="Mixed",
        num_requests=32,
        input_len=512,
        output_len=32,
        range_ratio=0.5,
    )

    # 3. Decode-heavy
    # 这里 selective lm_head 理论上收益很小
    benchmark(
        engine,
        name="Decode-heavy",
        num_requests=32,
        input_len=128,
        output_len=128,
        range_ratio=0.5,
    )


if __name__ == "__main__":
    main()


# ======================================================================
# Prefill-heavy
# requests=32, input≈1024, output=4, range_ratio=0.5
# ======================================================================
# [1] time=9.5052s | req/s=3.37 | input tok/s=3542.90 | output tok/s=13.47 | total tok/s=3556.37 | memory=1667.71 MB
# [2] time=10.6422s | req/s=3.01 | input tok/s=3164.37 | output tok/s=12.03 | total tok/s=3176.40 | memory=1667.71 MB
# [3] time=10.8976s | req/s=2.94 | input tok/s=3090.22 | output tok/s=11.75 | total tok/s=3101.97 | memory=1667.71 MB
# [4] time=11.2056s | req/s=2.86 | input tok/s=3005.29 | output tok/s=11.42 | total tok/s=3016.71 | memory=1667.71 MB
# [5] time=9.7328s | req/s=3.29 | input tok/s=3460.04 | output tok/s=13.15 | total tok/s=3473.20 | memory=1667.71 MB

# median = 10.6422s
# mean   = 10.3967s
# std    = 0.7418s

# ======================================================================
# Mixed
# requests=32, input≈512, output=32, range_ratio=0.5
# ======================================================================
# [1] time=29.2577s | req/s=1.09 | input tok/s=575.23 | output tok/s=35.00 | total tok/s=610.23 | memory=1667.71 MB
# [2] time=29.8984s | req/s=1.07 | input tok/s=562.91 | output tok/s=34.25 | total tok/s=597.16 | memory=1667.71 MB
# [3] time=30.6255s | req/s=1.04 | input tok/s=549.54 | output tok/s=33.44 | total tok/s=582.98 | memory=1667.71 MB
# [4] time=29.5173s | req/s=1.08 | input tok/s=570.17 | output tok/s=34.69 | total tok/s=604.86 | memory=1667.71 MB
# [5] time=30.9873s | req/s=1.03 | input tok/s=543.13 | output tok/s=33.05 | total tok/s=576.17 | memory=1667.71 MB

# median = 29.8984s
# mean   = 30.0572s
# std    = 0.7321s






# Selective Logits Computation


# ======================================================================
# Prefill-heavy
# requests=32, input≈1024, output=4, range_ratio=0.5
# ======================================================================
# [1] time=10.7223s | req/s=2.98 | input tok/s=3140.74 | output tok/s=11.94 | total tok/s=3152.67 | memory=1667.19 MB
# [2] time=10.7789s | req/s=2.97 | input tok/s=3124.25 | output tok/s=11.88 | total tok/s=3136.13 | memory=1667.19 MB
# [3] time=10.2744s | req/s=3.11 | input tok/s=3277.67 | output tok/s=12.46 | total tok/s=3290.13 | memory=1667.19 MB
# [4] time=9.8911s | req/s=3.24 | input tok/s=3404.68 | output tok/s=12.94 | total tok/s=3417.62 | memory=1667.19 MB
# [5] time=10.5877s | req/s=3.02 | input tok/s=3180.66 | output tok/s=12.09 | total tok/s=3192.75 | memory=1667.19 MB

# median = 10.5877s
# mean   = 10.4509s
# std    = 0.3690s

# ======================================================================
# Mixed
# requests=32, input≈512, output=32, range_ratio=0.5
# ======================================================================
# [1] time=29.3107s | req/s=1.09 | input tok/s=574.19 | output tok/s=34.94 | total tok/s=609.13 | memory=1634.24 MB
# [2] time=28.1478s | req/s=1.14 | input tok/s=597.92 | output tok/s=36.38 | total tok/s=634.29 | memory=1634.24 MB
# [3] time=27.5619s | req/s=1.16 | input tok/s=610.62 | output tok/s=37.15 | total tok/s=647.78 | memory=1634.24 MB
# [4] time=29.9599s | req/s=1.07 | input tok/s=561.75 | output tok/s=34.18 | total tok/s=595.93 | memory=1634.24 MB
# [5] time=27.8759s | req/s=1.15 | input tok/s=603.75 | output tok/s=36.73 | total tok/s=640.48 | memory=1634.24 MB

# median = 28.1478s
# mean   = 28.5712s
# std    = 1.0194s
