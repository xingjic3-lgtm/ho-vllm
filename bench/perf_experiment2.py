import torch
import os
import sys
import time
import statistics
from pathlib import Path
if __package__ in (None, ""):
    project_dir = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project_dir.parent))
    __package__ = f"{project_dir.name}.bench"
from ..config import load_config
from ..engine import Engine
from ..sampler import SamplingParams

WARMUP_RUNS = 1
MEASURE_RUNS = 5

WORKLOADS = {
    "prefill": {
        "num_requests": 8,
        "input_len": 1024,
        "output_len": 16,
    },
    "mixed": {
        "num_requests": 8,
        "input_len": 512,
        "output_len": 64,
    },
    "decode": {
        "num_requests": 8,
        "input_len": 128,
        "output_len": 256,
    },
}
def main():
    torch.manual_seed(0)

    model_path = os.environ.get("QWEN3_MODEL_PATH", "/home/chen/projects/dzyy-vllm-local/models/Qwen3-0.6B")
    config = load_config(model_path)

    engine = Engine(model_path, config, device="cuda")

    for workload_name, workload in WORKLOADS.items():
        # 固定这组输入，预热和正式测量都使用同一批 token。
        inputs = [
            create_synthetic_request(
                config["vocab_size"],
                workload["input_len"],
                workload["output_len"],
            )
            for _ in range(workload["num_requests"])
        ]

        for _ in range(WARMUP_RUNS):
            for token_ids, params in inputs:
                engine.add_request_tokens(token_ids, params)
            engine.run()

        times = []
        for repeat in range(MEASURE_RUNS):
            # run() 会处理完队列，每轮必须重新创建请求。
            for token_ids, params in inputs:
                engine.add_request_tokens(token_ids, params)

            torch.cuda.synchronize()
            start = time.perf_counter()

            engine.run()

            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            times.append(elapsed)
            print(f"{workload_name} [{repeat + 1}/{MEASURE_RUNS}]: {elapsed:.3f} s", flush=True)

        print(
            f"{workload_name} median: {statistics.median(times):.3f} s "
            f"(min={min(times):.3f}, max={max(times):.3f})",
            flush=True,
        )


    # torch.cuda.synchronize() 
    # torch.cuda.cudart().cudaProfilerStart()
    # torch.cuda.nvtx.range_push("profile2")
    # start = time.perf_counter()
    # engine.run()
    # torch.cuda.synchronize()
    # end = time.perf_counter()
    # torch.cuda.nvtx.range_pop()
    # torch.cuda.cudart().cudaProfilerStop()
    # elapsed = end-start

    # print(f"耗时：{elapsed:.3f} s")

def create_synthetic_request(
    vocab_size,
    input_len,
    output_len,
):
    token_ids = torch.randint(
        100,
        vocab_size,
        (input_len,),
        dtype=torch.long,
    )

    params = SamplingParams(
        temperature=0.0,
        top_k=50,
        top_p=0.8,
        max_tokens=output_len,
    )

    return token_ids, params

if __name__ == "__main__":
    main()
