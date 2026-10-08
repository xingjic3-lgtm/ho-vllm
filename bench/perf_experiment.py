"""Fixed engine workload; timing never includes profiler or output serialization."""
import argparse
import importlib
import json
import platform
import statistics
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))
pkg = ROOT.name
Engine = importlib.import_module(f"{pkg}.engine").Engine
load_config = importlib.import_module(f"{pkg}.config").load_config
SamplingParams = importlib.import_module(f"{pkg}.sampler").SamplingParams
workloads = importlib.import_module(f"{pkg}.bench.bench_serving")
MODEL = "/home/chen/projects/dzyy-vllm-local/models/Qwen3-0.6B"
CASES = {"prefill": (32, 1024, 4), "mixed": (32, 512, 32), "decode": (32, 128, 128), "profile": (4, 128, 8)}


def run(engine, lengths, output_len, label, profile=False):
    params = SamplingParams(temperature=0.0, top_k=50, top_p=0.8, max_tokens=output_len)
    requests = [engine.add_request(workloads.make_prompt(engine.tokenizer, n), params) for n in lengths]
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    if profile:
        torch.cuda.cudart().cudaProfilerStart()
        torch.cuda.nvtx.range_push(label)
    start = time.perf_counter()
    engine.run()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    if profile:
        torch.cuda.nvtx.range_pop()
        torch.cuda.cudart().cudaProfilerStop()
    result = {"seconds": elapsed, "input_tokens": sum(r.num_prompt_tokens for r in requests),
              "output_tokens": sum(r.num_output_tokens for r in requests),
              "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
              "peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
              "outputs": [r.output_token_ids for r in requests]}
    result["output_tokens_per_second"] = result["output_tokens"] / elapsed
    result["total_tokens_per_second"] = (result["input_tokens"] + result["output_tokens"]) / elapsed
    assert not engine.scheduler.running and not engine.scheduler.waiting
    assert engine.kv_cache_manager.get_num_free_blocks() == 256
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--reference")
    parser.add_argument("--cases", nargs="+", default=["prefill", "mixed", "decode"])
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()
    torch.manual_seed(0)
    engine = Engine(MODEL, load_config(MODEL))
    result = {"environment": {"torch": torch.__version__, "gpu": torch.cuda.get_device_name(),
                              "python": platform.python_version(), "model": MODEL},
              "profiled": args.profile, "warmup": args.warmup, "cases": {}}
    reference = json.loads(Path(args.reference).read_text()) if args.reference else None
    for case in args.cases:
        count, length, output_len = CASES[case]
        lengths = workloads.make_lengths(count, length, 0.5, 0)
        for _ in range(args.warmup):
            run(engine, lengths, output_len, "warmup")
        rows = []
        for i in range(args.repeat):
            row = run(engine, lengths, output_len, case, args.profile)
            if reference:
                assert row["outputs"] == reference["cases"][case]["runs"][0]["outputs"], f"token mismatch: {case}"
            if rows:
                assert row["outputs"] == rows[0]["outputs"], f"nondeterministic: {case}"
            rows.append(row)
            print(case, i, {k: v for k, v in row.items() if k != "outputs"}, flush=True)
        result["cases"][case] = {"lengths": lengths, "runs": rows, "median_seconds": statistics.median(r["seconds"] for r in rows)}
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
