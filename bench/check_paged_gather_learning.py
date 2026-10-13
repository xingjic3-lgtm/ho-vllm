"""Compare the learning kernel and integrated engine with eager gather/repeat."""
import importlib
import sys
import tempfile
from pathlib import Path

import torch

from perf_experiment import pkg
import check_execution

attention = importlib.import_module(f"{pkg}.model.attention")
fused = attention.paged_gather_gqa


def eager(cache, table, length, heads, kv_heads, dim, block_size):
    positions = torch.arange(length, device=cache.device)
    physical = table[positions // block_size].long()
    offsets = positions % block_size
    return tuple(cache[i, physical, offsets].repeat_interleave(heads // kv_heads, dim=1)
                 for i in range(2))


def main():
    torch.manual_seed(7)
    cases = 0
    for dtype in (torch.float32, torch.float16, torch.bfloat16):
        # Same non-contiguous K/V plane layout as cache[:, layer_id].
        storage = torch.randn(2, 3, 12, 16, 8, 128, device="cuda", dtype=dtype)
        cache = storage[:, 1]
        tables = torch.tensor([[7, 1, 9, 4], [2, 11, 0, 6]], device="cuda", dtype=torch.int32)
        try:
            fused(cache, tables, 17, 16, 8, 128, 16)
        except ValueError:
            pass
        else:
            raise AssertionError("A whole batch table must be rejected")
        for row in tables:
            for length in (1, 15, 16, 17, 49):
                actual = fused(cache, row, length, 16, 8, 128, 16)
                expected = eager(cache, row, length, 16, 8, 128, 16)
                assert all(torch.equal(a, b) for a, b in zip(actual, expected))
                cases += 1
    print(f"PASS: {cases} gather cases, including distinct request pages and layer strides", flush=True)

    old_argv = sys.argv
    try:
        with tempfile.TemporaryDirectory(prefix="gather-reference-") as directory:
            reference = str(Path(directory) / "eager.pt")
            attention.paged_gather_gqa = eager
            sys.argv = ["check_execution.py", "--save", reference]
            check_execution.main()
            torch.cuda.empty_cache()
            attention.paged_gather_gqa = fused
            sys.argv = ["check_execution.py", "--reference", reference]
            check_execution.main()
    finally:
        attention.paged_gather_gqa = fused
        sys.argv = old_argv


if __name__ == "__main__":
    main()
