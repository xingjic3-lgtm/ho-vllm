"""Capture every scheduled step's logits; compare unchanged execution bitwise."""
import argparse
import json
from pathlib import Path

import torch
from perf_experiment import Engine, MODEL, SamplingParams, load_config, workloads


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--save")
    p.add_argument("--reference")
    p.add_argument("--hf", action="store_true")
    args = p.parse_args()
    torch.manual_seed(42)
    engine = Engine(MODEL, load_config(MODEL))
    captured = []
    original = engine.model_runner.execute_model

    def execute(inputs):
        logits = original(inputs)
        captured.append(logits.cpu().clone())
        return logits

    engine.model_runner.execute_model = execute
    outputs = []
    for temperature in [0.0, 0.7]:
        requests = [engine.add_request(workloads.make_prompt(engine.tokenizer, n),
                    SamplingParams(temperature=temperature, top_k=20, top_p=0.9, max_tokens=8))
                    for n in [17, 257, 33, 65, 513]]
        engine.run()
        outputs.extend(r.output_token_ids for r in requests)
        assert engine.kv_cache_manager.get_num_free_blocks() == 256
    result = {"logits": captured, "outputs": outputs}
    if args.save:
        torch.save(result, args.save)
    if args.reference:
        ref = torch.load(args.reference, weights_only=True)
        assert len(captured) == len(ref["logits"])
        for step, (actual, expected) in enumerate(zip(captured, ref["logits"])):
            assert torch.equal(actual, expected), f"logits differ at step {step}: {(actual.float()-expected.float()).abs().max()}"
        assert outputs == ref["outputs"]
        print(f"PASS: {len(captured)} steps bitwise logits, greedy and seeded sampling, chunked/mixed prefill, page reuse")
    if args.hf:
        from transformers import AutoModelForCausalLM
        hf = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=engine.dtype, local_files_only=True,
                                                attn_implementation="eager").cuda().eval()
        rows = []
        for prompt in ["你好，请介绍一下自己。", "The capital of France is", "1 + 1 ="]:
            request = engine.add_request(prompt, SamplingParams(temperature=0, max_tokens=1))
            engine.run()
            ours = captured[-1].float()
            ids = torch.tensor([request.prompt_token_ids], device="cuda")
            with torch.inference_mode():
                expected = hf(ids).logits[:, -1].float().cpu()
            relative_l2 = ((ours - expected).norm() / expected.norm()).item()
            match = ours.argmax(-1).item() == expected.argmax(-1).item()
            rows.append({"prompt": prompt, "max_abs": (ours-expected).abs().max().item(),
                         "relative_l2": relative_l2, "top1_equal": match})
            assert relative_l2 < 0.03 and match
        print("HF eager reference (BF16 rounding tolerance, not bitwise):", json.dumps(rows, ensure_ascii=False))
    print("PASS: execution correctness checks completed")


if __name__ == "__main__":
    main()
