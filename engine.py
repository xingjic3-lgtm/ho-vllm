"""沿用 v0 的职责划分，组织自己的模型、加载器和采样器。"""
import torch
from transformers import AutoTokenizer

from .model.loader import load_weight
from .sampler import Sampler, SamplingParams
from collections import deque
from .kv_cache_manager import KVCacheManager
from .scheduler import Request, Scheduler
from .model_runner import ModelRunner

engine_config = {
    "num_blocks": 256,
    "block_size": 16,
    "max_num_seqs": 4,
    "max_num_batched_tokens": 256,
}


class Engine:
    def __init__(self, model_path: str, config: dict, device: str = "cuda"):
        self.config = config
        self.device = torch.device(device)
        self.dtype = getattr(torch, config.get("dtype") or config["torch_dtype"])
        self.eos_token_id = config["eos_token_id"]

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    
        self.model = load_weight(model_path, config, self.device, self.dtype)
        self.sampler = Sampler()

        self.kv_cache_manager = KVCacheManager(
            num_blocks=engine_config["num_blocks"],
            block_size=engine_config["block_size"],
        )

        self.scheduler = Scheduler(
            kv_cache_manager=self.kv_cache_manager,
            max_num_seqs=engine_config["max_num_seqs"],
            max_num_batched_tokens=engine_config["max_num_batched_tokens"],
        )

        self.model_runner = ModelRunner(
            model=self.model,
            kv_cache_manager=self.kv_cache_manager,
            config=config,
            device=self.device,
            dtype=self.dtype,
        )
        self.next_request_id = 0

    def add_request(self, prompt: str, sampling_params: SamplingParams) -> Request:
        inputs = self.tokenizer(prompt, return_tensors="pt")
        prompt_token_ids = inputs["input_ids"][0].tolist()

        request = Request(
            request_id=self.next_request_id,
            prompt_token_ids=prompt_token_ids,
            sampling_params=sampling_params,
        )

        self.next_request_id += 1
        self.scheduler.add_request(request)

        return request

    def run(self) -> None:
        while self.scheduler.waiting or self.scheduler.running:
            self.step()

    def step(self) -> None:
        scheduler_output = self.scheduler.schedule()

        if not scheduler_output.scheduled_requests:
            return

        model_input = self.model_runner.prepare_inputs(scheduler_output)
        logits = self.model_runner.execute_model(model_input)

        finished_requests = []
        logit_index = 0

        for request in scheduler_output.scheduled_requests:
            request_id = request.request_id
            num_scheduled_tokens = scheduler_output.num_scheduled_tokens[request_id]

            reaches_sequence_end = request.num_computed_tokens + num_scheduled_tokens == request.num_tokens

            request.num_computed_tokens += num_scheduled_tokens

            if not reaches_sequence_end:
                continue

            next_token = self.sampler.sample(
                logits[logit_index:logit_index + 1],
                request.sampling_params,
            )
            logit_index += 1

            token_id = next_token.item()
            request.append_output_token(token_id)

            if token_id == self.eos_token_id:
                finished_requests.append(request)
                continue

            if request.num_output_tokens >= request.sampling_params.max_tokens:
                finished_requests.append(request)
                continue

            if request.num_tokens >= self.config["max_position_embeddings"]:
                finished_requests.append(request)
        for request in finished_requests:
            self.scheduler.finish_request(request)