"""沿用 v0 的职责划分，组织自己的模型、加载器和采样器。"""
import torch
from transformers import AutoTokenizer

from .model.loader import load_weight
from .sampler import Sampler, SamplingParams


class Engine:
    def __init__(self, model_path: str, config: dict, device: str = "cuda"):
        self.config = config
        self.device = torch.device(device)
        self.dtype = getattr(torch, config.get("dtype") or config["torch_dtype"])
        self.eos_token_id = config["eos_token_id"]
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, local_files_only=True
        )
        self.model = load_weight(
            model_path, config, self.device, self.dtype
        )
        self.sampler = Sampler()

    def generate(self, prompt: str, sampling_params: SamplingParams):
        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(self.device)


        for step in range(sampling_params.max_tokens):
            with torch.inference_mode():
                if input_ids.shape[1] >= self.config["max_position_embeddings"]:
                    break

                positions = torch.arange(input_ids.shape[1], device=self.device)
                output = self.model(input_ids, positions)
                logits = output[:, -1, :]
                next_token = self.sampler.sample(logits, sampling_params)
                input_ids = torch.cat([input_ids, next_token], dim=-1)

                if next_token.item() == self.eos_token_id:
                    break
            yield self.tokenizer.decode(input_ids[0])
