"""沿用 v0 的职责划分，组织自己的模型、加载器和采样器。"""
import torch
from transformers import AutoTokenizer

from .model.loader import load_weight
from .sampler import Sampler, SamplingParams
from collections import deque

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
        self.scheduler = Scheduler()

    def add_request(self, prompt, sampling_params):
        inputs = self.tokenizer(prompt, return_tensors='pt')
        request = Request(inputs["input_ids"][0], sampling_params.max_tokens)
        self.scheduler.add_request(request)

    def step(self, ):
        requests = self.scheduler.schedule()
        self.execute(requests)

    def execute(self, requests):
        max_requestlen = 0
        for request in requests:
            if request.num_tokens > max_requestlen:
                max_requestlen = request.num_tokens

        batch = torch.tensor((len(requests), max_requestlen))
        for i in range(len(requests)):
            batch[i] = requests[i].token_ids
            
        
        


    def generate(self, prompt: str, sampling_params: SamplingParams):
        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(self.device)

        kv_cache = torch.empty(kv, layer_id, num_block, block_size,num_kv_head, head_dim)   # 依次是kv=2  layer_id=28 num_block=? block_size=?    num_kv_head=kv的头数   head_dim=head的维度
        
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


class Request:
    def __init__(self, token_ids:torch.Tensor, max_tokens:int):
        self.token_ids = token_ids
        # self.block_ids = []
        self.max_tokens = max_tokens
        self.generated_tokens = 0

    @property
    def num_tokens(self):
        return self.token_ids.shape[-1]
        

    # def append_block(self, block:Block):
    #     self.block_ids.append(block)

class Scheduler:
    def __init__(self, max_running_requests:int):
        self.running_requests = []
        self.waiting_requests = deque()   # 先进先出
        self.max_running_requests = max_running_requests

    def add_request(self, request:Request):
        self.waiting_requests.append(request)

    def schedule(self,):
        while (len(self.running_requests) < self.max_running_requests and self.waiting_requests):
            request = self.waiting_requests.popleft()
            self.running_requests.append(request)
        return self.running_requests

# class Block:
#     def __init__(self, physical_id: int):
#         self.physical_id = physical_id