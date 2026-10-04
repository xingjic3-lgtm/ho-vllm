"""沿用 v0 的职责划分，组织自己的模型、加载器和采样器。"""
import torch
from transformers import AutoTokenizer

from .model.loader import load_weight
from .sampler import Sampler, SamplingParams
from collections import deque

engine_config = {
    "max_running_requests": 4,
}


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
        self.scheduler = Scheduler(engine_config["max_running_requests"])

    def add_request(self, prompt, sampling_params):
        inputs = self.tokenizer(prompt, return_tensors='pt')
        request = Request(inputs["input_ids"][0], sampling_params)
        self.scheduler.add_request(request)
        return request

    def run(self):
        while (self.scheduler.waiting_requests or self.scheduler.running_requests ):
            self.step()

    def step(self, ):
        requests = self.scheduler.schedule()
        if not requests:
            return
        self.execute(requests)

    def execute(self, requests):
        max_requestlen = 0
        for request in requests:
            if request.num_tokens > max_requestlen:
                max_requestlen = request.num_tokens

        B = len(requests)

        batch = torch.full((B, max_requestlen), self.tokenizer.pad_token_id, device=self.device, dtype=requests[0].token_ids.dtype)    # 初始化先填充一个pad的矩阵  后面再用索引batch[i, :num_tokens]覆盖填入
        mask = torch.zeros((B, max_requestlen), device=self.device, dtype=torch.bool)
        positions = torch.zeros((B, max_requestlen), device=self.device, dtype=torch.long)
        for i in range(B):
            n = requests[i].num_tokens
            batch[i, :n] = requests[i].token_ids.to(self.device)
            mask[i, :n] = True
            positions[i, :n] =  torch.arange(n, device=self.device)


        with torch.inference_mode():
            outputs = self.model(batch, positions, mask)

        filished_requests = []
        for i,request in enumerate(requests):
            logits = outputs[i, request.num_tokens - 1 ,:]
            next_token = self.sampler.sample(logits, request.sampling_params)
            request.token_ids = torch.cat([request.token_ids, next_token.cpu()])
            request.generated_tokens += 1 

            # 判断request是否结束
            if (next_token == self.tokenizer.eos_token_id or request.generated_tokens >= request.max_tokens):
                filished_requests.append(request)

        self.scheduler.remove_requests(filished_requests)

    # 第一版的benchmark测试用        
    def generate(self, prompt: str, sampling_params: SamplingParams):
        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(self.device)

        for step in range(sampling_params.max_tokens):
            with torch.inference_mode():
                if input_ids.shape[1] >= self.config["max_position_embeddings"]:
                    break
                B, T = input_ids.shape

                # 单 request，没有 padding，全部都是真实 token
                mask = torch.ones((B, T), device=self.device, dtype=torch.bool)

                # [B,T]
                positions = torch.arange( T, device=self.device)[None, :]

                output = self.model( input_ids, positions, mask, )

                logits = output[:, -1, :]
                next_token = self.sampler.sample(logits, sampling_params)

                input_ids = torch.cat([input_ids, next_token], dim=-1 )

                if next_token.item() == self.eos_token_id:
                    break
            yield self.tokenizer.decode(input_ids[0])


class Request:
    def __init__(self, token_ids:torch.Tensor, sampling_params:SamplingParams):
        self.token_ids = token_ids
        # self.block_ids = []
        self.max_tokens = sampling_params.max_tokens
        self.generated_tokens = 0
        self.sampling_params = sampling_params

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

    def remove_requests(self, requests:list):
        for request in requests:
            self.running_requests.remove(request)