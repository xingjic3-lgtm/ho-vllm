import torch
from torch.nn import functional as F
from dataclasses import dataclass


@dataclass
class SamplingParams:
    temperature: float = 1.0
    top_k: int = 0
    top_p: float = 1.0
    max_tokens: int = 16


class Sampler:

    def sample(
        self,
        logits: torch.Tensor,
        sampling_params: SamplingParams
    ):
        temperature = sampling_params.temperature
        top_k = sampling_params.top_k
        top_p = sampling_params.top_p

        if temperature < 0:
            raise ValueError("temperature 不能小于 0")

        if temperature == 0:
            return logits.argmax(dim=-1, keepdim=True)

        # temperature
        logits = logits / temperature

        # top-k
        logits = self.top_k(logits, top_k)

        # top-p
        logits = self.top_p(logits, top_p)

        # 转成概率
        probs = F.softmax(logits, dim=-1)

        # 按概率采样
        next_token = torch.multinomial(probs, num_samples=1)

        return next_token


    def top_k(self, logits: torch.Tensor, top_k: int):
        if not 0 <= top_k <= logits.shape[-1]:
            raise ValueError("top_k 必须在 0 到词表大小之间")

        if top_k == 0:
            return logits

        values, indices = torch.topk(logits, k=top_k, dim=-1)

        filtered = torch.full_like(logits, float("-inf"))

        filtered.scatter_(
            dim=-1,
            index=indices,
            src=values
        )

        return filtered


    def top_p(self, logits: torch.Tensor, top_p: float):
        if not 0 < top_p <= 1:
            raise ValueError("top_p 必须满足 0 < top_p <= 1")

        if top_p == 1:
            return logits

        # 1. logits 从大到小排序
        sorted_logits, sorted_indices = torch.sort(
            logits,
            descending=True,
            dim=-1
        )

        # 2. 转成概率
        sorted_probs = F.softmax(sorted_logits, dim=-1)

        # 3. 累计概率
        cumulative_probs = torch.cumsum(
            sorted_probs,
            dim=-1
        )

        # 4. 当前 token 加入之前，
        #    累计概率是否已经达到 top_p
        remove = (
            cumulative_probs - sorted_probs
        ) >= top_p
        remove[..., 0] = False

        # 5. 删除这些 token
        sorted_logits = sorted_logits.masked_fill(
            remove,
            float("-inf")
        )

        # 6. 恢复到原来的 vocab 位置
        filtered = torch.full_like(
            logits,
            float("-inf")
        )

        filtered.scatter_(
            dim=-1,
            index=sorted_indices,
            src=sorted_logits
        )

        return filtered
