from dataclasses import dataclass
from collections import deque
from enum import Enum, auto

from .sampler import SamplingParams
from .kv_cache_manager import KVCacheManager


class RequestStatus(Enum):
    WAITING = auto()
    RUNNING = auto()
    FINISHED = auto()


class Request:
    def __init__(self, request_id: int, prompt_token_ids: list[int], sampling_params: SamplingParams):
        self.request_id = request_id
        self.prompt_token_ids = prompt_token_ids
        self.output_token_ids: list[int] = []
        self.all_token_ids = prompt_token_ids.copy()

        self.sampling_params = sampling_params
        self.num_computed_tokens = 0
        self.status = RequestStatus.WAITING

    @property
    def num_tokens(self) -> int:
        return len(self.all_token_ids)

    @property
    def num_prompt_tokens(self) -> int:
        return len(self.prompt_token_ids)

    @property
    def num_output_tokens(self) -> int:
        return len(self.output_token_ids)

    def append_output_token(self, token_id: int) -> None:
        self.output_token_ids.append(token_id)
        self.all_token_ids.append(token_id)

    def is_finished(self) -> bool:
        return self.status == RequestStatus.FINISHED


@dataclass
class SchedulerOutput:
    scheduled_requests: list[Request]
    num_scheduled_tokens: dict[int, int]

    @property
    def total_num_scheduled_tokens(self) -> int:
        return sum(self.num_scheduled_tokens.values())


class Scheduler:
    def __init__(self, kv_cache_manager: KVCacheManager, max_num_seqs: int, max_num_batched_tokens: int):
        self.kv_cache_manager = kv_cache_manager
        self.max_num_seqs = max_num_seqs
        self.max_num_batched_tokens = max_num_batched_tokens

        self.waiting: deque[Request] = deque()
        self.running: list[Request] = []

    def add_request(self, request: Request) -> None:
        request.status = RequestStatus.WAITING
        self.waiting.append(request)

    def schedule(self) -> SchedulerOutput:
        scheduled_requests = []
        num_scheduled_tokens = {}

        token_budget = self.max_num_batched_tokens

        # 1. 先调度已经 running 的 request
        for request in self.running:
            if token_budget == 0:
                break

            num_new_tokens = request.num_tokens - request.num_computed_tokens
            num_new_tokens = min(num_new_tokens, token_budget)

            if num_new_tokens <= 0:
                continue

            new_blocks = self.kv_cache_manager.allocate_slots(request, num_new_tokens)

            if new_blocks is None:
                continue

            scheduled_requests.append(request)
            num_scheduled_tokens[request.request_id] = num_new_tokens
            token_budget -= num_new_tokens

        # 2. 剩余 sequence slot + token budget 给 waiting request
        while self.waiting and len(self.running) < self.max_num_seqs and token_budget > 0:
            request = self.waiting[0]

            num_new_tokens = request.num_tokens - request.num_computed_tokens
            num_new_tokens = min(num_new_tokens, token_budget)

            if num_new_tokens <= 0:
                break

            new_blocks = self.kv_cache_manager.allocate_slots(request, num_new_tokens)

            if new_blocks is None:
                break

            self.waiting.popleft()
            self.running.append(request)
            request.status = RequestStatus.RUNNING

            scheduled_requests.append(request)
            num_scheduled_tokens[request.request_id] = num_new_tokens
            token_budget -= num_new_tokens

        return SchedulerOutput(
            scheduled_requests=scheduled_requests,
            num_scheduled_tokens=num_scheduled_tokens,
        )

    def finish_request(self, request: Request) -> None:
        if request in self.running:
            self.running.remove(request)

        self.kv_cache_manager.free(request.request_id)
        request.status = RequestStatus.FINISHED
