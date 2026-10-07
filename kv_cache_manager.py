from dataclasses import dataclass
from collections import deque
from typing import TYPE_CHECKING


@dataclass(slots=True)
class KVCacheBlock:
    block_id: int
    ref_cnt: int = 0


class BlockPool:
    def __init__(self, num_blocks: int):
        self.num_blocks = num_blocks
        self.blocks = [KVCacheBlock(i) for i in range(num_blocks)]
        self.free_blocks = deque(self.blocks)

    def allocate(self, num_blocks: int) -> list[KVCacheBlock] | None:
        if num_blocks > len(self.free_blocks):
            return None

        blocks = []

        for _ in range(num_blocks):
            block = self.free_blocks.popleft()
            block.ref_cnt += 1
            blocks.append(block)

        return blocks

    def free(self, blocks: list[KVCacheBlock]) -> None:
        for block in blocks:
            block.ref_cnt -= 1
            if block.ref_cnt == 0:
                self.free_blocks.append(block)

    def get_num_free_blocks(self) -> int:
        return len(self.free_blocks)


class KVCacheManager:
    def __init__(self, num_blocks: int, block_size: int):
        self.block_size = block_size
        self.block_pool = BlockPool(num_blocks)
        self.req_to_blocks: dict[int, list[KVCacheBlock]] = {}

    def allocate_slots(self, request, num_new_tokens: int) -> list[KVCacheBlock] | None:
        request_id = request.request_id
        current_blocks = self.req_to_blocks.get(request_id, [])

        num_required_tokens = request.num_computed_tokens + num_new_tokens
        num_required_blocks = (num_required_tokens + self.block_size - 1) // self.block_size
        num_new_blocks = num_required_blocks - len(current_blocks)

        if num_new_blocks <= 0:
            return []

        new_blocks = self.block_pool.allocate(num_new_blocks)

        if new_blocks is None:
            return None

        if request_id not in self.req_to_blocks:
            self.req_to_blocks[request_id] = []

        self.req_to_blocks[request_id].extend(new_blocks)

        return new_blocks

    def get_blocks(self, request_id: int) -> list[KVCacheBlock]:
        return self.req_to_blocks.get(request_id, [])

    def get_block_ids(self, request_id: int) -> list[int]:
        return [block.block_id for block in self.get_blocks(request_id)]

    def free(self, request_id: int) -> None:
        blocks = self.req_to_blocks.pop(request_id, None)

        if blocks is None:
            return

        self.block_pool.free(blocks)

    def get_num_free_blocks(self) -> int:
        return self.block_pool.get_num_free_blocks()