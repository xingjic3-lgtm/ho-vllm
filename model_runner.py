from dataclasses import dataclass

import torch

from .scheduler import SchedulerOutput
from .kv_cache_manager import KVCacheManager
from .forward_context import AttentionMetadata, set_forward_context


@dataclass
class ModelInput:
    input_ids: torch.Tensor
    positions: torch.Tensor
    attn_metadata: AttentionMetadata
    logits_indices: torch.Tensor


class ModelRunner:
    def __init__(self, model, kv_cache_manager: KVCacheManager, config: dict, device: torch.device, dtype: torch.dtype):
        self.model = model
        self.kv_cache_manager = kv_cache_manager
        self.config = config
        self.device = device
        self.dtype = dtype

        self.kv_cache = torch.empty(
            2,
            config["num_hidden_layers"],
            kv_cache_manager.block_pool.num_blocks,
            kv_cache_manager.block_size,
            config["num_key_value_heads"],
            config["head_dim"],
            dtype=dtype,
            device=device,
        )

        self.bind_kv_cache()

    def bind_kv_cache(self):
        for layer_id, layer in enumerate(self.model.model.layers):
            layer.self_attn.bind_kv_cache(self.kv_cache[:, layer_id])

    def prepare_inputs(self, scheduler_output: SchedulerOutput) -> ModelInput:
        input_ids = []
        positions = []
        query_start_loc = [0]
        seq_lens = []
        block_tables = []
        slot_mapping = []
        logits_indices = []

        for request in scheduler_output.scheduled_requests:
            request_id = request.request_id
            num_scheduled_tokens = scheduler_output.num_scheduled_tokens[request_id]

            start = request.num_computed_tokens
            end = start + num_scheduled_tokens

            input_ids.extend(request.all_token_ids[start:end])
            positions.extend(range(start, end))
            query_start_loc.append(query_start_loc[-1] + num_scheduled_tokens)
            seq_lens.append(end)

            # 只有本轮已经算到当前 sequence 尾部，才需要 logits
            if end == request.num_tokens:
                logits_indices.append(query_start_loc[-1] - 1)

            block_ids = self.kv_cache_manager.get_block_ids(request_id)
            block_tables.append(block_ids)

            for position in range(start, end):
                logical_block_id = position // self.kv_cache_manager.block_size
                block_offset = position % self.kv_cache_manager.block_size
                physical_block_id = block_ids[logical_block_id]
                slot = physical_block_id * self.kv_cache_manager.block_size + block_offset
                slot_mapping.append(slot)

        max_num_blocks = max(len(blocks) for blocks in block_tables)
        block_table = [blocks + [-1] * (max_num_blocks - len(blocks)) for blocks in block_tables]

        input_ids = torch.tensor(input_ids, dtype=torch.long, device=self.device)
        positions = torch.tensor(positions, dtype=torch.long, device=self.device)
        query_start_loc = torch.tensor(query_start_loc, dtype=torch.int32, device=self.device)
        seq_lens = torch.tensor(seq_lens, dtype=torch.int32, device=self.device)
        block_table = torch.tensor(block_table, dtype=torch.int32, device=self.device)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.long, device=self.device)
        logits_indices = torch.tensor(logits_indices, dtype=torch.long, device=self.device)

        attn_metadata = AttentionMetadata(
            query_start_loc=query_start_loc,
            seq_lens=seq_lens,
            block_table=block_table,
            slot_mapping=slot_mapping,
        )

        return ModelInput(input_ids=input_ids, positions=positions, logits_indices=logits_indices, attn_metadata=attn_metadata)

    def execute_model(self, model_input: ModelInput) -> torch.Tensor:
        with torch.inference_mode():
            with set_forward_context(model_input.attn_metadata):
                hidden_states = self.model(model_input.input_ids, model_input.positions)


            sample_hidden_states = hidden_states[model_input.logits_indices]
            logits = self.model.compute_logits(sample_hidden_states)
        return logits