from dataclasses import dataclass
from contextlib import contextmanager

import torch


@dataclass
class AttentionMetadata:
    query_start_loc: torch.Tensor
    seq_lens: torch.Tensor
    block_table: torch.Tensor
    slot_mapping: torch.Tensor


@dataclass
class ForwardContext:
    attn_metadata: AttentionMetadata


_forward_context: ForwardContext | None = None


def get_forward_context() -> ForwardContext:
    assert _forward_context is not None
    return _forward_context


@contextmanager
def set_forward_context(attn_metadata: AttentionMetadata):
    global _forward_context
    _forward_context = ForwardContext(attn_metadata)

    try:
        yield
    finally:
        _forward_context = None