import torch
import triton
import triton.language as tl


@triton.jit
def _paged_gather_gqa_kernel(
    k_ptr,                  # [num_blocks, block_size, num_kv_heads, head_dim]
    v_ptr,                  # [num_blocks, block_size, num_kv_heads, head_dim]
    block_table_ptr,        # 当前 request 的 block_table

    k_out_ptr,              # [seq_len, num_heads, head_dim]
    v_out_ptr,

    NUM_HEADS: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,

    BLOCK_H: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    # 当前 program 负责第几个 token
    token_index = tl.program_id(0)

    # 当前 token 的所有 Q heads
    head_index = tl.arange(0, BLOCK_H)[:, None]

    # 每个 head 内的所有维度
    dim_index = tl.arange(0, BLOCK_D)[None, :]

    mask = ( (head_index < NUM_HEADS)  & (dim_index < HEAD_DIM) )

    # ------------------------------------------------
    # 1. Paged KV：逻辑 token → 物理 block
    # ------------------------------------------------

    logical_block = token_index // BLOCK_SIZE
    block_offset = token_index % BLOCK_SIZE

    physical_block = tl.load( block_table_ptr + logical_block )

    # ------------------------------------------------
    # 2. GQA：Q head → KV head
    # ------------------------------------------------

    repeat = NUM_HEADS // NUM_KV_HEADS

    kv_head_index = head_index // repeat

    # ------------------------------------------------
    # 3. 算原始 paged KV cache 地址
    #
    # k/v shape:
    # [num_blocks, block_size, num_kv_heads, head_dim]
    # ------------------------------------------------

    src_offset = (
        (
            (
                physical_block * BLOCK_SIZE
                + block_offset
            ) * NUM_KV_HEADS
            + kv_head_index
        ) * HEAD_DIM
        + dim_index
    )

    # 直接从 Paged KV Cache 读取
    k = tl.load(
        k_ptr + src_offset,
        mask=mask,
        other=0.0,
    )

    v = tl.load(
        v_ptr + src_offset,
        mask=mask,
        other=0.0,
    )

    # ------------------------------------------------
    # 4. 输出地址
    #
    # output shape:
    # [seq_len, num_heads, head_dim]
    # ------------------------------------------------

    dst_offset = (
        (
            token_index * NUM_HEADS
            + head_index
        ) * HEAD_DIM
        + dim_index
    )

    tl.store(
        k_out_ptr + dst_offset,
        k,
        mask=mask,
    )

    tl.store(
        v_out_ptr + dst_offset,
        v,
        mask=mask,
    )


def paged_gather_gqa(
    kv_cache,
    block_table,
    seq_len,
    num_heads,
    num_kv_heads,
    head_dim,
    block_size,
):
    # kv_cache:
    # [2, num_blocks, block_size, num_kv_heads, head_dim]

    if block_table.ndim != 1 or not block_table.is_contiguous():
        raise ValueError("block_table must be one contiguous request row: block_table[i]")

    k_cache = kv_cache[0]
    v_cache = kv_cache[1]

    k_out = torch.empty(
        (seq_len, num_heads, head_dim),
        device=kv_cache.device,
        dtype=kv_cache.dtype,
    )

    v_out = torch.empty_like(k_out)

    BLOCK_H = triton.next_power_of_2(num_heads)
    BLOCK_D = triton.next_power_of_2(head_dim)

    grid = (seq_len,)

    _paged_gather_gqa_kernel[grid](
        k_cache,
        v_cache,
        block_table,
        k_out,
        v_out,

        NUM_HEADS=num_heads,
        NUM_KV_HEADS=num_kv_heads,
        HEAD_DIM=head_dim,
        BLOCK_SIZE=block_size,

        BLOCK_H=BLOCK_H,
        BLOCK_D=BLOCK_D,

        num_warps=4,
    )

    return k_out, v_out
