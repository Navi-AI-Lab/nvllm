# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright contributors to the nvllm fork
"""Pure PyTorch reference paged attention with FP8 KV dequantization.

Used by the production `paged_attention_forward` as a fallback when the
CuTe DSL kernel is unavailable (CUTLASS import failure) or when the
current batch is a prefill (the CuTe prefill body is not implemented).
Also the ground-truth reference for kernel correctness tests.

Not optimized — clarity over performance. Moved out of the `tests/`
package on 2026-05-19 so the production backend no longer imports from
`tests.*` (audit Finding 1.5).
"""
import torch
import torch.nn.functional as F


def _gather_kv_pages(
    cache: torch.Tensor,     # [num_pages, page_size, num_kv_heads, head_dim] uint8
    page_table: torch.Tensor,  # [max_pages_per_seq] int32
    seq_len: int,
    page_size: int,
) -> torch.Tensor:
    """Gather KV tokens from paged cache for a single sequence.

    Returns: [seq_len, num_kv_heads, head_dim] FP8 (as float8_e4m3fn)
    """
    num_pages_needed = (seq_len + page_size - 1) // page_size
    chunks = []
    for p in range(num_pages_needed):
        page_idx = page_table[p].item()
        tokens_in_page = min(page_size, seq_len - p * page_size)
        chunk = cache[page_idx, :tokens_in_page]
        chunks.append(chunk)
    gathered = torch.cat(chunks, dim=0)
    return gathered.view(torch.float8_e4m3fn)


def reference_paged_attention(
    query: torch.Tensor,        # [num_tokens, num_q_heads, head_dim] BF16
    k_cache: torch.Tensor,      # [num_pages, page_size, num_kv_heads, head_dim] uint8
    v_cache: torch.Tensor,      # [num_pages, page_size, num_kv_heads, head_dim] uint8
    page_table: torch.Tensor,   # [num_seqs, max_pages_per_seq] int32
    seq_lens: torch.Tensor,     # [num_seqs] int32
    scale: float,
    k_scale: float = 1.0,
    v_scale: float = 1.0,
    page_size: int = 64,
    query_start_loc: torch.Tensor | None = None,  # [num_seqs + 1] int32
) -> torch.Tensor:
    """Full paged attention reference with GQA, causal mask, and page table.

    Matches the production paged_attention_forward() signature for direct
    comparison testing.

    Returns: [num_tokens, num_q_heads, head_dim] BF16
    """
    num_q_heads = query.shape[1]
    num_kv_heads = k_cache.shape[2]
    gqa_ratio = num_q_heads // num_kv_heads
    num_seqs = len(seq_lens)

    if query_start_loc is not None:
        qsl = query_start_loc.cpu()
        tokens_per_seq = (qsl[1:] - qsl[:-1]).tolist()
    else:
        tokens_per_seq = [1] * num_seqs

    outputs = []
    token_idx = 0

    for seq_idx in range(num_seqs):
        seq_len = seq_lens[seq_idx].item()
        num_query_tokens = tokens_per_seq[seq_idx]

        q = query[token_idx:token_idx + num_query_tokens]

        k_fp8 = _gather_kv_pages(
            k_cache, page_table[seq_idx], seq_len, page_size,
        )
        v_fp8 = _gather_kv_pages(
            v_cache, page_table[seq_idx], seq_len, page_size,
        )

        k_all = k_fp8.to(torch.bfloat16) * k_scale
        v_all = v_fp8.to(torch.bfloat16) * v_scale

        if gqa_ratio > 1:
            k_all = k_all.repeat_interleave(gqa_ratio, dim=1)
            v_all = v_all.repeat_interleave(gqa_ratio, dim=1)

        scores = torch.einsum(
            "qhd,shd->hqs", q.float(), k_all.float(),
        ) * scale

        for qi in range(num_query_tokens):
            kv_start = seq_len - num_query_tokens
            for si in range(seq_len):
                if si > kv_start + qi:
                    scores[:, qi, si] = float("-inf")

        attn_weights = F.softmax(scores, dim=-1)

        out = torch.einsum(
            "hqs,shd->qhd", attn_weights, v_all.float(),
        )

        outputs.append(out.to(torch.bfloat16))
        token_idx += num_query_tokens

    return torch.cat(outputs, dim=0)
