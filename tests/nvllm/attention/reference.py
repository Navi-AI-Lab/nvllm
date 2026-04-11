"""Pure PyTorch reference attention with FP8 KV dequantization.

Ground truth for testing the CuTe DSL paged attention kernel.
Not optimized — clarity over performance.
"""
import torch
import torch.nn.functional as F


def reference_paged_attention(
    query: torch.Tensor,        # [num_tokens, num_q_heads, head_dim] BF16
    kv_cache: torch.Tensor,     # [num_pages, page_size, num_kv_heads, head_dim] uint8 (FP8)
    page_table: torch.Tensor,   # [num_seqs, max_pages_per_seq] int32
    seq_lens: torch.Tensor,     # [num_seqs] int32
    scale: float,
    k_scale: float = 1.0,
    v_scale: float = 1.0,
    page_size: int = 64,
) -> torch.Tensor:
    """Reference paged attention with FP8 KV and per-layer descale.

    Handles GQA by repeating KV heads to match Q head count.
    Handles causal masking per sequence.
    Returns: [num_tokens, num_q_heads, head_dim] BF16

    Note: This simplified reference uses the same cache data for both K
    and V (clone). It will be refined when the actual kernel's KV layout
    is finalized (separate K/V halves or interleaved pages).
    """
    num_q_heads = query.shape[1]
    head_dim = query.shape[2]
    num_kv_heads = kv_cache.shape[2]
    gqa_ratio = num_q_heads // num_kv_heads

    outputs = []
    token_idx = 0
    total_tokens = query.shape[0]
    num_seqs = len(seq_lens)

    for seq_idx in range(num_seqs):
        seq_len = seq_lens[seq_idx].item()
        # Detect prefill vs decode: if total query tokens > num sequences,
        # this is prefill and each sequence has seq_len query tokens.
        if total_tokens > num_seqs:
            num_query_tokens = seq_len
        else:
            num_query_tokens = 1

        q = query[token_idx:token_idx + num_query_tokens]  # [nq, num_q_heads, hd]

        # Gather KV from pages
        num_pages_needed = (seq_len + page_size - 1) // page_size
        k_list, v_list = [], []
        for p in range(num_pages_needed):
            page_idx = page_table[seq_idx, p].item()
            tokens_in_page = min(page_size, seq_len - p * page_size)
            page_k = kv_cache[page_idx, :tokens_in_page]  # [tokens, nkv, hd]
            k_list.append(page_k)
            v_list.append(page_k.clone())  # Same page holds both K and V in this ref

        # For a real impl, K and V are separate cache halves.
        # Here we pass kv_cache as K-only and use a separate v_cache param.
        # Simplified: assume caller passes K cache as kv_cache.

        # Dequantize FP8 -> BF16
        k_all = torch.cat(k_list, dim=0).view(torch.float8_e4m3fn).to(torch.bfloat16) * k_scale
        v_all = torch.cat(v_list, dim=0).view(torch.float8_e4m3fn).to(torch.bfloat16) * v_scale

        # Repeat KV for GQA
        if gqa_ratio > 1:
            k_all = k_all.repeat_interleave(gqa_ratio, dim=1)  # [seq_len, num_q_heads, hd]
            v_all = v_all.repeat_interleave(gqa_ratio, dim=1)

        # Attention: Q @ K^T * scale, causal mask, softmax, @ V
        # q: [nq, num_q_heads, hd], k: [seq_len, num_q_heads, hd]
        scores = torch.einsum("qhd,shd->hqs", q.float(), k_all.float()) * scale

        # Causal mask
        for qi in range(num_query_tokens):
            kv_start = seq_len - num_query_tokens  # for prefill offset
            for si in range(seq_len):
                if si > kv_start + qi:
                    scores[:, qi, si] = float("-inf")

        attn_weights = F.softmax(scores, dim=-1)
        out = torch.einsum("hqs,shd->qhd", attn_weights, v_all.float())

        outputs.append(out.to(torch.bfloat16))
        token_idx += num_query_tokens

    return torch.cat(outputs, dim=0)
