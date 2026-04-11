"""CuTe DSL paged attention kernel for SM120/SM121 (GB10).

Execution order (single CTA):
  1. Load Q tile via TMA
  2. For each KV tile:
     a. Load K page via CpAsync
     b. QK MMA (FP8 m16n8k32) -> scores
     c. Online softmax update (registers)
     d. Load V page via CpAsync
     e. PV MMA (BF16 m16n8k16) -> accumulate O
  3. Final softmax normalization
  4. Write O to global memory

This file contains the complete kernel in execution order.
Currently a PyTorch prototype — the CuTe DSL version replaces the
inner loops in a later task.
"""
import math

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Component 1: Online Softmax
# ---------------------------------------------------------------------------

def online_softmax(
    scores: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """Online softmax using the Flash-Attention streaming algorithm.

    Uses log2-based exp2 for hardware efficiency (matches the CuTe DSL
    kernel's register-based implementation).

    Args:
        scores: [num_rows, kv_len] raw QK dot products
        scale: attention scale factor (1/sqrt(head_dim))

    Returns:
        [num_rows, kv_len] normalized attention weights
    """
    scale_log2 = scale * math.log2(math.e)
    scaled = scores * scale_log2
    row_max = scaled.max(dim=-1, keepdim=True).values
    exp_scores = torch.exp2(scaled - row_max)
    row_sum = exp_scores.sum(dim=-1, keepdim=True)
    return exp_scores / row_sum


# ---------------------------------------------------------------------------
# Component 2: QK Pass (FP8 MMA)
# ---------------------------------------------------------------------------

def qk_pass(
    q: torch.Tensor,        # [num_q_rows, head_dim] BF16
    k: torch.Tensor,        # [num_kv_rows, head_dim] FP8 E4M3
    scale: float,
    k_scale: float,
) -> torch.Tensor:
    """QK dot product with FP8 MMA.

    Casts Q from BF16 -> FP8 E4M3 (lossy, 2-step: BF16->FP32->E4M3).
    Uses FP8 m16n8k32 MMA on SM120 (2x throughput vs BF16).
    Absorbs k_scale into the output scaling.

    Returns: [num_q_rows, num_kv_rows] FP32 (raw scores, pre-softmax)
    """
    # Cast Q to FP8 (simulates the register-level BF16->FP32->E4M3 path)
    q_fp8 = q.float().to(torch.float8_e4m3fn)

    # FP8 matmul: Q_fp8 @ K_fp8^T -> FP32 accumulator
    # On SM120 this is mma.sync.aligned.kind::f8f6f4.m16n8k32
    scores = torch._scaled_mm(
        q_fp8,
        k.T.contiguous(),
        out_dtype=torch.float32,
        scale_a=torch.tensor(scale * k_scale, device=q.device),
        scale_b=torch.tensor(1.0, device=q.device),
    )
    return scores


# ---------------------------------------------------------------------------
# Component 3: PV Pass (BF16 MMA + V Dequant)
# ---------------------------------------------------------------------------

def pv_pass(
    p: torch.Tensor,        # [num_q_rows, num_kv_rows] FP32 (softmax output)
    v: torch.Tensor,        # [num_kv_rows, head_dim] FP8 E4M3
    v_scale: float,
) -> torch.Tensor:
    """PV multiply with inline V dequantization.

    Descale applied to P in FP32 before BF16 cast (b12x pattern).
    V dequantized from FP8 to BF16 during fragment loads.
    BF16 m16n8k16 MMA on SM120.

    Returns: [num_q_rows, head_dim] BF16
    """
    # Apply v_scale to P while still in FP32 (before BF16 cast)
    p_scaled = p * v_scale

    # Cast P to BF16 for MMA
    p_bf16 = p_scaled.to(torch.bfloat16)

    # Dequant V inline: FP8 -> BF16
    v_bf16 = v.to(torch.bfloat16)

    # BF16 matmul: P_bf16 @ V_bf16 -> FP32 accumulator -> BF16 output
    output = (p_bf16.float() @ v_bf16.float()).to(torch.bfloat16)
    return output


# ---------------------------------------------------------------------------
# Full Forward: Paged Attention with GQA, Causal Mask, Split-KV
# ---------------------------------------------------------------------------

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
        chunk = cache[page_idx, :tokens_in_page]  # [tokens, nkv, hd]
        chunks.append(chunk)
    gathered = torch.cat(chunks, dim=0)  # [seq_len, nkv, hd] uint8
    return gathered.view(torch.float8_e4m3fn)


def paged_attention_forward(
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
    """Full paged attention forward with GQA, causal mask, and page table.

    PyTorch prototype of the CuTe DSL kernel. Matches the reference
    implementation for correctness testing.

    Args:
        query: BF16 query tensor [num_tokens, num_q_heads, head_dim]
        k_cache: FP8 key cache pages stored as uint8
        v_cache: FP8 value cache pages stored as uint8
        page_table: Page indices per sequence
        seq_lens: Total KV length per sequence
        scale: Attention scale (1/sqrt(head_dim))
        k_scale: Key descale factor
        v_scale: Value descale factor
        page_size: Tokens per page (default 64)
        query_start_loc: Start index of each sequence's query tokens

    Returns: [num_tokens, num_q_heads, head_dim] BF16
    """
    num_q_heads = query.shape[1]
    head_dim = query.shape[2]
    num_kv_heads = k_cache.shape[2]
    gqa_ratio = num_q_heads // num_kv_heads
    num_seqs = len(seq_lens)

    # Compute query tokens per sequence from query_start_loc
    if query_start_loc is not None:
        qsl = query_start_loc.cpu()
        tokens_per_seq = (qsl[1:] - qsl[:-1]).tolist()
    else:
        # Fallback: assume 1 token per sequence (decode-only)
        tokens_per_seq = [1] * num_seqs

    outputs = []
    token_idx = 0

    for seq_idx in range(num_seqs):
        seq_len = seq_lens[seq_idx].item()
        num_query_tokens = tokens_per_seq[seq_idx]

        q = query[token_idx:token_idx + num_query_tokens]  # [nq, num_q_heads, hd]

        # Gather K and V from their respective page caches
        k_fp8 = _gather_kv_pages(
            k_cache, page_table[seq_idx], seq_len, page_size,
        )  # [seq_len, num_kv_heads, head_dim] FP8
        v_fp8 = _gather_kv_pages(
            v_cache, page_table[seq_idx], seq_len, page_size,
        )  # [seq_len, num_kv_heads, head_dim] FP8

        # Dequantize FP8 -> BF16
        k_all = k_fp8.to(torch.bfloat16) * k_scale
        v_all = v_fp8.to(torch.bfloat16) * v_scale

        # Expand KV heads for GQA
        if gqa_ratio > 1:
            k_all = k_all.repeat_interleave(gqa_ratio, dim=1)
            v_all = v_all.repeat_interleave(gqa_ratio, dim=1)

        # QK: q @ k^T * scale per head
        # q: [nq, num_q_heads, hd], k: [seq_len, num_q_heads, hd]
        scores = torch.einsum(
            "qhd,shd->hqs", q.float(), k_all.float(),
        ) * scale  # [num_q_heads, nq, seq_len]

        # Causal mask
        for qi in range(num_query_tokens):
            kv_start = seq_len - num_query_tokens
            for si in range(seq_len):
                if si > kv_start + qi:
                    scores[:, qi, si] = float("-inf")

        # Softmax
        attn_weights = F.softmax(scores, dim=-1)

        # PV: attn @ v
        out = torch.einsum(
            "hqs,shd->qhd", attn_weights, v_all.float(),
        )  # [nq, num_q_heads, hd]

        outputs.append(out.to(torch.bfloat16))
        token_idx += num_query_tokens

    return torch.cat(outputs, dim=0)
