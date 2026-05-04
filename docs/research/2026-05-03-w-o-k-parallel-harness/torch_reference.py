"""Torch FP32 references for the W_O microkernel.

Three references:
- ``reference_chained_fma``: production-order baseline. Mirrors the
  kernel's ``wo_split=1`` reduction order — per-KV-head chained
  accumulation along K (1 element at a time, in linear order),
  then a 4-way sum across KV heads. Bit-exact match for ``wo_split=1``.
- ``reference_split_order(wo_split)``: variant-order reference.
  Mirrors each variant's *own* reduction tree exactly — partitions
  K per CTA the same way ``microkernel.py:203-205`` does, performs
  chained-FMA accumulation per CTA partial, then sums slots in the
  same ``wo_slot = by * wo_split + bx`` order the gather pass uses
  (``phase_e_kernel.py:4233-4242``). At ``wo_split=1`` it is bit-
  identical to ``reference_chained_fma`` — verified by the Phase A
  sanity assert in the smoke harness.
- ``reference_matmul``: diagnostic only. Vectorised cuBLAS matmul,
  reported in artifacts but NOT a pass/fail oracle. cuBLAS uses an
  internal tree reduction whose order will not match the kernel's
  chained FMA along K.

All three references share the same dequant pipeline:

    w_dequant = fp4_decode(w) * fp8_decode(sf) * wo_gs

The dequant convention matches ``phase_e_kernel.py:4078-4082`` and
``microkernel.py:300``. Per ``feedback_nvfp4_dequant_convention``: the
kernel sees ``1/wgs`` and **multiplies**, so the reference also
multiplies (``wo_gs`` here is already in kernel-facing orientation —
the harness passes it that way).

Gate model (per README §4):

    | wo_split    | Authoritative gate                          | Diagnostic drift                        |
    |-------------|---------------------------------------------|-----------------------------------------|
    | 1           | reference_chained_fma  (production-order)   | vs reference_matmul                     |
    | 2 / 4 / 8   | reference_split_order(wo_split) (kernel-tree) | vs reference_chained_fma AND vs reference_matmul |

Cross-split drift (variant vs ``reference_chained_fma``) is FP32
reorder noise on K=6144 mixed-sign data — measured and reported,
not pass/fail.

NOTE: the chained K loop is intentionally a Python ``for`` over int
indices to preserve associativity. ``torch.einsum`` / ``torch.matmul``
both reorder, which is the entire reason for this file.
"""
from __future__ import annotations

import torch

# ---------------------------------------------------------------------
# FP4 E2M1 lookup table.
# ---------------------------------------------------------------------
# Mirrors ``_fp4_nibble_to_f32`` at
# ``vllm/v1/attention/backends/cute_paged/kernel.py:719-751``.
# Bit layout: [sign(1) | exp(2) | mant(1)].
# Unsigned table indexed by ``nibble & 0x07``:
#   {0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0}
# Sign bit at position 3 flips the sign.
_FP4_LUT: list[float] = [
    0.0,  0.5,  1.0,  1.5,  2.0,  3.0,  4.0,  6.0,    # +
    -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,   # -
]


# ---------------------------------------------------------------------
# Dequant primitives — shared by both references.
# ---------------------------------------------------------------------
def decode_fp4_weight(wo_weight_uint8: torch.Tensor) -> torch.Tensor:
    """Unpack NVFP4-packed uint8 weights to FP32.

    Input:  ``[hidden, K // 2]`` uint8.
    Output: ``[hidden, K]``     fp32.

    Per ``phase_e_kernel.py:4060-4078`` and ``microkernel.py:262-289``:
    even k goes in the LOW nibble, odd k in the HIGH nibble.
    """
    assert wo_weight_uint8.dtype == torch.uint8, (
        f"expected uint8, got {wo_weight_uint8.dtype}"
    )
    assert wo_weight_uint8.is_cuda, "weight must be on CUDA"
    H, KP = wo_weight_uint8.shape
    K = KP * 2
    device = wo_weight_uint8.device

    fp4_lut = torch.tensor(_FP4_LUT, dtype=torch.float32, device=device)

    lo_nib = (wo_weight_uint8 & 0x0F).long()              # [H, KP]
    hi_nib = ((wo_weight_uint8 >> 4) & 0x0F).long()       # [H, KP]
    w_lo = fp4_lut[lo_nib]                                # [H, KP]
    w_hi = fp4_lut[hi_nib]                                # [H, KP]

    w_f32 = torch.empty((H, K), dtype=torch.float32, device=device)
    w_f32[:, 0::2] = w_lo
    w_f32[:, 1::2] = w_hi
    return w_f32


def decode_fp8_scales_swizzled(
    wo_scales_uint8: torch.Tensor,
    hidden: int,
    K: int,
    num_k_tiles: int,
) -> torch.Tensor:
    """Unswizzle and decode FP8 E4M3 block scales to FP32.

    Args:
        wo_scales_uint8: ``[num_m_tiles, num_k_tiles, 32, 4, 4]`` uint8
            (FP8 E4M3 reinterpret-cast). May also be a ``float8_e4m3fn``
            tensor — both have ``element_size() == 1``.
        hidden: number of M-rows (= ``H`` of the weight matrix).
        K: number of K-cols (must be a multiple of 16).
        num_k_tiles: ``ceil(num_k_groups / 4)``, where
            ``num_k_groups = K // 16``.

    Output: ``[hidden, num_k_groups]`` fp32.

    Inverse of the swizzle layout at
    ``vllm/v1/attention/backends/cute_paged/kernel.py:773-803``::

        m_tile  = m >> 7
        outer_m = m & 31
        inner_m = (m >> 5) & 3
        k_tile  = k_group >> 2
        inner_k = k_group & 3
        sf_offset = (m_tile * num_k_tiles + k_tile) * 512
                  + outer_m * 16 + inner_m * 4 + inner_k
    """
    assert wo_scales_uint8.is_cuda, "scales must be on CUDA"
    assert wo_scales_uint8.element_size() == 1, (
        f"expected 1-byte dtype (uint8 / fp8_e4m3fn), got "
        f"{wo_scales_uint8.dtype}"
    )
    assert K % 16 == 0, f"K={K} must be a multiple of 16"
    device = wo_scales_uint8.device
    num_k_groups = K // 16

    # Flatten + reinterpret as fp8_e4m3fn -> fp32.
    sf_flat_bytes = wo_scales_uint8.contiguous().view(-1)
    if sf_flat_bytes.dtype == torch.uint8:
        sf_flat_fp32 = sf_flat_bytes.view(torch.float8_e4m3fn).float()
    else:
        # Already fp8_e4m3fn (or another 1-byte fp8).
        sf_flat_fp32 = sf_flat_bytes.float()

    # Build (m, k_grp) → flat_offset map via meshgrid.
    m_idx = torch.arange(hidden, device=device, dtype=torch.int64)
    k_idx = torch.arange(num_k_groups, device=device, dtype=torch.int64)
    M_g, K_g = torch.meshgrid(m_idx, k_idx, indexing="ij")
    m_tile = M_g >> 7
    outer_m = M_g & 31
    inner_m = (M_g >> 5) & 3
    k_tile = K_g >> 2
    inner_k = K_g & 3
    sf_offset = (m_tile * num_k_tiles + k_tile) * 512 \
        + outer_m * 16 + inner_m * 4 + inner_k

    assert sf_offset.max() < sf_flat_fp32.numel(), (
        f"max sf_offset {int(sf_offset.max())} >= sf buffer numel "
        f"{sf_flat_fp32.numel()}"
    )

    return sf_flat_fp32[sf_offset]                        # [H, num_k_groups]


def compute_weighted(
    weight_f32: torch.Tensor,
    scales_f32: torch.Tensor,
    wo_gs: torch.Tensor,
) -> torch.Tensor:
    """Combine into per-(hidden, K) ``weighted = w * sf * wo_gs``.

    Each scale covers 16 K elements (NVFP4 group size). The kernel
    re-loads ``sf`` once per K element (idempotently), so we replicate
    the scale across its 16 K columns and multiply elementwise.

    Args:
        weight_f32: ``[hidden, K]`` fp32 (output of decode_fp4_weight).
        scales_f32: ``[hidden, num_k_groups]`` fp32
            (output of decode_fp8_scales_swizzled).
        wo_gs: scalar fp32 tensor.
    """
    H, K = weight_f32.shape
    H2, num_k_groups = scales_f32.shape
    assert H == H2, f"hidden mismatch {H} vs {H2}"
    assert num_k_groups * 16 == K, (
        f"num_k_groups*16 ({num_k_groups * 16}) != K ({K})"
    )

    # Replicate each scale across its 16-element K group.
    k_col_to_grp = (
        torch.arange(K, device=weight_f32.device, dtype=torch.int64) >> 4
    )
    sf_full = scales_f32[:, k_col_to_grp]                 # [H, K]

    wo_gs_scalar = float(wo_gs.item())
    return weight_f32 * sf_full * wo_gs_scalar


# ---------------------------------------------------------------------
# Reference implementations.
# ---------------------------------------------------------------------
def reference_matmul(
    attn_output_bf16: torch.Tensor,
    wo_weight_uint8: torch.Tensor,
    wo_scales_uint8: torch.Tensor,
    wo_gs: torch.Tensor,
    hidden: int,
    K: int,
    num_k_tiles: int,
) -> torch.Tensor:
    """Diagnostic-only reference: vectorised cuBLAS matmul.

    Reduction order is cuBLAS-internal (typically tree reduction).
    NOT a pass/fail oracle for the kernel gate — the kernel uses
    chained FMA along K, this uses tree reduction. Use for diagnostic
    max abs / max rel reporting alongside ``reference_chained_fma``.

    Args:
        attn_output_bf16: ``[batch, K]`` bf16.
        wo_weight_uint8:  ``[hidden, K // 2]`` uint8.
        wo_scales_uint8:  ``[num_m_tiles, num_k_tiles, 32, 4, 4]`` uint8.
        wo_gs:            scalar fp32 tensor.
        hidden:           number of M-rows.
        K:                number of K-cols.
        num_k_tiles:      swizzle constant (== ``ceil(K/16/4)``).

    Returns: ``[batch, hidden]`` fp32.
    """
    assert attn_output_bf16.shape[1] == K
    assert wo_weight_uint8.shape == (hidden, K // 2)

    w_f32 = decode_fp4_weight(wo_weight_uint8)
    sf_f32 = decode_fp8_scales_swizzled(
        wo_scales_uint8, hidden, K, num_k_tiles
    )
    weighted = compute_weighted(w_f32, sf_f32, wo_gs)     # [H, K]

    a_f32 = attn_output_bf16.float()                      # [B, K]
    return a_f32 @ weighted.T                             # [B, H]


def reference_chained_fma(
    attn_output_bf16: torch.Tensor,
    wo_weight_uint8: torch.Tensor,
    wo_scales_uint8: torch.Tensor,
    wo_gs: torch.Tensor,
    hidden: int,
    K: int,
    num_kv_heads: int,
    num_k_tiles: int,
) -> torch.Tensor:
    """Authoritative reference: per-KV-head chained accumulation along K.

    Mirrors the kernel's wo_split=1 reduction order exactly:

        for kv_head in range(num_kv_heads):
            a = zeros[batch, hidden]
            for k_in_head in range(K // num_kv_heads):
                k_global = kv_head * (K // num_kv_heads) + k_in_head
                a = a + attn_fp32[:, k_global:k_global+1] * weighted[:, k_global]
            out += a

    The K loop is Python (preserves order); each step is an
    elementwise add (PyTorch does not reorder elementwise adds). The
    FP32 ``a + x*y`` ordering matches the kernel's
    ``a_oi = a_oi + w_dequant * attn_val`` chain. Whether that becomes
    a hardware FMA or two ops doesn't matter at rtol=1e-3 / atol=1e-4.

    For wo_split>1 the kernel splits each KV head's K-range across
    ``wo_split`` partial slots, then sums them in the gather pass. We
    intentionally do NOT model that split in the reference: the gate
    is "any partial-split pattern that still totals the chained sum
    within rtol=1e-3 / atol=1e-4 is fine". Reorder noise from summing
    partial slots will appear as a (small) drift vs this reference.
    If it exceeds the gate, that is exactly what we want to catch.

    Args:
        attn_output_bf16: ``[batch, K]`` bf16.
        wo_weight_uint8:  ``[hidden, K // 2]`` uint8.
        wo_scales_uint8:  ``[num_m_tiles, num_k_tiles, 32, 4, 4]`` uint8.
        wo_gs:            scalar fp32 tensor.
        hidden:           number of M-rows.
        K:                number of K-cols.
        num_kv_heads:     KV head count (kernel grid Y).
        num_k_tiles:      swizzle constant.

    Returns: ``[batch, hidden]`` fp32.
    """
    assert attn_output_bf16.shape[1] == K, (
        f"attn shape {attn_output_bf16.shape} does not match K={K}"
    )
    assert K % num_kv_heads == 0, (
        f"K={K} not divisible by num_kv_heads={num_kv_heads}"
    )
    device = attn_output_bf16.device
    batch = attn_output_bf16.shape[0]
    K_per_head = K // num_kv_heads

    # Dequant once.
    w_f32 = decode_fp4_weight(wo_weight_uint8)            # [H, K]
    sf_f32 = decode_fp8_scales_swizzled(
        wo_scales_uint8, hidden, K, num_k_tiles
    )                                                     # [H, K_groups]
    weighted = compute_weighted(w_f32, sf_f32, wo_gs)     # [H, K]

    a_f32 = attn_output_bf16.float()                      # [B, K]

    # Final accumulator (4-way sum across KV heads happens here).
    out = torch.zeros(
        (batch, hidden), dtype=torch.float32, device=device
    )

    for kv_head in range(num_kv_heads):
        # Per-KV-head chained accumulator. Mirrors a single CTA's
        # a0..a7 path collapsed into a [batch, hidden] tensor — the
        # ordering across hidden columns (a0..a7) is independent and
        # commutes; the only ordering that matters for FP32 drift is
        # along K, which we enforce with the Python for loop.
        a = torch.zeros(
            (batch, hidden), dtype=torch.float32, device=device
        )

        k_base = kv_head * K_per_head
        # Single linear K traversal — element-wise ``a = a + x * y``.
        for k_in_head in range(K_per_head):
            k_global = k_base + k_in_head
            # attn[:, k_global:k_global+1] shape (B, 1).
            # weighted[:, k_global] shape (H,) -> broadcast to (1, H).
            # Result (B, H), elementwise FMA: each (b, h) cell sees
            #   a[b,h] = a[b,h] + attn[b,k] * weighted[h,k]
            x = a_f32[:, k_global:k_global + 1]           # (B, 1)
            y = weighted[:, k_global].unsqueeze(0)        # (1, H)
            # Elementwise multiply broadcasts to (B, H).
            a = a + x * y

        # 4-way sum across KV heads. Elementwise add; order across
        # KV heads matches the gather pass (cta_i = 0,1,2,3 in
        # microkernel.py:399-406). Sub-tolerance reorder.
        out = out + a

    return out


def reference_split_order(
    attn_output_bf16: torch.Tensor,
    wo_weight_uint8: torch.Tensor,
    wo_scales_uint8: torch.Tensor,
    wo_gs: torch.Tensor,
    hidden: int,
    K: int,
    num_kv_heads: int,
    num_k_tiles: int,
    wo_split: int,
) -> torch.Tensor:
    """Per-variant reference mirroring the kernel's reduction tree.

    Implements the *exact* reduction order each ``wo_split`` variant
    produces. Each CTA owns a sub-range of one KV-head's K span,
    chained-FMA accumulates over that sub-range, writes its FP32
    partial into ``wo_output[seq_idx, wo_slot, :]``, and the gather
    pass left-folds ``total_wo_ctas`` slots in slot-id order.

    Partitioning (matches ``microkernel.py:203-205, 225-227``;
    mirrors ``phase_e_kernel.py:4103``)::

        total_wo_ctas = num_kv_heads * wo_split
        K_per_head = K // num_kv_heads

        for slot_id in range(total_wo_ctas):
            by = slot_id // wo_split
            bx = slot_id %  wo_split
            k_start_in_head = (K_per_head * bx)       // wo_split
            k_end_in_head   = (K_per_head * (bx + 1)) // wo_split
            k_start = by * K_per_head + k_start_in_head
            k_end   = by * K_per_head + k_end_in_head
            slot[slot_id] = chained_fma_K_range(
                weighted, attn, k_start, k_end
            )

    Gather (matches ``phase_e_kernel.py:4233-4242`` and
    ``microkernel.py:399-406``)::

        out = slot[0]
        for i in range(1, total_wo_ctas):
            out = out + slot[i]   # left-fold in slot_id order

    For ``wo_split == 1``: ``total_wo_ctas == num_kv_heads``, each
    slot covers one full KV-head's K range with no sub-split, and
    the slot-order left-fold is identical to ``reference_chained_fma``
    (KV-head order). This is verified by the Phase A sanity assert
    in the smoke harness.

    Args:
        attn_output_bf16: ``[batch, K]`` bf16.
        wo_weight_uint8:  ``[hidden, K // 2]`` uint8.
        wo_scales_uint8:  ``[num_m_tiles, num_k_tiles, 32, 4, 4]`` uint8.
        wo_gs:            scalar fp32 tensor.
        hidden:           number of M-rows.
        K:                number of K-cols.
        num_kv_heads:     KV head count (kernel grid Y).
        num_k_tiles:      swizzle constant.
        wo_split:         variant — int in {1, 2, 4, 8}.

    Returns: ``[batch, hidden]`` fp32.
    """
    if wo_split not in (1, 2, 4, 8):
        raise ValueError(
            f"wo_split must be in {{1,2,4,8}}, got {wo_split}"
        )
    assert attn_output_bf16.shape[1] == K, (
        f"attn shape {attn_output_bf16.shape} does not match K={K}"
    )
    assert K % num_kv_heads == 0, (
        f"K={K} not divisible by num_kv_heads={num_kv_heads}"
    )
    K_per_head = K // num_kv_heads
    device = attn_output_bf16.device
    batch = attn_output_bf16.shape[0]
    total_wo_ctas = num_kv_heads * wo_split

    # ----- Dequant once. Same pipeline as the other two refs. -----
    w_f32 = decode_fp4_weight(wo_weight_uint8)            # [H, K]
    sf_f32 = decode_fp8_scales_swizzled(
        wo_scales_uint8, hidden, K, num_k_tiles
    )                                                     # [H, K_groups]
    weighted = compute_weighted(w_f32, sf_f32, wo_gs)     # [H, K]
    a_f32 = attn_output_bf16.float()                      # [B, K]

    # ----- Per-CTA partial slots — chained K accumulation. -----
    # We materialise [total_wo_ctas, B, H] so the gather can left-fold
    # in slot_id order without recomputing partials.
    slots = torch.empty(
        (total_wo_ctas, batch, hidden),
        dtype=torch.float32, device=device,
    )

    for slot_id in range(total_wo_ctas):
        by = slot_id // wo_split
        bx = slot_id % wo_split
        # Match microkernel.py:225-226 / phase_e_kernel.py:4103.
        # Note Python integer-divide and PTX integer-divide agree on
        # non-negative operands.
        k_start_in_head = (K_per_head * bx) // wo_split
        k_end_in_head = (K_per_head * (bx + 1)) // wo_split
        k_start = by * K_per_head + k_start_in_head
        k_end = by * K_per_head + k_end_in_head

        a = torch.zeros(
            (batch, hidden), dtype=torch.float32, device=device
        )
        # Chained FMA over [k_start, k_end). Same Python-loop pattern
        # as reference_chained_fma — preserves elementwise add order.
        for k_global in range(k_start, k_end):
            x = a_f32[:, k_global:k_global + 1]           # (B, 1)
            y = weighted[:, k_global].unsqueeze(0)        # (1, H)
            a = a + x * y
        slots[slot_id] = a

    # ----- Gather: left-fold slots in slot_id order. -----
    # Matches microkernel.py:399-406 — `cta_i` increments from 0 to
    # total_wo_ctas-1, summing each slot into a per-thread accumulator.
    out = slots[0].clone()
    for i in range(1, total_wo_ctas):
        out = out + slots[i]

    return out
