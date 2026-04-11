"""CuTe DSL paged attention kernel for SM120/SM121 (GB10).

Replaces the PyTorch prototype with JIT-compiled PTX kernels using
BF16 m16n8k16 MMA for both QK and PV passes. Path B: K FP8->BF16
dequant via fp8x4_e4m3_to_bfloat2x2.

Reference: lukealonso/b12x@c469c66 (default FP8 KV path)
Spec: docs/superpowers/specs/2026-04-11-cute-dsl-kernel-replacement-design.md
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from functools import lru_cache

import torch

logger = logging.getLogger(__name__)

# --- CuTe DSL import guard -------------------------------------------------
# If CUTLASS is not installed (dev environment without SM120), fall back to
# the PyTorch reference. Runtime compilation failures are NOT caught -- they
# raise RuntimeError immediately.
_CUTE_AVAILABLE = False
try:
    import cutlass
    from cutlass import cute
    from cutlass.cute.typing import Float32, Int32, Uint32
    _CUTE_AVAILABLE = True
except ImportError:
    logger.warning(
        "CuTe DSL not available (CUTLASS not installed). "
        "paged_attention_forward will use PyTorch reference fallback."
    )


# --- Kernel config ----------------------------------------------------------

@dataclass(frozen=True)
class KernelConfig:
    """Compile-time kernel configuration. Frozen for use as lru_cache key."""

    cta_q: int          # Q tile rows per CTA (16=decode, 64=prefill)
    cta_kv: int         # KV tile rows per CTA (always 64 = page_size)
    head_dim: int       # Head dimension (128)
    block_size: int     # Page size in tokens (64)
    num_warps_q: int    # Warps along Q dimension
    num_warps_kv: int   # Warps along KV dimension


DECODE_CONFIG = KernelConfig(
    cta_q=16, cta_kv=64, head_dim=128, block_size=64,
    num_warps_q=1, num_warps_kv=4,
)

PREFILL_CONFIG = KernelConfig(
    cta_q=64, cta_kv=64, head_dim=128, block_size=64,
    num_warps_q=4, num_warps_kv=1,
)


# --- Inline PTX utilities (Task 8) -----------------------------------------
# Filled in by Task 8. When _CUTE_AVAILABLE is False, these are never called.

# --- DecodeKernel class (Task 9) -------------------------------------------
# Filled in by Task 9.

# --- PrefillKernel class (Task 10) -----------------------------------------
# Filled in by Task 10.


# --- Compilation cache ------------------------------------------------------

@lru_cache(maxsize=4)
def _get_compiled_kernel(config: KernelConfig):
    """Compile and cache a kernel for the given config.

    Uses lru_cache on the frozen KernelConfig dataclass for deduplication.
    Compilation may be called redundantly from concurrent threads on first
    miss -- this is harmless (same as disk cache behavior).
    """
    if not _CUTE_AVAILABLE:
        raise RuntimeError(
            "CuTe DSL kernel compilation requested but CUTLASS is not installed"
        )
    if config.cta_q <= 16:
        kernel = DecodeKernel(config)
    else:
        kernel = PrefillKernel(config)
    return kernel


# --- Public entry point -----------------------------------------------------

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
    """Paged attention forward -- CuTe JIT kernel or PyTorch fallback.

    Returns: [num_tokens, num_q_heads, head_dim] BF16
    """
    if not _CUTE_AVAILABLE:
        # Fallback to PyTorch reference (dev environments without CUTLASS)
        from tests.nvllm.attention.reference import reference_paged_attention
        return reference_paged_attention(
            query, k_cache, v_cache, page_table, seq_lens,
            scale=scale, k_scale=k_scale, v_scale=v_scale,
            page_size=page_size, query_start_loc=query_start_loc,
        )

    # Select config based on query token count
    num_tokens = query.shape[0]
    num_seqs = len(seq_lens)
    is_decode = (num_tokens == num_seqs) and (
        query_start_loc is None
        or query_start_loc[-1].item() - query_start_loc[-2].item() == 1
    )

    config = DECODE_CONFIG if is_decode else PREFILL_CONFIG
    kernel = _get_compiled_kernel(config)

    return kernel(
        query=query,
        k_cache=k_cache,
        v_cache=v_cache,
        page_table=page_table,
        seq_lens=seq_lens,
        scale=scale,
        k_scale=k_scale,
        v_scale=v_scale,
        page_size=page_size,
        query_start_loc=query_start_loc,
    )
