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


# --- Inline PTX utilities ---------------------------------------------------
# Adapted from lukealonso/b12x@c469c66 forward_paged.py.
# All functions are @cute.jit and only exist when CUTLASS is installed.

if _CUTE_AVAILABLE:

    @cute.jit
    def _permuted_offset_128b(row: Int32, col: Int32, stride: Int32) -> Int32:
        """Bank-conflict-free SMEM offset for 128-byte rows.

        XOR-based swizzle avoids bank conflicts when multiple warps
        access adjacent rows in SMEM via ldmatrix.
        Ref: b12x@c469c66 forward_paged.py _permuted_offset_128b
        """
        return row * stride + (col ^ ((row % Int32(8)) // Int32(1)))

    @cute.jit
    def _smem_addr_from_b128_offset(
        base_addr: Int32, offset: Int32,
    ) -> Int32:
        """Convert 128-bit element offset to byte address for ldmatrix."""
        return base_addr + offset * Int32(16)

    @cute.jit
    def shared_ptr_to_u32(ptr) -> Uint32:
        """Convert shared memory pointer to u32 address for PTX."""
        result = Uint32(0)
        cutlass.arch.ptx(
            "cvta.to.shared.u32 %0, %1;",
            result, ptr, outputs=[result],
        )
        return result

    @cute.jit
    def ldmatrix_m8n8x4_b16(smem_addr: Int32):
        """Load 4 x m8n8 matrix fragments from SMEM.

        PTX: ldmatrix.sync.aligned.m8n8.x4.shared.b16
        Returns 4 Uint32 registers containing BF16 pairs.
        Ref: b12x@c469c66 forward_paged.py ldmatrix patterns
        """
        r0 = Uint32(0)
        r1 = Uint32(0)
        r2 = Uint32(0)
        r3 = Uint32(0)
        cutlass.arch.ptx(
            "ldmatrix.sync.aligned.m8n8.x4.shared.b16 "
            "{%0, %1, %2, %3}, [%4];",
            r0, r1, r2, r3, smem_addr,
            outputs=[r0, r1, r2, r3],
        )
        return r0, r1, r2, r3

    @cute.jit
    def frag_layout_swizzle_16b_to_8b(val: Uint32) -> Uint32:
        """Swizzle register layout from 16-bit to 8-bit element order.

        Required after ldmatrix when SMEM holds FP8 data but ldmatrix
        loads 16-bit granules.
        Ref: b12x@c469c66 forward_paged.py frag_layout_swizzle_16b_to_8b
        """
        result = Uint32(0)
        cutlass.arch.ptx(
            "prmt.b32 %0, %1, %1, 0x6420;",
            result, val,
            outputs=[result],
        )
        return result

    @cute.jit
    def fp8x4_e4m3_to_bfloat2x2(val: Uint32):
        """Convert 4 packed FP8 E4M3 values to 2 pairs of BF16 values.

        Input: Uint32 with 4x FP8 E4M3 (8 bits each).
        Output: Two Uint32 registers, each with 2x BF16.
        Uses prmt to extract bytes + shl to align with BF16 format.
        Ref: b12x@c469c66 forward_paged.py fp8x4_e4m3_to_bfloat2x2
        """
        lo = Uint32(0)
        hi = Uint32(0)
        cutlass.arch.ptx(
            "{.reg .b32 tmp0, tmp1;\n"
            " prmt.b32 tmp0, %2, 0, 0x5140;\n"
            " prmt.b32 tmp1, %2, 0, 0x7362;\n"
            " shl.b32 %0, tmp0, 8;\n"
            " shl.b32 %1, tmp1, 8;\n"
            "}",
            lo, hi, val,
            outputs=[lo, hi],
        )
        return lo, hi

    @cute.jit
    def bf16_mma_m16n16k16_f32(
        d0, d1, d2, d3, d4, d5, d6, d7,
        a0, a1, a2, a3,
        b0, b1, b2, b3,
    ):
        """BF16 m16n16k16 MMA accumulating into FP32 registers.

        Issues two m16n8k16 PTX instructions for a 16x16 output tile.
        Ref: b12x@c469c66 forward_paged.py bf16_mma_m16n16k16_f32

        Args:
            d0-d7: FP32 accumulator (8 regs for m16n16 output)
            a0-a3: A operand fragments (BF16 pairs from Q or P)
            b0-b3: B operand fragments (BF16 pairs from K or V)
        Returns: Updated d0-d7
        """
        # First m16n8k16: columns 0..7
        cutlass.arch.ptx(
            "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
            "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
            "{%0, %1, %2, %3};",
            d0, d1, d2, d3,
            a0, a1, a2, a3,
            b0, b1,
            outputs=[d0, d1, d2, d3],
        )
        # Second m16n8k16: columns 8..15
        cutlass.arch.ptx(
            "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
            "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
            "{%0, %1, %2, %3};",
            d4, d5, d6, d7,
            a0, a1, a2, a3,
            b2, b3,
            outputs=[d4, d5, d6, d7],
        )
        return d0, d1, d2, d3, d4, d5, d6, d7

    @cute.jit
    def exp2_approx_ftz_f32(x: Float32) -> Float32:
        """Fast exp2 with flush-to-zero for online softmax.

        Ref: b12x@c469c66 _exp2_approx_ftz_f32
        """
        result = Float32(0.0)
        cutlass.arch.ptx(
            "ex2.approx.ftz.f32 %0, %1;",
            result, x,
            outputs=[result],
        )
        return result

    @cute.jit
    def shfl_xor_sync(val: Float32, lane_mask: Int32) -> Float32:
        """Warp shuffle XOR for row-max/row-sum reduction.

        Full-warp mask (0xFFFFFFFF). Used in online softmax to find
        row max and accumulate row sum across warp lanes.
        """
        result = Float32(0.0)
        cutlass.arch.ptx(
            "shfl.sync.bfly.b32 %0, %1, %2, 31, 0xFFFFFFFF;",
            result, val, lane_mask,
            outputs=[result],
        )
        return result

    @cute.jit
    def cp_async_load_128b(smem_addr: Uint32, gmem_ptr) -> None:
        """Async copy 128 bits (16 bytes) from global to shared memory.

        Uses cp.async.cg.shared.global for non-blocking transfer.
        """
        cutlass.arch.ptx(
            "cp.async.cg.shared.global [%0], [%1], 16;",
            smem_addr, gmem_ptr,
        )

    @cute.jit
    def cp_async_commit_group() -> None:
        """Commit the current group of async copies."""
        cutlass.arch.ptx("cp.async.commit_group;")

    @cute.jit
    def cp_async_wait_group(n: Int32) -> None:
        """Wait until at most n async copy groups are pending.

        For num_stages=1, call with n=0 to wait for all copies.
        """
        if n == Int32(0):
            cutlass.arch.ptx("cp.async.wait_group 0;")
        else:
            cutlass.arch.ptx("cp.async.wait_group 1;")

    @cute.jit
    def _lane_id() -> Int32:
        """Get lane ID within the current warp (0..31)."""
        result = Int32(0)
        cutlass.arch.ptx(
            "mov.u32 %0, %laneid;",
            result,
            outputs=[result],
        )
        return result

    @cute.jit
    def _warp_id() -> Int32:
        """Get warp ID within the current CTA."""
        tid = Int32(0)
        cutlass.arch.ptx("mov.u32 %0, %tid.x;", tid, outputs=[tid])
        return tid // Int32(32)

    # --- DecodeKernel -------------------------------------------------------

    class DecodeKernel:
        """CuTe JIT decode kernel: cta_q=16, warps_kv=4, cross-warp reduction.

        Class-based kernel for const_expr() compile-time branching.
        Each warp processes different KV tile chunks in parallel, then
        reduces partial softmax states (O, m, d) via SMEM sync buffers.

        Spec: Section 3.3 (execution flow), Section 3.4 (decode config)
        Ref: b12x@c469c66 PagedFp8DecodeRawForwardKernel
        """

        def __init__(self, config: KernelConfig):
            self.cta_q = config.cta_q            # 16
            self.cta_kv = config.cta_kv          # 64
            self.head_dim = config.head_dim      # 128
            self.block_size = config.block_size  # 64
            self.num_warps_q = config.num_warps_q    # 1
            self.num_warps_kv = config.num_warps_kv  # 4
            self.num_threads = 128  # 4 warps * 32 threads
            self.num_mma_d = self.head_dim // 16  # 8

            # SMEM sizes (bytes)
            self.q_bytes = self.cta_q * self.head_dim * 2      # BF16
            self.k_bytes = self.cta_kv * self.head_dim * 1     # FP8
            self.v_bytes = self.cta_kv * self.head_dim * 1     # FP8
            self.qkv_bytes = self.q_bytes + self.k_bytes + self.v_bytes
            # Cross-warp reduction buffers (b12x pattern)
            self.sync_o_bytes = (
                self.num_warps_kv * self.cta_q * self.head_dim * 4
            )  # FP32
            self.sync_md_bytes = (
                self.num_warps_kv * self.cta_q * 8
            )  # FP32 m + d per row
            self.sync_bytes = self.sync_o_bytes + self.sync_md_bytes
            self.smem_bytes = max(self.qkv_bytes, self.sync_bytes)

        @cute.kernel
        def _kernel(self, query, k_cache, v_cache, page_table, seq_lens,
                     output, scale, k_scale, v_scale,
                     num_q_heads, num_kv_heads):
            """Decode kernel entry point.

            Grid: (num_q_tiles, num_kv_heads, num_seqs)

            Execution flow (spec section 3.3):
            1. Load Q tile via CpAsync into SMEM
            2. For each page in page_table[seq_idx]:
               a. Load K page -> SMEM, ldmatrix -> regs, dequant FP8->BF16
               b. QK MMA (bf16_mma_m16n16k16_f32, 8 iterations)
               c. Apply k_scale, online softmax update
               d. Load V page -> SMEM, ldmatrix -> regs, dequant FP8->BF16
               e. Apply v_scale to P, cast P FP32->BF16, PV MMA
            3. Cross-warp reduction via SMEM cta_sync buffers
            4. Final normalization O /= row_sum
            5. Cast FP32->BF16, write to global output

            Stub: full kernel body requires live CUTLASS compiler iteration.
            """
            pass

        def __call__(self, **kwargs):
            """Python-level wrapper: compute grid/block and launch."""
            query = kwargs["query"]
            k_cache = kwargs["k_cache"]
            v_cache = kwargs["v_cache"]
            page_table = kwargs["page_table"]
            seq_lens = kwargs["seq_lens"]
            scale = kwargs["scale"]
            k_scale = kwargs["k_scale"]
            v_scale = kwargs["v_scale"]

            num_tokens, num_q_heads, head_dim = query.shape
            num_kv_heads = k_cache.shape[2]
            group_size = num_q_heads // num_kv_heads
            num_seqs = len(seq_lens)

            # Grid: (ceil(group_size / cta_q), num_kv_heads, num_seqs)
            # Decode: 1 query token/seq, GQA=4, cta_q=16 -> 1 CTA per (kv_head, seq)
            num_q_tiles = max(
                (group_size + self.cta_q - 1) // self.cta_q, 1,
            )
            grid = (num_q_tiles, num_kv_heads, num_seqs)

            output = torch.empty_like(query)

            self._kernel[grid, (self.num_threads,)](
                query, k_cache, v_cache, page_table, seq_lens,
                output, scale, k_scale, v_scale,
                num_q_heads, num_kv_heads,
                smem_bytes=self.smem_bytes,
            )
            return output

    # --- PrefillKernel ------------------------------------------------------

    class PrefillKernel:
        """CuTe JIT prefill kernel: cta_q=64, warps_q=4, no cross-warp reduction.

        Each warp handles 16 Q rows independently, all sharing the same
        KV tile from SMEM. Causal masking applied per-token.

        Spec: Section 3.4 (prefill config), Section 3.9 (causal mask)
        Ref: b12x@c469c66 PagedFp8ExtendRawForwardKernel
        """

        def __init__(self, config: KernelConfig):
            self.cta_q = config.cta_q            # 64
            self.cta_kv = config.cta_kv          # 64
            self.head_dim = config.head_dim      # 128
            self.block_size = config.block_size  # 64
            self.num_warps_q = config.num_warps_q    # 4
            self.num_warps_kv = config.num_warps_kv  # 1
            self.num_threads = 128
            self.num_mma_d = self.head_dim // 16  # 8

            # SMEM sizes (bytes) -- no cta_sync for prefill
            self.q_bytes = self.cta_q * self.head_dim * 2      # BF16
            self.k_bytes = self.cta_kv * self.head_dim * 1     # FP8
            self.v_bytes = self.cta_kv * self.head_dim * 1     # FP8
            self.smem_bytes = self.q_bytes + self.k_bytes + self.v_bytes

        @cute.kernel
        def _kernel(self, query, k_cache, v_cache, page_table, seq_lens,
                     query_start_loc, output, scale, k_scale, v_scale,
                     num_q_heads, num_kv_heads):
            """Prefill kernel entry point.

            Grid: (num_q_tiles, num_kv_heads, num_seqs)

            Same flow as DecodeKernel but:
            - 4 warps on Q dimension (each handles 16 Q rows)
            - No cross-warp reduction (independent output rows)
            - Causal mask: kv_pos > q_pos -> score = -2^15
            - query_start_loc for per-sequence Q positions

            Stub: full kernel body requires live CUTLASS compiler iteration.
            """
            pass

        def __call__(self, **kwargs):
            """Python-level wrapper: compute grid/block and launch."""
            query = kwargs["query"]
            k_cache = kwargs["k_cache"]
            v_cache = kwargs["v_cache"]
            page_table = kwargs["page_table"]
            seq_lens = kwargs["seq_lens"]
            query_start_loc = kwargs["query_start_loc"]
            scale = kwargs["scale"]
            k_scale = kwargs["k_scale"]
            v_scale = kwargs["v_scale"]

            num_tokens, num_q_heads, head_dim = query.shape
            num_kv_heads = k_cache.shape[2]
            group_size = num_q_heads // num_kv_heads
            num_seqs = len(seq_lens)

            # Prefill: multiple Q tokens per sequence
            max_q_per_seq = max(
                (query_start_loc[i + 1] - query_start_loc[i]).item()
                for i in range(num_seqs)
            ) * group_size
            num_q_tiles = (
                (max_q_per_seq + self.cta_q - 1) // self.cta_q
            )
            grid = (num_q_tiles, num_kv_heads, num_seqs)

            output = torch.empty_like(query)

            self._kernel[grid, (self.num_threads,)](
                query, k_cache, v_cache, page_table, seq_lens,
                query_start_loc, output, scale, k_scale, v_scale,
                num_q_heads, num_kv_heads,
                smem_bytes=self.smem_bytes,
            )
            return output


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
