# Copyright 2026 Navi Ai Labs
# SPDX-License-Identifier: Apache-2.0
"""CuTe DSL paged attention kernel for SM120/SM121 (GB10).

Replaces the PyTorch prototype with JIT-compiled PTX kernels using
BF16 m16n8k16 MMA for both QK and PV passes. Path B: K FP8->BF16
dequant via fp8x4_e4m3_to_bfloat2x2.

Reference: lukealonso/b12x@c469c66 (default FP8 KV path)
Spec: docs/superpowers/specs/2026-04-11-cute-dsl-kernel-replacement-design.md
"""
from __future__ import annotations

import logging
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
    from cutlass._mlir import ir as _mlir_ir
    from cutlass._mlir.dialects import llvm as _llvm_dialect
    from cutlass.cute.typing import BFloat16, Float32, Int32, Int64, Uint32, Uint64
    from cutlass.cutlass_dsl import T, dsl_user_op
    import cuda.bindings.driver as _cuda_driver
    _CUTE_AVAILABLE = True
except ImportError:
    logger.warning(
        "CuTe DSL not available (CUTLASS not installed). "
        "paged_attention_forward will use PyTorch reference fallback."
    )

# Set False to disable CuTe kernels and use PyTorch reference fallback.
_KERNELS_IMPLEMENTED = True


# --- Kernel config ----------------------------------------------------------

@dataclass(frozen=True)
class KernelConfig:
    """Compile-time kernel configuration. Frozen for use as lru_cache key."""

    cta_q: int          # Q tile rows per CTA (16=decode, 64=prefill)
    cta_kv: int         # KV tile rows per CTA (always 64 = page_size)
    head_dim: int       # Head dimension (256 for Qwen3.5-27B)
    block_size: int     # Page size in tokens (64)
    num_warps_q: int    # Warps along Q dimension
    num_warps_kv: int   # Warps along KV dimension


DECODE_CONFIG = KernelConfig(
    cta_q=16, cta_kv=64, head_dim=256, block_size=64,
    num_warps_q=1, num_warps_kv=4,
)

PREFILL_CONFIG = KernelConfig(
    cta_q=64, cta_kv=64, head_dim=256, block_size=64,
    num_warps_q=4, num_warps_kv=1,
)


# --- Inline PTX utilities ---------------------------------------------------
# Adapted from lukealonso/b12x@c469c66 forward_paged.py.
# PTX helpers use @dsl_user_op + llvm.inline_asm (CUTLASS 4.4.2 API).
# Pure-arithmetic helpers remain @cute.jit. All only exist when CUTLASS is
# installed.

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
        base_addr: Int64, offset: Int32,
    ) -> Int64:
        """Convert 128-bit element offset to byte address for ldmatrix."""
        return base_addr + offset * Int64(16)

    @dsl_user_op
    def shared_ptr_to_i64(ptr, *, loc=None, ip=None) -> Int64:
        """Convert shared memory pointer to i64 address for PTX (64-bit)."""
        ptr_ir = ptr.toint(loc=loc, ip=ip).ir_value(loc=loc, ip=ip)
        result_ir = _llvm_dialect.inline_asm(
            T.i64(),
            [ptr_ir],
            "cvta.to.shared.u64 $0, $1;",
            "=l,l",
            has_side_effects=True,
            asm_dialect=0,
            loc=loc,
            ip=ip,
        )
        return Int64(result_ir)

    @dsl_user_op
    def ldmatrix_m8n8x4_b16(smem_addr: Int64, *, loc=None, ip=None):
        """Load 4 x m8n8 matrix fragments from SMEM.

        PTX: ldmatrix.sync.aligned.m8n8.x4.shared.b16
        Returns 4 Uint32 registers containing BF16 pairs.
        Ref: b12x@c469c66 forward_paged.py ldmatrix patterns
        """
        addr_ir = Int64(smem_addr).ir_value(loc=loc, ip=ip)
        result_struct = _llvm_dialect.inline_asm(
            _mlir_ir.Type.parse("!llvm.struct<(i32, i32, i32, i32)>"),
            [addr_ir],
            "ldmatrix.sync.aligned.m8n8.x4.shared.b16 "
            "{$0, $1, $2, $3}, [$4];",
            "=r,=r,=r,=r,l",
            has_side_effects=True,
            asm_dialect=0,
            loc=loc,
            ip=ip,
        )
        r0 = Uint32(_llvm_dialect.extractvalue(
            T.i32(), result_struct, [0], loc=loc, ip=ip))
        r1 = Uint32(_llvm_dialect.extractvalue(
            T.i32(), result_struct, [1], loc=loc, ip=ip))
        r2 = Uint32(_llvm_dialect.extractvalue(
            T.i32(), result_struct, [2], loc=loc, ip=ip))
        r3 = Uint32(_llvm_dialect.extractvalue(
            T.i32(), result_struct, [3], loc=loc, ip=ip))
        return r0, r1, r2, r3

    @dsl_user_op
    def frag_layout_swizzle_16b_to_8b(val: Uint32, *, loc=None, ip=None
                                       ) -> Uint32:
        """Swizzle register layout from 16-bit to 8-bit element order.

        Required after ldmatrix when SMEM holds FP8 data but ldmatrix
        loads 16-bit granules.
        Ref: b12x@c469c66 forward_paged.py frag_layout_swizzle_16b_to_8b
        """
        val_ir = Uint32(val).ir_value(loc=loc, ip=ip)
        result_ir = _llvm_dialect.inline_asm(
            T.i32(),
            [val_ir],
            "prmt.b32 $0, $1, $1, 0x6420;",
            "=r,r",
            has_side_effects=True,
            asm_dialect=0,
            loc=loc,
            ip=ip,
        )
        return Uint32(result_ir)

    @dsl_user_op
    def fp8x4_e4m3_to_bfloat2x2(val: Uint32, *, loc=None, ip=None):
        """Convert 4 packed FP8 E4M3 values to 2 pairs of BF16 values.

        Input: Uint32 with 4x FP8 E4M3 (8 bits each), bytes [e0,e1,e2,e3].
        Output: (lo, hi) — two Uint32 registers, each with 2x BF16.
          lo = [bf16(e0), bf16(e1)], hi = [bf16(e2), bf16(e3)]

        Uses hardware conversion chain (SM89+):
          1. Split u32 into two u16 halves (each with 2 packed E4M3)
          2. cvt.rn.f16x2.e4m3x2: convert 2 packed E4M3 → 2 packed FP16
          3. Extract individual FP16 values
          4. cvt.f32.f16: convert each FP16 → FP32
          5. cvt.rn.bf16x2.f32: pack 2 FP32 → BF16x2

        Ref: replaces broken prmt+shl approach (only correct for E5M2).
        """
        val_ir = Uint32(val).ir_value(loc=loc, ip=ip)
        result_struct = _llvm_dialect.inline_asm(
            _mlir_ir.Type.parse("!llvm.struct<(i32, i32)>"),
            [val_ir],
            "{\n"
            "  .reg .b16 lo16, hi16;\n"
            "  .reg .b32 f16_lo, f16_hi;\n"
            "  .reg .f16 h0, h1, h2, h3;\n"
            "  .reg .f32 f0, f1, f2, f3;\n"
            "  // Split into two 16-bit halves: lo16=[e0,e1], hi16=[e2,e3]\n"
            "  mov.b32 {lo16, hi16}, $2;\n"
            "  // E4M3x2 -> F16x2 (hardware conversion, SM89+)\n"
            "  cvt.rn.f16x2.e4m3x2 f16_lo, lo16;\n"
            "  cvt.rn.f16x2.e4m3x2 f16_hi, hi16;\n"
            "  // Extract individual FP16 values\n"
            "  mov.b32 {h0, h1}, f16_lo;\n"
            "  mov.b32 {h2, h3}, f16_hi;\n"
            "  // FP16 -> FP32\n"
            "  cvt.f32.f16 f0, h0;\n"
            "  cvt.f32.f16 f1, h1;\n"
            "  cvt.f32.f16 f2, h2;\n"
            "  cvt.f32.f16 f3, h3;\n"
            "  // Pack FP32 pairs -> BF16x2\n"
            "  cvt.rn.bf16x2.f32 $0, f1, f0;\n"
            "  cvt.rn.bf16x2.f32 $1, f3, f2;\n"
            "}",
            "=r,=r,r",
            has_side_effects=True,
            asm_dialect=0,
            loc=loc,
            ip=ip,
        )
        lo = Uint32(_llvm_dialect.extractvalue(
            T.i32(), result_struct, [0], loc=loc, ip=ip))
        hi = Uint32(_llvm_dialect.extractvalue(
            T.i32(), result_struct, [1], loc=loc, ip=ip))
        return lo, hi

    @dsl_user_op
    def _mma_m16n8k16_f32(
        d0: Float32, d1: Float32, d2: Float32, d3: Float32,
        a0: Uint32, a1: Uint32, a2: Uint32, a3: Uint32,
        b0: Uint32, b1: Uint32,
        *, loc=None, ip=None,
    ):
        """Single m16n8k16 MMA: accumulate BF16 A*B into FP32 D.

        PTX fragment layout for A (m16×k16, row-major, bf16):
          a0 = {A[group,   sub*2..+1]}       rows 0-7,  K cols 0-7
          a1 = {A[group+8, sub*2..+1]}       rows 8-15, K cols 0-7
          a2 = {A[group,   sub*2+8..+9]}     rows 0-7,  K cols 8-15
          a3 = {A[group+8, sub*2+8..+9]}     rows 8-15, K cols 8-15

        Callers use the logical convention (row-major then k-major):
          a0 = rows 0-7  k_lo,  a1 = rows 0-7  k_hi,
          a2 = rows 8-15 k_lo,  a3 = rows 8-15 k_hi.

        This function swaps a1↔a2 to match the PTX hardware order
        (row-interleaved: a1=rows 8-15 k_lo, a2=rows 0-7 k_hi).
        """
        # NOTE: a2 and a1 are SWAPPED for $5/$6 to match PTX spec.
        # PTX expects {a0, a_row_hi_k_lo, a_row_lo_k_hi, a3} but
        # callers pass {a0, a_row_lo_k_hi, a_row_hi_k_lo, a3}.
        operands_ir = [
            Uint32(a0).ir_value(loc=loc, ip=ip),
            Uint32(a2).ir_value(loc=loc, ip=ip),  # $5 = row_hi k_lo
            Uint32(a1).ir_value(loc=loc, ip=ip),  # $6 = row_lo k_hi
            Uint32(a3).ir_value(loc=loc, ip=ip),
            Uint32(b0).ir_value(loc=loc, ip=ip),
            Uint32(b1).ir_value(loc=loc, ip=ip),
            Float32(d0).ir_value(loc=loc, ip=ip),
            Float32(d1).ir_value(loc=loc, ip=ip),
            Float32(d2).ir_value(loc=loc, ip=ip),
            Float32(d3).ir_value(loc=loc, ip=ip),
        ]
        result_struct = _llvm_dialect.inline_asm(
            _mlir_ir.Type.parse("!llvm.struct<(f32, f32, f32, f32)>"),
            operands_ir,
            "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
            "{$0, $1, $2, $3}, {$4, $5, $6, $7}, {$8, $9}, "
            "{$10, $11, $12, $13};",
            "=f,=f,=f,=f,r,r,r,r,r,r,f,f,f,f",
            has_side_effects=True,
            asm_dialect=0,
            loc=loc,
            ip=ip,
        )
        r0 = Float32(_llvm_dialect.extractvalue(
            T.f32(), result_struct, [0], loc=loc, ip=ip))
        r1 = Float32(_llvm_dialect.extractvalue(
            T.f32(), result_struct, [1], loc=loc, ip=ip))
        r2 = Float32(_llvm_dialect.extractvalue(
            T.f32(), result_struct, [2], loc=loc, ip=ip))
        r3 = Float32(_llvm_dialect.extractvalue(
            T.f32(), result_struct, [3], loc=loc, ip=ip))
        return r0, r1, r2, r3

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
        d0, d1, d2, d3 = _mma_m16n8k16_f32(
            d0, d1, d2, d3, a0, a1, a2, a3, b0, b1)
        # Second m16n8k16: columns 8..15
        d4, d5, d6, d7 = _mma_m16n8k16_f32(
            d4, d5, d6, d7, a0, a1, a2, a3, b2, b3)
        return d0, d1, d2, d3, d4, d5, d6, d7

    @cute.jit
    def exp2_approx_ftz_f32(x: Float32) -> Float32:
        """Fast exp2 with flush-to-zero for online softmax.

        Uses cute.arch.exp2 built-in (approx ftz semantics).
        Ref: b12x@c469c66 _exp2_approx_ftz_f32
        """
        return cute.arch.exp2(x)

    @cute.jit
    def shfl_xor_sync(val: Float32, lane_mask: Int32) -> Float32:
        """Warp shuffle XOR for row-max/row-sum reduction.

        Uses cute.arch.shuffle_sync_bfly built-in (full-warp mask).
        Used in online softmax to find row max and accumulate row sum
        across warp lanes.
        """
        return cute.arch.shuffle_sync_bfly(val, lane_mask)

    @dsl_user_op
    def cp_async_load_128b(smem_addr: Int64, gmem_ptr, *,
                           loc=None, ip=None) -> None:
        """Async copy 128 bits (16 bytes) from global to shared memory.

        Uses cp.async.cg.shared.global for non-blocking transfer.
        """
        addr_ir = Int64(smem_addr).ir_value(loc=loc, ip=ip)
        ptr_ir = gmem_ptr.toint(loc=loc, ip=ip).ir_value(loc=loc, ip=ip)
        _llvm_dialect.inline_asm(
            None,
            [addr_ir, ptr_ir],
            "cp.async.cg.shared.global [$0], [$1], 16;",
            "l,l",
            has_side_effects=True,
            asm_dialect=0,
            loc=loc,
            ip=ip,
        )

    @cute.jit
    def cp_async_commit_group() -> None:
        """Commit the current group of async copies.

        Uses cute.arch.cp_async_commit_group built-in.
        """
        cute.arch.cp_async_commit_group()

    @cute.jit
    def cp_async_wait_group(n: Int32) -> None:
        """Wait until at most n async copy groups are pending.

        Uses cute.arch.cp_async_wait_group built-in.
        For num_stages=1, call with n=0 to wait for all copies.
        """
        cute.arch.cp_async_wait_group(n)

    @cute.jit
    def _lane_id() -> Int32:
        """Get lane ID within the current warp (0..31).

        Thin wrapper around cute.arch.lane_idx() for backward compat.
        """
        return cute.arch.lane_idx()

    @cute.jit
    def _warp_id() -> Int32:
        """Get warp ID within the current CTA.

        Thin wrapper around cute.arch.warp_idx() for backward compat.
        """
        return cute.arch.warp_idx()

    # --- Additional PTX utilities for decode kernel -------------------------

    @dsl_user_op
    def _ld_shared_b32(addr: Int64, *, loc=None, ip=None) -> Uint32:
        """Load 32 bits from shared memory (requires 4-byte alignment)."""
        addr_ir = Int64(addr).ir_value(loc=loc, ip=ip)
        result_ir = _llvm_dialect.inline_asm(
            T.i32(),
            [addr_ir],
            "ld.shared.b32 $0, [$1];",
            "=r,l",
            has_side_effects=True,
            asm_dialect=0,
            loc=loc,
            ip=ip,
        )
        return Uint32(result_ir)

    @dsl_user_op
    def _ld_shared_b16(addr: Int64, *, loc=None, ip=None) -> Uint32:
        """Load 16 bits from shared memory, zero-extended to Uint32.

        Only requires 2-byte alignment — use for FP8 KV loads where
        sub-column offsets may not be 4-byte aligned.
        """
        addr_ir = Int64(addr).ir_value(loc=loc, ip=ip)
        result_ir = _llvm_dialect.inline_asm(
            T.i32(),
            [addr_ir],
            "{.reg .b16 t; ld.shared.b16 t, [$1]; cvt.u32.u16 $0, t;}",
            "=r,l",
            has_side_effects=True,
            asm_dialect=0,
            loc=loc,
            ip=ip,
        )
        return Uint32(result_ir)

    @dsl_user_op
    def _st_shared_b16_from_u32(addr: Int64, val: Uint32, *,
                                loc=None, ip=None) -> None:
        """Store low 16 bits of Uint32 to shared memory.

        Uses explicit .b16 intermediate register to guarantee exactly
        16 bits are written — SM121 relaxed type checking with .b32
        source may write 32 bits, corrupting adjacent SMEM values.
        """
        addr_ir = Int64(addr).ir_value(loc=loc, ip=ip)
        val_ir = Uint32(val).ir_value(loc=loc, ip=ip)
        _llvm_dialect.inline_asm(
            None,
            [addr_ir, val_ir],
            "{.reg .b16 t; cvt.u16.u32 t, $1; st.shared.b16 [$0], t;}",
            "l,r",
            has_side_effects=True,
            asm_dialect=0,
            loc=loc,
            ip=ip,
        )

    @dsl_user_op
    def _st_shared_b32(addr: Int64, val: Uint32, *,
                       loc=None, ip=None) -> None:
        """Store 32 bits to shared memory."""
        addr_ir = Int64(addr).ir_value(loc=loc, ip=ip)
        val_ir = Uint32(val).ir_value(loc=loc, ip=ip)
        _llvm_dialect.inline_asm(
            None,
            [addr_ir, val_ir],
            "st.shared.b32 [$0], $1;",
            "l,r",
            has_side_effects=True,
            asm_dialect=0,
            loc=loc,
            ip=ip,
        )

    @dsl_user_op
    def _st_shared_f32(addr: Int64, val: Float32, *,
                       loc=None, ip=None) -> None:
        """Store FP32 to shared memory."""
        addr_ir = Int64(addr).ir_value(loc=loc, ip=ip)
        val_ir = Float32(val).ir_value(loc=loc, ip=ip)
        _llvm_dialect.inline_asm(
            None,
            [addr_ir, val_ir],
            "st.shared.f32 [$0], $1;",
            "l,f",
            has_side_effects=True,
            asm_dialect=0,
            loc=loc,
            ip=ip,
        )

    @dsl_user_op
    def _ld_shared_f32(addr: Int64, *, loc=None, ip=None) -> Float32:
        """Load FP32 from shared memory."""
        addr_ir = Int64(addr).ir_value(loc=loc, ip=ip)
        result_ir = _llvm_dialect.inline_asm(
            T.f32(),
            [addr_ir],
            "ld.shared.f32 $0, [$1];",
            "=f,l",
            has_side_effects=True,
            asm_dialect=0,
            loc=loc,
            ip=ip,
        )
        return Float32(result_ir)

    @dsl_user_op
    def _pack_lo16(a: Uint32, b: Uint32, *, loc=None, ip=None) -> Uint32:
        """Pack low 16 bits of a and b: result = [a0, a1, b0, b1] bytes.

        Used to combine two 2-byte FP8 loads into a single Uint32 for
        fp8x4_e4m3_to_bfloat2x2.
        """
        a_ir = Uint32(a).ir_value(loc=loc, ip=ip)
        b_ir = Uint32(b).ir_value(loc=loc, ip=ip)
        result_ir = _llvm_dialect.inline_asm(
            T.i32(),
            [a_ir, b_ir],
            "prmt.b32 $0, $1, $2, 0x5410;",
            "=r,r,r",
            has_side_effects=True,
            asm_dialect=0,
            loc=loc,
            ip=ip,
        )
        return Uint32(result_ir)

    @dsl_user_op
    def _cvt_2f32_to_bf16x2(lo: Float32, hi: Float32, *,
                             loc=None, ip=None) -> Uint32:
        """Pack two FP32 values into a BF16x2 Uint32.

        lo -> bits [0:15], hi -> bits [16:31].
        """
        lo_ir = Float32(lo).ir_value(loc=loc, ip=ip)
        hi_ir = Float32(hi).ir_value(loc=loc, ip=ip)
        result_ir = _llvm_dialect.inline_asm(
            T.i32(),
            [lo_ir, hi_ir],
            "cvt.rn.bf16x2.f32 $0, $2, $1;",
            "=r,f,f",
            has_side_effects=True,
            asm_dialect=0,
            loc=loc,
            ip=ip,
        )
        return Uint32(result_ir)

    @dsl_user_op
    def _ld_global_b32(addr: Int64, *, loc=None, ip=None) -> Uint32:
        """Load 32 bits (4 bytes) from global memory at byte address.

        Uses ld.global.nc (non-coherent / read-only L2 cache path).
        Requires 4-byte alignment. Used to bypass CuTe DSL's broken
        uint8 tensor indexing for FP8 KV cache loads.
        """
        addr_ir = Int64(addr).ir_value(loc=loc, ip=ip)
        result_ir = _llvm_dialect.inline_asm(
            T.i32(),
            [addr_ir],
            "ld.global.nc.b32 $0, [$1];",
            "=r,l",
            has_side_effects=True,
            asm_dialect=0,
            loc=loc,
            ip=ip,
        )
        return Uint32(result_ir)

    @dsl_user_op
    def _scatter_4bytes_transposed(
        v_base: Int64, val: Uint32,
        col_byte: Int32, row: Int32, stride: Int32,
        *, loc=None, ip=None,
    ) -> None:
        """Scatter 4 bytes from val to transposed SMEM positions.

        Stores byte j of val at v_base + (col_byte + j) * stride + row,
        for j in 0..3. Used for V cache transposition during GMEM→SMEM
        copy. SM80+ relaxed type checking truncates .b32 source to .b8
        for st.shared.b8.
        """
        base_ir = Int64(v_base).ir_value(loc=loc, ip=ip)
        val_ir = Uint32(val).ir_value(loc=loc, ip=ip)
        col_ir = Int32(col_byte).ir_value(loc=loc, ip=ip)
        row_ir = Int32(row).ir_value(loc=loc, ip=ip)
        stride_ir = Int32(stride).ir_value(loc=loc, ip=ip)
        _llvm_dialect.inline_asm(
            None,
            [base_ir, val_ir, col_ir, row_ir, stride_ir],
            "{\n"
            "  .reg .b32 b0, b1, b2, b3, c1, c2, c3, off;\n"
            "  .reg .b64 addr;\n"
            "  // Extract individual bytes from packed u32\n"
            "  and.b32 b0, $1, 0xFF;\n"
            "  bfe.u32 b1, $1, 8, 8;\n"
            "  bfe.u32 b2, $1, 16, 8;\n"
            "  shr.b32 b3, $1, 24;\n"
            "  // Byte 0: addr = base + col*stride + row\n"
            "  mad.lo.s32 off, $2, $4, $3;\n"
            "  cvt.s64.s32 addr, off;\n"
            "  add.s64 addr, $0, addr;\n"
            "  st.shared.b8 [addr], b0;\n"
            "  // Byte 1: addr = base + (col+1)*stride + row\n"
            "  add.s32 c1, $2, 1;\n"
            "  mad.lo.s32 off, c1, $4, $3;\n"
            "  cvt.s64.s32 addr, off;\n"
            "  add.s64 addr, $0, addr;\n"
            "  st.shared.b8 [addr], b1;\n"
            "  // Byte 2: addr = base + (col+2)*stride + row\n"
            "  add.s32 c2, $2, 2;\n"
            "  mad.lo.s32 off, c2, $4, $3;\n"
            "  cvt.s64.s32 addr, off;\n"
            "  add.s64 addr, $0, addr;\n"
            "  st.shared.b8 [addr], b2;\n"
            "  // Byte 3: addr = base + (col+3)*stride + row\n"
            "  add.s32 c3, $2, 3;\n"
            "  mad.lo.s32 off, c3, $4, $3;\n"
            "  cvt.s64.s32 addr, off;\n"
            "  add.s64 addr, $0, addr;\n"
            "  st.shared.b8 [addr], b3;\n"
            "}",
            "l,r,r,r,r",
            has_side_effects=True,
            asm_dialect=0,
            loc=loc,
            ip=ip,
        )

    @dsl_user_op
    def _ld_shared_u8(addr: Int64, *, loc=None, ip=None) -> Uint32:
        """Load 1 byte from shared memory, zero-extended to Uint32.

        Explicit AND mask ensures upper 24 bits are zero even if
        SM121 ld.shared.b8 relaxed type checking doesn't zero-extend.
        """
        addr_ir = Int64(addr).ir_value(loc=loc, ip=ip)
        result_ir = _llvm_dialect.inline_asm(
            T.i32(),
            [addr_ir],
            "{\n"
            "  .reg .b32 tmp;\n"
            "  ld.shared.b8 tmp, [$1];\n"
            "  and.b32 $0, tmp, 0xFF;\n"
            "}",
            "=r,l",
            has_side_effects=True,
            asm_dialect=0,
            loc=loc,
            ip=ip,
        )
        return Uint32(result_ir)

    @dsl_user_op
    def _pack_4bytes(
        b0: Uint32, b1: Uint32, b2: Uint32, b3: Uint32,
        *, loc=None, ip=None,
    ) -> Uint32:
        """Pack 4 byte values (low byte of each Uint32) into one Uint32.

        result = b0 | (b1 << 8) | (b2 << 16) | (b3 << 24)
        Each input must have the value in its low 8 bits.
        """
        b0_ir = Uint32(b0).ir_value(loc=loc, ip=ip)
        b1_ir = Uint32(b1).ir_value(loc=loc, ip=ip)
        b2_ir = Uint32(b2).ir_value(loc=loc, ip=ip)
        b3_ir = Uint32(b3).ir_value(loc=loc, ip=ip)
        result_ir = _llvm_dialect.inline_asm(
            T.i32(),
            [b0_ir, b1_ir, b2_ir, b3_ir],
            "{\n"
            "  .reg .b32 t1, t2, t3;\n"
            "  shl.b32 t1, $2, 8;\n"
            "  shl.b32 t2, $3, 16;\n"
            "  shl.b32 t3, $4, 24;\n"
            "  or.b32 $0, $1, t1;\n"
            "  or.b32 $0, $0, t2;\n"
            "  or.b32 $0, $0, t3;\n"
            "}",
            "=r,r,r,r,r",
            has_side_effects=True,
            asm_dialect=0,
            loc=loc,
            ip=ip,
        )
        return Uint32(result_ir)

    @dsl_user_op
    def _extract_byte_from_b32(
        word: Uint32, byte_pos: Int32, *, loc=None, ip=None,
    ) -> Uint32:
        """Extract byte at byte_pos (0-3) from a 32-bit word.

        result = (word >> (byte_pos * 8)) & 0xFF
        Used as a replacement for _ld_shared_u8 byte loads.
        """
        word_ir = Uint32(word).ir_value(loc=loc, ip=ip)
        pos_ir = Int32(byte_pos).ir_value(loc=loc, ip=ip)
        result_ir = _llvm_dialect.inline_asm(
            T.i32(),
            [word_ir, pos_ir],
            "{\n"
            "  .reg .b32 shift, tmp;\n"
            "  shl.b32 shift, $2, 3;\n"
            "  shr.b32 tmp, $1, shift;\n"
            "  and.b32 $0, tmp, 0xFF;\n"
            "}",
            "=r,r,r",
            has_side_effects=True,
            asm_dialect=0,
            loc=loc,
            ip=ip,
        )
        return Uint32(result_ir)

    @cute.jit
    def _fmax(a: Float32, b: Float32) -> Float32:
        """Max of two FP32 values (NaN-safe).

        Uses cute.arch.fmax built-in.
        """
        return cute.arch.fmax(a, b)

    # --- FP4 E2M1 dequant + atomicAdd utilities ----------------------------
    # Helpers for NVFP4 weight dequantization and W_O output fusion.

    @dsl_user_op
    def _bitcast_i32_to_f32(bits, *, loc=None, ip=None) -> Float32:
        """Reinterpret Int32 bits as Float32 (mov.b32)."""
        bits_ir = bits.ir_value(loc=loc, ip=ip)
        result_ir = _llvm_dialect.inline_asm(
            T.f32(),
            [bits_ir],
            "mov.b32 $0, $1;",
            "=f,r",
            has_side_effects=True,
        )
        return Float32(result_ir)

    @cute.jit
    def _fp4_nibble_to_f32(nibble: Int32) -> Float32:
        """Convert a single FP4 E2M1 nibble (4 bits in an Int32) to Float32.

        E2M1 format: [sign(1) | exp(2) | mant(1)]
        Unsigned values: {0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0}

        Pure integer arithmetic — no comparison operators (CuTe DSL
        comparisons return predicates, not Int32 0/1 flags).
        """
        sign = (nibble >> Int32(3)) & Int32(1)
        exp2 = (nibble >> Int32(1)) & Int32(3)
        mant1 = nibble & Int32(1)

        # exp2_flag = 1 when exp2 > 0, 0 when exp2 == 0
        # For 2-bit value: OR the two bits together
        exp2_flag = (exp2 | (exp2 >> Int32(1))) & Int32(1)
        exp2_zero = Int32(1) - exp2_flag

        # Normal case (exp2 > 0): (exp2 + 126) << 23 | mant1 << 22
        normal_bits = ((exp2 + Int32(126)) << Int32(23)) | (mant1 << Int32(22))

        # Subnormal case (exp2 == 0, mant1 == 1): 0.5 = 0x3F000000
        special_bits = mant1 * Int32(0x3F000000)

        # Branchless select via multiply-by-flag
        unsigned_bits = normal_bits * exp2_flag + special_bits * exp2_zero

        # Apply sign
        result_bits = unsigned_bits | (sign << Int32(31))

        return _bitcast_i32_to_f32(result_bits)

    @cute.jit
    def _fp4_byte_to_f32x2(byte_val: Int32) -> tuple:
        """Unpack one uint8 (two packed FP4 E2M1 values) into (Float32, Float32).

        Low nibble = even-indexed element, High nibble = odd-indexed element.
        """
        lo = byte_val & Int32(0x0F)
        hi = (byte_val >> Int32(4)) & Int32(0x0F)
        return _fp4_nibble_to_f32(lo), _fp4_nibble_to_f32(hi)

    @cute.jit
    def _cvt_bf16x2_lo_to_f32(bf16x2: Uint32) -> Float32:
        """Extract the low BF16 from a packed BF16x2 Uint32 and convert to Float32.

        BF16 is the top 16 bits of IEEE 754 float32, so: mask low 16 bits,
        shift left 16, reinterpret as float. Pure DSL — no inline PTX.
        """
        masked = bf16x2 & Uint32(0xFFFF)
        shifted = Int32(masked << Uint32(16))
        return _bitcast_i32_to_f32(shifted)

    @cute.jit
    def _ld_swizzled_scale(
        sf_ptr: Int64,
        m: Int32,
        k_group: Int32,
        num_k_tiles: Int32,
    ) -> Float32:
        """Load one FP8 E4M3 block scale from swizzled layout, return as Float32.

        Swizzle layout: [numMTiles, numKTiles, 32, 4, 4]
        Offset formula from CUTLASS nvfp4_utils.cuh.
        """
        m_tile = m >> Int32(7)                    # m / 128
        outer_m = m & Int32(31)                   # m % 32
        inner_m = (m >> Int32(5)) & Int32(3)      # (m / 32) % 4
        k_tile = k_group >> Int32(2)              # k_group / 4
        inner_k = k_group & Int32(3)              # k_group % 4

        sf_offset = (m_tile * num_k_tiles + k_tile) * Int32(512) \
            + outer_m * Int32(16) + inner_m * Int32(4) + inner_k

        # Load byte via aligned b32 load + extract
        aligned_addr = sf_ptr + Int64(sf_offset & Int32(0xFFFFFFFC))
        raw_word = _ld_global_b32(aligned_addr)
        byte_pos = sf_offset & Int32(3)
        scale_byte = _extract_byte_from_b32(raw_word, byte_pos)

        # Convert FP8 E4M3 -> BF16x2 -> F32
        # Pack same byte twice, convert pair, extract low
        packed = _pack_lo16(scale_byte, scale_byte)
        bf16_lo, _bf16_hi = fp8x4_e4m3_to_bfloat2x2(packed)
        return _cvt_bf16x2_lo_to_f32(bf16_lo)

    @dsl_user_op
    def _atomic_add_f32(addr, val, *, loc=None, ip=None) -> Uint32:
        """atomicAdd a Float32 value to global memory. Returns old value (discarded)."""
        addr_ir = addr.ir_value(loc=loc, ip=ip)
        val_ir = val.ir_value(loc=loc, ip=ip)
        result_ir = _llvm_dialect.inline_asm(
            T.i32(),
            [addr_ir, val_ir],
            "atom.global.add.f32 $0, [$1], $2;",
            "=r,l,f",
            has_side_effects=True,
        )
        return Uint32(result_ir)

    # --- Phase C: RMSNorm fusion PTX helpers --------------------------------

    @dsl_user_op
    def _threadfence(*, loc=None, ip=None):
        """Global memory fence — membar.gl.

        Ensures prior stores (Phase B atomicAdd) are visible to all
        threads in the GPU before Phase C reads the accumulated result.
        """
        _llvm_dialect.inline_asm(
            T.i32(), [],
            "membar.gl; mov.u32 $0, 0;",
            "=r",
            has_side_effects=True, loc=loc, ip=ip)

    @dsl_user_op
    def _acquire_fence(*, loc=None, ip=None):
        """Acquire fence — fence.acq_rel.gpu.

        Lighter than membar.gl for post-barrier loads. Guarantees that
        prior releases (atomic_add into arrival counter + membar.gl)
        are visible to subsequent loads of the guarded region.

        Used by Phase E β kernel to acquire after spin-wait on the
        grid-barrier arrival counter.
        """
        _llvm_dialect.inline_asm(
            T.i32(), [],
            "fence.acq_rel.gpu; mov.u32 $0, 0;",
            "=r",
            has_side_effects=True, loc=loc, ip=ip)

    @dsl_user_op
    def _atomic_add_u32(addr: Int64, val: Int32, *, loc=None, ip=None) -> Int32:
        """Integer atomicAdd, returns old value.

        Used for cross-CTA arrival counting — last CTA runs Phase C.
        """
        addr_ir = Int64(addr).ir_value(loc=loc, ip=ip)
        val_ir = Int32(val).ir_value(loc=loc, ip=ip)
        result_ir = _llvm_dialect.inline_asm(
            T.i32(), [addr_ir, val_ir],
            "atom.global.add.u32 $0, [$1], $2;", "=r,l,r",
            has_side_effects=True, loc=loc, ip=ip)
        return Int32(result_ir)

    @dsl_user_op
    def _ld_volatile_u32(addr: Int64, *, loc=None, ip=None) -> Int32:
        """Volatile U32 load — ld.volatile.global.u32.

        Used by Phase E β kernel's grid-barrier spin-wait: each CTA loops
        reading the arrival counter until it sees all CTAs have arrived.
        `volatile` prevents the compiler from hoisting the load out of the
        loop, so every iteration re-reads from the global-visible value
        produced by other CTAs' _atomic_add_u32 + _threadfence pair.
        """
        addr_ir = Int64(addr).ir_value(loc=loc, ip=ip)
        result_ir = _llvm_dialect.inline_asm(
            T.i32(), [addr_ir],
            "ld.volatile.global.u32 $0, [$1];", "=r,l",
            has_side_effects=True, loc=loc, ip=ip)
        return Int32(result_ir)

    @dsl_user_op
    def _rsqrt_approx_f32(x: Float32, *, loc=None, ip=None) -> Float32:
        """Hardware reciprocal square root — rsqrt.approx.ftz.f32.

        Single-instruction, sufficient precision for RMSNorm.
        """
        x_ir = Float32(x).ir_value(loc=loc, ip=ip)
        result_ir = _llvm_dialect.inline_asm(
            T.f32(), [x_ir],
            "rsqrt.approx.ftz.f32 $0, $1;", "=f,f",
            has_side_effects=True, loc=loc, ip=ip)
        return Float32(result_ir)

    @dsl_user_op
    def _rcp_approx_f32(x: Float32, *, loc=None, ip=None) -> Float32:
        """Hardware reciprocal approximation — rcp.approx.ftz.f32.

        Used for sigmoid: 1 / (1 + exp2(-x * LOG2E)).
        Single-instruction, sufficient precision for gating.
        """
        x_ir = Float32(x).ir_value(loc=loc, ip=ip)
        result_ir = _llvm_dialect.inline_asm(
            T.f32(), [x_ir],
            "rcp.approx.ftz.f32 $0, $1;", "=f,f",
            has_side_effects=True, loc=loc, ip=ip)
        return Float32(result_ir)

    @dsl_user_op
    def _ld_global_f32(addr: Int64, *, loc=None, ip=None) -> Float32:
        """Load FP32 from global memory at byte address."""
        addr_ir = Int64(addr).ir_value(loc=loc, ip=ip)
        result_ir = _llvm_dialect.inline_asm(
            T.f32(), [addr_ir],
            "ld.global.f32 $0, [$1];", "=f,l",
            has_side_effects=True, loc=loc, ip=ip)
        return Float32(result_ir)

    @dsl_user_op
    def _st_global_f32(addr: Int64, val: Float32, *, loc=None, ip=None) -> None:
        """Store FP32 to global memory at byte address."""
        addr_ir = Int64(addr).ir_value(loc=loc, ip=ip)
        val_ir = Float32(val).ir_value(loc=loc, ip=ip)
        _llvm_dialect.inline_asm(
            None, [addr_ir, val_ir],
            "st.global.f32 [$0], $1;", "l,f",
            has_side_effects=True, loc=loc, ip=ip)

    @dsl_user_op
    def _ld_global_b16_to_f32(addr: Int64, *, loc=None, ip=None) -> Float32:
        """Load BF16 from global memory and convert to FP32.

        Loads 16-bit value, shifts left 16 to reconstruct FP32 bits.
        BF16 is the upper 16 bits of IEEE 754 float32.

        NOTE: Single-line brace pattern — matches proven _ld_shared_b16.
        Multi-line braces cause ptxas parse errors inside dynamic ifs.
        """
        addr_ir = Int64(addr).ir_value(loc=loc, ip=ip)
        result_ir = _llvm_dialect.inline_asm(
            T.f32(), [addr_ir],
            "{.reg .b16 t16; .reg .b32 t32;"
            " ld.global.b16 t16, [$1];"
            " cvt.u32.u16 t32, t16;"
            " shl.b32 t32, t32, 16;"
            " mov.b32 $0, t32;}",
            "=f,l",
            has_side_effects=True, loc=loc, ip=ip)
        return Float32(result_ir)

    @dsl_user_op
    def _st_global_bf16_from_f32(addr: Int64, val: Float32, *,
                                  loc=None, ip=None) -> None:
        """Convert FP32 to BF16 and store to global memory.

        Uses cvt.rn.bf16.f32 for round-to-nearest conversion.

        NOTE: Single-line brace pattern — matches proven
        _st_shared_b16_from_u32. Multi-line braces cause ptxas
        parse errors inside dynamic ifs.
        """
        addr_ir = Int64(addr).ir_value(loc=loc, ip=ip)
        val_ir = Float32(val).ir_value(loc=loc, ip=ip)
        _llvm_dialect.inline_asm(
            None, [addr_ir, val_ir],
            "{.reg .b16 t; cvt.rn.bf16.f32 t, $1; st.global.b16 [$0], t;}",
            "l,f",
            has_side_effects=True, loc=loc, ip=ip)

    @dsl_user_op
    def _read_globaltimer_u64(*, loc=None, ip=None) -> Uint64:
        """Read %globaltimer (64-bit nanosecond clock, globally synchronized).

        Available since Kepler. Synchronized across SMs — cross-CTA
        timeline diffs are meaningful (unlike %clock64 which is per-SM).
        Resolution is ~1 ns.
        """
        result_ir = _llvm_dialect.inline_asm(
            T.i64(), [],
            "mov.u64 $0, %globaltimer;", "=l",
            has_side_effects=True, loc=loc, ip=ip)
        return Uint64(result_ir)

    @dsl_user_op
    def _read_clock64_u64(*, loc=None, ip=None) -> Uint64:
        """Read %clock64 (64-bit per-SM cycle counter).

        Per-SM, NOT synchronized across SMs. CTA-local diffs are valid;
        cross-CTA absolute diffs are not. Use as fallback if globaltimer
        plumbing fails.
        """
        result_ir = _llvm_dialect.inline_asm(
            T.i64(), [],
            "mov.u64 $0, %clock64;", "=l",
            has_side_effects=True, loc=loc, ip=ip)
        return Uint64(result_ir)

    @dsl_user_op
    def _st_global_u64(addr: Int64, val: Uint64, *, loc=None, ip=None) -> None:
        """Store 64-bit unsigned to global memory at byte address."""
        addr_ir = Int64(addr).ir_value(loc=loc, ip=ip)
        val_ir = Uint64(val).ir_value(loc=loc, ip=ip)
        _llvm_dialect.inline_asm(
            None, [addr_ir, val_ir],
            "st.global.b64 [$0], $1;", "l,l",
            has_side_effects=True, loc=loc, ip=ip)

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
            self.head_dim = config.head_dim      # 256
            self.block_size = config.block_size  # 64
            self.num_warps_q = config.num_warps_q    # 1
            self.num_warps_kv = config.num_warps_kv  # 4
            self.num_threads = 128  # 4 warps * 32 threads
            self.num_mma_d = self.head_dim // 16  # 16 for head_dim=256

            # SMEM sizes (bytes) — non-overlapping layout
            # QKV stays resident; sync buffers placed AFTER QKV so
            # writing sync_o never corrupts K/V data during page loop.
            self.q_bytes = self.cta_q * self.head_dim * 2      # BF16
            self.k_bytes = self.cta_kv * self.head_dim * 1     # FP8
            self.v_bytes = self.cta_kv * self.head_dim * 1     # FP8
            self.qkv_bytes = self.q_bytes + self.k_bytes + self.v_bytes

            # Per-_md sync buffers (16 cols, not full 256)
            # sync_o_small: 4 warps × 16 rows × 16 cols × 4 bytes
            self.sync_o_small_bytes = (
                self.num_warps_kv * self.cta_q * 16 * 4
            )
            # sync_md_small: 4 warps × 16 rows × 8 bytes (m + d)
            self.sync_md_small_bytes = (
                self.num_warps_kv * self.cta_q * 8
            )
            self.sync_o_small_offset = self.qkv_bytes
            self.sync_md_small_offset = (
                self.qkv_bytes + self.sync_o_small_bytes
            )
            self.smem_bytes = (
                self.qkv_bytes
                + self.sync_o_small_bytes
                + self.sync_md_small_bytes
            )  # 40960 + 4096 + 512 = 45568, fits in 101 KB
            self._compiled = None

        @cute.jit
        def _jit_launch(self, query, k_ptr: Int64, v_ptr: Int64,
                        page_table, seq_lens, output,
                        scale, k_scale, v_scale,
                        num_q_heads, num_kv_heads,
                        kv_page_stride: Int32,
                        wo_weight_ptr: Int64, wo_scale_ptr: Int64,
                        wo_output_ptr: Int64, wo_gs_ptr: Int64,
                        wo_num_k_tiles: Int32,
                        wo_weight_row_stride: Int32,
                        wo_fused: Int32,
                        rmsnorm_gamma_ptr: Int64,
                        rmsnorm_residual_ptr: Int64,
                        rmsnorm_output_ptr: Int64,
                        residual_output_ptr: Int64,
                        arrival_count_ptr: Int64,
                        rmsnorm_eps,
                        hidden_dim: Int32,
                        total_ctas_per_seq: Int32,
                        rmsnorm_fused: Int32,
                        gate_ptr: Int64,
                        gate_fused: Int32,
                        grid_x: Int32, grid_y: Int32, grid_z: Int32,
                        stream):
            """JIT host wrapper: compiles kernel launch into MLIR.

            stream: cuda.CUstream — honored by CuTe DSL via `.launch(stream=...)`,
            which maps to `async_deps` on gpu.launch_func. Without this the
            kernel launches on CuTe's internal default stream and is invisible
            to PyTorch CUDA graph capture, breaking FULL_AND_PIECEWISE mode.
            """
            self._kernel(
                query, k_ptr, v_ptr, page_table, seq_lens,
                output, scale, k_scale, v_scale,
                num_q_heads, num_kv_heads,
                kv_page_stride,
                wo_weight_ptr, wo_scale_ptr,
                wo_output_ptr, wo_gs_ptr,
                wo_num_k_tiles, wo_weight_row_stride,
                wo_fused,
                rmsnorm_gamma_ptr,
                rmsnorm_residual_ptr,
                rmsnorm_output_ptr,
                residual_output_ptr,
                arrival_count_ptr,
                rmsnorm_eps,
                hidden_dim,
                total_ctas_per_seq,
                rmsnorm_fused,
                gate_ptr,
                gate_fused,
            ).launch(
                grid=[grid_x, grid_y, grid_z],
                block=[self.num_threads, 1, 1],
                smem=self.smem_bytes,
                stream=stream,
            )

        @cute.kernel
        def _kernel(self, query, k_ptr: Int64, v_ptr: Int64,
                     page_table, seq_lens,
                     output, scale, k_scale, v_scale,
                     num_q_heads, num_kv_heads,
                     kv_page_stride: Int32,
                     wo_weight_ptr: Int64, wo_scale_ptr: Int64,
                     wo_output_ptr: Int64, wo_gs_ptr: Int64,
                     wo_num_k_tiles: Int32,
                     wo_weight_row_stride: Int32,
                     wo_fused: Int32,
                     rmsnorm_gamma_ptr: Int64,
                     rmsnorm_residual_ptr: Int64,
                     rmsnorm_output_ptr: Int64,
                     residual_output_ptr: Int64,
                     arrival_count_ptr: Int64,
                     rmsnorm_eps,
                     hidden_dim: Int32,
                     total_ctas_per_seq: Int32,
                     rmsnorm_fused: Int32,
                     gate_ptr: Int64,
                     gate_fused: Int32):
            """CuTe DSL decode kernel for FP8 paged attention on SM121.

            Structure: outer _md loop over head_dim blocks, inner page loop.
            Each _md iteration processes a 16-column slice of head_dim using
            FP8 MMA for QK, softmax, then BF16 MMA for PV accumulation.
            Cross-warp reduction merges partial results across KV warps.

            Grid: (num_q_tiles, num_kv_heads, num_seqs)
            """
            # === Phase 0: Thread / block identification ===
            bx, by, bz = cute.arch.block_idx()
            lane = cute.arch.lane_idx()
            warp = cute.arch.warp_idx()
            tid = warp * Int32(32) + lane
            group = lane >> Int32(2)   # lane // 4 = MMA row group 0-7
            sub = lane & Int32(3)      # lane % 4 = MMA col sub 0-3

            kv_head_idx = by
            seq_idx = bz
            group_size = num_q_heads // num_kv_heads
            q_head_start = kv_head_idx * group_size + bx * Int32(self.cta_q)

            seq_len = seq_lens[seq_idx]
            num_pages = (seq_len + Int32(self.block_size - 1)) \
                // Int32(self.block_size)

            # Combined scale: attention_scale * k_scale * log2(e)
            LOG2E = Float32(1.4426950408889634)
            sm_scale_log2 = Float32(scale) * Float32(k_scale) * LOG2E
            v_scale_f32 = Float32(v_scale)

            # === Phase 1: SMEM pointers (non-overlapping layout) ===
            smem = cute.arch.get_dyn_smem(cutlass.Uint8, alignment=128)
            q_smem = shared_ptr_to_i64(smem)
            k_smem = shared_ptr_to_i64(
                smem + Int32(self.q_bytes))
            v_smem = shared_ptr_to_i64(
                smem + Int32(self.q_bytes + self.k_bytes))
            sync_o = shared_ptr_to_i64(
                smem + Int32(self.sync_o_small_offset))
            sync_md = shared_ptr_to_i64(
                smem + Int32(self.sync_md_small_offset))

            hd = Int32(self.head_dim)  # 256
            warp_kv_start = warp * Int32(16)

            # Strides for KV cache global memory
            kv_tok_stride = num_kv_heads * hd

            # === Phase 2: Load Q into SMEM (once, persists) ===
            q_stride_tok = num_q_heads * hd
            elems_per_thr_q = Int32(self.cta_q * self.head_dim
                                    // self.num_threads)
            for _i in cutlass.range_constexpr(
                self.cta_q * self.head_dim // self.num_threads
            ):
                flat = tid * elems_per_thr_q + Int32(_i)
                row = flat // hd
                col = flat % hd
                gmem_idx = (seq_idx * q_stride_tok
                            + (q_head_start + row) * hd + col)
                smem_byte = (row * hd + col) * Int32(2)
                val = query[gmem_idx]
                val_u32 = _cvt_2f32_to_bf16x2(
                    Float32(val), Float32(0.0))
                _st_shared_b16_from_u32(
                    q_smem + Int64(smem_byte), val_u32)

            cute.arch.sync_threads()

            # === Phase 3: Serialized _md loop ===
            # Process one 16-column output block at a time.
            # 8 scalar accumulators per iteration — NO rmem_tensor.
            # QK recomputed each _md (16x total) to avoid 128-reg accum.
            for _md_c in cutlass.range_constexpr(self.num_mma_d):
                _md_idx = Int32(_md_c)
                # Per-_md accumulators (8 scalars = 8 regs)
                o0 = Float32(0.0)
                o1 = Float32(0.0)
                o2 = Float32(0.0)
                o3 = Float32(0.0)
                o4 = Float32(0.0)
                o5 = Float32(0.0)
                o6 = Float32(0.0)
                o7 = Float32(0.0)
                m_r0 = Float32(-1e30)
                m_r1 = Float32(-1e30)
                d_r0 = Float32(0.0)
                d_r1 = Float32(0.0)

                # --- Inner page loop ---
                page_idx = Int32(0)
                while page_idx < num_pages:
                    phys_page = page_table[seq_idx, page_idx]

                    # -- Load K page (row-major, 4B/iter) --
                    elems_per_thr_kv4 = Int32(
                        self.cta_kv * self.head_dim
                        // 4 // self.num_threads)
                    for _i in cutlass.range_constexpr(
                        self.cta_kv * self.head_dim
                        // 4 // self.num_threads
                    ):
                        flat = tid * elems_per_thr_kv4 + Int32(_i)
                        row = flat >> Int32(6)
                        col4 = flat & Int32(63)
                        k_byte_off = (phys_page * kv_page_stride
                                      + row * kv_tok_stride
                                      + kv_head_idx * hd
                                      + col4 * Int32(4))
                        k_raw = _ld_global_b32(
                            k_ptr + Int64(k_byte_off))
                        smem_byte = row * hd + col4 * Int32(4)
                        _st_shared_b32(
                            k_smem + Int64(smem_byte), k_raw)

                    # -- Load V page (row-major, 4B/iter) --
                    for _i in cutlass.range_constexpr(
                        self.cta_kv * self.head_dim
                        // 4 // self.num_threads
                    ):
                        flat = tid * elems_per_thr_kv4 + Int32(_i)
                        row = flat >> Int32(6)
                        col4 = flat & Int32(63)
                        v_byte_off = (phys_page * kv_page_stride
                                      + row * kv_tok_stride
                                      + kv_head_idx * hd
                                      + col4 * Int32(4))
                        v_raw = _ld_global_b32(
                            v_ptr + Int64(v_byte_off))
                        v_smem_byte = row * hd + col4 * Int32(4)
                        _st_shared_b32(
                            v_smem + Int64(v_smem_byte), v_raw)

                    cute.arch.sync_threads()

                    # -- QK MMA (all 16 K-dim iterations) --
                    s0 = Float32(0.0)
                    s1 = Float32(0.0)
                    s2 = Float32(0.0)
                    s3 = Float32(0.0)
                    s4 = Float32(0.0)
                    s5 = Float32(0.0)
                    s6 = Float32(0.0)
                    s7 = Float32(0.0)

                    for _kd in cutlass.range_constexpr(
                        self.num_mma_d
                    ):
                        k_start = Int32(_kd * 16)
                        q_byte_a0 = (group * hd + k_start
                                     + sub * Int32(2)) * Int32(2)
                        a0 = _ld_shared_b32(
                            q_smem + Int64(q_byte_a0))
                        a1 = _ld_shared_b32(
                            q_smem + Int64(q_byte_a0 + Int32(16)))
                        q_byte_a2 = ((group + Int32(8)) * hd
                                     + k_start
                                     + sub * Int32(2)) * Int32(2)
                        a2 = _ld_shared_b32(
                            q_smem + Int64(q_byte_a2))
                        a3 = _ld_shared_b32(
                            q_smem + Int64(q_byte_a2 + Int32(16)))

                        n_t = group
                        kv_row_0 = warp_kv_start + n_t
                        k_off_0a = (kv_row_0 * hd + k_start
                                    + sub * Int32(2))
                        k_raw_0a = _ld_shared_b16(
                            k_smem + Int64(k_off_0a))
                        k_raw_0b = _ld_shared_b16(
                            k_smem + Int64(k_off_0a + Int32(8)))
                        k_packed_0 = _pack_lo16(k_raw_0a, k_raw_0b)
                        b0, b1 = fp8x4_e4m3_to_bfloat2x2(k_packed_0)

                        kv_row_1 = warp_kv_start + n_t + Int32(8)
                        k_off_1a = (kv_row_1 * hd + k_start
                                    + sub * Int32(2))
                        k_raw_1a = _ld_shared_b16(
                            k_smem + Int64(k_off_1a))
                        k_raw_1b = _ld_shared_b16(
                            k_smem + Int64(k_off_1a + Int32(8)))
                        k_packed_1 = _pack_lo16(k_raw_1a, k_raw_1b)
                        b2, b3 = fp8x4_e4m3_to_bfloat2x2(k_packed_1)

                        (s0, s1, s2, s3,
                         s4, s5, s6, s7) = bf16_mma_m16n16k16_f32(
                            s0, s1, s2, s3, s4, s5, s6, s7,
                            a0, a1, a2, a3,
                            b0, b1, b2, b3)

                    # -- Online softmax --
                    s0 = s0 * sm_scale_log2
                    s1 = s1 * sm_scale_log2
                    s2 = s2 * sm_scale_log2
                    s3 = s3 * sm_scale_log2
                    s4 = s4 * sm_scale_log2
                    s5 = s5 * sm_scale_log2
                    s6 = s6 * sm_scale_log2
                    s7 = s7 * sm_scale_log2

                    tok_base = page_idx * Int32(self.block_size) \
                        + warp_kv_start
                    NEG = Float32(-1e20)
                    tok0 = tok_base + sub * Int32(2)
                    tok1 = tok0 + Int32(1)
                    tok8 = tok0 + Int32(8)
                    tok9 = tok8 + Int32(1)
                    if tok0 >= seq_len:
                        s0 = NEG
                        s2 = NEG
                    if tok1 >= seq_len:
                        s1 = NEG
                        s3 = NEG
                    if tok8 >= seq_len:
                        s4 = NEG
                        s6 = NEG
                    if tok9 >= seq_len:
                        s5 = NEG
                        s7 = NEG

                    lm0 = _fmax(_fmax(s0, s1), _fmax(s4, s5))
                    lm1 = _fmax(_fmax(s2, s3), _fmax(s6, s7))
                    lm0 = _fmax(lm0, shfl_xor_sync(lm0, Int32(1)))
                    lm0 = _fmax(lm0, shfl_xor_sync(lm0, Int32(2)))
                    lm1 = _fmax(lm1, shfl_xor_sync(lm1, Int32(1)))
                    lm1 = _fmax(lm1, shfl_xor_sync(lm1, Int32(2)))

                    m0_new = _fmax(m_r0, lm0)
                    m1_new = _fmax(m_r1, lm1)
                    sc0 = exp2_approx_ftz_f32(m_r0 - m0_new)
                    sc1 = exp2_approx_ftz_f32(m_r1 - m1_new)
                    d_r0 = d_r0 * sc0
                    d_r1 = d_r1 * sc1

                    # Rescale 8 scalar accumulators (not 128!)
                    o0 = o0 * sc0
                    o1 = o1 * sc0
                    o2 = o2 * sc1
                    o3 = o3 * sc1
                    o4 = o4 * sc0
                    o5 = o5 * sc0
                    o6 = o6 * sc1
                    o7 = o7 * sc1

                    m_r0 = m0_new
                    m_r1 = m1_new

                    p0 = exp2_approx_ftz_f32(s0 - m_r0)
                    p1 = exp2_approx_ftz_f32(s1 - m_r0)
                    p2 = exp2_approx_ftz_f32(s2 - m_r1)
                    p3 = exp2_approx_ftz_f32(s3 - m_r1)
                    p4 = exp2_approx_ftz_f32(s4 - m_r0)
                    p5 = exp2_approx_ftz_f32(s5 - m_r0)
                    p6 = exp2_approx_ftz_f32(s6 - m_r1)
                    p7 = exp2_approx_ftz_f32(s7 - m_r1)

                    ls0 = (p0 + p1) + (p4 + p5)
                    ls1 = (p2 + p3) + (p6 + p7)
                    ls0 = ls0 + shfl_xor_sync(ls0, Int32(1))
                    ls0 = ls0 + shfl_xor_sync(ls0, Int32(2))
                    ls1 = ls1 + shfl_xor_sync(ls1, Int32(1))
                    ls1 = ls1 + shfl_xor_sync(ls1, Int32(2))
                    d_r0 = d_r0 + ls0
                    d_r1 = d_r1 + ls1

                    # -- PV MMA for current _md_idx only --
                    pa0 = _cvt_2f32_to_bf16x2(
                        p0 * v_scale_f32, p1 * v_scale_f32)
                    pa1 = _cvt_2f32_to_bf16x2(
                        p4 * v_scale_f32, p5 * v_scale_f32)
                    pa2 = _cvt_2f32_to_bf16x2(
                        p2 * v_scale_f32, p3 * v_scale_f32)
                    pa3 = _cvt_2f32_to_bf16x2(
                        p6 * v_scale_f32, p7 * v_scale_f32)

                    # V fragment for current _md_idx: load from SMEM
                    v_k_start = _md_idx * Int32(16)
                    v_tok0 = warp_kv_start + sub * Int32(2)

                    # First m16n8: V cols [v_k_start+group]
                    v_hd0 = v_k_start + group
                    v_off_0a = v_tok0 * hd + v_hd0
                    v_off_0b = (v_tok0 + Int32(1)) * hd + v_hd0
                    v_off_8a = (v_tok0 + Int32(8)) * hd + v_hd0
                    v_off_8b = (v_tok0 + Int32(9)) * hd + v_hd0
                    # 4-byte aligned loads + byte extract
                    vw0 = _ld_shared_b32(
                        v_smem + Int64(v_off_0a & Int32(0xFFFFFFFC)))
                    vw1 = _ld_shared_b32(
                        v_smem + Int64(v_off_0b & Int32(0xFFFFFFFC)))
                    vw8 = _ld_shared_b32(
                        v_smem + Int64(v_off_8a & Int32(0xFFFFFFFC)))
                    vw9 = _ld_shared_b32(
                        v_smem + Int64(v_off_8b & Int32(0xFFFFFFFC)))
                    v_byte_pos = v_hd0 & Int32(3)
                    vb0_0 = _extract_byte_from_b32(vw0, v_byte_pos)
                    vb0_1 = _extract_byte_from_b32(vw1, v_byte_pos)
                    vb0_8 = _extract_byte_from_b32(vw8, v_byte_pos)
                    vb0_9 = _extract_byte_from_b32(vw9, v_byte_pos)
                    v_packed_0 = _pack_4bytes(
                        vb0_0, vb0_1, vb0_8, vb0_9)
                    vb0, vb1 = fp8x4_e4m3_to_bfloat2x2(v_packed_0)

                    # Second m16n8: V cols [v_k_start+group+8]
                    v_hd1 = v_k_start + group + Int32(8)
                    v_off_0c = v_tok0 * hd + v_hd1
                    v_off_0d = (v_tok0 + Int32(1)) * hd + v_hd1
                    v_off_8c = (v_tok0 + Int32(8)) * hd + v_hd1
                    v_off_8d = (v_tok0 + Int32(9)) * hd + v_hd1
                    vw0b = _ld_shared_b32(
                        v_smem + Int64(v_off_0c & Int32(0xFFFFFFFC)))
                    vw1b = _ld_shared_b32(
                        v_smem + Int64(v_off_0d & Int32(0xFFFFFFFC)))
                    vw8b = _ld_shared_b32(
                        v_smem + Int64(v_off_8c & Int32(0xFFFFFFFC)))
                    vw9b = _ld_shared_b32(
                        v_smem + Int64(v_off_8d & Int32(0xFFFFFFFC)))
                    v_byte_pos1 = v_hd1 & Int32(3)
                    vb1_0 = _extract_byte_from_b32(vw0b, v_byte_pos1)
                    vb1_1 = _extract_byte_from_b32(vw1b, v_byte_pos1)
                    vb1_8 = _extract_byte_from_b32(vw8b, v_byte_pos1)
                    vb1_9 = _extract_byte_from_b32(vw9b, v_byte_pos1)
                    v_packed_1 = _pack_4bytes(
                        vb1_0, vb1_1, vb1_8, vb1_9)
                    vb2, vb3 = fp8x4_e4m3_to_bfloat2x2(v_packed_1)

                    # PV MMA — accumulate into scalar o0..o7
                    (t0, t1, t2, t3,
                     t4, t5, t6, t7) = bf16_mma_m16n16k16_f32(
                        Float32(0.0), Float32(0.0),
                        Float32(0.0), Float32(0.0),
                        Float32(0.0), Float32(0.0),
                        Float32(0.0), Float32(0.0),
                        pa0, pa1, pa2, pa3,
                        vb0, vb1, vb2, vb3)
                    o0 = o0 + t0
                    o1 = o1 + t1
                    o2 = o2 + t2
                    o3 = o3 + t3
                    o4 = o4 + t4
                    o5 = o5 + t5
                    o6 = o6 + t6
                    o7 = o7 + t7

                    cute.arch.sync_threads()
                    page_idx = page_idx + Int32(1)
                # end page loop for this _md

                # === Write 8 accum values + m,d to sync buffers ===
                # sync_o_small layout: [warp][row][col16], FP32
                # col16 = local column within this _md block (0..15)
                # MMA fragment → col mapping:
                #   o0: row=group,    col=sub*2
                #   o1: row=group,    col=sub*2+1
                #   o2: row=group+8,  col=sub*2
                #   o3: row=group+8,  col=sub*2+1
                #   o4: row=group,    col=sub*2+8
                #   o5: row=group,    col=sub*2+9
                #   o6: row=group+8,  col=sub*2+8
                #   o7: row=group+8,  col=sub*2+9
                W16 = Int32(16)  # cols per _md block
                so_warp_off = warp * Int32(self.cta_q) * W16 * Int32(4)
                so_r0 = so_warp_off + group * W16 * Int32(4)
                so_r1 = so_warp_off + (group + Int32(8)) * W16 * Int32(4)
                lc0 = sub * Int32(2)
                lc8 = sub * Int32(2) + Int32(8)

                _st_shared_f32(sync_o + Int64(
                    so_r0 + lc0 * Int32(4)), o0)
                _st_shared_f32(sync_o + Int64(
                    so_r0 + (lc0 + Int32(1)) * Int32(4)), o1)
                _st_shared_f32(sync_o + Int64(
                    so_r1 + lc0 * Int32(4)), o2)
                _st_shared_f32(sync_o + Int64(
                    so_r1 + (lc0 + Int32(1)) * Int32(4)), o3)
                _st_shared_f32(sync_o + Int64(
                    so_r0 + lc8 * Int32(4)), o4)
                _st_shared_f32(sync_o + Int64(
                    so_r0 + (lc8 + Int32(1)) * Int32(4)), o5)
                _st_shared_f32(sync_o + Int64(
                    so_r1 + lc8 * Int32(4)), o6)
                _st_shared_f32(sync_o + Int64(
                    so_r1 + (lc8 + Int32(1)) * Int32(4)), o7)

                # Write m, d (sub 0 only — all subs have same values)
                if sub == Int32(0):
                    md_w_off = warp * Int32(self.cta_q) * Int32(8)
                    _st_shared_f32(sync_md + Int64(
                        md_w_off + group * Int32(8)), m_r0)
                    _st_shared_f32(sync_md + Int64(
                        md_w_off + group * Int32(8)
                        + Int32(4)), d_r0)
                    _st_shared_f32(sync_md + Int64(
                        md_w_off + (group + Int32(8)) * Int32(8)),
                        m_r1)
                    _st_shared_f32(sync_md + Int64(
                        md_w_off + (group + Int32(8)) * Int32(8)
                        + Int32(4)), d_r1)

                cute.arch.sync_threads()

                # === Cross-warp reduction ===
                if True:
                    if warp == Int32(0):
                        red_row = lane >> Int32(1)
                        col_base = (lane & Int32(1)) * Int32(8)

                        for _e in cutlass.range_constexpr(8):
                            col16 = col_base + Int32(_e)

                            m_final = Float32(-1e30)
                            for _w in cutlass.range_constexpr(
                                self.num_warps_kv
                            ):
                                m_w = _ld_shared_f32(
                                    sync_md + Int64(
                                        Int32(_w * self.cta_q)
                                        * Int32(8)
                                        + red_row * Int32(8)))
                                m_final = _fmax(m_final, m_w)

                            o_final = Float32(0.0)
                            d_final = Float32(0.0)
                            for _w in cutlass.range_constexpr(
                                self.num_warps_kv
                            ):
                                w_base = Int32(
                                    _w * self.cta_q * 16)
                                o_w = _ld_shared_f32(
                                    sync_o + Int64(
                                        (w_base + red_row * W16
                                         + col16) * Int32(4)))
                                m_w = _ld_shared_f32(
                                    sync_md + Int64(
                                        Int32(_w * self.cta_q)
                                        * Int32(8)
                                        + red_row * Int32(8)))
                                d_w = _ld_shared_f32(
                                    sync_md + Int64(
                                        Int32(_w * self.cta_q)
                                        * Int32(8) + red_row * Int32(8)
                                        + Int32(4)))
                                rescale = exp2_approx_ftz_f32(
                                    m_w - m_final)
                                o_final = o_final + o_w * rescale
                                d_final = d_final + d_w * rescale

                            o_final = o_final / d_final
                            out_head = q_head_start + red_row
                            g_col = _md_idx * Int32(16) + col16
                            if red_row < group_size:
                                out_idx = (seq_idx * num_q_heads * hd
                                           + out_head * hd + g_col)
                                # Gate fusion: sigmoid(gate) * attn_output
                                # gate_buf layout: [num_seqs, num_q_heads * head_dim] BF16
                                if gate_fused != Int32(0):
                                    gate_elem_idx = (seq_idx * num_q_heads * hd
                                                     + out_head * hd + g_col)
                                    gate_f32 = _ld_global_b16_to_f32(
                                        gate_ptr + Int64(
                                            gate_elem_idx * Int32(2)))
                                    # sigmoid(x) = 1 / (1 + exp2(-x * LOG2E))
                                    neg_x_log2e = (Float32(0.0) - gate_f32
                                                   * Float32(1.4426950408889634))
                                    exp_val = exp2_approx_ftz_f32(neg_x_log2e)
                                    sigmoid_val = _rcp_approx_f32(
                                        Float32(1.0) + exp_val)
                                    o_final = o_final * sigmoid_val
                                output[out_idx] = BFloat16(o_final)

                cute.arch.sync_threads()
            # end _md loop

            # === Phase B: Fused W_O GEMV ===
            # After Phase A attention writes BF16 output to global memory,
            # Phase B reads it back (likely L2-cached) and multiplies by
            # the NVFP4 W_O weight matrix.  Each CTA handles one KV head
            # group's slice of the K dimension; partial products are
            # atomicAdd'd to a pre-zeroed FP32 output buffer.
            #
            # Thread tiling: 128 threads, each owns 40 output rows of
            # hidden_dim=5120 (5120/128=40).  Serialized over 5 groups
            # of 8 rows with explicit scalar accumulators (no arrays).
            if wo_fused != Int32(0):
                # wo_output is zeroed by Python (impl.wo_output.zero_())
                # before kernel launch — CUDA stream ordering guarantees
                # the memset completes before any CTA runs.
                # Self-zero inside the kernel is NOT safe: CTAs launch
                # in indeterminate order and a CTA's own slot must be
                # the zero baseline before it writes, not some other
                # CTA's slot mid-write.
                # 2026-04-20 deterministic-reduction fix: wo_output is
                # [num_seqs, total_ctas_per_seq, hd_wo] — each CTA
                # writes into its own `(bx, by)`-indexed slot (plain
                # FP32 stores, no atomicAdd). The cross-CTA reduction
                # becomes a deterministic gather inside the Phase C
                # last-CTA branch (Phase B.5 below) — see audit commit
                # 16475223f.

                # Attention output: [num_seqs, num_q_heads, head_dim] BF16
                # This CTA covers heads [q_head_start .. q_head_start+group_size-1]
                attn_base = seq_idx * num_q_heads * hd + q_head_start * hd

                # W_O column byte offset for this KV head group
                # Columns: [kv_head_idx * group_size * head_dim / 2 ...]
                wo_col_byte_off = kv_head_idx * group_size * hd // Int32(2)

                hd_wo = Int32(5120)  # hidden_dim (output dim)
                n_per_thr = Int32(40)  # outputs per thread (5120/128)
                my_row_base = tid * n_per_thr

                wo_gs = _ld_global_f32(wo_gs_ptr)

                # Serialize over 5 groups of 8 output rows
                for _out_group in cutlass.range_constexpr(5):
                    out_base = my_row_base + Int32(_out_group * 8)

                    # 8 FP32 accumulators
                    a0 = Float32(0.0)
                    a1 = Float32(0.0)
                    a2 = Float32(0.0)
                    a3 = Float32(0.0)
                    a4 = Float32(0.0)
                    a5 = Float32(0.0)
                    a6 = Float32(0.0)
                    a7 = Float32(0.0)

                    # Inner loop over K dimension (group_size * head_dim)
                    # Runtime while loop — constexpr would unroll 1536
                    # iterations × 8 accums × 5 groups = 61K ops, OOMs
                    k_dim = group_size * hd  # 6 * 256 = 1536
                    k_idx = Int32(0)
                    while k_idx < k_dim:
                        # Load attention output element (BF16 from global)
                        attn_val = Float32(output[attn_base + k_idx])

                        # Absolute K index in the full W_O matrix
                        abs_k = kv_head_idx * group_size * hd + k_idx
                        k_byte = abs_k >> Int32(1)  # which byte
                        k_is_hi = abs_k & Int32(1)  # 0=low nibble, 1=high

                        # Scale factor k_group (blockscale group = 16)
                        k_grp = abs_k >> Int32(4)  # abs_k / 16

                        # Process each of 8 output rows
                        for _oi in cutlass.range_constexpr(8):
                            out_row = out_base + Int32(_oi)
                            if out_row < hd_wo:
                                # Load weight byte via aligned b32
                                w_addr = wo_weight_ptr + Int64(
                                    out_row * wo_weight_row_stride
                                    + k_byte)
                                aligned = w_addr & Int64(
                                    0xFFFFFFFFFFFFFFFC)
                                raw = _ld_global_b32(aligned)
                                bpos = Int32(w_addr & Int64(3))
                                the_byte = _extract_byte_from_b32(
                                    raw, bpos)

                                # Extract nibble (branchless)
                                nib_shift = k_is_hi << Int32(2)  # 0 or 4
                                nib = (the_byte >> nib_shift) & Int32(
                                    0x0F)

                                w_f32 = _fp4_nibble_to_f32(nib)

                                # Load blockscale
                                sf = _ld_swizzled_scale(
                                    wo_scale_ptr, out_row, k_grp,
                                    wo_num_k_tiles)

                                w_dequant = w_f32 * sf * wo_gs

                                # Accumulate into correct register
                                if _oi == 0:
                                    a0 = a0 + w_dequant * attn_val
                                if _oi == 1:
                                    a1 = a1 + w_dequant * attn_val
                                if _oi == 2:
                                    a2 = a2 + w_dequant * attn_val
                                if _oi == 3:
                                    a3 = a3 + w_dequant * attn_val
                                if _oi == 4:
                                    a4 = a4 + w_dequant * attn_val
                                if _oi == 5:
                                    a5 = a5 + w_dequant * attn_val
                                if _oi == 6:
                                    a6 = a6 + w_dequant * attn_val
                                if _oi == 7:
                                    a7 = a7 + w_dequant * attn_val

                        k_idx = k_idx + Int32(1)

                    # Plain FP32 stores into this CTA's per-slot slice of
                    # wo_output. 2026-04-20 deterministic-reduction fix:
                    # wo_output is now shaped [num_seqs, total_ctas_per_seq,
                    # hd_wo] so every CTA owns a distinct slot keyed by
                    # `cta_idx = bx * num_kv_heads + by` (the flattened
                    # (num_q_tiles, num_kv_heads) grid index). Within a
                    # CTA all (tid, _out_group, _oi) rows are disjoint,
                    # so plain stores are race-free. The cross-CTA
                    # reduction is deferred to Phase B.5 (last-CTA
                    # gather) — see audit commit 16475223f.
                    cta_idx = bx * num_kv_heads + by
                    wo_slot_base = wo_output_ptr + Int64(
                        (seq_idx * total_ctas_per_seq + cta_idx)
                        * hd_wo * Int32(4))
                    for _oi in cutlass.range_constexpr(8):
                        out_row = out_base + Int32(_oi)
                        if out_row < hd_wo:
                            if _oi == 0:
                                _st_global_f32(
                                    wo_slot_base + Int64(
                                        out_row * Int32(4)), a0)
                            if _oi == 1:
                                _st_global_f32(
                                    wo_slot_base + Int64(
                                        out_row * Int32(4)), a1)
                            if _oi == 2:
                                _st_global_f32(
                                    wo_slot_base + Int64(
                                        out_row * Int32(4)), a2)
                            if _oi == 3:
                                _st_global_f32(
                                    wo_slot_base + Int64(
                                        out_row * Int32(4)), a3)
                            if _oi == 4:
                                _st_global_f32(
                                    wo_slot_base + Int64(
                                        out_row * Int32(4)), a4)
                            if _oi == 5:
                                _st_global_f32(
                                    wo_slot_base + Int64(
                                        out_row * Int32(4)), a5)
                            if _oi == 6:
                                _st_global_f32(
                                    wo_slot_base + Int64(
                                        out_row * Int32(4)), a6)
                            if _oi == 7:
                                _st_global_f32(
                                    wo_slot_base + Int64(
                                        out_row * Int32(4)), a7)

            # ═══════════════════════════════════════════════════
            # Phase B.5 + C: Gather + Residual Add + RMSNorm (last CTA only)
            # ═══════════════════════════════════════════════════
            # After Phase B's per-CTA plain stores, the last CTA to
            # arrive runs (a) a deterministic gather-collapse of all
            # slots into slot 0 — Phase B.5 — and (b) the fused residual
            # add + RMSNorm over slot 0 — Phase C. Other CTAs return
            # early.
            #
            # Game engine parallel: "deferred shading resolve" — after
            # tile-local light accumulation, one tile does the final
            # tonemap (RMSNorm) before writing to the framebuffer.
            #
            # NOTE: range_constexpr(5) assumes hidden_dim/128 = 40 and
            # group_size = 8, i.e. 5 groups of 8. This is correct for
            # Qwen3.5-27B (hidden_dim=5120). Models with different
            # hidden_dim would need this adjusted.
            if rmsnorm_fused != Int32(0):
                # Fence: ensure all Phase B per-CTA slot writes are
                # globally visible before the gather reads them.
                _threadfence()

                # Arrival counter: only thread 0 of each CTA bumps it,
                # then broadcasts "am I in the last-arriving CTA" via
                # SMEM to all 128 threads. Without this, every thread
                # atomicAdds (512 bumps per call instead of 4) and only
                # ONE thread matches old==total-1 → partial Phase C.
                if tid == Int32(0):
                    old_count = _atomic_add_u32(
                        arrival_count_ptr + Int64(seq_idx * Int32(4)),
                        Int32(1))
                    if old_count == total_ctas_per_seq - Int32(1):
                        _st_shared_f32(sync_md, Float32(1.0))
                    else:
                        _st_shared_f32(sync_md, Float32(0.0))
                cute.arch.sync_threads()

                is_last_cta = _ld_shared_f32(sync_md)

                if is_last_cta > Float32(0.5):
                    # I am in the last CTA — all Phase B writes are complete.

                    # Derive tiling from hidden_dim parameter (NOT hardcoded)
                    hd_c = hidden_dim
                    n_per_thr_c = hd_c // Int32(128)  # 40 for 5120

                    # Base pointers for this sequence.
                    # 2026-04-20 deterministic-reduction fix: wo_output
                    # is now [num_seqs, total_ctas_per_seq, hd_c]. Slot
                    # 0 holds the collapsed sum produced by Phase B.5
                    # (the gather loop below); Phase C reads from slot
                    # 0 — see audit commit 16475223f.
                    res_base = rmsnorm_residual_ptr + Int64(
                        seq_idx * hd_c * Int32(2))  # BF16 = 2 bytes
                    wo_base_c = wo_output_ptr + Int64(
                        seq_idx * total_ctas_per_seq
                        * hd_c * Int32(4))  # FP32 = 4 bytes; slot 0
                    gamma_base = rmsnorm_gamma_ptr  # [hidden_dim] BF16
                    out_base_c = rmsnorm_output_ptr + Int64(
                        seq_idx * hd_c * Int32(2))  # BF16 output
                    resout_base = residual_output_ptr + Int64(
                        seq_idx * hd_c * Int32(2))  # BF16 output

                    my_start_c = tid * n_per_thr_c

                    # ── Phase B.5: gather per-CTA slots into slot 0 ──
                    # All 128 threads participate; each owns 40 rows
                    # (n_per_thr_c). For each row, sum across CTA slots
                    # (`cta_i = 0 .. total_ctas_per_seq - 1`) in index
                    # order and write back to slot 0. Sum order is
                    # fixed, so the reduction is bit-identical across
                    # runs (fixes the pre-2026-04-20 cross-CTA
                    # atomicAdd_f32 non-determinism). Subsequent Phase
                    # C passes 1 and 3 read wo_base_c (= slot 0) as
                    # before.
                    for _grp in cutlass.range_constexpr(5):
                        for _ei in cutlass.range_constexpr(8):
                            idx_c = my_start_c + Int32(_grp * 8 + _ei)
                            # my_start_c <= 127*40 = 5080, _grp*8+_ei
                            # <= 39, idx_c <= 5119 < hd_c=5120 — no
                            # bounds check needed for Qwen3.5-27B.
                            gather_acc = Float32(0.0)
                            cta_i = Int32(0)
                            while cta_i < total_ctas_per_seq:
                                slot_addr = wo_output_ptr + Int64(
                                    (seq_idx * total_ctas_per_seq + cta_i)
                                    * hd_c * Int32(4)
                                    + idx_c * Int32(4))
                                gather_acc = gather_acc + _ld_global_f32(
                                    slot_addr)
                                cta_i = cta_i + Int32(1)
                            _st_global_f32(
                                wo_base_c + Int64(idx_c * Int32(4)),
                                gather_acc,
                            )
                    # Fence so Phase C's reads see the collapsed slot-0
                    # writes (cross-warp within this CTA).
                    _threadfence()
                    cute.arch.sync_threads()

                    # ── Pass 1: Residual add + sum-of-squares ──
                    # Re-reads from global in Pass 3 — NO SMEM staging.
                    # Writing 5 groups to same 8 SMEM slots would
                    # overwrite groups 0-3 with group 4.
                    ss = Float32(0.0)

                    for _grp in cutlass.range_constexpr(5):
                        base_idx = my_start_c + Int32(_grp * 8)

                        for _ei in cutlass.range_constexpr(8):
                            idx_c = base_idx + Int32(_ei)
                            # Load residual (BF16 → FP32) from global
                            res_f32 = _ld_global_b16_to_f32(
                                res_base + Int64(idx_c * Int32(2)))
                            # Load wo_output (FP32) from global
                            wo_f32 = _ld_global_f32(
                                wo_base_c + Int64(idx_c * Int32(4)))
                            nr = res_f32 + wo_f32
                            ss = ss + nr * nr

                    # ── Pass 2: Reduction (warp shuffle + cross-warp SMEM) ──
                    # Intra-warp butterfly reduction (5 steps for 32 lanes)
                    ss = ss + shfl_xor_sync(ss, Int32(1))
                    ss = ss + shfl_xor_sync(ss, Int32(2))
                    ss = ss + shfl_xor_sync(ss, Int32(4))
                    ss = ss + shfl_xor_sync(ss, Int32(8))
                    ss = ss + shfl_xor_sync(ss, Int32(16))

                    # Cross-warp: lane 0 of each warp writes to SMEM
                    # Reuse sync_md buffer (4 FP32 slots = 16 bytes)
                    if lane == Int32(0):
                        _st_shared_f32(
                            sync_md + Int64(warp * Int32(4)), ss)
                    cute.arch.sync_threads()

                    # Warp 0, lane 0: sum all 4 warp partials
                    if warp == Int32(0):
                        if lane == Int32(0):
                            total_ss = _ld_shared_f32(sync_md)
                            total_ss = total_ss + _ld_shared_f32(
                                sync_md + Int64(4))
                            total_ss = total_ss + _ld_shared_f32(
                                sync_md + Int64(8))
                            total_ss = total_ss + _ld_shared_f32(
                                sync_md + Int64(12))
                            # variance = total / hidden_dim
                            variance = total_ss / Float32(hd_c)
                            inv_rms = _rsqrt_approx_f32(
                                variance + Float32(rmsnorm_eps))
                            # Broadcast inv_rms via SMEM slot 0
                            _st_shared_f32(sync_md, inv_rms)
                    cute.arch.sync_threads()

                    # All threads read the broadcast inv_rms
                    inv_rms_val = _ld_shared_f32(sync_md)

                    # ── Pass 3: Re-read from global (L2-hot), scale + write ──
                    # Both residual and wo_output are L2-hot: Phase B
                    # just wrote wo_output, Pass 1 just read residual.
                    for _grp in cutlass.range_constexpr(5):
                        base_idx = my_start_c + Int32(_grp * 8)

                        for _oi in cutlass.range_constexpr(8):
                            idx_c = base_idx + Int32(_oi)
                            # Re-read and recompute new_residual (L2-hot)
                            res_f32 = _ld_global_b16_to_f32(
                                res_base + Int64(idx_c * Int32(2)))
                            wo_f32 = _ld_global_f32(
                                wo_base_c + Int64(idx_c * Int32(4)))
                            new_res = res_f32 + wo_f32

                            # Load gamma (BF16 → FP32)
                            gamma_f32 = _ld_global_b16_to_f32(
                                gamma_base + Int64(idx_c * Int32(2)))

                            # hidden = new_res * inv_rms * (1 + gamma)
                            # Qwen3_5RMSNorm uses x * (1 + γ) — see vllm/nvllm/layers/layernorm.py:78
                            hidden_val = new_res * inv_rms_val * (Float32(1.0) + gamma_f32)

                            # Write hidden_states (→BF16) to output
                            _st_global_bf16_from_f32(
                                out_base_c + Int64(idx_c * Int32(2)),
                                hidden_val)
                            # Write new_residual (→BF16) to residual output
                            _st_global_bf16_from_f32(
                                resout_base + Int64(idx_c * Int32(2)),
                                new_res)

                    # Self-reset arrival counter: atomicAdd -N to reset to 0.
                    # Eliminates need for caller zero_() per launch.
                    # (Still zero-init at allocation as safety net.)
                    if tid == Int32(0):
                        _atomic_add_u32(
                            arrival_count_ptr + Int64(
                                seq_idx * Int32(4)),
                            Int32(0) - total_ctas_per_seq)

        def __call__(self, **kwargs):
            """Python-level wrapper: compute grid/block and launch."""
            query = kwargs["query"]
            kv_cache = kwargs["kv_cache"]
            page_table = kwargs["page_table"]
            seq_lens = kwargs["seq_lens"]
            scale = kwargs["scale"]
            k_scale = kwargs["k_scale"]
            v_scale = kwargs["v_scale"]

            # Phase B: W_O fusion params (optional)
            wo_weight = kwargs.get("wo_weight", None)       # [N, K/2] uint8
            wo_scales = kwargs.get("wo_scales", None)        # [N, K_sf] fp8
            wo_global_scale = kwargs.get("wo_global_scale", None)  # scalar
            wo_output = kwargs.get("wo_output", None)        # [num_seqs, total_ctas_per_seq, N] f32 (deterministic stage + gather; slot 0 holds final sum)

            # Phase B's `range_constexpr(5)` loop + `hd_wo = Int32(5120)` +
            # `n_per_thr = Int32(40)` (kernel body below) hard-code the
            # Qwen3.5-27B hidden size. Fail loud rather than silently
            # mis-write when a different hidden size lands here.
            if wo_weight is not None:
                wo_hidden_dim = int(wo_weight.shape[0])
                if wo_hidden_dim != 5120:
                    raise AssertionError(
                        f"CuTe decode W_O fusion is hard-coded to "
                        f"hidden_dim=5120 (Qwen3.5/3.6-27B); got "
                        f"wo_weight.shape[0]={wo_hidden_dim}. Disable "
                        f"CUTE_ATTN_FUSION or rebuild the kernel."
                    )

            # Phase C: RMSNorm fusion params (optional)
            rmsnorm_gamma = kwargs.get("rmsnorm_gamma", None)       # [hidden_dim] BF16
            rmsnorm_residual = kwargs.get("rmsnorm_residual", None) # [num_seqs, hidden_dim] BF16
            rmsnorm_output = kwargs.get("rmsnorm_output", None)     # [num_seqs, hidden_dim] BF16
            residual_output = kwargs.get("residual_output", None)   # [num_seqs, hidden_dim] BF16
            arrival_count = kwargs.get("arrival_count", None)       # [num_seqs] int32
            rmsnorm_eps = kwargs.get("rmsnorm_eps", None)           # float (from config)

            num_q_heads = query.shape[1]
            # kv_cache: [num_pages, 2, page_size, num_kv_heads, head_dim]
            num_kv_heads = kv_cache.shape[3]
            group_size = num_q_heads // num_kv_heads
            num_seqs = len(seq_lens)

            # Unified base pointer + K/V byte offsets (atlas layer offsets)
            kv_base = Int64(kv_cache.data_ptr())
            kv_slot_stride = Int64(
                kv_cache.stride(1) * kv_cache.element_size()
            )
            k_ptr = kv_base                    # K is slot 0
            v_ptr = kv_base + kv_slot_stride   # V is slot 1

            # 5D page stride: stride(0) in bytes
            kv_page_stride = Int32(
                kv_cache.stride(0) * kv_cache.element_size()
            )

            # Flatten query to 1D — CuTe DSL does NOT support flat
            # element indexing on multi-dimensional tensors.  The
            # kernel computes gmem_idx as a manual flat offset, so
            # the tensor must be 1D for query[gmem_idx] to work.
            assert query.is_contiguous(), "CuTe decode kernel requires contiguous query tensor"
            q_flat = query.view(-1)

            # Grid: (ceil(group_size / cta_q), num_kv_heads, num_seqs)
            num_q_tiles = max(
                (group_size + self.cta_q - 1) // self.cta_q, 1,
            )
            padded_num_seqs = kwargs.get("padded_num_seqs", num_seqs)
            grid = (num_q_tiles, num_kv_heads, padded_num_seqs)

            output = kwargs.get("output_buf", None)
            if output is None:
                output = torch.empty_like(query)
            # Output also needs 1D for the same reason
            out_flat = output.view(-1)

            # Phase B: build W_O pointer args — real values when fusion
            # is enabled, zeros when not (kernel guards on wo_fused_flag)
            if wo_weight is not None:
                wo_weight_ptr = Int64(wo_weight.data_ptr())
                wo_scale_ptr = Int64(wo_scales.data_ptr())
                wo_output_ptr = Int64(wo_output.data_ptr())
                wo_gs_ptr = Int64(wo_global_scale.data_ptr())
                wo_K = num_q_heads * self.head_dim  # 6144 for Qwen 27B
                wo_nkt = Int32((wo_K // 16 + 3) // 4)  # numKTiles swizzle
                wo_row_stride = Int32(wo_weight.shape[1])  # K/2 bytes/row
                wo_fused_flag = Int32(1)
            else:
                wo_weight_ptr = Int64(0)
                wo_scale_ptr = Int64(0)
                wo_output_ptr = Int64(0)
                wo_gs_ptr = Int64(0)
                wo_nkt = Int32(0)
                wo_row_stride = Int32(0)
                wo_fused_flag = Int32(0)

            # Phase C: build RMSNorm pointer args — real values when fusion
            # is enabled, zeros when not (kernel guards on rmsnorm_fused_flag).
            # hidden_dim derived from gamma tensor shape, NOT hardcoded.
            # total_ctas derived from grid dimensions, NOT hardcoded.
            if rmsnorm_gamma is not None:
                rmsnorm_gamma_ptr = Int64(rmsnorm_gamma.data_ptr())
                rmsnorm_residual_ptr = Int64(rmsnorm_residual.data_ptr())
                rmsnorm_output_ptr = Int64(rmsnorm_output.data_ptr())
                residual_output_ptr = Int64(residual_output.data_ptr())
                arrival_count_ptr = Int64(arrival_count.data_ptr())
                rmsnorm_eps_val = float(rmsnorm_eps) if rmsnorm_eps is not None else 1e-6
                hidden_dim_val = Int32(rmsnorm_gamma.shape[0])
                total_ctas = Int32(grid[0] * grid[1])
                rmsnorm_fused_flag = Int32(1)
            else:
                rmsnorm_gamma_ptr = Int64(0)
                rmsnorm_residual_ptr = Int64(0)
                rmsnorm_output_ptr = Int64(0)
                residual_output_ptr = Int64(0)
                arrival_count_ptr = Int64(0)
                rmsnorm_eps_val = 0.0
                hidden_dim_val = Int32(0)
                total_ctas = Int32(0)
                rmsnorm_fused_flag = Int32(0)

            # Output gate: pointer for sigmoid fusion
            gate_buf = kwargs.get("gate_buf", None)
            if gate_buf is not None:
                gate_ptr = Int64(gate_buf.data_ptr())
                gate_fused_flag = Int32(1)
            else:
                gate_ptr = Int64(0)
                gate_fused_flag = Int32(0)

            # Thread the current torch CUDA stream into the CuTe launch so
            # kernel launches participate in any active CUDA graph capture.
            # See stream_adapter.py: cuda.CUstream -> gpu.AsyncTokenType,
            # consumed by gpu.launch_func as its async_deps / launch stream.
            stream_arg = _cuda_driver.CUstream(
                int(torch.cuda.current_stream().cuda_stream)
            )

            all_args = (
                q_flat, k_ptr, v_ptr, page_table, seq_lens,
                out_flat,
                float(scale), float(k_scale), float(v_scale),
                Int32(num_q_heads), Int32(num_kv_heads),
                kv_page_stride,
                wo_weight_ptr, wo_scale_ptr, wo_output_ptr,
                wo_gs_ptr, wo_nkt, wo_row_stride,
                wo_fused_flag,
                rmsnorm_gamma_ptr, rmsnorm_residual_ptr,
                rmsnorm_output_ptr, residual_output_ptr,
                arrival_count_ptr, rmsnorm_eps_val,
                hidden_dim_val, total_ctas,
                rmsnorm_fused_flag,
                gate_ptr, gate_fused_flag,
                Int32(grid[0]), Int32(grid[1]), Int32(grid[2]),
                stream_arg,
            )

            if self._compiled is None:
                logger.info("Compiling CuTe decode kernel (first call)...")
                self._compiled = cute.compile(
                    self._jit_launch, *all_args)

            self._compiled(*all_args)
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
            self.head_dim = config.head_dim      # 256
            self.block_size = config.block_size  # 64
            self.num_warps_q = config.num_warps_q    # 4
            self.num_warps_kv = config.num_warps_kv  # 1
            self.num_threads = 128
            self.num_mma_d = self.head_dim // 16  # 16 for head_dim=256

            # SMEM sizes (bytes) -- no cta_sync for prefill
            self.q_bytes = self.cta_q * self.head_dim * 2      # BF16
            self.k_bytes = self.cta_kv * self.head_dim * 1     # FP8
            self.v_bytes = self.cta_kv * self.head_dim * 1     # FP8
            self.smem_bytes = self.q_bytes + self.k_bytes + self.v_bytes
            self._compiled = None

        @cute.jit
        def _jit_launch(self, query, k_cache, v_cache, page_table,
                        seq_lens, query_start_loc, output,
                        scale, k_scale, v_scale,
                        num_q_heads, num_kv_heads,
                        grid_x: Int32, grid_y: Int32, grid_z: Int32,
                        stream):
            """JIT host wrapper: compiles kernel launch into MLIR.

            stream: cuda.CUstream — required for CUDA graph capture to see
            the launch. Without it the prefill kernel would launch on CuTe's
            internal default stream, invisible to torch graph capture. Same
            fix as DecodeKernel; see that method for the full rationale.
            """
            self._kernel(
                query, k_cache, v_cache, page_table, seq_lens,
                query_start_loc, output, scale, k_scale, v_scale,
                num_q_heads, num_kv_heads,
            ).launch(
                grid=[grid_x, grid_y, grid_z],
                block=[self.num_threads, 1, 1],
                smem=self.smem_bytes,
                stream=stream,
            )

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

            # Thread current torch CUDA stream into the CuTe launch so the
            # kernel participates in any active CUDA graph capture. See
            # DecodeKernel for full rationale.
            stream_arg = _cuda_driver.CUstream(
                int(torch.cuda.current_stream().cuda_stream)
            )

            if self._compiled is None:
                logger.info("Compiling CuTe prefill kernel (first call)...")
                self._compiled = cute.compile(
                    self._jit_launch,
                    query, k_cache, v_cache, page_table, seq_lens,
                    query_start_loc, output,
                    float(scale), float(k_scale), float(v_scale),
                    Int32(num_q_heads), Int32(num_kv_heads),
                    Int32(grid[0]), Int32(grid[1]), Int32(grid[2]),
                    stream_arg,
                )

            self._compiled(
                query, k_cache, v_cache, page_table, seq_lens,
                query_start_loc, output,
                float(scale), float(k_scale), float(v_scale),
                Int32(num_q_heads), Int32(num_kv_heads),
                Int32(grid[0]), Int32(grid[1]), Int32(grid[2]),
                stream_arg,
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
    kernel = DecodeKernel(config) if config.cta_q <= 16 else PrefillKernel(config)
    return kernel


# --- Public entry point -----------------------------------------------------

def paged_attention_forward(
    query: torch.Tensor,        # [num_tokens, num_q_heads, head_dim] BF16
    kv_cache: torch.Tensor,     # [num_pages, 2, page_size, num_kv_heads, head_dim] uint8
    page_table: torch.Tensor,   # [num_seqs, max_pages_per_seq] int32
    seq_lens: torch.Tensor,     # [num_seqs] int32
    scale: float,
    k_scale: float = 1.0,
    v_scale: float = 1.0,
    page_size: int = 64,
    query_start_loc: torch.Tensor | None = None,  # [num_seqs + 1] int32
    # Phase B: W_O fusion (optional)
    wo_weight: torch.Tensor | None = None,       # [N, K/2] uint8
    wo_scales: torch.Tensor | None = None,        # [N, K_sf] fp8_e4m3fn
    wo_global_scale: torch.Tensor | None = None,  # scalar f32
    wo_output: torch.Tensor | None = None,        # [num_seqs, total_ctas_per_seq, N] f32 (deterministic stage + gather; slot 0 holds final sum)
    # Phase C: RMSNorm fusion (optional)
    rmsnorm_gamma: torch.Tensor | None = None,       # [hidden_dim] BF16
    rmsnorm_residual: torch.Tensor | None = None,    # [num_seqs, hidden_dim] BF16
    rmsnorm_output: torch.Tensor | None = None,      # [num_seqs, hidden_dim] BF16
    residual_output: torch.Tensor | None = None,     # [num_seqs, hidden_dim] BF16
    arrival_count: torch.Tensor | None = None,       # [num_seqs] int32
    rmsnorm_eps: float | None = None,                 # from config.rms_norm_eps
    # CUDA graph support (optional)
    gate_buf: torch.Tensor | None = None,            # [num_seqs, num_q_heads, head_dim] BF16
    padded_num_seqs: int | None = None,               # stable grid dim for graph capture
    # Caller-supplied persistent output buffer. When non-None the CuTe
    # decode kernel writes directly into it (no torch.empty_like per call,
    # no post-return copy_). Must match query's shape and dtype.
    output_buf: torch.Tensor | None = None,           # [num_tokens, num_q_heads, head_dim] BF16
) -> torch.Tensor:
    """Paged attention forward -- CuTe JIT kernel or PyTorch fallback.

    kv_cache is the unified 5D tensor [num_pages, 2, page_size, num_kv_heads, head_dim].
    Dim 1: 0=K, 1=V. The kernel computes K/V base pointers from stride(1).

    Returns: [num_tokens, num_q_heads, head_dim] BF16. When ``output_buf``
    is provided, the returned tensor IS ``output_buf`` (no copy).
    """
    if not _CUTE_AVAILABLE or not _KERNELS_IMPLEMENTED:
        # Fallback to PyTorch reference until CuTe kernel bodies are written
        from vllm.v1.attention.backends.cute_paged._pytorch_reference import (
            reference_paged_attention,
        )
        ref = reference_paged_attention(
            query, kv_cache[:, 0].contiguous(), kv_cache[:, 1].contiguous(),
            page_table, seq_lens,
            scale=scale, k_scale=k_scale, v_scale=v_scale,
            page_size=page_size, query_start_loc=query_start_loc,
        )
        if output_buf is not None:
            output_buf.copy_(ref)
            return output_buf
        return ref

    if page_size != 64:
        raise ValueError(
            f"CuTe paged attention requires page_size=64, got {page_size}"
        )

    # Select config: decode = one query token per sequence
    num_tokens = query.shape[0]
    num_seqs = len(seq_lens)
    is_decode = num_tokens == num_seqs

    if not is_decode:
        # Prefill: use PyTorch reference until CuTe prefill body is written
        from vllm.v1.attention.backends.cute_paged._pytorch_reference import (
            reference_paged_attention,
        )
        ref = reference_paged_attention(
            query, kv_cache[:, 0].contiguous(), kv_cache[:, 1].contiguous(),
            page_table, seq_lens,
            scale=scale, k_scale=k_scale, v_scale=v_scale,
            page_size=page_size, query_start_loc=query_start_loc,
        )
        if output_buf is not None:
            output_buf.copy_(ref)
            return output_buf
        return ref

    config = DECODE_CONFIG
    kernel = _get_compiled_kernel(config)

    cute_out = kernel(
        query=query,
        kv_cache=kv_cache,
        page_table=page_table,
        seq_lens=seq_lens,
        scale=scale,
        k_scale=k_scale,
        v_scale=v_scale,
        page_size=page_size,
        query_start_loc=query_start_loc,
        wo_weight=wo_weight,
        wo_scales=wo_scales,
        wo_global_scale=wo_global_scale,
        wo_output=wo_output,
        rmsnorm_gamma=rmsnorm_gamma,
        rmsnorm_residual=rmsnorm_residual,
        rmsnorm_output=rmsnorm_output,
        residual_output=residual_output,
        arrival_count=arrival_count,
        rmsnorm_eps=rmsnorm_eps,
        gate_buf=gate_buf,
        output_buf=output_buf,
        padded_num_seqs=padded_num_seqs,
    )

    return cute_out
