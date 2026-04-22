# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright contributors to the nvllm fork
"""Phase D fused MLP decode kernel — CuTe DSL.

PHASE 3b SCOPE: end-to-end fused MLP for nat<=1.
  - Input:
        x [nat, hidden] BF16,
        gate_w_fp4 [interm, hidden/2] u8 (2 FP4 nibbles/byte along hidden),
        gate_w_scale [interm, hidden/FP4_BLOCK_SIZE] u8 (UE4M3 per block),
        up_w_fp4, up_w_scale    (same layout as gate_w_*),
        down_w_fp4  [hidden, interm/2] u8,
        down_w_scale [hidden, interm/FP4_BLOCK_SIZE] u8,
        mlp_partial_fp32 [nat, slice_ctas, hidden] FP32 (zeroed by caller),
        mlp_arrival_count [nat, num_k_tiles] u32 (zeroed by caller),
        mlp_output [nat, hidden] BF16 (written by the last CTA only).

  - Pipeline (per CTA, grid (slice_ctas, num_k_tiles, nat), 128 threads):
        1. Stage x into SMEM FP32.
        2. For each slice s owned by this slice-group (bx):
              a. FC1: compute intermediate_bf16[tile_s] = silu(gate_j)*up_j
                 by dot-product over the full hidden dim. FP4 gate_w/up_w
                 are dequantized on-the-fly (nibble + blockscale).
              b. FP4 quantize intermediate into
                 (smem_intermediate_fp4 [tile_s/2], smem_intermediate_scale
                  [tile_s/FP4_BLOCK_SIZE]).
              c. FC2 (for this CTA's k_tile=by, k rows [by*tile_k,
                 (by+1)*tile_k)): compute partial_k = dot(intermediate_slice,
                 down_w[k, s_start + :tile_s]) and atomicAdd into
                 mlp_partial_fp32[token, bx, k] — per-CTA slot keyed by
                 the slice-group index `bx`, so cross-CTA reduction is
                 deferred to the last-CTA gather (step 4). Within a CTA
                 the atomicAdd target is unique per thread, so per-s
                 accumulation is deterministic (program order).
        3. __threadfence(); thread 0 atomicInc(arrival_count[token, by]).
        4. If this was the last-arriving CTA (old == slice_ctas-1),
           cooperatively read mlp_partial_fp32[token, bx, k:k+tile_k]
           for all bx in [0, slice_ctas), sum in constexpr bx order,
           cast to BF16, write to mlp_output. Gather order is fixed
           so the reduction is bit-identical across runs (fix for the
           pre-2026-04-20 non-deterministic cross-CTA atomicAdd — see
           audit commit 16475223f).

Spec: docs/superpowers/specs/2026-04-17-unreal-kernel-phase-d-mlp-fusion-design.md

Deviations from the full-sized plan (documented):
  - This first Phase-3b path targets the SMALL test config
    (hidden=128, interm=128, tile_s=64, tile_k=32, slice_ctas=2). The
    FC1/FC2 loops operate directly on registers (no cp_async H_CHUNK
    streaming) and use simple linear blockscale layouts (no swizzle).
  - FC2 output-row thread mapping: 4 threads per output row (128
    threads / tile_k=32). Each thread sums tile_s/4=16 elements.
  - The arrival-counter epilogue writes BF16 via the per-lane path
    (one thread per output element), matching the simple register
    layout above.
  - `CUTE_DEBUG_MLP_FUSION=1` env flag is wired as a Python-side
    switch that skips the FP4 quantize and writes BF16 intermediate
    directly into SMEM (debug path to isolate quant from FC2 math).
"""

from __future__ import annotations

import logging
import os
from typing import Optional

import torch

logger = logging.getLogger(__name__)

# Kernel tile constants (see spec §Target Dimensions).
H_CHUNK = 128
FP4_BLOCK_SIZE = 16
LOG2_E = 1.4426950408889634

# ---------------------------------------------------------------------------
# Phase D3a tile-preset registry.
# See docs/superpowers/specs/2026-04-19-phase-d3a-mlp-decode-retune-design.md
# for rationale per preset. `CUTE_MLP_TILE` env var picks one at kernel
# construct time; unset/empty → `_DEFAULT_PRESET_NAME`. Unknown name →
# ValueError at construct time (intentional: sweep runs must never silently
# use the wrong preset).
# ---------------------------------------------------------------------------
_TILE_PRESETS: dict[str, tuple[int, int, int]] = {
    # name              : (tile_s, tile_k, slice_ctas)
    "prefill-legacy":     (256, 640, 8),     # baseline; preserved verbatim
    "decode-balanced":    (128, 640, 16),    # half tile_s, 2× CTAs
    "decode-small":       (64,  640, 32),    # quarter tile_s, 4× CTAs
    "decode-narrow-grid": (256, 1280, 8),    # same tile_s, 2× tile_k → halve num_k_tiles
}

_DEFAULT_PRESET_NAME: str = "prefill-legacy"


def _resolve_tile_preset(name: Optional[str]) -> tuple[int, int, int]:
    """Return (tile_s, tile_k, slice_ctas) for the given preset name.

    `None` or empty → the default preset. Unknown name → ValueError with
    the full list of valid preset names in the message.
    """
    key = name if name else _DEFAULT_PRESET_NAME
    if key not in _TILE_PRESETS:
        valid = sorted(_TILE_PRESETS)
        raise ValueError(
            f"Unknown CUTE_MLP_TILE={name!r}; valid: {valid}"
        )
    return _TILE_PRESETS[key]


TILE_S_DEFAULT, TILE_K_DEFAULT, SLICE_CTAS_DEFAULT = _TILE_PRESETS[_DEFAULT_PRESET_NAME]

# Phase 3b debug switch: when set, host-side code may construct a kernel
# where the intermediate is passed BF16 (not FP4) and the quant stage is
# skipped. Exposed as a Python-side toggle so it can be flipped during
# investigation without recompiling.
DEBUG_SKIP_FP4_QUANT = (
    os.environ.get("CUTE_DEBUG_MLP_FUSION", "").lower()
    in ("1", "true", "yes")
)

# --- CuTe DSL import guard (mirrors kernel.py) -----------------------------
_CUTE_AVAILABLE = False
try:
    import cutlass
    from cutlass import cute
    from cutlass._mlir import ir as _mlir_ir  # noqa: F401
    from cutlass._mlir.dialects import llvm as _llvm_dialect
    from cutlass.cute.typing import (  # noqa: F401
        BFloat16,
        Float32,
        Int32,
        Int64,
        Uint32,
    )
    from cutlass.cutlass_dsl import T, dsl_user_op
    import cuda.bindings.driver as _cuda_driver

    # Import the shared PTX helpers that already exist in the attention kernel
    # module so we do NOT reimplement them here. They are module-level objects
    # once kernel.py is imported.
    from vllm.v1.attention.backends.cute_paged.kernel import (  # noqa: E501
        _atomic_add_f32,
        _atomic_add_u32,
        _bitcast_i32_to_f32,
        _cvt_2f32_to_bf16x2,
        _extract_byte_from_b32,
        _fp4_nibble_to_f32,
        _ld_global_f32,
        _ld_global_b16_to_f32,
        _ld_global_b32,
        _ld_shared_b16,
        _ld_shared_b32,
        _ld_shared_f32,
        _ld_shared_u8,
        _rcp_approx_f32,
        _rsqrt_approx_f32,
        _st_global_bf16_from_f32,
        _st_global_f32,
        _st_shared_b16_from_u32,
        _st_shared_b32,
        _st_shared_f32,
        _threadfence,
        exp2_approx_ftz_f32,
        fp8x4_e4m3_to_bfloat2x2,
        shared_ptr_to_i64,
        shfl_xor_sync,
    )
    from vllm.v1.attention.backends.cute_paged._fp4_writer import (  # noqa: E501
        _encode_ue4m3_f32_to_u8,
        _f32_to_fp4_nibble,
        _st_shared_u8,
    )

    _CUTE_AVAILABLE = True
except ImportError:
    logger.warning(
        "CuTe DSL not available (CUTLASS not installed). "
        "Phase_D_MLP_Kernel cannot be used."
    )


# --- Phase 3b: helpers that weren't already in kernel.py ------------------

if _CUTE_AVAILABLE:

    @dsl_user_op
    def _rcp_ieee_f32(x: Float32, *, loc=None, ip=None) -> Float32:
        """IEEE round-to-nearest reciprocal — rcp.rn.f32.

        Unlike ``_rcp_approx_f32`` (from kernel.py) this is IEEE-correct
        to the last bit. Used when the downstream op (FP4 e2m1 rounding)
        is sensitive to exact-midpoint cases; ``rcp.approx`` can produce
        a scaled value off the true midpoint by a few ULPs and push
        tie-break rounding in the opposite direction from the reference.
        """
        x_ir = Float32(x).ir_value(loc=loc, ip=ip)
        result_ir = _llvm_dialect.inline_asm(
            T.f32(), [x_ir],
            "rcp.rn.f32 $0, $1;", "=f,f",
            has_side_effects=True, loc=loc, ip=ip,
        )
        return Float32(result_ir)

    @dsl_user_op
    def _div_rn_f32(a: Float32, b: Float32, *, loc=None, ip=None) -> Float32:
        """IEEE round-to-nearest divide — div.rn.f32.

        One-instruction IEEE division. Matches ``torch.div``/``/`` on CPU
        bitwise. Used in the FP4 quant path where ``rcp + mul`` can deliver
        a result off by 1 ULP from a direct divide, flipping FP4 tie-break
        rounding away from the reference.
        """
        a_ir = Float32(a).ir_value(loc=loc, ip=ip)
        b_ir = Float32(b).ir_value(loc=loc, ip=ip)
        result_ir = _llvm_dialect.inline_asm(
            T.f32(), [a_ir, b_ir],
            "div.rn.f32 $0, $1, $2;", "=f,f,f",
            has_side_effects=True, loc=loc, ip=ip,
        )
        return Float32(result_ir)

    @dsl_user_op
    def _f32_div_to_fp4_nibble(
        value: Float32, scale: Float32, *, loc=None, ip=None,
    ) -> Int32:
        """IEEE ``value/scale`` → FP4 E2M1 nibble (low 4 bits of result).

        Equivalent to ``_f32_to_fp4_nibble(value, 1/scale)`` but uses a
        single ``div.rn.f32`` followed by ``cvt.rn.satfinite.e2m1x2.f32``
        — matches reference quantizer's FP32 divide-then-round behavior.

        Operand order: ``cvt byte, hi_src, lo_src`` puts lo_src in bits
        [3:0]. We place the real value as ``lo_src`` and a zero sentinel
        as ``hi_src``.
        """
        value_ir = Float32(value).ir_value(loc=loc, ip=ip)
        scale_ir = Float32(scale).ir_value(loc=loc, ip=ip)
        result_ir = _llvm_dialect.inline_asm(
            T.i32(),
            [value_ir, scale_ir],
            "{ .reg .f32 scaled, zero; .reg .b8 byte;"
            " div.rn.f32 scaled, $1, $2;"
            " mov.f32 zero, 0f00000000;"
            " cvt.rn.satfinite.e2m1x2.f32 byte, zero, scaled;"
            " cvt.u32.u8 $0, byte; and.b32 $0, $0, 0xF; }",
            "=r,f,f",
            has_side_effects=False,
            asm_dialect=0,
            loc=loc, ip=ip,
        )
        return Int32(result_ir)

    @dsl_user_op
    def _ld_global_u8(addr: Int64, *, loc=None, ip=None) -> Uint32:
        """Load 1 byte from global memory, zero-extend to Uint32.

        Mirrors _ld_shared_u8 but from the .global space. Used for per-byte
        FP4 packed weight loads where 4-byte alignment isn't guaranteed.
        """
        addr_ir = Int64(addr).ir_value(loc=loc, ip=ip)
        result_ir = _llvm_dialect.inline_asm(
            T.i32(), [addr_ir],
            "{ .reg .b32 tmp; ld.global.b8 tmp, [$1];"
            " and.b32 $0, tmp, 0xFF; }",
            "=r,l",
            has_side_effects=True,
            asm_dialect=0,
            loc=loc, ip=ip,
        )
        return Uint32(result_ir)

    @cute.jit
    def _decode_ue4m3_u8_to_f32(code: Uint32):
        """UE4M3 byte → FP32 scale (pure DSL math).

        Inverse of _fp4_writer.encode_ue4m3. Uses the existing
        fp8x4_e4m3_to_bfloat2x2 path by placing `code` in the low byte
        of a u32 and extracting the low BF16, then shifting back to FP32.
        """
        b0 = code & Uint32(0xFF)
        packed = b0 | (b0 << Uint32(8)) | (b0 << Uint32(16)) | (b0 << Uint32(24))
        bf16_lo, _bf16_hi = fp8x4_e4m3_to_bfloat2x2(packed)
        low16 = bf16_lo & Uint32(0xFFFF)
        as_f32_bits = Int32(low16 << Uint32(16))
        return _bitcast_i32_to_f32(as_f32_bits)



class Phase_D_MLP_Kernel:
    """CuTe DSL compiled kernel for fused MLP decode — Phase 3b.

    Call signature for Phase 3b:
        kernel(
            x_bf16, gate_w_fp4, gate_w_scale, up_w_fp4, up_w_scale,
            down_w_fp4, down_w_scale, mlp_partial_fp32,
            mlp_arrival_count, mlp_output, nat,
        )

    All global buffers are expected to be contiguous. Layout conventions:
      - x: [nat, hidden] BF16
      - gate_w_fp4 / up_w_fp4: [interm, hidden // 2] u8
      - gate_w_scale / up_w_scale: [interm, hidden // FP4_BLOCK_SIZE] u8
      - down_w_fp4: [hidden, interm // 2] u8
      - down_w_scale: [hidden, interm // FP4_BLOCK_SIZE] u8
      - mlp_partial_fp32: [nat, slice_ctas, hidden] FP32, zeroed by caller
        (slice_ctas-wide staging; each CTA atomic-adds into its own
         `bx` slot; last-CTA epilogue gathers slots in constexpr order
         for deterministic reduction — audit 16475223f)
      - mlp_arrival_count: [nat, num_k_tiles] u32, zeroed by caller
      - mlp_output: [nat, hidden] BF16, written by last CTA only
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        tile_s: Optional[int] = None,
        tile_k: Optional[int] = None,
        slice_ctas: Optional[int] = None,
    ):
        # Resolve tile constants from CUTE_MLP_TILE env var, with per-kwarg
        # override. Passing all three explicitly bypasses the env read (tests
        # and microbenches). Passing a subset fills remaining from the preset.
        preset_s, preset_k, preset_c = _resolve_tile_preset(
            os.environ.get("CUTE_MLP_TILE")
        )
        tile_s = tile_s if tile_s is not None else preset_s
        tile_k = tile_k if tile_k is not None else preset_k
        slice_ctas = slice_ctas if slice_ctas is not None else preset_c
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.tile_s = tile_s
        self.tile_k = tile_k
        self.slice_ctas = slice_ctas
        self.num_slices = intermediate_size // tile_s
        self.num_k_tiles = max(hidden_size // tile_k, 1)
        # Preset name for error-message context (captured from env here so
        # a bad entry in _TILE_PRESETS points operators at the right key).
        _preset_name = os.environ.get("CUTE_MLP_TILE") or _DEFAULT_PRESET_NAME
        assert intermediate_size % tile_s == 0, (
            f"preset={_preset_name!r}: intermediate_size={intermediate_size} "
            f"not multiple of tile_s={tile_s}"
        )
        assert hidden_size % tile_k == 0, (
            f"preset={_preset_name!r}: hidden_size={hidden_size} "
            f"not multiple of tile_k={tile_k}"
        )
        assert tile_s % FP4_BLOCK_SIZE == 0, (
            f"preset={_preset_name!r}: tile_s={tile_s} not multiple of "
            f"FP4_BLOCK_SIZE={FP4_BLOCK_SIZE}"
        )
        assert hidden_size % FP4_BLOCK_SIZE == 0, (
            f"preset={_preset_name!r}: hidden_size={hidden_size} not "
            f"multiple of FP4_BLOCK_SIZE={FP4_BLOCK_SIZE}"
        )
        # Phase E ε epilogue divides hidden across num_threads without
        # a tail loop; Qwen3.5/3.6-27B at hidden=5120, num_threads=128 is
        # clean (40 per thread). Assert guards future hidden sizes.
        assert hidden_size % 128 == 0, (
            f"hidden_size={hidden_size} must be divisible by num_threads=128 "
            f"for Phase E ε epilogue to cover all elements without a tail loop"
        )
        # Each CTA owns a contiguous chunk of slices.
        self.slices_per_cta = (self.num_slices + slice_ctas - 1) // slice_ctas
        self._num_threads = 128  # 4 warps
        # FC2 thread mapping (Phase 3c):
        #   Two supported patterns, chosen by tile_k vs num_threads:
        #   (A) tile_k >= num_threads AND tile_k % num_threads == 0:
        #       Each thread owns `rows_per_thread = tile_k / num_threads`
        #       consecutive output rows. No intra-warp reduction needed.
        #       This is the PRODUCTION path (tile_k=640, rows_per_thread=5).
        #   (B) tile_k < num_threads AND num_threads % tile_k == 0:
        #       Phase 3b fallback — `threads_per_row = num_threads / tile_k`
        #       threads share an output row, shfl_xor reduction at end.
        if tile_k >= self._num_threads:
            assert tile_k % self._num_threads == 0, (
                f"tile_k={tile_k} must be multiple of "
                f"num_threads={self._num_threads} when tile_k >= num_threads"
            )
            self._rows_per_thread = tile_k // self._num_threads
            self._threads_per_row = 1
        else:
            assert self._num_threads % tile_k == 0, (
                f"num_threads={self._num_threads} must be multiple of "
                f"tile_k={tile_k} when tile_k < num_threads"
            )
            self._rows_per_thread = 1
            self._threads_per_row = self._num_threads // tile_k
        # SMEM layout:
        #   [0, hidden*4)         -> smem_x FP32
        #   [+16)                 -> cross-warp reduce scratch (4 warps × FP32)
        #   [+tile_s*2)           -> smem_intermediate_bf16 (tile_s BF16)
        #   [+tile_s)             -> smem_intermediate_fp4 (tile_s/2 bytes)
        #   [+tile_s/FP4_BLOCK)   -> smem_intermediate_scale (u8 per block)
        #   [+4)                  -> smem_last_cta flag (u32)
        self._smem_x_bytes = hidden_size * 4
        self._smem_reduce_bytes = 4 * 4
        self._smem_intermediate_bf16_bytes = tile_s * 2
        self._smem_intermediate_fp4_bytes = tile_s // 2
        self._smem_intermediate_scale_bytes = tile_s // FP4_BLOCK_SIZE
        # pad to 4B for the last-CTA flag alignment
        self._smem_flag_bytes = 4
        self._smem_bytes = (
            self._smem_x_bytes
            + self._smem_reduce_bytes
            + self._smem_intermediate_bf16_bytes
            + self._smem_intermediate_fp4_bytes
            + self._smem_intermediate_scale_bytes
            + self._smem_flag_bytes
        )
        self._compiled = None

    # -----------------------------------------------------------------
    # Python-level launcher.
    # -----------------------------------------------------------------
    def __call__(
        self,
        x: torch.Tensor,                   # [nat, hidden] BF16
        gate_w_fp4: torch.Tensor,          # [interm, hidden//2] u8
        gate_w_scale: torch.Tensor,        # [interm, hidden//16] u8 UE4M3
        up_w_fp4: torch.Tensor,
        up_w_scale: torch.Tensor,
        down_w_fp4: torch.Tensor,          # [hidden, interm//2] u8
        down_w_scale: torch.Tensor,        # [hidden, interm//16] u8 UE4M3
        mlp_partial_fp32: torch.Tensor,    # [nat, slice_ctas, hidden] FP32 (zeroed)
        mlp_arrival_count: torch.Tensor,   # [nat, num_k_tiles] u32 (zeroed)
        mlp_output: torch.Tensor,          # [nat, hidden] BF16
        nat: int,
        # NVFP4 dequantization = fp4 × per_block_scale × weight_global_scale.
        # Without these factors kernel output is off by prod(1/wgs). gate
        # and up share one scale (MergedColumnParallelLinear). Default 1.0
        # → kernel-math smoke tests.
        gate_up_global_scale: float = 1.0,
        down_global_scale: float = 1.0,
        stream: Optional[object] = None,
        # --- NEW: Phase E ε epilogue (β-lite path 2) ---------------------
        # When `emit_epilogue=True`, after Phase D's last-CTA-per-k-tile
        # writes `mlp_output[nat, hidden]`, a second-level per-token barrier
        # elects ONE globally-last CTA per token to run the ε epilogue:
        #   residual_final = residual_post_ln + mlp_output   (BF16, FP32 acc)
        #   next_hidden     = emit_next_layernorm
        #                     ? RMSNorm(residual_final) * next_gamma
        #                     : residual_final
        # Legacy callers (emit_epilogue=False) are bit-identical to Phase D —
        # all Phase-E pointers are passed as Int64(0) and the epilogue
        # block is gated by `emit_ep_i32 == 0`.
        residual_post_ln: Optional[torch.Tensor] = None,      # [nat, hidden] BF16
        next_input_layernorm_gamma: Optional[torch.Tensor] = None,  # [hidden] BF16
        next_hidden_output: Optional[torch.Tensor] = None,    # [nat, hidden] BF16
        emit_epilogue: bool = False,
        emit_next_layernorm: bool = True,
        rms_eps: float = 1e-6,
    ) -> torch.Tensor:
        if not _CUTE_AVAILABLE:
            raise RuntimeError(
                "Phase_D_MLP_Kernel requires CUTLASS; not available."
            )
        for t in (
            x, gate_w_fp4, gate_w_scale, up_w_fp4, up_w_scale,
            down_w_fp4, down_w_scale, mlp_partial_fp32,
            mlp_arrival_count, mlp_output,
        ):
            assert t.is_contiguous(), f"tensor {t.shape} not contiguous"
        assert x.dtype == torch.bfloat16
        assert gate_w_fp4.dtype == torch.uint8
        assert gate_w_scale.dtype == torch.uint8
        assert up_w_fp4.dtype == torch.uint8
        assert up_w_scale.dtype == torch.uint8
        assert down_w_fp4.dtype == torch.uint8
        assert down_w_scale.dtype == torch.uint8
        assert mlp_partial_fp32.dtype == torch.float32
        assert mlp_arrival_count.dtype == torch.uint32 or \
               mlp_arrival_count.dtype == torch.int32, (
            f"mlp_arrival_count must be u32/i32, got {mlp_arrival_count.dtype}"
        )
        assert mlp_output.dtype == torch.bfloat16
        assert x.shape == (nat, self.hidden_size)
        assert gate_w_fp4.shape == (
            self.intermediate_size, self.hidden_size // 2)
        assert gate_w_scale.shape == (
            self.intermediate_size, self.hidden_size // FP4_BLOCK_SIZE)
        assert up_w_fp4.shape == gate_w_fp4.shape
        assert up_w_scale.shape == gate_w_scale.shape
        assert down_w_fp4.shape == (
            self.hidden_size, self.intermediate_size // 2)
        assert down_w_scale.shape == (
            self.hidden_size,
            self.intermediate_size // FP4_BLOCK_SIZE)
        assert mlp_partial_fp32.shape == (
            nat, self.slice_ctas, self.hidden_size,
        ), (
            f"mlp_partial_fp32 expects [nat, slice_ctas, hidden]; "
            f"got {tuple(mlp_partial_fp32.shape)} (nat={nat}, "
            f"slice_ctas={self.slice_ctas}, hidden={self.hidden_size})"
        )
        assert mlp_arrival_count.shape == (nat, self.num_k_tiles)
        assert mlp_output.shape == (nat, self.hidden_size)

        # x tensor is passed flat; all FP4 / scale buffers use byte ptrs.
        x_flat = x.view(-1)
        gate_fp4_ptr = Int64(gate_w_fp4.data_ptr())
        gate_sc_ptr = Int64(gate_w_scale.data_ptr())
        up_fp4_ptr = Int64(up_w_fp4.data_ptr())
        up_sc_ptr = Int64(up_w_scale.data_ptr())
        down_fp4_ptr = Int64(down_w_fp4.data_ptr())
        down_sc_ptr = Int64(down_w_scale.data_ptr())
        partial_ptr = Int64(mlp_partial_fp32.data_ptr())
        count_ptr = Int64(mlp_arrival_count.data_ptr())
        output_ptr = Int64(mlp_output.data_ptr())

        # Phase E ε epilogue pointers (optional).
        # `emit_epilogue=False` → all four ptrs are Int64(0) and the kernel
        # gates the epilogue block on the emit_ep_i32 flag so the legacy
        # call path is a byte-for-byte pass-through.
        # `emit_epilogue=True`  → validate shapes/dtypes and lazy-allocate
        # a kernel-owned int32[max_nat_seen] secondary arrival counter.
        # The barrier is zeroed by the globally-last-CTA itself at the end
        # of the ε epilogue (atomic_add(-num_k_tiles)), matching the
        # self-reset pattern used by the Phase B/C fused kernel — so the
        # caller never has to zero it.
        if emit_epilogue:
            assert residual_post_ln is not None, (
                "emit_epilogue=True requires residual_post_ln"
            )
            assert next_hidden_output is not None, (
                "emit_epilogue=True requires next_hidden_output"
            )
            assert residual_post_ln.shape == (nat, self.hidden_size)
            assert next_hidden_output.shape == (nat, self.hidden_size)
            assert residual_post_ln.dtype == torch.bfloat16
            assert next_hidden_output.dtype == torch.bfloat16
            assert residual_post_ln.is_contiguous()
            assert next_hidden_output.is_contiguous()
            if emit_next_layernorm:
                assert next_input_layernorm_gamma is not None, (
                    "emit_next_layernorm=True requires next_input_layernorm_gamma"
                )
                assert next_input_layernorm_gamma.shape == (self.hidden_size,)
                assert next_input_layernorm_gamma.dtype == torch.bfloat16
                assert next_input_layernorm_gamma.is_contiguous()
            # Lazy-allocate (or grow) the per-token secondary barrier.
            # TODO(task-16, CUDA-graph capture): the `torch.zeros(...)` path
            # below launches a memset kernel that would be recorded into any
            # graph capturing the first emit_epilogue=True call — subsequent
            # replays would NOT re-zero the barrier, and the self-reset
            # pattern below covers only post-first invocations.
            # Fix before β-coop CUDA-graph capture: either (a) pre-allocate
            # at attach_next_input_layernorm time with sizeof(max_num_seqs)
            # or (b) warmup-launch pre-capture. See spec §5.6.
            need_grow = (
                not hasattr(self, "_epilogue_barrier")
                or self._epilogue_barrier is None
                or self._epilogue_barrier.numel() < nat
            )
            if need_grow:
                self._epilogue_barrier = torch.zeros(
                    nat, dtype=torch.int32, device=mlp_output.device,
                )
            residual_ptr = Int64(residual_post_ln.data_ptr())
            next_gamma_ptr = Int64(
                next_input_layernorm_gamma.data_ptr()
                if next_input_layernorm_gamma is not None else 0
            )
            next_hidden_ptr = Int64(next_hidden_output.data_ptr())
            barrier_ptr = Int64(self._epilogue_barrier.data_ptr())
            emit_ep_i32 = Int32(1)
            emit_next_ln_i32 = Int32(1 if emit_next_layernorm else 0)
        else:
            residual_ptr = Int64(0)
            next_gamma_ptr = Int64(0)
            next_hidden_ptr = Int64(0)
            barrier_ptr = Int64(0)
            emit_ep_i32 = Int32(0)
            emit_next_ln_i32 = Int32(0)
        rms_eps_f32 = Float32(float(rms_eps))

        grid = (self.slice_ctas, self.num_k_tiles, nat)

        if stream is None:
            stream_arg = _cuda_driver.CUstream(
                int(torch.cuda.current_stream().cuda_stream)
            )
        else:
            stream_arg = stream

        gate_up_gs_f32 = Float32(float(gate_up_global_scale))
        down_gs_f32 = Float32(float(down_global_scale))

        if self._compiled is None:
            logger.info("Compiling CuTe Phase D MLP kernel (first call)...")
            self._compiled = cute.compile(
                self._jit_launch,
                x_flat,
                gate_fp4_ptr, gate_sc_ptr,
                up_fp4_ptr, up_sc_ptr,
                down_fp4_ptr, down_sc_ptr,
                partial_ptr, count_ptr, output_ptr,
                Int32(self.hidden_size),
                Int32(self.intermediate_size),
                Int32(self.num_slices),
                Int32(self.slices_per_cta),
                Int32(self.tile_s),
                Int32(self.tile_k),
                Int32(self.num_k_tiles),
                Int32(self.slice_ctas),
                gate_up_gs_f32, down_gs_f32,
                residual_ptr, next_gamma_ptr, next_hidden_ptr, barrier_ptr,
                emit_ep_i32, emit_next_ln_i32, rms_eps_f32,
                Int32(grid[0]), Int32(grid[1]), Int32(grid[2]),
                stream_arg,
            )

        self._compiled(
            x_flat,
            gate_fp4_ptr, gate_sc_ptr,
            up_fp4_ptr, up_sc_ptr,
            down_fp4_ptr, down_sc_ptr,
            partial_ptr, count_ptr, output_ptr,
            Int32(self.hidden_size),
            Int32(self.intermediate_size),
            Int32(self.num_slices),
            Int32(self.slices_per_cta),
            Int32(self.tile_s),
            Int32(self.tile_k),
            Int32(self.num_k_tiles),
            Int32(self.slice_ctas),
            gate_up_gs_f32, down_gs_f32,
            residual_ptr, next_gamma_ptr, next_hidden_ptr, barrier_ptr,
            emit_ep_i32, emit_next_ln_i32, rms_eps_f32,
            Int32(grid[0]), Int32(grid[1]), Int32(grid[2]),
            stream_arg,
        )
        return mlp_output

    # -----------------------------------------------------------------
    # JIT host wrapper + @cute.kernel body.
    # -----------------------------------------------------------------
    if _CUTE_AVAILABLE:

        @cute.jit
        def _jit_launch(
            self,
            x_flat,
            gate_fp4_ptr: Int64, gate_sc_ptr: Int64,
            up_fp4_ptr: Int64, up_sc_ptr: Int64,
            down_fp4_ptr: Int64, down_sc_ptr: Int64,
            partial_ptr: Int64, count_ptr: Int64, output_ptr: Int64,
            hidden: Int32, interm: Int32,
            num_slices: Int32, slices_per_cta: Int32,
            tile_s: Int32, tile_k: Int32,
            num_k_tiles: Int32, slice_ctas: Int32,
            gate_up_gs: Float32, down_gs: Float32,
            residual_ptr: Int64, next_gamma_ptr: Int64,
            next_hidden_ptr: Int64, barrier_ptr: Int64,
            emit_ep: Int32, emit_next_ln: Int32, rms_eps: Float32,
            grid_x: Int32, grid_y: Int32, grid_z: Int32,
            stream,
        ):
            """JIT host wrapper: compiles kernel launch into MLIR."""
            self._kernel(
                x_flat,
                gate_fp4_ptr, gate_sc_ptr,
                up_fp4_ptr, up_sc_ptr,
                down_fp4_ptr, down_sc_ptr,
                partial_ptr, count_ptr, output_ptr,
                hidden, interm,
                num_slices, slices_per_cta,
                tile_s, tile_k,
                num_k_tiles, slice_ctas,
                gate_up_gs, down_gs,
                residual_ptr, next_gamma_ptr, next_hidden_ptr, barrier_ptr,
                emit_ep, emit_next_ln, rms_eps,
            ).launch(
                grid=[grid_x, grid_y, grid_z],
                block=[self._num_threads, 1, 1],
                smem=self._smem_bytes,
                stream=stream,
            )

        @cute.kernel
        def _kernel(
            self,
            x_flat,
            gate_fp4_ptr: Int64, gate_sc_ptr: Int64,
            up_fp4_ptr: Int64, up_sc_ptr: Int64,
            down_fp4_ptr: Int64, down_sc_ptr: Int64,
            partial_ptr: Int64, count_ptr: Int64, output_ptr: Int64,
            hidden: Int32, interm: Int32,
            num_slices: Int32, slices_per_cta: Int32,
            tile_s: Int32, tile_k: Int32,
            num_k_tiles: Int32, slice_ctas: Int32,
            gate_up_gs: Float32, down_gs: Float32,
            residual_ptr: Int64, next_gamma_ptr: Int64,
            next_hidden_ptr: Int64, barrier_ptr: Int64,
            emit_ep: Int32, emit_next_ln: Int32, rms_eps: Float32,
        ):
            """Phase D end-to-end fused MLP kernel (small-dim path)."""
            # === Phase 0: Thread/block ids =====================
            bx, by, bz = cute.arch.block_idx()  # slice_group, k_tile, token
            lane = cute.arch.lane_idx()
            warp = cute.arch.warp_idx()
            tid = warp * Int32(32) + lane

            # === Phase 1: SMEM pointer layout ===
            smem = cute.arch.get_dyn_smem(cutlass.Uint8, alignment=128)
            smem_x = shared_ptr_to_i64(smem)
            smem_reduce = shared_ptr_to_i64(
                smem + Int32(self._smem_x_bytes)
            )
            smem_interm_bf16 = shared_ptr_to_i64(
                smem + Int32(self._smem_x_bytes
                              + self._smem_reduce_bytes)
            )
            smem_interm_fp4 = shared_ptr_to_i64(
                smem + Int32(self._smem_x_bytes
                              + self._smem_reduce_bytes
                              + self._smem_intermediate_bf16_bytes)
            )
            smem_interm_scale = shared_ptr_to_i64(
                smem + Int32(self._smem_x_bytes
                              + self._smem_reduce_bytes
                              + self._smem_intermediate_bf16_bytes
                              + self._smem_intermediate_fp4_bytes)
            )
            smem_last_flag = shared_ptr_to_i64(
                smem + Int32(self._smem_x_bytes
                              + self._smem_reduce_bytes
                              + self._smem_intermediate_bf16_bytes
                              + self._smem_intermediate_fp4_bytes
                              + self._smem_intermediate_scale_bytes)
            )

            # === Phase 2: Load x[bz, :] into smem_x as FP32 ===
            elems_per_thr = hidden // Int32(self._num_threads)
            _i = Int32(0)
            while _i < elems_per_thr:
                flat = tid + _i * Int32(self._num_threads)
                gmem_idx = bz * hidden + flat
                x_bf16 = x_flat[gmem_idx]
                x_f32 = Float32(x_bf16)
                _st_shared_f32(
                    smem_x + Int64(flat) * Int64(4),
                    x_f32,
                )
                _i = _i + Int32(1)

            cute.arch.sync_threads()

            # === Phase 3: Iterate slices assigned to this CTA ===
            s_start = bx * slices_per_cta
            s_end_raw = s_start + slices_per_cta
            s_end = s_end_raw
            if s_end > num_slices:
                s_end = num_slices

            # Useful constants.
            FP4_BS = Int32(FP4_BLOCK_SIZE)
            LOG2E_F = Float32(LOG2_E)
            # `num_h_blocks` = number of FP4 blocks along hidden dim.
            num_h_blocks = hidden // FP4_BS  # e.g. 128/16 = 8

            s = s_start
            while s < s_end:
                # -------- Stage 3a: FC1 -> smem_interm_bf16[tile_s] --------
                j_base = s * tile_s
                j_local = Int32(0)
                while j_local < tile_s:
                    j = j_base + j_local

                    # Per-thread FP32 partial sums (gate, up).
                    gate_acc = Float32(0.0)
                    up_acc = Float32(0.0)

                    # Each thread handles `elems_per_thr` elements of hidden.
                    # We iterate one element at a time (simple + correct).
                    k_i = Int32(0)
                    while k_i < elems_per_thr:
                        h = tid + k_i * Int32(self._num_threads)
                        x_val = _ld_shared_f32(
                            smem_x + Int64(h) * Int64(4)
                        )

                        # FP4 block index along hidden.
                        h_block = h // FP4_BS
                        # FP4 byte offset in gate_w_fp4[j, h/2]
                        byte_col = h >> Int32(1)
                        byte_addr_gate = gate_fp4_ptr + Int64(
                            j * (hidden >> Int32(1)) + byte_col
                        )
                        byte_addr_up = up_fp4_ptr + Int64(
                            j * (hidden >> Int32(1)) + byte_col
                        )
                        nib_lo_gate = _ld_global_u8(byte_addr_gate)
                        nib_lo_up = _ld_global_u8(byte_addr_up)
                        # even h → low nibble; odd h → high nibble.
                        is_odd = h & Int32(1)
                        nib_gate = Int32(
                            ((nib_lo_gate >> (Uint32(is_odd) * Uint32(4)))
                             & Uint32(0xF))
                        )
                        nib_up = Int32(
                            ((nib_lo_up >> (Uint32(is_odd) * Uint32(4)))
                             & Uint32(0xF))
                        )

                        # Decode blockscale (UE4M3 byte → FP32).
                        scale_byte_gate = _ld_global_u8(
                            gate_sc_ptr + Int64(
                                j * num_h_blocks + h_block
                            )
                        )
                        scale_byte_up = _ld_global_u8(
                            up_sc_ptr + Int64(
                                j * num_h_blocks + h_block
                            )
                        )
                        scale_gate = _decode_ue4m3_u8_to_f32(scale_byte_gate)
                        scale_up = _decode_ue4m3_u8_to_f32(scale_byte_up)

                        # NVFP4 dequant: fp4 × block_scale × weight_global_scale.
                        gw_f32 = (
                            _fp4_nibble_to_f32(nib_gate) * scale_gate
                            * gate_up_gs
                        )
                        uw_f32 = (
                            _fp4_nibble_to_f32(nib_up) * scale_up
                            * gate_up_gs
                        )

                        gate_acc = gate_acc + x_val * gw_f32
                        up_acc = up_acc + x_val * uw_f32
                        k_i = k_i + Int32(1)

                    # Warp-level reduction (32 lanes → lane 0).
                    gate_acc = gate_acc + shfl_xor_sync(gate_acc, Int32(1))
                    gate_acc = gate_acc + shfl_xor_sync(gate_acc, Int32(2))
                    gate_acc = gate_acc + shfl_xor_sync(gate_acc, Int32(4))
                    gate_acc = gate_acc + shfl_xor_sync(gate_acc, Int32(8))
                    gate_acc = gate_acc + shfl_xor_sync(gate_acc, Int32(16))
                    up_acc = up_acc + shfl_xor_sync(up_acc, Int32(1))
                    up_acc = up_acc + shfl_xor_sync(up_acc, Int32(2))
                    up_acc = up_acc + shfl_xor_sync(up_acc, Int32(4))
                    up_acc = up_acc + shfl_xor_sync(up_acc, Int32(8))
                    up_acc = up_acc + shfl_xor_sync(up_acc, Int32(16))

                    # Cross-warp reduce via smem_reduce (gate first, then up).
                    if lane == Int32(0):
                        _st_shared_f32(
                            smem_reduce + Int64(warp) * Int64(4),
                            gate_acc,
                        )
                    cute.arch.sync_threads()

                    gate_final = Float32(0.0)
                    up_final = Float32(0.0)
                    if warp == Int32(0) and lane == Int32(0):
                        g0 = _ld_shared_f32(smem_reduce + Int64(0) * Int64(4))
                        g1 = _ld_shared_f32(smem_reduce + Int64(1) * Int64(4))
                        g2 = _ld_shared_f32(smem_reduce + Int64(2) * Int64(4))
                        g3 = _ld_shared_f32(smem_reduce + Int64(3) * Int64(4))
                        gate_final = g0 + g1 + g2 + g3
                    cute.arch.sync_threads()

                    if lane == Int32(0):
                        _st_shared_f32(
                            smem_reduce + Int64(warp) * Int64(4),
                            up_acc,
                        )
                    cute.arch.sync_threads()

                    if warp == Int32(0) and lane == Int32(0):
                        u0 = _ld_shared_f32(smem_reduce + Int64(0) * Int64(4))
                        u1 = _ld_shared_f32(smem_reduce + Int64(1) * Int64(4))
                        u2 = _ld_shared_f32(smem_reduce + Int64(2) * Int64(4))
                        u3 = _ld_shared_f32(smem_reduce + Int64(3) * Int64(4))
                        up_final = u0 + u1 + u2 + u3

                        neg_g_log2e = Float32(0.0) - gate_final * LOG2E_F
                        exp_v = exp2_approx_ftz_f32(neg_g_log2e)
                        sig_v = _rcp_approx_f32(Float32(1.0) + exp_v)
                        silu_g = gate_final * sig_v
                        out_val = silu_g * up_final

                        # Store BF16 into smem_interm_bf16[j_local].
                        # Use _cvt_2f32_to_bf16x2 + _st_shared_b16_from_u32?
                        # Simpler: store as FP32, we'll reinterpret below.
                        # Actually we need BF16. Use a 1-element bf16 store.
                        # Easiest: use _st_global_bf16_from_f32 but that
                        # writes to global. For SMEM, we write 2 bytes.
                        # Approach: compute bf16x2 from (out_val, 0) and
                        # store the low 16 bits via st.shared.b16.
                        bf16x2 = _cvt_2f32_to_bf16x2(
                            out_val, Float32(0.0)
                        )
                        _st_shared_b16_from_u32(
                            smem_interm_bf16 + Int64(j_local) * Int64(2),
                            bf16x2,
                        )

                    cute.arch.sync_threads()
                    j_local = j_local + Int32(1)

                # -------- Stage 3b: FP4 quantize intermediate --------
                # tile_s is an arbitrary multiple of FP4_BLOCK_SIZE (16).
                # FP4 blocks along the slice: interm_nblocks = tile_s/16.
                # Assign one warp per block; when interm_nblocks > 4, each
                # warp iterates over multiple blocks via a strided loop
                # (warp w handles blocks {w, w+4, w+8, ...}).
                interm_nblocks = tile_s // Int32(FP4_BLOCK_SIZE)
                # Outer block loop; iterates ceil(interm_nblocks / 4)
                # times. On each pass all 4 warps collectively handle
                # 4 blocks (warp w → block w + blk_iter*4). The inner
                # `my_block_valid` predicate masks off out-of-range
                # tail blocks when interm_nblocks is not a multiple of 4.
                # Loop bound written as (interm_nblocks + 3) // 4.
                blk_iter_max = (interm_nblocks + Int32(3)) >> Int32(2)
                blk_iter = Int32(0)
                while blk_iter < blk_iter_max:
                    my_block = warp + blk_iter * Int32(4)  # stride=4 warps
                    # Only run the iteration if this warp's block is
                    # actually in-range (handles interm_nblocks < 4).
                    my_block_valid = my_block < interm_nblocks
                    # Load BF16 element for this lane (if active).
                    elem_idx = my_block * Int32(FP4_BLOCK_SIZE) + lane
                    my_val = Float32(0.0)
                    if my_block_valid and lane < Int32(FP4_BLOCK_SIZE):
                        addr = smem_interm_bf16 + Int64(elem_idx) * Int64(2)
                        bf16_u32 = _ld_shared_b16(addr)
                        f32_bits = Int32(
                            (bf16_u32 & Uint32(0xFFFF)) << Uint32(16)
                        )
                        my_val = _bitcast_i32_to_f32(f32_bits)

                    # Block-wide max_abs reduction across 16 lanes.
                    abs_val = my_val
                    if abs_val < Float32(0.0):
                        abs_val = Float32(0.0) - abs_val

                    r1 = shfl_xor_sync(abs_val, Int32(1))
                    if r1 > abs_val:
                        abs_val = r1
                    r2 = shfl_xor_sync(abs_val, Int32(2))
                    if r2 > abs_val:
                        abs_val = r2
                    r4 = shfl_xor_sync(abs_val, Int32(4))
                    if r4 > abs_val:
                        abs_val = r4
                    r8 = shfl_xor_sync(abs_val, Int32(8))
                    if r8 > abs_val:
                        abs_val = r8

                    max_abs = abs_val
                    FP4_MAX_F = Float32(6.0)
                    scale_f32 = _div_rn_f32(max_abs, FP4_MAX_F)
                    MIN_SCALE = Float32(1e-12)
                    if scale_f32 < MIN_SCALE:
                        scale_f32 = MIN_SCALE
                    if my_block_valid and lane == Int32(0):
                        scale_u8 = _encode_ue4m3_f32_to_u8(scale_f32)
                        _st_shared_u8(
                            smem_interm_scale + Int64(my_block) * Int64(1),
                            scale_u8,
                        )
                    cute.arch.sync_threads()

                    scale_rt = scale_f32
                    if my_block_valid and lane < Int32(FP4_BLOCK_SIZE):
                        scale_u8_rd = _ld_shared_u8(
                            smem_interm_scale + Int64(my_block) * Int64(1)
                        )
                        scale_rt = _decode_ue4m3_u8_to_f32(scale_u8_rd)
                        nib = _f32_div_to_fp4_nibble(my_val, scale_rt)
                        # Each lane writes its own nibble to a per-element
                        # byte in the (unused-after-FC1) smem_interm_bf16
                        # scratch; the packer (one thread per block)
                        # then combines 16 nibbles into 8 packed bytes.
                        _st_shared_u8(
                            smem_interm_bf16
                            + Int64(elem_idx) * Int64(1),
                            nib,
                        )
                    cute.arch.sync_threads()

                    # Packer: lane 0 of each warp reads 16 nibbles and
                    # writes 8 packed bytes into smem_interm_fp4.
                    if my_block_valid and lane == Int32(0):
                        byte_out_base = (
                            my_block * Int32(FP4_BLOCK_SIZE // 2)
                        )
                        pk_i = Int32(0)
                        while pk_i < Int32(FP4_BLOCK_SIZE // 2):
                            nib_lo = _ld_shared_u8(
                                smem_interm_bf16
                                + Int64(
                                    my_block * Int32(FP4_BLOCK_SIZE)
                                    + pk_i * Int32(2)
                                ) * Int64(1)
                            )
                            nib_hi = _ld_shared_u8(
                                smem_interm_bf16
                                + Int64(
                                    my_block * Int32(FP4_BLOCK_SIZE)
                                    + pk_i * Int32(2) + Int32(1)
                                ) * Int64(1)
                            )
                            packed = Int32(
                                (nib_lo & Uint32(0xF))
                                | ((nib_hi & Uint32(0xF)) << Uint32(4))
                            )
                            _st_shared_u8(
                                smem_interm_fp4
                                + Int64(byte_out_base + pk_i) * Int64(1),
                                packed,
                            )
                            pk_i = pk_i + Int32(1)
                    cute.arch.sync_threads()
                    blk_iter = blk_iter + Int32(1)

                # -------- Stage 3c: FC2 + atomicAdd --------
                # This CTA owns output rows [by*tile_k, (by+1)*tile_k).
                #
                # Phase 3c unified FC2 layout — two paths, selected at
                # compile time by the Python-side __init__.
                #
                # Path A (tile_k >= num_threads): each thread OWNS
                #   rows_per_thread = tile_k / num_threads consecutive
                #   rows. No intra-warp reduction (disjoint-row writes).
                #   This is the PRODUCTION layout (tile_k=640,
                #   rows_per_thread=5).
                # Path B (tile_k < num_threads): Phase 3b fallback —
                #   threads_per_row threads share a row and reduce via
                #   shfl_xor at the end. Only first tile_k threads are
                #   active.
                #
                # Both paths produce identical outputs; Path A just has
                # zero reduction traffic. The if/else below is a
                # host-side constexpr — must evaluate to a plain
                # Python bool BEFORE entering CuTe's `if` lowering so
                # only one branch is emitted.
                if cutlass.const_expr(self._threads_per_row == 1):
                    # ---------- Path A: per-thread-owns-N-rows ----------
                    rows_per_thread = Int32(self._rows_per_thread)
                    # Thread tid owns rows [tid * rows_per_thread,
                    # (tid+1) * rows_per_thread) within the k-tile.
                    row_base_local = tid * rows_per_thread

                    # One FP32 accumulator per owned row, kept in
                    # registers. We unroll over rows_per_thread which is
                    # a Python-side constant so the CuTe DSL produces
                    # disjoint register slots.
                    # Use a Python list of Float32s: CuTe DSL handles
                    # per-row independent accumulators correctly.
                    rpt = self._rows_per_thread
                    acc_list = [Float32(0.0) for _ in range(rpt)]

                    iter_i = Int32(0)
                    while iter_i < tile_s:
                        h = iter_i
                        # Dequant intermediate at h (SMEM).
                        interm_block = h >> Int32(4)   # h // 16
                        interm_byte_addr = (
                            smem_interm_fp4 + Int64(h >> Int32(1))
                        )
                        interm_byte = _ld_shared_u8(interm_byte_addr)
                        interm_is_odd = h & Int32(1)
                        interm_nib = Int32(
                            (interm_byte
                             >> (Uint32(interm_is_odd) * Uint32(4)))
                            & Uint32(0xF)
                        )
                        interm_scale_u8 = _ld_shared_u8(
                            smem_interm_scale + Int64(interm_block)
                        )
                        interm_scale_f32 = _decode_ue4m3_u8_to_f32(
                            interm_scale_u8
                        )
                        interm_val = (
                            _fp4_nibble_to_f32(interm_nib)
                            * interm_scale_f32
                        )

                        s_col_base = s * tile_s
                        global_col = s_col_base + h

                        # For each of this thread's owned output rows,
                        # load down_w[row, global_col] FP4 + scale and
                        # multiply-accumulate. `range_constexpr` unrolls
                        # at compile time so `r` is a Python int and
                        # `acc_list[r]` indexes a Python list.
                        for r in cutlass.range_constexpr(rpt):
                            k_row_global = (
                                by * tile_k + row_base_local + Int32(r)
                            )
                            dw_byte_addr = down_fp4_ptr + Int64(
                                k_row_global * (interm >> Int32(1))
                                + (global_col >> Int32(1))
                            )
                            dw_byte = _ld_global_u8(dw_byte_addr)
                            dw_is_odd = global_col & Int32(1)
                            dw_nib = Int32(
                                (dw_byte
                                 >> (Uint32(dw_is_odd) * Uint32(4)))
                                & Uint32(0xF)
                            )
                            dw_scale_addr = down_sc_ptr + Int64(
                                k_row_global * (interm // FP4_BS)
                                + (global_col // FP4_BS)
                            )
                            dw_scale_u8 = _ld_global_u8(dw_scale_addr)
                            dw_scale_f32 = _decode_ue4m3_u8_to_f32(
                                dw_scale_u8
                            )
                            dw_val = (
                                _fp4_nibble_to_f32(dw_nib) * dw_scale_f32
                                * down_gs
                            )
                            acc_list[r] = (
                                acc_list[r] + interm_val * dw_val
                            )
                        iter_i = iter_i + Int32(1)

                    # Each thread atomic-adds its owned rows. Rows are
                    # disjoint across threads, so there's no race.
                    # 2026-04-20: partial buffer is now [nat, slice_ctas,
                    # hidden] — each CTA writes to its own `bx` slot so
                    # the cross-CTA reduction becomes a deterministic
                    # gather in the last-CTA epilogue (see audit commit
                    # 16475223f / Option 1). Intra-CTA atomicAdd stays
                    # for the per-s accumulation (same thread, same
                    # address, program order → deterministic).
                    for r in cutlass.range_constexpr(rpt):
                        k_row_global = (
                            by * tile_k + row_base_local + Int32(r)
                        )
                        partial_idx = (
                            bz * slice_ctas * hidden
                            + bx * hidden
                            + k_row_global
                        )
                        _atomic_add_f32(
                            partial_ptr + Int64(partial_idx) * Int64(4),
                            acc_list[r],
                        )
                else:
                    # ---------- Path B: multiple-threads-per-row ----------
                    # Phase 3b behavior, preserved for tile_k < num_threads
                    # (small-tile test configs).
                    threads_per_row = Int32(self._threads_per_row)
                    row_local = tid // threads_per_row
                    thread_in_row = tid - row_local * threads_per_row
                    elems_per_in_row = tile_s // threads_per_row

                    k_row_global = by * tile_k + row_local
                    h_start = thread_in_row * elems_per_in_row

                    out_acc = Float32(0.0)
                    iter_i = Int32(0)
                    while iter_i < elems_per_in_row:
                        h = h_start + iter_i

                        interm_block = h >> Int32(4)
                        interm_byte_addr = (
                            smem_interm_fp4 + Int64(h >> Int32(1))
                        )
                        interm_byte = _ld_shared_u8(interm_byte_addr)
                        interm_is_odd = h & Int32(1)
                        interm_nib = Int32(
                            (interm_byte
                             >> (Uint32(interm_is_odd) * Uint32(4)))
                            & Uint32(0xF)
                        )
                        interm_scale_u8 = _ld_shared_u8(
                            smem_interm_scale + Int64(interm_block)
                        )
                        interm_scale_f32 = _decode_ue4m3_u8_to_f32(
                            interm_scale_u8
                        )
                        interm_val = (
                            _fp4_nibble_to_f32(interm_nib)
                            * interm_scale_f32
                        )

                        s_col_base = s * tile_s
                        global_col = s_col_base + h
                        dw_row = k_row_global
                        dw_byte_addr = down_fp4_ptr + Int64(
                            dw_row * (interm >> Int32(1))
                            + (global_col >> Int32(1))
                        )
                        dw_byte = _ld_global_u8(dw_byte_addr)
                        dw_is_odd = global_col & Int32(1)
                        dw_nib = Int32(
                            (dw_byte
                             >> (Uint32(dw_is_odd) * Uint32(4)))
                            & Uint32(0xF)
                        )
                        dw_scale_addr = down_sc_ptr + Int64(
                            dw_row * (interm // FP4_BS)
                            + (global_col // FP4_BS)
                        )
                        dw_scale_u8 = _ld_global_u8(dw_scale_addr)
                        dw_scale_f32 = _decode_ue4m3_u8_to_f32(dw_scale_u8)
                        dw_val = (
                            _fp4_nibble_to_f32(dw_nib) * dw_scale_f32
                            * down_gs
                        )

                        out_acc = out_acc + interm_val * dw_val
                        iter_i = iter_i + Int32(1)

                    # Static-bool reductions (threads_per_row is a
                    # Python-side constant; use const_expr so the DSL
                    # doesn't try to lower each comparison to dynamic IR).
                    if cutlass.const_expr(self._threads_per_row >= 2):
                        out_acc = out_acc + shfl_xor_sync(out_acc, Int32(1))
                    if cutlass.const_expr(self._threads_per_row >= 4):
                        out_acc = out_acc + shfl_xor_sync(out_acc, Int32(2))
                    if cutlass.const_expr(self._threads_per_row >= 8):
                        out_acc = out_acc + shfl_xor_sync(out_acc, Int32(4))
                    if cutlass.const_expr(self._threads_per_row >= 16):
                        out_acc = out_acc + shfl_xor_sync(out_acc, Int32(8))

                    # Hoist partial_idx outside the `if thread_in_row ==
                    # 0` region — CuTe's typed-if tracking can't see
                    # through Python control flow that assigns inside an
                    # MLIR if-region for the first time, especially when
                    # a `const_expr` outer-branch has already defined
                    # the name in an alternate Python branch.
                    # 2026-04-20: per-bx slot indexing — see Site M-A.
                    partial_idx = (
                        bz * slice_ctas * hidden
                        + bx * hidden
                        + k_row_global
                    )
                    if thread_in_row == Int32(0):
                        _atomic_add_f32(
                            partial_ptr + Int64(partial_idx) * Int64(4),
                            out_acc,
                        )

                cute.arch.sync_threads()
                s = s + Int32(1)

            # === Phase 4: Arrival counter + last-CTA epilogue ===
            _threadfence()
            cute.arch.sync_threads()

            if tid == Int32(0):
                count_idx = bz * num_k_tiles + by
                old = _atomic_add_u32(
                    count_ptr + Int64(count_idx) * Int64(4),
                    Int32(1),
                )
                # is_last?
                is_last_flag = Int32(0)
                if old == (slice_ctas - Int32(1)):
                    is_last_flag = Int32(1)
                # Broadcast via SMEM flag.
                _st_shared_b32(
                    smem_last_flag + Int64(0),
                    Uint32(is_last_flag),
                )
            cute.arch.sync_threads()

            # All threads read the flag.
            last_flag_u32 = _ld_shared_b32(smem_last_flag + Int64(0))
            is_last = Int32(last_flag_u32) == Int32(1)

            if is_last:
                # Phase 3c unified epilogue: two paths mirroring FC2.
                #   Path A (tile_k >= num_threads): each thread owns
                #     rows_per_thread consecutive rows; same thread-to-
                #     row assignment as the FC2 atomic writes, so we
                #     simply iterate.
                #   Path B (tile_k < num_threads): only first tile_k
                #     threads write one row each.
                # 2026-04-20: deterministic reduction — gather each
                # `bx` slot in constexpr index order and sum. Sum order
                # is fixed, so output is bit-identical across runs.
                if cutlass.const_expr(self._threads_per_row == 1):
                    rpt = self._rows_per_thread
                    rows_per_thread = Int32(rpt)
                    row_base_local = tid * rows_per_thread
                    for r in cutlass.range_constexpr(rpt):
                        k_row_global = (
                            by * tile_k + row_base_local + Int32(r)
                        )
                        output_idx = bz * hidden + k_row_global
                        val_f32 = Float32(0.0)
                        for bx_i in cutlass.range_constexpr(
                            self.slice_ctas
                        ):
                            slot_idx = (
                                bz * slice_ctas * hidden
                                + Int32(bx_i) * hidden
                                + k_row_global
                            )
                            val_f32 = val_f32 + _ld_global_f32(
                                partial_ptr + Int64(slot_idx) * Int64(4)
                            )
                        _st_global_bf16_from_f32(
                            output_ptr + Int64(output_idx) * Int64(2),
                            val_f32,
                        )
                else:
                    # Path B: only first tile_k threads write. Pre-
                    # declare val_f32/output_idx outside the dynamic
                    # if-region so CuTe's typed-if tracking sees the
                    # types before the inner body.
                    k_row_global = by * tile_k + tid
                    output_idx = bz * hidden + k_row_global
                    val_f32 = Float32(0.0)
                    if tid < tile_k:
                        for bx_i in cutlass.range_constexpr(
                            self.slice_ctas
                        ):
                            slot_idx = (
                                bz * slice_ctas * hidden
                                + Int32(bx_i) * hidden
                                + k_row_global
                            )
                            val_f32 = val_f32 + _ld_global_f32(
                                partial_ptr + Int64(slot_idx) * Int64(4)
                            )
                        _st_global_bf16_from_f32(
                            output_ptr + Int64(output_idx) * Int64(2),
                            val_f32,
                        )

            # === Phase E: ε epilogue (optional) ==========================
            # Input-mux: activated only when emit_ep==1 (runtime flag from
            # Python launcher). Preserves Phase D's byte-for-byte legacy
            # behavior when disabled.
            #
            # Design: each k-tile's local-last-CTA (the `is_last` branch
            # above) has just written its slice of mlp_output[bz, :]. A
            # per-token secondary barrier elects ONE globally-last CTA per
            # token — the one whose atomicAdd on barrier_ptr[bz] returns
            # `num_k_tiles - 1`. That CTA reads all of residual_post_ln +
            # mlp_output, computes residual_final + (optional) RMSNorm,
            # and writes residual_post_ln (in place) + next_hidden_output.
            #
            # Memory ordering: the `_threadfence` inside the `is_last`
            # branch below + atom.global.add.u32 (acq_rel) guarantees the
            # globally-last CTA sees all prior mlp_output writes.
            #
            # Self-reset: the globally-last CTA atomicAdd(-num_k_tiles)
            # restores barrier_ptr[bz] to 0 so the same tensor can be
            # reused across kernel invocations without a host-side zero_()
            # (matches arrival-count self-reset in Phase C).
            #
            # Cross-warp reduce pattern: mirrors Phase C RMSNorm at
            # kernel.py:~1850-1878 — 5-step warp shfl_xor butterfly, then
            # 4-slot SMEM exchange for cross-warp, then broadcast.
            if emit_ep == Int32(1):
                # Secondary barrier: only local-last CTAs participate.
                # Broadcast result via smem_last_flag. Non-last CTAs
                # already have flag=0 here (they never stored into
                # smem_last_flag after Phase 4's election) BUT we
                # re-initialize via a new atomic path anyway to be safe.
                is_global_last_local = Int32(0)
                if is_last:
                    _threadfence()  # flush mlp_output slice writes
                    cute.arch.sync_threads()
                    if tid == Int32(0):
                        old2 = _atomic_add_u32(
                            barrier_ptr + Int64(bz) * Int64(4),
                            Int32(1),
                        )
                        gl_flag = Int32(0)
                        if old2 == (num_k_tiles - Int32(1)):
                            gl_flag = Int32(1)
                        _st_shared_b32(
                            smem_last_flag + Int64(0),
                            Uint32(gl_flag),
                        )
                    cute.arch.sync_threads()
                    gl_u32 = _ld_shared_b32(smem_last_flag + Int64(0))
                    is_global_last_local = Int32(gl_u32)

                if is_global_last_local == Int32(1):
                    # --- Pass 1: residual_add + sum_of_squares --------
                    # Per-thread: iterate over hidden elements owned by
                    # this thread. Element idx owned by tid is
                    # {tid, tid+128, tid+256, ..., tid+128*(N-1)} where
                    # N = hidden/num_threads = 5120/128 = 40 for Qwen3.5
                    # (Python constexpr — <100, avoids constexpr-OOM).
                    n_per_thr = self.hidden_size // self._num_threads

                    res_base = (
                        residual_ptr + Int64(bz * self.hidden_size)
                        * Int64(2)  # BF16 = 2 bytes
                    )
                    mlp_base = (
                        output_ptr + Int64(bz * self.hidden_size)
                        * Int64(2)  # BF16 = 2 bytes
                    )
                    next_hidden_base = (
                        next_hidden_ptr + Int64(bz * self.hidden_size)
                        * Int64(2)  # BF16 = 2 bytes
                    )

                    ss = Float32(0.0)
                    for _i in cutlass.range_constexpr(n_per_thr):
                        idx = tid + Int32(_i * self._num_threads)
                        res_f32 = _ld_global_b16_to_f32(
                            res_base + Int64(idx) * Int64(2)
                        )
                        mlp_f32 = _ld_global_b16_to_f32(
                            mlp_base + Int64(idx) * Int64(2)
                        )
                        rf = res_f32 + mlp_f32
                        # Write residual_final back to residual_post_ln in place.
                        _st_global_bf16_from_f32(
                            res_base + Int64(idx) * Int64(2),
                            rf,
                        )
                        ss = ss + rf * rf

                    # --- Pass 2: cross-warp reduce sum_of_squares ------
                    # Intra-warp butterfly (5 steps for 32 lanes).
                    ss = ss + shfl_xor_sync(ss, Int32(1))
                    ss = ss + shfl_xor_sync(ss, Int32(2))
                    ss = ss + shfl_xor_sync(ss, Int32(4))
                    ss = ss + shfl_xor_sync(ss, Int32(8))
                    ss = ss + shfl_xor_sync(ss, Int32(16))

                    # Cross-warp exchange via 4-slot smem_reduce scratch.
                    if lane == Int32(0):
                        _st_shared_f32(
                            smem_reduce + Int64(warp) * Int64(4),
                            ss,
                        )
                    cute.arch.sync_threads()

                    # Warp 0 lane 0 reduces 4 warp-partials → variance → inv_rms.
                    if warp == Int32(0):
                        if lane == Int32(0):
                            total_ss = _ld_shared_f32(
                                smem_reduce + Int64(0) * Int64(4)
                            )
                            total_ss = total_ss + _ld_shared_f32(
                                smem_reduce + Int64(1) * Int64(4)
                            )
                            total_ss = total_ss + _ld_shared_f32(
                                smem_reduce + Int64(2) * Int64(4)
                            )
                            total_ss = total_ss + _ld_shared_f32(
                                smem_reduce + Int64(3) * Int64(4)
                            )
                            variance = total_ss / Float32(
                                float(self.hidden_size)
                            )
                            inv_rms = _rsqrt_approx_f32(variance + rms_eps)
                            # Broadcast inv_rms via smem_reduce slot 0.
                            _st_shared_f32(
                                smem_reduce + Int64(0),
                                inv_rms,
                            )
                    cute.arch.sync_threads()

                    inv_rms_val = _ld_shared_f32(smem_reduce + Int64(0))

                    # --- Pass 3: write next_hidden ---------------------
                    # Two-path: emit_next_ln determines whether we do
                    # RMSNorm * gamma or just memcpy residual_final.
                    # Dynamic if on runtime emit_next_ln flag.
                    if emit_next_ln == Int32(1):
                        for _i in cutlass.range_constexpr(n_per_thr):
                            idx = tid + Int32(_i * self._num_threads)
                            # Re-read residual_final (we just wrote it;
                            # L2-hot). Cheaper than keeping in registers
                            # across the cross-warp sync.
                            rf = _ld_global_b16_to_f32(
                                res_base + Int64(idx) * Int64(2)
                            )
                            gamma_f32 = _ld_global_b16_to_f32(
                                next_gamma_ptr + Int64(idx) * Int64(2)
                            )
                            # normed = (rf * inv_rms).to(bf16) * gamma
                            # Match Python ref byte-for-byte: the
                            # intermediate cast to BF16 (before gamma)
                            # produces a measurable diff at rtol=1e-2.
                            # The reference does `.to(torch.bfloat16)`
                            # after `rf * rstd` and then multiplies by
                            # gamma. We emulate by round-tripping the
                            # FP32 product through BF16.
                            normed_f32 = rf * inv_rms_val
                            # Round-trip through BF16: cvt.rn.bf16.f32
                            # then widen. Done inline via a single-
                            # element bf16 store + reload would cost a
                            # memory hop; simpler: cvt.rn.bf16.f32 +
                            # shift-back.
                            normed_bf16_u32 = _cvt_2f32_to_bf16x2(
                                normed_f32, Float32(0.0)
                            )
                            # low 16 bits hold the bf16-encoded value;
                            # reconstruct FP32 from it.
                            low16 = normed_bf16_u32 & Uint32(0xFFFF)
                            as_bits = Int32(low16 << Uint32(16))
                            normed_round = _bitcast_i32_to_f32(as_bits)
                            out_f32 = normed_round * gamma_f32
                            _st_global_bf16_from_f32(
                                next_hidden_base + Int64(idx) * Int64(2),
                                out_f32,
                            )
                    else:
                        # Last-layer case: next_hidden = residual_final.
                        for _i in cutlass.range_constexpr(n_per_thr):
                            idx = tid + Int32(_i * self._num_threads)
                            rf = _ld_global_b16_to_f32(
                                res_base + Int64(idx) * Int64(2)
                            )
                            _st_global_bf16_from_f32(
                                next_hidden_base + Int64(idx) * Int64(2),
                                rf,
                            )

                    # --- Self-reset secondary barrier ------------------
                    # atomicAdd(-num_k_tiles) restores barrier_ptr[bz]=0.
                    if tid == Int32(0):
                        _atomic_add_u32(
                            barrier_ptr + Int64(bz) * Int64(4),
                            Int32(0) - num_k_tiles,
                        )


