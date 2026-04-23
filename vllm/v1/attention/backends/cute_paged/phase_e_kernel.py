# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright contributors to the nvllm fork
"""Phase E β kernel — one cooperative launch per fusion-active decoder layer.

Phases (spec §3.1):
    0. prologue input_layernorm   (single-CTA-per-seq; broadcast via SMEM)
    1. attn (A+B+C)               (slice_ctas×num_k_tiles CTAs)
    2. grid barrier               (arrival counter + _threadfence)
    3. MLP (D)                    (same grid as phase 1, reuses SMEM)
    4. ε epilogue                 (residual_final + RMSNorm(next_γ))

Tasks: 11 (this file — skeleton + phase 0), 12 (phase 1), 13 (phase 2),
14 (phase 3), 15 (phase 4).

Plan: docs/superpowers/plans/2026-04-22-unreal-kernel-phase-e-d25.md
"""
from __future__ import annotations

import logging
from typing import Optional

import torch

logger = logging.getLogger(__name__)


# --- CuTe DSL import guard (mirrors kernel.py / mlp_kernel.py) --------------
_CUTE_AVAILABLE = False
try:
    import cutlass
    from cutlass import cute
    from cutlass.cute.typing import (  # noqa: F401
        Float32,
        Int32,
        Int64,
        Uint32,
    )
    import cuda.bindings.driver as _cuda_driver

    # Reuse PTX helpers already defined in kernel.py — they are module-level
    # and guarded by the same _CUTE_AVAILABLE pattern.
    from vllm.v1.attention.backends.cute_paged.kernel import (  # noqa: E501
        _ld_global_b16_to_f32,
        _ld_shared_f32,
        _rsqrt_approx_f32,
        _st_global_bf16_from_f32,
        _st_shared_f32,
        shared_ptr_to_i64,
        shfl_xor_sync,
    )

    _CUTE_AVAILABLE = True
except ImportError:
    logger.warning(
        "CuTe DSL not available (CUTLASS not installed). "
        "PhaseE_Beta_Kernel cannot be used."
    )


class PhaseE_Beta_Kernel:
    """Cooperative CuTe kernel fusing attn + MLP + residual + next-norm.

    β-coop grid (after Task 16): (slice_ctas=8, num_k_tiles=8, num_seqs)
        → 64 × num_seqs CTAs, single cooperative launch per layer.
    β-lite fallback (already shipped): two-kernel path (attn + Phase_D ε).

    Phase-0-only debug grid (this task): (1, 1, num_seqs).

    Block: (num_threads=128, 1, 1).
    SMEM: 45568 B budget (attn K/V tiles dominate; phase 0 uses only the
    cross-warp reduce scratch, 16 B).

    See spec §3.1 and plan
    docs/superpowers/plans/2026-04-22-unreal-kernel-phase-e-d25.md.
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_attn_heads: int,
        num_kv_heads: int,
        head_dim: int,
        rms_eps: float = 1e-6,
    ):
        assert hidden_size % 128 == 0, (
            f"hidden_size={hidden_size} must be divisible by num_threads=128 "
            f"for Phase E Phase 0 / ε epilogue to cover all elements without "
            f"a tail loop"
        )
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_attn_heads = num_attn_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.rms_eps = rms_eps
        # Grid constants — reserved for Tasks 12-16; phase-0-only uses (1,1,N).
        self.slice_ctas = 8
        self.num_k_tiles = 8
        self.num_threads = 128
        # SMEM budget (attn K/V tiles dominate once phases 1-4 land).
        self.smem_bytes = 45568
        # Phase-0-only SMEM: just 16 B for cross-warp reduction (4 warps × 4 B).
        self._smem_bytes_phase_0 = 16
        self._compiled_phase_0 = None

    # -----------------------------------------------------------------
    # Python-level debug entry point (phase-0-only).
    # -----------------------------------------------------------------
    def run_phase_0_only(
        self,
        hidden_in: torch.Tensor,    # [nat, hidden] BF16
        residual_in: torch.Tensor,  # [nat, hidden] BF16
        gamma: torch.Tensor,        # [hidden]      BF16
        normed_out: torch.Tensor,   # [nat, hidden] BF16 (written)
    ) -> torch.Tensor:
        """Launches only phase 0: hidden_normed = RMSNorm(hidden+residual) * γ.

        Task 11 debug harness. Real β-coop launch (Task 16) will cover all
        5 phases in a single cooperative kernel.
        """
        if not _CUTE_AVAILABLE:
            raise RuntimeError(
                "PhaseE_Beta_Kernel requires CUTLASS; not available."
            )
        nat, hidden = hidden_in.shape
        assert hidden == self.hidden_size
        assert residual_in.shape == hidden_in.shape
        assert gamma.shape == (hidden,)
        assert normed_out.shape == hidden_in.shape
        for t in (hidden_in, residual_in, gamma, normed_out):
            assert t.is_contiguous()
            assert t.dtype == torch.bfloat16

        hidden_in_ptr = Int64(hidden_in.data_ptr())
        residual_in_ptr = Int64(residual_in.data_ptr())
        gamma_ptr = Int64(gamma.data_ptr())
        normed_out_ptr = Int64(normed_out.data_ptr())
        rms_eps_f32 = Float32(float(self.rms_eps))

        stream_arg = _cuda_driver.CUstream(
            int(torch.cuda.current_stream().cuda_stream)
        )

        if self._compiled_phase_0 is None:
            logger.info(
                "Compiling PhaseE_Beta_Kernel phase-0-only (first call)…"
            )
            self._compiled_phase_0 = cute.compile(
                self._jit_launch_phase_0,
                hidden_in_ptr,
                residual_in_ptr,
                gamma_ptr,
                normed_out_ptr,
                Int32(nat),
                Int32(hidden),
                rms_eps_f32,
                stream_arg,
            )

        self._compiled_phase_0(
            hidden_in_ptr,
            residual_in_ptr,
            gamma_ptr,
            normed_out_ptr,
            Int32(nat),
            Int32(hidden),
            rms_eps_f32,
            stream_arg,
        )
        return normed_out

    # -----------------------------------------------------------------
    # JIT host wrapper + @cute.kernel body (phase-0-only for now).
    # -----------------------------------------------------------------
    if _CUTE_AVAILABLE:

        @cute.jit
        def _jit_launch_phase_0(
            self,
            hidden_in_ptr: Int64,
            residual_in_ptr: Int64,
            gamma_ptr: Int64,
            normed_out_ptr: Int64,
            nat: Int32,
            hidden_dim: Int32,
            rms_eps: Float32,
            stream,
        ):
            """JIT host wrapper for the phase-0-only kernel launch."""
            self._kernel_phase_0(
                hidden_in_ptr,
                residual_in_ptr,
                gamma_ptr,
                normed_out_ptr,
                hidden_dim,
                rms_eps,
            ).launch(
                grid=[1, 1, nat],
                block=[self.num_threads, 1, 1],
                smem=self._smem_bytes_phase_0,
                stream=stream,
            )

        @cute.kernel
        def _kernel_phase_0(
            self,
            hidden_in_ptr: Int64,
            residual_in_ptr: Int64,
            gamma_ptr: Int64,
            normed_out_ptr: Int64,
            hidden_dim: Int32,
            rms_eps: Float32,
        ):
            """Phase 0: single-CTA-per-seq input_layernorm prologue.

            Structure (mirrors Phase C RMSNorm in kernel.py):
                Pass 1 — per-thread sum-of-squares of (hidden+residual)
                Pass 2 — warp-shuffle reduction + cross-warp SMEM; one
                         thread computes inv_rms = rsqrt(var+ε) and
                         broadcasts via SMEM slot 0.
                Pass 3 — re-read (L2-hot), scale by inv_rms·γ, store BF16.

            Grid (1,1,nat), block (128,1,1): each thread owns
            `hidden // 128` = 40 elements (Qwen3.5/3.6-27B with hidden=5120).
            """
            # --- Phase 1: attn (A+B+C) ---   filled in Task 12 (β-coop path)
            # --- Phase 2: grid barrier ---   filled in Task 13 (β-coop path)
            # --- Phase 3: MLP (D) ---        filled in Task 14 (β-coop path)
            # --- Phase 4: ε epilogue ---     filled in Task 15 (β-coop path)

            # --- Thread / block identification ---
            seq_idx = cute.arch.block_idx()[2]
            tid = cute.arch.thread_idx()[0]
            warp = tid >> Int32(5)
            lane = tid & Int32(31)

            # --- SMEM: 4 FP32 slots for cross-warp reduction (16 B) ---
            smem = cute.arch.get_dyn_smem(cutlass.Uint8, alignment=128)
            sync_md = shared_ptr_to_i64(smem)

            hd_c = hidden_dim
            n_per_thr_c = hd_c // Int32(128)  # = 40 for hidden=5120

            # Base pointers (BF16 = 2 bytes per element).
            hidden_base = hidden_in_ptr + Int64(seq_idx * hd_c * Int32(2))
            residual_base = residual_in_ptr + Int64(
                seq_idx * hd_c * Int32(2))
            gamma_base = gamma_ptr  # γ shared across sequences
            out_base = normed_out_ptr + Int64(seq_idx * hd_c * Int32(2))

            my_start = tid * n_per_thr_c

            # --- Pass 1: sum-of-squares of (hidden + residual) ---
            # range_constexpr unroll (hidden//num_threads = 40 < 100 limit).
            ss = Float32(0.0)
            n_per_thr_py = self.hidden_size // self.num_threads
            for _i in cutlass.range_constexpr(n_per_thr_py):
                idx_c = my_start + Int32(_i)
                h_f32 = _ld_global_b16_to_f32(
                    hidden_base + Int64(idx_c * Int32(2)))
                r_f32 = _ld_global_b16_to_f32(
                    residual_base + Int64(idx_c * Int32(2)))
                s = h_f32 + r_f32
                ss = ss + s * s

            # --- Pass 2: warp-shuffle reduction + cross-warp SMEM ---
            ss = ss + shfl_xor_sync(ss, Int32(1))
            ss = ss + shfl_xor_sync(ss, Int32(2))
            ss = ss + shfl_xor_sync(ss, Int32(4))
            ss = ss + shfl_xor_sync(ss, Int32(8))
            ss = ss + shfl_xor_sync(ss, Int32(16))

            # Lane 0 of each warp writes its partial; warp 0 lane 0 sums.
            if lane == Int32(0):
                _st_shared_f32(sync_md + Int64(warp * Int32(4)), ss)
            cute.arch.sync_threads()

            if warp == Int32(0):
                if lane == Int32(0):
                    total_ss = _ld_shared_f32(sync_md)
                    total_ss = total_ss + _ld_shared_f32(
                        sync_md + Int64(4))
                    total_ss = total_ss + _ld_shared_f32(
                        sync_md + Int64(8))
                    total_ss = total_ss + _ld_shared_f32(
                        sync_md + Int64(12))
                    variance = total_ss / Float32(hd_c)
                    inv_rms = _rsqrt_approx_f32(variance + rms_eps)
                    _st_shared_f32(sync_md, inv_rms)
            cute.arch.sync_threads()

            inv_rms_val = _ld_shared_f32(sync_md)

            # --- Pass 3: re-read (L2-hot), scale by inv_rms·γ, write BF16 ---
            for _i in cutlass.range_constexpr(n_per_thr_py):
                idx_c = my_start + Int32(_i)
                h_f32 = _ld_global_b16_to_f32(
                    hidden_base + Int64(idx_c * Int32(2)))
                r_f32 = _ld_global_b16_to_f32(
                    residual_base + Int64(idx_c * Int32(2)))
                gamma_f32 = _ld_global_b16_to_f32(
                    gamma_base + Int64(idx_c * Int32(2)))
                normed = (h_f32 + r_f32) * inv_rms_val * gamma_f32
                _st_global_bf16_from_f32(
                    out_base + Int64(idx_c * Int32(2)), normed)
