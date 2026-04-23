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

    from cutlass.cute.typing import BFloat16  # noqa: F401

    # Reuse PTX helpers already defined in kernel.py — they are module-level
    # and guarded by the same _CUTE_AVAILABLE pattern.
    from vllm.v1.attention.backends.cute_paged.kernel import (  # noqa: E501
        _acquire_fence,
        _atomic_add_u32,
        _cvt_2f32_to_bf16x2,
        _extract_byte_from_b32,
        _fmax,
        _fp4_nibble_to_f32,
        _ld_global_b16_to_f32,
        _ld_global_b32,
        _ld_global_f32,
        _ld_shared_b16,
        _ld_shared_b32,
        _ld_shared_f32,
        _ld_swizzled_scale,
        _ld_volatile_u32,
        _pack_4bytes,
        _pack_lo16,
        _rcp_approx_f32,
        _rsqrt_approx_f32,
        _st_global_bf16_from_f32,
        _st_global_f32,
        _st_shared_b16_from_u32,
        _st_shared_b32,
        _st_shared_f32,
        _threadfence,
        bf16_mma_m16n16k16_f32,
        exp2_approx_ftz_f32,
        fp8x4_e4m3_to_bfloat2x2,
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

        # --- Phase-0+1 debug kernel (Task 12) constants ---------------------
        # Mirrors DecodeKernel SMEM layout: Q[cta_q,hd] BF16 + K[cta_kv,hd] FP8
        # + V[cta_kv,hd] FP8 + sync_o_small + sync_md_small. See kernel.py:967.
        self._cta_q = 16
        self._cta_kv = 64
        self._block_size = 64
        self._num_warps_kv = 4
        self._num_mma_d = head_dim // 16  # 16 for head_dim=256
        _q_bytes = self._cta_q * head_dim * 2      # BF16
        _k_bytes = self._cta_kv * head_dim * 1     # FP8
        _v_bytes = self._cta_kv * head_dim * 1     # FP8
        _qkv_bytes = _q_bytes + _k_bytes + _v_bytes
        _sync_o_small_bytes = self._num_warps_kv * self._cta_q * 16 * 4
        _sync_md_small_bytes = self._num_warps_kv * self._cta_q * 8
        self._q_bytes = _q_bytes
        self._k_bytes = _k_bytes
        self._v_bytes = _v_bytes
        self._sync_o_small_offset = _qkv_bytes
        self._sync_md_small_offset = _qkv_bytes + _sync_o_small_bytes
        self._smem_bytes_phase_01 = (
            _qkv_bytes + _sync_o_small_bytes + _sync_md_small_bytes
        )  # 45568 for Qwen3.5-27B decode config
        self._compiled_phase_01 = None

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
    # Python-level debug entry point (phase-0+1-only, Task 12).
    # -----------------------------------------------------------------
    def run_phase_01_only(
        self,
        hidden_in: torch.Tensor,        # [nat, hidden]   BF16
        residual_in: torch.Tensor,      # [nat, hidden]   BF16
        input_gamma: torch.Tensor,      # [hidden]        BF16
        post_attn_gamma: torch.Tensor,  # [hidden]        BF16
        attn_input_bf16: torch.Tensor,  # [nat, hidden]   BF16 (Phase 0 out)
        query: torch.Tensor,            # [nat, num_q_heads, head_dim] BF16
        kv_cache: torch.Tensor,         # [pg, 2, ps, kv, hd] uint8 FP8
        page_table: torch.Tensor,       # [nat, max_pages] int32
        seq_lens: torch.Tensor,         # [nat] int32
        wo_weight: torch.Tensor,        # [hidden, K/2]   uint8 (NVFP4)
        wo_scales: torch.Tensor,        # [hidden, K_sf]  fp8_e4m3fn (swizzled)
        wo_global_scale: torch.Tensor,  # scalar f32
        attn_output: torch.Tensor,      # [nat, hidden]   BF16 (written by Phase C)
        scale: float = None,
        k_scale: float = 1.0,
        v_scale: float = 1.0,
    ) -> torch.Tensor:
        """Launch phases 0+1 only: attn_input_bf16 = RMSNorm(h+r)*γ_in,
        attn_output = PostAttnRMSNorm(h+r + W_O @ Attn(Q,K,V)).

        Task 12 debug harness. Mirrors the Phase A+B+C fused path of
        DecodeKernel (see kernel.py:955). Grid: (1, 4, nat) — 4 kv-head
        CTAs per seq run attn; no idle CTAs in this debug launch.

        Phase 0 prologue is gated to cta_y==0 (single CTA per seq writes
        attn_input_bf16) to avoid 4-way write races on the same BF16 buffer.

        Note: QKV projection is NOT fused here; caller must pass `query`
        already projected. attn_input_bf16 is the input the next layer's
        QKV would consume — Phase 1 of THIS layer runs attn over `query`.
        """
        if not _CUTE_AVAILABLE:
            raise RuntimeError(
                "PhaseE_Beta_Kernel requires CUTLASS; not available."
            )
        nat, hidden = hidden_in.shape
        assert hidden == self.hidden_size
        assert residual_in.shape == hidden_in.shape
        assert input_gamma.shape == (hidden,)
        assert post_attn_gamma.shape == (hidden,)
        assert attn_input_bf16.shape == hidden_in.shape
        assert attn_output.shape == hidden_in.shape
        assert query.shape == (nat, self.num_attn_heads, self.head_dim)
        for t in (hidden_in, residual_in, input_gamma, post_attn_gamma,
                  attn_input_bf16, attn_output, query,
                  kv_cache, page_table, seq_lens,
                  wo_weight, wo_scales, wo_global_scale):
            assert t.is_contiguous(), f"{t.shape} must be contiguous"
        assert query.dtype == torch.bfloat16
        assert hidden_in.dtype == torch.bfloat16

        if scale is None:
            scale = 1.0 / (self.head_dim ** 0.5)

        # --- Allocate workspace buffers -----------------------------------
        # wo_output slots: 4 attn CTAs × nat; layout [nat, 4, hidden] FP32.
        total_ctas_per_seq = 4   # grid_x=1, grid_y=4
        wo_output = torch.zeros(
            nat, total_ctas_per_seq, hidden,
            dtype=torch.float32, device=hidden_in.device,
        )
        arrival_count = torch.zeros(
            nat, dtype=torch.int32, device=hidden_in.device,
        )
        # Phase C residual_output (new_residual for next block's input).
        residual_output = torch.empty(
            nat, hidden, dtype=torch.bfloat16, device=hidden_in.device,
        )

        # --- Flatten multi-dim tensors: CuTe DSL cannot flat-index >1D. ----
        q_flat = query.view(-1)

        # --- Derived pointers / scalars -----------------------------------
        # KV cache: [pg, 2, ps, kv, hd] — K is slot 0, V is slot 1.
        kv_base = Int64(kv_cache.data_ptr())
        kv_slot_stride = Int64(
            kv_cache.stride(1) * kv_cache.element_size()
        )
        k_ptr = kv_base
        v_ptr = kv_base + kv_slot_stride
        kv_page_stride = Int32(
            kv_cache.stride(0) * kv_cache.element_size()
        )

        hidden_in_ptr = Int64(hidden_in.data_ptr())
        residual_in_ptr = Int64(residual_in.data_ptr())
        input_gamma_ptr = Int64(input_gamma.data_ptr())
        post_attn_gamma_ptr = Int64(post_attn_gamma.data_ptr())
        attn_input_bf16_ptr = Int64(attn_input_bf16.data_ptr())
        attn_output_ptr = Int64(attn_output.data_ptr())
        residual_output_ptr = Int64(residual_output.data_ptr())
        wo_weight_ptr = Int64(wo_weight.data_ptr())
        wo_scale_ptr = Int64(wo_scales.data_ptr())
        wo_output_ptr = Int64(wo_output.data_ptr())
        wo_gs_ptr = Int64(wo_global_scale.data_ptr())
        arrival_count_ptr = Int64(arrival_count.data_ptr())

        # NVFP4 scale-swizzle numKTiles = ceil(K/64), K = num_q_heads * head_dim.
        wo_K = self.num_attn_heads * self.head_dim
        wo_nkt = Int32((wo_K // 16 + 3) // 4)
        wo_row_stride = Int32(wo_weight.shape[1])

        rms_eps_f32 = Float32(float(self.rms_eps))

        # Combine attn scale with k_scale for the kernel (it multiplies by
        # log2(e) internally — Phase 3 in DecodeKernel._kernel).
        stream_arg = _cuda_driver.CUstream(
            int(torch.cuda.current_stream().cuda_stream)
        )

        all_args = (
            hidden_in_ptr,
            residual_in_ptr,
            input_gamma_ptr,
            post_attn_gamma_ptr,
            attn_input_bf16_ptr,
            attn_output_ptr,
            residual_output_ptr,
            q_flat,
            k_ptr,
            v_ptr,
            page_table,
            seq_lens,
            wo_weight_ptr,
            wo_scale_ptr,
            wo_output_ptr,
            wo_gs_ptr,
            arrival_count_ptr,
            Int32(self.num_attn_heads),
            Int32(self.num_kv_heads),
            kv_page_stride,
            wo_nkt,
            wo_row_stride,
            Int32(total_ctas_per_seq),
            Int32(hidden),
            Float32(float(scale)),
            Float32(float(k_scale)),
            Float32(float(v_scale)),
            rms_eps_f32,
            Int32(nat),
            stream_arg,
        )

        if self._compiled_phase_01 is None:
            logger.info(
                "Compiling PhaseE_Beta_Kernel phase-0+1 (first call)…"
            )
            self._compiled_phase_01 = cute.compile(
                self._jit_launch_phase_01,
                *all_args,
            )

        self._compiled_phase_01(*all_args)
        return attn_output

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

        # -----------------------------------------------------------------
        # Task 12: Phase-0+1 debug kernel — prologue + attn (A+B+C).
        # Launched with grid=(1, 4, nat) so all 4 kv-head CTAs run attn;
        # Phase 0 prologue gated to cta_y==0 so only one CTA per seq writes
        # attn_input_bf16. Uber-kernel launch (Task 16) will replace this
        # with grid=(8, 8, nat) and add idle-CTA gating.
        # -----------------------------------------------------------------
        @cute.jit
        def _jit_launch_phase_01(
            self,
            hidden_in_ptr: Int64,
            residual_in_ptr: Int64,
            input_gamma_ptr: Int64,
            post_attn_gamma_ptr: Int64,
            attn_input_bf16_ptr: Int64,
            attn_output_ptr: Int64,
            residual_output_ptr: Int64,
            query,
            k_ptr: Int64,
            v_ptr: Int64,
            page_table,
            seq_lens,
            wo_weight_ptr: Int64,
            wo_scale_ptr: Int64,
            wo_output_ptr: Int64,
            wo_gs_ptr: Int64,
            arrival_count_ptr: Int64,
            num_q_heads: Int32,
            num_kv_heads: Int32,
            kv_page_stride: Int32,
            wo_num_k_tiles: Int32,
            wo_weight_row_stride: Int32,
            total_ctas_per_seq: Int32,
            hidden_dim: Int32,
            scale: Float32,
            k_scale: Float32,
            v_scale: Float32,
            rms_eps: Float32,
            nat: Int32,
            stream,
        ):
            """JIT host wrapper for the phase-0+1 debug launch."""
            self._kernel_phase_01(
                hidden_in_ptr,
                residual_in_ptr,
                input_gamma_ptr,
                post_attn_gamma_ptr,
                attn_input_bf16_ptr,
                attn_output_ptr,
                residual_output_ptr,
                query,
                k_ptr,
                v_ptr,
                page_table,
                seq_lens,
                wo_weight_ptr,
                wo_scale_ptr,
                wo_output_ptr,
                wo_gs_ptr,
                arrival_count_ptr,
                num_q_heads,
                num_kv_heads,
                kv_page_stride,
                wo_num_k_tiles,
                wo_weight_row_stride,
                total_ctas_per_seq,
                hidden_dim,
                scale,
                k_scale,
                v_scale,
                rms_eps,
            ).launch(
                grid=[1, 4, nat],
                block=[self.num_threads, 1, 1],
                smem=self._smem_bytes_phase_01,
                stream=stream,
            )

        @cute.kernel
        def _kernel_phase_01(
            self,
            hidden_in_ptr: Int64,
            residual_in_ptr: Int64,
            input_gamma_ptr: Int64,
            post_attn_gamma_ptr: Int64,
            attn_input_bf16_ptr: Int64,
            attn_output_ptr: Int64,
            residual_output_ptr: Int64,
            query,
            k_ptr: Int64,
            v_ptr: Int64,
            page_table,
            seq_lens,
            wo_weight_ptr: Int64,
            wo_scale_ptr: Int64,
            wo_output_ptr: Int64,
            wo_gs_ptr: Int64,
            arrival_count_ptr: Int64,
            num_q_heads: Int32,
            num_kv_heads: Int32,
            kv_page_stride: Int32,
            wo_num_k_tiles: Int32,
            wo_weight_row_stride: Int32,
            total_ctas_per_seq: Int32,
            hidden_dim: Int32,
            scale: Float32,
            k_scale: Float32,
            v_scale: Float32,
            rms_eps: Float32,
        ):
            """Phase 0 (input RMSNorm) + Phase 1 (attn A+B+C) combined.

            Structure:
              Phase 0 (cta_y==0 only): same as _kernel_phase_0 — writes
                  attn_input_bf16 = RMSNorm(hidden_in + residual_in) * γ_in.
              Phase 1 (cta_x==0 && cta_y<4 — all 4 here in debug mode):
                  Mirrors DecodeKernel._kernel body: Q load, _md serialized
                  loop with QK+softmax+PV, Phase B W_O GEMV, Phase B.5 gather,
                  Phase C post-attn RMSNorm. Output written to attn_output.
            """
            bx, by, bz = cute.arch.block_idx()
            lane = cute.arch.lane_idx()
            warp = cute.arch.warp_idx()
            tid = warp * Int32(32) + lane
            group = lane >> Int32(2)
            sub = lane & Int32(3)

            kv_head_idx = by
            seq_idx = bz
            group_size = num_q_heads // num_kv_heads
            q_head_start = kv_head_idx * group_size + bx * Int32(self._cta_q)

            # -----------------------------------------------------------------
            # SMEM layout (same as DecodeKernel): Q | K | V | sync_o | sync_md.
            # Phase 0 reuses the sync_md slot as its 4-warp reduce scratch
            # (it only runs when no attn state is live in that region, so
            # overlapping is safe — same pattern as kernel.py's Phase C).
            # -----------------------------------------------------------------
            smem = cute.arch.get_dyn_smem(cutlass.Uint8, alignment=128)
            q_smem = shared_ptr_to_i64(smem)
            k_smem = shared_ptr_to_i64(
                smem + Int32(self._q_bytes))
            v_smem = shared_ptr_to_i64(
                smem + Int32(self._q_bytes + self._k_bytes))
            sync_o = shared_ptr_to_i64(
                smem + Int32(self._sync_o_small_offset))
            sync_md = shared_ptr_to_i64(
                smem + Int32(self._sync_md_small_offset))

            # =================================================================
            # Phase 0: input_layernorm (single CTA per seq — cta_y==0).
            # =================================================================
            if by == Int32(0):
                hd_c0 = hidden_dim
                n_per_thr_0 = self.hidden_size // self.num_threads  # 40
                hidden_base_0 = hidden_in_ptr + Int64(
                    seq_idx * hd_c0 * Int32(2))
                residual_base_0 = residual_in_ptr + Int64(
                    seq_idx * hd_c0 * Int32(2))
                gamma_base_0 = input_gamma_ptr
                out_base_0 = attn_input_bf16_ptr + Int64(
                    seq_idx * hd_c0 * Int32(2))
                my_start_0 = tid * Int32(n_per_thr_0)

                ss0 = Float32(0.0)
                for _i in cutlass.range_constexpr(n_per_thr_0):
                    idx_c0 = my_start_0 + Int32(_i)
                    h_f32 = _ld_global_b16_to_f32(
                        hidden_base_0 + Int64(idx_c0 * Int32(2)))
                    r_f32 = _ld_global_b16_to_f32(
                        residual_base_0 + Int64(idx_c0 * Int32(2)))
                    s = h_f32 + r_f32
                    ss0 = ss0 + s * s

                ss0 = ss0 + shfl_xor_sync(ss0, Int32(1))
                ss0 = ss0 + shfl_xor_sync(ss0, Int32(2))
                ss0 = ss0 + shfl_xor_sync(ss0, Int32(4))
                ss0 = ss0 + shfl_xor_sync(ss0, Int32(8))
                ss0 = ss0 + shfl_xor_sync(ss0, Int32(16))

                if lane == Int32(0):
                    _st_shared_f32(sync_md + Int64(warp * Int32(4)), ss0)
                cute.arch.sync_threads()

                if warp == Int32(0):
                    if lane == Int32(0):
                        total_ss0 = _ld_shared_f32(sync_md)
                        total_ss0 = total_ss0 + _ld_shared_f32(
                            sync_md + Int64(4))
                        total_ss0 = total_ss0 + _ld_shared_f32(
                            sync_md + Int64(8))
                        total_ss0 = total_ss0 + _ld_shared_f32(
                            sync_md + Int64(12))
                        variance0 = total_ss0 / Float32(hd_c0)
                        inv_rms0 = _rsqrt_approx_f32(variance0 + rms_eps)
                        _st_shared_f32(sync_md, inv_rms0)
                cute.arch.sync_threads()

                inv_rms_val_0 = _ld_shared_f32(sync_md)

                for _i in cutlass.range_constexpr(n_per_thr_0):
                    idx_c0 = my_start_0 + Int32(_i)
                    h_f32 = _ld_global_b16_to_f32(
                        hidden_base_0 + Int64(idx_c0 * Int32(2)))
                    r_f32 = _ld_global_b16_to_f32(
                        residual_base_0 + Int64(idx_c0 * Int32(2)))
                    gamma_f32 = _ld_global_b16_to_f32(
                        gamma_base_0 + Int64(idx_c0 * Int32(2)))
                    normed = (h_f32 + r_f32) * inv_rms_val_0 * gamma_f32
                    _st_global_bf16_from_f32(
                        out_base_0 + Int64(idx_c0 * Int32(2)), normed)

                # Sync before Phase 1 reuses sync_md/sync_o for attn state.
                cute.arch.sync_threads()

            # =================================================================
            # Phase 1: Attn A+B+C — gated to attn-active CTAs.
            # Grid=(1,4,nat), so cta_x==0 and cta_y<4 unconditionally here,
            # but we keep the gate so Task 16's full (8,8,N) grid is correct.
            # =================================================================
            if bx == Int32(0):
                if by < Int32(4):
                    # --- Combined scales (mirrors DecodeKernel Phase 0 tail) ---
                    LOG2E = Float32(1.4426950408889634)
                    sm_scale_log2 = scale * k_scale * LOG2E
                    v_scale_f32 = v_scale

                    seq_len = seq_lens[seq_idx]
                    num_pages = (seq_len
                                 + Int32(self._block_size - 1)) \
                        // Int32(self._block_size)

                    hd = Int32(self.head_dim)
                    warp_kv_start = warp * Int32(16)
                    kv_tok_stride = num_kv_heads * hd

                    # --- Phase 2 of DecodeKernel: load Q into SMEM ---
                    q_stride_tok = num_q_heads * hd
                    elems_per_thr_q = Int32(
                        self._cta_q * self.head_dim
                        // self.num_threads)
                    for _i in cutlass.range_constexpr(
                        self._cta_q * self.head_dim
                        // self.num_threads
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

                    # --- Phase 3: serialized _md loop ---
                    for _md_c in cutlass.range_constexpr(self._num_mma_d):
                        _md_idx = Int32(_md_c)
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

                        page_idx = Int32(0)
                        while page_idx < num_pages:
                            phys_page = page_table[seq_idx, page_idx]

                            # Load K page (row-major, 4B/iter)
                            elems_per_thr_kv4 = Int32(
                                self._cta_kv * self.head_dim
                                // 4 // self.num_threads)
                            for _i in cutlass.range_constexpr(
                                self._cta_kv * self.head_dim
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

                            # Load V page
                            for _i in cutlass.range_constexpr(
                                self._cta_kv * self.head_dim
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

                            # QK MMA (all 16 K-dim iterations)
                            s0 = Float32(0.0)
                            s1 = Float32(0.0)
                            s2 = Float32(0.0)
                            s3 = Float32(0.0)
                            s4 = Float32(0.0)
                            s5 = Float32(0.0)
                            s6 = Float32(0.0)
                            s7 = Float32(0.0)

                            for _kd in cutlass.range_constexpr(
                                self._num_mma_d
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
                                k_packed_0 = _pack_lo16(
                                    k_raw_0a, k_raw_0b)
                                b0, b1 = fp8x4_e4m3_to_bfloat2x2(
                                    k_packed_0)

                                kv_row_1 = warp_kv_start + n_t + Int32(8)
                                k_off_1a = (kv_row_1 * hd + k_start
                                            + sub * Int32(2))
                                k_raw_1a = _ld_shared_b16(
                                    k_smem + Int64(k_off_1a))
                                k_raw_1b = _ld_shared_b16(
                                    k_smem + Int64(k_off_1a + Int32(8)))
                                k_packed_1 = _pack_lo16(
                                    k_raw_1a, k_raw_1b)
                                b2, b3 = fp8x4_e4m3_to_bfloat2x2(
                                    k_packed_1)

                                (s0, s1, s2, s3,
                                 s4, s5, s6, s7) = bf16_mma_m16n16k16_f32(
                                    s0, s1, s2, s3, s4, s5, s6, s7,
                                    a0, a1, a2, a3,
                                    b0, b1, b2, b3)

                            # Online softmax
                            s0 = s0 * sm_scale_log2
                            s1 = s1 * sm_scale_log2
                            s2 = s2 * sm_scale_log2
                            s3 = s3 * sm_scale_log2
                            s4 = s4 * sm_scale_log2
                            s5 = s5 * sm_scale_log2
                            s6 = s6 * sm_scale_log2
                            s7 = s7 * sm_scale_log2

                            tok_base = page_idx * Int32(self._block_size) \
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

                            # PV MMA for current _md_idx
                            pa0 = _cvt_2f32_to_bf16x2(
                                p0 * v_scale_f32, p1 * v_scale_f32)
                            pa1 = _cvt_2f32_to_bf16x2(
                                p4 * v_scale_f32, p5 * v_scale_f32)
                            pa2 = _cvt_2f32_to_bf16x2(
                                p2 * v_scale_f32, p3 * v_scale_f32)
                            pa3 = _cvt_2f32_to_bf16x2(
                                p6 * v_scale_f32, p7 * v_scale_f32)

                            v_k_start = _md_idx * Int32(16)
                            v_tok0 = warp_kv_start + sub * Int32(2)

                            v_hd0 = v_k_start + group
                            v_off_0a = v_tok0 * hd + v_hd0
                            v_off_0b = (v_tok0 + Int32(1)) * hd + v_hd0
                            v_off_8a = (v_tok0 + Int32(8)) * hd + v_hd0
                            v_off_8b = (v_tok0 + Int32(9)) * hd + v_hd0
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

                        # === Write 8 accum values + m,d to sync buffers ===
                        W16 = Int32(16)
                        so_warp_off = warp * Int32(self._cta_q) * W16 \
                            * Int32(4)
                        so_r0 = so_warp_off + group * W16 * Int32(4)
                        so_r1 = so_warp_off + (group + Int32(8)) \
                            * W16 * Int32(4)
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

                        if sub == Int32(0):
                            md_w_off = warp * Int32(self._cta_q) \
                                * Int32(8)
                            _st_shared_f32(sync_md + Int64(
                                md_w_off + group * Int32(8)), m_r0)
                            _st_shared_f32(sync_md + Int64(
                                md_w_off + group * Int32(8)
                                + Int32(4)), d_r0)
                            _st_shared_f32(sync_md + Int64(
                                md_w_off
                                + (group + Int32(8)) * Int32(8)),
                                m_r1)
                            _st_shared_f32(sync_md + Int64(
                                md_w_off
                                + (group + Int32(8)) * Int32(8)
                                + Int32(4)), d_r1)

                        cute.arch.sync_threads()

                        # === Cross-warp reduction (warp 0 only) ===
                        if warp == Int32(0):
                            red_row = lane >> Int32(1)
                            col_base = (lane & Int32(1)) * Int32(8)

                            for _e in cutlass.range_constexpr(8):
                                col16 = col_base + Int32(_e)

                                m_final = Float32(-1e30)
                                for _w in cutlass.range_constexpr(
                                    self._num_warps_kv
                                ):
                                    m_w = _ld_shared_f32(
                                        sync_md + Int64(
                                            Int32(_w * self._cta_q)
                                            * Int32(8)
                                            + red_row * Int32(8)))
                                    m_final = _fmax(m_final, m_w)

                                o_final = Float32(0.0)
                                d_final = Float32(0.0)
                                for _w in cutlass.range_constexpr(
                                    self._num_warps_kv
                                ):
                                    w_base = Int32(
                                        _w * self._cta_q * 16)
                                    o_w = _ld_shared_f32(
                                        sync_o + Int64(
                                            (w_base + red_row * W16
                                             + col16) * Int32(4)))
                                    m_w = _ld_shared_f32(
                                        sync_md + Int64(
                                            Int32(_w * self._cta_q)
                                            * Int32(8)
                                            + red_row * Int32(8)))
                                    d_w = _ld_shared_f32(
                                        sync_md + Int64(
                                            Int32(_w * self._cta_q)
                                            * Int32(8)
                                            + red_row * Int32(8)
                                            + Int32(4)))
                                    rescale = exp2_approx_ftz_f32(
                                        m_w - m_final)
                                    o_final = o_final + o_w * rescale
                                    d_final = d_final + d_w * rescale

                                o_final = o_final / d_final
                                out_head = q_head_start + red_row
                                g_col = _md_idx * Int32(16) + col16
                                if red_row < group_size:
                                    # Stage attn output in the per-seq
                                    # wo_output slot 0 as a scratch BF16
                                    # intermediate? No — DecodeKernel writes
                                    # to a real output tensor. For β-coop
                                    # Phase 1 we stage via sync_o-adjacent
                                    # SMEM to skip the global round-trip,
                                    # but that requires the Phase B loop
                                    # to read SMEM. DecodeKernel reads
                                    # global (Phase B body uses
                                    # `output[attn_base + k_idx]`) so for
                                    # byte-identical math we write global
                                    # BF16 into attn_output too — temp
                                    # scratch reusing the same buffer (is
                                    # overwritten by Phase C below).
                                    out_idx = (seq_idx * num_q_heads * hd
                                               + out_head * hd + g_col)
                                    _st_global_bf16_from_f32(
                                        attn_output_ptr
                                        + Int64(out_idx * Int32(2)),
                                        o_final)

                        cute.arch.sync_threads()
                    # end _md loop

                    # === Phase B: Fused W_O GEMV ===
                    # attn_output holds the attn BF16 scratch; read back
                    # (likely L2-hot) for the W_O multiply. Per-CTA slot
                    # writes into wo_output[seq, cta_idx, :] FP32.
                    _threadfence()
                    cute.arch.sync_threads()

                    attn_base = seq_idx * num_q_heads * hd \
                        + q_head_start * hd
                    hd_wo = Int32(self.hidden_size)
                    n_per_thr_wo = Int32(
                        self.hidden_size // self.num_threads)
                    my_row_base = tid * n_per_thr_wo

                    wo_gs = _ld_global_f32(wo_gs_ptr)

                    # 5 groups of 8 rows (matches DecodeKernel;
                    # requires hidden_size % 128 == 0).
                    n_groups_wo = self.hidden_size // self.num_threads // 8
                    for _out_group in cutlass.range_constexpr(
                        self.hidden_size // self.num_threads // 8
                    ):
                        out_base_wo = my_row_base \
                            + Int32(_out_group * 8)

                        a0 = Float32(0.0)
                        a1 = Float32(0.0)
                        a2 = Float32(0.0)
                        a3 = Float32(0.0)
                        a4 = Float32(0.0)
                        a5 = Float32(0.0)
                        a6 = Float32(0.0)
                        a7 = Float32(0.0)

                        k_dim = group_size * hd
                        k_idx = Int32(0)
                        while k_idx < k_dim:
                            attn_val = _ld_global_b16_to_f32(
                                attn_output_ptr
                                + Int64((attn_base + k_idx) * Int32(2)))
                            abs_k = kv_head_idx * group_size * hd + k_idx
                            k_byte = abs_k >> Int32(1)
                            k_is_hi = abs_k & Int32(1)
                            k_grp = abs_k >> Int32(4)

                            for _oi in cutlass.range_constexpr(8):
                                out_row = out_base_wo + Int32(_oi)
                                if out_row < hd_wo:
                                    w_addr = wo_weight_ptr + Int64(
                                        out_row * wo_weight_row_stride
                                        + k_byte)
                                    aligned = w_addr & Int64(
                                        0xFFFFFFFFFFFFFFFC)
                                    raw = _ld_global_b32(aligned)
                                    bpos = Int32(w_addr & Int64(3))
                                    the_byte = _extract_byte_from_b32(
                                        raw, bpos)
                                    nib_shift = k_is_hi << Int32(2)
                                    nib = (the_byte >> nib_shift) \
                                        & Int32(0x0F)
                                    w_f32 = _fp4_nibble_to_f32(nib)
                                    sf = _ld_swizzled_scale(
                                        wo_scale_ptr, out_row, k_grp,
                                        wo_num_k_tiles)
                                    w_dequant = w_f32 * sf * wo_gs

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

                        # Per-CTA slot write (deterministic gather).
                        cta_idx = bx * num_kv_heads + by
                        wo_slot_base = wo_output_ptr + Int64(
                            (seq_idx * total_ctas_per_seq + cta_idx)
                            * hd_wo * Int32(4))
                        for _oi in cutlass.range_constexpr(8):
                            out_row = out_base_wo + Int32(_oi)
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

                    # === Phase B.5 + C: last-CTA gather + RMSNorm ===
                    _threadfence()

                    if tid == Int32(0):
                        old_count = _atomic_add_u32(
                            arrival_count_ptr
                            + Int64(seq_idx * Int32(4)),
                            Int32(1))
                        if old_count == total_ctas_per_seq - Int32(1):
                            _st_shared_f32(sync_md, Float32(1.0))
                        else:
                            _st_shared_f32(sync_md, Float32(0.0))
                    cute.arch.sync_threads()

                    is_last_cta = _ld_shared_f32(sync_md)

                    if is_last_cta > Float32(0.5):
                        hd_c = hidden_dim
                        n_per_thr_c = hd_c // Int32(128)

                        res_base_c = residual_in_ptr + Int64(
                            seq_idx * hd_c * Int32(2))
                        wo_base_c = wo_output_ptr + Int64(
                            seq_idx * total_ctas_per_seq
                            * hd_c * Int32(4))
                        gamma_base_c = post_attn_gamma_ptr
                        out_base_c = attn_output_ptr + Int64(
                            seq_idx * hd_c * Int32(2))
                        resout_base_c = residual_output_ptr + Int64(
                            seq_idx * hd_c * Int32(2))

                        my_start_c = tid * n_per_thr_c

                        # Phase B.5: gather per-CTA slots into slot 0.
                        for _grp in cutlass.range_constexpr(
                            self.hidden_size // self.num_threads // 8
                        ):
                            for _ei in cutlass.range_constexpr(8):
                                idx_c = my_start_c + Int32(_grp * 8 + _ei)
                                gather_acc = Float32(0.0)
                                cta_i = Int32(0)
                                while cta_i < total_ctas_per_seq:
                                    slot_addr = wo_output_ptr + Int64(
                                        (seq_idx * total_ctas_per_seq
                                         + cta_i)
                                        * hd_c * Int32(4)
                                        + idx_c * Int32(4))
                                    gather_acc = gather_acc \
                                        + _ld_global_f32(slot_addr)
                                    cta_i = cta_i + Int32(1)
                                _st_global_f32(
                                    wo_base_c
                                    + Int64(idx_c * Int32(4)),
                                    gather_acc,
                                )
                        _threadfence()
                        cute.arch.sync_threads()

                        # Pass 1: residual add + sum-of-squares
                        ss = Float32(0.0)
                        for _grp in cutlass.range_constexpr(
                            self.hidden_size // self.num_threads // 8
                        ):
                            base_idx = my_start_c + Int32(_grp * 8)
                            for _ei in cutlass.range_constexpr(8):
                                idx_c = base_idx + Int32(_ei)
                                res_f32 = _ld_global_b16_to_f32(
                                    res_base_c
                                    + Int64(idx_c * Int32(2)))
                                wo_f32 = _ld_global_f32(
                                    wo_base_c
                                    + Int64(idx_c * Int32(4)))
                                nr = res_f32 + wo_f32
                                ss = ss + nr * nr

                        # Pass 2: reduction
                        ss = ss + shfl_xor_sync(ss, Int32(1))
                        ss = ss + shfl_xor_sync(ss, Int32(2))
                        ss = ss + shfl_xor_sync(ss, Int32(4))
                        ss = ss + shfl_xor_sync(ss, Int32(8))
                        ss = ss + shfl_xor_sync(ss, Int32(16))

                        if lane == Int32(0):
                            _st_shared_f32(
                                sync_md + Int64(warp * Int32(4)), ss)
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
                                inv_rms = _rsqrt_approx_f32(
                                    variance + rms_eps)
                                _st_shared_f32(sync_md, inv_rms)
                        cute.arch.sync_threads()

                        inv_rms_val = _ld_shared_f32(sync_md)

                        # Pass 3: re-read, scale, write BF16 output
                        for _grp in cutlass.range_constexpr(
                            self.hidden_size // self.num_threads // 8
                        ):
                            base_idx = my_start_c + Int32(_grp * 8)
                            for _oi in cutlass.range_constexpr(8):
                                idx_c = base_idx + Int32(_oi)
                                res_f32 = _ld_global_b16_to_f32(
                                    res_base_c
                                    + Int64(idx_c * Int32(2)))
                                wo_f32 = _ld_global_f32(
                                    wo_base_c
                                    + Int64(idx_c * Int32(4)))
                                new_res = res_f32 + wo_f32

                                gamma_f32 = _ld_global_b16_to_f32(
                                    gamma_base_c
                                    + Int64(idx_c * Int32(2)))
                                hidden_val = new_res * inv_rms_val \
                                    * gamma_f32

                                _st_global_bf16_from_f32(
                                    out_base_c
                                    + Int64(idx_c * Int32(2)),
                                    hidden_val)
                                _st_global_bf16_from_f32(
                                    resout_base_c
                                    + Int64(idx_c * Int32(2)),
                                    new_res)

                        # Reset arrival counter for next call.
                        if tid == Int32(0):
                            _atomic_add_u32(
                                arrival_count_ptr
                                + Int64(seq_idx * Int32(4)),
                                Int32(0) - total_ctas_per_seq)

        # -----------------------------------------------------------------
        # Task 13: Phase 2 grid-barrier stress kernel.
        # -----------------------------------------------------------------
        # Standalone barrier test: grid (total_ctas,1,1) launched with
        # cooperative=True. Every CTA writes its block-id into scratch[bx],
        # passes through the grid barrier, then CTA 0 sums all scratch
        # values. Expected sum = total_ctas*(total_ctas-1)/2. Without the
        # barrier, CTA 0 may read stale scratch entries and the sum diverges
        # (races are non-deterministic — this test asserts PASS with barrier;
        # the race-free failure mode is proven structurally, not by timing).
        @cute.kernel
        def _kernel_barrier_stress(
            self,
            scratch_ptr: Int64,      # [total_ctas] FP32
            result_ptr: Int64,       # [1]          FP32
            barrier_ptr: Int64,      # [1]          I32 (zeroed by caller)
            total_ctas: Int32,
        ):
            """Minimal grid-barrier stress kernel.

            Phase 1 (one write per CTA) → Phase 2 (barrier) → Phase 3
            (CTA 0 sum). Matches the release/acquire pattern the β-coop
            kernel will use between real Phase 1 attn and Phase 3 MLP.
            """
            bx = cute.arch.block_idx()[0]
            tid = cute.arch.thread_idx()[0]

            # --- "Phase 1": each CTA's tid-0 writes bx (as FP32) to scratch[bx] ---
            if tid == Int32(0):
                _st_global_f32(
                    scratch_ptr + Int64(bx * Int32(4)),
                    Float32(bx),
                )
            cute.arch.sync_threads()

            # --- Phase 2: grid barrier ---
            # Release: make the Phase 1 store globally visible.
            _threadfence()
            # Arrival: tid-0 of every CTA bumps the counter once.
            if tid == Int32(0):
                _atomic_add_u32(barrier_ptr, Int32(1))
            # Spin-wait: every thread of every CTA loops on a volatile
            # load until all CTAs have arrived. Runtime while-loop (per
            # memory:feedback_constexpr_oom — range_constexpr on variable
            # N OOMs the JIT compiler).
            arrived = Int32(0)
            while arrived < total_ctas:
                arrived = _ld_volatile_u32(barrier_ptr)
            # Acquire: subsequent loads observe all prior releases.
            _acquire_fence()

            # --- "Phase 3": CTA 0's tid-0 sums the scratch ---
            if bx == Int32(0):
                if tid == Int32(0):
                    acc = Float32(0.0)
                    i = Int32(0)
                    while i < total_ctas:
                        v = _ld_global_f32(
                            scratch_ptr + Int64(i * Int32(4)))
                        acc = acc + v
                        i = i + Int32(1)
                    _st_global_f32(result_ptr, acc)

        @cute.jit
        def _jit_launch_barrier_stress(
            self,
            scratch_ptr: Int64,
            result_ptr: Int64,
            barrier_ptr: Int64,
            total_ctas: Int32,
            stream,
        ):
            """JIT host wrapper: grid-barrier stress launch with
            cooperative=True (the dispatch knob Task 16 will reuse)."""
            self._kernel_barrier_stress(
                scratch_ptr, result_ptr, barrier_ptr, total_ctas,
            ).launch(
                grid=[total_ctas, 1, 1],
                block=[self.num_threads, 1, 1],
                smem=16,  # minimal — test kernel uses no SMEM
                stream=stream,
                cooperative=True,
            )

    # -----------------------------------------------------------------
    # Python-level debug entry: grid-barrier stress.
    # -----------------------------------------------------------------
    def run_barrier_stress_debug(
        self,
        total_ctas: int = 8,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Launch the grid-barrier stress kernel. Returns (scratch, result).

        Correctness criterion: `result[0] == total_ctas*(total_ctas-1)/2`.
        """
        if not _CUTE_AVAILABLE:
            raise RuntimeError(
                "PhaseE_Beta_Kernel requires CUTLASS; not available."
            )
        assert total_ctas >= 2, "grid-barrier test needs at least 2 CTAs"

        device = "cuda"
        scratch = torch.zeros(total_ctas, dtype=torch.float32,
                              device=device)
        result = torch.zeros(1, dtype=torch.float32, device=device)
        barrier = torch.zeros(1, dtype=torch.int32, device=device)

        scratch_ptr = Int64(scratch.data_ptr())
        result_ptr = Int64(result.data_ptr())
        barrier_ptr = Int64(barrier.data_ptr())
        total_ctas_i = Int32(total_ctas)
        stream_arg = _cuda_driver.CUstream(
            int(torch.cuda.current_stream().cuda_stream)
        )

        if getattr(self, "_compiled_barrier_stress", None) is None:
            logger.info(
                "Compiling PhaseE_Beta_Kernel barrier-stress (first call)…"
            )
            self._compiled_barrier_stress = cute.compile(
                self._jit_launch_barrier_stress,
                scratch_ptr, result_ptr, barrier_ptr,
                total_ctas_i, stream_arg,
            )

        self._compiled_barrier_stress(
            scratch_ptr, result_ptr, barrier_ptr,
            total_ctas_i, stream_arg,
        )
        return scratch, result
