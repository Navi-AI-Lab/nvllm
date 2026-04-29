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

import contextlib
import logging
import os
import threading
import time as _time_mod
from typing import Optional

import torch

logger = logging.getLogger(__name__)

# Phase D MLP tile constants — pulled from mlp_kernel.py for Task 14 port.
# Keep in sync if mlp_kernel.py evolves.
FP4_BLOCK_SIZE = 16
LOG2_E = 1.4426950408889634


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
        _atomic_add_f32,
        _atomic_add_u32,
        _bitcast_i32_to_f32,
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
        _ld_shared_u8,
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

    # Phase D helpers defined inside mlp_kernel.py's `if _CUTE_AVAILABLE`
    # block — module-level once the CuTe DSL import succeeds. Imported here
    # so the Task 14 Phase 3 body can reuse them verbatim.
    from vllm.v1.attention.backends.cute_paged.mlp_kernel import (  # noqa: E501
        _decode_ue4m3_u8_to_f32,
        _div_rn_f32,
        _f32_div_to_fp4_nibble,
        _ld_global_u8,
        _rcp_ieee_f32,  # noqa: F401
        _resolve_tile_preset,
    )
    from vllm.v1.attention.backends.cute_paged._fp4_writer import (  # noqa: E501
        _encode_ue4m3_f32_to_u8,
        _f32_to_fp4_nibble,  # noqa: F401
        _st_shared_u8,
    )

    _CUTE_AVAILABLE = True
except ImportError:
    logger.warning(
        "CuTe DSL not available (CUTLASS not installed). "
        "PhaseE_Beta_Kernel cannot be used."
    )
    # Fallback so `_resolve_tile_preset` is still callable for __init__ —
    # but since __init__ requires CUTLASS-side tile resolution anyway,
    # we only need it for the env-override path. If the DSL is absent,
    # the kernel cannot run; these stubs just let module import succeed.
    def _resolve_tile_preset(name):  # type: ignore[no-redef]
        raise RuntimeError(
            "PhaseE_Beta_Kernel requires CUTLASS; _resolve_tile_preset "
            "unavailable."
        )


# --- Task 7: cold-compile heartbeat (operator-visibility) --------------------
# β-coop full compile can take >95min cold on Spark (per
# project_beta_coop_full_compile_wall.md). Without a heartbeat, a stuck
# cute.compile() looks indistinguishable from a deadlock to the operator
# tailing docker logs. This context manager spawns a daemon thread that
# prints ``[β-coop compile] t=Xs alive (#N)`` every ``period_s`` seconds.
@contextlib.contextmanager
def _coop_full_compile_heartbeat(period_s: float = 300.0):
    """Daemon-thread heartbeat that prints ``[β-coop compile] t=Xs alive (#N)``
    every ``period_s`` seconds while inside the context.

    Design risk: if cute.compile holds the GIL the entire time, the daemon
    thread will never run. This is verified empirically on first invocation;
    the fallback if needed is a subprocess-based watchdog (file follow-on).
    """
    counter = {"n": 0}
    stop = threading.Event()
    t0 = _time_mod.monotonic()

    def _beat():
        while not stop.wait(period_s):
            counter["n"] += 1
            elapsed = int(_time_mod.monotonic() - t0)
            logger.info(
                "[β-coop compile] t=%ds alive (#%d)", elapsed, counter["n"]
            )

    thread = threading.Thread(
        target=_beat, daemon=True, name="beta-coop-compile-heartbeat",
    )
    thread.start()
    try:
        yield
    finally:
        stop.set()
        thread.join(timeout=period_s + 1.0)


# --- Phase E.1 follow-up #1: process-wide β-coop compile cache --------------
# Each full_attention layer attaches its own PhaseE_Beta_Kernel instance
# (see _backend.py:754), so without a shared cache cute.compile() fires
# once per layer — 16 × ~23 s = ~6 min cold-start stall on Qwen3.5-27B.
# All instances with matching constexpr config can share one compiled
# handle; this dict keys them by the tuple returned from
# ``PhaseE_Beta_Kernel._coop_full_compile_key()``.
_PHASE_E_COOP_FULL_COMPILE_CACHE: dict = {}


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
        tile_s: Optional[int] = None,
        tile_k: Optional[int] = None,
        slice_ctas: Optional[int] = None,
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
        self.num_threads = 128

        # --- Phase D MLP tile resolution (Task 14) -----------------------
        # Matches Phase_D_MLP_Kernel.__init__ sequence (mlp_kernel.py:313):
        #   unset tile_s/tile_k/slice_ctas kwargs fall back to CUTE_MLP_TILE
        #   env var → preset; explicit kwargs bypass the env read.
        preset_s, preset_k, preset_c = _resolve_tile_preset(
            os.environ.get("CUTE_MLP_TILE")
        )
        tile_s = tile_s if tile_s is not None else preset_s
        tile_k = tile_k if tile_k is not None else preset_k
        slice_ctas = slice_ctas if slice_ctas is not None else preset_c
        self.tile_s = tile_s
        self.tile_k = tile_k
        self.slice_ctas = slice_ctas
        self.num_slices = intermediate_size // tile_s
        self.num_k_tiles = max(hidden_size // tile_k, 1)
        assert intermediate_size % tile_s == 0, (
            f"intermediate_size={intermediate_size} not multiple of "
            f"tile_s={tile_s}"
        )
        assert hidden_size % tile_k == 0, (
            f"hidden_size={hidden_size} not multiple of tile_k={tile_k}"
        )
        assert tile_s % FP4_BLOCK_SIZE == 0, (
            f"tile_s={tile_s} not multiple of FP4_BLOCK_SIZE={FP4_BLOCK_SIZE}"
        )
        assert hidden_size % FP4_BLOCK_SIZE == 0, (
            f"hidden_size={hidden_size} not multiple of "
            f"FP4_BLOCK_SIZE={FP4_BLOCK_SIZE}"
        )
        # Each CTA owns a contiguous chunk of slices (mirrors Phase_D).
        self.slices_per_cta = (self.num_slices + slice_ctas - 1) // slice_ctas
        # FC2 thread mapping — same two-path choice as Phase_D (see
        # mlp_kernel.py:374).
        if tile_k >= self.num_threads:
            assert tile_k % self.num_threads == 0, (
                f"tile_k={tile_k} must be multiple of "
                f"num_threads={self.num_threads} when tile_k >= num_threads"
            )
            self._rows_per_thread = tile_k // self.num_threads
            self._threads_per_row = 1
        else:
            assert self.num_threads % tile_k == 0, (
                f"num_threads={self.num_threads} must be multiple of "
                f"tile_k={tile_k} when tile_k < num_threads"
            )
            self._rows_per_thread = 1
            self._threads_per_row = self.num_threads // tile_k

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

        # --- Phase 3 (MLP D) SMEM layout (Task 14) --------------------------
        # Mirrors Phase_D_MLP_Kernel.__init__ (mlp_kernel.py:389):
        #   [0, hidden*4)        -> smem_x FP32
        #   [+16)                -> cross-warp reduce scratch (4 warps × FP32)
        #   [+tile_s*2)          -> smem_intermediate_bf16 (tile_s BF16)
        #   [+tile_s)            -> smem_intermediate_fp4 (tile_s/2 bytes)
        #   [+tile_s/FP4_BLOCK)  -> smem_intermediate_scale (u8 per block)
        #   [+4)                 -> smem_last_cta flag (u32)
        self._smem_x_bytes = hidden_size * 4
        self._smem_reduce_bytes = 4 * 4
        self._smem_intermediate_bf16_bytes = tile_s * 2
        self._smem_intermediate_fp4_bytes = tile_s // 2
        self._smem_intermediate_scale_bytes = tile_s // FP4_BLOCK_SIZE
        self._smem_flag_bytes = 4
        self._smem_bytes_phase_3 = (
            self._smem_x_bytes
            + self._smem_reduce_bytes
            + self._smem_intermediate_bf16_bytes
            + self._smem_intermediate_fp4_bytes
            + self._smem_intermediate_scale_bytes
            + self._smem_flag_bytes
        )
        self._compiled_phase_3 = None

        # --- β-coop unified kernel SMEM budget (Task 16) --------------------
        # Phase 1 (attn) SMEM layout and Phase 3 (MLP) SMEM layout are
        # time-disjoint (separated by a grid barrier), so they union-alias
        # the same dynamic SMEM region. Total dynamic SMEM = max of the two.
        self._smem_bytes_phase_coop_full = max(
            self._smem_bytes_phase_01,     # Phase 1 attn layout
            self._smem_bytes_phase_3,      # Phase 3 MLP layout
        )
        self._compiled_phase_coop_full = None

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
                # Qwen3_5RMSNorm uses x * (1 + γ) — see vllm/nvllm/layers/layernorm.py:78
                normed = (h_f32 + r_f32) * inv_rms_val * (Float32(1.0) + gamma_f32)
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
                    # Qwen3_5RMSNorm uses x * (1 + γ) — see vllm/nvllm/layers/layernorm.py:78
                    normed = (h_f32 + r_f32) * inv_rms_val_0 * (Float32(1.0) + gamma_f32)
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
                                # Qwen3_5RMSNorm uses x * (1 + γ) — see vllm/nvllm/layers/layernorm.py:78
                                hidden_val = new_res * inv_rms_val \
                                    * (Float32(1.0) + gamma_f32)

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
        # Task 14: Phase 3 MLP (D) standalone debug kernel.
        # Byte-for-byte port of Phase_D_MLP_Kernel._kernel legacy path
        # (mlp_kernel.py:680–1328), MINUS the ε epilogue block which
        # Task 15 will add. Grid: (slice_ctas, num_k_tiles, num_seqs).
        # Block: (128, 1, 1). SMEM: self._smem_bytes_phase_3.
        # -----------------------------------------------------------------
        @cute.kernel
        def _kernel_phase_3_only(
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
        ):
            """Phase 3: fused MLP (legacy Phase D path) embedded in β kernel.

            Byte-for-byte port of Phase_D_MLP_Kernel._kernel, excluding the
            ε epilogue (Task 15 will add that). See mlp_kernel.py:680 for
            the canonical docstring and pipeline overview.
            """
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
            elems_per_thr = hidden // Int32(self.num_threads)
            _i = Int32(0)
            while _i < elems_per_thr:
                flat = tid + _i * Int32(self.num_threads)
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
            num_h_blocks = hidden // FP4_BS

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

                    k_i = Int32(0)
                    while k_i < elems_per_thr:
                        h = tid + k_i * Int32(self.num_threads)
                        x_val = _ld_shared_f32(
                            smem_x + Int64(h) * Int64(4)
                        )

                        h_block = h // FP4_BS
                        byte_col = h >> Int32(1)
                        byte_addr_gate = gate_fp4_ptr + Int64(
                            j * (hidden >> Int32(1)) + byte_col
                        )
                        byte_addr_up = up_fp4_ptr + Int64(
                            j * (hidden >> Int32(1)) + byte_col
                        )
                        nib_lo_gate = _ld_global_u8(byte_addr_gate)
                        nib_lo_up = _ld_global_u8(byte_addr_up)
                        is_odd = h & Int32(1)
                        nib_gate = Int32(
                            ((nib_lo_gate >> (Uint32(is_odd) * Uint32(4)))
                             & Uint32(0xF))
                        )
                        nib_up = Int32(
                            ((nib_lo_up >> (Uint32(is_odd) * Uint32(4)))
                             & Uint32(0xF))
                        )

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
                interm_nblocks = tile_s // Int32(FP4_BLOCK_SIZE)
                blk_iter_max = (interm_nblocks + Int32(3)) >> Int32(2)
                blk_iter = Int32(0)
                while blk_iter < blk_iter_max:
                    my_block = warp + blk_iter * Int32(4)
                    my_block_valid = my_block < interm_nblocks
                    elem_idx = my_block * Int32(FP4_BLOCK_SIZE) + lane
                    my_val = Float32(0.0)
                    if my_block_valid and lane < Int32(FP4_BLOCK_SIZE):
                        addr = smem_interm_bf16 + Int64(elem_idx) * Int64(2)
                        bf16_u32 = _ld_shared_b16(addr)
                        f32_bits = Int32(
                            (bf16_u32 & Uint32(0xFFFF)) << Uint32(16)
                        )
                        my_val = _bitcast_i32_to_f32(f32_bits)

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
                if cutlass.const_expr(self._threads_per_row == 1):
                    # ---------- Path A: per-thread-owns-N-rows ----------
                    rows_per_thread = Int32(self._rows_per_thread)
                    row_base_local = tid * rows_per_thread

                    rpt = self._rows_per_thread
                    acc_list = [Float32(0.0) for _ in range(rpt)]

                    iter_i = Int32(0)
                    while iter_i < tile_s:
                        h = iter_i
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

                    if cutlass.const_expr(self._threads_per_row >= 2):
                        out_acc = out_acc + shfl_xor_sync(out_acc, Int32(1))
                    if cutlass.const_expr(self._threads_per_row >= 4):
                        out_acc = out_acc + shfl_xor_sync(out_acc, Int32(2))
                    if cutlass.const_expr(self._threads_per_row >= 8):
                        out_acc = out_acc + shfl_xor_sync(out_acc, Int32(4))
                    if cutlass.const_expr(self._threads_per_row >= 16):
                        out_acc = out_acc + shfl_xor_sync(out_acc, Int32(8))

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
                is_last_flag = Int32(0)
                if old == (slice_ctas - Int32(1)):
                    is_last_flag = Int32(1)
                _st_shared_b32(
                    smem_last_flag + Int64(0),
                    Uint32(is_last_flag),
                )
            cute.arch.sync_threads()

            last_flag_u32 = _ld_shared_b32(smem_last_flag + Int64(0))
            is_last = Int32(last_flag_u32) == Int32(1)

            if is_last:
                # Path-A / Path-B last-CTA gather (deterministic constexpr
                # bx order — see Phase_D audit 16475223f / Option 1).
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
            # NOTE: the ε epilogue block (emit_ep == 1) is deliberately
            # omitted here — Task 15 will add it. The legacy Phase D
            # counter self-reset was also ε-epilogue-only in mlp_kernel.py,
            # so for Task 14 the arrival counter is caller-zeroed between
            # calls (matches Phase_D_MLP_Kernel legacy behavior).

        @cute.jit
        def _jit_launch_phase_3_only(
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
            nat: Int32,
            stream,
        ):
            """JIT host wrapper: Phase 3 (standalone MLP) launch."""
            self._kernel_phase_3_only(
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
            ).launch(
                grid=[slice_ctas, num_k_tiles, nat],
                block=[self.num_threads, 1, 1],
                smem=self._smem_bytes_phase_3,
                stream=stream,
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

    # -----------------------------------------------------------------
    # Python-level debug entry: Phase 3 (standalone MLP) only (Task 14).
    # -----------------------------------------------------------------
    def run_phase_3_only(
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
        gate_up_global_scale: float = 1.0,
        down_global_scale: float = 1.0,
    ) -> torch.Tensor:
        """Launch only Phase 3 (fused MLP — legacy Phase D path).

        Byte-for-byte compatible with `Phase_D_MLP_Kernel.__call__(...,
        emit_epilogue=False)`. Used by Task 14 tests to prove the β
        kernel's Phase 3 body matches standalone Phase D. Zero-init of
        `mlp_partial_fp32` and `mlp_arrival_count` is the caller's
        responsibility — same contract as Phase_D_MLP_Kernel.
        """
        if not _CUTE_AVAILABLE:
            raise RuntimeError(
                "PhaseE_Beta_Kernel requires CUTLASS; not available."
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
        )
        assert mlp_arrival_count.shape == (nat, self.num_k_tiles)
        assert mlp_output.shape == (nat, self.hidden_size)

        # Flat x + byte ptrs for FP4 / scales (matches Phase_D convention).
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

        gate_up_gs_f32 = Float32(float(gate_up_global_scale))
        down_gs_f32 = Float32(float(down_global_scale))

        stream_arg = _cuda_driver.CUstream(
            int(torch.cuda.current_stream().cuda_stream)
        )

        all_args = (
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
            Int32(nat),
            stream_arg,
        )

        if self._compiled_phase_3 is None:
            logger.info(
                "Compiling PhaseE_Beta_Kernel phase-3-only (first call)…"
            )
            self._compiled_phase_3 = cute.compile(
                self._jit_launch_phase_3_only,
                *all_args,
            )

        self._compiled_phase_3(*all_args)
        return mlp_output

    # -----------------------------------------------------------------
    # Task 15: Phase 4 ε epilogue (isolated debug kernel).
    # -----------------------------------------------------------------
    def run_phase_4_only(
        self,
        residual_post_ln: torch.Tensor,    # [nat, hidden] BF16 (in-place → residual_final)
        mlp_output: torch.Tensor,          # [nat, hidden] BF16 (input)
        next_input_layernorm_gamma: Optional[torch.Tensor],  # [hidden] BF16 or None
        next_hidden_output: torch.Tensor,  # [nat, hidden] BF16 (written)
        emit_next_layernorm: bool = True,
    ) -> torch.Tensor:
        """Launch ε epilogue only. Single CTA per seq (grid=(1,1,nat)).

        Math (ports mlp_kernel.py:1355 minus the secondary barrier since
        (1,1,nat) implies every CTA is already "last" for its seq):

            residual_final = residual_post_ln + mlp_output   # BF16, FP32 acc
            if emit_next_layernorm:
                next_hidden = RMSNorm(residual_final) * γ_next
            else:
                next_hidden = residual_final                  # memcpy
        """
        if not _CUTE_AVAILABLE:
            raise RuntimeError(
                "PhaseE_Beta_Kernel requires CUTLASS; not available."
            )
        nat, hidden = residual_post_ln.shape
        assert hidden == self.hidden_size
        assert mlp_output.shape == residual_post_ln.shape
        assert next_hidden_output.shape == residual_post_ln.shape
        for t in (residual_post_ln, mlp_output, next_hidden_output):
            assert t.is_contiguous()
            assert t.dtype == torch.bfloat16
        if emit_next_layernorm:
            assert next_input_layernorm_gamma is not None, (
                "emit_next_layernorm=True requires next_input_layernorm_gamma"
            )
            assert next_input_layernorm_gamma.shape == (hidden,)
            assert next_input_layernorm_gamma.dtype == torch.bfloat16
            assert next_input_layernorm_gamma.is_contiguous()
            gamma_ptr_val = next_input_layernorm_gamma.data_ptr()
        else:
            # Last-layer: γ not needed; pass Int64(0) — runtime branch skips loads.
            gamma_ptr_val = 0

        residual_ptr = Int64(residual_post_ln.data_ptr())
        mlp_output_ptr = Int64(mlp_output.data_ptr())
        next_gamma_ptr = Int64(gamma_ptr_val)
        next_hidden_ptr = Int64(next_hidden_output.data_ptr())
        rms_eps_f32 = Float32(float(self.rms_eps))
        emit_next_ln_i32 = Int32(1 if emit_next_layernorm else 0)

        stream_arg = _cuda_driver.CUstream(
            int(torch.cuda.current_stream().cuda_stream)
        )

        if getattr(self, "_compiled_phase_4", None) is None:
            logger.info(
                "Compiling PhaseE_Beta_Kernel phase-4-only (first call)…"
            )
            self._compiled_phase_4 = cute.compile(
                self._jit_launch_phase_4_only,
                residual_ptr,
                mlp_output_ptr,
                next_gamma_ptr,
                next_hidden_ptr,
                Int32(nat),
                Int32(hidden),
                rms_eps_f32,
                emit_next_ln_i32,
                stream_arg,
            )

        self._compiled_phase_4(
            residual_ptr,
            mlp_output_ptr,
            next_gamma_ptr,
            next_hidden_ptr,
            Int32(nat),
            Int32(hidden),
            rms_eps_f32,
            emit_next_ln_i32,
            stream_arg,
        )
        return next_hidden_output

    if _CUTE_AVAILABLE:

        @cute.jit
        def _jit_launch_phase_4_only(
            self,
            residual_post_ln_ptr: Int64,
            mlp_output_ptr: Int64,
            next_gamma_ptr: Int64,
            next_hidden_out_ptr: Int64,
            nat: Int32,
            hidden_dim: Int32,
            rms_eps: Float32,
            emit_next_ln: Int32,
            stream,
        ):
            """JIT host wrapper for ε epilogue launch."""
            self._kernel_phase_4_only(
                residual_post_ln_ptr,
                mlp_output_ptr,
                next_gamma_ptr,
                next_hidden_out_ptr,
                hidden_dim,
                rms_eps,
                emit_next_ln,
            ).launch(
                grid=[1, 1, nat],
                block=[self.num_threads, 1, 1],
                smem=16,  # 4-warp cross-warp reduce scratch
                stream=stream,
            )

        @cute.kernel
        def _kernel_phase_4_only(
            self,
            residual_post_ln_ptr: Int64,
            mlp_output_ptr: Int64,
            next_gamma_ptr: Int64,
            next_hidden_out_ptr: Int64,
            hidden_dim: Int32,
            rms_eps: Float32,
            emit_next_ln: Int32,
        ):
            """Phase 4 ε epilogue: residual_final + optional next-layer RMSNorm.

            Port of mlp_kernel.py:1355's epilogue (Task 8). Grid=(1,1,nat)
            means every CTA is the single participant per seq, so the
            secondary barrier / last-CTA election is dropped.
            """
            bz = cute.arch.block_idx()[2]
            tid = cute.arch.thread_idx()[0]
            lane = cute.arch.lane_idx()
            warp = cute.arch.warp_idx()

            # SMEM: 4 warp × 4 B = 16 B cross-warp reduction scratch.
            smem = cute.arch.get_dyn_smem(cutlass.Uint8, alignment=128)
            smem_reduce = shared_ptr_to_i64(smem)

            n_per_thr_py = self.hidden_size // self.num_threads  # 40 @ hidden=5120

            res_base = (
                residual_post_ln_ptr
                + Int64(bz * Int32(self.hidden_size)) * Int64(2)
            )
            mlp_base = (
                mlp_output_ptr
                + Int64(bz * Int32(self.hidden_size)) * Int64(2)
            )
            next_hidden_base = (
                next_hidden_out_ptr
                + Int64(bz * Int32(self.hidden_size)) * Int64(2)
            )

            # --- Pass 1: residual_final = residual_post + mlp_out (in-place);
            #             accumulate sum-of-squares ---------------------
            ss = Float32(0.0)
            for _i in cutlass.range_constexpr(n_per_thr_py):
                idx = tid + Int32(_i * self.num_threads)
                res_f32 = _ld_global_b16_to_f32(
                    res_base + Int64(idx) * Int64(2))
                mlp_f32 = _ld_global_b16_to_f32(
                    mlp_base + Int64(idx) * Int64(2))
                rf = res_f32 + mlp_f32
                _st_global_bf16_from_f32(
                    res_base + Int64(idx) * Int64(2), rf)
                ss = ss + rf * rf

            # --- Pass 2: warp-shuffle reduction + cross-warp SMEM →
            #             variance → inv_rms, broadcast via smem slot 0 -
            ss = ss + shfl_xor_sync(ss, Int32(1))
            ss = ss + shfl_xor_sync(ss, Int32(2))
            ss = ss + shfl_xor_sync(ss, Int32(4))
            ss = ss + shfl_xor_sync(ss, Int32(8))
            ss = ss + shfl_xor_sync(ss, Int32(16))

            if lane == Int32(0):
                _st_shared_f32(
                    smem_reduce + Int64(warp) * Int64(4), ss)
            cute.arch.sync_threads()

            if warp == Int32(0):
                if lane == Int32(0):
                    total_ss = _ld_shared_f32(smem_reduce + Int64(0))
                    total_ss = total_ss + _ld_shared_f32(
                        smem_reduce + Int64(4))
                    total_ss = total_ss + _ld_shared_f32(
                        smem_reduce + Int64(8))
                    total_ss = total_ss + _ld_shared_f32(
                        smem_reduce + Int64(12))
                    variance = total_ss / Float32(
                        float(self.hidden_size))
                    inv_rms = _rsqrt_approx_f32(variance + rms_eps)
                    _st_shared_f32(smem_reduce + Int64(0), inv_rms)
            cute.arch.sync_threads()

            inv_rms_val = _ld_shared_f32(smem_reduce + Int64(0))

            # --- Pass 3: write next_hidden — two paths gated by emit_next_ln ---
            if emit_next_ln == Int32(1):
                # RMSNorm(residual_final) * γ with a BF16 round-trip to
                # match Python ref's `(rf * rstd).to(bf16) * gamma.float()`.
                for _i in cutlass.range_constexpr(n_per_thr_py):
                    idx = tid + Int32(_i * self.num_threads)
                    rf = _ld_global_b16_to_f32(
                        res_base + Int64(idx) * Int64(2))
                    gamma_f32 = _ld_global_b16_to_f32(
                        next_gamma_ptr + Int64(idx) * Int64(2))
                    normed_f32 = rf * inv_rms_val
                    # Round-trip through BF16 via cvt.rn.bf16.f32 + widen.
                    normed_bf16_u32 = _cvt_2f32_to_bf16x2(
                        normed_f32, Float32(0.0))
                    low16 = normed_bf16_u32 & Uint32(0xFFFF)
                    as_bits = Int32(low16 << Uint32(16))
                    normed_round = _bitcast_i32_to_f32(as_bits)
                    # Qwen3_5RMSNorm uses x * (1 + γ) — see vllm/nvllm/layers/layernorm.py:78
                    out_f32 = normed_round * (Float32(1.0) + gamma_f32)
                    _st_global_bf16_from_f32(
                        next_hidden_base + Int64(idx) * Int64(2), out_f32)
            else:
                # Last-layer: next_hidden = residual_final (memcpy).
                for _i in cutlass.range_constexpr(n_per_thr_py):
                    idx = tid + Int32(_i * self.num_threads)
                    rf = _ld_global_b16_to_f32(
                        res_base + Int64(idx) * Int64(2))
                    _st_global_bf16_from_f32(
                        next_hidden_base + Int64(idx) * Int64(2), rf)

    # =====================================================================
    # Task 16: β-coop UNIFIED kernel — Phases 0 + 1 + 3 + 4 in one
    # cooperative launch. Grid: (slice_ctas, num_k_tiles, num_seqs),
    # typically (8, 8, nat) = 64 CTAs per seq.
    # =====================================================================
    #
    # Design (see subagent brief 2026-04-23):
    #   * Phase 0 (input_layernorm): gated to bx==0 && by==0 (single CTA
    #     per seq). Writes attn_input_bf16 as a side-channel output for
    #     external consumers (next-layer QKV would fuse this in a fully-
    #     integrated world). No data dependency with Phase 1 in the
    #     debug harness (query is pre-projected).
    #   * Phase 1 (attn A+B+C): gated to bx==0 && by<4 (4 kv-head CTAs).
    #     Byte-for-byte copy of _kernel_phase_01 Phase 1 body. Writes
    #     attn_output (Phase C BF16) + residual_output (BF16).
    #   * GRID BARRIER: all 64 CTAs arrive; ensures Phase 3 sees
    #     attn_output before loading it as its MLP input `x`.
    #   * Phase 3 (MLP D): all 64 CTAs participate. Byte-for-byte copy of
    #     _kernel_phase_3_only body, reading `x = attn_output[bz, :]`,
    #     writing mlp_output + per-CTA partials + per-k-tile arrival.
    #   * SECONDARY BARRIER (Phase D pattern): each k-tile's local-last-
    #     CTA increments a secondary counter; the globally-last CTA
    #     (counter old == num_k_tiles-1) proceeds to Phase 4.
    #   * Phase 4 (ε epilogue): globally-last CTA per seq does
    #     residual_final (in-place) + optional next-layer RMSNorm.
    #
    # SMEM: union-alias Phase 1 layout (45568 B) with Phase 3 layout
    # (21156 B). Total dynamic SMEM = max = 45568 B. The two layouts are
    # time-disjoint (separated by the grid barrier).
    #
    # Barrier workspace (one contiguous tensor):
    #   [nat]        i32  phase1_arrival_count  (4 attn CTAs arrive)
    #   [nat]        i32  grid_barrier          (64 CTAs arrive)
    #   [nat, nkt]   u32  phase3_arrival_count  (slice_ctas per k-tile)
    #   [nat]        i32  secondary_barrier     (num_k_tiles last-CTAs)
    #
    # Self-reset (mirrors mlp_kernel.py:1519): globally-last CTA at end of
    # Phase 4 zeroes ALL counters for this seq — grid_barrier (via
    # atomic_add(-total_ctas_per_seq)), secondary_barrier (via
    # atomic_add(-num_k_tiles)). phase1_arrival_count is already self-
    # reset by Phase 1's Phase C block (see _kernel_phase_01 line 1536).
    # phase3_arrival_count is reset inline (per-k-tile last-CTA after it
    # has written its slice of mlp_output).
    # =====================================================================
    def run_beta_coop_full(
        self,
        # Phase 0 / Phase 1 inputs (mirror run_phase_01_only):
        hidden_in: torch.Tensor,        # [nat, hidden]  BF16 (pre-input_LN hidden)
        residual_in: torch.Tensor,      # [nat, hidden]  BF16 (residual stream)
        input_gamma: torch.Tensor,      # [hidden]       BF16 (γ for Phase 0)
        post_attn_gamma: torch.Tensor,  # [hidden]       BF16 (γ for Phase C)
        attn_input_bf16: torch.Tensor,  # [nat, hidden]  BF16 (Phase 0 output — side channel)
        query: torch.Tensor,            # [nat, num_q_heads, head_dim] BF16 (pre-projected)
        kv_cache: torch.Tensor,         # [pg, 2, ps, kv, hd] uint8 FP8
        page_table: torch.Tensor,       # [nat, max_pages] int32
        seq_lens: torch.Tensor,         # [nat] int32
        wo_weight: torch.Tensor,        # [hidden, K/2] uint8 NVFP4
        wo_scales: torch.Tensor,        # [hidden, K_sf] fp8_e4m3fn (swizzled)
        wo_global_scale: torch.Tensor,  # scalar f32
        attn_output: torch.Tensor,      # [nat, hidden] BF16 (Phase C output = Phase 3 input)
        # Phase 3 inputs (mirror run_phase_3_only):
        gate_w_fp4: torch.Tensor,
        gate_w_scale: torch.Tensor,
        up_w_fp4: torch.Tensor,
        up_w_scale: torch.Tensor,
        down_w_fp4: torch.Tensor,
        down_w_scale: torch.Tensor,
        mlp_output: torch.Tensor,       # [nat, hidden] BF16
        # C1.5: Phase 4 inputs (next-LN gamma, next-hidden output, emit
        # flag) deleted — kernel returns at end of Phase 3.
        # Scalars / flags:
        scale: float = None,
        k_scale: float = 1.0,
        v_scale: float = 1.0,
        gate_up_global_scale: float = 1.0,
        down_global_scale: float = 1.0,
        # Task 16: allow caller to supply a pre-allocated residual_output.
        # When None (default), allocated internally — backward-compatible
        # with the standalone test. When provided, the kernel writes the
        # Phase 1 Phase-C residual (= old_residual + attn_out) into this
        # buffer.
        # C1.5 NOTE: pre-C1.5 Phase 4 mutated residual_output in-place to
        # residual_final (= residual_post + mlp_out). Phase 4 is now gone,
        # so residual_output here holds residual_post_attn (the Phase-C
        # residual = old_residual + attn_out). The next layer's input_LN
        # in Python takes care of adding mlp_out.
        residual_output: Optional[torch.Tensor] = None,
        # C2: Qwen3.5 attn output gate. When supplied (BF16
        # [nat, num_q_heads * head_dim]), the kernel multiplies the
        # per-head attn output by sigmoid(gate) before the W_O GEMV —
        # mirrors paged kernel.py:1555-1569.
        gate_buf: Optional[torch.Tensor] = None,
        # When True, prime the disk cache for this config and return
        # immediately after _compile_coop_full. The actual launch
        # (self._compiled_phase_coop_full) is skipped. Used by
        # scripts/precompile-cute-coop-full.py to populate the cache
        # without booting vLLM.
        compile_only: bool = False,
    ) -> None:
        """Launch the unified β-coop kernel for a single fusion-active decoder layer.

        Phases 0 → 1 → grid barrier → 3 execute in one cooperative kernel
        launch. Returns None (mutates attn_input_bf16, attn_output,
        mlp_output, residual_output).

        C1.5 NOTE: Phase 4 (ε epilogue / next-layer input_LN bake) was
        removed. The kernel now ends at Phase 3 (MLP write). The next
        layer's input_LN runs from Python at layer entry instead of
        being baked into the previous layer's epilogue.

        NOTE: like run_phase_01_only, `residual_in` is NOT mutated — the
        Phase-C residual `residual_post = residual_in + attn_out` is
        written to `residual_output`.
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
        assert mlp_output.shape == hidden_in.shape
        assert query.shape == (nat, self.num_attn_heads, self.head_dim)
        for t in (
            hidden_in, residual_in, input_gamma, post_attn_gamma,
            attn_input_bf16, attn_output, query,
            kv_cache, page_table, seq_lens,
            wo_weight, wo_scales, wo_global_scale,
            gate_w_fp4, gate_w_scale, up_w_fp4, up_w_scale,
            down_w_fp4, down_w_scale, mlp_output,
        ):
            assert t.is_contiguous(), f"{t.shape} must be contiguous"
        assert hidden_in.dtype == torch.bfloat16
        assert query.dtype == torch.bfloat16
        assert gate_w_fp4.dtype == torch.uint8
        assert gate_w_scale.dtype == torch.uint8

        if scale is None:
            scale = 1.0 / (self.head_dim ** 0.5)

        # --- Workspace buffers -------------------------------------------
        # wo_output: 4 attn CTAs per seq — matches Phase 1 of phase_01.
        total_ctas_per_seq_attn = 4  # bx==0 && by<4
        wo_output = torch.zeros(
            nat, total_ctas_per_seq_attn, hidden,
            dtype=torch.float32, device=hidden_in.device,
        )
        # residual_output: written by Phase 1 Phase C (residual_post =
        # residual_in + attn_out). Task 16: allow caller-supplied.
        # C1.5: Phase 4 deletion ended in-place mutation; residual_output
        # is now write-once during Phase C.
        if residual_output is None:
            residual_output = torch.empty(
                nat, hidden, dtype=torch.bfloat16, device=hidden_in.device,
            )
        else:
            assert residual_output.shape == hidden_in.shape
            assert residual_output.dtype == torch.bfloat16
            assert residual_output.is_contiguous()
        # MLP partials + arrival counter (Phase D):
        mlp_partial_fp32 = torch.zeros(
            nat, self.slice_ctas, hidden,
            dtype=torch.float32, device=hidden_in.device,
        )
        mlp_arrival_count = torch.zeros(
            nat, self.num_k_tiles, dtype=torch.uint32,
            device=hidden_in.device,
        )
        # Grid-barrier counter: 64 CTAs per seq arrive.
        grid_barrier_i32 = torch.zeros(
            nat, dtype=torch.int32, device=hidden_in.device,
        )
        # Phase 1 Phase C last-CTA election (4 attn CTAs per seq arrive).
        phase1_arrival_count = torch.zeros(
            nat, dtype=torch.int32, device=hidden_in.device,
        )
        # C1.5: secondary_barrier removed with Phase 4.

        # --- Flatten (DSL cannot flat-index >1D multidim tensors) --------
        q_flat = query.view(-1)

        # --- Derived pointers / scalars ----------------------------------
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
        phase1_arrival_ptr = Int64(phase1_arrival_count.data_ptr())

        gate_fp4_ptr = Int64(gate_w_fp4.data_ptr())
        gate_sc_ptr = Int64(gate_w_scale.data_ptr())
        up_fp4_ptr = Int64(up_w_fp4.data_ptr())
        up_sc_ptr = Int64(up_w_scale.data_ptr())
        down_fp4_ptr = Int64(down_w_fp4.data_ptr())
        down_sc_ptr = Int64(down_w_scale.data_ptr())
        mlp_partial_ptr = Int64(mlp_partial_fp32.data_ptr())
        mlp_arrival_ptr = Int64(mlp_arrival_count.data_ptr())
        mlp_output_ptr = Int64(mlp_output.data_ptr())

        grid_barrier_ptr = Int64(grid_barrier_i32.data_ptr())
        # C1.5: secondary_barrier_ptr / next_gamma_ptr / next_hidden_ptr /
        # emit_next_ln_i32 dropped with Phase 4.

        wo_K = self.num_attn_heads * self.head_dim
        wo_nkt = Int32((wo_K // 16 + 3) // 4)
        wo_row_stride = Int32(wo_weight.shape[1])

        # C2: gate_buf optional — back-compat for callers that don't pass it
        # (gate_fused == 0 disables the multiply inside the kernel).
        if gate_buf is not None:
            assert gate_buf.dtype == torch.bfloat16
            assert gate_buf.is_contiguous()
            assert gate_buf.shape[-1] == self.num_attn_heads * self.head_dim
            gate_ptr = Int64(gate_buf.data_ptr())
            gate_fused_flag = Int32(1)
        else:
            gate_ptr = Int64(0)
            gate_fused_flag = Int32(0)

        # C2 DIAG removed: the host-side gate_buf check fires at trace time
        # (compile path) and reads the uninitialised buffer, not the
        # runtime-populated one. Misleading — keep diagnostics inside the
        # CUTE kernel proper if needed (write-then-read via a debug global).

        rms_eps_f32 = Float32(float(self.rms_eps))
        gate_up_gs_f32 = Float32(float(gate_up_global_scale))
        down_gs_f32 = Float32(float(down_global_scale))

        stream_arg = _cuda_driver.CUstream(
            int(torch.cuda.current_stream().cuda_stream)
        )

        # Total CTAs per seq for the grid barrier = slice_ctas × num_k_tiles.
        total_ctas_per_seq_grid = self.slice_ctas * self.num_k_tiles

        all_args = (
            # Phase 0/1 args
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
            phase1_arrival_ptr,
            Int32(self.num_attn_heads),
            Int32(self.num_kv_heads),
            kv_page_stride,
            wo_nkt,
            wo_row_stride,
            Int32(total_ctas_per_seq_attn),
            Int32(hidden),
            Float32(float(scale)),
            Float32(float(k_scale)),
            Float32(float(v_scale)),
            rms_eps_f32,
            # Grid barrier
            grid_barrier_ptr,
            Int32(total_ctas_per_seq_grid),
            # Phase 3 args
            gate_fp4_ptr, gate_sc_ptr,
            up_fp4_ptr, up_sc_ptr,
            down_fp4_ptr, down_sc_ptr,
            mlp_partial_ptr, mlp_arrival_ptr, mlp_output_ptr,
            Int32(self.intermediate_size),
            Int32(self.num_slices),
            Int32(self.slices_per_cta),
            Int32(self.tile_s),
            Int32(self.tile_k),
            Int32(self.num_k_tiles),
            Int32(self.slice_ctas),
            gate_up_gs_f32, down_gs_f32,
            # C2: Qwen3.5 attn output gate.
            gate_ptr,
            gate_fused_flag,
            # C1.5: Phase 4 args (secondary_barrier_ptr, next_gamma_ptr,
            # next_hidden_ptr, emit_next_ln_i32) dropped.
            # Grid z-dim
            Int32(nat),
            stream_arg,
        )

        self._compile_coop_full(*all_args)
        if compile_only:
            return None
        self._compiled_phase_coop_full(*all_args)
        return None

    # -----------------------------------------------------------------
    # Phase E.1 follow-up #1 — shared β-coop compile cache.
    # -----------------------------------------------------------------
    def _coop_full_compile_key(self) -> tuple:
        """Tuple of every ``self.`` constexpr value read by
        ``_jit_launch_phase_0_to_4``. Two instances with identical keys
        can share one cute-compiled kernel (see
        _PHASE_E_COOP_FULL_COMPILE_CACHE).
        """
        return (
            "phase_coop_full",
            self.hidden_size,
            self.intermediate_size,
            self.num_attn_heads,
            self.num_kv_heads,
            self.head_dim,
            self.num_threads,
            self.tile_s,
            self.tile_k,
            self.slice_ctas,
            self.num_slices,
            self.num_k_tiles,
            self.slices_per_cta,
            self._rows_per_thread,
            self._threads_per_row,
            self._cta_q,
            self._cta_kv,
            self._block_size,
            self._num_warps_kv,
            self._num_mma_d,
            self._q_bytes,
            self._k_bytes,
            self._v_bytes,
            self._sync_o_small_offset,
            self._sync_md_small_offset,
            self._smem_bytes_phase_01,
            self._smem_x_bytes,
            self._smem_reduce_bytes,
            self._smem_intermediate_bf16_bytes,
            self._smem_intermediate_fp4_bytes,
            self._smem_intermediate_scale_bytes,
            self._smem_flag_bytes,
            self._smem_bytes_phase_3,
            self._smem_bytes_phase_coop_full,
        )

    def _compile_coop_full(self, *compile_args):
        """Cached ``cute.compile`` for the β-coop unified kernel.

        First call for a given config key compiles; subsequent calls
        (including from a different ``PhaseE_Beta_Kernel`` instance with
        matching constexpr config) reuse the cached handle. Populates
        ``self._compiled_phase_coop_full`` for back-compat readers.
        """
        key = self._coop_full_compile_key()
        cached = _PHASE_E_COOP_FULL_COMPILE_CACHE.get(key)
        if cached is None:
            logger.info(
                "Compiling PhaseE_Beta_Kernel β-coop full (first call for "
                "this config)…"
            )
            with _coop_full_compile_heartbeat():
                cached = cute.compile(
                    self._jit_launch_phase_0_to_4,
                    *compile_args,
                )
            _PHASE_E_COOP_FULL_COMPILE_CACHE[key] = cached
        self._compiled_phase_coop_full = cached
        return cached

    if _CUTE_AVAILABLE:

        @cute.jit
        def _jit_launch_phase_0_to_4(
            self,
            # Phase 0/1
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
            phase1_arrival_ptr: Int64,
            num_q_heads: Int32,
            num_kv_heads: Int32,
            kv_page_stride: Int32,
            wo_num_k_tiles: Int32,
            wo_weight_row_stride: Int32,
            total_ctas_per_seq_attn: Int32,
            hidden_dim: Int32,
            scale: Float32,
            k_scale: Float32,
            v_scale: Float32,
            rms_eps: Float32,
            # Grid barrier
            grid_barrier_ptr: Int64,
            total_ctas_per_seq_grid: Int32,
            # Phase 3
            gate_fp4_ptr: Int64, gate_sc_ptr: Int64,
            up_fp4_ptr: Int64, up_sc_ptr: Int64,
            down_fp4_ptr: Int64, down_sc_ptr: Int64,
            mlp_partial_ptr: Int64,
            mlp_arrival_ptr: Int64,
            mlp_output_ptr: Int64,
            interm: Int32,
            num_slices: Int32,
            slices_per_cta: Int32,
            tile_s: Int32,
            tile_k: Int32,
            num_k_tiles: Int32,
            slice_ctas: Int32,
            gate_up_gs: Float32,
            down_gs: Float32,
            # C2: Qwen3.5 attn output gate (BF16 [nat, num_q_heads*hd]).
            gate_ptr: Int64,
            gate_fused: Int32,
            # C1.5: Phase 4 args (secondary_barrier_ptr, next_gamma_ptr,
            # next_hidden_ptr, emit_next_ln) dropped — kernel ends at
            # Phase 3 (MLP write).
            # Grid z
            nat: Int32,
            stream,
        ):
            """JIT host wrapper for the unified β-coop launch.

            Cooperative launch (cooperative=True) is REQUIRED because the
            kernel uses a grid-wide barrier that spans all CTAs of one
            seq (actually of one grid-z slab). On GB10 the barrier is
            implemented via _threadfence + atomic counter + spin-wait
            (see Task 13's _kernel_barrier_stress).
            """
            self._kernel_phase_0_to_4(
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
                phase1_arrival_ptr,
                num_q_heads,
                num_kv_heads,
                kv_page_stride,
                wo_num_k_tiles,
                wo_weight_row_stride,
                total_ctas_per_seq_attn,
                hidden_dim,
                scale,
                k_scale,
                v_scale,
                rms_eps,
                grid_barrier_ptr,
                total_ctas_per_seq_grid,
                gate_fp4_ptr, gate_sc_ptr,
                up_fp4_ptr, up_sc_ptr,
                down_fp4_ptr, down_sc_ptr,
                mlp_partial_ptr, mlp_arrival_ptr, mlp_output_ptr,
                interm,
                num_slices,
                slices_per_cta,
                tile_s,
                tile_k,
                num_k_tiles,
                slice_ctas,
                gate_up_gs,
                down_gs,
                gate_ptr,
                gate_fused,
            ).launch(
                grid=[self.slice_ctas, self.num_k_tiles, nat],
                block=[self.num_threads, 1, 1],
                smem=self._smem_bytes_phase_coop_full,
                stream=stream,
                cooperative=True,
            )

        @cute.kernel
        def _kernel_phase_0_to_4(
            self,
            # Phase 0/1
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
            phase1_arrival_ptr: Int64,
            num_q_heads: Int32,
            num_kv_heads: Int32,
            kv_page_stride: Int32,
            wo_num_k_tiles: Int32,
            wo_weight_row_stride: Int32,
            total_ctas_per_seq_attn: Int32,
            hidden_dim: Int32,
            scale: Float32,
            k_scale: Float32,
            v_scale: Float32,
            rms_eps: Float32,
            # Grid barrier
            grid_barrier_ptr: Int64,
            total_ctas_per_seq_grid: Int32,
            # Phase 3
            gate_fp4_ptr: Int64, gate_sc_ptr: Int64,
            up_fp4_ptr: Int64, up_sc_ptr: Int64,
            down_fp4_ptr: Int64, down_sc_ptr: Int64,
            mlp_partial_ptr: Int64,
            mlp_arrival_ptr: Int64,
            mlp_output_ptr: Int64,
            interm: Int32,
            num_slices: Int32,
            slices_per_cta: Int32,
            tile_s: Int32,
            tile_k: Int32,
            num_k_tiles: Int32,
            slice_ctas: Int32,
            gate_up_gs: Float32,
            down_gs: Float32,
            # C2: Qwen3.5 attn output gate (BF16 [nat, num_q_heads*hd]).
            # gate_fused == 0 disables the multiply (back-compat for
            # callers that don't supply gate_buf).
            gate_ptr: Int64,
            gate_fused: Int32,
            # C1.5: Phase 4 params (secondary_barrier_ptr, next_gamma_ptr,
            # next_hidden_ptr, emit_next_ln) dropped.
        ):
            """β-coop unified kernel — Phase 0 → 1 → grid barrier → 3."""
            bx, by, bz = cute.arch.block_idx()
            lane = cute.arch.lane_idx()
            warp = cute.arch.warp_idx()
            tid = warp * Int32(32) + lane
            group = lane >> Int32(2)
            sub = lane & Int32(3)

            seq_idx = bz

            # -----------------------------------------------------------------
            # SMEM: Phase 1 layout is the superset (45568 B). Phase 3 aliases
            # the same region after the grid barrier.
            # Phase 1 layout: Q | K | V | sync_o | sync_md.
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
            # Phase 0: input_layernorm. Single CTA per seq (bx==0, by==0).
            # Writes attn_input_bf16 as a side-channel output (not consumed
            # by Phase 1 in this debug harness — Phase 1 uses pre-projected
            # `query`). Byte-for-byte copy of _kernel_phase_01 Phase 0 body.
            # =================================================================
            if bx == Int32(0):
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
                        _st_shared_f32(
                            sync_md + Int64(warp * Int32(4)), ss0)
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
                        # Qwen3_5RMSNorm uses x * (1 + γ) — see vllm/nvllm/layers/layernorm.py:78
                        normed = (h_f32 + r_f32) * inv_rms_val_0 * (Float32(1.0) + gamma_f32)
                        _st_global_bf16_from_f32(
                            out_base_0 + Int64(idx_c0 * Int32(2)), normed)

                    # Sync before Phase 1 reuses sync_md/sync_o for attn state.
                    cute.arch.sync_threads()

            # =================================================================
            # Phase 1: Attn A+B+C — gated to bx==0 && by<4 (4 attn CTAs).
            # kv_head_idx == by (each of the 4 attn CTAs takes one kv head).
            # Byte-for-byte port of _kernel_phase_01 Phase 1 body.
            # =================================================================
            if bx == Int32(0):
                if by < Int32(4):
                    # Recompute Phase 1-local constants (shadowed vars from
                    # Phase 0 not needed — most vars were in scope-limited
                    # by Python-side `if by==0`, but we redeclare for clarity).
                    kv_head_idx = by
                    group_size_p1 = num_q_heads // num_kv_heads
                    q_head_start = (kv_head_idx * group_size_p1
                                     + bx * Int32(self._cta_q))

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

                    # --- Load Q into SMEM ---
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

                    # --- Serialized _md loop ---
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
                                if red_row < group_size_p1:
                                    out_idx = (seq_idx * num_q_heads * hd
                                               + out_head * hd + g_col)
                                    # C2: Qwen3.5 attn output gate fusion —
                                    # mirrors paged kernel.py:1555-1569.
                                    # gate_buf layout matches `output`:
                                    # [num_seqs, num_q_heads * head_dim] BF16
                                    if gate_fused != Int32(0):
                                        gate_elem_idx = (
                                            seq_idx * num_q_heads * hd
                                            + out_head * hd + g_col)
                                        gate_f32 = _ld_global_b16_to_f32(
                                            gate_ptr + Int64(
                                                gate_elem_idx * Int32(2)))
                                        # sigmoid(x) = 1/(1+exp2(-x*LOG2E))
                                        neg_x_log2e = (
                                            Float32(0.0) - gate_f32
                                            * Float32(1.4426950408889634))
                                        exp_val = exp2_approx_ftz_f32(
                                            neg_x_log2e)
                                        sigmoid_val = _rcp_approx_f32(
                                            Float32(1.0) + exp_val)
                                        o_final = o_final * sigmoid_val
                                    _st_global_bf16_from_f32(
                                        attn_output_ptr
                                        + Int64(out_idx * Int32(2)),
                                        o_final)

                        cute.arch.sync_threads()
                    # end _md loop

                    # === Phase B: Fused W_O GEMV ===
                    _threadfence()
                    cute.arch.sync_threads()

                    attn_base = seq_idx * num_q_heads * hd \
                        + q_head_start * hd
                    hd_wo = Int32(self.hidden_size)
                    n_per_thr_wo = Int32(
                        self.hidden_size // self.num_threads)
                    my_row_base = tid * n_per_thr_wo

                    wo_gs = _ld_global_f32(wo_gs_ptr)

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

                        k_dim = group_size_p1 * hd
                        k_idx = Int32(0)
                        while k_idx < k_dim:
                            attn_val = _ld_global_b16_to_f32(
                                attn_output_ptr
                                + Int64((attn_base + k_idx) * Int32(2)))
                            abs_k = (kv_head_idx * group_size_p1 * hd
                                     + k_idx)
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

                        cta_idx = bx * num_kv_heads + by
                        wo_slot_base = wo_output_ptr + Int64(
                            (seq_idx * total_ctas_per_seq_attn + cta_idx)
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
                            phase1_arrival_ptr
                            + Int64(seq_idx * Int32(4)),
                            Int32(1))
                        if old_count == total_ctas_per_seq_attn - Int32(1):
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
                            seq_idx * total_ctas_per_seq_attn
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
                                while cta_i < total_ctas_per_seq_attn:
                                    slot_addr = wo_output_ptr + Int64(
                                        (seq_idx * total_ctas_per_seq_attn
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
                                # Qwen3_5RMSNorm uses x * (1 + γ) — see vllm/nvllm/layers/layernorm.py:78
                                hidden_val = new_res * inv_rms_val \
                                    * (Float32(1.0) + gamma_f32)

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
                                phase1_arrival_ptr
                                + Int64(seq_idx * Int32(4)),
                                Int32(0) - total_ctas_per_seq_attn)

            # =================================================================
            # GRID BARRIER: between Phase 1 (attn) and Phase 3 (MLP).
            # All 64 CTAs per seq must arrive before any CTA reads
            # attn_output / residual_output (Phase 3 + Phase 4 inputs).
            # Release/acquire pattern (mirrors _kernel_barrier_stress).
            # =================================================================
            _threadfence()
            cute.arch.sync_threads()

            if tid == Int32(0):
                _atomic_add_u32(
                    grid_barrier_ptr + Int64(seq_idx * Int32(4)),
                    Int32(1),
                )
            # Spin-wait: every thread of every CTA loops on a volatile
            # load until all CTAs for this seq have arrived.
            arrived = Int32(0)
            while arrived < total_ctas_per_seq_grid:
                arrived = _ld_volatile_u32(
                    grid_barrier_ptr + Int64(seq_idx * Int32(4))
                )
            _acquire_fence()
            # Block-internal sync: ensure all threads of this CTA have
            # observed the barrier release before re-aliasing SMEM.
            cute.arch.sync_threads()

            # =================================================================
            # Phase 3: MLP (D). All 64 CTAs participate. Reuses SMEM (Phase
            # 1 layout is disjoint in time). Byte-for-byte port of
            # _kernel_phase_3_only body. Input: `attn_output` (Phase 1's
            # Phase C write). Output: partials + per-k-tile last-CTA
            # election writes mlp_output.
            # =================================================================
            # Re-alias SMEM pointers for Phase 3 layout.
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

            # === Phase 3.2: Load attn_output[bz, :] into smem_x as FP32 ===
            # In Phase 3 standalone, input is `x_flat` (a CuTe tensor view
            # of x.view(-1)). Here we read `attn_output` directly via
            # pointer arithmetic — skips needing to plumb a second tensor
            # view. Phase 3's standalone uses `x_flat[gmem_idx]` which
            # emits a BF16 load; we emit `_ld_global_b16_to_f32` instead.
            hidden_p3 = Int32(self.hidden_size)
            elems_per_thr_p3 = hidden_p3 // Int32(self.num_threads)
            _i3 = Int32(0)
            while _i3 < elems_per_thr_p3:
                flat3 = tid + _i3 * Int32(self.num_threads)
                gmem_byte = (seq_idx * self.hidden_size + flat3) * Int32(2)
                x_f32 = _ld_global_b16_to_f32(
                    attn_output_ptr + Int64(gmem_byte)
                )
                _st_shared_f32(
                    smem_x + Int64(flat3) * Int64(4),
                    x_f32,
                )
                _i3 = _i3 + Int32(1)

            cute.arch.sync_threads()

            # === Phase 3.3: Iterate slices assigned to this CTA ===
            s_start_p3 = bx * slices_per_cta
            s_end_raw_p3 = s_start_p3 + slices_per_cta
            s_end_p3 = s_end_raw_p3
            if s_end_p3 > num_slices:
                s_end_p3 = num_slices

            FP4_BS = Int32(FP4_BLOCK_SIZE)
            LOG2E_F = Float32(LOG2_E)
            num_h_blocks = hidden_p3 // FP4_BS

            s = s_start_p3
            while s < s_end_p3:
                # -------- Stage 3a: FC1 -> smem_interm_bf16[tile_s] --------
                j_base = s * tile_s
                j_local = Int32(0)
                while j_local < tile_s:
                    j = j_base + j_local

                    gate_acc = Float32(0.0)
                    up_acc = Float32(0.0)

                    k_i = Int32(0)
                    while k_i < elems_per_thr_p3:
                        h = tid + k_i * Int32(self.num_threads)
                        x_val = _ld_shared_f32(
                            smem_x + Int64(h) * Int64(4)
                        )

                        h_block = h // FP4_BS
                        byte_col = h >> Int32(1)
                        byte_addr_gate = gate_fp4_ptr + Int64(
                            j * (hidden_p3 >> Int32(1)) + byte_col
                        )
                        byte_addr_up = up_fp4_ptr + Int64(
                            j * (hidden_p3 >> Int32(1)) + byte_col
                        )
                        nib_lo_gate = _ld_global_u8(byte_addr_gate)
                        nib_lo_up = _ld_global_u8(byte_addr_up)
                        is_odd = h & Int32(1)
                        nib_gate = Int32(
                            ((nib_lo_gate >> (Uint32(is_odd) * Uint32(4)))
                             & Uint32(0xF))
                        )
                        nib_up = Int32(
                            ((nib_lo_up >> (Uint32(is_odd) * Uint32(4)))
                             & Uint32(0xF))
                        )

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
                interm_nblocks = tile_s // Int32(FP4_BLOCK_SIZE)
                blk_iter_max = (interm_nblocks + Int32(3)) >> Int32(2)
                blk_iter = Int32(0)
                while blk_iter < blk_iter_max:
                    my_block = warp + blk_iter * Int32(4)
                    my_block_valid = my_block < interm_nblocks
                    elem_idx = my_block * Int32(FP4_BLOCK_SIZE) + lane
                    my_val = Float32(0.0)
                    if my_block_valid and lane < Int32(FP4_BLOCK_SIZE):
                        addr = smem_interm_bf16 + Int64(elem_idx) * Int64(2)
                        bf16_u32 = _ld_shared_b16(addr)
                        f32_bits = Int32(
                            (bf16_u32 & Uint32(0xFFFF)) << Uint32(16)
                        )
                        my_val = _bitcast_i32_to_f32(f32_bits)

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
                        _st_shared_u8(
                            smem_interm_bf16
                            + Int64(elem_idx) * Int64(1),
                            nib,
                        )
                    cute.arch.sync_threads()

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
                if cutlass.const_expr(self._threads_per_row == 1):
                    rows_per_thread = Int32(self._rows_per_thread)
                    row_base_local = tid * rows_per_thread

                    rpt = self._rows_per_thread
                    acc_list = [Float32(0.0) for _ in range(rpt)]

                    iter_i = Int32(0)
                    while iter_i < tile_s:
                        h = iter_i
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

                    for r in cutlass.range_constexpr(rpt):
                        k_row_global = (
                            by * tile_k + row_base_local + Int32(r)
                        )
                        partial_idx = (
                            bz * slice_ctas * hidden_p3
                            + bx * hidden_p3
                            + k_row_global
                        )
                        _atomic_add_f32(
                            mlp_partial_ptr
                            + Int64(partial_idx) * Int64(4),
                            acc_list[r],
                        )
                else:
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

                    if cutlass.const_expr(self._threads_per_row >= 2):
                        out_acc = out_acc + shfl_xor_sync(out_acc, Int32(1))
                    if cutlass.const_expr(self._threads_per_row >= 4):
                        out_acc = out_acc + shfl_xor_sync(out_acc, Int32(2))
                    if cutlass.const_expr(self._threads_per_row >= 8):
                        out_acc = out_acc + shfl_xor_sync(out_acc, Int32(4))
                    if cutlass.const_expr(self._threads_per_row >= 16):
                        out_acc = out_acc + shfl_xor_sync(out_acc, Int32(8))

                    partial_idx = (
                        bz * slice_ctas * hidden_p3
                        + bx * hidden_p3
                        + k_row_global
                    )
                    if thread_in_row == Int32(0):
                        _atomic_add_f32(
                            mlp_partial_ptr
                            + Int64(partial_idx) * Int64(4),
                            out_acc,
                        )

                cute.arch.sync_threads()
                s = s + Int32(1)

            # === Phase 3.4: per-k-tile arrival counter + last-CTA gather ===
            _threadfence()
            cute.arch.sync_threads()

            if tid == Int32(0):
                count_idx = bz * num_k_tiles + by
                old = _atomic_add_u32(
                    mlp_arrival_ptr + Int64(count_idx) * Int64(4),
                    Int32(1),
                )
                is_last_flag = Int32(0)
                if old == (slice_ctas - Int32(1)):
                    is_last_flag = Int32(1)
                _st_shared_b32(
                    smem_last_flag + Int64(0),
                    Uint32(is_last_flag),
                )
            cute.arch.sync_threads()

            last_flag_u32 = _ld_shared_b32(smem_last_flag + Int64(0))
            is_last = Int32(last_flag_u32) == Int32(1)

            if is_last:
                if cutlass.const_expr(self._threads_per_row == 1):
                    rpt = self._rows_per_thread
                    rows_per_thread = Int32(rpt)
                    row_base_local = tid * rows_per_thread
                    for r in cutlass.range_constexpr(rpt):
                        k_row_global = (
                            by * tile_k + row_base_local + Int32(r)
                        )
                        output_idx = bz * hidden_p3 + k_row_global
                        val_f32 = Float32(0.0)
                        for bx_i in cutlass.range_constexpr(
                            self.slice_ctas
                        ):
                            slot_idx = (
                                bz * slice_ctas * hidden_p3
                                + Int32(bx_i) * hidden_p3
                                + k_row_global
                            )
                            val_f32 = val_f32 + _ld_global_f32(
                                mlp_partial_ptr
                                + Int64(slot_idx) * Int64(4)
                            )
                        _st_global_bf16_from_f32(
                            mlp_output_ptr + Int64(output_idx) * Int64(2),
                            val_f32,
                        )
                else:
                    k_row_global = by * tile_k + tid
                    output_idx = bz * hidden_p3 + k_row_global
                    val_f32 = Float32(0.0)
                    if tid < tile_k:
                        for bx_i in cutlass.range_constexpr(
                            self.slice_ctas
                        ):
                            slot_idx = (
                                bz * slice_ctas * hidden_p3
                                + Int32(bx_i) * hidden_p3
                                + k_row_global
                            )
                            val_f32 = val_f32 + _ld_global_f32(
                                mlp_partial_ptr
                                + Int64(slot_idx) * Int64(4)
                            )
                        _st_global_bf16_from_f32(
                            mlp_output_ptr + Int64(output_idx) * Int64(2),
                            val_f32,
                        )

            # =================================================================
            # C1.5: Phase 4 (secondary barrier + ε epilogue + next-layer
            # input_LN bake) deleted. Kernel ends at the Phase 3 MLP write
            # above. The next layer's input_LN runs from Python at layer
            # entry (see _backend.py + qwen3_5.py:Qwen3_5DecoderLayer.forward).
            #
            # Counter reset: pre-C1.5 the globally-last CTA reset
            # secondary_barrier and grid_barrier here. Both buffers are
            # now allocated fresh (torch.zeros) per call in
            # run_beta_coop_full, so no in-kernel reset is required.
            # phase3_arrival_count is reset inline by per-k-tile last-CTA
            # writers above; phase1_arrival_count is reset inside Phase 1
            # Phase C (see _kernel_phase_01).
            # =================================================================
