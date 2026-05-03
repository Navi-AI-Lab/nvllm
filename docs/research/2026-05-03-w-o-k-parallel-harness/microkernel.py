"""Standalone W_O-only CuTe DSL microkernel for the K-parallel harness.

Adapts the W_O+gather portion of the production β-coop kernel
(``vllm/v1/attention/backends/cute_paged/phase_e_kernel.py``,
commit ``46ad9bbc5``, lines 4004-4254) into a self-contained kernel
parameterized by ``wo_split``. v1 deliberately strips:

* Phase 0 (input_layernorm)
* Phase 1 attention production (Q×K, softmax, ×V)
* Phase 3 (MLP), grid barrier between Phase 1 and Phase 3
* Phase B.5 RMSNorm + residual epilogue

Per-K-group block scales (``_ld_swizzled_scale``) ARE applied —
production parity is required for the round-trip claim and to keep
the NCU memory-vs-compute classification unbiased
(``README.md`` §3 input synthesis: "Per-K-group scales must be
included — dropping them makes the memory stream ~11% lighter and
biases the NCU memory-vs-compute classification.").

The microkernel has three phases:

  1. **W_O GEMV** — gated by ``bx < wo_split && by < num_kv_heads``.
     Each W_O CTA reads its slice of ``attn_output``, dequants the
     corresponding NVFP4 W_O weights via
     ``w_dequant = w_f32 * sf * wo_gs`` (mirrors
     ``phase_e_kernel.py:4078-4082``), and writes its FP32 partial
     into ``wo_output[seq_idx, wo_slot, :]``. K range per CTA is
     ``[k_start, k_end)`` with robust integer-divide bounds.
  2. **Post-W_O grid barrier** — atomic-counter increment + spin-wait
     primitive (mirrors ``phase_e_kernel.py:4371``). Cooperative
     launch is required.
  3. **Gather** — every CTA in the grid reads the ``total_wo_ctas``
     slots of ``wo_output[seq_idx, :, :]`` and sums them into
     ``final_out[seq_idx, :]``. Stores are 4-byte aligned and every
     CTA computes the same value, so the redundant writes converge.

Compile-time elision:

* For ``wo_split == 1`` the W_O gate collapses to ``bx == 0``, the
  per-CTA K range collapses to ``[0, k_dim)``, and the slot id
  collapses to ``by`` — structurally equivalent to today's
  W_O+gather.

NVFP4 dequant convention (per ``feedback_nvfp4_dequant_convention``):
the loader inverts the global scale at load time, so the kernel sees
``1/wgs`` and **multiplies** by it. The FP8 E4M3 per-K-group block
scale (``sf``) is applied with the same orientation as production
(multiplied alongside ``wo_gs``).

Disk-cache keying: the constexpr config (``wo_split``, ``hidden_size``,
etc.) is captured by Python closure, so the JIT body uses Python
``int`` for all compile-time arithmetic and ``range_constexpr`` works
naturally. To make ``apply_disk_cache_patch`` differentiate variants,
the JIT host wrapper takes the same config as ``cutlass.Constexpr[int]``
arguments — those flow into ``_structural_args_cache_key``
(``disk_cache.py:401``) and force a unique key per variant. The
constexpr args are otherwise unused inside the body (the closure
values are the source of truth for codegen). ``wo_num_k_tiles`` is a
*runtime* Int32 (not Constexpr) — the kernel CODE is unchanged when
it shifts, only the runtime offset arithmetic uses it. It therefore
does NOT need to enter the cache key.
"""

from __future__ import annotations

from typing import Callable

import torch

# All CuTe DSL machinery is guarded by ``_CUTE_AVAILABLE`` per the
# pattern in ``vllm/v1/attention/backends/cute_paged/kernel.py:26``.
_CUTE_AVAILABLE = False
try:
    import cutlass
    from cutlass import cute
    from cutlass.cute.typing import Float32, Int32, Int64  # noqa: F401
    import cuda.bindings.driver as _cuda_driver

    # Module-level PTX helpers from the production kernel module.
    from vllm.v1.attention.backends.cute_paged.kernel import (
        _atomic_add_u32,
        _extract_byte_from_b32,
        _fp4_nibble_to_f32,
        _ld_global_b16_to_f32,
        _ld_global_b32,
        _ld_global_f32,
        _ld_swizzled_scale,
        _ld_volatile_u32,
        _st_global_f32,
        _threadfence,
    )

    _CUTE_AVAILABLE = True
except ImportError as _exc:  # pragma: no cover - exercised on CPU-only hosts
    _IMPORT_ERROR = _exc


def make_w_o_microkernel(
    *,
    wo_split: int,
    hidden_size: int,
    num_kv_heads: int,
    num_q_heads: int,
    head_dim: int,
    num_threads: int,
    slice_ctas: int,
) -> Callable[..., None]:
    """Build a JIT-compiled CuTe DSL W_O+gather microkernel.

    See module docstring for the kernel structure. Returns a Python
    callable with the signature documented in the harness spec
    (README.md §1, §1a). The callable launches the compiled kernel on
    the current CUDA stream and returns ``None``; the caller is
    responsible for synchronization and timing.

    The launchable callable's signature is:

        kernel(
            attn_output, wo_weight,
            wo_scales,                  # uint8 [num_m_tiles, num_k_tiles, 32, 4, 4]
            wo_gs,
            wo_output, final_out, grid_barrier,
            num_active_tokens,
            wo_num_k_tiles,             # Int32 runtime arg
        )
    """
    if not _CUTE_AVAILABLE:
        raise RuntimeError(
            "CuTe DSL not available; microkernel cannot be built."
        ) from _IMPORT_ERROR

    # ---------- Validate constexpr parameters ----------
    if wo_split not in (1, 2, 4, 8):
        raise ValueError(f"wo_split must be in {{1,2,4,8}}, got {wo_split}")
    if slice_ctas < wo_split:
        raise ValueError(
            f"slice_ctas ({slice_ctas}) must be >= wo_split ({wo_split})"
        )
    if hidden_size % num_threads != 0:
        raise ValueError(
            f"hidden_size ({hidden_size}) must be a multiple of "
            f"num_threads ({num_threads})"
        )
    n_per_thr = hidden_size // num_threads
    if n_per_thr % 8 != 0:
        raise ValueError(
            f"hidden_size/num_threads ({n_per_thr}) must be a multiple of 8"
        )
    if num_q_heads % num_kv_heads != 0:
        raise ValueError(
            f"num_q_heads ({num_q_heads}) must be a multiple of "
            f"num_kv_heads ({num_kv_heads})"
        )

    # ---------- Closure-captured Python int constants ----------
    # All compile-time arithmetic uses these directly so the DSL sees
    # ``int`` rather than ``ArithValue`` (which would force runtime
    # codegen and break ``range_constexpr``).
    group_size_p1 = num_q_heads // num_kv_heads
    k_dim_const = group_size_p1 * head_dim
    total_wo_ctas = num_kv_heads * wo_split
    total_ctas_per_seq_grid = slice_ctas * num_kv_heads

    # ---------- @cute.kernel and @cute.jit defined as closures ----------
    @cute.kernel
    def _wo_kernel_body(
        attn_output_ptr: Int64,
        wo_weight_ptr: Int64,
        wo_scale_ptr: Int64,
        wo_gs_ptr: Int64,
        wo_output_ptr: Int64,
        final_out_ptr: Int64,
        grid_barrier_ptr: Int64,
        wo_weight_row_stride: Int32,
        wo_num_k_tiles: Int32,
        # Cache-keying tags. Body does not consume; closure-captured
        # constants are the source of truth for codegen. These flow
        # into ``apply_disk_cache_patch``'s structural args fingerprint
        # so each (wo_split, hidden_size, ...) tuple gets a distinct
        # disk-cache key.
        _tag_wo_split: cutlass.Constexpr[int],
        _tag_hidden_size: cutlass.Constexpr[int],
        _tag_num_kv_heads: cutlass.Constexpr[int],
        _tag_num_q_heads: cutlass.Constexpr[int],
        _tag_head_dim: cutlass.Constexpr[int],
        _tag_num_threads: cutlass.Constexpr[int],
        _tag_slice_ctas: cutlass.Constexpr[int],
    ):
        # Block / thread identification (mirror phase_e_kernel.py:3375).
        bx, by, bz = cute.arch.block_idx()
        lane = cute.arch.lane_idx()
        warp = cute.arch.warp_idx()
        tid = warp * Int32(32) + lane
        seq_idx = bz

        # =========================================================
        # Phase 1: W_O GEMV body. Gate at runtime; the closure-
        # captured ``wo_split`` is a Python int so codegen for
        # wo_split == 1 lets the compiler fold ``bx < 1`` to a
        # cheap ``bx == 0`` test.
        # =========================================================
        if bx < Int32(wo_split):
            if by < Int32(num_kv_heads):
                hd_wo = Int32(hidden_size)
                n_per_thr_wo = Int32(n_per_thr)
                my_row_base = tid * n_per_thr_wo

                # Spec: kernel sees 1/wgs and MULTIPLIES.
                wo_gs = _ld_global_f32(wo_gs_ptr)

                k_dim = Int32(k_dim_const)
                # Predeclare so the DSL's variable-flow analysis sees
                # the names regardless of which Python branch runs.
                # Constexpr-elide the K bounds for wo_split == 1.
                # Python-side branch on the closure value: only one
                # branch is traced.
                k_start = Int32(0)
                k_end = k_dim
                wo_slot = by
                if wo_split == 1:
                    # Identical to phase_e_kernel.py:4053 shape:
                    # ``while k_idx < k_dim`` with wo_slot == by.
                    pass
                else:
                    k_start = (k_dim * bx) // Int32(wo_split)
                    k_end = (k_dim * (bx + Int32(1))) // Int32(wo_split)
                    wo_slot = by * Int32(wo_split) + bx

                kv_head_idx = by

                # Outer-row group loop unrolled at compile time.
                # n_per_thr // 8 = 5 for hidden=5120, num_threads=128.
                for _out_group in cutlass.range_constexpr(
                    n_per_thr // 8
                ):
                    out_base_wo = my_row_base + Int32(_out_group * 8)

                    a0 = Float32(0.0)
                    a1 = Float32(0.0)
                    a2 = Float32(0.0)
                    a3 = Float32(0.0)
                    a4 = Float32(0.0)
                    a5 = Float32(0.0)
                    a6 = Float32(0.0)
                    a7 = Float32(0.0)

                    k_idx = k_start
                    while k_idx < k_end:
                        q_head_start = kv_head_idx * Int32(group_size_p1)
                        attn_base = (
                            seq_idx * Int32(num_q_heads) * Int32(head_dim)
                            + q_head_start * Int32(head_dim)
                        )
                        attn_val = _ld_global_b16_to_f32(
                            attn_output_ptr
                            + Int64((attn_base + k_idx) * Int32(2))
                        )
                        # W_O is [hidden, num_q_heads*hd]; the K
                        # column for this kv-head starts at
                        # q_head_start*hd, so abs_k = q_head_start *
                        # hd + k_idx. Mirror phase_e_kernel.py:4057.
                        abs_k = q_head_start * Int32(head_dim) + k_idx
                        k_byte = abs_k >> Int32(1)
                        k_is_hi = abs_k & Int32(1)
                        # Per-K-group block-scale group index — every
                        # 16 K elements share one FP8 E4M3 scale.
                        # Mirror phase_e_kernel.py:4061.
                        k_grp = abs_k >> Int32(4)

                        for _oi in cutlass.range_constexpr(8):
                            out_row = out_base_wo + Int32(_oi)
                            if out_row < hd_wo:
                                w_addr = (
                                    wo_weight_ptr
                                    + Int64(
                                        out_row * wo_weight_row_stride
                                        + k_byte
                                    )
                                )
                                aligned = w_addr & Int64(
                                    0xFFFFFFFFFFFFFFFC
                                )
                                raw = _ld_global_b32(aligned)
                                bpos = Int32(w_addr & Int64(3))
                                the_byte = _extract_byte_from_b32(
                                    raw, bpos
                                )
                                nib_shift = k_is_hi << Int32(2)
                                nib = (the_byte >> nib_shift) & Int32(0x0F)
                                w_f32 = _fp4_nibble_to_f32(nib)
                                # Per-K-group FP8 E4M3 block scale.
                                # Mirror phase_e_kernel.py:4079-4082:
                                #   w_dequant = w_f32 * sf * wo_gs
                                sf = _ld_swizzled_scale(
                                    wo_scale_ptr,
                                    out_row,
                                    k_grp,
                                    wo_num_k_tiles,
                                )
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

                    # Write per-CTA partial into wo_output slot.
                    wo_slot_base = wo_output_ptr + Int64(
                        (seq_idx * Int32(total_wo_ctas) + wo_slot)
                        * hd_wo * Int32(4)
                    )
                    for _oi in cutlass.range_constexpr(8):
                        out_row = out_base_wo + Int32(_oi)
                        if out_row < hd_wo:
                            if _oi == 0:
                                _st_global_f32(
                                    wo_slot_base
                                    + Int64(out_row * Int32(4)), a0)
                            if _oi == 1:
                                _st_global_f32(
                                    wo_slot_base
                                    + Int64(out_row * Int32(4)), a1)
                            if _oi == 2:
                                _st_global_f32(
                                    wo_slot_base
                                    + Int64(out_row * Int32(4)), a2)
                            if _oi == 3:
                                _st_global_f32(
                                    wo_slot_base
                                    + Int64(out_row * Int32(4)), a3)
                            if _oi == 4:
                                _st_global_f32(
                                    wo_slot_base
                                    + Int64(out_row * Int32(4)), a4)
                            if _oi == 5:
                                _st_global_f32(
                                    wo_slot_base
                                    + Int64(out_row * Int32(4)), a5)
                            if _oi == 6:
                                _st_global_f32(
                                    wo_slot_base
                                    + Int64(out_row * Int32(4)), a6)
                            if _oi == 7:
                                _st_global_f32(
                                    wo_slot_base
                                    + Int64(out_row * Int32(4)), a7)

        # ============================================================
        # Phase 2: Post-W_O grid barrier. Mirrors phase_e_kernel.py:4368.
        # All slice_ctas * num_kv_heads CTAs per seq_idx must arrive.
        # ============================================================
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
        while arrived < Int32(total_ctas_per_seq_grid):
            arrived = _ld_volatile_u32(
                grid_barrier_ptr + Int64(seq_idx * Int32(4))
            )

        # Block-internal sync ensures every thread sees the release
        # before reading the slots in the gather pass.
        cute.arch.sync_threads()

        # ============================================================
        # Phase 3: gather. All slice_ctas * num_kv_heads CTAs read all
        # ``total_wo_ctas`` slots of wo_output[seq_idx, :, :] and sum
        # into final_out[seq_idx, :].
        # ============================================================
        hd_c = Int32(hidden_size)
        n_per_thr_c = Int32(n_per_thr)
        my_start_c = tid * n_per_thr_c

        for _grp in cutlass.range_constexpr(n_per_thr // 8):
            for _ei in cutlass.range_constexpr(8):
                idx_c = my_start_c + Int32(_grp * 8 + _ei)
                gather_acc = Float32(0.0)
                cta_i = Int32(0)
                while cta_i < Int32(total_wo_ctas):
                    slot_addr = wo_output_ptr + Int64(
                        (seq_idx * Int32(total_wo_ctas) + cta_i)
                        * hd_c * Int32(4)
                        + idx_c * Int32(4)
                    )
                    gather_acc = gather_acc + _ld_global_f32(slot_addr)
                    cta_i = cta_i + Int32(1)
                # Race-tolerant store: every CTA computes the same
                # value at idx_c, 4-byte stores are atomic at the
                # hardware level, so the final_out slot converges.
                _st_global_f32(
                    final_out_ptr
                    + Int64(seq_idx * hd_c * Int32(4))
                    + Int64(idx_c * Int32(4)),
                    gather_acc,
                )

    @cute.jit
    def _wo_jit_launch(
        attn_output_ptr: Int64,
        wo_weight_ptr: Int64,
        wo_scale_ptr: Int64,
        wo_gs_ptr: Int64,
        wo_output_ptr: Int64,
        final_out_ptr: Int64,
        grid_barrier_ptr: Int64,
        wo_weight_row_stride: Int32,
        wo_num_k_tiles: Int32,
        nat: Int32,
        stream,
        # Cache-keying tags — see _wo_kernel_body docstring.
        _tag_wo_split: cutlass.Constexpr[int],
        _tag_hidden_size: cutlass.Constexpr[int],
        _tag_num_kv_heads: cutlass.Constexpr[int],
        _tag_num_q_heads: cutlass.Constexpr[int],
        _tag_head_dim: cutlass.Constexpr[int],
        _tag_num_threads: cutlass.Constexpr[int],
        _tag_slice_ctas: cutlass.Constexpr[int],
    ):
        _wo_kernel_body(
            attn_output_ptr,
            wo_weight_ptr,
            wo_scale_ptr,
            wo_gs_ptr,
            wo_output_ptr,
            final_out_ptr,
            grid_barrier_ptr,
            wo_weight_row_stride,
            wo_num_k_tiles,
            _tag_wo_split,
            _tag_hidden_size,
            _tag_num_kv_heads,
            _tag_num_q_heads,
            _tag_head_dim,
            _tag_num_threads,
            _tag_slice_ctas,
        ).launch(
            grid=[slice_ctas, num_kv_heads, nat],
            block=[num_threads, 1, 1],
            # No SMEM consumed by the microkernel; cute.kernel needs
            # a slot allocation regardless.
            smem=128,
            stream=stream,
            cooperative=True,
        )

    # ---------- Compile + return launchable callable ----------
    placeholder_i64 = Int64(0)
    placeholder_i32 = Int32(1)
    placeholder_stream = _cuda_driver.CUstream(
        int(torch.cuda.current_stream().cuda_stream)
    )
    compiled = cute.compile(
        _wo_jit_launch,
        placeholder_i64,  # attn_output_ptr
        placeholder_i64,  # wo_weight_ptr
        placeholder_i64,  # wo_scale_ptr
        placeholder_i64,  # wo_gs_ptr
        placeholder_i64,  # wo_output_ptr
        placeholder_i64,  # final_out_ptr
        placeholder_i64,  # grid_barrier_ptr
        placeholder_i32,  # wo_weight_row_stride
        placeholder_i32,  # wo_num_k_tiles
        placeholder_i32,  # nat
        placeholder_stream,
        wo_split,
        hidden_size,
        num_kv_heads,
        num_q_heads,
        head_dim,
        num_threads,
        slice_ctas,
    )

    def _launch(
        attn_output: torch.Tensor,
        wo_weight: torch.Tensor,
        wo_scales: torch.Tensor,
        wo_gs: torch.Tensor,
        wo_output: torch.Tensor,
        final_out: torch.Tensor,
        grid_barrier: torch.Tensor,
        num_active_tokens: int,
        wo_num_k_tiles: int,
    ) -> None:
        """Launch the W_O microkernel on the current CUDA stream.

        The harness owns synchronization and timing; this function
        returns immediately after enqueuing the launch.

        ``wo_scales`` shape: ``(num_m_tiles, num_k_tiles, 32, 4, 4)``,
        dtype ``torch.uint8`` (FP8 E4M3 reinterpret-cast). Layout:
        swizzled, see ``_ld_swizzled_scale`` at
        ``vllm/v1/attention/backends/cute_paged/kernel.py:773-803``.
        """
        # Light validation — ``assert`` only; harness handles richer
        # error reporting.
        assert attn_output.is_cuda and attn_output.dtype == torch.bfloat16
        assert attn_output.shape[0] >= num_active_tokens
        assert attn_output.shape[1] == num_q_heads * head_dim
        assert wo_weight.is_cuda and wo_weight.dtype == torch.uint8
        assert wo_weight.shape == (
            hidden_size,
            (num_q_heads * head_dim) // 2,
        )
        # wo_scales is uint8-typed (or fp8_e4m3fn-typed; both are 1
        # byte per element). Accept either; verify the byte count.
        assert wo_scales.is_cuda
        assert wo_scales.element_size() == 1, (
            f"wo_scales must be 1-byte dtype (uint8 or fp8_e4m3fn), "
            f"got {wo_scales.dtype}"
        )
        num_m_tiles = (hidden_size + 127) // 128
        num_k_groups = (num_q_heads * head_dim + 15) // 16
        expected_num_k_tiles = (num_k_groups + 3) // 4
        assert wo_num_k_tiles == expected_num_k_tiles, (
            f"wo_num_k_tiles={wo_num_k_tiles} but expected "
            f"{expected_num_k_tiles} for K={num_q_heads * head_dim}"
        )
        assert wo_scales.numel() * wo_scales.element_size() == (
            num_m_tiles * expected_num_k_tiles * 32 * 4 * 4
        ), (
            f"wo_scales byte count {wo_scales.numel()} != "
            f"num_m_tiles*num_k_tiles*32*4*4 = "
            f"{num_m_tiles * expected_num_k_tiles * 32 * 4 * 4}"
        )
        assert wo_gs.is_cuda and wo_gs.dtype == torch.float32
        assert wo_output.is_cuda and wo_output.dtype == torch.float32
        assert wo_output.shape == (
            num_active_tokens,
            total_wo_ctas,
            hidden_size,
        )
        assert final_out.is_cuda and final_out.dtype == torch.float32
        assert final_out.shape == (num_active_tokens, hidden_size)
        assert grid_barrier.is_cuda and grid_barrier.dtype == torch.int32
        assert grid_barrier.shape[0] >= num_active_tokens

        # All tensors must be contiguous so .data_ptr() is the start
        # of a row-major buffer.
        for t in (attn_output, wo_weight, wo_scales, wo_output,
                  final_out, grid_barrier, wo_gs):
            assert t.is_contiguous(), (
                f"tensor {tuple(t.shape)} must be contiguous"
            )

        stream = _cuda_driver.CUstream(
            int(torch.cuda.current_stream().cuda_stream)
        )
        compiled(
            Int64(attn_output.data_ptr()),
            Int64(wo_weight.data_ptr()),
            Int64(wo_scales.data_ptr()),
            Int64(wo_gs.data_ptr()),
            Int64(wo_output.data_ptr()),
            Int64(final_out.data_ptr()),
            Int64(grid_barrier.data_ptr()),
            Int32(int(wo_weight.shape[1])),
            Int32(int(wo_num_k_tiles)),
            Int32(int(num_active_tokens)),
            stream,
            wo_split,
            hidden_size,
            num_kv_heads,
            num_q_heads,
            head_dim,
            num_threads,
            slice_ctas,
        )

    # Expose useful constants as attributes for the harness.
    _launch.wo_split = wo_split
    _launch.total_wo_ctas = total_wo_ctas
    _launch.total_ctas_per_seq_grid = total_ctas_per_seq_grid
    _launch.k_dim = k_dim_const
    return _launch
