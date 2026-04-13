#!/usr/bin/env python3
"""Session 11: Minimal cross-warp reduction probe.

Tests ONLY the sync_o store + reduction logic from the decode kernel.
No QK, no softmax, no PV MMA — just store known values to sync_o SMEM,
sync_threads, then reduce and write output.

Three tests:
1. All warps store 1.0 → output should be 1.0 everywhere
2. Each warp stores its warp_id → output should be sum/4 with equal m,d
3. Exact decode kernel reduction code with known sync_o values

Volume-mount into container and run:
  python /app/nvllm/tests/nvllm/attention/test_reduction_only.py
"""
import sys
sys.path.insert(0, "/app/nvllm")

import torch
import logging

logging.basicConfig(level=logging.WARNING)

try:
    import cutlass
    from cutlass import cute
    from cutlass._mlir import ir as _mlir_ir
    from cutlass._mlir.dialects import llvm as _llvm_dialect
    from cutlass.cute.typing import BFloat16, Float32, Int32, Int64, Uint32
    from cutlass.cutlass_dsl import T, dsl_user_op
except ImportError:
    print("CUTLASS not available")
    sys.exit(0)

from vllm.v1.attention.backends.cute_paged.kernel import (
    _st_shared_f32,
    _ld_shared_f32,
    shared_ptr_to_i64,
    exp2_approx_ftz_f32,
    _fmax,
)


class ReductionOnlyProbe:
    """Minimal 4-warp probe: tests sync_o store + reduction only."""

    def __init__(self):
        self.cta_q = 16
        self.num_warps_kv = 4
        self.num_threads = 128

        # sync_o: 4 warps × 16 rows × 16 cols × 4 bytes
        self.sync_o_bytes = self.num_warps_kv * self.cta_q * 16 * 4  # 4096
        # sync_md: 4 warps × 16 rows × 8 bytes (m + d)
        self.sync_md_bytes = self.num_warps_kv * self.cta_q * 8  # 512
        self.smem_bytes = self.sync_o_bytes + self.sync_md_bytes
        self._compiled = None

    @cute.jit
    def _jit_launch(self, output, debug_dump,
                    group_size: Int32, gx: Int32):
        self._kernel(output, debug_dump, group_size).launch(
            grid=[gx, Int32(1), Int32(1)],
            block=[self.num_threads, 1, 1],
            smem=self.smem_bytes,
        )

    @cute.kernel
    def _kernel(self, output, debug_dump, group_size: Int32):
        """4-warp test: store known values to sync_o, reduce, write output.

        Each warp stores value = (warp+1) * 1.0 to ALL its sync_o positions.
        m = 0.0, d = 1.0 for all warps (uniform softmax = no rescaling).
        Expected output: o_final = (1+2+3+4) / (1+1+1+1) = 2.5 everywhere.
        """
        lane = cute.arch.lane_idx()
        warp = cute.arch.warp_idx()
        tid = warp * Int32(32) + lane
        group = lane >> Int32(2)
        sub = lane & Int32(3)

        smem = cute.arch.get_dyn_smem(cutlass.Uint8, alignment=128)
        sync_o = shared_ptr_to_i64(smem)
        sync_md = shared_ptr_to_i64(
            smem + Int32(self.sync_o_bytes))

        W16 = Int32(16)

        # === Phase 1: ALL warps store known values to sync_o ===
        # Value = (warp+1) * 1.0 for all positions
        val = Float32(1.0) + Float32(1.0) * Float32(warp)

        so_warp_off = warp * Int32(self.cta_q) * W16 * Int32(4)
        so_r0 = so_warp_off + group * W16 * Int32(4)
        so_r1 = so_warp_off + (group + Int32(8)) * W16 * Int32(4)
        lc0 = sub * Int32(2)
        lc8 = sub * Int32(2) + Int32(8)

        _st_shared_f32(sync_o + Int64(
            so_r0 + lc0 * Int32(4)), val)
        _st_shared_f32(sync_o + Int64(
            so_r0 + (lc0 + Int32(1)) * Int32(4)), val)
        _st_shared_f32(sync_o + Int64(
            so_r1 + lc0 * Int32(4)), val)
        _st_shared_f32(sync_o + Int64(
            so_r1 + (lc0 + Int32(1)) * Int32(4)), val)
        _st_shared_f32(sync_o + Int64(
            so_r0 + lc8 * Int32(4)), val)
        _st_shared_f32(sync_o + Int64(
            so_r0 + (lc8 + Int32(1)) * Int32(4)), val)
        _st_shared_f32(sync_o + Int64(
            so_r1 + lc8 * Int32(4)), val)
        _st_shared_f32(sync_o + Int64(
            so_r1 + (lc8 + Int32(1)) * Int32(4)), val)

        # m = 0.0, d = 1.0 for all warps (sub 0 only)
        if sub == Int32(0):
            md_w_off = warp * Int32(self.cta_q) * Int32(8)
            _st_shared_f32(sync_md + Int64(
                md_w_off + group * Int32(8)), Float32(0.0))
            _st_shared_f32(sync_md + Int64(
                md_w_off + group * Int32(8) + Int32(4)), Float32(1.0))
            _st_shared_f32(sync_md + Int64(
                md_w_off + (group + Int32(8)) * Int32(8)),
                Float32(0.0))
            _st_shared_f32(sync_md + Int64(
                md_w_off + (group + Int32(8)) * Int32(8)
                + Int32(4)), Float32(1.0))

        cute.arch.sync_threads()

        # === Phase 2: Dump sync_o for warp 0 (before reduction) ===
        # debug_dump[0..255] = sync_o for warp 0
        if warp == Int32(0):
            for _e in cutlass.range_constexpr(
                self.cta_q * 16 // 32
            ):
                elem = lane * Int32(self.cta_q * 16 // 32) \
                    + Int32(_e)
                row = elem >> Int32(4)
                col16 = elem & Int32(15)
                w_base = Int32(0 * self.cta_q * 16)
                o_read = _ld_shared_f32(sync_o + Int64(
                    (w_base + row * W16 + col16) * Int32(4)))
                debug_dump[elem] = o_read

        cute.arch.sync_threads()

        # === Phase 3: Cross-warp reduction (exact copy from kernel) ===
        if warp == Int32(0):
            for _e in cutlass.range_constexpr(
                self.cta_q * 16 // 32
            ):
                elem = lane * Int32(self.cta_q * 16 // 32) \
                    + Int32(_e)
                row = elem >> Int32(4)
                col16 = elem & Int32(15)

                m_final = Float32(-1e30)
                for _w in cutlass.range_constexpr(
                    self.num_warps_kv
                ):
                    m_w = _ld_shared_f32(sync_md + Int64(
                        Int32(_w * self.cta_q) * Int32(8)
                        + row * Int32(8)))
                    m_final = _fmax(m_final, m_w)

                o_final = Float32(0.0)
                d_final = Float32(0.0)
                for _w in cutlass.range_constexpr(
                    self.num_warps_kv
                ):
                    w_base = Int32(
                        _w * self.cta_q * 16)
                    o_w = _ld_shared_f32(sync_o + Int64(
                        (w_base + row * W16
                         + col16) * Int32(4)))
                    m_w = _ld_shared_f32(sync_md + Int64(
                        Int32(_w * self.cta_q) * Int32(8)
                        + row * Int32(8)))
                    d_w = _ld_shared_f32(sync_md + Int64(
                        Int32(_w * self.cta_q) * Int32(8)
                        + row * Int32(8) + Int32(4)))
                    rescale = exp2_approx_ftz_f32(
                        m_w - m_final)
                    o_final = o_final + o_w * rescale
                    d_final = d_final + d_w * rescale

                o_final = o_final / d_final

                if row < group_size:
                    # Write to flat output: row * 16 + col16
                    out_idx = row * Int32(16) + col16
                    output[out_idx] = o_final

                # Also dump to debug: offset 256 + elem
                debug_dump[Int32(256) + elem] = o_final

        cute.arch.sync_threads()

    def __call__(self, group_size=6):
        output = torch.zeros(16 * 16, dtype=torch.float32, device="cuda")
        debug_dump = torch.zeros(512, dtype=torch.float32, device="cuda")

        if self._compiled is None:
            print("Compiling reduction-only probe...")
            self._compiled = cute.compile(
                self._jit_launch,
                output, debug_dump,
                Int32(group_size), Int32(1),
            )
        self._compiled(
            output, debug_dump,
            Int32(group_size), Int32(1),
        )
        return (output.reshape(16, 16),
                debug_dump[:256].reshape(16, 16),
                debug_dump[256:].reshape(16, 16))


def main():
    print("=" * 60)
    print("Session 11: Reduction-only probe")
    print("=" * 60)

    probe = ReductionOnlyProbe()

    # Test: all warps store (warp+1)*1.0, m=0, d=1
    # Expected: o_final = sum(warp_val * rescale) / sum(d * rescale)
    # With m=0 for all, rescale = exp2(0-0) = 1.0
    # o_final = (1+2+3+4) / (1+1+1+1) = 10/4 = 2.5
    output, sync_o_dump, reduction_dump = probe(group_size=16)

    print("\n=== sync_o dump (warp 0 values, before reduction) ===")
    print("Expected: 1.0 everywhere (warp 0 stored 1.0)")
    print(f"Row 0: {[f'{x:.2f}' for x in sync_o_dump[0].tolist()]}")
    print(f"Row 1: {[f'{x:.2f}' for x in sync_o_dump[1].tolist()]}")
    nz_synco = (sync_o_dump.abs() > 1e-6).sum().item()
    print(f"Nonzero: {nz_synco}/256")

    print("\n=== Reduction output (all rows, o_final values) ===")
    print("Expected: 2.5 everywhere (with m=0, d=1 for all warps)")
    print(f"Row 0: {[f'{x:.2f}' for x in reduction_dump[0].tolist()]}")
    print(f"Row 1: {[f'{x:.2f}' for x in reduction_dump[1].tolist()]}")
    nz_red = (reduction_dump.abs() > 1e-6).sum().item()
    print(f"Nonzero: {nz_red}/256")

    print("\n=== Global output (row < group_size only) ===")
    print(f"Row 0: {[f'{x:.2f}' for x in output[0].tolist()]}")
    print(f"Row 1: {[f'{x:.2f}' for x in output[1].tolist()]}")
    nz_out = (output.abs() > 1e-6).sum().item()
    print(f"Nonzero: {nz_out}/256")

    # Check if all values are 2.5
    expected = 2.5
    diff = (reduction_dump - expected).abs()
    max_diff = diff.max().item()
    print(f"\nMax diff from expected {expected}: {max_diff:.6f}")

    # Pattern analysis
    nz_mask = reduction_dump.abs() > 1e-6
    nz_positions = nz_mask.nonzero(as_tuple=False)
    if len(nz_positions) < 256:
        nz_cols = sorted(set(nz_positions[:, 1].tolist()))
        nz_rows = sorted(set(nz_positions[:, 0].tolist()))
        print(f"Nonzero rows: {nz_rows}")
        print(f"Nonzero cols: {nz_cols}")

    if max_diff < 0.01:
        print("\nPASS: Reduction produces correct output")
    else:
        print(f"\nFAIL: Reduction is broken (max_diff={max_diff:.4f})")

        # Detailed analysis per position
        print("\nDetailed nonzero positions:")
        for r in range(min(4, 16)):
            for c in range(16):
                val = reduction_dump[r, c].item()
                if abs(val) > 1e-6:
                    print(f"  [{r},{c}] = {val:.4f}")


if __name__ == "__main__":
    main()
