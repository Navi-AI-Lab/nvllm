#!/usr/bin/env python3
"""MMA probe with non-uniform B: set B columns to different BF16 values
directly (no FP8, no SMEM loading) and verify D output.

If this works: the FP8 dequant / SMEM loading path is broken.
If this also fails: the MMA hardware has a bug or the fragment mapping is wrong.
"""
import torch
import sys
import logging

logging.basicConfig(level=logging.WARNING)
sys.path.insert(0, "/app/nvllm")

try:
    import cutlass
    from cutlass import cute
    from cutlass.cute.typing import Float32, Int32, Int64, Uint32
except ImportError:
    print("CUTLASS not available")
    sys.exit(0)

from vllm.v1.attention.backends.cute_paged.kernel import (
    _mma_m16n8k16_f32, bf16_mma_m16n16k16_f32,
    _cvt_2f32_to_bf16x2, shared_ptr_to_i64,
    _st_shared_f32, _ld_shared_f32,
)

# Unique values per B-column (N-dimension = group for first m16n8k16)
# B col 0: 0.5, col 1: 0.25, col 2: 0.125, col 3: 2.0,
# col 4: 1.0, col 5: 0.375, col 6: 0.1875, col 7: 4.0
COL_VALS = [0.5, 0.25, 0.125, 2.0, 1.0, 0.375, 0.1875, 4.0]


class MultiValueBProbe:
    def __init__(self):
        self.num_threads = 32
        self.smem_bytes = 256  # for output
        self._compiled = None

    @cute.jit
    def _jit_launch(self, output, gx: Int32):
        self._kernel(output).launch(
            grid=[gx, Int32(1), Int32(1)],
            block=[self.num_threads, 1, 1],
            smem=self.smem_bytes,
        )

    @cute.kernel
    def _kernel(self, output):
        """Single m16n8k16 MMA with A=1.0, B=unique per column.
        B-column n (n=group) gets value COL_VALS[group].
        Expected: D[m, n] = 16 * COL_VALS[n]."""
        lane = cute.arch.lane_idx()
        group = lane >> Int32(2)
        sub = lane & Int32(3)

        one = _cvt_2f32_to_bf16x2(Float32(1.0), Float32(1.0))
        a0 = one
        a1 = one
        a2 = one
        a3 = one

        # Set B based on group value
        # Each group gets its unique value
        b_val = Float32(0.0)
        if group == Int32(0):
            b_val = Float32(0.5)
        if group == Int32(1):
            b_val = Float32(0.25)
        if group == Int32(2):
            b_val = Float32(0.125)
        if group == Int32(3):
            b_val = Float32(2.0)
        if group == Int32(4):
            b_val = Float32(1.0)
        if group == Int32(5):
            b_val = Float32(0.375)
        if group == Int32(6):
            b_val = Float32(0.1875)
        if group == Int32(7):
            b_val = Float32(4.0)

        b0 = _cvt_2f32_to_bf16x2(b_val, b_val)
        b1 = _cvt_2f32_to_bf16x2(b_val, b_val)

        d0, d1, d2, d3 = _mma_m16n8k16_f32(
            Float32(0.0), Float32(0.0),
            Float32(0.0), Float32(0.0),
            a0, a1, a2, a3, b0, b1)

        # Dump: output[lane * 4 + 0..3] = d0..d3
        base = lane * Int32(4)
        output[base] = d0
        output[base + Int32(1)] = d1
        output[base + Int32(2)] = d2
        output[base + Int32(3)] = d3

    def __call__(self):
        output = torch.zeros(128, dtype=torch.float32, device="cuda")
        if self._compiled is None:
            print("Compiling multi-value B probe...")
            self._compiled = cute.compile(
                self._jit_launch, output, Int32(1))
        self._compiled(output, Int32(1))
        return output.reshape(32, 4)


def main():
    probe = MultiValueBProbe()
    r = probe()

    print("=" * 70)
    print("MMA m16n8k16 with non-uniform B (BF16, no FP8)")
    print("  A = all 1.0, B col n = " + str(COL_VALS))
    print("  Expected D[m, n] = 16 * COL_VALS[n]")
    print("=" * 70)
    print()

    # For each sub, d0 should be D[group, sub*2]
    # Expected: 16 * COL_VALS[sub*2]
    print(f"{'Lane':>4} {'g':>2} {'s':>2} | "
          f"{'d0':>8} {'exp':>8} {'col':>4} | "
          f"{'d1':>8} {'exp':>8} {'col':>4}")
    print("-" * 70)

    all_ok = True
    for lane_id in range(8):  # first 8 lanes (group 0-1, all subs)
        g = lane_id >> 2
        s = lane_id & 3
        d0 = r[lane_id, 0].item()
        d1 = r[lane_id, 1].item()

        exp_d0 = 16 * COL_VALS[s * 2]
        exp_d1 = 16 * COL_VALS[s * 2 + 1]

        ok0 = abs(d0 - exp_d0) < 0.1
        ok1 = abs(d1 - exp_d1) < 0.1

        # Find which col d0 actually matches
        act_col0 = min(range(8),
                       key=lambda c: abs(d0 - 16*COL_VALS[c]))
        act_col1 = min(range(8),
                       key=lambda c: abs(d1 - 16*COL_VALS[c]))

        m0 = "  " if ok0 else "!!"
        m1 = "  " if ok1 else "!!"
        if not ok0 or not ok1:
            all_ok = False

        print(f"{lane_id:4d} {g:2d} {s:2d} | "
              f"{d0:8.1f} {exp_d0:8.1f} {act_col0:4d}{m0} | "
              f"{d1:8.1f} {exp_d1:8.1f} {act_col1:4d}{m1}")

    print()
    if all_ok:
        print("RESULT: MMA with direct BF16 B values is CORRECT")
        print("  → Bug is in the FP8 dequant / SMEM loading path")
    else:
        print("RESULT: MMA with direct BF16 B values is WRONG")
        print("  → Bug is in the MMA fragment mapping or hardware")


if __name__ == "__main__":
    main()
