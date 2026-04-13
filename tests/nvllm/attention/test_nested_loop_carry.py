#!/usr/bin/env python3
"""Session 11: Test range_constexpr outer loop + while inner loop.

The decode kernel has:
  for _md_c in range_constexpr(16):
      o0 = 0.0
      while page_idx < num_pages:
          o0 = o0 + value
      # use o0 here

This probe tests if the while loop exit values are correct across
multiple range_constexpr iterations.

Volume-mount into container and run:
  python /app/nvllm/tests/nvllm/attention/test_nested_loop_carry.py
"""
import sys
sys.path.insert(0, "/app/nvllm")

import torch
import logging

logging.basicConfig(level=logging.WARNING)

try:
    import cutlass
    from cutlass import cute
    from cutlass.cute.typing import Float32, Int32, Int64
except ImportError:
    print("CUTLASS not available")
    sys.exit(0)


class NestedLoopProbe:
    """range_constexpr(N) outer × while inner, dump per-iteration results."""

    def __init__(self, num_outer=4, num_carried=8):
        self.num_outer = num_outer
        self.num_carried = num_carried  # Must be 8 to match o0..o7
        self.num_threads = 128
        self.smem_bytes = 128
        self._compiled = None

    @cute.jit
    def _jit_launch(self, output, num_iters: Int32, gx: Int32):
        self._kernel(output, num_iters).launch(
            grid=[gx, Int32(1), Int32(1)],
            block=[self.num_threads, 1, 1],
            smem=self.smem_bytes,
        )

    @cute.kernel
    def _kernel(self, output, num_iters: Int32):
        """For each outer iteration _c, accumulate 8 values through
        a while loop, then write to output[_c * 128*8 + tid*8 + i]."""
        lane = cute.arch.lane_idx()
        warp = cute.arch.warp_idx()
        tid = warp * Int32(32) + lane

        for _c in cutlass.range_constexpr(self.num_outer):
            # Unique increment per _c AND per value index
            # val_i = (_c + 1) * (i + 1) per iteration
            o0 = Float32(0.0)
            o1 = Float32(0.0)
            o2 = Float32(0.0)
            o3 = Float32(0.0)
            o4 = Float32(0.0)
            o5 = Float32(0.0)
            o6 = Float32(0.0)
            o7 = Float32(0.0)

            idx = Int32(0)
            while idx < num_iters:
                o0 = o0 + Float32((_c + 1) * 1.0)
                o1 = o1 + Float32((_c + 1) * 2.0)
                o2 = o2 + Float32((_c + 1) * 3.0)
                o3 = o3 + Float32((_c + 1) * 4.0)
                o4 = o4 + Float32((_c + 1) * 5.0)
                o5 = o5 + Float32((_c + 1) * 6.0)
                o6 = o6 + Float32((_c + 1) * 7.0)
                o7 = o7 + Float32((_c + 1) * 8.0)
                cute.arch.sync_threads()
                idx = idx + Int32(1)

            # Write results for this outer iteration
            base = Int32(_c * 128 * 8) + tid * Int32(8)
            output[base + Int32(0)] = o0
            output[base + Int32(1)] = o1
            output[base + Int32(2)] = o2
            output[base + Int32(3)] = o3
            output[base + Int32(4)] = o4
            output[base + Int32(5)] = o5
            output[base + Int32(6)] = o6
            output[base + Int32(7)] = o7

            cute.arch.sync_threads()

    def __call__(self, num_iters=1):
        total = self.num_outer * 128 * 8
        output = torch.zeros(total, dtype=torch.float32, device="cuda")
        if self._compiled is None:
            print("Compiling nested loop probe...")
            self._compiled = cute.compile(
                self._jit_launch,
                output, Int32(num_iters), Int32(1),
            )
        self._compiled(output, Int32(num_iters), Int32(1))
        return output.reshape(self.num_outer, 128, 8)


def main():
    print("=" * 60)
    print("Session 11: Nested loop (range_constexpr + while) probe")
    print("=" * 60)

    probe = NestedLoopProbe(num_outer=4, num_carried=8)

    for num_iters in [1, 3]:
        result = probe(num_iters=num_iters)

        print(f"\n=== num_iters={num_iters} ===")
        all_pass = True
        for outer_c in range(4):
            # Expected: o_i = num_iters * (outer_c+1) * (i+1)
            expected = torch.tensor([
                num_iters * (outer_c + 1) * (i + 1)
                for i in range(8)
            ], dtype=torch.float32, device="cuda")

            # Check thread 0
            t0 = result[outer_c, 0]
            match = (t0 - expected).abs().max().item() < 0.001

            # Check ALL threads
            all_match = True
            for tid in range(128):
                diff = (result[outer_c, tid] - expected).abs()
                if diff.max().item() > 0.001:
                    all_match = False
                    break

            status = "PASS" if all_match else "FAIL"
            if not all_match:
                all_pass = False

            print(f"  outer_c={outer_c}: {status}")
            print(f"    Expected: {expected.tolist()}")
            print(f"    Thread 0: {t0.tolist()}")
            if not all_match:
                # Find first mismatch
                for tid in range(128):
                    diff = (result[outer_c, tid] - expected).abs()
                    if diff.max().item() > 0.001:
                        w = tid // 32
                        l = tid % 32
                        print(f"    First mismatch at tid={tid} (warp={w}, lane={l}):")
                        print(f"      Got: {result[outer_c, tid].tolist()}")
                        print(f"      Diff: {diff.tolist()}")
                        break

        print(f"\n  Overall: {'PASS' if all_pass else 'FAIL'}")

    # Test with 16 outer iterations (matching decode kernel)
    print("\n=== 16 outer iterations (matching decode kernel) ===")
    probe16 = NestedLoopProbe(num_outer=16, num_carried=8)
    result16 = probe16(num_iters=1)

    all_pass = True
    for outer_c in range(16):
        expected = torch.tensor([
            (outer_c + 1) * (i + 1) for i in range(8)
        ], dtype=torch.float32, device="cuda")

        ok = True
        for tid in range(128):
            diff = (result16[outer_c, tid] - expected).abs()
            if diff.max().item() > 0.001:
                ok = False
                break

        if not ok:
            all_pass = False
            print(f"  FAIL at outer_c={outer_c}:")
            print(f"    Expected: {expected.tolist()}")
            print(f"    Thread 0: {result16[outer_c, 0].tolist()}")

    if all_pass:
        print("  All 16 outer iterations: PASS")
    else:
        # Count how many iterations are wrong
        wrong = sum(1 for c in range(16) if (
            result16[c, 0] - torch.tensor([
                (c + 1) * (i + 1) for i in range(8)
            ], dtype=torch.float32, device="cuda")
        ).abs().max().item() > 0.001)
        print(f"  {wrong}/16 iterations wrong")


if __name__ == "__main__":
    main()
