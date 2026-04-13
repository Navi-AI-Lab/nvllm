#!/usr/bin/env python3
"""Session 11: Test CuTe DSL while loop value propagation.

The decode kernel accumulates 13 values (o0..o7 + m_r0/1 + d_r0/1 + page_idx)
through a runtime while loop. This probe tests if all values survive
the loop exit correctly.

Volume-mount into container and run:
  python /app/nvllm/tests/nvllm/attention/test_while_loop_carry.py
"""
import sys
sys.path.insert(0, "/app/nvllm")

import torch
import logging

logging.basicConfig(level=logging.WARNING)

try:
    import cutlass
    from cutlass import cute
    from cutlass.cute.typing import Float32, Int32, Int64, BFloat16
except ImportError:
    print("CUTLASS not available")
    sys.exit(0)

from vllm.v1.attention.backends.cute_paged.kernel import (
    _st_shared_f32, _ld_shared_f32, shared_ptr_to_i64,
)


class WhileLoopProbe:
    """Test while loop with N loop-carried FP32 values.

    Each iteration adds (lane+1)*0.1 to all values.
    After K iterations, each value should be K*(lane+1)*0.1.
    """

    def __init__(self):
        self.num_threads = 128  # 4 warps
        self.smem_bytes = 128  # small scratch
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
        """While loop with 13 loop-carried values (matching decode kernel)."""
        lane = cute.arch.lane_idx()
        warp = cute.arch.warp_idx()
        tid = warp * Int32(32) + lane

        # Match decode kernel: 8 accum + 2 m + 2 d + 1 counter = 13
        o0 = Float32(0.0)
        o1 = Float32(0.0)
        o2 = Float32(0.0)
        o3 = Float32(0.0)
        o4 = Float32(0.0)
        o5 = Float32(0.0)
        o6 = Float32(0.0)
        o7 = Float32(0.0)
        m_r0 = Float32(0.0)
        m_r1 = Float32(0.0)
        d_r0 = Float32(0.0)
        d_r1 = Float32(0.0)

        # Increment per iteration = (lane+1) * 0.1 for o0..o7
        # Different base for each to detect swaps:
        # o0 += 1.0, o1 += 2.0, ..., o7 += 8.0
        # m_r0 += 0.5, m_r1 += 0.25, d_r0 += 0.125, d_r1 += 0.0625
        idx = Int32(0)
        while idx < num_iters:
            o0 = o0 + Float32(1.0)
            o1 = o1 + Float32(2.0)
            o2 = o2 + Float32(3.0)
            o3 = o3 + Float32(4.0)
            o4 = o4 + Float32(5.0)
            o5 = o5 + Float32(6.0)
            o6 = o6 + Float32(7.0)
            o7 = o7 + Float32(8.0)
            m_r0 = m_r0 + Float32(0.5)
            m_r1 = m_r1 + Float32(0.25)
            d_r0 = d_r0 + Float32(0.125)
            d_r1 = d_r1 + Float32(0.0625)
            idx = idx + Int32(1)

        cute.arch.sync_threads()

        # Write all 12 values to output
        # output layout: [128 threads × 12 values]
        base = tid * Int32(12)
        output[base + Int32(0)] = o0
        output[base + Int32(1)] = o1
        output[base + Int32(2)] = o2
        output[base + Int32(3)] = o3
        output[base + Int32(4)] = o4
        output[base + Int32(5)] = o5
        output[base + Int32(6)] = o6
        output[base + Int32(7)] = o7
        output[base + Int32(8)] = m_r0
        output[base + Int32(9)] = m_r1
        output[base + Int32(10)] = d_r0
        output[base + Int32(11)] = d_r1

    def __call__(self, num_iters=1):
        output = torch.zeros(128 * 12, dtype=torch.float32, device="cuda")
        if self._compiled is None:
            print("Compiling while-loop probe...")
            self._compiled = cute.compile(
                self._jit_launch,
                output, Int32(num_iters), Int32(1),
            )
        self._compiled(output, Int32(num_iters), Int32(1))
        return output.reshape(128, 12)


class WhileLoopWithSyncProbe:
    """Same as above but adds sync_threads INSIDE the while loop
    (matching decode kernel which has sync_threads at end of page loop body).
    """

    def __init__(self):
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
        """While loop with sync_threads in body + 13 carried values."""
        lane = cute.arch.lane_idx()
        warp = cute.arch.warp_idx()
        tid = warp * Int32(32) + lane

        o0 = Float32(0.0)
        o1 = Float32(0.0)
        o2 = Float32(0.0)
        o3 = Float32(0.0)
        o4 = Float32(0.0)
        o5 = Float32(0.0)
        o6 = Float32(0.0)
        o7 = Float32(0.0)
        m_r0 = Float32(0.0)
        m_r1 = Float32(0.0)
        d_r0 = Float32(0.0)
        d_r1 = Float32(0.0)

        idx = Int32(0)
        while idx < num_iters:
            o0 = o0 + Float32(1.0)
            o1 = o1 + Float32(2.0)
            o2 = o2 + Float32(3.0)
            o3 = o3 + Float32(4.0)
            o4 = o4 + Float32(5.0)
            o5 = o5 + Float32(6.0)
            o6 = o6 + Float32(7.0)
            o7 = o7 + Float32(8.0)
            m_r0 = m_r0 + Float32(0.5)
            m_r1 = m_r1 + Float32(0.25)
            d_r0 = d_r0 + Float32(0.125)
            d_r1 = d_r1 + Float32(0.0625)
            cute.arch.sync_threads()  # <-- KEY DIFFERENCE
            idx = idx + Int32(1)

        # Write all 12 values
        base = tid * Int32(12)
        output[base + Int32(0)] = o0
        output[base + Int32(1)] = o1
        output[base + Int32(2)] = o2
        output[base + Int32(3)] = o3
        output[base + Int32(4)] = o4
        output[base + Int32(5)] = o5
        output[base + Int32(6)] = o6
        output[base + Int32(7)] = o7
        output[base + Int32(8)] = m_r0
        output[base + Int32(9)] = m_r1
        output[base + Int32(10)] = d_r0
        output[base + Int32(11)] = d_r1

    def __call__(self, num_iters=1):
        output = torch.zeros(128 * 12, dtype=torch.float32, device="cuda")
        if self._compiled is None:
            print("Compiling while-loop-with-sync probe...")
            self._compiled = cute.compile(
                self._jit_launch,
                output, Int32(num_iters), Int32(1),
            )
        self._compiled(output, Int32(num_iters), Int32(1))
        return output.reshape(128, 12)


def check_results(result, num_iters, label):
    expected = torch.tensor([
        1.0 * num_iters,
        2.0 * num_iters,
        3.0 * num_iters,
        4.0 * num_iters,
        5.0 * num_iters,
        6.0 * num_iters,
        7.0 * num_iters,
        8.0 * num_iters,
        0.5 * num_iters,
        0.25 * num_iters,
        0.125 * num_iters,
        0.0625 * num_iters,
    ], device="cuda")

    print(f"\n=== {label} (num_iters={num_iters}) ===")
    print(f"Expected: {[f'{x:.4f}' for x in expected.tolist()]}")

    # Check thread 0 (warp 0, lane 0)
    t0 = result[0]
    print(f"Thread 0: {[f'{x:.4f}' for x in t0.tolist()]}")

    # Check thread 32 (warp 1, lane 0)
    t32 = result[32]
    print(f"Thread 32: {[f'{x:.4f}' for x in t32.tolist()]}")

    # Check ALL threads
    all_match = True
    for tid in range(128):
        diff = (result[tid] - expected).abs()
        if diff.max().item() > 0.001:
            warp = tid // 32
            lane = tid % 4
            group = (tid % 32) // 4
            sub = tid % 4
            print(f"  MISMATCH thread {tid} (warp={warp} group={group} sub={sub}): "
                  f"{[f'{x:.4f}' for x in result[tid].tolist()]}")
            print(f"    diff: {[f'{x:.4f}' for x in diff.tolist()]}")
            all_match = False
            if tid > 10:
                print(f"  ... (truncated, first 10 mismatches shown)")
                break

    # Count which values are wrong across all threads
    wrong_counts = torch.zeros(12, dtype=torch.int32)
    for tid in range(128):
        diff = (result[tid] - expected).abs()
        for i in range(12):
            if diff[i].item() > 0.001:
                wrong_counts[i] += 1

    if not all_match:
        print(f"  Wrong count per value: {wrong_counts.tolist()}")
        print(f"  Labels: [o0, o1, o2, o3, o4, o5, o6, o7, m0, m1, d0, d1]")

    print(f"{'PASS' if all_match else 'FAIL'}")
    return all_match


def main():
    print("=" * 60)
    print("Session 11: While loop carry-value probe")
    print("=" * 60)

    # Test 1: Simple while loop, 1 iteration
    probe1 = WhileLoopProbe()
    r1 = probe1(num_iters=1)
    check_results(r1, 1, "Simple while, 1 iter")

    # Test 2: Simple while loop, 3 iterations
    r2 = probe1(num_iters=3)
    check_results(r2, 3, "Simple while, 3 iters")

    # Test 3: While loop with sync_threads, 1 iteration
    probe2 = WhileLoopWithSyncProbe()
    r3 = probe2(num_iters=1)
    check_results(r3, 1, "While + sync_threads, 1 iter")

    # Test 4: While loop with sync_threads, 3 iterations
    r4 = probe2(num_iters=3)
    check_results(r4, 3, "While + sync_threads, 3 iters")


if __name__ == "__main__":
    main()
