"""Probe whether cute.make_rmem_tensor has register aliasing for 64-element Float32."""

import torch
import logging
logging.basicConfig(level=logging.WARNING)

import cutlass
from cutlass import cute
from cutlass._mlir.dialects import llvm as _llvm_dialect
from cutlass.cute.typing import Float32, Int32


class RmemProbe:
    def __init__(self):
        self.num_threads = 32
        self._compiled = None

    @cute.jit
    def _jit_launch(self, output):
        self._kernel(output).launch(
            grid=[1, 1, 1], block=[self.num_threads, 1, 1],
            smem=0,
        )

    @cute.kernel
    def _kernel(self, output):
        lane = cute.arch.lane_idx()

        t = cute.make_rmem_tensor(cute.make_layout((64,)), Float32)
        t.fill(Float32(0.0))

        # Write known values
        t[0] = Float32(1.0)
        t[1] = Float32(2.0)
        t[7] = Float32(3.0)

        # Read back all 64 elements (only lane 0)
        if lane == Int32(0):
            for _i in cutlass.range_constexpr(64):
                output[Int32(_i)] = t[_i]

    def __call__(self):
        output = torch.zeros(64, dtype=torch.float32, device="cuda")
        if self._compiled is None:
            self._compiled = cute.compile(self._jit_launch, output)
        self._compiled(output)
        return output


def main():
    print("rmem_tensor(64, Float32) aliasing probe")
    print("=" * 50)
    probe = RmemProbe()
    out = probe()
    vals = out.cpu().tolist()

    print("Expected: [0]=1.0, [1]=2.0, [7]=3.0, all others=0.0")
    print()
    for i, v in enumerate(vals):
        if abs(v) > 1e-6:
            print(f"  [{i:2d}] = {v:.1f}", end="")
            if i == 0:
                print(" (written 1.0)", end="")
            elif i == 1:
                print(" (written 2.0)", end="")
            elif i == 7:
                print(" (written 3.0)", end="")
            else:
                print(f" *** ALIAS with [{i % 8 if i >= 8 else '?'}]? ***", end="")
            print()

    # Check for specific aliasing patterns
    for stride in [8, 16, 32]:
        aliases = []
        for i in range(stride, 64):
            if abs(vals[i] - vals[i % stride]) > 1e-6 and abs(vals[i]) < 1e-6:
                continue
            if abs(vals[i] - vals[i % stride]) < 1e-6 and abs(vals[i]) > 1e-6:
                aliases.append(i)
        if aliases:
            print(f"\nStride-{stride} aliasing detected at: {aliases}")
        else:
            print(f"No stride-{stride} aliasing")


if __name__ == "__main__":
    main()
