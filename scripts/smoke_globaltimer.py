"""Standalone CuTe-DSL smoke test for the globaltimer / clock64 / st.b64
PTX helpers added in Task 1.

Runs inside an existing nvllm:gb10 container with the source bind-mounted
(no rebuild). Compiles a minimal cooperative kernel that:
  1. Reads globaltimer at entry, writes to scratch[bx, 0]
  2. Spins for ~1us
  3. Reads globaltimer at exit, writes to scratch[bx, 1]
Then verifies on host that scratch[bx, 1] > scratch[bx, 0] for every CTA
and that the median delta is in a sane range (~10^3-10^5 ns).

Repeat with %clock64 fallback. Confirm both compile and emit monotonic u64.

NOTE: CuTe DSL variable scoping does not let names defined inside a
Constexpr branch escape (variables in `if use_gt: t0 = a` get treated as
defined in dynamic control flow even when use_gt is constexpr). To avoid
that, the two timer probes are written as separate @cute.kernel functions
rather than one parameterized kernel.

Run inside container (mount the host repo over the editable vllm install
so the Python source is fresh while the in-image _C.abi3.so stays valid):
  docker run --rm --gpus all --privileged \
    -v $(pwd):/app/nvllm -w /app/nvllm nvllm:gb10 \
    --entrypoint /bin/bash -c \
    '/opt/venv/bin/python /app/nvllm/scripts/smoke_globaltimer.py'
"""
from __future__ import annotations

import sys

import cutlass
import cutlass.cute as cute
import torch
from cutlass import Int32, Int64, Uint64

from vllm.v1.attention.backends.cute_paged.kernel import (
    _read_clock64_u64,
    _read_globaltimer_u64,
    _st_global_u64,
)


@cute.kernel
def smoke_kernel_globaltimer(scratch_ptr: Int64):
    bx = cute.arch.block_idx()[0]
    tid = cute.arch.thread_idx()[0]
    if tid == Int32(0):
        t0 = _read_globaltimer_u64()
        # Write t0 BEFORE the spin so it is consumed (defeats DCE) and
        # provides a memory dependency that prevents the compiler from
        # reordering the t1 read above the spin.
        _st_global_u64(scratch_ptr + Int64(bx * Int32(16) + Int32(0)), t0)
        # Spin a few hundred thousand iterations. Each loop body has a
        # global st (to scratch[bx*16+8] preview) so it has side effects
        # the compiler must keep.
        i = Int32(0)
        while i < Int32(20000):
            _st_global_u64(scratch_ptr + Int64(bx * Int32(16) + Int32(8)),
                           Uint64(i))
            i = i + Int32(1)
        t1 = _read_globaltimer_u64()
        _st_global_u64(scratch_ptr + Int64(bx * Int32(16) + Int32(8)), t1)


@cute.kernel
def smoke_kernel_clock64(scratch_ptr: Int64):
    bx = cute.arch.block_idx()[0]
    tid = cute.arch.thread_idx()[0]
    if tid == Int32(0):
        t0 = _read_clock64_u64()
        _st_global_u64(scratch_ptr + Int64(bx * Int32(16) + Int32(0)), t0)
        i = Int32(0)
        while i < Int32(20000):
            _st_global_u64(scratch_ptr + Int64(bx * Int32(16) + Int32(8)),
                           Uint64(i))
            i = i + Int32(1)
        t1 = _read_clock64_u64()
        _st_global_u64(scratch_ptr + Int64(bx * Int32(16) + Int32(8)), t1)


@cute.jit
def smoke_launch_globaltimer(scratch_ptr: Int64):
    smoke_kernel_globaltimer(scratch_ptr).launch(
        grid=[Int32(64), Int32(1), Int32(1)],
        block=[Int32(32), Int32(1), Int32(1)],
    )


@cute.jit
def smoke_launch_clock64(scratch_ptr: Int64):
    smoke_kernel_clock64(scratch_ptr).launch(
        grid=[Int32(64), Int32(1), Int32(1)],
        block=[Int32(32), Int32(1), Int32(1)],
    )


def run_one(use_globaltimer: bool) -> tuple[bool, float, float]:
    """Returns (monotonic_ok, median_delta, n_ctas)."""
    scratch = torch.zeros(64 * 2, dtype=torch.int64, device="cuda")
    label = "globaltimer" if use_globaltimer else "clock64"
    launch_fn = smoke_launch_globaltimer if use_globaltimer else smoke_launch_clock64
    # Wrap data_ptr in Int64 — proven pattern from
    # tests/kernels/cute/test_cute_cooperative_launch_capture.py:128
    print(f"[smoke] compiling {label} kernel...", flush=True)
    compiled = cute.compile(launch_fn, Int64(scratch.data_ptr()))
    print(f"[smoke] launching...", flush=True)
    compiled(Int64(scratch.data_ptr()))
    torch.cuda.synchronize()
    h = scratch.cpu().view(64, 2)
    t0 = h[:, 0].numpy().astype("uint64")
    t1 = h[:, 1].numpy().astype("uint64")
    deltas = (t1.astype("int64") - t0.astype("int64"))
    monotonic = bool((deltas > 0).all())
    median = float(sorted(deltas)[len(deltas) // 2])
    print(f"[smoke] {label}: monotonic={monotonic} median_delta={median} "
          f"min={int(deltas.min())} max={int(deltas.max())}")
    return monotonic, median, 64


def main() -> int:
    rc = 0
    for use_globaltimer in (True, False):
        try:
            ok, median, _n = run_one(use_globaltimer)
        except Exception as e:
            label = "globaltimer" if use_globaltimer else "clock64"
            print(f"[smoke] FAIL: {label} failed to compile/run: "
                  f"{type(e).__name__}: {e}", flush=True)
            rc = 1
            continue
        if not ok:
            print(f"[smoke] FAIL: non-monotonic ticks for "
                  f"{'globaltimer' if use_globaltimer else 'clock64'}")
            rc = 1
            continue
        # Sanity bounds: globaltimer is ns (~10^3 expected for 2k-iter spin).
        # clock64 is cycles (~10^4 expected at ~1.5GHz for the same spin).
        if use_globaltimer and not (100 < median < 10**6):
            print(f"[smoke] WARN: globaltimer median {median} ns out of "
                  f"expected 1e2..1e6 ns range")
        if (not use_globaltimer) and not (10**3 < median < 10**7):
            print(f"[smoke] WARN: clock64 median {median} cycles out of "
                  f"expected 1e3..1e7 range")
    return rc


if __name__ == "__main__":
    sys.exit(main())
