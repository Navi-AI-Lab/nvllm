"""Regression test: PTX %globaltimer / %clock64 / st.b64 helpers compile
and produce monotonic u64 ticks. Mirrors scripts/smoke_globaltimer.py
but as a pytest suite.

GPU-required; skipped if no CUDA device available.

Implementation notes (matched to the smoke script):
  - The two timer probes are split into separate @cute.kernel functions
    rather than one parameterized kernel — CuTe DSL scoping does not let
    names defined in a Constexpr branch escape the branch (see smoke).
  - Each kernel reads t0, stores it, runs a side-effect spin loop (so the
    compiler cannot DCE the spin), then reads t1 and stores it. Without
    the side-effect spin, the back-to-back reads collapse and the deltas
    register as 0/1 ticks — monotonic check (>0) fails spuriously.
  - data_ptr() is wrapped in Int64() before passing to cute.compile —
    proven pattern from
    tests/kernels/cute/test_cute_cooperative_launch_capture.py.
"""
from __future__ import annotations

import pytest

cutlass = pytest.importorskip("cutlass")
torch = pytest.importorskip("torch")

if not torch.cuda.is_available():
    pytest.skip("no CUDA device", allow_module_level=True)


def _run_smoke(use_globaltimer: bool):
    import cutlass.cute as cute
    from cutlass import Int32, Int64, Uint64
    from vllm.v1.attention.backends.cute_paged.kernel import (
        _read_clock64_u64,
        _read_globaltimer_u64,
        _st_global_u64,
    )

    @cute.kernel
    def kern_gt(scratch_ptr: Int64):
        bx = cute.arch.block_idx()[0]
        tid = cute.arch.thread_idx()[0]
        if tid == Int32(0):
            t0 = _read_globaltimer_u64()
            _st_global_u64(scratch_ptr + Int64(bx * Int32(16) + Int32(0)),
                           t0)
            i = Int32(0)
            while i < Int32(20000):
                _st_global_u64(
                    scratch_ptr + Int64(bx * Int32(16) + Int32(8)),
                    Uint64(i))
                i = i + Int32(1)
            t1 = _read_globaltimer_u64()
            _st_global_u64(scratch_ptr + Int64(bx * Int32(16) + Int32(8)),
                           t1)

    @cute.kernel
    def kern_ck(scratch_ptr: Int64):
        bx = cute.arch.block_idx()[0]
        tid = cute.arch.thread_idx()[0]
        if tid == Int32(0):
            t0 = _read_clock64_u64()
            _st_global_u64(scratch_ptr + Int64(bx * Int32(16) + Int32(0)),
                           t0)
            i = Int32(0)
            while i < Int32(20000):
                _st_global_u64(
                    scratch_ptr + Int64(bx * Int32(16) + Int32(8)),
                    Uint64(i))
                i = i + Int32(1)
            t1 = _read_clock64_u64()
            _st_global_u64(scratch_ptr + Int64(bx * Int32(16) + Int32(8)),
                           t1)

    @cute.jit
    def launch_gt(scratch_ptr: Int64):
        kern_gt(scratch_ptr).launch(
            grid=[Int32(64), Int32(1), Int32(1)],
            block=[Int32(32), Int32(1), Int32(1)],
        )

    @cute.jit
    def launch_ck(scratch_ptr: Int64):
        kern_ck(scratch_ptr).launch(
            grid=[Int32(64), Int32(1), Int32(1)],
            block=[Int32(32), Int32(1), Int32(1)],
        )

    launch = launch_gt if use_globaltimer else launch_ck
    scratch = torch.zeros(64 * 2, dtype=torch.int64, device="cuda")
    compiled = cute.compile(launch, Int64(scratch.data_ptr()))
    compiled(Int64(scratch.data_ptr()))
    torch.cuda.synchronize()
    h = scratch.cpu().view(64, 2)
    deltas = (h[:, 1].numpy().astype("int64")
              - h[:, 0].numpy().astype("int64"))
    return deltas


def test_globaltimer_monotonic():
    deltas = _run_smoke(use_globaltimer=True)
    assert (deltas > 0).all(), (
        f"globaltimer non-monotonic: {deltas}"
    )


def test_clock64_monotonic():
    deltas = _run_smoke(use_globaltimer=False)
    assert (deltas > 0).all(), (
        f"clock64 non-monotonic: {deltas}"
    )
