# tests/kernels/cute/test_cute_cooperative_launch_capture.py
"""Probe: can CuTe DSL cooperative-launch kernels be captured by
torch.cuda.graph()? Informs whether Phase E β-coop is
eager-only or full-graph-compatible.

Gate 6.2.5 from spec.

Outcomes (any is a VALID Task 2 result):
  - PASS: β-coop can run under CUDA graph capture.
  - FAIL (capture raises): `cooperative=True` incompatible with
    torch.cuda.graph() on this stack. β-coop must run eager-only;
    β-lite handles all graph modes.
  - FAIL (empty/silent capture): the capture context accepts the
    kernel launch without raising, but the launch is NOT recorded
    into the graph (it eager-executes during the capture window).
    On replay, nothing happens. β-coop cannot run under naive
    torch.cuda.graph() capture on this stack. See
    memory:project_cute_not_capturing for the broader pattern
    across CuTe kernels (not cooperative-specific).
  - FAIL (replay mismatch): kernel behavior differs under replay.
    Report DONE_WITH_CONCERNS with eager vs replay diff.
"""
from __future__ import annotations

import pytest
import torch

CUTE_AVAILABLE = True
try:
    from cutlass import cute
    from cutlass._mlir.dialects import llvm as _llvm_dialect
    from cutlass.cute.typing import Int32, Int64
    from cutlass.cutlass_dsl import dsl_user_op
    import cuda.bindings.driver as _cuda_driver
except ImportError:
    CUTE_AVAILABLE = False


if CUTE_AVAILABLE:

    @dsl_user_op
    def _st_global_u32(addr: Int64, val: Int32, *, loc=None, ip=None) -> None:
        """Store u32 to global memory at byte address.

        Mirrors _st_global_f32 in cute_paged/kernel.py:903 but for u32.
        PTX ref: `st.global.u32 [addr], val`.
        """
        addr_ir = Int64(addr).ir_value(loc=loc, ip=ip)
        val_ir = Int32(val).ir_value(loc=loc, ip=ip)
        _llvm_dialect.inline_asm(
            None, [addr_ir, val_ir],
            "st.global.u32 [$0], $1;", "l,r",
            has_side_effects=True, loc=loc, ip=ip,
        )

    @cute.kernel
    def trivial_coop_kernel(out_ptr: Int64):
        """Trivial cooperative kernel: each CTA writes its block index + 1
        to out[bx]. Uses sync_threads (intra-CTA) as a minimal proxy for
        the structure of a real cooperative kernel; cooperative=True on
        the launch side is what we actually probe.

        Observable behavior: out[bx] == bx + 1 for bx in [0, grid_x).
        """
        tid_x, _, _ = cute.arch.thread_idx()
        bx, _, _ = cute.arch.block_idx()

        # Only lane 0 of each CTA writes — avoid 32 racing stores.
        if tid_x == Int32(0):
            # byte address of out[bx]: out_ptr + bx * 4
            byte_off = Int64(bx) * Int64(4)
            addr = out_ptr + byte_off
            _st_global_u32(addr, bx + Int32(1))

        # Barrier exercises an intra-CTA sync; combined with
        # cooperative=True on the launch side this verifies the capture
        # path handles the cooperative-launch attribute on the config.
        cute.arch.sync_threads()

    @cute.jit
    def launch_fn(out_ptr: Int64, stream):
        """Host JIT wrapper. cooperative=True is the bit under test."""
        trivial_coop_kernel(out_ptr).launch(
            grid=[8, 1, 1],
            block=[32, 1, 1],
            smem=0,
            stream=stream,
            cooperative=True,
        )


@pytest.mark.skipif(not CUTE_AVAILABLE,
                    reason="CUTLASS CuTe DSL not available")
@pytest.mark.xfail(
    strict=True,
    reason=(
        "CuTe DSL kernel launches are not captured by naive "
        "torch.cuda.graph() on this stack — see "
        "memory:project_cute_not_capturing. β-coop must route through "
        "vLLM's custom-op/PIECEWISE path or run eager-only. XPASS here "
        "means the gap has closed and β-coop can use graphs directly; "
        "flip the dispatcher."
    ),
)
def test_cooperative_launch_captures_under_cuda_graph():
    """Minimal cooperative kernel → torch.cuda.graph() capture → replay.

    Three-stage check — any failure stage is a valid Task 2 result:
      1. eager launch produces expected observable output
      2. torch.cuda.graph() context does not RAISE
      3. after capture, replay-only (eager launch absent) produces the
         expected output — i.e. the launch was actually recorded
    """
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    out = torch.zeros(8, dtype=torch.int32, device='cuda')

    stream = _cuda_driver.CUstream(
        int(torch.cuda.current_stream().cuda_stream)
    )

    expected = torch.arange(1, 9, dtype=torch.int32, device='cuda')

    # --- Stage 1: warm-up eager -----------------------------------------
    # First call JIT-compiles; cache so graph capture does not have to
    # compile mid-capture (compilation may allocate, which breaks capture).
    compiled = cute.compile(launch_fn, Int64(out.data_ptr()), stream)
    compiled(Int64(out.data_ptr()), stream)
    torch.cuda.synchronize()
    eager_result = out.clone()

    assert torch.equal(eager_result, expected), (
        f"Eager cooperative launch produced unexpected output. "
        f"got={eager_result.tolist()} expected={expected.tolist()}"
    )

    # --- Stage 2: attempt graph capture ---------------------------------
    out.zero_()
    g = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(g):
            compiled(Int64(out.data_ptr()), stream)
    except Exception as e:
        pytest.fail(
            f"Cooperative-launch capture raised: "
            f"{type(e).__name__}: {e}\n"
            "Phase E β-coop will need to run eager-only. Flag this in "
            "_backend.py dispatch."
        )

    # --- Stage 3: replay-only must reproduce the eager result ------------
    # This catches the silent "graph empty — launch eager-executed inside
    # capture window but was never recorded" mode. After zeroing `out` we
    # rely *solely* on the graph to fill it back in.
    out.zero_()
    torch.cuda.synchronize()

    g.replay()
    torch.cuda.synchronize()
    replay_result = out.clone()

    assert torch.equal(replay_result, expected), (
        "Graph replay did not reproduce the kernel's output. The "
        "`with torch.cuda.graph(g)` block did not raise, but the "
        "cooperative launch was NOT recorded into the graph (likely "
        "eager-executed on a stream torch capture did not observe, "
        "cf. memory:project_cute_not_capturing). "
        f"replay_result={replay_result.tolist()} "
        f"expected={expected.tolist()}. "
        "Phase E β-coop cannot run under naive torch.cuda.graph() "
        "capture on this stack — must run eager-only or via the same "
        "custom-op/PIECEWISE path the production paged-attention "
        "kernel uses."
    )
