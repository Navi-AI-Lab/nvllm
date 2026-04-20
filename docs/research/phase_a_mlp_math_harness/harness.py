"""Phase A vs D2e MLP math harness — direct Phase_D_MLP_Kernel.__call__.

Runs the kernel under each image with three progressively more
production-like inputs and saves the output tensor. A sibling driver
runs the harness on both images; a diff step compares the saved
outputs elementwise.

Purpose: localize the Phase A Q2 math break. Today's PTX-diag
(commit 1f008b4fe) proved both CuTe kernels emit byte-identical
PTX between the two images, so the break must be in non-kernel
code (Python glue, marshalling, timing) OR in a runtime path the
PTX-diag harness does not exercise. This harness exercises the
production code path at the Phase_D_MLP_Kernel.__call__ boundary
with deterministic seeded tensors.

The harness intentionally avoids the `_mlp_op.py` opaque-op
wrapper — we call `Phase_D_MLP_Kernel.__call__` directly. If the
direct call produces matching output across images, the break is
in the wrapper / dispatch layer; if it diverges, the break is in
the kernel-call glue itself (init-time eager compile, Constexpr
marshalling, etc).

Usage (inside container, invoked by run_diagnostic.sh):
    python3 /workdir/harness.py
"""

from __future__ import annotations

import os
import sys
import time
import traceback
from importlib.metadata import version

import torch


HIDDEN = 5120            # Qwen3.5-27B hidden_size
INTERM = 17408           # Qwen3.5-27B intermediate_size
TILE_S = 256             # prefill-legacy preset
TILE_K = 640             # prefill-legacy preset
SLICE_CTAS = 8           # prefill-legacy preset
FP4_BLOCK_SIZE = 16      # NVFP4 standard


def alloc_zero(shape, dtype):
    return torch.zeros(shape, dtype=dtype, device="cuda").contiguous()


def seeded_bf16(shape, seed: int) -> torch.Tensor:
    g = torch.Generator(device="cpu").manual_seed(seed)
    t = torch.randn(shape, dtype=torch.float32, generator=g)
    return t.to(torch.bfloat16).cuda().contiguous()


def seeded_u8(shape, seed: int) -> torch.Tensor:
    g = torch.Generator(device="cpu").manual_seed(seed)
    t = (torch.rand(shape, generator=g) * 256.0).to(torch.uint8)
    return t.cuda().contiguous()


def allocate_kernel_buffers(nat: int):
    num_k_tiles = max(HIDDEN // TILE_K, 1)
    partial = alloc_zero((nat, HIDDEN), torch.float32)
    count = alloc_zero((nat, num_k_tiles), torch.int32)
    output = alloc_zero((nat, HIDDEN), torch.bfloat16)
    return partial, count, output


def run_once(kernel, name, x, gate_w, gate_s, up_w, up_s, down_w, down_s,
             gate_up_gs, down_gs, out_dir):
    nat = x.shape[0]
    partial, count, output = allocate_kernel_buffers(nat)
    t0 = time.monotonic()
    try:
        kernel(
            x, gate_w, gate_s, up_w, up_s, down_w, down_s,
            partial, count, output, nat,
            gate_up_global_scale=gate_up_gs,
            down_global_scale=down_gs,
        )
        torch.cuda.synchronize()
    except Exception as e:
        print(f"[harness] {name} FAIL: {type(e).__name__}: {e!r}",
              flush=True)
        traceback.print_exc()
        return False
    dt = time.monotonic() - t0
    out_host = output.detach().cpu().clone()
    # Fingerprint metrics — cheap comparators that surface before the
    # full elementwise diff step.
    out_f32 = out_host.to(torch.float32)
    finite = torch.isfinite(out_f32).all().item()
    abs_max = out_f32.abs().max().item()
    mean = out_f32.mean().item()
    sum_ = out_f32.sum().item()
    sum_sq = (out_f32 * out_f32).sum().item()
    print(
        f"[harness] {name}: nat={nat} dt={dt*1000:.1f}ms "
        f"finite={finite} absmax={abs_max:.6g} mean={mean:.6g} "
        f"sum={sum_:.6g} sumsq={sum_sq:.6g}",
        flush=True,
    )
    path = os.path.join(out_dir, f"{name}.pt")
    torch.save(out_host, path)
    print(f"[harness] {name}: saved {path} ({os.path.getsize(path)} B)",
          flush=True)
    return True


def main() -> int:
    print(f"[harness] torch {torch.__version__}", flush=True)
    print(f"[harness] cutlass-dsl {version('nvidia-cutlass-dsl')}",
          flush=True)
    print(f"[harness] CUDA device: {torch.cuda.get_device_name(0)}",
          flush=True)

    out_dir = "/workdir/out"
    os.makedirs(out_dir, exist_ok=True)

    for k in (
        "CUTE_DSL_NO_CACHE", "CUTE_MLP_KEEP_PTX", "CUTE_MLP_TILE",
    ):
        print(f"[harness] env {k}={os.environ.get(k, '<unset>')}",
              flush=True)

    from vllm.v1.attention.backends.cute_paged.mlp_kernel import (
        Phase_D_MLP_Kernel,
    )

    print("[harness] constructing Phase_D_MLP_Kernel at Qwen3.5-27B dims",
          flush=True)
    t0 = time.monotonic()
    kernel = Phase_D_MLP_Kernel(
        hidden_size=HIDDEN,
        intermediate_size=INTERM,
        tile_s=TILE_S,
        tile_k=TILE_K,
        slice_ctas=SLICE_CTAS,
    )
    dt = time.monotonic() - t0
    print(
        f"[harness] constructed in {dt:.2f}s "
        f"(D2e: lazy compile, no kernel work here; "
        f"Phase A: eager compile runs inside __init__)",
        flush=True,
    )
    print(
        f"[harness] _compiled state after __init__: "
        f"{'set' if getattr(kernel, '_compiled', None) is not None else 'None'}",
        flush=True,
    )

    # --- Case 1: all-zero tensors ---
    # Matches the PTX-diag harness exactly. If D2e and Phase A produce
    # different outputs here, the break is present on the simplest
    # possible path — zero weights, zero input, scale=1.0.
    print("[harness] --- case zero_nat1 ---", flush=True)
    nat1 = 1
    x = alloc_zero((nat1, HIDDEN), torch.bfloat16)
    gate_w = alloc_zero((INTERM, HIDDEN // 2), torch.uint8)
    gate_s = alloc_zero((INTERM, HIDDEN // FP4_BLOCK_SIZE), torch.uint8)
    up_w = alloc_zero((INTERM, HIDDEN // 2), torch.uint8)
    up_s = alloc_zero((INTERM, HIDDEN // FP4_BLOCK_SIZE), torch.uint8)
    down_w = alloc_zero((HIDDEN, INTERM // 2), torch.uint8)
    down_s = alloc_zero((HIDDEN, INTERM // FP4_BLOCK_SIZE), torch.uint8)
    if not run_once(
        kernel, "zero_nat1", x, gate_w, gate_s, up_w, up_s, down_w, down_s,
        1.0, 1.0, out_dir,
    ):
        return 3

    # --- Case 2: seeded FP4 weights + seeded input at nat=1 ---
    # Exercises the full compute path with non-trivial data.
    print("[harness] --- case seed_nat1 ---", flush=True)
    x = seeded_bf16((nat1, HIDDEN), 0xBEEF)
    gate_w = seeded_u8((INTERM, HIDDEN // 2), 0xC0FFEE_1)
    gate_s = seeded_u8((INTERM, HIDDEN // FP4_BLOCK_SIZE), 0xC0FFEE_2)
    up_w = seeded_u8((INTERM, HIDDEN // 2), 0xC0FFEE_3)
    up_s = seeded_u8((INTERM, HIDDEN // FP4_BLOCK_SIZE), 0xC0FFEE_4)
    down_w = seeded_u8((HIDDEN, INTERM // 2), 0xC0FFEE_5)
    down_s = seeded_u8((HIDDEN, INTERM // FP4_BLOCK_SIZE), 0xC0FFEE_6)
    # Plausible NVFP4 weight_global_scale — doesn't need to be quant-
    # correct, just non-trivial and consistent across images.
    gate_up_gs = 3.125e-3
    down_gs = 4.1667e-3
    if not run_once(
        kernel, "seed_nat1", x, gate_w, gate_s, up_w, up_s, down_w, down_s,
        gate_up_gs, down_gs, out_dir,
    ):
        return 4

    # --- Case 3: same seeded weights + nat=8 input (decode batch) ---
    # Production decode hits `nat` up to max_num_seqs. PTX-diag only
    # tested nat=1. If the break is shape-dependent, nat=8 catches it.
    print("[harness] --- case seed_nat8 ---", flush=True)
    nat8 = 8
    x8 = seeded_bf16((nat8, HIDDEN), 0xBEEF)
    if not run_once(
        kernel, "seed_nat8", x8, gate_w, gate_s, up_w, up_s, down_w, down_s,
        gate_up_gs, down_gs, out_dir,
    ):
        return 5

    # --- Case 4: repeat seed_nat1 — determinism check ---
    # If the first and second call of the same inputs differ, state
    # leaks across calls (e.g., partial/count not zeroed properly, or
    # stream-ordering bug).
    print("[harness] --- case seed_nat1_repeat ---", flush=True)
    x = seeded_bf16((nat1, HIDDEN), 0xBEEF)
    if not run_once(
        kernel, "seed_nat1_repeat", x, gate_w, gate_s, up_w, up_s, down_w,
        down_s, gate_up_gs, down_gs, out_dir,
    ):
        return 6

    print("[harness] OK — all cases completed", flush=True)
    print("[harness] listing /workdir/out", flush=True)
    for name in sorted(os.listdir(out_dir)):
        path = os.path.join(out_dir, name)
        print(f"[harness]   {name}  {os.path.getsize(path)}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
