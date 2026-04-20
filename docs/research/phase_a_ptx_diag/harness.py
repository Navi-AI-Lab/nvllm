"""Production-dim PTX capture harness for Phase A vs D2e MLP fusion diff.

Constructs Phase_D_MLP_Kernel at Qwen3.5-27B dimensions with the
prefill-legacy tile preset (256, 640, 8) and invokes it once with
zero-filled tensors to force JIT compile. When run with
CUTE_DSL_KEEP_PTX=1 + CUTE_DSL_DUMP_DIR=/workdir/ir_dump, the emitted
PTX/IR/CUBIN lands in the dump dir.

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


HIDDEN = 5120           # Qwen3.5-27B hidden_size
INTERM = 17408          # Qwen3.5-27B intermediate_size
TILE_S = 256            # prefill-legacy preset
TILE_K = 640            # prefill-legacy preset
SLICE_CTAS = 8          # prefill-legacy preset
FP4_BLOCK_SIZE = 16     # NVFP4 standard
NAT = 1                 # single-token batch — plenty to trigger compile


def alloc_zero(shape, dtype, device="cuda"):
    return torch.zeros(shape, dtype=dtype, device=device)


def main() -> int:
    print(f"[harness] torch {torch.__version__}", flush=True)
    print(f"[harness] nvidia_cutlass_dsl {version('nvidia-cutlass-dsl')}", flush=True)
    print(f"[harness] CUDA available: {torch.cuda.is_available()}", flush=True)
    if not torch.cuda.is_available():
        print("[harness] FAIL: CUDA not available", flush=True)
        return 2
    print(f"[harness] device: {torch.cuda.get_device_name(0)}", flush=True)

    # Echo env vars so each image's dump dir log records them.
    for k in (
        "CUTE_DSL_KEEP_PTX", "CUTE_DSL_KEEP_IR", "CUTE_DSL_KEEP_CUBIN",
        "CUTE_DSL_DUMP_DIR", "CUTE_DSL_NO_CACHE", "CUTE_DSL_ARCH",
        "CUTE_MLP_TILE", "CUTE_MLP_FUSION",
    ):
        print(f"[harness] env {k}={os.environ.get(k, '<unset>')}", flush=True)

    from vllm.v1.attention.backends.cute_paged.mlp_kernel import (
        Phase_D_MLP_Kernel,
    )

    print("[harness] constructing kernel at production dims", flush=True)
    t0 = time.monotonic()
    kernel = Phase_D_MLP_Kernel(
        hidden_size=HIDDEN,
        intermediate_size=INTERM,
        tile_s=TILE_S,
        tile_k=TILE_K,
        slice_ctas=SLICE_CTAS,
    )
    print(
        f"[harness] constructed in {time.monotonic() - t0:.2f}s; "
        f"_compiled={'set' if getattr(kernel, '_compiled', None) is not None else 'None'}",
        flush=True,
    )

    # Allocate zero tensors with the shapes __call__ expects.
    x = alloc_zero((NAT, HIDDEN), torch.bfloat16)
    gate_w_fp4 = alloc_zero((INTERM, HIDDEN // 2), torch.uint8)
    gate_w_scale = alloc_zero((INTERM, HIDDEN // FP4_BLOCK_SIZE), torch.uint8)
    up_w_fp4 = alloc_zero((INTERM, HIDDEN // 2), torch.uint8)
    up_w_scale = alloc_zero((INTERM, HIDDEN // FP4_BLOCK_SIZE), torch.uint8)
    down_w_fp4 = alloc_zero((HIDDEN, INTERM // 2), torch.uint8)
    down_w_scale = alloc_zero((HIDDEN, INTERM // FP4_BLOCK_SIZE), torch.uint8)
    mlp_partial_fp32 = alloc_zero((NAT, HIDDEN), torch.float32)
    num_k_tiles = max(HIDDEN // TILE_K, 1)
    mlp_arrival_count = alloc_zero((NAT, num_k_tiles), torch.int32)
    mlp_output = alloc_zero((NAT, HIDDEN), torch.bfloat16)

    total_bytes = sum(
        t.element_size() * t.numel() for t in (
            x, gate_w_fp4, gate_w_scale, up_w_fp4, up_w_scale,
            down_w_fp4, down_w_scale, mlp_partial_fp32,
            mlp_arrival_count, mlp_output,
        )
    )
    print(f"[harness] allocated {total_bytes / 1024**2:.1f} MB on GPU", flush=True)

    print("[harness] invoking kernel (triggers compile on first call for D2e, "
          "noop-compile-already-done for Phase A)", flush=True)
    t1 = time.monotonic()
    try:
        kernel(
            x, gate_w_fp4, gate_w_scale, up_w_fp4, up_w_scale,
            down_w_fp4, down_w_scale, mlp_partial_fp32,
            mlp_arrival_count, mlp_output, NAT,
        )
        torch.cuda.synchronize()
    except Exception as e:
        print(f"[harness] FAIL: kernel call raised {type(e).__name__}: {e!r}", flush=True)
        traceback.print_exc()
        return 3
    print(
        f"[harness] compile+launch completed in {time.monotonic() - t1:.2f}s",
        flush=True,
    )

    dump_dir = os.environ.get("CUTE_DSL_DUMP_DIR", ".")
    print(f"[harness] listing {dump_dir}", flush=True)
    if os.path.isdir(dump_dir):
        for name in sorted(os.listdir(dump_dir)):
            path = os.path.join(dump_dir, name)
            size = os.path.getsize(path) if os.path.isfile(path) else -1
            print(f"[harness]   {name}  {size}", flush=True)
    else:
        print(f"[harness] dump dir does not exist", flush=True)

    print("[harness] OK", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
