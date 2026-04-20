"""Production-dim PTX capture harness for Phase A vs D2e attention diff.

Constructs the CuTe paged-attention DecodeKernel at Qwen3.5-27B dims
and invokes it once with zero-filled tensors to force JIT compile.
When run with CUTE_DSL_KEEP_PTX=1 + CUTE_DSL_DUMP_DIR=/workdir/ir_dump,
the emitted PTX/IR/CUBIN lands in the dump dir.

Sibling to docs/research/phase_a_ptx_diag/harness.py (MLP). The MLP
diff (commit 396c3bbcf) falsified source-hash drift for the MLP
kernel; this harness exercises the same diagnostic on the attention
kernel in the same cute_paged/ package.

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


# Qwen3.5-27B attention dims — from
# natfii/Qwen3.5-27B-NVFP4-Opus-GB10/config.json (snapshot 1496fc6e...).
NUM_Q_HEADS = 24        # num_attention_heads
NUM_KV_HEADS = 4        # num_key_value_heads
HEAD_DIM = 256          # head_dim (matches DECODE_CONFIG.head_dim)
PAGE_SIZE = 64          # DECODE_CONFIG.block_size — kernel hard-requires 64

# Single-decode workload — compile-trigger only. The kernel's JIT
# specialization is on config + arg types, not on grid extents, so a
# one-seq/one-token invocation emits the same PTX as a larger batch.
NUM_SEQS = 1
NUM_TOKENS = NUM_SEQS
MAX_PAGES_PER_SEQ = 4
NUM_PAGES = 16


def alloc_zero(shape, dtype, device="cuda"):
    return torch.zeros(shape, dtype=dtype, device=device)


def main() -> int:
    print(f"[harness] torch {torch.__version__}", flush=True)
    print(f"[harness] nvidia_cutlass_dsl {version('nvidia-cutlass-dsl')}",
          flush=True)
    print(f"[harness] CUDA available: {torch.cuda.is_available()}", flush=True)
    if not torch.cuda.is_available():
        print("[harness] FAIL: CUDA not available", flush=True)
        return 2
    print(f"[harness] device: {torch.cuda.get_device_name(0)}", flush=True)

    # Echo env vars so each image's dump dir log records them.
    for k in (
        "CUTE_DSL_KEEP_PTX", "CUTE_DSL_KEEP_IR", "CUTE_DSL_KEEP_CUBIN",
        "CUTE_DSL_DUMP_DIR", "CUTE_DSL_NO_CACHE", "CUTE_DSL_ARCH",
    ):
        print(f"[harness] env {k}={os.environ.get(k, '<unset>')}", flush=True)

    from vllm.v1.attention.backends.cute_paged.kernel import (
        DECODE_CONFIG, _get_compiled_kernel,
    )

    print(f"[harness] DECODE_CONFIG={DECODE_CONFIG}", flush=True)
    print("[harness] instantiating DecodeKernel via _get_compiled_kernel",
          flush=True)
    t0 = time.monotonic()
    kernel = _get_compiled_kernel(DECODE_CONFIG)
    print(
        f"[harness] instantiated in {time.monotonic() - t0:.2f}s; "
        f"_compiled={'set' if getattr(kernel, '_compiled', None) is not None else 'None'}",
        flush=True,
    )

    # Allocate zero tensors with shapes __call__ expects.
    # query: [num_tokens, num_q_heads, head_dim] BF16
    query = alloc_zero((NUM_TOKENS, NUM_Q_HEADS, HEAD_DIM), torch.bfloat16)
    # kv_cache: [num_pages, 2, page_size, num_kv_heads, head_dim] uint8 (FP8)
    kv_cache = alloc_zero(
        (NUM_PAGES, 2, PAGE_SIZE, NUM_KV_HEADS, HEAD_DIM), torch.uint8,
    )
    # page_table: [num_seqs, max_pages_per_seq] int32
    page_table = alloc_zero((NUM_SEQS, MAX_PAGES_PER_SEQ), torch.int32)
    # seq_lens: [num_seqs] int32 — zero is fine for compile trigger
    seq_lens = alloc_zero((NUM_SEQS,), torch.int32)

    total_bytes = sum(
        t.element_size() * t.numel() for t in (
            query, kv_cache, page_table, seq_lens,
        )
    )
    print(f"[harness] allocated {total_bytes / 1024**2:.1f} MB on GPU",
          flush=True)

    # Invoke the kernel — unfused path (no wo_/rmsnorm_/gate_ kwargs).
    # The kernel body is the same PTX whether fusion is enabled or not
    # (fused/unfused branches are selected at runtime via Int32 flags).
    print("[harness] invoking kernel (triggers compile on first call)",
          flush=True)
    t1 = time.monotonic()
    try:
        kernel(
            query=query,
            kv_cache=kv_cache,
            page_table=page_table,
            seq_lens=seq_lens,
            scale=1.0,
            k_scale=1.0,
            v_scale=1.0,
        )
        torch.cuda.synchronize()
    except Exception as e:
        print(
            f"[harness] FAIL: kernel call raised {type(e).__name__}: {e!r}",
            flush=True,
        )
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
        print("[harness] dump dir does not exist", flush=True)

    print("[harness] OK", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
