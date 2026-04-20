"""Gate-0 mechanical pre-flight for Phase D cute.compile+Constexpr refactor.

Uses the Phase 3b small-dim test config (hidden=128, interm=128, tile_s=64,
tile_k=32, slice_ctas=2) that already satisfies Phase_D_MLP_Kernel.__init__
asserts. Validates that the refactor:

  0a — imports without DSL type errors
  0b — constructs (eager cute.compile) without DSL type errors
  0c — produces distinct PTX per distinct Constexpr tuple under
       CUTE_MLP_KEEP_PTX=1 (the refactor thesis in miniature)
  0d — invokes compiled kernel on small dummy tensors without crash,
       output finite (no NaN/Inf)

Does NOT validate Qwen3.5-27B FP4 math correctness — that requires
the full 27B model and stays on Docker gates 2, 4, 5.

Run: `.venv/bin/python docs/research/phase_d_cute_compile_audit/jupyter_preflight.py`
     (also importable into a Jupyter notebook cell-by-cell.)
"""

from __future__ import annotations

import os
import pathlib
import sys
import time
from typing import Optional

import torch


CACHE_DIR = pathlib.Path.home() / ".cache" / "cutlass_dsl"


def gate_0a_import() -> tuple[bool, str]:
    """0a — module imports without DSL type errors."""
    try:
        from vllm.v1.attention.backends.cute_paged.mlp_kernel import (
            Phase_D_MLP_Kernel,
        )
        return True, f"OK: Phase_D_MLP_Kernel imported ({Phase_D_MLP_Kernel})"
    except Exception as e:
        return False, f"FAIL: import raised {type(e).__name__}: {e!r}"


def gate_0b_construct() -> tuple[bool, str, Optional[object]]:
    """0b — eager construct with small-dim config succeeds."""
    from vllm.v1.attention.backends.cute_paged.mlp_kernel import (
        Phase_D_MLP_Kernel,
    )
    try:
        t0 = time.monotonic()
        k = Phase_D_MLP_Kernel(
            hidden_size=128,
            intermediate_size=128,
            tile_s=64,
            tile_k=32,
            slice_ctas=2,
        )
        elapsed = time.monotonic() - t0
        if k._compiled is None:
            return False, (
                f"FAIL: construct returned but self._compiled is None "
                f"(eager compile didn't run — expected pre-refactor)"
            ), None
        return True, (
            f"OK: constructed + eagerly compiled in {elapsed:.2f}s; "
            f"self._compiled is populated"
        ), k
    except Exception as e:
        return False, f"FAIL: construct raised {type(e).__name__}: {e!r}", None


def gate_0c_module_hash_distinctness() -> tuple[bool, str]:
    """0c — two constructs with distinct Constexpr tuples produce distinct
    CuTe DSL module hashes (= distinct cache entries).

    The refactor's thesis is that the cache key varies with the Constexpr
    tuple, not the module source hash. We prove this directly by
    intercepting `BaseDSL.get_module_hash` — the function the DSL calls
    internally to compute the cache key during `cute.compile`.

    Note: `KeepPTX()` file-on-disk detection would be a more natural
    probe, but `nvidia_cutlass_dsl` does not actually write PTX files
    when `KeepPTX()` is passed as a Python compile option (despite the
    option being accepted — it's a silent no-op in the Python API;
    PTX dumps via env vars land elsewhere). Hash-interception is the
    reliable in-process probe.
    """
    from cutlass.base_dsl.dsl import BaseDSL

    original_get_module_hash = BaseDSL.get_module_hash
    captured_hashes: list[str] = []

    def capturing_get_module_hash(self, module, function_name):
        h = original_get_module_hash(self, module, function_name)
        captured_hashes.append(h)
        return h

    BaseDSL.get_module_hash = capturing_get_module_hash
    try:
        from vllm.v1.attention.backends.cute_paged.mlp_kernel import (
            Phase_D_MLP_Kernel,
        )
        _k1 = Phase_D_MLP_Kernel(
            hidden_size=128, intermediate_size=128,
            tile_s=64, tile_k=32, slice_ctas=2,
        )
        hashes_after_k1 = list(captured_hashes)
        _k2 = Phase_D_MLP_Kernel(
            hidden_size=128, intermediate_size=128,
            tile_s=128, tile_k=32, slice_ctas=1,
        )
        hashes_after_k2 = list(captured_hashes)
    finally:
        BaseDSL.get_module_hash = original_get_module_hash

    k1_hashes = set(hashes_after_k1)
    k2_new_hashes = set(hashes_after_k2) - k1_hashes

    if len(k1_hashes) >= 1 and len(k2_new_hashes) >= 1:
        sample = next(iter(k1_hashes))[:16]
        sample2 = next(iter(k2_new_hashes))[:16]
        return True, (
            f"OK: distinct module hashes per Constexpr tuple. "
            f"k1={len(k1_hashes)} hash(es), k2 added {len(k2_new_hashes)} "
            f"new hash(es). sample k1=<{sample}...> "
            f"sample k2=<{sample2}...>"
        )
    return False, (
        f"FAIL: module-hash probe captured "
        f"{len(hashes_after_k1)} for k1, {len(hashes_after_k2)} total, "
        f"no new hashes for k2 → cache key is NOT varying with Constexpr"
    )


def gate_0d_invoke(k: object) -> tuple[bool, str]:
    """0d — invoke compiled kernel on small dummy tensors, output finite."""
    if k is None:
        return False, "FAIL: kernel not available from 0b"
    device = "cuda"
    hidden, interm = 128, 128
    nat = 1
    tile_s, tile_k, slice_ctas = 64, 32, 2
    fp4_bs = 16
    num_k_tiles = hidden // tile_k

    x = torch.zeros((nat, hidden), dtype=torch.bfloat16, device=device)
    x[0, 0] = 1.0
    gate_w_fp4 = torch.zeros(
        (interm, hidden // 2), dtype=torch.uint8, device=device,
    )
    gate_w_scale = torch.zeros(
        (interm, hidden // fp4_bs), dtype=torch.uint8, device=device,
    )
    up_w_fp4 = torch.zeros_like(gate_w_fp4)
    up_w_scale = torch.zeros_like(gate_w_scale)
    down_w_fp4 = torch.zeros(
        (hidden, interm // 2), dtype=torch.uint8, device=device,
    )
    down_w_scale = torch.zeros(
        (hidden, interm // fp4_bs), dtype=torch.uint8, device=device,
    )
    partial = torch.zeros((nat, hidden), dtype=torch.float32, device=device)
    arrival = torch.zeros(
        (nat, num_k_tiles), dtype=torch.uint32, device=device,
    )
    output = torch.zeros((nat, hidden), dtype=torch.bfloat16, device=device)

    try:
        k(
            x, gate_w_fp4, gate_w_scale, up_w_fp4, up_w_scale,
            down_w_fp4, down_w_scale, partial, arrival, output,
            nat,
            gate_up_global_scale=1.0, down_global_scale=1.0,
        )
        torch.cuda.synchronize()
    except Exception as e:
        return False, (
            f"FAIL: kernel invocation raised "
            f"{type(e).__name__}: {e!r}"
        )

    finite = torch.isfinite(output).all().item()
    return (finite, (
        f"{'OK' if finite else 'FAIL'}: kernel invoked; "
        f"output.isfinite().all()={finite}; "
        f"output.abs().max()={output.abs().max().item()}"
    ))


def main() -> int:
    print("=" * 60)
    print("Gate 0 — Phase D cute.compile+Constexpr pre-flight")
    print("=" * 60)

    ok_0a, msg_0a = gate_0a_import()
    print(f"[0a] {msg_0a}")
    if not ok_0a:
        return 1

    ok_0b, msg_0b, k = gate_0b_construct()
    print(f"[0b] {msg_0b}")
    if not ok_0b:
        # Keep going — 0c and 0d may still be informative even if
        # 0b fails on unrefactored code (lazy-compile path).
        pass

    ok_0c, msg_0c = gate_0c_module_hash_distinctness()
    print(f"[0c] {msg_0c}")

    ok_0d, msg_0d = gate_0d_invoke(k) if k is not None else (
        False, "FAIL: skipped (0b did not produce kernel)",
    )
    print(f"[0d] {msg_0d}")

    print("=" * 60)
    all_ok = all([ok_0a, ok_0b, ok_0c, ok_0d])
    print(f"RESULT: {'ALL GATES PASSED' if all_ok else 'GATES FAILED'}")
    print("=" * 60)
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
