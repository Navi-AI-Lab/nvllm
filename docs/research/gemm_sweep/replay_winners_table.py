# docs/research/gemm_sweep/replay_winners_table.py
"""Primary perf evidence for the winners table (spec §5 step 4).

Runs (shape × M) grid twice: baseline (NVLLM_FP4_GEMM_CONFIG_M256=11 forces
smoke_M256 on all shapes) vs table (no env var — hits lookup_m_mid_winner).
Calls ops.cutlass_scaled_fp4_mm directly so it exercises the real production
dispatcher; the standalone microbench binary bypasses it.

Input construction mirrors `benchmarks/kernels/benchmark_nvfp4_gemm.py` —
per-tensor global scales + `alpha = 1 / (a_gs * b_gs)`.

Must be run INSIDE the rebuilt nvllm:gb10 container — bare python cannot find
the torch op.
"""
from __future__ import annotations

import argparse
import csv
import os
import statistics
from pathlib import Path

import torch

# nvllm/vLLM torch bindings — matches benchmarks/kernels/benchmark_nvfp4_gemm.py.
from vllm import _custom_ops as ops
from vllm.scalar_type import scalar_types

FLOAT4_E2M1_MAX = scalar_types.float4_e2m1f.max()
FLOAT8_E4M3_MAX = torch.finfo(torch.float8_e4m3fn).max

SHAPES = [
    # (name, N, K) — matches benchmarks/.../2026-04-21-qwen35-27b/microbench.csv
    ("qkv_proj",     8192,  5120),
    ("o_proj",       5120,  6144),
    ("gate_up_proj", 34816, 5120),
    ("down_proj",    5120,  17408),
]
M_VALUES = [16, 32, 64, 128, 192, 256]

WARMUP = 50
TIMED  = 200


def make_inputs(m: int, n: int, k: int, device: str):
    """Build FP4-quantized A, B + their block-scales + alpha. Mirrors
    `benchmarks/kernels/benchmark_nvfp4_gemm.py:64-122` (build_nvfp4_runner)."""
    a_bf16 = torch.randn((m, k), device=device, dtype=torch.bfloat16) * 0.1
    b_bf16 = torch.randn((n, k), device=device, dtype=torch.bfloat16) * 0.1

    a_amax = torch.abs(a_bf16).max().to(torch.float32)
    b_amax = torch.abs(b_bf16).max().to(torch.float32)
    a_global_scale = FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX / a_amax
    b_global_scale = FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX / b_amax

    a_fp4, a_sf = ops.scaled_fp4_quant(a_bf16, a_global_scale)
    b_fp4, b_sf = ops.scaled_fp4_quant(b_bf16, b_global_scale)
    alpha = 1.0 / (a_global_scale * b_global_scale)
    return a_fp4, b_fp4, a_sf, b_sf, alpha


def time_op(m: int, n: int, k: int, warmup: int, timed: int,
            out_dtype: torch.dtype, device: str) -> list[float]:
    a, b, a_sf, b_sf, alpha = make_inputs(m, n, k, device)

    for _ in range(warmup):
        ops.cutlass_scaled_fp4_mm(a, b, a_sf, b_sf, alpha, out_dtype)
    torch.cuda.synchronize()

    start = [torch.cuda.Event(enable_timing=True) for _ in range(timed)]
    end   = [torch.cuda.Event(enable_timing=True) for _ in range(timed)]
    for i in range(timed):
        start[i].record()
        ops.cutlass_scaled_fp4_mm(a, b, a_sf, b_sf, alpha, out_dtype)
        end[i].record()
    torch.cuda.synchronize()
    return [s.elapsed_time(e) * 1000.0 for s, e in zip(start, end)]  # ms -> us


def run_grid(label: str) -> list[dict]:
    out_dtype = torch.bfloat16
    device = "cuda"
    rows = []
    for shape, n, k in SHAPES:
        for m in M_VALUES:
            us = time_op(m, n, k, WARMUP, TIMED, out_dtype, device)
            rows.append({
                "shape": shape, "M": m, "N": n, "K": k,
                "label": label,
                "min_us": min(us),
                "mean_us": statistics.mean(us),
                "p50_us": statistics.median(us),
            })
            print(f"  [{label}] {shape} M={m:>3}: min={min(us):.2f} mean={statistics.mean(us):.2f} μs")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--label", required=True, choices=["baseline", "table"],
                    help="baseline = NVLLM_FP4_GEMM_CONFIG_M256=11 forces smoke_M256; "
                         "table = no env var, hits lookup_m_mid_winner")
    args = ap.parse_args()

    env_idx = os.environ.get("NVLLM_FP4_GEMM_CONFIG_M256", "")
    if args.label == "baseline" and env_idx != "11":
        raise SystemExit("baseline mode requires NVLLM_FP4_GEMM_CONFIG_M256=11")
    if args.label == "table" and env_idx != "":
        raise SystemExit("table mode requires NVLLM_FP4_GEMM_CONFIG_M256 unset")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    rows = run_grid(args.label)
    with args.output.open("w") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {args.output} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
