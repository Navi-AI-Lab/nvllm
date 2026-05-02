"""Reduce region_timings.npy + profile_kernels.csv into a calibrated
per-region CSV.

Inputs:
  - region_timings.npy: (num_ctas, 11, 2) int64 raw buffer.
  - profile_kernels.csv: per-kernel μs from
    docs/research/gemm_sweep/extract_e2e_kernels.py. Column name is
    `kernel_symbol` (NOT `Kernel Name`); rows include CuTe kernel name
    suffixes — match by substring.

Output:
  - region_breakdown.csv: per-region rows with median_ticks, p99_ticks,
    median_us (NaN unless globaltimer + calibrated), frac_of_kernel.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from vllm.v1.attention.backends.cute_paged.region_timing import (
    reduce_region_timings,
)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--buf", required=True)
    p.add_argument("--kernels", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--slice-ctas", type=int, default=8)
    p.add_argument("--num-k-tiles", type=int, default=8)
    p.add_argument("--num-seqs", type=int, default=1)
    p.add_argument("--tick-source", choices=("globaltimer", "clock64"),
                   default="globaltimer",
                   help="Set by the orchestrator from Task 2 smoke result")
    p.add_argument("--kernel-symbol-regex", default="PhaseE_Beta_Kernel",
                   help="Substring used to find the β-coop row in "
                        "profile_kernels.csv.kernel_symbol")
    args = p.parse_args()

    buf = np.load(args.buf)
    kernels = pd.read_csv(args.kernels)
    # The extractor at docs/research/gemm_sweep/extract_e2e_kernels.py
    # emits column `kernel_symbol`, NOT "Kernel Name".
    if "kernel_symbol" not in kernels.columns:
        raise SystemExit(
            f"ERROR: profile_kernels.csv has columns "
            f"{list(kernels.columns)} — expected `kernel_symbol`. "
            f"Verify --kernels path points at the output of "
            f"docs/research/gemm_sweep/extract_e2e_kernels.py."
        )
    kern_row = kernels[
        kernels["kernel_symbol"].str.contains(
            args.kernel_symbol_regex, case=False, na=False, regex=True,
        )
    ]
    if len(kern_row) == 0:
        raise SystemExit(
            f"ERROR: no row matched /{args.kernel_symbol_regex}/ in "
            f"kernel_symbol. Top-5 kernels by total_ms:\n"
            + kernels.nlargest(5, 'total_ms')[
                ['kernel_symbol', 'mean_us', 'total_ms']
              ].to_string(index=False)
        )
    # If multiple matches (e.g. multi-arch JIT compiles), take the row
    # with the largest total_ms — that is the production symbol.
    kern_row = kern_row.nlargest(1, 'total_ms').iloc[0]
    nsys_total_us = float(kern_row["mean_us"])

    df = reduce_region_timings(
        buf,
        slice_ctas=args.slice_ctas,
        num_k_tiles=args.num_k_tiles,
        num_seqs=args.num_seqs,
        tick_source=args.tick_source,
        nsys_total_us=nsys_total_us,
    )
    df.to_csv(args.out, index=False)

    print(f"[extract] matched kernel_symbol: {kern_row['kernel_symbol']}")
    print(f"[extract] nsys total μs (mean per-call): {nsys_total_us:.2f}")
    print(f"[extract] tick source: {args.tick_source}")
    print(f"[extract] regions:")
    for _, row in df.iterrows():
        med_us = (
            f"{row.median_us:7.2f}μs" if not np.isnan(row.median_us)
            else f"{row.median_ticks:9.0f} ticks"
        )
        frac = (
            f"{row.frac_of_kernel*100:5.1f}%"
            if not np.isnan(row.frac_of_kernel)
            else "  n/a "
        )
        print(
            f"  {row.region:32s}  n={row.n_active_ctas:3d}  "
            f"median={med_us}  frac={frac}"
        )

    # Headline: K-reducible fraction (regions 2 + 7 + 9)
    k_red_rows = df[df.region.isin([
        "phase1_wo_gemv",
        "phase3_3a_fc1_silu",
        "phase3_3c_fc2_atomic",
    ])]
    if k_red_rows.frac_of_kernel.isna().any():
        print("\n[extract] WARN: frac_of_kernel is NaN for some K-reducible "
              "regions (likely tick_source=clock64 without calibration). "
              "Verdict will use ticks-ratio against region 4-excluded sum "
              "as a coarse proxy.")
        # Coarse proxy: sum of K-reducible median_ticks divided by sum
        # of all WORK regions' median_ticks (excluding wait region 4).
        work = df[df.cta_class != "barrier_wait"]
        k_red_ticks = k_red_rows.median_ticks.sum()
        all_work_ticks = work.median_ticks.sum()
        k_red = k_red_ticks / all_work_ticks if all_work_ticks > 0 else 0.0
    else:
        k_red = float(k_red_rows.frac_of_kernel.sum())
    print(f"\n[extract] K-reducible region sum (regions 2 + 7 + 9): "
          f"{k_red*100:.1f}% of kernel")
    if k_red >= 0.50:
        verdict = "STRONG GO — prototype Extra Blocks split-K on FC1 first"
    elif k_red >= 0.40:
        verdict = "PROCEED — prototype FC1, 2 weeks budget"
    elif k_red >= 0.25:
        verdict = "CONDITIONAL — pursue only if NCU shows memory-bound"
    else:
        verdict = "NO-GO for K-parallel alone — broader restructuring needed"
    print(f"[extract] VERDICT: {verdict}")


if __name__ == "__main__":
    main()
