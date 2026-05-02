"""Render side-by-side per-kernel μs comparison from two kernels.csv files.

Outputs:
  - markdown comparison table (top N by total_ms diff)
  - JSON summary with headline numbers

Usage:
    python render_comparison.py \
        --piecewise piecewise_kernels.csv \
        --full full_kernels.csv \
        --out-md comparison.md \
        --out-json comparison.json
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path


def read_csv(path: Path) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            sym = row["kernel_symbol"]
            rows[sym] = {
                "n_calls": int(row["n_calls"]),
                "mean_us": float(row["mean_us"]),
                "p50_us": float(row["p50_us"]),
                "p95_us": float(row["p95_us"]),
                "total_ms": float(row["total_ms"]),
            }
    return rows


def short_name(sym: str) -> str:
    """Best-effort short name for the markdown table."""
    if "PhaseE_Beta_Kernel" in sym:
        return "PhaseE_Beta_Kernel (β-coop fused attn+MLP)"
    if "Phase_D_MLP_Kernel" in sym:
        return "Phase_D_MLP_Kernel"
    if "DecodeKernel" in sym:
        return "DecodeKernel (CuTe paged attn)"
    if "fused_recurrent_gated_delta_rule" in sym:
        return "GDN linear-attn (fused_recurrent_gated_delta_rule)"
    if "_causal_conv1d_update_kernel" in sym:
        return "causal_conv1d_update"
    if "cvt_fp16_to_fp4" in sym:
        return "cvt_fp16_to_fp4"
    if "GEMM" in sym or "gemm" in sym or "f4f4bf16" in sym or "fp4" in sym.lower():
        return f"FP4 GEMM ({sym[:40]}…)"
    return sym[:80] + ("…" if len(sym) > 80 else "")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--piecewise", required=True, type=Path)
    ap.add_argument("--full", required=True, type=Path)
    ap.add_argument("--out-md", required=True, type=Path)
    ap.add_argument("--out-json", required=True, type=Path)
    ap.add_argument("--top-n", type=int, default=20)
    args = ap.parse_args()

    pw = read_csv(args.piecewise)
    fl = read_csv(args.full)

    common = sorted(set(pw) & set(fl))
    only_pw = sorted(set(pw) - set(fl))
    only_fl = sorted(set(fl) - set(pw))

    rows = []
    for sym in common:
        p, f = pw[sym], fl[sym]
        delta_mean = f["mean_us"] - p["mean_us"]
        delta_pct = (delta_mean / p["mean_us"] * 100.0) if p["mean_us"] > 0 else 0.0
        delta_total = f["total_ms"] - p["total_ms"]
        rows.append({
            "symbol": sym,
            "short": short_name(sym),
            "pw_calls": p["n_calls"],
            "pw_mean_us": p["mean_us"],
            "pw_total_ms": p["total_ms"],
            "fl_calls": f["n_calls"],
            "fl_mean_us": f["mean_us"],
            "fl_total_ms": f["total_ms"],
            "delta_mean_us": delta_mean,
            "delta_pct": delta_pct,
            "delta_total_ms": delta_total,
        })

    # Sort by absolute delta_total_ms — biggest workload-level shift first
    rows.sort(key=lambda r: abs(r["delta_total_ms"]), reverse=True)
    top = rows[: args.top_n]

    # Aggregate totals (summed total_ms across common kernels)
    pw_total_common = sum(pw[s]["total_ms"] for s in common)
    fl_total_common = sum(fl[s]["total_ms"] for s in common)

    md_lines = [
        "## Per-kernel comparison (PIECEWISE vs FULL+blessed)",
        "",
        f"Common kernels: {len(common)}. Top {min(args.top_n, len(top))} by absolute total_ms shift.",
        "",
        "| Kernel | PW calls | PW mean μs | FL calls | FL mean μs | Δ μs | Δ % | Δ total ms |",
        "|---|--:|--:|--:|--:|--:|--:|--:|",
    ]
    for r in top:
        md_lines.append(
            f"| `{r['short']}` | {r['pw_calls']} | {r['pw_mean_us']:.3f} | "
            f"{r['fl_calls']} | {r['fl_mean_us']:.3f} | "
            f"{r['delta_mean_us']:+.3f} | {r['delta_pct']:+.1f}% | "
            f"{r['delta_total_ms']:+.3f} |"
        )

    if only_pw:
        md_lines += [
            "",
            "### Kernels in PIECEWISE only (top 10 by total_ms)",
            "",
            "| Kernel | calls | mean μs | total ms |",
            "|---|--:|--:|--:|",
        ]
        only_pw_rows = sorted(only_pw, key=lambda s: pw[s]["total_ms"], reverse=True)[:10]
        for s in only_pw_rows:
            md_lines.append(
                f"| `{short_name(s)}` | {pw[s]['n_calls']} | "
                f"{pw[s]['mean_us']:.3f} | {pw[s]['total_ms']:.3f} |"
            )

    if only_fl:
        md_lines += [
            "",
            "### Kernels in FULL only (top 10 by total_ms)",
            "",
            "| Kernel | calls | mean μs | total ms |",
            "|---|--:|--:|--:|",
        ]
        only_fl_rows = sorted(only_fl, key=lambda s: fl[s]["total_ms"], reverse=True)[:10]
        for s in only_fl_rows:
            md_lines.append(
                f"| `{short_name(s)}` | {fl[s]['n_calls']} | "
                f"{fl[s]['mean_us']:.3f} | {fl[s]['total_ms']:.3f} |"
            )

    md_lines += [
        "",
        "### Aggregate (sum across common kernels)",
        "",
        f"- PIECEWISE total kernel time: **{pw_total_common:,.3f} ms**",
        f"- FULL total kernel time: **{fl_total_common:,.3f} ms**",
        f"- Δ: **{fl_total_common - pw_total_common:+.3f} ms** "
        f"({(fl_total_common - pw_total_common) / pw_total_common * 100.0:+.2f}%)",
    ]

    args.out_md.write_text("\n".join(md_lines) + "\n")

    summary = {
        "n_common_kernels": len(common),
        "n_only_piecewise": len(only_pw),
        "n_only_full": len(only_fl),
        "pw_total_ms_common": pw_total_common,
        "fl_total_ms_common": fl_total_common,
        "delta_total_ms": fl_total_common - pw_total_common,
        "delta_pct": (fl_total_common - pw_total_common) / pw_total_common * 100.0
                     if pw_total_common > 0 else 0.0,
        "top_kernels": [
            {k: r[k] for k in (
                "symbol", "short", "pw_calls", "pw_mean_us", "pw_total_ms",
                "fl_calls", "fl_mean_us", "fl_total_ms",
                "delta_mean_us", "delta_pct", "delta_total_ms",
            )} for r in top
        ],
    }
    args.out_json.write_text(json.dumps(summary, indent=2))

    print(f"Wrote {args.out_md} and {args.out_json}", file=sys.stderr)
    print(f"Common: {len(common)}  PW-only: {len(only_pw)}  FL-only: {len(only_fl)}", file=sys.stderr)
    print(f"Aggregate Δ total_ms: {fl_total_common - pw_total_common:+.3f} "
          f"({(fl_total_common - pw_total_common) / pw_total_common * 100.0:+.2f}%)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
