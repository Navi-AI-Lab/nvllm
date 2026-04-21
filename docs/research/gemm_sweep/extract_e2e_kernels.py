"""Extract per-kernel μs stats from a torch profiler trace (.pt.trace.json.gz).

Usage:
    .venv/bin/python docs/research/gemm_sweep/extract_e2e_kernels.py \\
        --trace <path-to-pt.trace.json.gz> \\
        --config <config_id> \\
        --out <path-to-output.csv>

Emits CSV rows: config_id,kernel_symbol,n_calls,mean_us,p50_us,p95_us,total_ms
"""
from __future__ import annotations

import argparse
import gzip
import json
from collections import defaultdict
from pathlib import Path


def extract(trace_path: Path, config_id: str) -> list[dict]:
    with gzip.open(trace_path, "rt") as f:
        data = json.load(f)  # ~180MB compressed but fits in RAM uncompressed
    events = data.get("traceEvents", data)
    kernels: dict[str, list[float]] = defaultdict(list)
    for ev in events:
        if ev.get("cat") != "kernel":
            continue
        name = ev.get("name", "")
        dur = ev.get("dur", 0)
        if dur <= 0 or not name:
            continue
        kernels[name].append(dur)
    rows = []
    for name, durs in kernels.items():
        srt = sorted(durs)
        rows.append({
            "config_id": config_id,
            "kernel_symbol": name,
            "n_calls": len(durs),
            "mean_us": sum(durs) / len(durs),
            "p50_us": srt[len(srt) // 2],
            "p95_us": srt[int(len(srt) * 0.95)],
            "total_ms": sum(durs) / 1000.0,
        })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", type=Path, required=True)
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    rows = extract(args.trace, args.config)
    rows.sort(key=lambda r: -r["total_ms"])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        f.write("config_id,kernel_symbol,n_calls,mean_us,p50_us,p95_us,total_ms\n")
        for r in rows:
            symbol_escaped = r["kernel_symbol"].replace('"', '""')
            f.write(
                f'{r["config_id"]},"{symbol_escaped}",{r["n_calls"]},'
                f'{r["mean_us"]:.3f},{r["p50_us"]:.3f},{r["p95_us"]:.3f},'
                f'{r["total_ms"]:.3f}\n'
            )
    print(f"Wrote {len(rows)} kernel rows -> {args.out}")


if __name__ == "__main__":
    main()
