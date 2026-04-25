#!/usr/bin/env python3
"""Summarize tile_sweep_*/{decode_small,decode_balanced}/ vs probe-4 baseline.

Reads:
  - decode_small/timing_lines.txt
  - decode_balanced/timing_lines.txt
  - decode_small/completion.json
  - decode_balanced/completion.json

Computes per-fused-full-attn-layer avg mlp_end - mlp_start in ms and prints
a side-by-side vs probe-4 baseline (25.5 ms / fused layer, prefill-legacy).
"""
from __future__ import annotations

import json
import re
import statistics
import sys
from pathlib import Path

BASELINE_PROBE4 = {
    "name": "prefill-legacy (probe-4 baseline)",
    "slice_ctas": 8,
    "per_layer_ms": 25.5,       # from findings.md — sync_end avg
    "mlp_kernel_avg_ms": 26.07,  # from torch profile
}

# Per-CTA grid parallelism estimates for Qwen3.5-27B (hidden=5120, interm=17408):
PRESET_GRID = {
    "prefill-legacy":     (8,  8, 1, 64),
    "decode-balanced":    (16, 8, 1, 128),
    "decode-small":       (32, 8, 1, 256),
}


def extract_mlp_per_layer_ms(timing_txt: Path) -> dict[str, float]:
    """Parse `[CUTE_TIMING] <tag>=<us>` lines and return mean μs per tag."""
    if not timing_txt.exists():
        return {}
    pat = re.compile(r"\[CUTE_TIMING\]\s+(\S+)=([\d.]+)us")
    buckets: dict[str, list[float]] = {}
    for line in timing_txt.read_text().splitlines():
        m = pat.search(line)
        if not m:
            continue
        tag, val = m.group(1), float(m.group(2))
        buckets.setdefault(tag, []).append(val)
    return {k: statistics.mean(v) for k, v in buckets.items()}


def main(argv: list[str]) -> int:
    here = Path(__file__).resolve().parent
    print(f"# Tile-preset sweep analysis — {here.name}\n")
    print("Baseline (probe-4, prefill-legacy, slice_ctas=8):")
    print(f"  per-fused-layer {BASELINE_PROBE4['per_layer_ms']:.1f} ms  "
          f"torch_profile MLP/call {BASELINE_PROBE4['mlp_kernel_avg_ms']:.2f} ms\n")
    header = f"| {'preset':<18} | {'slice_ctas':>10} | {'grid CTAs':>9} | {'mlp_kernel avg (us)':>20} | {'Q2 output':<40} |"
    print(header)
    print("|" + "-" * (len(header) - 2) + "|")
    for sub, preset in [("decode_small", "decode-small"),
                        ("decode_balanced", "decode-balanced")]:
        d = here / sub
        timings = extract_mlp_per_layer_ms(d / "timing_lines.txt")
        compl = d / "completion.json"
        out_snippet = ""
        if compl.exists():
            try:
                j = json.loads(compl.read_text())
                out_snippet = j["choices"][0]["text"].replace("\n", "\\n")[:40]
            except Exception as e:
                out_snippet = f"<parse error: {e!r}>"
        grid = PRESET_GRID.get(preset, ("?", "?", "?", "?"))
        # Our instrumented qwen3_5.py timestamps checkpoint tags; the MLP kernel
        # itself is visible either via the tag `mlp_op_end - mlp_op_start` or in
        # torch_profile. For the timing_lines.txt it's simplest to look for
        # `mlp_kernel_us` if the instrumentation logs it; otherwise synthesize
        # from the broad per-layer tag.
        mlp_avg = "?"
        for k in ("mlp_kernel_us", "mlp_forward_us", "mlp_end-mlp_start"):
            if k in timings:
                mlp_avg = f"{timings[k]:.0f}"
                break
        print(f"| {preset:<18} | {grid[0]:>10} | {grid[3]:>9} | {mlp_avg:>20} | {out_snippet:<40} |")
    print("")
    print("Raw timing tags found (per preset):")
    for sub in ("decode_small", "decode_balanced"):
        d = here / sub
        t = extract_mlp_per_layer_ms(d / "timing_lines.txt")
        print(f"  {sub}: {sorted(t)}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
