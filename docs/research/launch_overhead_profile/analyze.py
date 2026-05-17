"""Per-β-coop-call launch overhead breakdown from torch profiler trace.

Friend's breakdown columns (committed in summary.md):
    host_event_wall_us       — from host_launch_walls.npy (per-call CUDA event)
    kernel_dur_us            — GPU kernel duration from torch profiler event
    launch_api_dur_us        — cudaLaunch{...}Kernel{...} CPU op duration
    queue_gap_us             — gpu_kernel_start_ts - cpu_launch_op_end_ts
    region_sum_us            — sum of per-CTA medians for work regions
                               (from region_timings.npy + reduce_region_timings)
    wall_minus_regions_us    — host_event_wall - region_sum (B' diagnostic)

We pair each per-call CUDA event wall (host_launch_walls.npy) with a torch
profiler β-coop call from the same window. host_launch_walls accumulates
across warmup + timed window since the queue isn't drained until the
sentinel touch — but the npy is overwritten only on the dump call, so the
file as captured by capture.sh contains ALL walls. analyzer takes the LAST
TIMED_N entries as the timed-window subset (warmup ahead of them).

Usage:
    .venv/bin/python docs/research/launch_overhead_profile/analyze.py \\
        --out-dir benchmarks/nvllm/traces/cute_paged_attn/2026-05-16-launch-overhead-profile \\
        --timed-n 200
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import sys
from pathlib import Path

import numpy as np


# Cooperative launch can appear under any of these runtime API names
# depending on torch / driver version. Match on prefix.
LAUNCH_OP_PREFIXES = (
    "cudaLaunchCooperativeKernel",
    "cudaLaunchKernelExC",
    "cudaLaunchKernel",
)

# Outer NVTX range emitted by vLLM via torch.profiler.record_function() in
# the Phase E β-coop attention dispatch. Naming: "PhaseE_Beta.coop." + layer
# qualified name + ".self_attn.attn". This is what torch profiler captures
# as user_annotation.
#
# The INNER B' NVTX `phase_e_beta_kernel` (phase_e_kernel.py:3158) uses raw
# torch.cuda.nvtx.range_push which torch profiler does NOT capture as
# user_annotation — only record_function markers appear there. So we anchor
# pairing on the outer range and filter to cudaLaunchKernelExC (cooperative
# launch API) inside it.
OUTER_NVTX_PREFIX = "PhaseE_Beta.coop."
# Cooperative launches go through cudaLaunchKernelExC specifically; the
# generic cudaLaunchKernel inside the outer range is the prep zero_ kernel.
COOP_LAUNCH_NAME = "cudaLaunchKernelExC"


def load_trace_events(path: Path) -> list[dict]:
    """Stream-load trace events from a .pt.trace.json.gz file."""
    print(f"[load] reading {path} ({path.stat().st_size / 1e6:.1f} MB gzipped)", flush=True)
    with gzip.open(path, "rt") as f:
        data = json.load(f)
    events = data.get("traceEvents", [])
    print(f"[load] {len(events):,} events", flush=True)
    return events


def extract_beta_coop_calls(events: list[dict]) -> list[dict]:
    """Pair each outer PhaseE_Beta.coop.* NVTX range with the β-coop launch.

    The B' inner NVTX `phase_e_beta_kernel` uses raw torch.cuda.nvtx which
    torch profiler does not capture; we anchor on the outer record_function
    range and filter to cudaLaunchKernelExC (cooperative launch API).

    Returns one dict per β-coop call with timing columns. The outer NVTX
    captures more than just the launch (CPU prep + KV update + post-launch
    bookkeeping), so launch_api_dur and queue_gap are reported separately
    from nvtx_dur for clean attribution.
    """
    nvtx_ranges: list[tuple[float, float, str]] = []  # (ts, dur, name)
    launch_ops: list[tuple[float, float, int, str]] = []  # (ts, dur, correlation, name)
    gpu_kernels: dict[int, tuple[float, float, str]] = {}  # correlation -> (ts, dur, name)

    for e in events:
        cat = e.get("cat", "")
        name = e.get("name", "")
        if cat == "user_annotation" and name.startswith(OUTER_NVTX_PREFIX):
            ts = e.get("ts")
            dur = e.get("dur")
            if ts is not None and dur is not None:
                nvtx_ranges.append((float(ts), float(dur), name))
        elif cat == "cuda_runtime" and name.startswith(LAUNCH_OP_PREFIXES):
            ts = e.get("ts")
            dur = e.get("dur")
            corr = (e.get("args") or {}).get("correlation")
            if ts is not None and dur is not None and corr is not None:
                launch_ops.append((float(ts), float(dur), int(corr), name))
        elif cat == "kernel":
            ts = e.get("ts")
            dur = e.get("dur")
            corr = (e.get("args") or {}).get("correlation")
            if ts is not None and dur is not None and corr is not None:
                gpu_kernels[int(corr)] = (float(ts), float(dur), name)

    print(
        f"[extract] {len(nvtx_ranges)} outer `{OUTER_NVTX_PREFIX}*` NVTX ranges, "
        f"{len(launch_ops)} launch ops, {len(gpu_kernels)} GPU kernels",
        flush=True,
    )

    if not nvtx_ranges:
        print(f"[extract] WARN: no NVTX ranges matched prefix `{OUTER_NVTX_PREFIX}`",
              flush=True)
        return []

    nvtx_ranges.sort(key=lambda r: r[0])
    launch_ops.sort(key=lambda l: l[0])

    calls: list[dict] = []
    j = 0  # pointer into launch_ops
    skipped_no_launch = 0
    skipped_no_kernel = 0
    for (n_ts, n_dur, n_name) in nvtx_ranges:
        n_end = n_ts + n_dur
        while j < len(launch_ops) and launch_ops[j][0] + launch_ops[j][1] < n_ts:
            j += 1
        # Find the cudaLaunchKernelExC inside this outer range (the β-coop
        # cooperative launch). Other cudaLaunchKernel events inside the
        # range are CPU prep kernels (zero_, fill_, etc.) — skip them.
        k = j
        launch_match = None
        while k < len(launch_ops) and launch_ops[k][0] < n_end:
            if launch_ops[k][0] >= n_ts and launch_ops[k][3] == COOP_LAUNCH_NAME:
                launch_match = launch_ops[k]
                break
            k += 1
        if launch_match is None:
            skipped_no_launch += 1
            continue
        l_ts, l_dur, corr, l_name = launch_match
        gpu = gpu_kernels.get(corr)
        if gpu is None:
            skipped_no_kernel += 1
            continue
        g_ts, g_dur, g_name = gpu
        calls.append({
            "nvtx_name": n_name,
            "nvtx_ts_us": n_ts,
            "nvtx_dur_us": n_dur,
            "launch_api_ts_us": l_ts,
            "launch_api_dur_us": l_dur,
            "gpu_kernel_name": g_name,
            "gpu_kernel_ts_us": g_ts,
            "kernel_dur_us": g_dur,
            "queue_gap_us": g_ts - (l_ts + l_dur),
            "correlation": corr,
        })

    print(
        f"[extract] paired {len(calls)} β-coop calls "
        f"(skipped {skipped_no_launch} no-launch, {skipped_no_kernel} no-kernel)",
        flush=True,
    )
    return calls


def stats(arr: np.ndarray) -> dict:
    """Return median / p10 / p90 / mean / std as floats (μs)."""
    if arr.size == 0:
        return {"n": 0, "median": float("nan"), "p10": float("nan"),
                "p90": float("nan"), "mean": float("nan"), "std": float("nan")}
    return {
        "n": int(arr.size),
        "median": float(np.median(arr)),
        "p10": float(np.percentile(arr, 10)),
        "p90": float(np.percentile(arr, 90)),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
    }


def fmt_stats(name: str, s: dict, unit: str = "us") -> str:
    if s["n"] == 0:
        return f"  {name:30s}  n=0 (no data)"
    return (
        f"  {name:30s}  n={s['n']:5d}  "
        f"median={s['median']:8.2f} {unit}  "
        f"p10={s['p10']:8.2f}  p90={s['p90']:8.2f}  "
        f"mean={s['mean']:8.2f}  std={s['std']:7.2f}"
    )


def reduce_region_npy(
    npy_path: Path,
    host_launch_wall_us: float,
    wo_split: int = 8,
    num_kv_heads: int = 4,
    num_seqs: int = 1,
    slice_ctas: int = 8,
    num_k_tiles: int = 8,
) -> "pd.DataFrame":
    """Wrap region_timing.reduce_region_timings with our captured params."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from vllm.v1.attention.backends.cute_paged.region_timing import (
        reduce_region_timings,
    )
    buf = np.load(npy_path)
    print(f"[regions] {npy_path.name}: shape={buf.shape} dtype={buf.dtype}", flush=True)
    df = reduce_region_timings(
        buf,
        slice_ctas=slice_ctas,
        num_k_tiles=num_k_tiles,
        num_seqs=num_seqs,
        tick_source="globaltimer",
        wo_split=wo_split,
        num_kv_heads=num_kv_heads,
        host_launch_wall_us=host_launch_wall_us,
    )
    return df


def analyze_leg(out_dir: Path, label: str, timed_n: int) -> dict:
    """Compute per-call stats + region breakdown for one leg."""
    print(f"\n{'='*64}")
    print(f"=== Leg: {label}")
    print(f"{'='*64}", flush=True)

    trace_path = out_dir / f"{label}.pt.trace.json.gz"
    walls_path = out_dir / f"{label}_host_launch_walls.npy"
    regions_path = out_dir / f"{label}_region_timings.npy"

    result: dict = {"label": label}

    # Per-call breakdown from torch trace.
    if trace_path.exists():
        events = load_trace_events(trace_path)
        calls = extract_beta_coop_calls(events)
        del events  # free memory before next leg
        result["n_paired_calls"] = len(calls)
        if calls:
            arr_launch = np.array([c["launch_api_dur_us"] for c in calls])
            arr_queue = np.array([c["queue_gap_us"] for c in calls])
            arr_kernel = np.array([c["kernel_dur_us"] for c in calls])
            arr_nvtx = np.array([c["nvtx_dur_us"] for c in calls])
            result["launch_api_dur"] = stats(arr_launch)
            result["queue_gap"] = stats(arr_queue)
            result["kernel_dur"] = stats(arr_kernel)
            result["nvtx_dur"] = stats(arr_nvtx)
            # Per-call CSV for committed evidence.
            csv_path = out_dir / f"{label}_per_call.csv"
            import csv
            with open(csv_path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow([
                    "nvtx_name", "nvtx_ts_us", "nvtx_dur_us", "launch_api_ts_us",
                    "launch_api_dur_us", "gpu_kernel_ts_us", "kernel_dur_us",
                    "queue_gap_us", "gpu_kernel_name", "correlation",
                ])
                for c in calls:
                    w.writerow([
                        c["nvtx_name"],
                        f"{c['nvtx_ts_us']:.3f}", f"{c['nvtx_dur_us']:.3f}",
                        f"{c['launch_api_ts_us']:.3f}", f"{c['launch_api_dur_us']:.3f}",
                        f"{c['gpu_kernel_ts_us']:.3f}", f"{c['kernel_dur_us']:.3f}",
                        f"{c['queue_gap_us']:.3f}", c["gpu_kernel_name"], c["correlation"],
                    ])
            print(f"[csv] wrote {csv_path} ({len(calls)} rows)", flush=True)
    else:
        print(f"[trace] WARN: {trace_path} missing", flush=True)

    # Host CUDA-event walls (timing_on only).
    if walls_path.exists():
        walls = np.load(walls_path)
        print(f"[walls] {walls_path.name}: n={walls.size}", flush=True)
        # Take last timed_n+small_buffer entries as timed-window subset.
        # (Warmup + post-profile drain reqs sit ahead of this slice.)
        if walls.size > timed_n:
            timed_walls = walls[-(timed_n):]
        else:
            timed_walls = walls
            print(f"[walls] WARN: only {walls.size} walls, expected ≥{timed_n} "
                  "(timed-window slice may include warmup)", flush=True)
        result["host_event_wall_all"] = stats(walls)
        result["host_event_wall_timed"] = stats(timed_walls)
        # Median timed wall used as host_launch_wall_us for region reducer.
        median_wall = float(np.median(timed_walls))
    else:
        median_wall = float("nan")
        if label.startswith("timing_on"):
            print(f"[walls] WARN: {walls_path} missing for timing_on leg", flush=True)

    # Region breakdown via reducer (timing_on only).
    if regions_path.exists():
        df = reduce_region_npy(regions_path, median_wall)
        df_csv = out_dir / f"{label}_region_breakdown.csv"
        df.to_csv(df_csv, index=False)
        print(f"[regions] wrote {df_csv}", flush=True)
        # Sum work-region medians (B' wall_minus_regions accounting).
        work_mask = ~df["region_id"].isin({4, 11})  # WAIT_NOT_WORK
        sum_work_us = float((df.loc[work_mask, "median_ticks"] / 1000.0).sum())
        result["region_sum_us"] = sum_work_us
        result["wall_minus_regions_us"] = median_wall - sum_work_us
        # Per-region table for summary.md.
        result["regions_df_csv"] = str(df_csv.relative_to(out_dir.parent.parent.parent.parent))
    return result


def write_summary(out_dir: Path, results: list[dict], timed_n: int, commit: str) -> None:
    """Emit summary.md aggregating both legs."""
    md = out_dir / "summary.md"
    lines: list[str] = []
    lines.append(f"# β-coop launch-overhead profile (2026-05-16)")
    lines.append("")
    lines.append(f"**Commit:** `{commit}` (PR #17 merged)  ")
    lines.append(f"**Image:** `nvllm:gb10-bprime` sha256:`4fccbd915044a8f5f7db8268b0ec645323eb3d7063fd66233e64b1882e7c2539` (clean rebuild 2026-05-16)  ")
    lines.append(f"**Model:** `ig1/Qwen3.5-27B-NVFP4`  ")
    lines.append(f"**Hardware:** DGX Spark (GB10, SM120/121, 48 SMs, 128 GB unified)  ")
    lines.append(f"**Backend:** CuTe Paged, FP8 E4M3 KV, PIECEWISE cudagraphs  ")
    lines.append(f"**Config:** `CUTE_PHASE_E_FUSION=1 CUTE_PHASE_E_LAYERS=3,7 CUTE_WO_SPLIT=8`, `max_num_seqs=1`, decode `max_tokens=64`  ")
    lines.append(f"**Profiler:** torch profiler (vLLM V1 EngineCore — nsys cannot follow spawn boundary per `feedback_vllm_profiling`)  ")
    lines.append(f"**Window:** 15-req warmup outside profiler, then profiler-active window auto-bounded by `active_iterations=200` (~3 decode reqs captured per leg). `--timed 100` for `timing_on` (host launch walls need samples), `--timed 20` for `timing_off` (control needs only profiler-window samples). `record_shapes=False with_stack=False`.")
    lines.append("")
    lines.append("## Purpose")
    lines.append("")
    lines.append("Convert the B' single-call diagnostic (PR #17, ~23 ms `wall_minus_regions_us` per "
                 "β-coop call) into a steady-state AGENTS.md §4 perf claim, AND identify whether the "
                 "unaccounted budget is CPU launch overhead, GPU scheduling, or in-kernel time.")
    lines.append("")
    lines.append("Two legs bound the instrumentation tax: `timing_on` has B' R-region writes + host "
                 "CUDA-event recording (`CUTE_BETA_REGION_TIMING=1`); `timing_off` is the production "
                 "control (`CUTE_BETA_REGION_TIMING=0`, `--timed 20` — only profiler-window samples "
                 "needed since walls aren't captured).")
    lines.append("")
    lines.append("## Headline finding — the 24 ms is IN-KERNEL, not launch overhead")
    lines.append("")
    lines.append("Torch profiler kernel events show `kernel_dur ≈ host_event_wall`. The CPU "
                 "`cudaLaunchKernelExC` API takes ~7 μs; the OUTER NVTX wrapping the launch is ~400 μs "
                 "(includes KV update + Python bookkeeping). So the 24 ms `wall_minus_regions_us` budget "
                 "is **in-kernel time that the R-regions don't cover** — either CTA-concurrency-amortized "
                 "regions (per-CTA medians summed underweight wall-clock), or uninstrumented kernel work "
                 "(barrier waits, register/SMEM spills, scheduling gaps between R-region exits and entries).")
    lines.append("")
    lines.append("This **revises the B' memory framing** of \"87% cooperative-launch/grid/tail overhead\" "
                 "— the cost lever is inside the kernel, not the launch path. Persistent kernel / smaller "
                 "cooperative grid / batched launches will not help; the next-bet candidates are R-region "
                 "coverage expansion (instrument the gaps) and an SMEM-spill / register-pressure audit.")
    lines.append("")
    lines.append("## Per-call breakdown (median across timed window)")
    lines.append("")
    lines.append("| Metric | timing_on | timing_off | Δ (tax) |")
    lines.append("|---|---:|---:|---:|")
    on = next((r for r in results if r["label"] == "timing_on"), None)
    off = next((r for r in results if r["label"] == "timing_off"), None)
    def cell(r: dict | None, key: str, sub: str = "median") -> str:
        if r is None or key not in r:
            return "—"
        v = r[key]
        if isinstance(v, dict):
            v = v.get(sub, float("nan"))
        if v != v:  # NaN
            return "—"
        return f"{v:.2f}"
    def delta(on_r, off_r, key: str, sub: str = "median") -> str:
        if on_r is None or off_r is None or key not in on_r or key not in off_r:
            return "—"
        a = on_r[key].get(sub) if isinstance(on_r[key], dict) else on_r[key]
        b = off_r[key].get(sub) if isinstance(off_r[key], dict) else off_r[key]
        if a != a or b != b:
            return "—"
        return f"{a - b:+.2f}"
    for key, label in [
        ("launch_api_dur", "CPU `cudaLaunchKernelExC` API duration"),
        ("kernel_dur", "GPU β-coop kernel duration (correlation-paired)"),
        ("nvtx_dur", "Outer `PhaseE_Beta.coop.*` NVTX range (CPU wall)"),
        ("host_event_wall_timed", "Host CUDA-event wall (B' instrumentation)"),
    ]:
        lines.append(f"| {label} | {cell(on, key)} μs | {cell(off, key)} μs | {delta(on, off, key)} μs |")
    lines.append("")
    lines.append("**Note on `queue_gap`:** the per-call CSV reports `queue_gap_us` (CPU launch-return "
                 "→ GPU kernel-start), but at our workload heavy β-coop kernels (~27 ms each) saturate "
                 "the device stream so this measures **device backlog**, not scheduling latency. Excluded "
                 "from the summary table to avoid misinterpretation.")
    lines.append("")
    if on:
        lines.append("**Region breakdown (timing_on leg only — `region_timings.npy` is a one-call snapshot, walls are timed-window medians):**")
        lines.append("")
        lines.append("| Metric | Value |")
        lines.append("|---|---:|")
        if "region_sum_us" in on:
            lines.append(f"| Sum of per-CTA work-region medians | {on['region_sum_us']:.2f} μs |")
        if "wall_minus_regions_us" in on:
            lines.append(f"| `wall_minus_regions_us` (B' diagnostic) | **{on['wall_minus_regions_us']:.2f} μs** |")
        lines.append("")
    lines.append("## Instrumentation tax — verdict")
    lines.append("")
    # Compute the verdict line dynamically from the data.
    if on and off and "kernel_dur" in on and "kernel_dur" in off:
        on_k = on["kernel_dur"]["median"]
        off_k = off["kernel_dur"]["median"]
        delta_k = on_k - off_k
        pct = (delta_k / off_k) * 100.0 if off_k else float("nan")
        std_floor = max(on["kernel_dur"]["std"], off["kernel_dur"]["std"])
        verdict = "NEGLIGIBLE" if abs(delta_k) < std_floor else "MATERIAL"
        lines.append(f"`kernel_dur` Δ = **{delta_k:+.0f} μs ({pct:+.2f}%)** on timing_on vs timing_off; "
                     f"both legs have std ≈ {std_floor:.0f} μs so this is **{verdict}**.")
        if verdict == "NEGLIGIBLE":
            lines.append("")
            lines.append("The B' R-region writes + host CUDA-event recording do **not** perturb the "
                         "β-coop kernel. The `wall_minus_regions_us` figure reflects production cost, "
                         "not measurement artifact, and is safe to use as the perf-claim denominator.")
        else:
            lines.append("")
            lines.append("The instrumentation IS perturbing the kernel. The `wall_minus_regions_us` "
                         "figure includes measurement overhead; treat as upper bound on real "
                         "unaccounted budget.")
    else:
        lines.append("Insufficient data — both legs must be present.")
    lines.append("")
    lines.append("## How to reproduce")
    lines.append("")
    lines.append("The two legs were captured separately to keep wall-clock manageable (β-coop runs "
                 "at ~2.5 tok/s, so each timed window is rate-limited):")
    lines.append("")
    lines.append("```bash")
    lines.append("# Leg 1 — timing_on (B' R-region writes + host CUDA-event recording).")
    lines.append("# --timed 100 captures enough host launch walls for stable median.")
    lines.append("bash docs/research/launch_overhead_profile/capture.sh smoke")
    lines.append("")
    lines.append("# Leg 2 — timing_off (CUTE_BETA_REGION_TIMING=0 production control).")
    lines.append("# --timed 20 since only profiler-window samples drive kernel_dur comparison.")
    lines.append("bash docs/research/launch_overhead_profile/capture.sh control")
    lines.append("")
    lines.append("# Analysis — pairs outer PhaseE_Beta.coop NVTX with cudaLaunchKernelExC,")
    lines.append("# reduces B' region_timings.npy + host_launch_walls.npy, emits summary.md.")
    lines.append(".venv/bin/python docs/research/launch_overhead_profile/analyze.py \\")
    lines.append(f"    --out-dir {out_dir} \\")
    lines.append(f"    --timed-n {timed_n}")
    lines.append("```")
    lines.append("")
    lines.append("For a single bundled run (~70 min): "
                 "`bash docs/research/launch_overhead_profile/capture.sh full`.")
    lines.append("")
    lines.append("## Caveats")
    lines.append("")
    lines.append("- **Torch profiler fallback called out.** vLLM V1 EngineCore is spawned; "
                 "nsys cannot follow without an explicit `cudaProfilerApi` hook (not installed). "
                 "Torch profiler captures CPU + GPU events with correlation IDs, which gives us "
                 "the same per-call breakdown nsys would, but without grid-residency / SM-level detail.")
    lines.append("- **`region_timings.npy` is one snapshot.** B' sentinel-file dump is one-shot per "
                 "touch. host launch walls are the full per-call queue (timed-window slice taken in "
                 "analyzer); region buf is the last β-coop call before the drain trigger.")
    lines.append("- **wall_minus_regions_us interpretation.** Per the headline, the unaccounted budget "
                 "is **in-kernel time** (kernel_dur ≈ host_event_wall), not pre-launch CPU or "
                 "post-launch tail. Candidate causes: (a) per-CTA medians underweight wall-clock when "
                 "CTAs run regions concurrently; (b) **R-region entry/exit likely bracket ONE "
                 "iteration of inner K-tile loops, not the full per-call sum** — sum-of-medians "
                 "undercounts by the loop trip count (Phase 3 R7-R9 in particular run inside the "
                 "FC1/FC2 K-tile loop); (c) uninstrumented kernel work between R-region exits and "
                 "the next R-region entry (barrier waits, register/SMEM spills). The B' "
                 "instrumentation does not currently distinguish (a)/(b)/(c) — that's the next "
                 "investigation.")
    lines.append("- **β-coop solo-only at coop launch.** `max_num_seqs=1`; `num_seqs=2` is the next "
                 "blocked target (`project_num_seqs_2_target`) but out of scope for this trace.")
    lines.append("")
    md.write_text("\n".join(lines))
    print(f"[summary] wrote {md}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--timed-n", type=int, default=200)
    ap.add_argument("--commit", default=None,
                    help="Override commit hash (default: git rev-parse --short HEAD)")
    args = ap.parse_args()

    if args.commit is None:
        import subprocess
        args.commit = subprocess.check_output(
            ["git", "-C", str(args.out_dir.parent.parent.parent.parent),
             "rev-parse", "--short", "HEAD"]
        ).decode().strip()

    results = []
    for leg in ("timing_on", "timing_off"):
        trace_path = args.out_dir / f"{leg}.pt.trace.json.gz"
        if not trace_path.exists():
            print(f"[skip] {leg}: no trace file", flush=True)
            continue
        results.append(analyze_leg(args.out_dir, leg, args.timed_n))

    if not results:
        print("[err] no legs analyzed", file=sys.stderr)
        sys.exit(1)

    # Print summary table to stdout.
    print(f"\n{'='*64}")
    print("=== Summary")
    print(f"{'='*64}")
    for r in results:
        print(f"\nLeg: {r['label']}  (n_paired_calls={r.get('n_paired_calls', 0)})")
        for key in ("launch_api_dur", "queue_gap", "kernel_dur", "nvtx_dur",
                    "host_event_wall_timed"):
            if key in r:
                print(fmt_stats(key, r[key]))
        if "wall_minus_regions_us" in r:
            print(f"  wall_minus_regions_us (median wall)   {r['wall_minus_regions_us']:.2f}")
        if "region_sum_us" in r:
            print(f"  region_sum_us (work regions)          {r['region_sum_us']:.2f}")

    write_summary(args.out_dir, results, args.timed_n, args.commit)


if __name__ == "__main__":
    main()
