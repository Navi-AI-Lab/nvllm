#!/usr/bin/env python3
"""Aggregate soak evidence into summary tables for summary.md.

Expected input layout (per plan.md):
    benchmarks/nvllm/traces/wo_split_prod_soak/<date>-soak/
      wo1/  wo2/  wo4/  wo8/
        primary/
          gsm8k.json
          run01/  ... run05/
            sharegpt.json    # {"turns": [{"wall_s", "token_deltas_ms"...}]}
            longdecode.json  # {"turns": [{"wall_s", "token_deltas_ms"...}]}
          concurrent/
            summary.json     # {"total_wall_s", per-request stats}

Per-replay JSONs come from runner.sh (driven by a small Python helper that
issues /v1/completions requests and records per-token timestamps).

Usage:
    .venv/bin/python parse_results.py \
        --evidence-dir benchmarks/nvllm/traces/wo_split_prod_soak/2026-05-04-soak \
        --out         benchmarks/nvllm/traces/wo_split_prod_soak/2026-05-04-soak/summary.md
"""

import argparse
import json
import sys
from pathlib import Path
from statistics import mean, stdev


# --- helpers ----------------------------------------------------------------

def percentile(data: list[float], p: float) -> float:
    if not data:
        return 0.0
    sd = sorted(data)
    k = (len(sd) - 1) * p
    f = int(k)
    c = min(f + 1, len(sd) - 1)
    if f == c:
        return sd[f]
    return sd[f] + (sd[c] - sd[f]) * (k - f)


def load_replay_jsons(arm_dir: Path, phase: str) -> list[dict]:
    primary = arm_dir / "primary"
    if not primary.exists():
        return []
    out = []
    for run in sorted(primary.glob("run*")):
        jf = run / f"{phase}.json"
        if jf.exists():
            out.append(json.loads(jf.read_text()))
    return out


def all_token_deltas(replays: list[dict]) -> list[float]:
    out: list[float] = []
    for r in replays:
        for turn in r.get("turns", []):
            out.extend(turn.get("token_deltas_ms", []))
    return out


def per_replay_p95(replays: list[dict]) -> list[float]:
    out: list[float] = []
    for r in replays:
        deltas: list[float] = []
        for turn in r.get("turns", []):
            deltas.extend(turn.get("token_deltas_ms", []))
        if deltas:
            out.append(percentile(deltas, 0.95))
    return out


def replay_walls(replays: list[dict]) -> list[float]:
    return [
        sum(t.get("wall_s", 0.0) for t in r.get("turns", []))
        for r in replays
    ]


# --- per-arm aggregation ----------------------------------------------------

def arm_stats(arm_dir: Path) -> dict:
    sharegpt = load_replay_jsons(arm_dir, "sharegpt")
    longdecode = load_replay_jsons(arm_dir, "longdecode")

    walls = replay_walls(sharegpt)
    sg_deltas = all_token_deltas(sharegpt)
    ld_deltas = all_token_deltas(longdecode)
    sg_replay_p95 = per_replay_p95(sharegpt)

    out: dict = {
        "arm": arm_dir.name,
        "n_replays": len(sharegpt),
        "gsm8k_score": None,
        "concurrent_wall_s": None,
        "sharegpt_wall_mean_s":   mean(walls) if walls else 0.0,
        "sharegpt_wall_stddev_s": stdev(walls) if len(walls) > 1 else 0.0,
        "sharegpt_tpot_p50_ms":   percentile(sg_deltas, 0.50),
        "sharegpt_tpot_p95_ms":   percentile(sg_deltas, 0.95),
        "sharegpt_tpot_p99_ms":   percentile(sg_deltas, 0.99),
        # Stddev across per-replay p95 values; this is what the plan's
        # "TPOT p95 not worse than baseline (within 1× baseline stddev)" means.
        "sharegpt_tpot_p95_replay_stddev_ms":
            stdev(sg_replay_p95) if len(sg_replay_p95) > 1 else 0.0,
        "longdecode_tpot_p50_ms": percentile(ld_deltas, 0.50),
        "longdecode_tpot_p95_ms": percentile(ld_deltas, 0.95),
        "longdecode_tpot_p99_ms": percentile(ld_deltas, 0.99),
    }
    gsm8k_f = arm_dir / "primary" / "gsm8k.json"
    if gsm8k_f.exists():
        # gsm8k_eval_50.py writes 'correct' (count out of n), not 'score'.
        out["gsm8k_score"] = json.loads(gsm8k_f.read_text()).get("correct")
    cf = arm_dir / "primary" / "concurrent" / "summary.json"
    if cf.exists():
        out["concurrent_wall_s"] = json.loads(cf.read_text()).get(
            "total_wall_s")
    return out


# --- per-arm verdict per plan.md decision criteria --------------------------

def verdict(arm: dict, baseline: dict, wall_threshold_pct: float) -> str:
    g = arm.get("gsm8k_score")
    if g is None:
        return "no gsm8k score"
    if g < 30:
        return f"investigate (gsm8k {g} < 30 floor)"
    bg = baseline.get("gsm8k_score")
    if bg is not None and (bg - g) > 2:
        return f"investigate (gsm8k drop {bg - g} > 2 questions)"

    bw = baseline.get("sharegpt_wall_mean_s", 0.0)
    aw = arm.get("sharegpt_wall_mean_s", 0.0)
    wall_pct = (bw - aw) / bw * 100 if bw > 0 else 0.0
    bp95 = baseline.get("sharegpt_tpot_p95_ms", 0.0)
    ap95 = arm.get("sharegpt_tpot_p95_ms", 0.0)
    p95_noise = baseline.get("sharegpt_tpot_p95_replay_stddev_ms", 0.0)

    bw_stddev = baseline.get("sharegpt_wall_stddev_s", 0.0)
    wall_noise_pct = (bw_stddev / bw * 100) if bw > 0 else 0.0
    regression_floor = max(2.0, wall_noise_pct)
    if wall_pct < -regression_floor:
        return (f"investigate (wall regression {wall_pct:.1f}% "
                f"exceeds baseline noise floor {regression_floor:.1f}%)")
    if (wall_pct >= wall_threshold_pct
            and ap95 <= bp95 + p95_noise):
        return f"default candidate (wall +{wall_pct:.1f}%, tpot p95 ok)"
    return (f"keep opt-in (wall {wall_pct:+.1f}%, "
            f"tpot p95 {ap95 - bp95:+.2f} ms vs baseline noise {p95_noise:.2f})")


# --- markdown rendering -----------------------------------------------------

def render_md(arms: list[dict], baseline_arm: str,
              wall_threshold_pct: float,
              expected_replays: int) -> str:
    by_arm = {a["arm"]: a for a in arms}
    baseline = by_arm.get(baseline_arm, {})
    lines: list[str] = []

    missing = [a for a in arms if a["n_replays"] < expected_replays]
    if missing:
        lines.append(f"## ⚠ Missing replays (expected {expected_replays})")
        lines.append("")
        for a in missing:
            lines.append(
                f"- **{a['arm']}**: {a['n_replays']}/{expected_replays} "
                "replays — verdict may be unreliable"
            )
        lines.append("")

    lines.append("## Per-arm summary")
    lines.append("")
    lines.append("| arm | gsm8k | replays | wall mean (s) | wall stddev | tpot p50 (ms) | tpot p95 (ms) | tpot p99 (ms) | longdecode p95 |")
    lines.append("|-----|-------|---------|---------------|-------------|---------------|---------------|---------------|----------------|")
    for a in arms:
        score = "—" if a["gsm8k_score"] is None else f"{a['gsm8k_score']}/50"
        lines.append(
            f"| {a['arm']} | {score} | {a['n_replays']} | "
            f"{a['sharegpt_wall_mean_s']:.2f} | "
            f"{a['sharegpt_wall_stddev_s']:.2f} | "
            f"{a['sharegpt_tpot_p50_ms']:.2f} | "
            f"{a['sharegpt_tpot_p95_ms']:.2f} | "
            f"{a['sharegpt_tpot_p99_ms']:.2f} | "
            f"{a['longdecode_tpot_p95_ms']:.2f} |"
        )
    lines.append("")

    if baseline:
        lines.append(f"## Pairwise vs baseline ({baseline_arm})")
        lines.append("")
        lines.append("| arm | wall % change | tpot p95 Δ (ms) | gsm8k Δ |")
        lines.append("|-----|---------------|-----------------|---------|")
        for a in arms:
            if a["arm"] == baseline_arm:
                continue
            bw = baseline["sharegpt_wall_mean_s"]
            aw = a["sharegpt_wall_mean_s"]
            wall_pct = (bw - aw) / bw * 100 if bw > 0 else 0.0
            tpot_delta = (a["sharegpt_tpot_p95_ms"]
                          - baseline["sharegpt_tpot_p95_ms"])
            ag = a.get("gsm8k_score")
            bg = baseline.get("gsm8k_score")
            gd = "—" if ag is None or bg is None else f"{ag - bg:+d}"
            lines.append(
                f"| {a['arm']} | {wall_pct:+.1f}% | "
                f"{tpot_delta:+.2f} | {gd} |"
            )
        lines.append("")

    lines.append("## Verdicts")
    lines.append("")
    lines.append(f"_Wall threshold: {wall_threshold_pct:.1f}% _")
    lines.append("")
    for a in arms:
        if a["arm"] == baseline_arm:
            lines.append(f"- **{a['arm']}**: baseline")
        else:
            lines.append(
                f"- **{a['arm']}**: "
                f"{verdict(a, baseline, wall_threshold_pct)}"
            )
    lines.append("")
    return "\n".join(lines)


# --- main -------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--evidence-dir", type=Path, required=True,
                    help="benchmarks/nvllm/traces/wo_split_prod_soak/<date>-soak")
    ap.add_argument("--baseline-arm", default="wo1",
                    help="arm to compare others against (default: wo1)")
    ap.add_argument("--wall-threshold-pct", type=float, default=5.0,
                    help="default-candidate wall improvement threshold")
    ap.add_argument("--expected-replays", type=int, default=5,
                    help="warn if any arm has fewer replays than this")
    ap.add_argument("--out", type=Path, default=None,
                    help="write markdown here (default: stdout)")
    args = ap.parse_args()

    if not args.evidence_dir.exists():
        print(f"error: evidence dir not found: {args.evidence_dir}",
              file=sys.stderr)
        return 1

    # Sort numerically (wo1, wo2, wo4, wo8) — lexicographic puts wo10 before wo2.
    arm_dirs = sorted(
        (d for d in args.evidence_dir.iterdir()
         if d.is_dir() and d.name.startswith("wo")),
        key=lambda d: int(d.name[2:]),
    )
    if not arm_dirs:
        print(f"error: no wo* arm dirs in {args.evidence_dir}",
              file=sys.stderr)
        return 1
    arms = [arm_stats(d) for d in arm_dirs]

    md = render_md(arms, args.baseline_arm, args.wall_threshold_pct,
                   args.expected_replays)

    if args.out:
        args.out.write_text(md)
        print(f"wrote {args.out}", file=sys.stderr)
    else:
        sys.stdout.write(md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
