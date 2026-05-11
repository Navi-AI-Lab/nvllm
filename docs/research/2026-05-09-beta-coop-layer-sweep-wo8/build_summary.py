"""Build the top-level summary.md for the β-coop layer-count sweep.

Reads each arm's gsm8k.json + verdict.json under sweep/<arm>/ and writes
sweep/summary.md with:
  1. Headline accuracy/dispatch/error table.
  2. Per-question miss table — every question index that any arm got
     non-OK on, with status + pred per arm. Prevents over-claiming from
     a single arm's accuracy delta when the spread is decode noise.
  3. Verdict framing baked from the 2026-05-10 user-validated read.

Usage:
    .venv/bin/python docs/research/2026-05-09-beta-coop-layer-sweep-wo8/build_summary.py \\
        [--sweep-dir docs/research/2026-05-09-beta-coop-layer-sweep-wo8/sweep] \\
        [--out docs/research/2026-05-09-beta-coop-layer-sweep-wo8/sweep/summary.md]
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path


def _arm_layer_count(arm: str) -> int:
    # arm names: "2L_3_7", "8L_3_31", etc. Sort by leading integer.
    n = ""
    for ch in arm:
        if ch.isdigit():
            n += ch
        else:
            break
    return int(n) if n else -1


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--sweep-dir", default=None,
                   help="Directory containing per-arm subdirs. "
                        "Defaults to the script's dirname/sweep.")
    p.add_argument("--out", default=None,
                   help="Output summary.md path. Defaults to "
                        "<sweep-dir>/summary.md.")
    args = p.parse_args()

    here = Path(__file__).resolve().parent
    sweep_dir = Path(args.sweep_dir) if args.sweep_dir else here / "sweep"
    out_path = Path(args.out) if args.out else sweep_dir / "summary.md"

    if not sweep_dir.is_dir():
        sys.stderr.write(f"sweep dir not found: {sweep_dir}\n")
        return 2

    arms = sorted(
        (d for d in sweep_dir.iterdir() if d.is_dir()),
        key=lambda d: _arm_layer_count(d.name),
    )
    if not arms:
        sys.stderr.write(f"no arms found under {sweep_dir}\n")
        return 2

    arm_data: dict[str, dict] = {}
    for arm_dir in arms:
        arm = arm_dir.name
        v = arm_dir / "verdict.json"
        g = arm_dir / "gsm8k.json"
        verdict = json.loads(v.read_text()) if v.is_file() else {}
        gsm = json.loads(g.read_text()) if g.is_file() else None
        arm_data[arm] = {"verdict": verdict, "gsm8k": gsm}

    # Per-question miss table: union of non-OK indices across arms.
    miss_idx: set[int] = set()
    for arm, d in arm_data.items():
        if d["gsm8k"] is None:
            continue
        for r in d["gsm8k"]["results"]:
            if r["status"] != "OK":
                miss_idx.add(r["i"])

    # Also keep all results indexed for table emission.
    by_arm_idx: dict[str, dict[int, dict]] = {}
    for arm, d in arm_data.items():
        if d["gsm8k"] is None:
            by_arm_idx[arm] = {}
            continue
        by_arm_idx[arm] = {r["i"]: r for r in d["gsm8k"]["results"]}

    # Find the gold for each missed index (any arm's record will have it).
    gold_for: dict[int, str] = {}
    for i in miss_idx:
        for arm in by_arm_idx:
            r = by_arm_idx[arm].get(i)
            if r is not None:
                gold_for[i] = r["expected"]
                break

    # Generation context.
    git_sha = ""
    image_id = ""
    if arm_data:
        first = next(iter(arm_data.values()))["verdict"]
        git_sha = first.get("git_sha", "")
        image_id = first.get("image_id", "")
    arms_csv = sweep_dir.parent / "arms.csv"

    lines: list[str] = []
    lines.append("# β-coop layer-count sweep under wo_split=8")
    lines.append("")
    lines.append(f"- generated: {dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"- git_sha: {git_sha}")
    lines.append(f"- image_id: {image_id}")
    lines.append(f"- arms manifest: `{arms_csv}`")
    lines.append("- region-timing buffer: DISABLED for the sweep "
                 "(plan Risk #2). Per-call β median was captured separately "
                 "in Stage 0c (5.538 ms vs ≤7 ms gate).")
    lines.append("")

    # Headline arm table.
    lines.append("## Per-arm headline")
    lines.append("")
    lines.append(
        "| arm | fusion | layers | dispatch | GSM8K | errors | "
        "wall (s) | ok |"
    )
    lines.append(
        "|---|---|---|---|---|---|---|---|"
    )
    for arm_dir in arms:
        arm = arm_dir.name
        v = arm_data[arm]["verdict"]
        g = arm_data[arm]["gsm8k"]
        wall = (
            f"{g['total_seconds']:.0f}"
            if g and "total_seconds" in g else "?"
        )
        floor = v.get("gsm8k_floor", "?")
        correct = v.get("gsm8k_correct", "?")
        errors = v.get("gsm8k_errors", "?")
        layers = v.get("phase_e_layers", "")
        dispatch = v.get("dispatch_audit", "?")
        fusion = v.get("fusion", "?")
        ok = "true" if v.get("ok") is True else "false"
        lines.append(
            f"| {arm} | {fusion} | [{layers}] | {dispatch} | "
            f"{correct}/50 (≥{floor}) | {errors} | {wall} | {ok} |"
        )
    lines.append("")

    # Per-question miss table.
    lines.append("## Per-question miss table")
    lines.append("")
    lines.append(
        "Union of every question any arm got non-OK on. Cells show the "
        "model's predicted answer (or `OK` if that arm answered correctly). "
        "This table prevents accidental over-claiming from a single arm's "
        "accuracy delta — a 49/50 vs 47/50 spread looks impressive in the "
        "headline but the miss-pattern below shows whether the spread is "
        "layer-correlated quality or just decode variance."
    )
    lines.append("")
    if not miss_idx:
        lines.append("_All arms scored 50/50 on the seed=42 sample — no "
                     "miss table to emit._")
    else:
        header = "| Q (gold) |"
        sep = "|---|"
        for arm_dir in arms:
            header += f" {arm_dir.name} |"
            sep += "---|"
        lines.append(header)
        lines.append(sep)
        for i in sorted(miss_idx):
            row = f"| Q{i} (gold={gold_for.get(i,'?')}) |"
            for arm_dir in arms:
                arm = arm_dir.name
                r = by_arm_idx[arm].get(i)
                if r is None:
                    row += " ? |"
                elif r["status"] == "OK":
                    row += " OK |"
                else:
                    pred = r.get("got") or "(empty)"
                    row += f" {r['status']} pred=`{pred}` |"
            lines.append(row)
    lines.append("")

    # Verdict framing.
    lines.append("## Verdict framing (2026-05-10)")
    lines.append("")
    lines.append(
        "- **Q0 (gold=2280, pred=2180 on 2L / 4L / 8L / 12L)** is a stable "
        "model/eval miss across the passing β configs, not a regression "
        "signal. (16L's Q0 produced a different wrong answer — see the miss "
        "table — consistent with 16L's broader quality break, not a Q0-specific "
        "signal.)"
    )
    lines.append(
        "- **Q44 and the one-off Q7 / Q21 flips** look like knife-edge "
        "decode variance among 2L/4L/8L/12L, not layer-count-correlated "
        "quality."
    )
    lines.append(
        "- **The 47–49/50 spread among 2L/4L/8L/12L** should be treated as "
        "noise around a passing band, not evidence that 8L is better "
        "quality."
    )
    lines.append(
        "- **16L (36/50, 0 errors) is NOT noise.** A 12-question drop "
        "on the same seed=42 GSM8K-50 sample is real signal — 16L is "
        "**quality-blocked under Stage 1c** and excluded from the "
        "dev-baseline pick."
    )
    lines.append(
        "- **Stage 1c verdict:** 2L, 4L, 8L, 12L pass (≥45/50 with 0 "
        "errors). 16L fails."
    )
    lines.append(
        "- **Stage 2a dev-baseline pick:** **12L_3_47** "
        "(`CUTE_PHASE_E_LAYERS=3,7,11,15,19,23,27,31,35,39,43,47`) — "
        "highest β-capable full-attention layer count that passes "
        "Stage 1c. Per the plan, perf comparison among passing arms is "
        "deferred to the long-soak loop; this loop only proves "
        "correctness across the layer ladder."
    )
    lines.append("")
    lines.append("### Follow-up: bisect the upper-quartet regression")
    lines.append("")
    lines.append(
        "16L = 12L + {51, 55, 59, 63} regressed quality. We are NOT "
        "bisecting the offending upper-quartet layer in this loop — "
        "scope was \"sweep then pick\". A future loop should test "
        "13L (12L + 51), 14L (12L + 51, 55), 15L (12L + 51, 55, 59), "
        "or rotate which single upper layer is added to 12L, to "
        "isolate whether the regression is one bad layer or a "
        "cumulative effect."
    )
    lines.append("")

    # Per-arm artifacts.
    lines.append("## Per-arm artifacts")
    lines.append("")
    for arm_dir in arms:
        arm = arm_dir.name
        lines.append(
            f"- [{arm}/summary.md]({arm}/summary.md), "
            f"[{arm}/verdict.json]({arm}/verdict.json), "
            f"[{arm}/gsm8k.json]({arm}/gsm8k.json), "
            f"[{arm}/dispatch_audit.json]({arm}/dispatch_audit.json)"
        )
    lines.append("")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines))
    print(f"[summary] wrote {out_path} ({sum(1 for _ in lines)} lines, "
          f"{len(arms)} arms, {len(miss_idx)} miss indices)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
