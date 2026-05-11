"""Build the Stage 2b survival-soak summary.md for the 12L_3_47 arm.

Reads soak/verdict.json + soak/run<N>/gsm8k.json and emits soak/summary.md
with:
  1. Per-run headline table (correct/errors/wall/pass).
  2. Per-question miss table — every index any run got non-OK on, with
     status + pred per run. Same anti-overclaim discipline as
     build_summary.py: a 50/50 run inside a soak where one other run
     drops 5 questions tells you stability, not quality.
  3. Verdict framing baked from Gate 2b spec (all runs >= floor, 0
     errors, container alive, no docker.log corruption hits).

Usage:
    .venv/bin/python docs/research/2026-05-09-beta-coop-layer-sweep-wo8/build_soak_summary.py \\
        [--soak-dir docs/research/2026-05-09-beta-coop-layer-sweep-wo8/soak] \\
        [--out docs/research/2026-05-09-beta-coop-layer-sweep-wo8/soak/summary.md]
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import sys
from pathlib import Path


RUN_DIR_RE = re.compile(r"^run(\d+)$")


def _run_index(name: str) -> int:
    m = RUN_DIR_RE.match(name)
    return int(m.group(1)) if m else -1


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--soak-dir", default=None)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    here = Path(__file__).resolve().parent
    soak_dir = Path(args.soak_dir) if args.soak_dir else here / "soak"
    out_path = Path(args.out) if args.out else soak_dir / "summary.md"

    if not soak_dir.is_dir():
        sys.stderr.write(f"soak dir not found: {soak_dir}\n")
        return 2

    verdict_path = soak_dir / "verdict.json"
    if not verdict_path.is_file():
        sys.stderr.write(f"verdict.json missing at {verdict_path}\n")
        return 2
    verdict = json.loads(verdict_path.read_text())

    run_dirs = sorted(
        (d for d in soak_dir.iterdir()
         if d.is_dir() and RUN_DIR_RE.match(d.name)),
        key=lambda d: _run_index(d.name),
    )
    if not run_dirs:
        sys.stderr.write(f"no run<N> subdirs under {soak_dir}\n")
        return 2

    # Read each run's gsm8k.json. Value is None when the run never wrote
    # an output file (e.g. it failed before scoring).
    by_run: dict[str, dict | None] = {}
    for d in run_dirs:
        g = d / "gsm8k.json"
        by_run[d.name] = json.loads(g.read_text()) if g.is_file() else None

    # Per-question miss-index union across runs.
    miss_idx: set[int] = set()
    for run_name, data in by_run.items():
        if data is None:
            continue
        for r in data["results"]:
            if r["status"] != "OK":
                miss_idx.add(r["i"])

    by_run_idx: dict[str, dict[int, dict]] = {}
    for run_name, data in by_run.items():
        if data is None:
            by_run_idx[run_name] = {}
            continue
        by_run_idx[run_name] = {r["i"]: r for r in data["results"]}

    gold_for: dict[int, str] = {}
    for i in miss_idx:
        for run_name in by_run_idx:
            r = by_run_idx[run_name].get(i)
            if r is not None:
                gold_for[i] = r["expected"]
                break

    arm = verdict.get("arm", "")
    git_sha = verdict.get("git_sha", "")
    image_id = verdict.get("image_id", "")
    phase_e_layers = verdict.get("phase_e_layers", "")
    wo_split = verdict.get("wo_split", "")
    n_runs = verdict.get("n_runs", "")
    floor = verdict.get("gsm8k_floor", "")
    container_alive = verdict.get("container_alive_at_end", "")
    corruption_hits = verdict.get("docker_log_corruption_hits", "")
    gate_pass = verdict.get("gate_2b_pass", "")

    lines: list[str] = []
    lines.append(f"# Stage 2b survival soak — {arm}")
    lines.append("")
    lines.append(f"- generated: {dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"- arm: {arm}")
    lines.append(f"- git_sha: {git_sha}")
    lines.append(f"- image_id: {image_id}")
    lines.append(f"- phase_e_layers: `{phase_e_layers}`")
    lines.append(f"- wo_split: {wo_split}")
    lines.append(f"- n_runs: {n_runs}")
    lines.append(f"- gsm8k_floor: {floor}")
    lines.append(f"- container_alive_at_end: {container_alive}")
    lines.append(f"- docker_log_corruption_hits: {corruption_hits}")
    lines.append(f"- **gate_2b_pass: {gate_pass}**")
    lines.append("")

    # Per-run headline.
    lines.append("## Per-run headline")
    lines.append("")
    lines.append("| run | correct | errors | wall (s) | pass |")
    lines.append("|---|---|---|---|---|")
    runs_arr = {r["run"]: r for r in verdict.get("runs", [])}
    for d in run_dirs:
        idx = _run_index(d.name)
        r = runs_arr.get(idx, {})
        gsm = by_run[d.name]
        correct = r.get("correct", "?")
        errors = r.get("errors", "?")
        wall = f"{gsm['total_seconds']:.0f}" if (gsm and "total_seconds" in gsm) else "?"
        run_pass = r.get("pass", r.get("ok", False))
        lines.append(
            f"| {idx} | {correct}/50 | {errors} | {wall} | "
            f"{'true' if run_pass else 'false'} |"
        )
    lines.append("")

    # Per-question miss table.
    lines.append("## Per-question miss table")
    lines.append("")
    lines.append(
        "Union of every question any run got non-OK on. Cells show the "
        "model's predicted answer (or `OK` if that run answered correctly). "
        "Same anti-overclaim discipline as the sweep summary — a stable "
        "miss set across all 5 runs is the model's blind spot, not a "
        "stability problem. A miss set that grows from run to run is a "
        "state-corruption / drift signal."
    )
    lines.append("")
    if not miss_idx:
        lines.append("_All runs scored 50/50 on the seed=42 sample — no "
                     "miss table to emit._")
    else:
        header = "| Q (gold) |"
        sep = "|---|"
        for d in run_dirs:
            header += f" {d.name} |"
            sep += "---|"
        lines.append(header)
        lines.append(sep)
        for i in sorted(miss_idx):
            row = f"| Q{i} (gold={gold_for.get(i,'?')}) |"
            for d in run_dirs:
                r = by_run_idx[d.name].get(i)
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
    lines.append("## Verdict framing")
    lines.append("")
    lines.append(
        f"- **Gate 2b spec:** all {n_runs} runs ≥ {floor}/50 AND 0 errors "
        f"AND container alive at end AND `docker.log` corruption hits = 0."
    )
    lines.append(
        f"- **Observed:** container_alive={container_alive}, "
        f"corruption_hits={corruption_hits}, gate_2b_pass={gate_pass}."
    )
    sweep_overlap = (
        "Stable miss set across all runs => model/eval blind spot, not "
        "soak drift. New questions failing in later runs => state "
        "corruption candidate; bisect by run order."
    )
    lines.append(f"- **Miss-table read:** {sweep_overlap}")
    lines.append("")

    # Per-run artifacts.
    lines.append("## Per-run artifacts")
    lines.append("")
    for d in run_dirs:
        lines.append(
            f"- [{d.name}/gsm8k.json]({d.name}/gsm8k.json), "
            f"[{d.name}/gsm8k.log]({d.name}/gsm8k.log)"
        )
    lines.append(
        "- [dispatch_audit.json](dispatch_audit.json), "
        "[verdict.json](verdict.json), "
        "[serve.log](serve.log), "
        "[docker.log](docker.log)"
    )
    lines.append("")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines))
    print(f"[soak-summary] wrote {out_path} ({sum(1 for _ in lines)} lines, "
          f"{len(run_dirs)} runs, {len(miss_idx)} miss indices)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
