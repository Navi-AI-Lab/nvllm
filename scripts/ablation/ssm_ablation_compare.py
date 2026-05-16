"""
SSM zero-on-realloc ablation comparison tool.

Reads each arm's verdict.json + per-question JSONL trace, emits a markdown
comparison at <OUT_DIR>/ANALYSIS.md with:
  - Verdict table (arm x run x correct/errors/gate_pass)
  - Q1-Q50 per-question table for Run 4 (collapse window) across all 4 arms:
    latency_s, completion_tokens, decode_tok_s, finish_reason, correct
  - Aggregate per-arm steady-state stats: median decode_tok_s, p95 latency,
    mean completion_tokens
  - Friend's interpretation thresholds applied: which arm matches "real
    pipeline win" vs "shortened generations"
  - Drained KV invariant check from /metrics pre vs post snapshot.

Usage:
    python3 /tmp/ssm_ablation_compare.py [OUT_DIR]

Default OUT_DIR: /tmp/ssm_ablation_suite

Reads:
    <OUT_DIR>/<arm>/verdict.json
    <OUT_DIR>/<arm>/perq.jsonl            (concatenated by runner)
    <OUT_DIR>/<arm>/run<i>/perq.jsonl     (per-run, used for Run-4 table)
    <OUT_DIR>/<arm>/run<i>/metrics_*.json (pre / q10..q50 / post snapshots)

Writes:
    <OUT_DIR>/ANALYSIS.md
"""

from __future__ import annotations

import json
import os
import statistics
import sys
from typing import Any

# Arm presentation order = same as the runner.
ARM_ORDER = ("both", "neither", "ssm_only", "kv_only")
RUN_INDICES = (1, 2, 3, 4, 5)
COLLAPSE_RUN = 4  # the friend's collapse-window pin
METRIC_KEY_KV_USAGE = "vllm:kv_cache_usage_perc"
METRIC_KEY_KV_USAGE_TOL = 0.05  # 5 percentage-point tolerance for "drained"

# Friend's interpretation thresholds.
# - "real pipeline win" = decode_tok_s materially higher AND completion_tokens
#   not shortened relative to neither/baseline.
# - "shortened generations" = decode_tok_s higher BUT mean completion_tokens
#   notably lower (typical max_tokens=512 with finish_reason=length disappearing).
TPOT_WIN_RATIO = 1.30          # >=30% decode_tok_s vs baseline = "win"
SHORTEN_RATIO = 0.85           # <=85% of baseline mean completion_tokens = "shortened"


def _load_json(path: str) -> Any:
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def _load_jsonl(path: str) -> list[dict]:
    out: list[dict] = []
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(json.loads(line))
                except Exception:
                    continue
    except FileNotFoundError:
        pass
    return out


def _p95(values: list[float]) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    idx = max(0, int(round(0.95 * (len(s) - 1))))
    return s[idx]


def _fmt(v: Any, prec: int = 2) -> str:
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.{prec}f}"
    return str(v)


def _arm_stats(perq_records: list[dict]) -> dict:
    if not perq_records:
        return {
            "n_questions": 0,
            "median_decode_tok_s": 0.0,
            "p95_wall_time_s": 0.0,
            "mean_completion_tokens": 0.0,
            "finish_reason_counts": {},
        }
    decode_rates = [
        float(r.get("decode_tok_s", 0) or 0)
        for r in perq_records
        if float(r.get("decode_tok_s", 0) or 0) > 0
    ]
    walls = [
        float(r.get("wall_time_s", 0) or 0)
        for r in perq_records
        if float(r.get("wall_time_s", 0) or 0) > 0
    ]
    comp_tokens = [int(r.get("completion_tokens", 0) or 0) for r in perq_records]
    finish_reasons: dict[str, int] = {}
    for r in perq_records:
        fr = str(r.get("finish_reason"))
        finish_reasons[fr] = finish_reasons.get(fr, 0) + 1
    return {
        "n_questions": len(perq_records),
        "median_decode_tok_s": (statistics.median(decode_rates) if decode_rates else 0.0),
        "p95_wall_time_s": _p95(walls),
        "mean_completion_tokens": (statistics.mean(comp_tokens) if comp_tokens else 0.0),
        "finish_reason_counts": finish_reasons,
    }


def _drained_invariant(metric_pre: dict, metric_post: dict) -> dict:
    """Did KV usage % return to baseline at the post snapshot?

    Returns dict with pre, post, delta_pp, drained (bool).
    """
    if not isinstance(metric_pre, dict) or not isinstance(metric_post, dict):
        return {"pre": None, "post": None, "delta_pp": None, "drained": None}
    m_pre = (metric_pre or {}).get("metrics", {}) or {}
    m_post = (metric_post or {}).get("metrics", {}) or {}
    pre_val = m_pre.get(METRIC_KEY_KV_USAGE)
    post_val = m_post.get(METRIC_KEY_KV_USAGE)
    if pre_val is None or post_val is None:
        return {"pre": pre_val, "post": post_val, "delta_pp": None, "drained": None}
    try:
        delta = float(post_val) - float(pre_val)
    except (TypeError, ValueError):
        return {"pre": pre_val, "post": post_val, "delta_pp": None, "drained": None}
    return {
        "pre": float(pre_val),
        "post": float(post_val),
        "delta_pp": delta,
        "drained": abs(delta) <= METRIC_KEY_KV_USAGE_TOL,
    }


def _load_arm(out_dir: str, arm: str) -> dict:
    arm_dir = os.path.join(out_dir, arm)
    verdict = _load_json(os.path.join(arm_dir, "verdict.json")) or {}
    perq_concat_path = os.path.join(arm_dir, "perq.jsonl")
    perq_concat = _load_jsonl(perq_concat_path)

    # Per-run breakouts.
    runs: dict[int, dict] = {}
    for run_idx in RUN_INDICES:
        run_dir = os.path.join(arm_dir, f"run{run_idx}")
        gsm = _load_json(os.path.join(run_dir, "gsm8k.json"))
        perq = _load_jsonl(os.path.join(run_dir, "perq.jsonl"))
        runs[run_idx] = {"gsm8k": gsm, "perq": perq, "dir": run_dir}

    # /metrics snapshots: typically Run 1 holds the pre/post pair. We aggregate
    # from each run's own pre/post if they exist (the eval writes them next to
    # the run's gsm8k.json).
    arm_metrics = {}
    for run_idx in RUN_INDICES:
        run_dir = runs[run_idx]["dir"]
        arm_metrics[run_idx] = {
            "pre": _load_json(os.path.join(run_dir, "metrics_pre.json")),
            "post": _load_json(os.path.join(run_dir, "metrics_post.json")),
        }
        for tag in ("q10", "q20", "q30", "q40", "q50"):
            snap = _load_json(os.path.join(run_dir, f"metrics_{tag}.json"))
            if snap is not None:
                arm_metrics[run_idx][tag] = snap

    return {
        "verdict": verdict,
        "perq_concat": perq_concat,
        "runs": runs,
        "metrics": arm_metrics,
    }


def _interpretation(stats_by_arm: dict[str, dict]) -> dict[str, str]:
    """Apply friend's win/shortened thresholds, with 'neither' as baseline."""
    out: dict[str, str] = {}
    baseline = stats_by_arm.get("neither", {})
    base_decode = float(baseline.get("median_decode_tok_s") or 0.0) or None
    base_compt = float(baseline.get("mean_completion_tokens") or 0.0) or None

    for arm in ARM_ORDER:
        s = stats_by_arm.get(arm, {})
        decode = float(s.get("median_decode_tok_s") or 0.0)
        compt = float(s.get("mean_completion_tokens") or 0.0)
        if base_decode is None or base_compt is None or arm == "neither":
            verdict = "baseline" if arm == "neither" else "no baseline available"
            out[arm] = verdict
            continue
        decode_ratio = decode / base_decode if base_decode > 0 else 0.0
        compt_ratio = compt / base_compt if base_compt > 0 else 0.0
        if decode_ratio >= TPOT_WIN_RATIO and compt_ratio >= SHORTEN_RATIO:
            verdict = (
                f"REAL pipeline win "
                f"(decode {decode_ratio:.2f}x baseline, compt {compt_ratio:.2f}x)"
            )
        elif decode_ratio >= TPOT_WIN_RATIO and compt_ratio < SHORTEN_RATIO:
            verdict = (
                f"SHORTENED generations - speed inflated "
                f"(decode {decode_ratio:.2f}x, compt only {compt_ratio:.2f}x)"
            )
        elif decode_ratio < TPOT_WIN_RATIO and compt_ratio >= SHORTEN_RATIO:
            verdict = (
                f"no decode win vs baseline "
                f"(decode {decode_ratio:.2f}x, compt {compt_ratio:.2f}x)"
            )
        else:
            verdict = (
                f"no win + shortened "
                f"(decode {decode_ratio:.2f}x, compt {compt_ratio:.2f}x)"
            )
        out[arm] = verdict
    return out


def _render_verdict_table(arms_data: dict[str, dict]) -> str:
    """Verdict table: arm x run x correct/errors/gate_pass."""
    lines: list[str] = []
    lines.append("| Arm | SSM | KV | Run 1 | Run 2 | Run 3 | Run 4 | Run 5 | Gate |")
    lines.append("|-----|-----|----|-------|-------|-------|-------|-------|------|")
    for arm in ARM_ORDER:
        d = arms_data.get(arm, {})
        v = d.get("verdict", {})
        ssm = v.get("ssm_zero_on_realloc", v.get("ssm_sentinel", "?"))
        kv = v.get("kv_zero_for_mamba_ids", v.get("kv_sentinel", "?"))
        gate = v.get("gate_pass", "?")
        cells: list[str] = []
        runs_arr = v.get("runs") or []
        run_by_idx = {int(r.get("run", -1)): r for r in runs_arr if isinstance(r, dict)}
        for run_idx in RUN_INDICES:
            r = run_by_idx.get(run_idx)
            if r is None:
                cells.append("-")
                continue
            if "correct" not in r:
                cells.append(f"FAIL({r.get('reason', '?')})")
                continue
            cells.append(f"{r['correct']}/{r.get('errors', 0)}err")
        lines.append(
            "| " + " | ".join([arm, str(ssm), str(kv), *cells, str(gate)]) + " |"
        )
    return "\n".join(lines)


def _render_run_table(arms_data: dict[str, dict], run_idx: int) -> str:
    """Per-question table for one run across all 4 arms. Columns:
    Q | <arm>:lat | <arm>:ct | <arm>:dtok/s | <arm>:fr | <arm>:ok
    """
    arm_perq: dict[str, list[dict]] = {
        arm: arms_data.get(arm, {}).get("runs", {}).get(run_idx, {}).get("perq") or []
        for arm in ARM_ORDER
    }
    # Build index by prompt_index per arm.
    indexed: dict[str, dict[int, dict]] = {
        arm: {int(r.get("prompt_index", -1)): r for r in arm_perq[arm]}
        for arm in ARM_ORDER
    }
    # Union of seen prompt indices, sorted.
    all_qs: set[int] = set()
    for arm in ARM_ORDER:
        all_qs.update(indexed[arm].keys())
    if not all_qs:
        return "_(no per-Q records for run "f"{run_idx}_)"

    header_cells = ["Q"]
    for arm in ARM_ORDER:
        header_cells += [
            f"{arm}:lat", f"{arm}:ct", f"{arm}:dtok/s", f"{arm}:fr", f"{arm}:ok"
        ]
    lines: list[str] = []
    lines.append("| " + " | ".join(header_cells) + " |")
    lines.append("|" + "|".join(["---"] * len(header_cells)) + "|")
    for q in sorted(all_qs):
        row = [str(q)]
        for arm in ARM_ORDER:
            r = indexed[arm].get(q)
            if r is None:
                row += ["-", "-", "-", "-", "-"]
                continue
            row += [
                _fmt(r.get("wall_time_s"), 2),
                _fmt(r.get("completion_tokens"), 0),
                _fmt(r.get("decode_tok_s"), 2),
                _fmt(r.get("finish_reason")),
                "Y" if r.get("correct") else "N",
            ]
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def _render_steady_state(stats_by_arm: dict[str, dict]) -> str:
    """Aggregate per-arm steady-state stats."""
    lines: list[str] = []
    lines.append(
        "| Arm | N | median dtok/s | p95 wall_s | mean completion_tokens | finish_reason hist |"
    )
    lines.append("|-----|---|---------------|------------|------------------------|--------------------|")
    for arm in ARM_ORDER:
        s = stats_by_arm.get(arm, {})
        fr = s.get("finish_reason_counts", {}) or {}
        fr_str = ", ".join(f"{k}={v}" for k, v in sorted(fr.items()))
        lines.append(
            f"| {arm} | {s.get('n_questions', 0)} | "
            f"{_fmt(s.get('median_decode_tok_s'), 2)} | "
            f"{_fmt(s.get('p95_wall_time_s'), 2)} | "
            f"{_fmt(s.get('mean_completion_tokens'), 1)} | {fr_str} |"
        )
    return "\n".join(lines)


def _render_drained_section(arms_data: dict[str, dict]) -> str:
    lines: list[str] = []
    lines.append(
        f"Tolerance: |delta| <= {METRIC_KEY_KV_USAGE_TOL:.2f} (5 pp) "
        f"counts as drained."
    )
    lines.append("")
    lines.append("| Arm | Run | KV pre | KV post | delta | drained |")
    lines.append("|-----|-----|--------|---------|-------|---------|")
    for arm in ARM_ORDER:
        d = arms_data.get(arm, {})
        metrics = d.get("metrics", {})
        for run_idx in RUN_INDICES:
            m = metrics.get(run_idx, {})
            di = _drained_invariant(m.get("pre"), m.get("post"))
            drained = di.get("drained")
            drained_s = "-" if drained is None else ("Y" if drained else "N")
            lines.append(
                f"| {arm} | {run_idx} | {_fmt(di.get('pre'), 4)} | "
                f"{_fmt(di.get('post'), 4)} | {_fmt(di.get('delta_pp'), 4)} | "
                f"{drained_s} |"
            )
    return "\n".join(lines)


def main():
    out_dir = sys.argv[1] if len(sys.argv) > 1 else "/tmp/ssm_ablation_suite"
    if not os.path.isdir(out_dir):
        print(f"ERROR: OUT_DIR not found: {out_dir}", file=sys.stderr)
        sys.exit(2)

    arms_data: dict[str, dict] = {
        arm: _load_arm(out_dir, arm) for arm in ARM_ORDER
    }
    # Steady-state stats per arm computed on perq concat.
    stats_by_arm = {
        arm: _arm_stats(arms_data[arm].get("perq_concat") or [])
        for arm in ARM_ORDER
    }
    interp = _interpretation(stats_by_arm)

    md: list[str] = []
    md.append("# SSM zero-on-realloc ablation: 4-arm comparison")
    md.append("")
    md.append(f"- OUT_DIR: `{out_dir}`")
    comp = _load_json(os.path.join(out_dir, "comparison.json")) or {}
    md.append(f"- git_sha: `{comp.get('git_sha', '?')}`")
    md.append(f"- image: `{comp.get('image', '?')}`")
    md.append(f"- N runs per arm: {comp.get('n_runs', '?')}")
    md.append(f"- gsm8k_floor: {comp.get('gsm8k_floor', '?')}")
    md.append("")
    md.append("## Verdict table (run x correct/errors)")
    md.append("")
    md.append(_render_verdict_table(arms_data))
    md.append("")
    md.append(
        f"## Per-question table - Run {COLLAPSE_RUN} (collapse window)"
    )
    md.append("")
    md.append(
        "Columns per arm: lat (wall_time_s), ct (completion_tokens), "
        "dtok/s (decode_tok_s), fr (finish_reason), ok (correct)."
    )
    md.append("")
    md.append(_render_run_table(arms_data, COLLAPSE_RUN))
    md.append("")
    md.append("## Aggregate per-arm steady-state stats (concat across runs)")
    md.append("")
    md.append(_render_steady_state(stats_by_arm))
    md.append("")
    md.append("## Friend's interpretation thresholds applied")
    md.append("")
    md.append(
        f"- 'real pipeline win' iff median decode_tok_s >= "
        f"{TPOT_WIN_RATIO:.2f}x baseline ('neither') AND mean completion_tokens "
        f">= {SHORTEN_RATIO:.2f}x baseline"
    )
    md.append(
        f"- 'shortened generations' iff decode rate up but completion_tokens "
        f"< {SHORTEN_RATIO:.2f}x baseline"
    )
    md.append("")
    for arm in ARM_ORDER:
        md.append(f"- **{arm}**: {interp.get(arm, '-')}")
    md.append("")
    md.append("## Drained KV invariant (per-run pre vs post)")
    md.append("")
    md.append(_render_drained_section(arms_data))
    md.append("")

    out_md = os.path.join(out_dir, "ANALYSIS.md")
    with open(out_md, "w") as f:
        f.write("\n".join(md) + "\n")
    print(out_md)


if __name__ == "__main__":
    main()
