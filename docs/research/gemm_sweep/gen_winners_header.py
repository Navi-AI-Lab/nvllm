# docs/research/gemm_sweep/gen_winners_header.py
"""Generate csrc/libtorch_stable/quantization/fp4/nvfp4_winners_table.hpp
and benchmarks/.../winners.json from shortlist.json + microbench.csv.

The top-1 winner per (shape, bucket) is the first entry in
shortlist.json::by_shape[shape][bucket]; analyze.py:62 already sorts by
rank, so no re-aggregation is needed. (N, K) per shape is read from
microbench.csv rows (column `config_id,shape,M,N,K,min_us`).

Usage:
    gen_winners_header.py \\
        --sweep-dir benchmarks/.../2026-04-21-qwen35-27b \\
        --shortlist-header csrc/.../nvfp4_shortlist_configs.hpp \\
        --model-tag qwen35_27b [--check]
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

BUCKETS = ["16-32", "64-128", "192-256"]  # mid-batch buckets from shortlist.json
SHAPE_ORDER = ["qkv_proj", "o_proj", "gate_up_proj", "down_proj"]  # stable emission order

# Small-M buckets (Phase 6b 2026-04-29). Computed directly from microbench.csv
# rather than from shortlist.json — the original shortlist analysis skipped
# small-M as the "Stream-K band". Per-cell deltas vs Cfg_128x128x256_TmaWSCoop_SK
# range from -1.16% to -26.36%.
SMALL_BUCKETS: dict[str, list[int]] = {
    "1-2": [1, 2],
    "4-8": [4, 8],
    "16":  [16],
}

# Phase 6b small-M-only shapes: discovered post-rebuild via NVLLM_FP4_GEMM_LOG_TABLE=1
# logging that they were falling through to Stream-K. NOT in the original
# 2026-04-21 mid-M sweep, so mid-M buckets are skipped (lookup_m_mid_winner
# returns -1 for these — they hit M256 default at mp2 > 16, unchanged).
# Each entry: shape_name -> (N, K, supplemental_microbench.csv path)
SMALL_ONLY_SHAPES: dict[str, tuple[int, int, str]] = {
    "gdn_in_proj_qkv": (
        14336, 5120,
        "benchmarks/nvllm/traces/gemm_sweep_sm120_phase6b_gdn/2026-04-29/microbench.csv",
    ),
}


def _load_shortlist(p: Path) -> dict:
    data = json.loads(p.read_text())
    for shape in SHAPE_ORDER:
        if shape not in data["by_shape"]:
            raise SystemExit(f"shortlist.json missing shape: {shape}")
        for bucket in BUCKETS:
            if bucket not in data["by_shape"][shape]:
                raise SystemExit(f"shortlist.json missing {shape}/{bucket}")
    return data


def _load_nk_from_microbench(csv_path: Path) -> dict[str, tuple[int, int]]:
    by_shape: dict[str, tuple[int, int]] = {}
    with csv_path.open() as f:
        for row in csv.DictReader(f):
            shape = row["shape"]
            nk = (int(row["N"]), int(row["K"]))
            if shape in by_shape and by_shape[shape] != nk:
                raise SystemExit(
                    f"microbench.csv: inconsistent (N,K) for {shape}: "
                    f"{by_shape[shape]} vs {nk}"
                )
            by_shape[shape] = nk
    for shape in SHAPE_ORDER:
        if shape not in by_shape:
            raise SystemExit(f"microbench.csv missing shape: {shape}")
    return by_shape


def _compute_small_only_winners(repo_root: Path) -> dict[str, dict[str, object]]:
    """Phase 6b small-M-only shapes (discovered via runtime log_table miss).
    For each, read its supplemental microbench CSV and pick top-1 per
    small-M bucket. Returns: {shape: {"N": n, "K": k, "1-2": cfg, ...}}.
    """
    out: dict[str, dict[str, object]] = {}
    for shape, (n, k, csv_rel) in SMALL_ONLY_SHAPES.items():
        csv_path = repo_root / csv_rel
        if not csv_path.exists():
            raise SystemExit(
                f"SMALL_ONLY_SHAPES[{shape!r}]: supplemental CSV missing: {csv_path}"
            )
        rows: list[dict] = []
        with csv_path.open() as f:
            for row in csv.DictReader(f):
                if float(row["min_us"]) < 0:
                    continue
                rows.append(
                    {
                        "shape": row["shape"],
                        "M": int(row["M"]),
                        "config": row["config_id"],
                        "us": float(row["min_us"]),
                    }
                )
        entry: dict[str, object] = {"N": n, "K": k}
        for bucket_name, m_vals in SMALL_BUCKETS.items():
            cfg_to_avg: dict[str, list[float]] = {}
            for r in rows:
                if r["M"] not in m_vals:
                    continue
                cfg_to_avg.setdefault(r["config"], []).append(r["us"])
            if not cfg_to_avg:
                raise SystemExit(
                    f"SMALL_ONLY_SHAPES[{shape!r}]: missing data for bucket {bucket_name}"
                )
            cfg_avg = {c: sum(v) / len(v) for c, v in cfg_to_avg.items()}
            best = min(cfg_avg, key=cfg_avg.get)
            entry[bucket_name] = best
        out[shape] = entry
    return out


def _compute_small_winners(csv_path: Path) -> dict[str, dict[str, str]]:
    """Read microbench.csv directly and pick top-1 config per (shape, small-M
    bucket). Small-M was excluded from analyze.py::shortlist_top3 because it
    was the historical Stream-K band — but Phase 6b confirmed every (shape,
    small-M) cell has a Persistent winner that beats the hardcoded Stream-K
    config. Aggregation rule: per (shape, bucket), average min_us across the
    bucket's M values for each config, pick lowest average.
    """
    rows: list[dict] = []
    with csv_path.open() as f:
        for row in csv.DictReader(f):
            if float(row["min_us"]) < 0:
                continue
            rows.append(
                {
                    "shape": row["shape"],
                    "M": int(row["M"]),
                    "config": row["config_id"],
                    "us": float(row["min_us"]),
                }
            )

    by_shape_bucket: dict[str, dict[str, str]] = {}
    for shape in SHAPE_ORDER:
        by_shape_bucket[shape] = {}
        for bucket_name, m_vals in SMALL_BUCKETS.items():
            cfg_to_avg: dict[str, list[float]] = {}
            for r in rows:
                if r["shape"] != shape or r["M"] not in m_vals:
                    continue
                cfg_to_avg.setdefault(r["config"], []).append(r["us"])
            if not cfg_to_avg:
                raise SystemExit(
                    f"microbench.csv missing data for {shape}/{bucket_name}"
                )
            cfg_avg = {c: sum(v) / len(v) for c, v in cfg_to_avg.items()}
            best = min(cfg_avg, key=cfg_avg.get)
            by_shape_bucket[shape][bucket_name] = best
    return by_shape_bucket


def _parse_header_idx_map(hpp_path: Path) -> dict[int, str]:
    """Parse struct ShortlistCfg_<idx> declarations + their static name =
    "..." line. Used only as a drift check against shortlist.json::all_configs."""
    text = hpp_path.read_text()
    by_idx: dict[int, str] = {}
    # Combined regex: match the whole struct body up to the name= line. The
    # non-greedy `[^}]*?` stops at the first `}` so each struct matches exactly
    # once. Avoids any fixed-size window (some structs, e.g. the SK variants
    # with an extra TileScheduler line, spill past a naive 500-char tail).
    for m in re.finditer(
        r'struct ShortlistCfg_(\d+)\s*\{[^}]*?name\s*=\s*"([^"]+)"',
        text, re.DOTALL,
    ):
        by_idx[int(m.group(1))] = m.group(2)
    if not by_idx:
        raise SystemExit(f"No ShortlistCfg_<idx> structs found in {hpp_path}")
    return by_idx


def build_winners(sweep_dir: Path, shortlist_header: Path, model_tag: str) -> dict:
    shortlist = _load_shortlist(sweep_dir / "shortlist.json")
    nk = _load_nk_from_microbench(sweep_dir / "microbench.csv")

    name_to_idx = {name: i for i, name in enumerate(shortlist["all_configs"])}

    header_idx_to_name = _parse_header_idx_map(shortlist_header)
    header_name_to_idx = {v: k for k, v in header_idx_to_name.items()}
    # Drift check — both maps must agree.
    for name, i in name_to_idx.items():
        if header_name_to_idx.get(name) != i:
            raise SystemExit(
                f"idx drift: shortlist.json all_configs gives {name}->{i} "
                f"but {shortlist_header.name} gives {name}->{header_name_to_idx.get(name)}"
            )

    small_winners = _compute_small_winners(sweep_dir / "microbench.csv")
    repo_root = Path(__file__).resolve().parents[3]
    small_only_winners = _compute_small_only_winners(repo_root)

    by_shape: dict = {}
    emit_order = list(SHAPE_ORDER) + list(SMALL_ONLY_SHAPES.keys())
    for shape in emit_order:
        if shape in SHAPE_ORDER:
            n, k = nk[shape]
            entry = {"N": n, "K": k}
            for bucket in BUCKETS:
                top1_name = shortlist["by_shape"][shape][bucket][0]
                if top1_name not in name_to_idx:
                    raise SystemExit(
                        f"{shape}/{bucket} top-1 {top1_name!r} not in shortlist all_configs"
                    )
                entry[bucket] = {"cfg": top1_name, "idx": name_to_idx[top1_name]}
            for bucket, top1_name in small_winners[shape].items():
                if top1_name not in name_to_idx:
                    raise SystemExit(
                        f"{shape}/small-{bucket} top-1 {top1_name!r} not in shortlist all_configs"
                    )
                entry[bucket] = {"cfg": top1_name, "idx": name_to_idx[top1_name]}
        else:
            # SMALL_ONLY_SHAPES: small-M only; mid-M buckets emit -1 idx so
            # lookup_m_mid_winner returns -1 -> M256 default (unchanged).
            so_entry = small_only_winners[shape]
            entry = {"N": so_entry["N"], "K": so_entry["K"]}
            for bucket in BUCKETS:
                entry[bucket] = {"cfg": "(none)", "idx": -1}
            for bucket in SMALL_BUCKETS:
                top1_name = so_entry[bucket]
                if top1_name not in name_to_idx:
                    raise SystemExit(
                        f"{shape}/small-{bucket} top-1 {top1_name!r} not in shortlist all_configs"
                    )
                entry[bucket] = {"cfg": top1_name, "idx": name_to_idx[top1_name]}
        by_shape[shape] = entry

    return {
        "model_tag": model_tag,
        "source_sweep": sweep_dir.name,
        "source_ranking_rule": "mid-M (16-32, 64-128, 192-256): shortlist.json top-1 from analyze.py::shortlist_top3(); small-M (1-2, 4-8, 16): per-bucket avg min_us across microbench.csv, top-1 picked directly (Phase 6b 2026-04-29)",
        "by_shape": by_shape,
    }


def emit_header(winners: dict, model_tag: str, source_sweep: str) -> str:
    rows = []
    emit_order = list(SHAPE_ORDER) + list(SMALL_ONLY_SHAPES.keys())
    for shape in emit_order:
        e = winners["by_shape"][shape]
        rows.append(
            f"    {{ {e['N']:>6}, {e['K']:>6}, "
            f"/*1-2*/ {e['1-2']['idx']:>2}, "
            f"/*4-8*/ {e['4-8']['idx']:>2}, "
            f"/*16*/ {e['16']['idx']:>2}, "
            f"/*16-32*/ {e['16-32']['idx']:>2}, "
            f"/*64-128*/ {e['64-128']['idx']:>2}, "
            f"/*192-256*/ {e['192-256']['idx']:>2}, "
            f'"{model_tag}/{shape}" }},'
        )
    rows_text = "\n".join(rows)
    return f"""#pragma once
// Auto-generated by docs/research/gemm_sweep/gen_winners_header.py
// Source: benchmarks/nvllm/traces/gemm_sweep_sm120/{source_sweep}/winners.json
// Do not edit by hand — regenerate after winners.json changes.

#include <cstdio>
#include <cstdlib>

namespace nvllm::fp4 {{

struct ShapeWinners {{
    int N;
    int K;
    int idx_1_2;       // small-M: mp2 in {{1,2}}
    int idx_4_8;       // small-M: mp2 in {{4,8}}
    int idx_16;        // small-M: mp2 == 16
    int idx_16_32;     // mid-M:   mp2 == 32 (bucket name "16-32" historical)
    int idx_64_128;    // mid-M:   mp2 in {{64,128}}
    int idx_192_256;   // mid-M:   mp2 == 256
    const char* shape_name;  // debug only — NVLLM_FP4_GEMM_LOG_TABLE=1
}};

inline constexpr ShapeWinners kWinnersTable[] = {{
{rows_text}
}};

// Returns ShortlistCfg idx or -1 if (n,k) not in table / mp2 not in small band.
// Small band: mp2 in {{1, 2, 4, 8, 16}}.
inline int lookup_m_small_winner(int n, int k, int mp2) {{
    for (auto const& w : kWinnersTable) {{
        if (w.N != n || w.K != k) continue;
        int idx;
        switch (mp2) {{
            case 1: case 2:  idx = w.idx_1_2; break;
            case 4: case 8:  idx = w.idx_4_8; break;
            case 16:         idx = w.idx_16;  break;
            default:         return -1;
        }}
        if (const char* dbg = std::getenv("NVLLM_FP4_GEMM_LOG_TABLE"); dbg && dbg[0] == '1') {{
            std::fprintf(stderr, "[nvllm] fp4 small-M table: %s mp2=%d -> idx=%d\\n",
                         w.shape_name, mp2, idx);
        }}
        return idx;
    }}
    return -1;
}}

// Returns ShortlistCfg idx or -1 if (n,k) not in table / mp2 not in mid band.
inline int lookup_m_mid_winner(int n, int k, int mp2) {{
    for (auto const& w : kWinnersTable) {{
        if (w.N != n || w.K != k) continue;
        int idx;
        switch (mp2) {{
            case 32:            idx = w.idx_16_32;  break;
            case 64: case 128:  idx = w.idx_64_128; break;
            case 256:           idx = w.idx_192_256; break;
            default:            return -1;
        }}
        if (const char* dbg = std::getenv("NVLLM_FP4_GEMM_LOG_TABLE"); dbg && dbg[0] == '1') {{
            std::fprintf(stderr, "[nvllm] fp4 table: %s mp2=%d -> idx=%d\\n",
                         w.shape_name, mp2, idx);
        }}
        return idx;
    }}
    return -1;
}}

}} // namespace nvllm::fp4
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep-dir", type=Path, required=True)
    ap.add_argument("--shortlist-header", type=Path, required=True)
    ap.add_argument("--model-tag", required=True)
    ap.add_argument("--winners-out", type=Path,
                    help="Override winners.json path (default: <sweep-dir>/winners.json)")
    ap.add_argument("--header-out", type=Path,
                    help="Override header path (default: csrc/.../nvfp4_winners_table.hpp)")
    ap.add_argument("--check", action="store_true",
                    help="Exit non-zero if committed header differs from fresh emit")
    args = ap.parse_args()

    winners = build_winners(args.sweep_dir, args.shortlist_header, args.model_tag)
    header_text = emit_header(winners, args.model_tag, args.sweep_dir.name)

    repo_root = Path(__file__).resolve().parents[3]
    default_winners = args.sweep_dir / "winners.json"
    default_header = repo_root / "csrc/libtorch_stable/quantization/fp4/nvfp4_winners_table.hpp"
    winners_path = args.winners_out or default_winners
    header_path = args.header_out or default_header

    if args.check:
        if not header_path.exists():
            sys.stderr.write(f"--check: committed header missing: {header_path}\n")
            sys.exit(2)
        committed = header_path.read_text()
        if committed != header_text:
            sys.stdout.write(header_text)
            sys.stderr.write("--check: committed header is stale (see stdout for fresh emit)\n")
            sys.exit(1)
        print(f"--check OK: {header_path} is up-to-date")
        return

    winners_path.write_text(json.dumps(winners, indent=2) + "\n")
    header_path.write_text(header_text)
    print(f"Wrote {winners_path}")
    print(f"Wrote {header_path}")


if __name__ == "__main__":
    main()
