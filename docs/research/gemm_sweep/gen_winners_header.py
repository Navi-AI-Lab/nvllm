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

BUCKETS = ["16-32", "64-128", "192-256"]  # 1-8 is the Stream-K band — skipped.
SHAPE_ORDER = ["qkv_proj", "o_proj", "gate_up_proj", "down_proj"]  # stable emission order

STRUCT_IDX_RE = re.compile(r"struct ShortlistCfg_(\d+)\b")


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

    by_shape: dict = {}
    for shape in SHAPE_ORDER:
        n, k = nk[shape]
        entry = {"N": n, "K": k}
        for bucket in BUCKETS:
            top1_name = shortlist["by_shape"][shape][bucket][0]
            if top1_name not in name_to_idx:
                raise SystemExit(
                    f"{shape}/{bucket} top-1 {top1_name!r} not in shortlist all_configs"
                )
            entry[bucket] = {"cfg": top1_name, "idx": name_to_idx[top1_name]}
        by_shape[shape] = entry

    return {
        "model_tag": model_tag,
        "source_sweep": sweep_dir.name,
        "source_ranking_rule": "shortlist.json top-1 per (shape, bucket) from analyze.py::shortlist_top3() -> ranked by bucket_min_us",
        "by_shape": by_shape,
    }


def emit_header(winners: dict, model_tag: str, source_sweep: str) -> str:
    rows = []
    for shape in SHAPE_ORDER:
        e = winners["by_shape"][shape]
        rows.append(
            f"    {{ {e['N']:>6}, {e['K']:>6}, "
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
    int idx_16_32;
    int idx_64_128;
    int idx_192_256;
    const char* shape_name;  // debug only — NVLLM_FP4_GEMM_LOG_TABLE=1
}};

inline constexpr ShapeWinners kWinnersTable[] = {{
{rows_text}
}};

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
