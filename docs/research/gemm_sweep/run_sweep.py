"""Run the full NVFP4 GEMM microbench sweep and aggregate into a CSV.

Reads Qwen3.5-27B dense GEMM shapes from the model's config.json (per memory
feedback_verify_model_config - no hardcoded dims). Enumerates (shape x M x
config) combinations, shells out to the container-built microbench binary
per combo, aggregates into microbench.csv.

Usage (from repo root):
    .venv/bin/python docs/research/gemm_sweep/run_sweep.py \\
        --config-json $HOME/.cache/huggingface/hub/models--ig1--Qwen3.5-27B-NVFP4/snapshots/*/config.json \\
        --output benchmarks/nvllm/traces/gemm_sweep_sm120/2026-04-21-qwen35-27b/microbench.csv
"""
from __future__ import annotations

import argparse
import glob
import json
import re
import subprocess
import sys
import time
from pathlib import Path


def load_shapes(config_json: Path) -> list[dict]:
    cfg = json.loads(config_json.read_text())
    tc = cfg["text_config"]
    H = tc["hidden_size"]
    nh = tc["num_attention_heads"]
    nkv = tc["num_key_value_heads"]
    hd = tc["head_dim"]
    I = tc["intermediate_size"]
    return [
        {"name": "qkv_proj",     "N": (nh + 2 * nkv) * hd, "K": H},
        {"name": "o_proj",       "N": H,                   "K": nh * hd},
        {"name": "gate_up_proj", "N": 2 * I,               "K": H},
        {"name": "down_proj",    "N": H,                   "K": I},
    ]


def load_config_names(hpp_path: Path) -> list[str]:
    text = hpp_path.read_text()
    names = re.findall(r"^struct (Cfg_\S+) \{", text, flags=re.M)
    names.append("smoke_M256")
    return names


def run_batch(calls: list[tuple]) -> list[str]:
    """Run a batch of microbench invocations in a single container (for speed).

    calls is a list of (config_name, M, N, K).
    Returns the list of CSV lines (one per call, in order).
    """
    # Compose a shell script that runs all calls in one container.
    # Each call prints one CSV line; unknown configs exit non-zero but
    # we wrap with || echo sentinel so the loop continues.
    lines = []
    for cfg, M, N, K in calls:
        lines.append(
            f'./build/gemm_microbench {cfg} {M} {N} {K} '
            f'|| echo "{cfg},{M},{N},{K},-9.0"'
        )
    script = "\n".join(lines)

    proc = subprocess.run(
        [
            "docker", "run", "--rm", "--gpus", "all",
            "-v", "/home/natfii/docker/nvllm:/workspace",
            "-v", f"{Path.home()}/.cache/huggingface:/root/.cache/huggingface",
            "-w", "/workspace/docs/research/gemm_sweep/microbench",
            "--entrypoint", "bash",
            "nvllm:gb10", "-c", script,
        ],
        capture_output=True, text=True, timeout=600,
    )
    if proc.returncode != 0 and not proc.stdout.strip():
        # Complete failure (not just per-call errors)
        print(f"BATCH FAILED: {proc.stderr[:500]}", file=sys.stderr)
        raise RuntimeError(f"batch failed: {proc.stderr[:500]}")

    return [l for l in proc.stdout.splitlines() if l.strip() and "," in l]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config-json", type=str, required=True,
                    help="Path to Qwen3.5-27B config.json (globs allowed)")
    ap.add_argument("--configs-hpp", type=Path,
                    default=Path("docs/research/gemm_sweep/microbench/configs_generated.hpp"))
    ap.add_argument("--output", type=Path,
                    default=Path("benchmarks/nvllm/traces/gemm_sweep_sm120/2026-04-21-qwen35-27b/microbench.csv"))
    ap.add_argument("--m-values", type=int, nargs="+",
                    default=[1, 2, 4, 8, 16, 32, 64, 128, 192, 256])
    ap.add_argument("--batch-size", type=int, default=30,
                    help="Number of microbench calls per docker invocation")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print plan and exit without running")
    args = ap.parse_args()

    # Resolve glob in --config-json
    matches = glob.glob(args.config_json)
    if not matches:
        print(f"ERROR: config.json not found: {args.config_json}", file=sys.stderr)
        sys.exit(1)
    config_json = Path(matches[0])
    print(f"Config.json: {config_json}")

    shapes = load_shapes(config_json)
    configs = load_config_names(args.configs_hpp)
    total = len(shapes) * len(args.m_values) * len(configs)
    print(f"Sweep plan: {len(shapes)} shapes x {len(args.m_values)} M x {len(configs)} configs = {total} runs")
    print(f"  shapes: {[s['name'] for s in shapes]}")
    for s in shapes:
        print(f"    {s['name']}: N={s['N']}, K={s['K']}")
    print(f"  M values: {args.m_values}")
    print(f"  configs: {len(configs)} (first 3: {configs[:3]})")
    print(f"  output: {args.output}")

    if args.dry_run:
        print("(dry-run, exiting)")
        return

    args.output.parent.mkdir(parents=True, exist_ok=True)

    # Build the full call list
    all_calls = []
    for sh in shapes:
        for M in args.m_values:
            for cfg in configs:
                all_calls.append((cfg, M, sh["N"], sh["K"], sh["name"]))

    # Run in batches, aggregate
    start = time.time()
    with args.output.open("w") as out:
        out.write("config_id,shape,M,N,K,min_us\n")
        i = 0
        while i < len(all_calls):
            batch = all_calls[i:i + args.batch_size]
            batch_calls = [(c[0], c[1], c[2], c[3]) for c in batch]  # strip shape name for the binary
            batch_names = [c[4] for c in batch]
            lines = run_batch(batch_calls)
            # Re-emit with shape column
            for line, shape_name in zip(lines, batch_names):
                parts = line.split(",")
                # binary emits: <cfg>,<M>,<N>,<K>,<us>
                if len(parts) != 5:
                    print(f"  WARN: malformed line: {line!r}", file=sys.stderr)
                    continue
                cfg, M, N, K, us = parts
                out.write(f"{cfg},{shape_name},{M},{N},{K},{us}\n")
                out.flush()
            i += args.batch_size
            elapsed = time.time() - start
            rate = i / elapsed if elapsed > 0 else 0
            eta = (len(all_calls) - i) / rate if rate > 0 else 0
            print(f"  [{i}/{len(all_calls)}] elapsed={elapsed:.1f}s  rate={rate:.1f}/s  eta={eta:.0f}s", flush=True)

    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
