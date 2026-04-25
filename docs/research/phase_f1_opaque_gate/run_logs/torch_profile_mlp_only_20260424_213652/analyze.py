"""Top-kernel summary from a vLLM torch profiler trace.

Usage:
    .venv/bin/python analyze.py path/to/trace.json.gz
"""

import gzip
import json
import sys
from collections import defaultdict
from pathlib import Path


def load_trace(path: Path) -> dict:
    if path.suffix == ".gz":
        with gzip.open(path, "rt") as f:
            return json.load(f)
    with open(path) as f:
        return json.load(f)


def summarize(trace: dict, top: int = 30) -> None:
    events = trace.get("traceEvents", [])
    print(f"  total events: {len(events)}")
    print()

    # Group device-side kernel events by name. Heuristic:
    # cat == "kernel" OR cat == "gpu_op" — these are CUDA kernel launches.
    cuda_dur = defaultdict(int)
    cuda_count = defaultdict(int)
    cpu_dur = defaultdict(int)
    cpu_count = defaultdict(int)
    for ev in events:
        if ev.get("ph") != "X":
            continue
        name = ev.get("name", "?")
        dur = ev.get("dur", 0)
        cat = ev.get("cat", "")
        if cat == "kernel" or cat == "gpu_op" or "Kernel" in name:
            cuda_dur[name] += dur
            cuda_count[name] += 1
        elif cat in ("cpu_op", "Runtime", "user_annotation"):
            cpu_dur[name] += dur
            cpu_count[name] += 1

    print("=== TOP CUDA KERNELS (sorted by total μs) ===")
    print(f"{'kernel':<70} {'total_us':>12} {'n':>6} {'avg_us':>10}")
    rows = sorted(cuda_dur.items(), key=lambda kv: -kv[1])
    for name, total in rows[:top]:
        n = cuda_count[name]
        avg = total / n if n else 0
        print(f"{name[:70]:<70} {total:>12} {n:>6} {avg:>10.1f}")

    print()
    print("=== TOP CPU OPS (sorted by total μs) ===")
    print(f"{'op':<70} {'total_us':>12} {'n':>6} {'avg_us':>10}")
    rows = sorted(cpu_dur.items(), key=lambda kv: -kv[1])
    for name, total in rows[:top]:
        n = cpu_count[name]
        avg = total / n if n else 0
        print(f"{name[:70]:<70} {total:>12} {n:>6} {avg:>10.1f}")


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    path = Path(sys.argv[1])
    print(f"Loading {path} ({path.stat().st_size / 1024 / 1024:.1f} MiB)...")
    trace = load_trace(path)
    summarize(trace)


if __name__ == "__main__":
    main()
