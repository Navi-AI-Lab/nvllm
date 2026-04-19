#!/usr/bin/env python3
"""Silly Streaming microbench #2 — cuFile vs mmap on GB10 unified memory.

Hypothesis: on GB10 there is no CPU bounce buffer to bypass (GPU memory is
system memory), so cuFile should NOT dramatically outperform plain mmap +
madvise(SEQUENTIAL). We measure the actual delta so we know how much of
the eventual streaming win comes from the I/O path alone (before
compression / expert prediction).

Expected signal:
  - cuFile ≈ mmap within 10-15%: unified-memory hypothesis holds, pick
    whichever API is cleaner to integrate.
  - cuFile >> mmap:              unexpected; something is wrong with the
    mmap path (maybe page-cache thrash) — investigate before trusting it.

Important: the Linux page cache will mask cold-read bandwidth on repeated
runs. Drop caches between runs with:
    sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'

Install (one-time):
    uv pip install kvikio-cu13 cupy-cuda13x

Preparation (create a large test file on NVMe):
    dd if=/dev/urandom of=/tmp/silly_streaming_4g.bin bs=1M count=4096

Run (not yet):
    .venv/bin/python docs/research/silly_streaming/02_cufile_vs_mmap_bench.py \\
        --path /tmp/silly_streaming_4g.bin --size-gb 4
"""

from __future__ import annotations

import argparse
import json
import mmap
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class IoResult:
    method: str
    bytes_read: int
    wall_ms: float
    gbs: float


def _gbs(size_bytes: int, wall_ms: float) -> float:
    return size_bytes / (1 << 30) / (wall_ms / 1000.0)


def bench_posix_read(path: Path, size: int) -> IoResult:
    buf = bytearray(size)
    start = time.perf_counter()
    with path.open("rb") as f:
        f.readinto(buf)
    wall_ms = (time.perf_counter() - start) * 1000.0
    return IoResult("posix_read", size, wall_ms, _gbs(size, wall_ms))


def bench_mmap(path: Path, size: int) -> IoResult:
    start = time.perf_counter()
    with path.open("rb") as f, mmap.mmap(f.fileno(), size, prot=mmap.PROT_READ) as mm:
        if hasattr(mm, "madvise"):
            mm.madvise(mmap.MADV_SEQUENTIAL)
        # Force page-in of all pages by touching one byte per page.
        page = mmap.PAGESIZE
        total = 0
        for offset in range(0, size, page):
            total += mm[offset]
        _ = total  # prevent optimization
    wall_ms = (time.perf_counter() - start) * 1000.0
    return IoResult("mmap_sequential", size, wall_ms, _gbs(size, wall_ms))


def bench_cufile(path: Path, size: int, qdepth: int) -> IoResult:
    import cupy as cp
    from kvikio import CuFile

    dst = cp.empty(size, dtype=cp.uint8)
    start = time.perf_counter()
    with CuFile(str(path), "r") as f:
        chunk = size // qdepth
        futures = []
        for i in range(qdepth):
            offset = i * chunk
            n = chunk if i < qdepth - 1 else size - offset
            futures.append(f.pread(dst[offset : offset + n], n, offset))
        for fut in futures:
            fut.get()
    cp.cuda.runtime.deviceSynchronize()
    wall_ms = (time.perf_counter() - start) * 1000.0
    return IoResult(f"cufile_qdepth{qdepth}", size, wall_ms, _gbs(size, wall_ms))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Silly Streaming #2: cuFile vs mmap on unified memory.",
    )
    parser.add_argument("--path", type=Path, required=True)
    parser.add_argument("--size-gb", type=float, default=4.0)
    parser.add_argument("--qdepth", type=int, default=16)
    parser.add_argument(
        "--skip-cufile",
        action="store_true",
        help="Skip cuFile; useful if kvikio/cupy aren't installed yet.",
    )
    args = parser.parse_args(argv)

    size = int(args.size_gb * (1 << 30))
    if not args.path.exists() or args.path.stat().st_size < size:
        print(
            f"ERROR: {args.path} must exist and be >= {args.size_gb} GB",
            file=sys.stderr,
        )
        return 2

    results: list[IoResult] = [
        bench_posix_read(args.path, size),
        bench_mmap(args.path, size),
    ]
    if not args.skip_cufile:
        results.append(bench_cufile(args.path, size, args.qdepth))

    print(json.dumps([asdict(r) for r in results], indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
