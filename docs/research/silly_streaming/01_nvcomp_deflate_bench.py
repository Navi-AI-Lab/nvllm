#!/usr/bin/env python3
"""Silly Streaming microbench #1 — nvCOMP Deflate decompression throughput on GB10.

Answers the gatekeeper question: is the Blackwell Hardware Decompress Engine
(HW-DE) actually exposed on GB10 (SM120/121), or does nvCOMP silently fall
back to software (SM) decompression?

Expected signal:
  - HW-DE active:    ~50-80 GB/s decomp, low SM utilization during decode
  - SW fallback:     ~15-25 GB/s decomp, SMs saturated during decode

The script is deliberately codec-agnostic so you can flip between Deflate,
Snappy, and Gzip (the three HW-DE-accelerated codecs per nvcomp 4.2.0
release notes) to triangulate.

Install (one-time):
    uv pip install kvikio-cu13 cupy-cuda13x

Run (not yet — exists for the next session):
    .venv/bin/python docs/research/silly_streaming/01_nvcomp_deflate_bench.py \\
        --size-gb 1 --iterations 5 --codec Deflate

Notes:
  - API surface assumed: kvikio.nvcomp_codec.NvCompBatchCodec; verify at run
    time before relying on it for the real bench.
  - SM utilization is best observed with `nsys profile --gpu-metrics-device=0`
    wrapping the script; this file only reports decomp throughput.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class BenchResult:
    codec: str
    buffer_kind: str
    input_size_bytes: int
    compressed_size_bytes: int
    compression_ratio: float
    decomp_gbs_samples: list[float]
    decomp_gbs_mean: float
    decomp_gbs_min: float
    decomp_gbs_max: float


def make_buffer(size_bytes: int, weights_path: Path | None):
    import cupy as cp

    if weights_path is not None:
        raw = weights_path.read_bytes()[:size_bytes]
        return cp.asarray(bytearray(raw), dtype=cp.uint8), "nvfp4_weights"

    rng = cp.random.default_rng(seed=0)
    return (
        rng.integers(0, 256, size=size_bytes, dtype=cp.uint8),
        "uniform_random",
    )


def run(
    size_bytes: int,
    iterations: int,
    codec_name: str,
    weights_path: Path | None,
) -> BenchResult:
    import cupy as cp
    from kvikio.nvcomp_codec import NvCompBatchCodec

    device_input, buffer_kind = make_buffer(size_bytes, weights_path)

    codec = NvCompBatchCodec(codec_name)
    compressed = codec.encode([device_input])
    compressed_size = sum(int(c.nbytes) for c in compressed)
    ratio = size_bytes / max(compressed_size, 1)

    # Warm-up pass to amortize JIT / allocator setup.
    _ = codec.decode(compressed)
    cp.cuda.runtime.deviceSynchronize()

    samples: list[float] = []
    for _ in range(iterations):
        start = cp.cuda.Event()
        stop = cp.cuda.Event()
        start.record()
        _ = codec.decode(compressed)
        stop.record()
        stop.synchronize()
        ms = cp.cuda.get_elapsed_time(start, stop)
        gbs = size_bytes / (1 << 30) / (ms / 1000.0)
        samples.append(gbs)

    return BenchResult(
        codec=codec_name,
        buffer_kind=buffer_kind,
        input_size_bytes=size_bytes,
        compressed_size_bytes=compressed_size,
        compression_ratio=ratio,
        decomp_gbs_samples=samples,
        decomp_gbs_mean=sum(samples) / len(samples),
        decomp_gbs_min=min(samples),
        decomp_gbs_max=max(samples),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Silly Streaming #1: nvCOMP decomp throughput on GB10.",
    )
    parser.add_argument("--size-gb", type=float, default=1.0)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument(
        "--codec",
        choices=["Deflate", "Snappy", "Gzip"],
        default="Deflate",
        help="HW-DE accelerated codecs per nvcomp 4.2.0.",
    )
    parser.add_argument(
        "--weights-path",
        type=Path,
        default=None,
        help="Optional path to an NVFP4 tensor blob for real-weight entropy.",
    )
    args = parser.parse_args(argv)

    if args.size_gb <= 0 or args.iterations <= 0:
        print("ERROR: --size-gb and --iterations must be positive", file=sys.stderr)
        return 2

    size_bytes = int(args.size_gb * (1 << 30))
    result = run(size_bytes, args.iterations, args.codec, args.weights_path)
    print(json.dumps(asdict(result), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
