#!/usr/bin/env python3
"""Silly Streaming microbench #3 — NVFP4 weight compression-ratio survey.

Hypothesis: already-quantized NVFP4 mantissas are high-entropy (~3.9 bits
of information per 4-bit code) so Deflate yields ~1.0-1.1x at best. Block
scales (FP8 e4m3 or FP32 per-block) have local structure and should
compress much better (~2-3x). Metadata (shape, names) is trivial.

The per-class breakdown tells us whether splitting the weights into
separate streams (mantissa vs scale) would meaningfully improve the
overall streaming budget, or whether a single Deflate pass is good
enough.

Runs against any safetensors model directory — does not need NVFP4
specifically, but is calibrated for that data layout.

Install (one-time):
    uv pip install safetensors torch

Run (not yet):
    cd docs/research/silly_streaming && \\
      .venv/bin/python 03_nvfp4_compression_ratio_survey.py \\
        --model ~/models/Qwen3.5-27B-NVFP4-Opus-GB10
"""

from __future__ import annotations

import argparse
import json
import sys
import zlib
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class ClassStats:
    class_name: str
    uncompressed_bytes: int
    compressed_bytes: int
    ratio: float
    tensor_count: int


def classify_tensor(name: str) -> str:
    n = name.lower()
    if "_scale" in n or "scale_2" in n or "input_scale" in n or "weight_scale" in n:
        return "scale"
    if "weight" in n:
        return "weight_mantissa"
    if "bias" in n:
        return "bias"
    if "norm" in n:
        return "norm"
    return "metadata"


def survey(model_path: Path, deflate_level: int) -> dict[str, ClassStats]:
    try:
        from safetensors import safe_open
    except ImportError:
        print(
            "ERROR: safetensors not installed. `uv pip install safetensors`",
            file=sys.stderr,
        )
        sys.exit(1)

    shards = sorted(model_path.glob("*.safetensors"))
    if not shards:
        print(f"ERROR: no .safetensors files in {model_path}", file=sys.stderr)
        sys.exit(2)

    # [uncompressed_bytes, compressed_bytes, count]
    totals: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0])

    for shard in shards:
        with safe_open(shard, framework="pt") as f:
            for name in f.keys():  # noqa: SIM118 — safe_open is not a dict
                tensor = f.get_tensor(name)
                raw = tensor.cpu().contiguous().view(-1).numpy().tobytes()
                compressed = zlib.compress(raw, level=deflate_level)
                klass = classify_tensor(name)
                totals[klass][0] += len(raw)
                totals[klass][1] += len(compressed)
                totals[klass][2] += 1

    return {
        klass: ClassStats(
            class_name=klass,
            uncompressed_bytes=u,
            compressed_bytes=c,
            ratio=u / max(c, 1),
            tensor_count=count,
        )
        for klass, (u, c, count) in totals.items()
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Silly Streaming #3: per-class compression ratio on safetensors.",
    )
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--deflate-level", type=int, default=6)
    args = parser.parse_args(argv)

    if not args.model.is_dir():
        print(f"ERROR: {args.model} is not a directory", file=sys.stderr)
        return 2

    stats = survey(args.model, args.deflate_level)
    total_u = sum(s.uncompressed_bytes for s in stats.values())
    total_c = sum(s.compressed_bytes for s in stats.values())

    out = {
        "by_class": {k: asdict(v) for k, v in stats.items()},
        "overall": {
            "uncompressed_bytes": total_u,
            "compressed_bytes": total_c,
            "ratio": total_u / max(total_c, 1),
        },
    }
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
