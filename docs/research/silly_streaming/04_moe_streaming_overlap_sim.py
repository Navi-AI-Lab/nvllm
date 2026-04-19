#!/usr/bin/env python3
"""Silly Streaming microbench #4 — MoE expert streaming overlap simulation.

Pure simulation (no hardware required). Given a target model's MoE shape
and assumed bandwidths for NVMe, HW-DE, and compute, decide whether
expert streaming fits in the per-layer compute budget at decode time.

Outputs a verdict per model so we know which frontier MoE is worth
prototyping first and which are unreachable at current bandwidth
assumptions.

Run (not yet — stdlib only, no install needed):
    .venv/bin/python docs/research/silly_streaming/04_moe_streaming_overlap_sim.py \\
        --model kimi-k2.5

All registry numbers below should be sanity-checked against each model's
published config.json before citing in a report; layer counts are the
coarsest estimate.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass

# Coarse MoE specs for April 2026 frontier models. VERIFY before citing.
MODEL_REGISTRY: dict[str, dict[str, float]] = {
    "kimi-k2.5": {
        "total_params_b": 1040.0,
        "active_params_b": 32.0,
        "layers": 61.0,
        "num_experts": 384.0,
        "experts_per_token": 8.0,
    },
    "glm-5.1": {
        "total_params_b": 744.0,
        "active_params_b": 40.0,
        "layers": 80.0,
        "num_experts": 256.0,
        "experts_per_token": 8.0,
    },
    "deepseek-v4": {
        "total_params_b": 1000.0,
        "active_params_b": 37.0,
        "layers": 61.0,
        "num_experts": 256.0,
        "experts_per_token": 8.0,
    },
    "qwen3.6-35b-a3b": {
        "total_params_b": 35.0,
        "active_params_b": 3.0,
        "layers": 40.0,
        "num_experts": 256.0,
        "experts_per_token": 9.0,
    },
}


@dataclass
class SimConfig:
    model: str
    nvme_gbs: float
    hw_de_gbs: float
    compression_ratio: float
    compute_bw_gbs: float
    resident_expert_fraction: float


@dataclass
class SimResult:
    model: str
    per_layer_active_fp4_mb: float
    per_layer_streamed_compressed_mb: float
    stream_ms_per_layer: float
    decomp_ms_per_layer: float
    effective_stream_ms_per_layer: float
    compute_ms_per_layer: float
    overlap_fraction: float
    net_latency_penalty_x: float
    verdict: str


def simulate(cfg: SimConfig) -> SimResult:
    spec = MODEL_REGISTRY[cfg.model]
    active_b = spec["active_params_b"]
    layers = spec["layers"]

    # NVFP4 ≈ 0.5 bytes/param.
    active_fp4_mb_per_layer = (active_b / layers) * 0.5 * 1024.0

    # Only the non-resident fraction is streamed.
    stream_fraction = max(0.0, 1.0 - cfg.resident_expert_fraction)
    streamed_mb = active_fp4_mb_per_layer * stream_fraction
    compressed_mb = streamed_mb / max(cfg.compression_ratio, 1e-6)

    stream_ms = (compressed_mb / 1024.0) / cfg.nvme_gbs * 1000.0
    decomp_ms = (compressed_mb / 1024.0) / cfg.hw_de_gbs * 1000.0
    # Stream and decomp pipeline; the slower stage dominates.
    effective_stream_ms = max(stream_ms, decomp_ms)

    # Compute time per layer: proxy = active weight bytes / compute bandwidth.
    compute_ms = (active_fp4_mb_per_layer / 1024.0) / cfg.compute_bw_gbs * 1000.0

    if max(effective_stream_ms, compute_ms) <= 0:
        overlap_fraction = 0.0
    else:
        overlap_fraction = min(effective_stream_ms, compute_ms) / max(
            effective_stream_ms, compute_ms
        )

    # Penalty vs fully-resident decode: bounded below by 1.0.
    penalty = max(effective_stream_ms / max(compute_ms, 1e-6), 1.0)

    if penalty < 1.3:
        verdict = "feasible — stream hides under compute with healthy margin"
    elif penalty < 2.0:
        verdict = "tight — stream dominates but not crippling; tune prefetch"
    elif penalty < 3.5:
        verdict = "marginal — significant latency penalty; only if no alternative"
    else:
        verdict = "not feasible at these bandwidth assumptions"

    return SimResult(
        model=cfg.model,
        per_layer_active_fp4_mb=active_fp4_mb_per_layer,
        per_layer_streamed_compressed_mb=compressed_mb,
        stream_ms_per_layer=stream_ms,
        decomp_ms_per_layer=decomp_ms,
        effective_stream_ms_per_layer=effective_stream_ms,
        compute_ms_per_layer=compute_ms,
        overlap_fraction=overlap_fraction,
        net_latency_penalty_x=penalty,
        verdict=verdict,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Silly Streaming #4: MoE expert streaming overlap simulation.",
    )
    parser.add_argument("--model", choices=sorted(MODEL_REGISTRY.keys()), required=True)
    parser.add_argument(
        "--nvme-gbs",
        type=float,
        default=6.5,
        help="Effective NVMe read bandwidth (GB/s). Gen4 x4 ~7 theoretical.",
    )
    parser.add_argument(
        "--hw-de-gbs",
        type=float,
        default=60.0,
        help="HW-DE decomp throughput. Use ~20 to simulate SW fallback.",
    )
    parser.add_argument("--compression-ratio", type=float, default=1.15)
    parser.add_argument(
        "--compute-bw-gbs",
        type=float,
        default=200.0,
        help="Proxy for per-layer compute-bound bandwidth. GB10 ~273 peak.",
    )
    parser.add_argument(
        "--resident-expert-fraction",
        type=float,
        default=0.7,
        help="Fraction of experts kept hot in unified memory.",
    )
    args = parser.parse_args(argv)

    cfg = SimConfig(
        model=args.model,
        nvme_gbs=args.nvme_gbs,
        hw_de_gbs=args.hw_de_gbs,
        compression_ratio=args.compression_ratio,
        compute_bw_gbs=args.compute_bw_gbs,
        resident_expert_fraction=args.resident_expert_fraction,
    )
    result = simulate(cfg)
    print(json.dumps({"config": asdict(cfg), "result": asdict(result)}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
