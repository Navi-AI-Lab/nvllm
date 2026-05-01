#!/usr/bin/env python
"""Bless candidate torch.compile AOT cache for FULL+β-coop production.

Two-phase flow per spec:
  docs/superpowers/specs/2026-05-01-cute-full-cache-production-workaround-design.md

Phase 1: RW bootstrap → cold compile → AOT artifact populated.
Phase 2: K fresh-container :ro validations with c2_replay_coherence + cache-reuse signals.
Phase 3: accept (atomic install + manifest) or reject (preserve evidence, no manifest).

Invoked by scripts/bless-cute-full-cache.sh with the config-hash already derived.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class TrialResult:
    trial_n: int
    c2_pass: bool
    cache_reused: bool
    aot_sha256_post: str
    c2_json: dict[str, Any] = field(default_factory=dict)
    log_paths: dict[str, str] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return self.c2_pass and self.cache_reused


@dataclass
class BlessConfig:
    config_hash: str
    image_id: str
    hf_revision: str
    rebless: bool
    k_trials: int
    unsafe_dev_trials: bool


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Bless a candidate torch.compile AOT cache for "
                    "FULL+β-coop production serve.",
    )
    p.add_argument("--config-hash", required=True,
                   help="64-char sha256 hex (computed by bless-cute-full-cache.sh).")
    p.add_argument("--image-id", required=True,
                   help="Full Docker image ID (e.g. sha256:d3ddffea3c...).")
    p.add_argument("--hf-revision", required=True,
                   help="Resolved HF model revision sha.")
    p.add_argument("--rebless", action="store_true",
                   help="Allow overwrite of existing manifest (atomic, archived).")
    p.add_argument("--unsafe-trials", type=int, default=None,
                   help="Override K with N<5 (writes unsafe_dev_trials=true; "
                        "production serve refuses such manifests).")
    return p.parse_args(argv)


def _resolve_k_trials(args: argparse.Namespace) -> tuple[int, bool]:
    """Return (k_trials, unsafe_dev_trials)."""
    import os
    env_k = os.environ.get("NVLLM_BLESS_VALIDATION_TRIALS")
    if args.unsafe_trials is not None:
        if args.unsafe_trials >= 5:
            print("ERROR: --unsafe-trials only allowed with N<5; use "
                  "NVLLM_BLESS_VALIDATION_TRIALS to RAISE K", file=sys.stderr)
            sys.exit(2)
        return args.unsafe_trials, True
    if env_k is not None:
        k = int(env_k)
        if k < 5:
            print(f"ERROR: NVLLM_BLESS_VALIDATION_TRIALS={k} < 5 without "
                  "--unsafe-trials. Refusing.", file=sys.stderr)
            sys.exit(2)
        return k, False
    return 5, False


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    k_trials, unsafe = _resolve_k_trials(args)
    cfg = BlessConfig(
        config_hash=args.config_hash,
        image_id=args.image_id,
        hf_revision=args.hf_revision,
        rebless=args.rebless,
        k_trials=k_trials,
        unsafe_dev_trials=unsafe,
    )
    print(f"[bless] config_hash={cfg.config_hash[:7]}... "
          f"K={cfg.k_trials} rebless={cfg.rebless} unsafe={cfg.unsafe_dev_trials}",
          flush=True)
    # Phase 1, 2, 3 wired in subsequent tasks.
    print("[bless] (skeleton — Phase 1/2/3 not yet wired)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
