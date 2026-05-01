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
import hashlib
import json
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
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


CONTAINER_NAME = "nvllm"
DEFAULT_IMAGE = "nvllm:gb10"
PHASE1_PROMPT = "The capital of France is"
PHASE1_MAX_TOKENS = 8


def expected_cache_files() -> list[dict[str, str]]:
    """Return the 4 expected files (relative_path may be a glob — torch's
    AOT artifact path includes a hash that varies per torch/vLLM version;
    Phase 1 resolves the actual path post-bootstrap)."""
    return [
        {"role": "aot_model",
         "relative_path_glob":
             "torch_compile_cache/torch_aot_compile/*/rank_0_0/model"},
        {"role": "computation_graph",
         "relative_path_glob":
             "torch_compile_cache/*/rank_0_0/backbone/computation_graph.py"},
        {"role": "cache_key_factors",
         "relative_path_glob":
             "torch_compile_cache/*/rank_0_0/backbone/cache_key_factors.json"},
        {"role": "model_info",
         "relative_path_glob": "modelinfos/*.json"},
    ]


def build_phase1_docker_args(
    *, container_name: str, image: str,
    hf_cache: Path, flashinfer_cache: Path, cute_compile_host_cache: Path,
    staging_dir: Path, model_id: str, kv_cache_dtype: str,
    attention_backend: str, max_model_len: int, max_num_seqs: int,
    max_num_batched_tokens: int, cute_phase_e_layers: str,
) -> list[str]:
    """Build the docker run argv for the RW Phase-1 container.

    Mirrors scripts/serve-cute-full.sh defaults but with probes OFF and
    a RW mount of staging_dir at /root/.cache/vllm.
    """
    return [
        "docker", "run", "-d",
        "--name", container_name,
        "--gpus", "all",
        "--ipc=host",
        "--network", "host",
        "-v", f"{hf_cache}:/root/.cache/huggingface",
        "-v", f"{flashinfer_cache}:/root/.cache/flashinfer",
        "-v", f"{cute_compile_host_cache}:/opt/vllm/kernel_cache",
        "-v", f"{staging_dir}:/root/.cache/vllm",
        "-e", "B12X_CUTE_COMPILE_DISK_CACHE=1",
        "-e", "B12X_CUTE_COMPILE_CACHE_DIR=/opt/vllm/kernel_cache",
        "-e", "VLLM_NVFP4_GEMM_BACKEND=cutlass",
        "-e", "VLLM_ALLOW_LONG_MAX_MODEL_LEN=1",
        "-e", "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True",
        "-e", "CUTE_MLP_FUSION=1",
        "-e", "CUTE_ATTN_FUSION=1",
        "-e", "CUTE_BETA_MIN_FREE_GB=8",
        "-e", "CUTE_PHASE_E_FUSION=1",
        "-e", f"CUTE_PHASE_E_LAYERS={cute_phase_e_layers}",
        "-e", "CUTE_PHASE_E_FALLBACK_RAISE=1",
        "-e", "CUTE_FULL_GRAPH_PROBE=0",
        "-e", "CUTE_WO_RESET_LOG=0",
        "-e", "CUTE_DISPATCH_AUDIT=0",
        image,
        "serve",
        "--model", model_id,
        "--served-model-name", "default",
        "--host", "0.0.0.0", "--port", "8000",
        "--kv-cache-dtype", kv_cache_dtype,
        "--attention-backend", attention_backend,
        "--max-model-len", str(max_model_len),
        "--max-num-seqs", str(max_num_seqs),
        "--language-model-only",
        "--limit-mm-per-prompt", '{"image": 0, "video": 0}',
        "--mamba-cache-mode", "align",
        "--trust-remote-code",
        "--gpu-memory-utilization", "0.65",
        "--max-num-batched-tokens", str(max_num_batched_tokens),
        "--kernel-config", '{"enable_flashinfer_autotune":false}',
        "--compilation-config",
        '{"cudagraph_mode":"FULL_AND_PIECEWISE","cudagraph_capture_sizes":[1]}',
    ]


def _docker_run(args: list[str]) -> None:
    r = subprocess.run(args, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"docker run failed: {r.stderr}")


def _docker_stop(name: str, timeout: int = 60) -> str:
    """Stop container gracefully; capture full container log. Returns log text."""
    log = subprocess.run(["docker", "logs", name],
                          capture_output=True, text=True).stdout + \
          subprocess.run(["docker", "logs", name],
                          capture_output=True, text=True).stderr
    subprocess.run(["docker", "stop", "-t", str(timeout), name],
                   capture_output=True)
    subprocess.run(["docker", "rm", "-f", name], capture_output=True)
    return log


def _poll_models(host: str, port: int, timeout_s: int = 600,
                 attempt_max: int = 3) -> None:
    """Block until /v1/models responds 200, with up to attempt_max transient retries."""
    url = f"http://{host}:{port}/v1/models"
    deadline = time.time() + timeout_s
    attempt = 0
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=10) as r:
                if r.status == 200:
                    return
        except (urllib.error.URLError, ConnectionError, TimeoutError):
            attempt += 1
            if attempt > attempt_max * 50:  # ~1 retry per 2s
                raise
            time.sleep(2)
    raise TimeoutError(f"/v1/models did not respond within {timeout_s}s")


def _trigger_completion(host: str, port: int) -> None:
    """One fixed completion to force prefill + decode + AOT artifact write."""
    body = json.dumps({
        "model": "default",
        "prompt": PHASE1_PROMPT,
        "max_tokens": PHASE1_MAX_TOKENS,
        "temperature": 0,
    }).encode()
    req = urllib.request.Request(
        f"http://{host}:{port}/v1/completions",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=600) as r:
        _ = r.read()


def _resolve_glob(staging: Path, glob: str) -> Path | None:
    matches = list(staging.glob(glob))
    return matches[0] if len(matches) == 1 else None


def _sha256_size(p: Path) -> tuple[str, int]:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest(), p.stat().st_size


def phase1_bootstrap(
    *, staging_dir: Path, image: str, hf_cache: Path, flashinfer_cache: Path,
    cute_compile_host_cache: Path, model_id: str, kv_cache_dtype: str = "fp8_e4m3",
    attention_backend: str = "CUTE_PAGED", max_model_len: int = 16384,
    max_num_seqs: int = 1, max_num_batched_tokens: int = 65536,
    cute_phase_e_layers: str = "0,1,2,3,4,5,6,7",
    bootstrap_log_path: Path,
) -> dict[str, Any]:
    """Run Phase 1; return {'aot_sha': str, 'aot_size': int, 'resolved_paths': {role: rel_path}}."""
    # Clean staging.
    if staging_dir.exists():
        shutil.rmtree(staging_dir)
    staging_dir.mkdir(parents=True, exist_ok=True)

    # Make sure no stray container is running.
    subprocess.run(["docker", "rm", "-f", CONTAINER_NAME],
                   capture_output=True)

    args = build_phase1_docker_args(
        container_name=CONTAINER_NAME, image=image,
        hf_cache=hf_cache, flashinfer_cache=flashinfer_cache,
        cute_compile_host_cache=cute_compile_host_cache,
        staging_dir=staging_dir, model_id=model_id,
        kv_cache_dtype=kv_cache_dtype, attention_backend=attention_backend,
        max_model_len=max_model_len, max_num_seqs=max_num_seqs,
        max_num_batched_tokens=max_num_batched_tokens,
        cute_phase_e_layers=cute_phase_e_layers,
    )
    print(f"[bless/phase1] docker run (RW staging={staging_dir})", flush=True)
    _docker_run(args)

    try:
        _poll_models("localhost", 8000, timeout_s=600)
        print("[bless/phase1] /v1/models ready, triggering completion", flush=True)
        _trigger_completion("localhost", 8000)
        print("[bless/phase1] completion done; stopping container "
              "(graceful, allows torch to flush AOT)", flush=True)
    finally:
        log = _docker_stop(CONTAINER_NAME, timeout=60)
        bootstrap_log_path.parent.mkdir(parents=True, exist_ok=True)
        bootstrap_log_path.write_text(log)

    # Resolve the 4 expected files via globs (path-suffix hashes vary).
    resolved = {}
    for ent in expected_cache_files():
        p = _resolve_glob(staging_dir, ent["relative_path_glob"])
        if p is None:
            raise RuntimeError(
                f"[bless/phase1] expected file role={ent['role']} not found "
                f"(or non-unique) under {staging_dir} with glob "
                f"{ent['relative_path_glob']}"
            )
        if p.stat().st_size == 0:
            raise RuntimeError(
                f"[bless/phase1] expected file role={ent['role']} is zero-byte: {p}"
            )
        resolved[ent["role"]] = str(p.relative_to(staging_dir))

    aot_path = staging_dir / resolved["aot_model"]
    aot_sha, aot_size = _sha256_size(aot_path)
    print(f"[bless/phase1] AOT model: {aot_path.relative_to(staging_dir)} "
          f"({aot_size} B, sha={aot_sha[:12]}…)", flush=True)
    return {
        "aot_sha": aot_sha,
        "aot_size": aot_size,
        "resolved_paths": resolved,
    }


if __name__ == "__main__":
    sys.exit(main())
