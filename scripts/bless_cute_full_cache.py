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
    import os
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

    home = Path(os.environ["HOME"])
    cache_root = Path(os.environ.get("NVLLM_BLESSED_CACHE_ROOT",
                                       str(home / ".cache/nvllm")))
    blessed_root = cache_root / "blessed"
    staging_root = cache_root / "staging"
    evidence_root = cache_root / "evidence"
    manifest_root = REPO_ROOT / "docs/blessed-caches"
    staging_dir = staging_root / cfg.config_hash

    blessed_dir = blessed_root / cfg.config_hash
    if blessed_dir.exists() and not cfg.rebless:
        print(f"[bless] manifest+cache already exist for {cfg.config_hash[:7]}…",
              file=sys.stderr)
        print(f"[bless] re-run with --rebless to replace, or delete: {blessed_dir}",
              file=sys.stderr)
        return 1
    previous_manifest = None
    previous_blessed_dir = None
    if cfg.rebless:
        try:
            r = subprocess.run(
                ["bash", "-c",
                 f"source {REPO_ROOT}/scripts/common.sh && "
                 f"nvllm_resolve_blessed_manifest {cfg.config_hash}"],
                capture_output=True, text=True, check=True,
            )
            previous_manifest = Path(r.stdout.strip())
        except subprocess.CalledProcessError:
            previous_manifest = None
        if blessed_dir.exists():
            previous_blessed_dir = blessed_dir

    image = os.environ.get("NVLLM_IMAGE", DEFAULT_IMAGE)
    hf_cache = home / ".cache/huggingface"
    flashinfer_cache = home / ".cache/flashinfer"
    cute_compile_host_cache = Path(
        os.environ.get("CUTE_COMPILE_HOST_CACHE_DIR", "/tmp/nvllm-cute-cache")
    )
    cute_compile_host_cache.mkdir(parents=True, exist_ok=True)

    # Launch config — must match serve-cute-full.sh defaults exactly.
    launch_config: dict[str, Any] = {
        "model_id": os.environ.get("HF_MODEL", "ig1/Qwen3.5-27B-NVFP4"),
        "kv_cache_dtype": "fp8_e4m3",
        "attention_backend": "CUTE_PAGED",
        "cudagraph_mode": "FULL_AND_PIECEWISE",
        "cudagraph_capture_sizes": [1],
        "max_num_seqs": 1,
        "max_model_len": 16384,
        "max_num_batched_tokens": 65536,
        "cute_phase_e_fusion": 1,
        "cute_phase_e_layers": "0,1,2,3,4,5,6,7",
        "cute_phase_e_fallback_raise": 1,
        "cute_full_graph_probe": 0,
        "cute_wo_reset_log": 0,
        "cute_dispatch_audit": 0,
        "cute_mlp_fusion": 1,
        "cute_attn_fusion": 1,
    }
    ts = _dt.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    bless_evidence_dir = evidence_root / f"{cfg.config_hash}_{ts}"

    print(f"[bless] config_hash={cfg.config_hash} K={cfg.k_trials} "
          f"rebless={cfg.rebless}", flush=True)
    print(f"[bless] staging  : {staging_dir}", flush=True)
    print(f"[bless] blessed  : {blessed_dir}", flush=True)
    print(f"[bless] evidence : {bless_evidence_dir}", flush=True)
    print(f"[bless] manifest : {manifest_root}", flush=True)

    # ---- Phase 1 ----
    try:
        ph1 = phase1_bootstrap(
            staging_dir=staging_dir, image=image,
            hf_cache=hf_cache, flashinfer_cache=flashinfer_cache,
            cute_compile_host_cache=cute_compile_host_cache,
            model_id=launch_config["model_id"],
            kv_cache_dtype=launch_config["kv_cache_dtype"],
            attention_backend=launch_config["attention_backend"],
            max_model_len=launch_config["max_model_len"],
            max_num_seqs=launch_config["max_num_seqs"],
            max_num_batched_tokens=launch_config["max_num_batched_tokens"],
            cute_phase_e_layers=launch_config["cute_phase_e_layers"],
            bootstrap_log_path=bless_evidence_dir / "_bless_logs/phase1_bootstrap.log",
        )
    except Exception as e:
        print(f"[bless/phase1] FAIL: {e}", file=sys.stderr)
        reject(staging_dir=staging_dir, evidence_root=evidence_root,
                cfg=cfg, trial_results=[])
        return 2

    # ---- Phase 2 ----
    results = phase2_validate(
        staging_dir=staging_dir, image=image,
        hf_cache=hf_cache, flashinfer_cache=flashinfer_cache,
        cute_compile_host_cache=cute_compile_host_cache,
        model_id=launch_config["model_id"],
        kv_cache_dtype=launch_config["kv_cache_dtype"],
        attention_backend=launch_config["attention_backend"],
        max_model_len=launch_config["max_model_len"],
        max_num_seqs=launch_config["max_num_seqs"],
        max_num_batched_tokens=launch_config["max_num_batched_tokens"],
        cute_phase_e_layers=launch_config["cute_phase_e_layers"],
        aot_relpath=ph1["resolved_paths"]["aot_model"],
        expected_aot_sha=ph1["aot_sha"],
        k_trials=cfg.k_trials,
        evidence_dir=bless_evidence_dir,
    )

    if all(r.passed for r in results):
        print(f"[bless] all {cfg.k_trials} trials PASS — accepting", flush=True)
        manifest_path = accept(
            staging_dir=staging_dir,
            blessed_root=blessed_root,
            manifest_root=manifest_root,
            cfg=cfg,
            resolved_paths=ph1["resolved_paths"],
            trial_results=results,
            launch_config=launch_config,
            archive_root=blessed_root,
            previous_manifest=previous_manifest,
            previous_blessed_dir=previous_blessed_dir,
        )
        print(f"[bless] DONE. To commit:", flush=True)
        print(f"  git add docs/blessed-caches/{manifest_path.name}", flush=True)
        return 0
    else:
        n_fail = sum(1 for r in results if not r.passed)
        print(f"[bless] {n_fail}/{cfg.k_trials} trials FAIL — rejecting",
              file=sys.stderr)
        reject(staging_dir=staging_dir, evidence_root=evidence_root,
                cfg=cfg, trial_results=results)
        return 3


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
    """Stop container gracefully; capture full container log. Returns log text.

    Single docker logs invocation captures both stdout and stderr (combined).
    No -f on rm: a successfully stopped container removes cleanly without it.
    """
    logs = subprocess.run(["docker", "logs", name],
                          capture_output=True, text=True)
    subprocess.run(["docker", "stop", "-t", str(timeout), name],
                   capture_output=True)
    subprocess.run(["docker", "rm", name], capture_output=True)
    return logs.stdout + logs.stderr


def _poll_models(host: str, port: int, timeout_s: int = 600,
                 max_transient_retries: int = 3) -> None:
    """Block until /v1/models responds 200.

    Bounded by:
      - timeout_s: overall wall-clock deadline (raises TimeoutError on expiry).
      - max_transient_retries: number of transient (URLError/ConnectionError/
        TimeoutError) failures tolerated before re-raising. Spec §7.3 caps
        this at 3 — a real boot does NOT flap repeatedly.

    Successful 200 → return. Non-200 status → keep polling (server may still
    be initializing). Sleep 2s between polls.
    """
    url = f"http://{host}:{port}/v1/models"
    deadline = time.time() + timeout_s
    transient_failures = 0
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=10) as r:
                if r.status == 200:
                    return
        except (urllib.error.URLError, ConnectionError, TimeoutError):
            transient_failures += 1
            if transient_failures > max_transient_retries:
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

    # Defensive cleanup of orchestrator-owned leftovers from a prior crash.
    # NOT force-remove: the launcher's preflight (nvllm_refuse_if_container_exists)
    # already refused operator-owned containers. A stop+rm without -f surfaces
    # any unexpected running container as a loud failure rather than silently
    # destroying it.
    subprocess.run(["docker", "stop", "-t", "10", CONTAINER_NAME],
                   capture_output=True)
    subprocess.run(["docker", "rm", CONTAINER_NAME],
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


# The exact reuse log line emitted by torch.compile when AOT cache hits;
# see Z1 evidence
# (docs/research/2026-04-29-full-graph-spike/evidence/2026-04-30-2016-pathb-z1-good-trial-1/docker_logs_full.txt:2484
#  — "Directly load AOT compilation from path /root/.cache/vllm/torch_compile_cache/torch_aot_compile/<hash>/rank_0_0/model").
AOT_LOAD_MARKER = "Directly load AOT compilation from path"
AOT_SAVED_PATTERN = "saved AOT compiled function"


def build_phase2_docker_args(
    *, container_name: str, image: str,
    hf_cache: Path, flashinfer_cache: Path, cute_compile_host_cache: Path,
    staging_dir: Path, model_id: str, kv_cache_dtype: str,
    attention_backend: str, max_model_len: int, max_num_seqs: int,
    max_num_batched_tokens: int, cute_phase_e_layers: str,
) -> list[str]:
    """Same as Phase 1 but :ro mount on staging."""
    args = build_phase1_docker_args(
        container_name=container_name, image=image,
        hf_cache=hf_cache, flashinfer_cache=flashinfer_cache,
        cute_compile_host_cache=cute_compile_host_cache,
        staging_dir=staging_dir, model_id=model_id,
        kv_cache_dtype=kv_cache_dtype, attention_backend=attention_backend,
        max_model_len=max_model_len, max_num_seqs=max_num_seqs,
        max_num_batched_tokens=max_num_batched_tokens,
        cute_phase_e_layers=cute_phase_e_layers,
    )
    # Replace the staging volume arg with :ro variant.
    rw_arg = f"{staging_dir}:/root/.cache/vllm"
    ro_arg = f"{staging_dir}:/root/.cache/vllm:ro"
    return [a if a != rw_arg else ro_arg for a in args]


def classify_cache_reuse(*, container_log: str, sha_pre: str,
                          sha_post: str,
                          expected_aot_path: str) -> tuple[bool, list[str]]:
    """Return (passed, list_of_failure_reasons).

    Cache reuse is signalled by a single log line:
        Directly load AOT compilation from path <expected_aot_path>
    AND no `saved AOT compiled function` lines (which would mean the cache
    was rebuilt) AND the AOT model sha256 unchanged across the trial.
    """
    reasons: list[str] = []
    expected_load_line = f"{AOT_LOAD_MARKER} {expected_aot_path}"
    if expected_load_line not in container_log:
        # Marker absent or pointing at a different path — both are failures.
        if AOT_LOAD_MARKER not in container_log:
            reasons.append(
                f"AOT load marker absent (expected: '{AOT_LOAD_MARKER}'). "
                "Cache miss — torch.compile recompiled instead of reusing."
            )
        else:
            reasons.append(
                "AOT load marker present but path mismatch "
                f"(expected ends with '{expected_aot_path}'). "
                "Different AOT artifact loaded than blessed."
            )
    if AOT_SAVED_PATTERN in container_log:
        reasons.append(
            "'saved AOT compiled function' line present — cache was rebuilt"
        )
    if sha_pre != sha_post:
        reasons.append(
            f"AOT sha drift: pre={sha_pre[:12]}… post={sha_post[:12]}…"
        )
    return (len(reasons) == 0), reasons


def parse_c2_json(json_path: Path) -> tuple[bool, dict[str, Any]]:
    """Return (c2_pass, full_summary_dict)."""
    summary = json.loads(json_path.read_text())
    c2_pass = bool(
        summary.get("same_prompt_pass", False)
        and summary.get("cross_prompt_pass", False)
        and summary.get("same_prompt_unique_count", -1) == 1
    )
    return c2_pass, summary


REPO_ROOT = Path(__file__).resolve().parents[1]


def _run_c2_subprocess(json_out: Path, evidence_dir: Path) -> int:
    repo_root = REPO_ROOT
    c2 = repo_root / "docs/research/2026-04-29-full-graph-spike/c2_replay_coherence.py"
    py = repo_root / ".venv/bin/python"
    r = subprocess.run(
        [str(py), str(c2),
         "--json-out", str(json_out),
         "--evidence-dir", str(evidence_dir)],
        capture_output=True, text=True,
    )
    print(r.stdout)
    if r.stderr:
        print(r.stderr, file=sys.stderr)
    return r.returncode


def phase2_validate(
    *, staging_dir: Path, image: str, hf_cache: Path, flashinfer_cache: Path,
    cute_compile_host_cache: Path, model_id: str, kv_cache_dtype: str,
    attention_backend: str, max_model_len: int, max_num_seqs: int,
    max_num_batched_tokens: int, cute_phase_e_layers: str,
    aot_relpath: str, expected_aot_sha: str,
    k_trials: int, evidence_dir: Path,
) -> list[TrialResult]:
    """Run K fresh-container :ro validation trials."""
    bless_logs = evidence_dir / "_bless_logs"
    bless_logs.mkdir(parents=True, exist_ok=True)
    results: list[TrialResult] = []
    for i in range(1, k_trials + 1):
        print(f"\n[bless/phase2] === Trial {i}/{k_trials} ===", flush=True)
        # Defensive cleanup of orchestrator-owned leftovers from a prior crash;
        # non-force per ea0046dde — refusal of unexpected operator-owned
        # containers is the launcher's job.
        subprocess.run(["docker", "stop", "-t", "10", CONTAINER_NAME],
                       capture_output=True)
        subprocess.run(["docker", "rm", CONTAINER_NAME],
                       capture_output=True)
        args = build_phase2_docker_args(
            container_name=CONTAINER_NAME, image=image,
            hf_cache=hf_cache, flashinfer_cache=flashinfer_cache,
            cute_compile_host_cache=cute_compile_host_cache,
            staging_dir=staging_dir, model_id=model_id,
            kv_cache_dtype=kv_cache_dtype, attention_backend=attention_backend,
            max_model_len=max_model_len, max_num_seqs=max_num_seqs,
            max_num_batched_tokens=max_num_batched_tokens,
            cute_phase_e_layers=cute_phase_e_layers,
        )
        _docker_run(args)
        c2_json_out = bless_logs / f"trial_{i}_c2.json"
        c2_evidence = bless_logs / f"trial_{i}_c2_evidence"
        try:
            _poll_models("localhost", 8000, timeout_s=600)
            c2_rc = _run_c2_subprocess(c2_json_out, c2_evidence)
        finally:
            log = _docker_stop(CONTAINER_NAME, timeout=60)
            (bless_logs / f"trial_{i}_container.log").write_text(log)

        sha_post, _ = _sha256_size(staging_dir / aot_relpath)
        c2_pass, summary = parse_c2_json(c2_json_out) if c2_json_out.exists() \
            else (False, {"error": "c2 json not written"})
        # The container path of the staging dir is /root/.cache/vllm; the AOT
        # marker line embeds the absolute container path, so we reconstruct it.
        expected_container_aot_path = f"/root/.cache/vllm/{aot_relpath}"
        cache_ok, reasons = classify_cache_reuse(
            container_log=log,
            sha_pre=expected_aot_sha, sha_post=sha_post,
            expected_aot_path=expected_container_aot_path,
        )
        tr = TrialResult(
            trial_n=i, c2_pass=c2_pass, cache_reused=cache_ok,
            aot_sha256_post=sha_post, c2_json=summary,
            log_paths={
                "container": str(bless_logs / f"trial_{i}_container.log"),
                "c2_json":   str(c2_json_out),
            },
        )
        if not cache_ok:
            print(f"[bless/phase2] trial {i} cache-reuse FAIL: {reasons}",
                  flush=True)
        if not c2_pass:
            print(f"[bless/phase2] trial {i} c2 FAIL", flush=True)
        results.append(tr)
        print(f"[bless/phase2] trial {i} → "
              f"{'PASS' if tr.passed else 'FAIL'}", flush=True)
    return results


import datetime as _dt


def _human_label_from_config(cfg: BlessConfig, model_id: str,
                              cudagraph_mode: str,
                              cute_phase_e_layers: str) -> str:
    model_short = model_id.split("/")[-1].lower().replace(".", "")
    cgmode_short = "fap" if cudagraph_mode == "FULL_AND_PIECEWISE" else \
                   cudagraph_mode.lower()
    layer_short = "lower" + str(len(cute_phase_e_layers.split(",")))
    img_sha7 = cfg.image_id.replace("sha256:", "")[:7]
    return f"{model_short}_{cgmode_short}_{layer_short}_image-{img_sha7}"


def accept(
    *, staging_dir: Path, blessed_root: Path, manifest_root: Path,
    cfg: BlessConfig, resolved_paths: dict[str, str],
    trial_results: list[TrialResult],
    launch_config: dict[str, Any],
    archive_root: Path | None = None,
    previous_manifest: Path | None = None,
    previous_blessed_dir: Path | None = None,
) -> Path:
    """Build manifest, install staging→blessed, write manifest JSON.

    Re-bless ordering is **promote-then-archive** to keep the prior config
    active until the new one is fully on disk:

      1. Stage new cache as `<blessed_root>/<hash>.new` (rename of staging).
      2. Stage new manifest as `<manifest_root>/<filename>.new` (write).
      3. PROMOTE cache: rename old `<hash>` → `<hash>.old`,
         then rename `<hash>.new` → `<hash>` (small window where neither exists).
      4. PROMOTE manifest: rename old `<filename>` → `<filename>.old`,
         then rename `<filename>.new` → `<filename>`.
      5. ARCHIVE: rename `.old` artifacts under `_archive/` (no active config
         depends on these by the time we get here).

    If any step fails between 3 and 5, operator can recover: `.new` and `.old`
    artifacts are inspectable, and the old config is still functional unless
    promotion completed for one half (cache or manifest) but not the other.
    Recovery in that case is a manual rename — never an unrecoverable loss.

    Return manifest path."""
    blessed_root.mkdir(parents=True, exist_ok=True)
    manifest_root.mkdir(parents=True, exist_ok=True)

    files = []
    for role, rel in resolved_paths.items():
        full = staging_dir / rel
        sha, sz = _sha256_size(full)
        files.append({
            "relative_path": rel, "sha256": sha,
            "size_bytes": sz, "role": role,
        })

    blessed_dir = blessed_root / cfg.config_hash

    aot_sha = next(f["sha256"] for f in files if f["role"] == "aot_model")
    human_label = _human_label_from_config(
        cfg=cfg, model_id=launch_config["model_id"],
        cudagraph_mode=launch_config["cudagraph_mode"],
        cute_phase_e_layers=launch_config["cute_phase_e_layers"],
    )
    manifest_filename = f"{human_label}_{cfg.config_hash[:7]}.json"
    manifest_path = manifest_root / manifest_filename

    archive_paths: dict[str, str] | None = None
    replaces_manifest_filename: str | None = None
    replaces_artifact_sha256: str | None = None
    is_rebless = cfg.rebless and previous_manifest is not None
    if is_rebless:
        assert archive_root is not None, "archive_root required for --rebless"
        prev_manifest_data = json.loads(previous_manifest.read_text())
        prev_aot_sha = next(
            (f["sha256"] for f in prev_manifest_data["files"]
             if f["role"] == "aot_model"), "unknown",
        )
        replaces_manifest_filename = previous_manifest.name
        replaces_artifact_sha256 = prev_aot_sha

    manifest = {
        "schema_version": 1,
        "config_hash": cfg.config_hash,
        "human_label": human_label,
        "blessed_at": _dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "blessed_by": _blessed_by(),
        "blessed_image_id": cfg.image_id,
        "config": {
            **{k: v for k, v in launch_config.items()
               if k != "human_label_inputs"},
            "model_revision_resolved": cfg.hf_revision,
        },
        "mount": {
            "host_path": str(blessed_dir),
            "container_path": "/root/.cache/vllm",
            "mode": "ro",
        },
        "files": files,
        "validation": {
            "trials": len(trial_results),
            "trials_passed": sum(1 for r in trial_results if r.passed),
            "c2_replay_n_replays_per_trial": 8,
            "criterion": ("all-K trials: c2 unique=1 AND cross_prompt=independent "
                          "AND cache_reused=true"),
            "cache_reuse_signals": [
                "aot_load_observed_in_logs",
                "zero_saved_aot_compiled_function_lines",
                "post_trial_aot_sha256_unchanged",
            ],
            "unsafe_dev_trials": cfg.unsafe_dev_trials,
            "per_trial": [
                {"trial": r.trial_n, "c2_pass": r.c2_pass,
                 "cache_reused": r.cache_reused,
                 "aot_sha256_post": r.aot_sha256_post}
                for r in trial_results
            ],
        },
        "replaces_manifest": replaces_manifest_filename,
        "replaces_artifact_sha256": replaces_artifact_sha256,
        "archive_paths": None,  # filled in after archive step below
    }

    # ---- Step 1: stage new cache + manifest under .new names. ---------------
    blessed_dir_new = blessed_dir.with_name(blessed_dir.name + ".new")
    manifest_path_new = manifest_path.with_name(manifest_path.name + ".new")
    if blessed_dir_new.exists():
        raise RuntimeError(
            f"orphan staging from prior bless: {blessed_dir_new}; "
            "remove it before retrying"
        )
    if manifest_path_new.exists():
        raise RuntimeError(
            f"orphan manifest staging from prior bless: {manifest_path_new}; "
            "remove it before retrying"
        )
    if not is_rebless and blessed_dir.exists():
        raise RuntimeError(
            f"blessed dir already exists for first-bless (race?): {blessed_dir}"
        )
    if not is_rebless and manifest_path.exists():
        raise RuntimeError(
            f"manifest already exists for first-bless (race?): {manifest_path}"
        )
    staging_dir.rename(blessed_dir_new)
    manifest_path_new.write_text(json.dumps(manifest, indent=2) + "\n")

    # ---- Step 2: promote — atomic rename of new → canonical names. ----------
    # Order: cache first, manifest second. If anything fails, the operator
    # can inspect .new and .old artifacts and finish the rename manually
    # without losing data.
    blessed_dir_old: Path | None = None
    manifest_path_old: Path | None = None
    if is_rebless:
        # 2a. Move old cache aside.
        if previous_blessed_dir is not None and previous_blessed_dir.exists():
            blessed_dir_old = blessed_dir.with_name(blessed_dir.name + ".old")
            previous_blessed_dir.rename(blessed_dir_old)
        # 2b. Promote new cache.
        blessed_dir_new.rename(blessed_dir)
        # 2c. Move old manifest aside.
        manifest_path_old = manifest_path.with_name(manifest_path.name + ".old")
        previous_manifest.rename(manifest_path_old)
        # 2d. Promote new manifest.
        manifest_path_new.rename(manifest_path)
    else:
        blessed_dir_new.rename(blessed_dir)
        manifest_path_new.rename(manifest_path)

    # ---- Step 3: archive .old artifacts (failures here do not break the
    #              now-active blessed config). ------------------------------
    if is_rebless:
        ts = _dt.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        archive_manifest_dir = manifest_root / "_archive"
        archive_manifest_dir.mkdir(parents=True, exist_ok=True)
        archive_blessed_root = archive_root / "_archive"
        archive_blessed_root.mkdir(parents=True, exist_ok=True)
        archive_manifest_path = archive_manifest_dir / (
            previous_manifest.stem + f"_{ts}_{prev_aot_sha[:8]}.json"
        )
        archive_blessed_path = archive_blessed_root / (
            f"{cfg.config_hash}_{ts}_{prev_aot_sha[:8]}"
        )
        if manifest_path_old is not None and manifest_path_old.exists():
            manifest_path_old.rename(archive_manifest_path)
        if blessed_dir_old is not None and blessed_dir_old.exists():
            blessed_dir_old.rename(archive_blessed_path)
        archive_paths = {
            "manifest": str(archive_manifest_path),
            "blessed_dir": str(archive_blessed_path),
        }
        # Patch the manifest in place to record archive paths (the .new file
        # was written before archive existed). Same fs => write is atomic.
        m = json.loads(manifest_path.read_text())
        m["archive_paths"] = archive_paths
        manifest_path.write_text(json.dumps(m, indent=2) + "\n")

    print(f"[bless/accept] manifest: {manifest_path}", flush=True)
    print(f"[bless/accept] blessed cache: {blessed_dir}", flush=True)
    regenerate_readme_table(manifest_root)
    return manifest_path


def _blessed_by() -> str:
    import getpass, socket
    return f"{getpass.getuser()}@{socket.gethostname()}"


def reject(
    *, staging_dir: Path, evidence_root: Path,
    cfg: BlessConfig, trial_results: list[TrialResult],
) -> Path:
    """Preserve staging dir as evidence; write failure summary; no manifest."""
    ts = _dt.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    evidence_dir = evidence_root / f"{cfg.config_hash}_{ts}"
    evidence_dir.parent.mkdir(parents=True, exist_ok=True)
    if staging_dir.exists():
        staging_dir.rename(evidence_dir)
    else:
        evidence_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "config_hash": cfg.config_hash,
        "image_id": cfg.image_id,
        "rejected_at": _dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "trial_results": [
            {"trial_n": r.trial_n, "c2_pass": r.c2_pass,
             "cache_reused": r.cache_reused,
             "passed": r.passed, "log_paths": r.log_paths}
            for r in trial_results
        ],
    }
    (evidence_dir / "bless_failure.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(f"[bless/reject] preserved at {evidence_dir}", flush=True)
    return evidence_dir


TABLE_BEGIN = "<!-- BEGIN AUTO-GENERATED TABLE -->"
TABLE_END = "<!-- END AUTO-GENERATED TABLE -->"


def regenerate_readme_table(manifest_root: Path) -> None:
    """Replace the auto-generated active-manifests table block in
    docs/blessed-caches/README.md."""
    readme = manifest_root / "README.md"
    if not readme.exists():
        return
    rows: list[str] = []
    for m_path in sorted(manifest_root.glob("*.json")):
        m = json.loads(m_path.read_text())
        cfg = m.get("config", {})
        rows.append(
            f"| `{m_path.name}` | {cfg.get('model_id','?')} | "
            f"{cfg.get('cudagraph_mode','?')} | "
            f"{cfg.get('cute_phase_e_layers','?')} | "
            f"{m.get('blessed_image_id','sha256:?')[:14]}… | "
            f"{m.get('blessed_at','?')} | "
            f"{'unsafe-dev' if m.get('validation',{}).get('unsafe_dev_trials') else 'active'} |"
        )
    if rows:
        table = ("| Filename | Model | cgmode | Layer set | Image ID | "
                 "Blessed at | Status |\n"
                 "|---|---|---|---|---|---|---|\n"
                 + "\n".join(rows) + "\n")
    else:
        table = "_None yet — run `./scripts/bless-cute-full-cache.sh` to bless the first config._\n"
    src = readme.read_text()
    if TABLE_BEGIN not in src or TABLE_END not in src:
        return
    pre, _, rest = src.partition(TABLE_BEGIN)
    _, _, post = rest.partition(TABLE_END)
    readme.write_text(
        f"{pre}{TABLE_BEGIN}\n\n{table}\n{TABLE_END}{post}"
    )


if __name__ == "__main__":
    sys.exit(main())
