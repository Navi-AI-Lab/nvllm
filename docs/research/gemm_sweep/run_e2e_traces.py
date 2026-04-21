"""Per-config E2E orchestration for Phase B.2.3.

For each shortlisted config (from shortlist.json), start a fresh container
with NVLLM_FP4_GEMM_CONFIG_M256=<idx>, wait for readiness, run GSM8K sanity
(drop the config if GSM8K fails), run the timed profiler workload, let the
profiler flush CUPTI buffers, rename the trace to <config_id>.pt.trace.json.gz,
tear the container down, move to the next config.

Each config takes ~15 min wall. Full 12-config sweep: ~3 hours unattended.

Usage (from repo root):
    .venv/bin/python docs/research/gemm_sweep/run_e2e_traces.py
    .venv/bin/python docs/research/gemm_sweep/run_e2e_traces.py --dry-run
    .venv/bin/python docs/research/gemm_sweep/run_e2e_traces.py --only-idx 0 2 5
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path("/home/natfii/docker/nvllm")
SHORTLIST = REPO / "benchmarks/nvllm/traces/gemm_sweep_sm120/2026-04-21-qwen35-27b/shortlist.json"
OUT_ROOT = REPO / "benchmarks/nvllm/traces/gemm_sweep_sm120/2026-04-21-qwen35-27b"
TRACE_WORKLOAD = REPO / "docs/research/gemm_sweep/trace_workload.py"
GSM8K_SCRIPT = REPO / "scripts/gsm8k_sanity.py"

CONTAINER_NAME = "nvllm-gemm-b23"
PORT = 8000
IMAGE = os.environ.get("NVLLM_IMAGE", "nvllm:gb10")

PROFILER_CONFIG = (
    '{"profiler":"torch","torch_profiler_dir":"/tmp/profiles","ignore_frontend":true,'
    '"delay_iterations":0,"active_iterations":600,'
    '"torch_profiler_with_stack":false,"torch_profiler_use_gzip":true}'
)


def docker_cleanup():
    """Remove any stale container that would hold our port or name."""
    for name in (CONTAINER_NAME, "nvllm", "nvllm-gemm-a3-streamk", "nvllm-gemm-a3-baseline"):
        subprocess.run(["docker", "rm", "-f", name], capture_output=True, check=False)


def start_container(config_idx: int, out_dir: Path, log_path: Path) -> str:
    """Launch the serve container with NVLLM_FP4_GEMM_CONFIG_M256 set.

    Returns the container ID.
    """
    cmd = [
        "docker", "run", "-d",
        "--name", CONTAINER_NAME,
        "--gpus", "all",
        "--ipc=host", "--network", "host", "--privileged",
        "-v", f"{Path.home()}/.cache/huggingface:/root/.cache/huggingface",
        "-v", f"{Path.home()}/.cache/flashinfer:/root/.cache/flashinfer",
        "-v", f"{out_dir}:/tmp/profiles",
        "-e", "VLLM_NVFP4_GEMM_BACKEND=cutlass",
        "-e", "VLLM_ALLOW_LONG_MAX_MODEL_LEN=1",
        "-e", "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True",
        "-e", f"NVLLM_FP4_GEMM_CONFIG_M256={config_idx}",
        IMAGE,
        "serve",
        "--model", "ig1/Qwen3.5-27B-NVFP4",
        "--served-model-name", "default",
        "--host", "0.0.0.0", "--port", str(PORT),
        "--kv-cache-dtype", "auto",
        "--attention-backend", "triton_attn",
        "--max-model-len", "65536",
        "--max-num-seqs", "4",
        "--language-model-only",
        "--mamba-cache-mode", "align",
        "--trust-remote-code",
        "--gpu-memory-utilization", "0.80",
        "--max-num-batched-tokens", "65536",
        "--compilation-config", '{"cudagraph_mode":"PIECEWISE"}',
        "--profiler-config", PROFILER_CONFIG,
    ]
    r = subprocess.run(cmd, capture_output=True, text=True, check=True)
    container_id = r.stdout.strip()
    log_path.write_text(f"Container: {container_id}\n")
    return container_id


def wait_ready(port: int, timeout_s: int = 900, log_every_s: int = 60) -> bool:
    """Poll /v1/models until 200 or timeout."""
    import urllib.request
    start = time.time()
    last_log = 0.0
    while time.time() - start < timeout_s:
        try:
            urllib.request.urlopen(f"http://localhost:{port}/v1/models", timeout=5)
            print(f"    ready at t={time.time() - start:.0f}s", flush=True)
            return True
        except Exception:
            pass
        elapsed = time.time() - start
        if elapsed - last_log >= log_every_s:
            print(f"    ... still waiting at t={elapsed:.0f}s", flush=True)
            last_log = elapsed
        time.sleep(5)
    return False


def run_gsm8k_sanity(sanity_log: Path, label: str, save_path: Path) -> bool:
    """Invoke scripts/gsm8k_sanity.py. Returns True if passed.

    Matches phase_d_trace_capture.sh invocation pattern:
      python scripts/gsm8k_sanity.py --api URL --model default --label L --save F
    """
    cmd = [
        str(REPO / ".venv/bin/python"), str(GSM8K_SCRIPT),
        "--api", f"http://localhost:{PORT}/v1",
        "--model", "default",
        "--label", label,
        "--save", str(save_path),
    ]
    with sanity_log.open("w") as f:
        r = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, timeout=1800)
    # gsm8k_sanity.py exits 0 iff passed; phase_d uses `|| true` to not abort,
    # but we want the pass/fail signal.
    return r.returncode == 0


def run_trace(port: int) -> bool:
    cmd = [
        str(REPO / ".venv/bin/python"), str(TRACE_WORKLOAD),
        "--base-url", f"http://localhost:{port}/v1",
        "--model", "default",
        "--warmup", "30", "--timed", "30", "--concurrent", "4",
        "--max-tokens", "256",
        "--profile-start", f"http://localhost:{port}/start_profile",
        "--profile-stop", f"http://localhost:{port}/stop_profile",
    ]
    r = subprocess.run(cmd, capture_output=False, timeout=1800)
    return r.returncode == 0


def flush_and_rename(out_dir: Path, config_id: str, flush_seconds: int = 120) -> Path | None:
    """Wait for CUPTI to finish, then rename rank0.*.pt.trace.json.gz to
    <config_id>.pt.trace.json.gz. Returns the renamed path or None."""
    print(f"    flushing CUPTI buffers ({flush_seconds}s)...", flush=True)
    for i in range(flush_seconds // 30):
        time.sleep(30)
        matches = list(out_dir.glob("rank*.pt.trace.json.gz"))
        sizes = [m.stat().st_size for m in matches]
        print(f"      [+{(i+1)*30}s] rank files: {len(matches)}, sizes: {sizes}", flush=True)

    matches = list(out_dir.glob("rank*.pt.trace.json.gz"))
    if not matches:
        print(f"    ERROR: no rank*.pt.trace.json.gz found in {out_dir}", flush=True)
        return None
    newest = max(matches, key=lambda p: p.stat().st_mtime)
    target = out_dir / f"{config_id}.pt.trace.json.gz"
    newest.rename(target)
    print(f"    renamed {newest.name} -> {target.name}", flush=True)
    return target


def collect_logs(container_id: str, dest: Path):
    with dest.open("w") as f:
        subprocess.run(["docker", "logs", CONTAINER_NAME],
                       stdout=f, stderr=subprocess.STDOUT, check=False)


def stop_container():
    subprocess.run(["docker", "stop", CONTAINER_NAME], capture_output=True, check=False, timeout=60)
    subprocess.run(["docker", "rm", CONTAINER_NAME], capture_output=True, check=False)


def process_one(config_idx: int, config_id: str, out_root: Path) -> dict:
    """Run one config end-to-end. Returns a results dict."""
    status = "ok"
    note = ""
    t0 = time.time()

    e2e_dir = out_root / "e2e"
    sanity_dir = out_root / "gsm8k_sanity"
    decode_dir = out_root / "decode_logs"
    for d in (e2e_dir, sanity_dir, decode_dir):
        d.mkdir(parents=True, exist_ok=True)

    serve_log = Path(f"/tmp/gemm-b23-serve-{config_idx}.log")
    decode_log = decode_dir / f"{config_id}.txt"
    sanity_log = sanity_dir / f"{config_id}.log"
    sanity_json = sanity_dir / f"{config_id}.json"

    print(f"\n===== [{config_idx}/12] {config_id} =====", flush=True)
    docker_cleanup()
    try:
        start_container(config_idx, e2e_dir, serve_log)
        if not wait_ready(PORT, timeout_s=900):
            status = "startup_timeout"
            note = "server did not become ready within 15 min"
            return {"idx": config_idx, "config_id": config_id, "status": status,
                    "note": note, "elapsed": time.time() - t0}

        # GSM8K sanity — fail-closed
        print(f"  GSM8K sanity...", flush=True)
        passed = run_gsm8k_sanity(sanity_log, f"gemm_sweep_{config_id}", sanity_json)
        if not passed:
            status = "sanity_failed"
            note = f"GSM8K sanity returned non-zero; see {sanity_log.name}"
            return {"idx": config_idx, "config_id": config_id, "status": status,
                    "note": note, "elapsed": time.time() - t0}

        # Trace workload
        print(f"  trace workload (30 warmup + 30 timed @ 4)...", flush=True)
        if not run_trace(PORT):
            status = "trace_failed"
            note = "trace_workload.py returned non-zero"
            # continue to flush+rename anyway — partial trace may still be usable

        # Flush + rename
        trace = flush_and_rename(e2e_dir, config_id)
        if trace is None and status == "ok":
            status = "trace_missing"
            note = "no rank*.pt.trace.json.gz artifact after flush"
    finally:
        try:
            collect_logs("", decode_log)
        except Exception as e:
            print(f"    warn: log collection failed: {e}", flush=True)
        stop_container()
        time.sleep(10)  # let ports free up

    return {"idx": config_idx, "config_id": config_id, "status": status,
            "note": note, "elapsed": time.time() - t0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                    help="Print plan and exit")
    ap.add_argument("--only-idx", type=int, nargs="+",
                    help="Run only these config indices (default: all)")
    ap.add_argument("--shortlist", type=Path, default=SHORTLIST)
    args = ap.parse_args()

    data = json.loads(args.shortlist.read_text())
    configs = data["all_configs"]
    indices = args.only_idx if args.only_idx else list(range(len(configs)))

    print(f"Shortlist: {args.shortlist}")
    print(f"Configs: {len(configs)}")
    for i, c in enumerate(configs):
        marker = " <-- run" if i in indices else ""
        print(f"  [{i}] {c}{marker}")
    print(f"Will run: {len(indices)} configs, ETA ~{len(indices) * 15} min")

    if args.dry_run:
        return

    results = []
    overall_start = time.time()
    for idx in indices:
        r = process_one(idx, configs[idx], OUT_ROOT)
        results.append(r)
        elapsed = time.time() - overall_start
        done = len(results)
        remaining = len(indices) - done
        rate = done / elapsed if elapsed else 0
        eta = remaining / rate if rate else 0
        print(f"  [{done}/{len(indices)}] status={r['status']} cfg_elapsed={r['elapsed']:.0f}s  total_elapsed={elapsed:.0f}s  eta={eta:.0f}s", flush=True)

    # Write a summary JSON for downstream B.3.1 to consume
    summary_path = OUT_ROOT / "e2e_results.json"
    summary_path.write_text(json.dumps({"results": results}, indent=2))
    print(f"\nWrote {summary_path}")

    ok = [r for r in results if r["status"] == "ok"]
    print(f"\nSummary: {len(ok)}/{len(results)} configs OK")
    for r in results:
        if r["status"] != "ok":
            print(f"  FAIL [{r['idx']}] {r['config_id']}: {r['status']} ({r['note']})")


if __name__ == "__main__":
    main()
