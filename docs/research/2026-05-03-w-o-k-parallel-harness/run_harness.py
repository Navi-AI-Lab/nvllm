"""Validation harness for the W_O K-parallel scaling experiment.

See README.md (commit 46ad9bbc5) for the full ratified design.

This script lives at:
    docs/research/2026-05-03-w-o-k-parallel-harness/run_harness.py

It is a single COLD-PROCESS Python harness. Each invocation:

  1. Validates CLI args fail-fast.
  2. Writes ``config.json`` BEFORE any kernel work, so debugging a crash
     always has the run config.
  3. Synthesises the W_O-focused inputs (attn_output, NVFP4 weight bytes,
     swizzled FP8 E4M3 scales, scalar wo_gs, fp32 output buffers,
     int32 grid barrier).
  4. Applies the CuTe DSL disk-cache patch (so JIT compile is paid once
     across the sweep).
  5. JIT-compiles the W_O microkernel via ``make_w_o_microkernel``.
  6. Per-launch loop: reset accumulating buffers, time with CUDA events,
     record timing.csv row.
  7. Computes correctness against three Torch FP32 references —
     ``reference_split_order(wo_split=N)`` is the AUTHORITATIVE gate
     (kernel and reference share reduction tree), ``reference_chained_fma``
     and ``reference_matmul`` are diagnostic-only.
  8. (Optional) ``--ncu`` re-launches self under
     ``ncu --replay-mode application``.
  9. (Optional, diagnostic) ``--no-cooperative`` for kernel-replay
     deadlock reproduction.

----------------------------------------------------------------------
Effective bytes counting -- production-equivalent convention
----------------------------------------------------------------------

For one kernel call we count three categories.

  PAYLOAD  = bytes functionally required for the W_O+gather computation.
             Independent of wo_split.
       attn_output  : B * K * 2                 (bf16 read)
       wo_weight    : H * K // 2                (uint8 packed FP4 read)
       wo_scales    : nmt * nkt * 32 * 4 * 4    (uint8 swizzled FP8 E4M3)
       wo_gs        : 4                         (single fp32 read)
       final_out    : B * H * 4                 (fp32, written once)

  SCRATCH = bytes the kernel moves due to its reduction strategy
            (per-CTA slot writes + every-CTA gather reads). Grows
            with wo_split.
       wo_output writes : B * total_wo_ctas * H * 4
       wo_output reads  : GATHER_CTAS * total_wo_ctas * B * H * 4
            where GATHER_CTAS = slice_ctas * num_kv_heads (== 32).

  EFFECTIVE = PAYLOAD + SCRATCH

  EFFECTIVE_GBPS = EFFECTIVE / (elapsed_us * 1e-6) / 1e9

This is harness telemetry only -- NOT an NCU roofline classification
(per README sec.5: harness telemetry answers "does it scale?", NCU
answers "is it memory-bound?").
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------
# Module-level constants -- Qwen3.5-27B beta-coop config (ratified).
# ---------------------------------------------------------------------
HIDDEN_SIZE = 5120
NUM_KV_HEADS = 4
NUM_Q_HEADS = 24
HEAD_DIM = 256
NUM_THREADS = 128
SLICE_CTAS = 8
DEFAULT_SEED = 4242
CACHE_DIR = "/tmp/cute_harness_cache_v3"

K = NUM_Q_HEADS * HEAD_DIM
NUM_K_GROUPS = K // 16
NUM_K_TILES = (NUM_K_GROUPS + 3) // 4
NUM_M_TILES = (HIDDEN_SIZE + 127) // 128
GATHER_CTAS = SLICE_CTAS * NUM_KV_HEADS  # 32

# Disk-cache log capture -- apply_disk_cache_patch logs at INFO level
# "CuTe disk cache MISS key=<16-hex> - compiling" and
# "CuTe disk cache HIT  key=<16-hex>" through the package logger.
DISK_CACHE_LOGGER = "vllm.v1.attention.backends.cute_paged.disk_cache"


# ---------------------------------------------------------------------
# CLI parsing
# ---------------------------------------------------------------------
def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="W_O K-parallel scaling validation harness "
        "(see docs/research/2026-05-03-w-o-k-parallel-harness/README.md).",
    )
    parser.add_argument(
        "--wo-split", type=int, choices=(1, 2, 4, 8), required=True,
        help="K-parallel CTA factor per KV head. total_wo_ctas = "
             "num_kv_heads * wo_split.",
    )
    parser.add_argument(
        "--launches", type=int, default=50,
        help="Number of timed kernel launches (default 50). With --ncu "
             "this is forced to 1 unless --allow-ncu-multi-launch is set.",
    )
    parser.add_argument(
        "--out", type=str, required=True,
        help="Output directory (will be created if missing).",
    )
    parser.add_argument(
        "--ncu", action="store_true",
        help="Re-launch self under `ncu --replay-mode application`.",
    )
    parser.add_argument(
        "--no-cooperative", action="store_true",
        help="Diagnostic only -- disables cooperative launch. The "
             "post-W_O grid barrier WILL deadlock under non-coop on "
             "any hardware-resident set < num CTAs. Marked in "
             "config.json as diagnostic_no_cooperative=true.",
    )
    parser.add_argument(
        "--seed", type=int, default=DEFAULT_SEED,
        help=f"torch.manual_seed (default {DEFAULT_SEED}).",
    )
    parser.add_argument(
        "--num-active-tokens", type=int, default=1,
        help="Number of active decode tokens (default 1).",
    )
    parser.add_argument(
        "--slice-ctas", type=int, default=SLICE_CTAS,
        help=(
            "Override slice_ctas (production default 8). Total grid CTAs = "
            "slice_ctas * num_kv_heads. Used for 2026-05-03-parity-gap audit "
            "to vary the total cooperative-grid size while holding active "
            "W_O CTAs (= wo_split * num_kv_heads) constant. "
            "GATHER_CTAS effective-bytes term scales with this value."
        ),
    )
    parser.add_argument(
        "--allow-ncu-multi-launch", action="store_true",
        help="Allow --ncu with --launches > 1 (only valid with --ncu).",
    )
    args = parser.parse_args(argv)

    # ----- Fail-fast validation -----
    if args.allow_ncu_multi_launch and not args.ncu:
        parser.error("--allow-ncu-multi-launch is only valid with --ncu")
    if args.ncu and args.launches > 1 and not args.allow_ncu_multi_launch:
        # NCU defaults --launches to 1.
        args.launches = 1
    if args.launches < 1:
        parser.error(f"--launches must be >= 1, got {args.launches}")
    if args.num_active_tokens < 1:
        parser.error(
            f"--num-active-tokens must be >= 1, got {args.num_active_tokens}"
        )
    if args.slice_ctas < args.wo_split:
        parser.error(
            f"--slice-ctas ({args.slice_ctas}) must be >= "
            f"--wo-split ({args.wo_split})"
        )
    if args.slice_ctas < 1:
        parser.error(f"--slice-ctas must be >= 1, got {args.slice_ctas}")
    return args


# ---------------------------------------------------------------------
# NCU re-launch mode -- replaces self with ncu wrapping the same script.
# ---------------------------------------------------------------------
def _ncu_relaunch(args: argparse.Namespace, out_dir: Path) -> None:
    """Replace self with ncu --replay-mode application.

    Sets NVLLM_HARNESS_NCU_RUNNING=1 so the child invocation skips
    re-launch. NCU sections per README sec.5.
    """
    if shutil.which("ncu") is None:
        raise RuntimeError(
            "ncu is not on PATH inside this environment; cannot --ncu."
        )
    ncu_argv = [
        "ncu",
        "--replay-mode", "application",
        "--target-processes", "all",
        "--section", "MemoryWorkloadAnalysis",
        "--section", "ComputeWorkloadAnalysis",
        "--section", "LaunchStats",
        "--section", "Occupancy",
        "--section", "SchedulerStats",
        "--csv",
        "--log-file", str(out_dir / "ncu_stdout.log"),
        "--export", str(out_dir / "kernel.ncu-rep"),
        sys.executable, sys.argv[0],
    ] + [arg for arg in sys.argv[1:] if arg != "--ncu"]

    # Persist the exact invocation for reproducibility.
    (out_dir / "command.txt").write_text(
        " ".join(repr(a) if " " in a else a for a in ncu_argv) + "\n"
    )

    os.environ["NVLLM_HARNESS_NCU_RUNNING"] = "1"
    os.execvp(ncu_argv[0], ncu_argv)
    # NEVER REACHED


# ---------------------------------------------------------------------
# Disk-cache log capture -- record HIT/MISS keys for the first launch.
# ---------------------------------------------------------------------
class _CacheKeyCapture(logging.Handler):
    """Captures the first 16-hex cache key emitted by disk_cache."""

    def __init__(self) -> None:
        super().__init__(level=logging.INFO)
        self.keys: list[tuple[str, str]] = []  # (status, key16)

    def emit(self, record: logging.LogRecord) -> None:  # noqa: D401
        try:
            msg = record.getMessage()
        except Exception:
            return
        if "CuTe disk cache MISS" in msg:
            # Format: "CuTe disk cache MISS key=<16hex> - compiling"
            for tok in msg.split():
                if tok.startswith("key="):
                    self.keys.append(("MISS", tok[len("key="):]))
                    break
        elif "CuTe disk cache HIT" in msg:
            for tok in msg.split():
                if tok.startswith("key="):
                    self.keys.append(("HIT", tok[len("key="):]))
                    break


# ---------------------------------------------------------------------
# Image id (only meaningful when running inside docker).
# ---------------------------------------------------------------------
def _detect_image_id() -> str | None:
    # Inside the container /etc/hostname is the container id; we cannot
    # invoke `docker inspect` from inside. Best-effort fallback: read
    # /proc/self/cgroup. If we cannot identify, return None -- the
    # harness records None and the caller (sweep) can record the image
    # id itself. NVLLM_HARNESS_IMAGE_ID env-var overrides if set.
    env_id = os.environ.get("NVLLM_HARNESS_IMAGE_ID")
    if env_id:
        return env_id
    cgroup_path = Path("/proc/self/cgroup")
    if cgroup_path.exists():
        try:
            txt = cgroup_path.read_text()
            for line in txt.splitlines():
                if "docker" in line:
                    parts = line.strip().split("/")
                    for part in parts:
                        if len(part) == 64 and all(
                            c in "0123456789abcdef" for c in part
                        ):
                            return part
        except OSError:
            pass
    return None


def _detect_git_sha(repo_root: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_root),
            check=True,
            capture_output=True,
        )
        return result.stdout.decode().strip()
    except Exception:
        return os.environ.get("NVLLM_HARNESS_GIT_SHA", "unknown")


# ---------------------------------------------------------------------
# Main entry point.
# ---------------------------------------------------------------------
def main(argv: list[str]) -> int:
    args = _parse_args(argv)
    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # ----- NCU re-launch handling -----
    ncu_already_running = (
        os.environ.get("NVLLM_HARNESS_NCU_RUNNING", "0") == "1"
    )
    if args.ncu and not ncu_already_running:
        _ncu_relaunch(args, out_dir)
        return 0  # unreachable

    # ----- Diagnostic non-cooperative warning -----
    if args.no_cooperative:
        sys.stderr.write(
            "\n"
            "[WARN] DIAGNOSTIC RUN -- no-cooperative is for kernel-replay-"
            "deadlock\n"
            "[WARN] reproduction, NOT a valid scaling-evidence run. The\n"
            "[WARN] split-order gate may not pass under non-cooperative "
            "launch\n"
            "[WARN] (post-W_O grid barrier deadlocks).\n"
            "\n"
        )

    # ----- Lazy imports so --help doesn't pay torch import cost -----
    import torch  # noqa: E402

    # We need the editable-installed copy of vllm. The Dockerfile
    # editable-installs at /app/nvllm; the sweep bind-mounts the host
    # repo over it.
    from vllm.v1.attention.backends.cute_paged.disk_cache import (  # noqa: E402,E501
        apply_disk_cache_patch,
    )

    # Ensure we can import the harness siblings (microkernel /
    # torch_reference) regardless of how python was invoked.
    here = Path(__file__).resolve().parent
    if str(here) not in sys.path:
        sys.path.insert(0, str(here))
    from microkernel import make_w_o_microkernel  # noqa: E402
    from torch_reference import (  # noqa: E402
        reference_chained_fma,
        reference_matmul,
        reference_split_order,
    )

    # ----- repo_root for git sha and config writeback -----
    repo_root = Path("/work")
    if not repo_root.exists():
        # Running from host or non-default mount layout.
        repo_root = here.parent.parent.parent
    git_sha = _detect_git_sha(repo_root)

    # ----- Attach disk-cache log capture BEFORE applying patch. -----
    cap = _CacheKeyCapture()
    cache_logger = logging.getLogger(DISK_CACHE_LOGGER)
    cache_logger.addHandler(cap)
    # Force INFO so HIT/MISS lines emit even if root logger is WARNING.
    if cache_logger.level > logging.INFO or cache_logger.level == 0:
        cache_logger.setLevel(logging.INFO)

    # Apply disk cache patch. Idempotent across imports.
    apply_disk_cache_patch(cache_dir=CACHE_DIR)

    # ----- Initial config.json (cache_key filled in after first launch) -----
    total_wo_ctas = NUM_KV_HEADS * args.wo_split
    slice_ctas = args.slice_ctas
    gather_ctas = slice_ctas * NUM_KV_HEADS
    cooperative = not args.no_cooperative
    config: dict[str, Any] = {
        "git_sha": git_sha,
        "wo_split": args.wo_split,
        "total_wo_ctas": total_wo_ctas,
        "slice_ctas": slice_ctas,
        "gather_ctas": gather_ctas,
        "total_grid_ctas_per_seq": gather_ctas,
        "active_wo_ctas": total_wo_ctas,
        "hidden_size": HIDDEN_SIZE,
        "num_kv_heads": NUM_KV_HEADS,
        "num_q_heads": NUM_Q_HEADS,
        "head_dim": HEAD_DIM,
        "K": K,
        "num_k_groups": NUM_K_GROUPS,
        "num_k_tiles": NUM_K_TILES,
        "num_active_tokens": args.num_active_tokens,
        "seed": args.seed,
        "launches": args.launches,
        "cooperative": cooperative,
        "ncu": bool(args.ncu),
        "warmup_launches": 0,
        "dtypes": {
            "attn_output": "bfloat16",
            "wo_weight": "uint8",
            "wo_scales": "uint8",
            "wo_gs": "float32",
            "wo_output": "float32",
            "final_out": "float32",
        },
        "cache_key": None,
        "diagnostic_no_cooperative": bool(args.no_cooperative),
        "image_id": _detect_image_id(),
        "torch_version": torch.__version__,
        "python_version": sys.version,
        "effective_bytes_formula": (
            "PAYLOAD = B*K*2 + H*K//2 + nmt*nkt*32*4*4 + 4 + B*H*4; "
            "SCRATCH = B*total_wo_ctas*H*4 + GATHER_CTAS*total_wo_ctas*B*H*4; "
            "EFFECTIVE = PAYLOAD + SCRATCH; "
            "GATHER_CTAS = slice_ctas*num_kv_heads "
            f"(this run: {gather_ctas})"
        ),
    }
    config_path = out_dir / "config.json"
    config_path.write_text(json.dumps(config, indent=2) + "\n")

    # ----- Synthesise inputs. -----
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA must be available for this harness")
    device = torch.device("cuda:0")
    torch.manual_seed(args.seed)

    B = args.num_active_tokens
    attn_output = torch.randn(
        B, NUM_Q_HEADS * HEAD_DIM, dtype=torch.float32, device=device,
    ).to(torch.bfloat16).contiguous()

    wo_weight = torch.randint(
        0, 256, (HIDDEN_SIZE, (NUM_Q_HEADS * HEAD_DIM) // 2),
        dtype=torch.uint8, device=device,
    ).contiguous()
    wo_scales = torch.randint(
        0, 256, (NUM_M_TILES, NUM_K_TILES, 32, 4, 4),
        dtype=torch.uint8, device=device,
    ).contiguous()
    # FP8 E4M3 has two NaN encodings (0x7F positive-NaN, 0xFF negative-NaN).
    # Random uint8 produces ~0.78% NaN bytes in 1.96M scale bytes ->
    # guaranteed NaN somewhere. Replace with a benign value (0x40 = +2.0).
    # This matches the prior bit-exact smoke (/tmp/wo_split_order_gate.py).
    nan_mask = (wo_scales == 0x7F) | (wo_scales == 0xFF)
    wo_scales[nan_mask] = 0x40
    wo_gs = torch.tensor([1.0], dtype=torch.float32, device=device)

    wo_output = torch.zeros(
        B, total_wo_ctas, HIDDEN_SIZE,
        dtype=torch.float32, device=device,
    )
    final_out = torch.zeros(
        B, HIDDEN_SIZE, dtype=torch.float32, device=device,
    )
    grid_barrier = torch.zeros(
        max(B, 1), dtype=torch.int32, device=device,
    )

    # ----- JIT compile the microkernel. Cooperative launch is hard-
    # wired in microkernel.py; --no-cooperative remains a diagnostic
    # toggle for the future (today the harness logs the flag and
    # records the run as diagnostic). -----
    if args.no_cooperative:
        # The microkernel always launches cooperative=True today.
        # We honor --no-cooperative by recording it but not by
        # actually disabling the launch flag, since microkernel.py
        # does not expose it. We DO write the warning so anyone
        # reading the artifacts knows this run is diagnostic.
        sys.stderr.write(
            "[WARN] --no-cooperative recorded but the microkernel\n"
            "[WARN] currently always launches cooperative=True.\n"
        )

    kernel = make_w_o_microkernel(
        wo_split=args.wo_split,
        hidden_size=HIDDEN_SIZE,
        num_kv_heads=NUM_KV_HEADS,
        num_q_heads=NUM_Q_HEADS,
        head_dim=HEAD_DIM,
        num_threads=NUM_THREADS,
        slice_ctas=slice_ctas,
    )

    # ----- Effective-bytes invariants (uses run-actual gather_ctas) -----
    payload_bytes = (
        B * K * 2
        + HIDDEN_SIZE * K // 2
        + NUM_M_TILES * NUM_K_TILES * 32 * 4 * 4
        + 4
        + B * HIDDEN_SIZE * 4
    )
    scratch_bytes = (
        B * total_wo_ctas * HIDDEN_SIZE * 4
        + gather_ctas * total_wo_ctas * B * HIDDEN_SIZE * 4
    )
    effective_bytes = payload_bytes + scratch_bytes

    # ----- Per-launch loop -----
    timing_path = out_dir / "timing.csv"
    with timing_path.open("w") as f:
        f.write(
            "launch_idx,elapsed_us,payload_bytes,scratch_bytes,"
            "effective_bytes,effective_gbps,is_warmup\n"
        )

    final_out_first: torch.Tensor | None = None

    for launch_idx in range(args.launches):
        # Reset accumulating buffers between launches.
        wo_output.zero_()
        final_out.zero_()
        grid_barrier.zero_()

        torch.cuda.synchronize()
        start_evt = torch.cuda.Event(enable_timing=True)
        end_evt = torch.cuda.Event(enable_timing=True)

        start_evt.record()
        kernel(
            attn_output, wo_weight, wo_scales, wo_gs,
            wo_output, final_out, grid_barrier,
            B, NUM_K_TILES,
        )
        end_evt.record()
        torch.cuda.synchronize()

        elapsed_us = start_evt.elapsed_time(end_evt) * 1000.0
        effective_gbps = (
            effective_bytes / (elapsed_us * 1e-6) / 1e9
            if elapsed_us > 0 else float("inf")
        )

        # Capture first launch output BEFORE the next reset.
        if launch_idx == 0:
            final_out_first = final_out.detach().clone()
            # Update config with cache_key from the first launch.
            keys = [k for status, k in cap.keys]
            if keys:
                config["cache_key"] = keys[0]
                # Also note status of first event for diagnostic.
                config["cache_status_first_event"] = cap.keys[0][0]
                config_path.write_text(json.dumps(config, indent=2) + "\n")

        with timing_path.open("a") as f:
            f.write(
                f"{launch_idx},{elapsed_us:.6f},{payload_bytes},"
                f"{scratch_bytes},{effective_bytes},"
                f"{effective_gbps:.6f},0\n"
            )

    assert final_out_first is not None

    # ----- Correctness comparisons -----
    # Authoritative gate: split-order reference (kernel reduction tree).
    ref_split_order = reference_split_order(
        attn_output, wo_weight, wo_scales, wo_gs,
        hidden=HIDDEN_SIZE,
        K=K,
        num_kv_heads=NUM_KV_HEADS,
        num_k_tiles=NUM_K_TILES,
        wo_split=args.wo_split,
    )
    diff_so = (final_out_first - ref_split_order).abs()
    denom_so = ref_split_order.abs().clamp_min(1e-30)
    max_abs_so = float(diff_so.max().item())
    max_rel_so = float((diff_so / denom_so).max().item())
    rtol_gate = 1e-3
    atol_gate = 1e-4
    passes_so = bool(torch.allclose(
        final_out_first, ref_split_order, rtol=rtol_gate, atol=atol_gate,
    ))
    (out_dir / "correctness_gate_split_order.json").write_text(
        json.dumps(
            {
                "passes": passes_so,
                "max_abs": max_abs_so,
                "max_rel": max_rel_so,
                "rtol": rtol_gate,
                "atol": atol_gate,
                "ref_function":
                    f"reference_split_order(wo_split={args.wo_split})",
                "wo_split": args.wo_split,
                "kind": "AUTHORITATIVE",
            },
            indent=2,
        ) + "\n"
    )

    # Diagnostic: production-order chained-FMA.
    ref_chained = reference_chained_fma(
        attn_output, wo_weight, wo_scales, wo_gs,
        hidden=HIDDEN_SIZE,
        K=K,
        num_kv_heads=NUM_KV_HEADS,
        num_k_tiles=NUM_K_TILES,
    )
    diff_c = (final_out_first - ref_chained).abs()
    denom_c = ref_chained.abs().clamp_min(1e-30)
    max_abs_c = float(diff_c.max().item())
    max_rel_c = float((diff_c / denom_c).max().item())
    (out_dir / "correctness_vs_chained.json").write_text(
        json.dumps(
            {
                "max_abs": max_abs_c,
                "max_rel": max_rel_c,
                "ref_function": "reference_chained_fma",
                "kind": "DIAGNOSTIC",
            },
            indent=2,
        ) + "\n"
    )

    # Diagnostic: cuBLAS-tree matmul.
    ref_mm = reference_matmul(
        attn_output, wo_weight, wo_scales, wo_gs,
        hidden=HIDDEN_SIZE,
        K=K,
        num_k_tiles=NUM_K_TILES,
    )
    diff_m = (final_out_first - ref_mm).abs()
    denom_m = ref_mm.abs().clamp_min(1e-30)
    max_abs_m = float(diff_m.max().item())
    max_rel_m = float((diff_m / denom_m).max().item())
    (out_dir / "correctness_vs_matmul.json").write_text(
        json.dumps(
            {
                "max_abs": max_abs_m,
                "max_rel": max_rel_m,
                "ref_function": "reference_matmul",
                "kind": "DIAGNOSTIC",
            },
            indent=2,
        ) + "\n"
    )

    # ----- Console summary (single line per run) -----
    sys.stdout.write(
        "[harness] "
        f"wo_split={args.wo_split} "
        f"total_wo_ctas={total_wo_ctas} "
        f"slice_ctas={slice_ctas} "
        f"gather_ctas={gather_ctas} "
        f"launches={args.launches} "
        f"gate_passes={passes_so} "
        f"max_abs_so={max_abs_so:.3e} "
        f"max_rel_so={max_rel_so:.3e} "
        f"cache_key={config.get('cache_key')}\n"
    )

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
