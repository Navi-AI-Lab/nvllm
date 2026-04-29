"""
Cache-smoke for the CuTe DSL disk cache.

Drives the same two-kernel (decode + prefill) compile path that
vllm.v1.attention.backends.cute_paged.warmup.warmup() uses, under an
explicit apply_disk_cache_patch(...) call.

NOTE: warmup.warmup() itself currently passes the unified `kv_cache=`
kwarg to the prefill kernel, but the prefill kernel signature was
updated to expect split `k_cache=`+`v_cache=` (zero-copy stride
addressing). build-time warmup runs under `|| true` so this regression
went silent. We mirror warmup.warmup()'s approach but pass the right
kwargs per-kernel, so Gate G1 (disk-cache plumbing) can verdict
independent of the orthogonal warmup.py bug. Fix for warmup.py is
out-of-scope for this task; tracked separately.

Verifies that:
  cold phase  - populates the per-run cache subdir AND emits at least
                one 'CuTe disk cache MISS' log line.
  warm phase  - emits at least one 'CuTe disk cache HIT' line AND emits
                no MISS lines AND wallclock < 5s for both compiles
                combined.

Cache isolation: each run uses a per-run subdir under
B12X_CUTE_COMPILE_CACHE_DIR_ROOT (default /opt/vllm/kernel_cache_smoke).
The cold phase creates the subdir; the warm phase reuses it. This avoids
'rm -rf' on the shared /tmp/nvllm-cute-cache.

Run inside the nvllm container:
  python3 docs/research/2026-04-29-full-graph-spike/cache_smoke.py \\
      --phase=cold --run-id=<id>
  # ... container restart between phases, same --run-id ...
  python3 docs/research/2026-04-29-full-graph-spike/cache_smoke.py \\
      --phase=warm --run-id=<id>
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

# Ensure the bind-mounted source tree at /workspace takes precedence over
# the image-baked editable install at /app/nvllm. Without this, an
# editable-install path-hook routes `import vllm` to /app/nvllm/vllm,
# which contains the pre-Task-2 disk_cache.py (no HIT/MISS log lines).
# Detected during Task 4: serve-time logs showed [disk_cache.py:458]
# (old enabled-line) instead of [disk_cache.py:477] (post-Task-2 line).
_WORKSPACE = Path("/workspace")
if _WORKSPACE.exists() and (_WORKSPACE / "vllm" / "__init__.py").exists():
    sys.path.insert(0, str(_WORKSPACE))


# Capture log lines emitted by the disk_cache patch so we can assert on
# HIT/MISS counts. Set up before any disk_cache import.
class _LineCapture(logging.Handler):
    def __init__(self):
        super().__init__()
        self.lines: list[str] = []

    def emit(self, record):
        self.lines.append(record.getMessage())


def _find_artifacts(cache_dir: Path) -> list[tuple[str, float, int]]:
    out = []
    if not cache_dir.exists():
        return out
    for p in cache_dir.rglob("*"):
        if p.is_file():
            st = p.stat()
            out.append((str(p.relative_to(cache_dir)), st.st_mtime, st.st_size))
    return sorted(out)


def _drive_compiles(arch: str) -> int:
    """Mirror warmup.warmup() but with prefill-correct kwargs.

    Returns number of kernels compiled. Raises on failure.
    """
    import torch
    from vllm.v1.attention.backends.cute_paged.kernel import (
        DECODE_CONFIG,
        PREFILL_CONFIG,
        _get_compiled_kernel,
    )

    logger = logging.getLogger(__name__)

    configs = [
        ("decode", DECODE_CONFIG),
        ("prefill", PREFILL_CONFIG),
    ]

    compiled = 0
    for name, config in configs:
        logger.info("Compiling %s kernel: %s", name, config)
        kernel = _get_compiled_kernel(config)
        num_q_heads = 32
        num_kv_heads = 8
        head_dim = config.head_dim
        page_size = config.block_size
        num_pages = 2

        is_decode = config.cta_q <= 16
        if is_decode:
            q_tokens = 1
            num_seqs = 1
        else:
            q_tokens = config.cta_q
            num_seqs = 1

        q = torch.zeros(
            q_tokens, num_q_heads, head_dim,
            dtype=torch.bfloat16, device="cuda",
        )
        # Decode reads unified kv_cache: [num_pages, 2, page_size,
        # num_kv_heads, head_dim]. Prefill reads split k_cache/v_cache:
        # [num_pages, page_size, num_kv_heads, head_dim] each.
        kv_cache_unified = torch.zeros(
            num_pages, 2, page_size, num_kv_heads, head_dim,
            dtype=torch.uint8, device="cuda",
        )
        page_table = torch.zeros(
            num_seqs, num_pages, dtype=torch.int32, device="cuda",
        )
        seq_lens = torch.tensor(
            [q_tokens] * num_seqs, dtype=torch.int32, device="cuda",
        )
        query_start_loc = torch.tensor(
            [0, q_tokens], dtype=torch.int32, device="cuda",
        )

        common = dict(
            query=q,
            page_table=page_table,
            seq_lens=seq_lens,
            scale=1.0 / (head_dim ** 0.5),
            k_scale=1.0,
            v_scale=1.0,
            page_size=page_size,
            query_start_loc=query_start_loc,
        )

        if is_decode:
            kernel(kv_cache=kv_cache_unified, **common)
        else:
            k_cache = kv_cache_unified[:, 0].contiguous()
            v_cache = kv_cache_unified[:, 1].contiguous()
            kernel(k_cache=k_cache, v_cache=v_cache, **common)

        compiled += 1
        logger.info("Successfully compiled %s kernel", name)

    return compiled


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("cold", "warm"), required=True)
    parser.add_argument(
        "--run-id",
        required=True,
        help="Unique id used to namespace the per-run cache subdir.",
    )
    parser.add_argument("--arch", default="sm_121")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")

    cache_root = Path(
        os.environ.get(
            "B12X_CUTE_COMPILE_CACHE_DIR_ROOT", "/opt/vllm/kernel_cache_smoke"
        )
    )
    cache_dir = cache_root / args.run_id
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Override env so apply_disk_cache_patch and warmup.warmup both see the
    # per-run subdir, not the shared serve cache.
    os.environ["B12X_CUTE_COMPILE_DISK_CACHE"] = "1"
    os.environ["B12X_CUTE_COMPILE_CACHE_DIR"] = str(cache_dir)

    # Import vllm BEFORE attaching the capture handler. vllm's logging
    # init clears handlers on its named loggers, so any handler attached
    # before `import vllm` is silently dropped.
    from vllm.v1.attention.backends.cute_paged.disk_cache import (
        apply_disk_cache_patch,
    )

    capture = _LineCapture()
    logging.getLogger("vllm.v1.attention.backends.cute_paged.disk_cache").addHandler(
        capture
    )

    apply_disk_cache_patch(cache_dir=str(cache_dir))

    before = _find_artifacts(cache_dir)
    t0 = time.monotonic()

    n_compiled = _drive_compiles(args.arch)

    elapsed = time.monotonic() - t0
    after = _find_artifacts(cache_dir)
    new_files = [f for f in after if f not in before]

    msgs = capture.lines
    miss_count = sum(1 for m in msgs if "CuTe disk cache MISS" in m)
    hit_count = sum(1 for m in msgs if "CuTe disk cache HIT" in m)

    result = {
        "phase": args.phase,
        "run_id": args.run_id,
        "ok": True,
        "elapsed_s": round(elapsed, 3),
        "n_compiled": n_compiled,
        "files_before": len(before),
        "files_after": len(after),
        "new_files": [n[0] for n in new_files],
        "miss_count": miss_count,
        "hit_count": hit_count,
    }

    if args.phase == "cold":
        if miss_count == 0 and len(before) == 0:
            result["ok"] = False
            result["err"] = "cold phase emitted no MISS lines on empty cache"
        if not new_files and len(before) == 0:
            result["ok"] = False
            result["err"] = "cold phase produced no new artifacts"
    elif args.phase == "warm":
        if hit_count == 0:
            result["ok"] = False
            result["err"] = "warm phase emitted no HIT lines"
        if miss_count > 0:
            result["ok"] = False
            result["err"] = (
                f"warm phase still emitted {miss_count} MISS lines - "
                "cache key drift between cold and warm"
            )
        if elapsed > 5.0:
            result["ok"] = False
            result["err"] = (
                f"warm wallclock {elapsed:.1f}s > 5s; cache likely missed"
            )

    print(json.dumps(result))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
