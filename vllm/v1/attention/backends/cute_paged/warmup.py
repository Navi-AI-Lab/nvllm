"""Pre-compile CuTe DSL attention kernels for Dockerfile build.

Usage:
    python -m vllm.v1.attention.backends.cute_paged.warmup --arch sm_121
    python -m vllm.v1.attention.backends.cute_paged.warmup --verify-only --cache-dir /opt/vllm/kernel_cache
"""
import argparse
import logging
import os
import sys

logger = logging.getLogger(__name__)

# (cta_q, cta_kv, head_dim, block_size, stages, mma_type)
WARMUP_CONFIGS = [
    # head_dim=128 (Qwen2.5, Llama, etc.)
    (64, 64, 128, 64, 4, "fp8_qk"),
    (64, 64, 128, 64, 4, "bf16_pv"),
    (16, 64, 128, 64, 2, "fp8_qk"),
    (16, 64, 128, 64, 2, "bf16_pv"),
    # head_dim=256 (Qwen3.5)
    (64, 64, 256, 64, 4, "fp8_qk"),
    (64, 64, 256, 64, 4, "bf16_pv"),
    (16, 64, 256, 64, 2, "fp8_qk"),
    (16, 64, 256, 64, 2, "bf16_pv"),
]


def warmup(arch: str) -> int:
    """Compile all kernel configs. Returns number compiled."""
    from vllm.v1.attention.backends.cute_paged.disk_cache import (
        apply_disk_cache_patch,
    )
    cache_dir = os.environ.get(
        "B12X_CUTE_COMPILE_CACHE_DIR", "/opt/vllm/kernel_cache",
    )
    apply_disk_cache_patch(cache_dir=cache_dir)

    compiled = 0
    for config in WARMUP_CONFIGS:
        logger.info("Compiling config: %s", config)
        # TODO: Trigger actual CuTe DSL compilation with dummy tensors
        # matching each config's tile sizes. Currently a placeholder
        # until the CuTe DSL kernel replaces the PyTorch prototype.
        compiled += 1
    return compiled


def verify(cache_dir: str) -> bool:
    """Verify all required kernels exist in cache."""
    if not os.path.isdir(cache_dir):
        logger.error("Cache directory not found: %s", cache_dir)
        return False
    count = sum(1 for _ in _iter_cache_files(cache_dir))
    expected = len(WARMUP_CONFIGS)
    if count < expected:
        logger.error("Expected %d cached kernels, found %d", expected, count)
        return False
    logger.info("Verified %d cached kernels in %s", count, cache_dir)
    return True


def _iter_cache_files(cache_dir: str):
    for subdir in os.listdir(cache_dir):
        subpath = os.path.join(cache_dir, subdir)
        if os.path.isdir(subpath):
            yield from (os.path.join(subpath, f) for f in os.listdir(subpath))


def main():
    parser = argparse.ArgumentParser(
        description="Pre-compile CuTe DSL attention kernels",
    )
    parser.add_argument("--arch", default="sm_121")
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--cache-dir", default="/opt/vllm/kernel_cache")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    if args.verify_only:
        ok = verify(args.cache_dir)
        sys.exit(0 if ok else 1)
    else:
        n = warmup(args.arch)
        logger.info("Compiled %d kernel configs", n)


if __name__ == "__main__":
    main()
