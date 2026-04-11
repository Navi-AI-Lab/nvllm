"""Pre-compile CuTe DSL attention kernels for Dockerfile build.

Compiles exactly 2 kernel variants (decode + prefill) with dummy tensors
to trigger NVRTC compilation. Results are captured by the disk cache for
zero cold-start serving.

If compilation fails, the script exits with code 1 -- Docker build fails.

Usage:
    python -m vllm.v1.attention.backends.cute_paged.warmup --arch sm_121
    python -m vllm.v1.attention.backends.cute_paged.warmup \\
        --verify-only --cache-dir /opt/vllm/kernel_cache
"""
import argparse
import logging
import os
import sys

import torch

logger = logging.getLogger(__name__)


def warmup(arch: str) -> int:
    """Compile all kernel configs. Returns number compiled.

    Raises RuntimeError if any compilation fails.
    """
    from vllm.v1.attention.backends.cute_paged.disk_cache import (
        apply_disk_cache_patch,
    )
    from vllm.v1.attention.backends.cute_paged.kernel import (
        DECODE_CONFIG,
        PREFILL_CONFIG,
        _get_compiled_kernel,
    )

    cache_dir = os.environ.get(
        "B12X_CUTE_COMPILE_CACHE_DIR", "/opt/vllm/kernel_cache",
    )
    apply_disk_cache_patch(cache_dir=cache_dir)

    configs = [
        ("decode", DECODE_CONFIG),
        ("prefill", PREFILL_CONFIG),
    ]

    compiled = 0
    for name, config in configs:
        logger.info("Compiling %s kernel: %s", name, config)
        try:
            kernel = _get_compiled_kernel(config)
            # Trigger compilation with dummy tensors.
            # Shapes must be valid but values don't matter.
            num_q_heads = 32
            num_kv_heads = 8
            head_dim = config.head_dim
            page_size = config.block_size
            num_pages = 2

            # Decode: 1 query token per sequence (valid decode shape).
            # Prefill: cta_q tokens for 1 sequence.
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
            k_cache = torch.zeros(
                num_pages, page_size, num_kv_heads, head_dim,
                dtype=torch.uint8, device="cuda",
            )
            v_cache = torch.zeros_like(k_cache)
            page_table = torch.zeros(
                num_seqs, num_pages, dtype=torch.int32, device="cuda",
            )
            seq_lens = torch.tensor(
                [q_tokens] * num_seqs, dtype=torch.int32, device="cuda",
            )
            query_start_loc = torch.tensor(
                [0, q_tokens], dtype=torch.int32, device="cuda",
            )

            kernel(
                query=q,
                k_cache=k_cache,
                v_cache=v_cache,
                page_table=page_table,
                seq_lens=seq_lens,
                scale=1.0 / (head_dim ** 0.5),
                k_scale=1.0,
                v_scale=1.0,
                page_size=page_size,
                query_start_loc=query_start_loc,
            )
            compiled += 1
            logger.info("Successfully compiled %s kernel", name)
        except Exception as e:
            logger.error("FAILED to compile %s kernel: %s", name, e)
            raise RuntimeError(
                f"CuTe kernel compilation failed for {name}: {e}"
            ) from e

    return compiled


def verify(cache_dir: str) -> bool:
    """Verify all required kernels exist in cache."""
    if not os.path.isdir(cache_dir):
        logger.error("Cache directory not found: %s", cache_dir)
        return False
    count = sum(1 for _ in _iter_cache_files(cache_dir))
    expected = 2  # decode + prefill
    if count < expected:
        logger.error("Expected %d cached kernels, found %d", expected, count)
        return False
    logger.info("Verified %d cached kernels in %s", count, cache_dir)
    return True


def _iter_cache_files(cache_dir: str):
    for subdir in os.listdir(cache_dir):
        subpath = os.path.join(cache_dir, subdir)
        if os.path.isdir(subpath):
            yield from (
                os.path.join(subpath, f) for f in os.listdir(subpath)
            )


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
