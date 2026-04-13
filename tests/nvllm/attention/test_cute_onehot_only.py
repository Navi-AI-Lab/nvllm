#!/usr/bin/env python3
"""Minimal one-hot V test — runs in isolation (no prior kernel calls).

Tests whether the PV MMA correctly computes the full product P×V
when V has different columns per dim (one-hot pattern).
"""
import torch
import sys
import logging

logging.basicConfig(level=logging.WARNING)


def main():
    device = "cuda"
    num_q_heads = 24
    num_kv_heads = 4
    head_dim = 256
    page_size = 64
    seq_len = 6
    scale = 1.0 / (head_dim ** 0.5)

    # Deterministic Q: dim 0 = 1.0
    query = torch.zeros(1, num_q_heads, head_dim, dtype=torch.bfloat16,
                         device=device)
    query[0, :, 0] = 1.0

    # Deterministic K: K[tok, :, 0] = tok+1 in FP8
    kv_shape = (2, page_size, num_kv_heads, head_dim)
    k_cache = torch.zeros(kv_shape, dtype=torch.uint8, device=device)
    fp8_vals = {1: 0x38, 2: 0x40, 3: 0x42, 4: 0x44, 5: 0x45, 6: 0x46}
    for t in range(seq_len):
        for h in range(num_kv_heads):
            k_cache[0, t, h, 0] = fp8_vals[t + 1]

    # One-hot V
    v_cache = torch.zeros(kv_shape, dtype=torch.uint8, device=device)
    for t in range(seq_len):
        v_cache[0, t, :, t] = 0x38

    page_table = torch.zeros(1, 2, dtype=torch.int32, device=device)
    seq_lens_t = torch.tensor([seq_len], dtype=torch.int32, device=device)

    # Expected softmax
    k_float = k_cache[0, :seq_len, 0, 0].view(torch.float8_e4m3fn).float()
    probs = torch.softmax(k_float * scale, dim=0)
    print(f"Expected softmax: {probs.tolist()}")

    from vllm.v1.attention.backends.cute_paged.kernel import (
        _get_compiled_kernel, DECODE_CONFIG, _CUTE_AVAILABLE,
    )
    if not _CUTE_AVAILABLE:
        print("CUTLASS not available")
        sys.exit(1)

    kernel = _get_compiled_kernel(DECODE_CONFIG)
    print("Compiling kernel (fresh, no prior calls)...")

    cute_out = kernel(
        query=query, k_cache=k_cache.contiguous(),
        v_cache=v_cache.contiguous(), page_table=page_table,
        seq_lens=seq_lens_t, scale=scale, k_scale=1.0, v_scale=1.0,
        page_size=page_size,
    )
    c = cute_out.float()

    print(f"\ncute[0,0,:8] = {c[0,0,:8].tolist()}")
    print(f"cute[0,1,:8] = {c[0,1,:8].tolist()}")
    print(f"cute[0,2,:8] = {c[0,2,:8].tolist()}")

    print(f"\nPer-head cute_sum (dims 0-5):")
    gs = num_q_heads // num_kv_heads
    for h in range(min(num_q_heads, 8)):
        s = c[0, h, :6].sum().item()
        vals = c[0, h, :8].tolist()
        print(f"  head {h} (kv={h//gs}): sum={s:.4f} vals={vals}")

    # Check if output is diagonal or full
    nonzero_per_head = []
    for h in range(6):
        nz = (c[0, h, :6].abs() > 1e-4).sum().item()
        nonzero_per_head.append(nz)
    print(f"\nNonzero dims (0-5) per head: {nonzero_per_head}")
    print(f"Expected: [6, 6, 6, 6, 6, 6] (all 6 dims nonzero)")

    if all(n == 6 for n in nonzero_per_head):
        print("PASS — full PV product computed")
    elif all(n <= 1 for n in nonzero_per_head):
        print("FAIL — diagonal only (each head sees only its own token)")
    else:
        print(f"PARTIAL — mixed pattern")


if __name__ == "__main__":
    main()
