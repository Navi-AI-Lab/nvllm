#!/usr/bin/env python3
"""Minimal probe: test the cross-warp scatter/reduction with known o values.

Approach: Modify the kernel to write KNOWN values into the o accumulators
(overriding PV MMA results) right before the scatter. Then check if the
output correctly reflects those known values.

Since we can't easily inject into the kernel, instead we test indirectly:
If V=all-ones gives O=1.0 at all dims (Test 1 PASSES), but V=one-hot
concentrates O at dims {0,32,64,...} instead of {0,1,2,3,4,5}, then
the scatter/reduction IS correct (it correctly propagates whatever o has).
The bug must be in the PV MMA or the V loading.

This test directly probes: does thread (group=g, sub=s) at _md=m
correctly load V from the expected SMEM position?

We craft V so that each SMEM position has a unique value. Then check
which values appear in the output.
"""
import torch


def test_v_loading():
    """Craft V with unique values per (token, dim) and check PV output."""
    from vllm.v1.attention.backends.cute_paged.kernel import (
        _get_compiled_kernel, DECODE_CONFIG,
    )

    num_q_heads = 24
    num_kv_heads = 4
    head_dim = 256
    scale = 1.0 / (head_dim ** 0.5)
    device = "cuda"
    num_pages = 2
    num_tokens = 6
    kv_shape = (num_pages, 64, num_kv_heads, head_dim)

    # Q = constant 1.0 → QK scores depend on K, softmax gives non-uniform P
    query = torch.ones(1, num_q_heads, head_dim, dtype=torch.bfloat16,
                       device=device)

    # K: all zeros → QK = 0 for all tokens → softmax = uniform = 1/6 each
    k_cache = torch.zeros(kv_shape, dtype=torch.uint8, device=device)

    page_table = torch.zeros(1, 2, dtype=torch.int32, device=device)
    seq_lens = torch.tensor([num_tokens], dtype=torch.int32, device=device)
    kernel = _get_compiled_kernel(DECODE_CONFIG)

    # === Test A: V = column-specific pattern ===
    # V[tok, kv_h, dim] = 1.0 only at specific dim per token
    # Token 0 → dim 0: V[0, :, 0] = 1.0
    # Token 1 → dim 1: V[1, :, 1] = 1.0
    # Token 2 → dim 2: V[2, :, 2] = 1.0
    # ...
    print("=" * 60)
    print("Test A: V one-hot (token t → dim t)")
    print("With K=0 → uniform P=1/6 → O[d] = P[d] = 1/6 for d<6")
    print("=" * 60)

    v_cache = torch.zeros(kv_shape, dtype=torch.uint8, device=device)
    for t in range(num_tokens):
        v_cache[0, t, :, t] = 0x38  # FP8 E4M3 1.0

    out = kernel(
        query=query, k_cache=k_cache.contiguous(),
        v_cache=v_cache.contiguous(), page_table=page_table,
        seq_lens=seq_lens, scale=scale, k_scale=1.0, v_scale=1.0,
        page_size=64,
    )

    o = out[0, 0].float()
    print(f"  O[0:16]   = {o[:16].tolist()}")
    print(f"  O[16:32]  = {o[16:32].tolist()}")
    print(f"  O[32:48]  = {o[32:48].tolist()}")
    nz = (o.abs() > 1e-4).nonzero(as_tuple=False).flatten()
    print(f"  nonzero dims: {nz.tolist()}")
    for d in nz.tolist()[:16]:
        print(f"    dim {d}: O = {o[d].item():.6f}")

    # === Test B: V = only dim 0 = 1.0, all tokens ===
    print("\n" + "=" * 60)
    print("Test B: V[all_tokens, all_heads, dim=0] = 1.0")
    print("Expected: O[0]=1.0, all others=0")
    print("=" * 60)

    v_cache2 = torch.zeros(kv_shape, dtype=torch.uint8, device=device)
    for t in range(num_tokens):
        v_cache2[0, t, :, 0] = 0x38

    out2 = kernel(
        query=query, k_cache=k_cache.contiguous(),
        v_cache=v_cache2.contiguous(), page_table=page_table,
        seq_lens=seq_lens, scale=scale, k_scale=1.0, v_scale=1.0,
        page_size=64,
    )

    o2 = out2[0, 0].float()
    print(f"  O[0:16]  = {o2[:16].tolist()}")
    print(f"  O[32:48] = {o2[32:48].tolist()}")
    nz2 = (o2.abs() > 1e-4).nonzero(as_tuple=False).flatten()
    print(f"  nonzero: {nz2.tolist()}")
    for d in nz2.tolist()[:16]:
        print(f"    dim {d}: O = {o2[d].item():.6f}")

    # === Test C: V = only dim 1 = 1.0, all tokens ===
    print("\n" + "=" * 60)
    print("Test C: V[all_tokens, all_heads, dim=1] = 1.0")
    print("Expected: O[1]=1.0, all others=0")
    print("=" * 60)

    v_cache3 = torch.zeros(kv_shape, dtype=torch.uint8, device=device)
    for t in range(num_tokens):
        v_cache3[0, t, :, 1] = 0x38

    out3 = kernel(
        query=query, k_cache=k_cache.contiguous(),
        v_cache=v_cache3.contiguous(), page_table=page_table,
        seq_lens=seq_lens, scale=scale, k_scale=1.0, v_scale=1.0,
        page_size=64,
    )

    o3 = out3[0, 0].float()
    print(f"  O[0:16]  = {o3[:16].tolist()}")
    nz3 = (o3.abs() > 1e-4).nonzero(as_tuple=False).flatten()
    print(f"  nonzero: {nz3.tolist()}")
    for d in nz3.tolist()[:16]:
        print(f"    dim {d}: O = {o3[d].item():.6f}")

    # === Test D: V = only dim 16 = 1.0, all tokens ===
    print("\n" + "=" * 60)
    print("Test D: V[all_tokens, all_heads, dim=16] = 1.0")
    print("Expected: O[16]=1.0, all others=0")
    print("=" * 60)

    v_cache4 = torch.zeros(kv_shape, dtype=torch.uint8, device=device)
    for t in range(num_tokens):
        v_cache4[0, t, :, 16] = 0x38

    out4 = kernel(
        query=query, k_cache=k_cache.contiguous(),
        v_cache=v_cache4.contiguous(), page_table=page_table,
        seq_lens=seq_lens, scale=scale, k_scale=1.0, v_scale=1.0,
        page_size=64,
    )

    o4 = out4[0, 0].float()
    print(f"  O[0:16]  = {o4[:16].tolist()}")
    print(f"  O[16:32] = {o4[16:32].tolist()}")
    nz4 = (o4.abs() > 1e-4).nonzero(as_tuple=False).flatten()
    print(f"  nonzero: {nz4.tolist()}")

    # === Test E: V = diagonal block in dims 0-15 ===
    # V[tok=t, dim=t] for t=0..5 (same as Test A)
    # V[tok=t, dim=t+16] for t=0..5 (shifted)
    print("\n" + "=" * 60)
    print("Test E: V[tok=t, dim=t+16] = 1.0 (shifted one-hot)")
    print("Expected: O[16]=P[0], O[17]=P[1], ..., O[21]=P[5]")
    print("=" * 60)

    v_cache5 = torch.zeros(kv_shape, dtype=torch.uint8, device=device)
    for t in range(num_tokens):
        v_cache5[0, t, :, t + 16] = 0x38

    out5 = kernel(
        query=query, k_cache=k_cache.contiguous(),
        v_cache=v_cache5.contiguous(), page_table=page_table,
        seq_lens=seq_lens, scale=scale, k_scale=1.0, v_scale=1.0,
        page_size=64,
    )

    o5 = out5[0, 0].float()
    print(f"  O[0:16]  = {o5[:16].tolist()}")
    print(f"  O[16:32] = {o5[16:32].tolist()}")
    print(f"  O[32:48] = {o5[32:48].tolist()}")
    nz5 = (o5.abs() > 1e-4).nonzero(as_tuple=False).flatten()
    print(f"  nonzero: {nz5.tolist()}")
    for d in nz5.tolist()[:16]:
        print(f"    dim {d}: O = {o5[d].item():.6f}")


if __name__ == "__main__":
    test_v_loading()
