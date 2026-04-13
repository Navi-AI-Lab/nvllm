#!/usr/bin/env python3
"""Deterministic CuTe kernel test: known Q/K → predictable softmax.

Uses controlled Q and K values so softmax weights are hand-verifiable,
then tests with two V patterns:
  1. V = all-ones → output should be 1.0 everywhere (sum-of-softmax check)
  2. V = per-token constant → output should be constant across dims
     (routes V values through MMA to check per-dim routing)
  3. V = one-hot → output reveals individual softmax weights

Volume-mount into container and run:
  docker cp tests/nvllm/attention/test_cute_deterministic.py \
      nvllm:/app/nvllm/tests/nvllm/attention/
  docker exec nvllm python \
      /app/nvllm/tests/nvllm/attention/test_cute_deterministic.py
"""
import torch
import sys
import logging

logging.basicConfig(level=logging.WARNING)


def reference_paged_attention(
    query, k_cache, v_cache, page_table, seq_lens,
    scale, k_scale=1.0, v_scale=1.0, page_size=64,
):
    """Minimal PyTorch reference for decode-only paged attention."""
    num_q_heads = query.shape[1]
    head_dim = query.shape[2]
    num_kv_heads = k_cache.shape[2]
    group_size = num_q_heads // num_kv_heads
    output = torch.zeros_like(query)

    for s in range(len(seq_lens)):
        sl = seq_lens[s].item()
        num_pages = (sl + page_size - 1) // page_size
        k_list, v_list = [], []
        for p in range(num_pages):
            phys = page_table[s, p].item()
            end = min(page_size, sl - p * page_size)
            k_list.append(k_cache[phys, :end])
            v_list.append(v_cache[phys, :end])
        k_seq = torch.cat(k_list, dim=0)
        v_seq = torch.cat(v_list, dim=0)

        k_f = k_seq.view(torch.float8_e4m3fn).to(torch.bfloat16).float()
        k_f = k_f * k_scale
        v_f = v_seq.view(torch.float8_e4m3fn).to(torch.bfloat16).float()
        v_f = v_f * v_scale

        q_s = query[s].float()
        for qh in range(num_q_heads):
            kvh = qh // group_size
            scores = (q_s[qh] @ k_f[:, kvh, :].T) * scale
            probs = torch.softmax(scores, dim=0)
            out = probs @ v_f[:, kvh, :]
            output[s, qh] = out.to(query.dtype)
    return output


def make_deterministic_kv(num_kv_heads, head_dim, page_size, seq_len,
                          device):
    """Create K cache with known FP8 values.

    K[tok, head, dim=0] = tok+1 in FP8, rest = 0.
    This makes QK[tok] = Q[0] * (tok+1) — linearly increasing.
    """
    kv_shape = (2, page_size, num_kv_heads, head_dim)
    k_cache = torch.zeros(kv_shape, dtype=torch.uint8, device=device)

    # FP8 E4M3 encoding for small integers:
    # 1.0 = 0x38, 2.0 = 0x40, 3.0 = 0x42, 4.0 = 0x44
    # 5.0 = 0x45, 6.0 = 0x46
    fp8_vals = {
        1: 0x38, 2: 0x40, 3: 0x42, 4: 0x44, 5: 0x45, 6: 0x46,
    }
    for t in range(seq_len):
        for h in range(num_kv_heads):
            k_cache[0, t, h, 0] = fp8_vals[t + 1]

    return k_cache


def main():
    device = "cuda"
    num_q_heads = 24
    num_kv_heads = 4
    head_dim = 256
    page_size = 64
    seq_len = 6
    scale = 1.0 / (head_dim ** 0.5)  # 0.0625

    print("=" * 60)
    print("Deterministic CuTe Kernel Test")
    print("=" * 60)

    # --- Deterministic Q: only dim 0 = 1.0, rest = 0 ---
    query = torch.zeros(1, num_q_heads, head_dim, dtype=torch.bfloat16,
                         device=device)
    query[0, :, 0] = 1.0

    # --- Deterministic K: K[tok, :, 0] = tok+1, rest 0 ---
    k_cache = make_deterministic_kv(num_kv_heads, head_dim, page_size,
                                     seq_len, device)

    page_table = torch.zeros(1, 2, dtype=torch.int32, device=device)
    seq_lens_t = torch.tensor([seq_len], dtype=torch.int32, device=device)

    # Verify K encoding
    k_float = k_cache[0, :seq_len, 0, 0].view(torch.float8_e4m3fn).float()
    print(f"K[tok, head=0, dim=0] = {k_float.tolist()}")
    print(f"Expected QK scores (before scale): {k_float.tolist()}")
    print(f"After scale ({scale}): "
          f"{(k_float * scale).tolist()}")

    # Expected softmax
    scores = k_float * scale
    probs = torch.softmax(scores, dim=0)
    print(f"Expected softmax: {probs.tolist()}")
    print(f"Expected softmax sum: {probs.sum().item():.4f}")

    from vllm.v1.attention.backends.cute_paged.kernel import (
        _get_compiled_kernel, DECODE_CONFIG, _CUTE_AVAILABLE,
    )
    if not _CUTE_AVAILABLE:
        print("CUTLASS not available")
        sys.exit(1)

    kernel = _get_compiled_kernel(DECODE_CONFIG)

    # ============================================================
    # Test 1: V = all-ones → output should be 1.0 everywhere
    # ============================================================
    print("\n--- Test 1: V = all-ones ---")
    v_ones = torch.full((2, page_size, num_kv_heads, head_dim),
                         0x38, dtype=torch.uint8, device=device)

    cute_out = kernel(
        query=query, k_cache=k_cache.contiguous(),
        v_cache=v_ones.contiguous(), page_table=page_table,
        seq_lens=seq_lens_t, scale=scale, k_scale=1.0, v_scale=1.0,
        page_size=page_size,
    )
    c = cute_out.float()
    print(f"  cute[0,0,:8] = {c[0,0,:8].tolist()}")
    print(f"  mean = {c[0,0,:].mean().item():.4f}  "
          f"std = {c[0,0,:].std().item():.6f}")

    # ============================================================
    # Test 2: V = per-token constant → output constant across dims
    # ============================================================
    print("\n--- Test 2: V = per-token constant ---")
    v_ptc = torch.zeros(2, page_size, num_kv_heads, head_dim,
                          dtype=torch.uint8, device=device)
    # V[tok=t, :, :] = fp8(t+1) for all dims
    fp8_vals = {1: 0x38, 2: 0x40, 3: 0x42, 4: 0x44, 5: 0x45, 6: 0x46}
    for t in range(seq_len):
        v_ptc[0, t, :, :] = fp8_vals[t + 1]

    ref_out2 = reference_paged_attention(
        query, k_cache, v_ptc, page_table, seq_lens_t,
        scale=scale, k_scale=1.0, v_scale=1.0, page_size=page_size,
    )
    cute_out2 = kernel(
        query=query, k_cache=k_cache.contiguous(),
        v_cache=v_ptc.contiguous(), page_table=page_table,
        seq_lens=seq_lens_t, scale=scale, k_scale=1.0, v_scale=1.0,
        page_size=page_size,
    )
    c2 = cute_out2.float()
    r2 = ref_out2.float()
    print(f"  cute[0,0,:8] = {c2[0,0,:8].tolist()}")
    print(f"   ref[0,0,:8] = {r2[0,0,:8].tolist()}")
    print(f"  cute std across dims = {c2[0,0,:].std().item():.6f} "
          f"(should be ~0)")
    print(f"   ref std across dims = {r2[0,0,:].std().item():.6f}")

    # Expected: sum_t(P[t] * (t+1)) = same for all dims
    expected_val = (probs * k_float).sum().item()
    print(f"  Expected value (all dims): {expected_val:.4f}")
    print(f"  Cute mean: {c2[0,0,:].mean().item():.4f}")
    print(f"   Ref mean: {r2[0,0,:].mean().item():.4f}")

    # ============================================================
    # Test 3: V = one-hot → reveals individual softmax weights
    # ============================================================
    print("\n--- Test 3: V = one-hot ---")
    v_oh = torch.zeros(2, page_size, num_kv_heads, head_dim,
                        dtype=torch.uint8, device=device)
    for t in range(seq_len):
        v_oh[0, t, :, t] = 0x38  # 1.0 at dim=t

    ref_out3 = reference_paged_attention(
        query, k_cache, v_oh, page_table, seq_lens_t,
        scale=scale, k_scale=1.0, v_scale=1.0, page_size=page_size,
    )
    cute_out3 = kernel(
        query=query, k_cache=k_cache.contiguous(),
        v_cache=v_oh.contiguous(), page_table=page_table,
        seq_lens=seq_lens_t, scale=scale, k_scale=1.0, v_scale=1.0,
        page_size=page_size,
    )
    c3 = cute_out3.float()
    r3 = ref_out3.float()
    diff = (c3 - r3).abs()

    print(f"  cute[0,0,:8] = {c3[0,0,:8].tolist()}")
    print(f"   ref[0,0,:8] = {r3[0,0,:8].tolist()}")
    print(f"  cute sum (d0-5) = {c3[0,0,:6].sum().item():.4f}")
    print(f"   ref sum (d0-5) = {r3[0,0,:6].sum().item():.4f}")
    print(f"  max diff = {diff.max().item():.6f}")

    # Per-dim comparison for head 0
    print("\n  Per-dim head 0 (dims 0-5):")
    for d in range(6):
        print(f"    dim {d}: cute={c3[0,0,d].item():.6f} "
              f"ref={r3[0,0,d].item():.6f} "
              f"diff={diff[0,0,d].item():.6f}")

    # Check multiple heads
    print("\n  Per-head max diff:")
    gs = num_q_heads // num_kv_heads
    for h in range(num_q_heads):
        hdiff = diff[0, h].max().item()
        hsum = c3[0, h, :6].sum().item()
        print(f"    head {h:2d} (kv={h//gs}): "
              f"max_diff={hdiff:.4f} cute_sum={hsum:.4f}")

    print("\n" + "=" * 60)
    if diff.max().item() < 0.02:
        print("ALL TESTS PASS")
    else:
        print(f"FAIL: max diff = {diff.max().item():.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
