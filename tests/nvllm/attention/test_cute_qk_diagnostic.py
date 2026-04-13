#!/usr/bin/env python3
"""Diagnostic: isolate QK and PV addressing bugs in CuTe paged attention.

Three layered tests:
1. V=all-ones: O should be ~1.0 everywhere (tests normalization + PV scatter)
2. V=one-hot (constant Q): O[d]=1/6 for d<6 (tests PV extraction with uniform P)
3. V=one-hot (basis Q): O reveals both QK and PV addressing

Volume-mount into container and run:
  python /app/nvllm/tests/nvllm/attention/test_cute_qk_diagnostic.py
"""
import torch
import sys


def make_test_data(device="cuda"):
    """Create shared test fixtures."""
    num_q_heads = 24
    num_kv_heads = 4
    head_dim = 256
    page_size = 64
    scale = 1.0 / (head_dim ** 0.5)
    num_pages = 2
    num_tokens = 6
    kv_shape = (num_pages, page_size, num_kv_heads, head_dim)

    # K: deterministic pattern
    k_float = torch.zeros(kv_shape, device=device)
    for t in range(num_tokens):
        for d in range(head_dim):
            val = ((t * 256 + d) % 97 - 48) / 48.0
            k_float[0, t, :, d] = val
    k_cache = k_float.to(torch.float8_e4m3fn).view(torch.uint8)

    page_table = torch.zeros(1, 2, dtype=torch.int32, device=device)
    seq_lens = torch.tensor([num_tokens], dtype=torch.int32, device=device)

    return {
        "num_q_heads": num_q_heads,
        "num_kv_heads": num_kv_heads,
        "head_dim": head_dim,
        "scale": scale,
        "k_cache": k_cache,
        "kv_shape": kv_shape,
        "page_table": page_table,
        "seq_lens": seq_lens,
        "num_tokens": num_tokens,
        "device": device,
    }


def run_kernel(query, k_cache, v_cache, page_table, seq_lens, scale):
    """Run CuTe kernel, return output tensor."""
    from vllm.v1.attention.backends.cute_paged.kernel import (
        _get_compiled_kernel, DECODE_CONFIG,
    )
    kernel = _get_compiled_kernel(DECODE_CONFIG)
    return kernel(
        query=query,
        k_cache=k_cache.contiguous(),
        v_cache=v_cache.contiguous(),
        page_table=page_table,
        seq_lens=seq_lens,
        scale=scale,
        k_scale=1.0,
        v_scale=1.0,
        page_size=64,
    )


def test_1_v_allones():
    """Test 1: V = all-ones (FP8 1.0 = 0x38).

    With V=1 everywhere: O[h, d] = sum_t P[h,t] * 1.0 = 1.0 for all d.
    Any deviation reveals normalization or PV scatter bugs.
    """
    print("\n" + "=" * 60)
    print("TEST 1: V = all-ones (expect O ≈ 1.0 everywhere)")
    print("=" * 60)

    ctx = make_test_data()
    # Constant Q (all 1.0) for simplicity
    query = torch.ones(1, ctx["num_q_heads"], ctx["head_dim"],
                       dtype=torch.bfloat16, device=ctx["device"])

    # V = all ones (0x38 = E4M3 1.0) for valid tokens
    v_cache = torch.zeros(ctx["kv_shape"], dtype=torch.uint8, device=ctx["device"])
    for t in range(ctx["num_tokens"]):
        v_cache[0, t, :, :] = 0x38

    out = run_kernel(query, ctx["k_cache"], v_cache,
                     ctx["page_table"], ctx["seq_lens"], ctx["scale"])

    o = out[0].float()
    print(f"  head 0 dims[0:16]  = {o[0, :16].tolist()}")
    print(f"  head 0 dims[16:32] = {o[0, 16:32].tolist()}")
    print(f"  head 0 dims[240:256] = {o[0, 240:256].tolist()}")
    print(f"  head 0 mean = {o[0].mean().item():.6f}")
    print(f"  head 0 std  = {o[0].std().item():.6f}")
    print(f"  head 0 min  = {o[0].min().item():.6f}")
    print(f"  head 0 max  = {o[0].max().item():.6f}")
    print(f"  head 0 nonzero = {(o[0].abs() > 1e-6).sum().item()}/256")

    # All dims should be ~1.0
    expected = 1.0
    diff = (o[0] - expected).abs()
    print(f"  max |O - 1.0| = {diff.max().item():.6f}")

    if diff.max().item() < 0.1:
        print("  PASS: all dims ≈ 1.0")
    else:
        print("  FAIL: V=all-ones should give O=1.0 everywhere")
        # Show which dims are wrong
        bad = (diff > 0.1).nonzero(as_tuple=False).flatten()
        print(f"  Bad dims (|O-1| > 0.1): first 20 = {bad[:20].tolist()}")
        good = (diff <= 0.1).nonzero(as_tuple=False).flatten()
        print(f"  Good dims: first 20 = {good[:20].tolist()}")

    return diff.max().item() < 0.1


def test_2_onehot_constant_q():
    """Test 2: V=one-hot, Q=constant (all 1.0).

    Constant Q gives uniform P. V one-hot extracts P[h,d].
    Expected: O[h, d] ≈ 1/6 for d=0..5, ≈0 for d>=6.
    If this fails but test 1 passes → PV dim addressing bug.
    """
    print("\n" + "=" * 60)
    print("TEST 2: V=one-hot, Q=constant (expect O[d<6]≈1/6)")
    print("=" * 60)

    ctx = make_test_data()
    query = torch.ones(1, ctx["num_q_heads"], ctx["head_dim"],
                       dtype=torch.bfloat16, device=ctx["device"])

    v_cache = torch.zeros(ctx["kv_shape"], dtype=torch.uint8, device=ctx["device"])
    for t in range(ctx["num_tokens"]):
        v_cache[0, t, :, t] = 0x38

    out = run_kernel(query, ctx["k_cache"], v_cache,
                     ctx["page_table"], ctx["seq_lens"], ctx["scale"])

    o = out[0, 0].float()  # head 0
    print(f"  O[h0, 0:8]  = {o[:8].tolist()}")
    print(f"  O[h0, 8:16] = {o[8:16].tolist()}")
    print(f"  sum(0:6)  = {o[:6].sum().item():.4f} (expect 1.0)")
    print(f"  sum(0:16) = {o[:16].sum().item():.4f}")
    print(f"  nonzero dims = {(o.abs() > 1e-6).nonzero(as_tuple=False).flatten().tolist()}")

    expected_val = 1.0 / ctx["num_tokens"]  # 1/6
    print(f"  expected per-token prob = {expected_val:.6f}")

    # Check which dims have the probability values
    prob_dims = (o.abs() > 0.01).nonzero(as_tuple=False).flatten()
    print(f"  dims with |O| > 0.01: {prob_dims.tolist()}")
    for d in prob_dims.tolist()[:10]:
        print(f"    dim {d}: O = {o[d].item():.6f}")

    # Verify the constant-Q softmax reference
    k_dequant = ctx["k_cache"][0, :ctx["num_tokens"], 0, :].view(
        torch.float8_e4m3fn).to(torch.bfloat16).float()
    q_h0 = query[0, 0].float()
    scores = (q_h0 @ k_dequant.T) * ctx["scale"]
    probs = torch.softmax(scores, dim=0)
    print(f"  ref probs = {probs.tolist()}")
    print(f"  ref sum   = {probs.sum().item():.4f}")

    ok = o[:6].sum().item() > 0.9
    if ok:
        print("  PASS")
    else:
        print("  FAIL: one-hot extraction broken")
    return ok


def test_3_onehot_basis_q():
    """Test 3: V=one-hot, Q=basis vectors at specific dims.

    This isolates the QK addressing — each basis dim should give
    scores proportional to K[:, that_dim].
    """
    print("\n" + "=" * 60)
    print("TEST 3: V=one-hot, Q=basis (QK dim addressing)")
    print("=" * 60)

    ctx = make_test_data()

    v_cache = torch.zeros(ctx["kv_shape"], dtype=torch.uint8, device=ctx["device"])
    for t in range(ctx["num_tokens"]):
        v_cache[0, t, :, t] = 0x38

    k_dequant = ctx["k_cache"][0, :ctx["num_tokens"], 0, :].view(
        torch.float8_e4m3fn).to(torch.bfloat16).float()

    probe_dims = [0, 1, 8, 16, 128, 255]

    for probe_dim in probe_dims:
        query = torch.zeros(1, ctx["num_q_heads"], ctx["head_dim"],
                            dtype=torch.bfloat16, device=ctx["device"])
        query[0, :, probe_dim] = 1.0

        out = run_kernel(query, ctx["k_cache"], v_cache,
                         ctx["page_table"], ctx["seq_lens"], ctx["scale"])

        o = out[0, 0].float()

        # Reference
        ref_scores = k_dequant[:, probe_dim] * ctx["scale"]
        ref_probs = torch.softmax(ref_scores, dim=0).cpu()

        print(f"\n  probe_dim={probe_dim}:")
        print(f"    O[0:8]   = {o[:8].tolist()}")
        print(f"    ref P    = {ref_probs.tolist()}")
        print(f"    O sum(0:6) = {o[:6].sum().item():.4f}")
        nonzero = (o.abs() > 1e-4).nonzero(as_tuple=False).flatten()
        print(f"    nonzero dims: {nonzero.tolist()[:20]}")


def test_4_v_identity_block():
    """Test 4: V = identity for first 16 dims (diag block).

    V[t, d] = 1.0 if t==d and t<6 and d<16, else 0.
    With Q=1 (constant): O[h, d] = P[h, d] for d<6.
    But V is zero for d>=16, so those dims should be 0.

    Compare: V[t, d] = 1.0 for ALL d in a row (row-ones).
    V[t, :] = 1 for t=0..5.
    Then O[h, d] = sum_t P[h,t] = 1.0 for all d.
    """
    print("\n" + "=" * 60)
    print("TEST 4: Row-by-row V probe (which tokens contribute to which dims)")
    print("=" * 60)

    ctx = make_test_data()
    query = torch.ones(1, ctx["num_q_heads"], ctx["head_dim"],
                       dtype=torch.bfloat16, device=ctx["device"])

    # V: only token 0 is 1.0, all dims
    v_cache = torch.zeros(ctx["kv_shape"], dtype=torch.uint8, device=ctx["device"])
    v_cache[0, 0, :, :] = 0x38  # token 0, all dims = 1.0

    out = run_kernel(query, ctx["k_cache"], v_cache,
                     ctx["page_table"], ctx["seq_lens"], ctx["scale"])

    o = out[0, 0].float()
    # Expected: O[h, d] = P[h, 0] * 1.0 ≈ 1/6 for all d
    print(f"  V: only token 0 = 1.0 (all dims)")
    print(f"  O[0:8]     = {o[:8].tolist()}")
    print(f"  O[128:136] = {o[128:136].tolist()}")
    print(f"  O[248:256] = {o[248:256].tolist()}")
    print(f"  mean = {o.mean().item():.6f}, std = {o.std().item():.6f}")
    print(f"  nonzero = {(o.abs() > 1e-4).sum().item()}/256")

    # Now V: only token 1 is 1.0, all dims
    v_cache2 = torch.zeros(ctx["kv_shape"], dtype=torch.uint8, device=ctx["device"])
    v_cache2[0, 1, :, :] = 0x38

    out2 = run_kernel(query, ctx["k_cache"], v_cache2,
                      ctx["page_table"], ctx["seq_lens"], ctx["scale"])

    o2 = out2[0, 0].float()
    print(f"\n  V: only token 1 = 1.0 (all dims)")
    print(f"  O[0:8]     = {o2[:8].tolist()}")
    print(f"  O[128:136] = {o2[128:136].tolist()}")
    print(f"  mean = {o2.mean().item():.6f}, std = {o2.std().item():.6f}")
    print(f"  nonzero = {(o2.abs() > 1e-4).sum().item()}/256")

    # Now V: only dim 0 is 1.0 (all tokens)
    v_cache3 = torch.zeros(ctx["kv_shape"], dtype=torch.uint8, device=ctx["device"])
    for t in range(ctx["num_tokens"]):
        v_cache3[0, t, :, 0] = 0x38

    out3 = run_kernel(query, ctx["k_cache"], v_cache3,
                      ctx["page_table"], ctx["seq_lens"], ctx["scale"])

    o3 = out3[0, 0].float()
    # Expected: O[h, 0] = sum_t P[h,t] * 1.0 = 1.0, O[h, d>0] = 0
    print(f"\n  V: only dim 0 = 1.0 (all tokens)")
    print(f"  O[0:8]   = {o3[:8].tolist()}")
    print(f"  O[0] = {o3[0].item():.6f} (expect 1.0)")
    print(f"  nonzero = {(o3.abs() > 1e-4).nonzero(as_tuple=False).flatten().tolist()[:20]}")


def main():
    print("CuTe Paged Attention — Layered Diagnostic")
    print("=" * 60)

    # Import and compile kernel once
    from vllm.v1.attention.backends.cute_paged.kernel import (
        _get_compiled_kernel, DECODE_CONFIG,
    )
    kernel = _get_compiled_kernel(DECODE_CONFIG)
    print("Kernel compiled.")

    t1 = test_1_v_allones()
    t2 = test_2_onehot_constant_q()
    test_3_onehot_basis_q()
    test_4_v_identity_block()

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    if not t1:
        print("  Test 1 FAILED → normalization or PV scatter broken")
    if not t2:
        print("  Test 2 FAILED → PV dim addressing broken (even with uniform P)")
    if t1 and t2:
        print("  Tests 1+2 PASS → issue is QK-specific, check test 3")


if __name__ == "__main__":
    main()
