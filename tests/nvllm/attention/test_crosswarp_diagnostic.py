#!/usr/bin/env python3
"""Session 11 diagnostic: Cross-warp reduction bug investigation.

Three tests to triangulate the stride-32 bug:

1. seq_len=64 test: All 4 warps have valid data (no masking).
   - PASS → bug is in masked-warp interaction
   - FAIL → bug is in reduction itself or per-warp PV with 4 warps

2. seq_len=6 test: Same as standalone test (reproduces known failure).
   - Baseline for comparison.

3. Per-warp output analysis: Runs seq_len=6, inspects each warp's
   sync_o contribution by comparing with per-warp reference.

Volume-mount into container and run:
  python /app/nvllm/tests/nvllm/attention/test_crosswarp_diagnostic.py
"""
import sys
sys.path.insert(0, "/app/nvllm")

import torch
import logging

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


def reference_paged_attention_per_warp(
    query, k_cache, v_cache, page_table, seq_lens,
    scale, k_scale=1.0, v_scale=1.0, page_size=64,
    warp_kv_start=0, warp_kv_end=16,
):
    """Reference attention for a SINGLE warp's KV token range.

    Computes partial online softmax state (O, m, d) for tokens
    [warp_kv_start, warp_kv_end) only. Returns unnormalized O, m, d
    so the caller can do cross-warp reduction in Python.
    """
    num_seqs = len(seq_lens)
    num_q_heads = query.shape[1]
    head_dim = query.shape[2]
    num_kv_heads = k_cache.shape[2]
    group_size = num_q_heads // num_kv_heads

    results = []
    for s in range(num_seqs):
        sl = seq_lens[s].item()
        # Gather K, V for this sequence
        num_pages = (sl + page_size - 1) // page_size
        k_list, v_list = [], []
        for p in range(num_pages):
            phys = page_table[s, p].item()
            end = min(page_size, sl - p * page_size)
            k_list.append(k_cache[phys, :end])
            v_list.append(v_cache[phys, :end])
        k_seq = torch.cat(k_list, dim=0)
        v_seq = torch.cat(v_list, dim=0)

        k_f = k_seq.view(torch.float8_e4m3fn).to(torch.bfloat16).float() * k_scale
        v_f = v_seq.view(torch.float8_e4m3fn).to(torch.bfloat16).float() * v_scale

        q_s = query[s].float()
        per_head = []
        for qh in range(num_q_heads):
            kvh = qh // group_size
            q_vec = q_s[qh]
            k_mat = k_f[:, kvh, :]
            v_mat = v_f[:, kvh, :]

            # Full scores for masking
            scores = (q_vec @ k_mat.T) * scale  # [sl]

            # Only consider tokens in this warp's range
            # Use log2(e) scaling to match kernel
            LOG2E = 1.4426950408889634
            scores_log2 = scores * LOG2E * k_scale / k_scale  # already scaled

            # Create warp-local view
            warp_end = min(warp_kv_end, sl)
            if warp_kv_start >= sl:
                # All tokens masked for this warp
                m = torch.tensor(-1e30)
                d = torch.tensor(0.0)
                o = torch.zeros(head_dim)
                # But kernel computes exp2(NEG - NEG) = 1.0, not 0
                # So d = 16 * 1.0 = 16, o = P * V where P=1, V=0 = 0
                per_head.append((o, m.item(), d.item()))
                continue

            # Scores for this warp's tokens
            warp_scores = scores[warp_kv_start:warp_end]
            # Mask tokens beyond seq_len (shouldn't happen if warp_end <= sl)
            # and pad to 16 tokens with -inf
            padded = torch.full((16,), -1e20)
            n_valid = warp_end - warp_kv_start
            padded[:n_valid] = warp_scores[:n_valid] * LOG2E

            m_val = padded.max().item()
            p = torch.pow(2.0, padded - m_val)
            d_val = p.sum().item()

            # PV product (unnormalized)
            v_warp = torch.zeros(16, head_dim)
            if warp_end > warp_kv_start:
                v_warp[:n_valid] = v_mat[warp_kv_start:warp_end]
            o = (p.unsqueeze(1) * v_warp).sum(0)

            per_head.append((o, m_val, d_val))
        results.append(per_head)
    return results


def test_kernel_at_seqlen(seq_len, label=""):
    """Run 4-warp decode kernel at given seq_len, compare to reference."""
    from vllm.v1.attention.backends.cute_paged.kernel import (
        _get_compiled_kernel, DECODE_CONFIG, _CUTE_AVAILABLE,
    )
    if not _CUTE_AVAILABLE:
        print("CUTLASS not available, skipping")
        return None

    num_q_heads = 24
    num_kv_heads = 4
    head_dim = 256
    page_size = 64
    scale = 1.0 / (head_dim ** 0.5)
    device = "cuda"

    torch.manual_seed(42)
    seq_lens_t = torch.tensor([seq_len], dtype=torch.int32, device=device)
    query = torch.randn(1, num_q_heads, head_dim, dtype=torch.bfloat16,
                         device=device)

    num_pages = (seq_len + page_size - 1) // page_size
    # Allocate enough pages
    max_pages = max(num_pages, 2)
    kv_shape = (max_pages, page_size, num_kv_heads, head_dim)

    # Random K
    torch.manual_seed(42)
    query = torch.randn(1, num_q_heads, head_dim, dtype=torch.bfloat16,
                         device=device)
    k_float = torch.randn(*kv_shape, device=device).clamp(-10, 10)
    k_cache = k_float.to(torch.float8_e4m3fn).view(torch.uint8)

    # Non-uniform V: linearly varying pattern (NOT identity, NOT uniform)
    # V[tok, kv_head, dim] = (tok * 256 + dim) % 240 mapped to FP8
    v_float = torch.zeros(*kv_shape, device=device)
    for t in range(min(seq_len, page_size * max_pages)):
        page_idx = t // page_size
        tok_in_page = t % page_size
        for h in range(num_kv_heads):
            for d in range(head_dim):
                # Varied but bounded values
                val = ((t * 7 + d * 3 + h * 13) % 19 - 9) * 0.5
                v_float[page_idx, tok_in_page, h, d] = val
    v_cache = v_float.clamp(-448, 448).to(torch.float8_e4m3fn).view(
        torch.uint8)

    page_table = torch.zeros(1, max_pages, dtype=torch.int32, device=device)
    for p in range(num_pages):
        page_table[0, p] = p

    # Reference
    from tests.nvllm.attention.reference import (
        reference_paged_attention as official_ref,
    )
    ref_out = official_ref(
        query, k_cache, v_cache, page_table, seq_lens_t,
        scale=scale, k_scale=1.0, v_scale=1.0, page_size=page_size,
    )

    # CuTe kernel
    # Force fresh compilation by clearing cache
    from vllm.v1.attention.backends.cute_paged.kernel import _get_compiled_kernel
    _get_compiled_kernel.cache_clear()
    kernel = _get_compiled_kernel(DECODE_CONFIG)
    cute_out = kernel(
        query=query,
        k_cache=k_cache.contiguous(),
        v_cache=v_cache.contiguous(),
        page_table=page_table,
        seq_lens=seq_lens_t,
        scale=scale,
        k_scale=1.0,
        v_scale=1.0,
        page_size=page_size,
    )

    # Analysis
    cute_f = cute_out.float()
    ref_f = ref_out.float()

    # NaN check
    cute_nan = torch.isnan(cute_f).sum().item()
    ref_nan = torch.isnan(ref_f).sum().item()

    cute_clean = cute_f.nan_to_num(0)
    ref_clean = ref_f.nan_to_num(0)
    diff = (cute_clean - ref_clean).abs()

    print(f"\n{'='*60}")
    print(f"TEST: {label} (seq_len={seq_len})")
    print(f"{'='*60}")
    print(f"NaN: cute={cute_nan} ref={ref_nan}")
    print(f"Max diff: {diff.max().item():.6f}")
    print(f"Mean diff: {diff.mean().item():.6f}")

    # Per-head analysis (first 6 heads)
    gs = num_q_heads // num_kv_heads
    for h in range(min(6, num_q_heads)):
        hdiff = diff[0, h]
        hnan = torch.isnan(cute_f[0, h]).sum().item()
        # Nonzero pattern
        nz_mask = cute_clean[0, h].abs() > 1e-6
        nz_count = nz_mask.sum().item()
        nz_positions = nz_mask.nonzero(as_tuple=False).flatten().tolist()
        # Check for stride pattern
        if len(nz_positions) >= 2:
            strides = [nz_positions[i+1] - nz_positions[i]
                       for i in range(min(len(nz_positions)-1, 5))]
            stride_str = str(strides)
        else:
            stride_str = "N/A"
        print(f"  head {h:2d} (kv={h//gs}): "
              f"max_diff={hdiff.max().item():.4f} "
              f"nz={nz_count}/256 "
              f"nan={hnan} "
              f"strides={stride_str}")

    # Detailed first head dump
    h0_cute = cute_clean[0, 0]
    h0_ref = ref_clean[0, 0]
    print(f"\n  Head 0 cute[:16]: {[f'{x:.3f}' for x in h0_cute[:16].tolist()]}")
    print(f"  Head 0  ref[:16]: {[f'{x:.3f}' for x in h0_ref[:16].tolist()]}")
    print(f"  Head 0 cute[16:32]: {[f'{x:.3f}' for x in h0_cute[16:32].tolist()]}")
    print(f"  Head 0  ref[16:32]: {[f'{x:.3f}' for x in h0_ref[16:32].tolist()]}")

    # Overall nonzero analysis
    h0_nz = (h0_cute.abs() > 1e-6).nonzero(as_tuple=False).flatten().tolist()
    print(f"\n  Head 0 nonzero dims ({len(h0_nz)}): {h0_nz[:30]}{'...' if len(h0_nz) > 30 else ''}")

    passed = diff.max().item() < 0.5
    print(f"\n  {'PASS' if passed else 'FAIL'}: max diff = {diff.max().item():.4f}")

    return {
        "passed": passed,
        "max_diff": diff.max().item(),
        "cute_out": cute_out,
        "ref_out": ref_out,
        "query": query,
        "k_cache": k_cache,
        "v_cache": v_cache,
        "page_table": page_table,
        "seq_lens": seq_lens_t,
    }


def test_v_allones_seqlen6():
    """Confirm V=all-ones still passes (regression check)."""
    from vllm.v1.attention.backends.cute_paged.kernel import (
        _get_compiled_kernel, DECODE_CONFIG, _CUTE_AVAILABLE,
    )
    if not _CUTE_AVAILABLE:
        return

    num_q_heads = 24
    num_kv_heads = 4
    head_dim = 256
    page_size = 64
    scale = 1.0 / (head_dim ** 0.5)
    device = "cuda"

    torch.manual_seed(42)
    seq_lens_t = torch.tensor([6], dtype=torch.int32, device=device)
    query = torch.randn(1, num_q_heads, head_dim, dtype=torch.bfloat16,
                         device=device)

    kv_shape = (2, page_size, num_kv_heads, head_dim)
    k_float = torch.randn(*kv_shape, device=device).clamp(-10, 10)
    k_cache = k_float.to(torch.float8_e4m3fn).view(torch.uint8)
    # V = all 1.0 (FP8 E4M3)
    v_cache = torch.full(kv_shape, 0x38, dtype=torch.uint8, device=device)

    page_table = torch.zeros(1, 2, dtype=torch.int32, device=device)

    from tests.nvllm.attention.reference import reference_paged_attention
    ref_out = reference_paged_attention(
        query, k_cache, v_cache, page_table, seq_lens_t,
        scale=scale, k_scale=1.0, v_scale=1.0, page_size=page_size,
    )

    _get_compiled_kernel.cache_clear()
    kernel = _get_compiled_kernel(DECODE_CONFIG)
    cute_out = kernel(
        query=query,
        k_cache=k_cache.contiguous(),
        v_cache=v_cache.contiguous(),
        page_table=page_table,
        seq_lens=seq_lens_t,
        scale=scale,
        k_scale=1.0,
        v_scale=1.0,
        page_size=page_size,
    )

    diff = (cute_out.float().nan_to_num(0) - ref_out.float().nan_to_num(0)).abs()
    max_d = diff.max().item()
    print(f"\n{'='*60}")
    print(f"TEST: V=all-ones regression check (seq_len=6)")
    print(f"{'='*60}")
    print(f"Max diff: {max_d:.6f}")
    # Check uniformity across dims
    h0 = cute_out[0, 0].float()
    print(f"Head 0 mean={h0.mean().item():.4f} std={h0.std().item():.6f}")
    print(f"{'PASS' if max_d < 0.1 else 'FAIL'}")


def main():
    print("=" * 60)
    print("Session 11: Cross-warp reduction diagnostic")
    print("=" * 60)

    # Test 0: V=all-ones regression check
    test_v_allones_seqlen6()

    # Test 1: seq_len=64 (all warps valid, no masking)
    result64 = test_kernel_at_seqlen(64, "All warps valid (no masking)")

    # Test 2: seq_len=6 (reproduce known failure)
    result6 = test_kernel_at_seqlen(6, "Reproduce stride-32 bug")

    # Test 3: seq_len=32 (2 warps valid, 2 masked)
    result32 = test_kernel_at_seqlen(32, "Half warps valid")

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for label, result in [
        ("seq_len=64 (all valid)", result64),
        ("seq_len=6 (known fail)", result6),
        ("seq_len=32 (half valid)", result32),
    ]:
        if result:
            status = "PASS" if result["passed"] else "FAIL"
            print(f"  {status}: {label} (max_diff={result['max_diff']:.4f})")

    print("\nDiagnostic interpretation:")
    if result64 and result6:
        if result64["passed"] and not result6["passed"]:
            print("  → Bug is in MASKED WARP interaction")
            print("  → Warps with all-masked tokens contribute bad data")
            print("  → Check: exp2(-1e20 - m_final) flush-to-zero, or")
            print("           masked warps' sync_o contains garbage")
        elif not result64["passed"] and not result6["passed"]:
            print("  → Bug is in REDUCTION ITSELF or per-warp PV with 4 warps")
            print("  → Next: dump sync_o before reduction")
        elif result64["passed"] and result6["passed"]:
            print("  → Both pass! Bug may be fixed or test data differs")
        else:
            print("  → Unexpected: seq_len=64 fails but seq_len=6 passes")


if __name__ == "__main__":
    main()
