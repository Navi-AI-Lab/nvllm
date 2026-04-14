#!/usr/bin/env python3
"""Standalone test for CuTe paged attention kernel vs PyTorch reference.

Volume-mount into container and run:
  python /app/nvllm/tests/nvllm/attention/test_cute_kernel_standalone.py
"""
import torch
import logging

logging.basicConfig(level=logging.WARNING)

def reference_paged_attention(
    query, k_cache, v_cache, page_table, seq_lens,
    scale, k_scale=1.0, v_scale=1.0, page_size=64,
    query_start_loc=None,
):
    """Simple PyTorch reference for paged attention (decode only)."""
    num_seqs = len(seq_lens)
    num_q_heads = query.shape[1]
    head_dim = query.shape[2]
    num_kv_heads = k_cache.shape[2]
    group_size = num_q_heads // num_kv_heads

    output = torch.zeros_like(query)

    for s in range(num_seqs):
        sl = seq_lens[s].item()
        # Gather K, V for this sequence
        num_pages = (sl + page_size - 1) // page_size
        k_list = []
        v_list = []
        for p in range(num_pages):
            phys = page_table[s, p].item()
            end = min(page_size, sl - p * page_size)
            k_list.append(k_cache[phys, :end])
            v_list.append(v_cache[phys, :end])
        k_seq = torch.cat(k_list, dim=0)  # [sl, kv_heads, hd] uint8
        v_seq = torch.cat(v_list, dim=0)

        # Dequant FP8 E4M3 → BF16 (matching kernel's E4M3→F16→F32→BF16 path)
        # then → float for dot product
        k_f = k_seq.view(torch.float8_e4m3fn).to(torch.bfloat16).float() * k_scale
        v_f = v_seq.view(torch.float8_e4m3fn).to(torch.bfloat16).float() * v_scale

        # Per Q head
        q_s = query[s].float()  # [num_q_heads, hd]
        for qh in range(num_q_heads):
            kvh = qh // group_size
            q_vec = q_s[qh]  # [hd]
            k_mat = k_f[:, kvh, :]  # [sl, hd]
            v_mat = v_f[:, kvh, :]  # [sl, hd]

            scores = (q_vec @ k_mat.T) * scale  # [sl]
            probs = torch.softmax(scores, dim=0)  # [sl]
            out = probs @ v_mat  # [hd]
            output[s, qh] = out.to(query.dtype)

    return output


def test_cute_kernel():
    """Test CuTe kernel with known data."""
    from vllm.v1.attention.backends.cute_paged.kernel import (
        paged_attention_forward, _CUTE_AVAILABLE,
    )

    if not _CUTE_AVAILABLE:
        print("CUTLASS not available, skipping")
        return

    # Config matching Qwen3.5-27B
    num_q_heads = 24
    num_kv_heads = 4
    head_dim = 256
    page_size = 64
    scale = 1.0 / (head_dim ** 0.5)  # 0.0625

    # Create test data
    torch.manual_seed(42)
    device = "cuda"

    # 1 sequence, 6 tokens (like "The capital of France is" + 1 decode)
    num_seqs = 1
    seq_lens = torch.tensor([6], dtype=torch.int32, device=device)

    # Query: 1 decode token
    query = torch.randn(1, num_q_heads, head_dim, dtype=torch.bfloat16,
                         device=device)

    # KV cache: 2 pages (only 1 used)
    num_pages = 2
    kv_shape_4d = (num_pages, page_size, num_kv_heads, head_dim)
    # Unified 5D KV cache: [num_pages, 2, page_size, num_kv_heads, head_dim]
    kv_cache = torch.zeros(
        num_pages, 2, page_size, num_kv_heads, head_dim,
        dtype=torch.uint8, device=device,
    )
    # PROBABILITY EXTRACTION TEST: V = one-hot → output = softmax probs
    torch.manual_seed(42)
    query = torch.randn(1, num_q_heads, head_dim, dtype=torch.bfloat16,
                         device=device)
    k_float = torch.randn(*kv_shape_4d, device=device).clamp(-10, 10)
    kv_cache[:, 0] = k_float.to(torch.float8_e4m3fn).view(torch.uint8)
    # V: one-hot encoding per token (byte 0x38 = E4M3 1.0)
    for t in range(6):
        kv_cache[0, 1, t, :, t] = 0x38  # V[page0, V_slot, tok_t, :, dim_t] = 1.0
    print(f"V one-hot check: V[0,1,0,:,0]={kv_cache[0,1,0,0,0].item()} "
          f"V[0,1,0,:,1]={kv_cache[0,1,0,0,1].item()} "
          f"V[0,1,1,:,1]={kv_cache[0,1,1,0,1].item()}")

    # Separate views for reference (no copy)
    k_cache = kv_cache[:, 0]
    v_cache = kv_cache[:, 1]

    # Page table: seq 0 uses page 0
    page_table = torch.zeros(1, 2, dtype=torch.int32, device=device)
    page_table[0, 0] = 0

    k_scale = 1.0
    v_scale = 1.0

    k_nan_count = 0
    v_nan_count = ((v_cache == 0x7F) | (v_cache == 0xFF)).sum().item()

    print(f"Config: Q={num_q_heads}h KV={num_kv_heads}h hd={head_dim} "
          f"sl=6 scale={scale:.4f}")
    print(f"E4M3 NaN bytes: k={k_nan_count} v={v_nan_count}")
    print(f"k_cache[0,0,0,:8] bytes = {k_cache[0,0,0,:8].tolist()}")
    print(f"k_cache[0,0,0,:4] fp8 = "
          f"{k_cache[0,0,0,:4].view(torch.float8_e4m3fn).float().tolist()}")

    # Reference — use the SAME one the kernel uses in DEBUG_COMPARE
    import sys
    sys.path.insert(0, "/app/nvllm")
    from tests.nvllm.attention.reference import (
        reference_paged_attention as official_ref,
    )
    ref_out = official_ref(
        query, k_cache.contiguous(), v_cache.contiguous(),
        page_table, seq_lens,
        scale=scale, k_scale=k_scale, v_scale=v_scale,
        page_size=page_size,
    )

    # CuTe kernel — pass unified 5D kv_cache (no .contiguous()!)
    from vllm.v1.attention.backends.cute_paged.kernel import (
        _get_compiled_kernel, DECODE_CONFIG,
    )
    kernel = _get_compiled_kernel(DECODE_CONFIG)
    print("Compiling CuTe kernel...")
    cute_out = kernel(
        query=query,
        kv_cache=kv_cache,
        page_table=page_table,
        seq_lens=seq_lens,
        scale=scale,
        k_scale=k_scale,
        v_scale=v_scale,
        page_size=page_size,
    )

    # Compare
    cute_f = cute_out.float()
    ref_f = ref_out.float()

    # Find NaN
    cute_nan = torch.isnan(cute_f)
    ref_nan = torch.isnan(ref_f)
    print(f"\nNaN check: cute={cute_nan.sum().item()} ref={ref_nan.sum().item()}")
    if cute_nan.any():
        # Find first NaN position
        pos = cute_nan.nonzero(as_tuple=False)[0]
        print(f"  First cute NaN at: seq={pos[0]}, head={pos[1]}, dim={pos[2]}")
        # Show NaN dims for head 0
        h0_nan_dims = cute_nan[0, 0].nonzero(as_tuple=False).flatten().tolist()
        print(f"  Head 0 NaN dims: {h0_nan_dims}")
        # MMA tile mapping
        nan_tiles = sorted(set(d // 16 for d in h0_nan_dims))
        print(f"  NaN in MMA tiles: {nan_tiles} "
              f"(tile_dim = tile*16..tile*16+15)")

    # Replace NaN for diff computation
    cute_clean = cute_f.nan_to_num(0)
    ref_clean = ref_f.nan_to_num(0)
    diff = (cute_clean - ref_clean).abs()

    print(f"\nResults (NaN→0 for diff):")
    print(f"  cute[0,0,:8] = {cute_f[0,0,:8].tolist()}")
    print(f"   ref[0,0,:8] = {ref_f[0,0,:8].tolist()}")
    print(f"  max diff = {diff.max().item():.6f}")
    print(f"  mean diff = {diff.mean().item():.6f}")

    # Per-head analysis (all heads)
    gs = num_q_heads // num_kv_heads
    for h in range(num_q_heads):
        hdiff = diff[0, h]
        hnan = cute_nan[0, h].sum().item()
        print(f"  head {h:2d} (kv={h//gs}): "
              f"max={hdiff.max().item():.4f} "
              f"mean={hdiff.mean().item():.6f} "
              f"nan={hnan}")

    # Show extracted probabilities for head 0 (dims 0-5 = P[h0, tok0..5])
    print(f"\nExtracted attention probs (head 0, dims 0-5):")
    print(f"  cute P = {cute_f[0, 0, :6].tolist()}")
    print(f"   ref P = {ref_f[0, 0, :6].tolist()}")
    print(f"  cute sum = {cute_f[0, 0, :6].sum().item():.4f}")
    print(f"   ref sum = {ref_f[0, 0, :6].sum().item():.4f}")

    max_diff = diff.max().item()
    if max_diff < 0.5:
        print(f"\nPASS: max diff = {max_diff:.4f}")
    else:
        print(f"\nFAIL: max diff = {max_diff:.4f}")


def test_wo_fusion():
    """Test fused attention + W_O GEMV vs unfused (attention then matmul)."""
    from vllm.v1.attention.backends.cute_paged.kernel import (
        paged_attention_forward, _CUTE_AVAILABLE,
    )

    if not _CUTE_AVAILABLE:
        print("CUTLASS not available, skipping W_O fusion test")
        return

    # Config matching Qwen3.5-27B
    num_q_heads = 24
    num_kv_heads = 4
    head_dim = 256
    hidden_dim = 5120
    page_size = 64
    scale = 1.0 / (head_dim ** 0.5)  # 0.0625
    group_size = num_q_heads // num_kv_heads  # 6

    torch.manual_seed(42)
    device = "cuda"

    # 1 sequence, 6 tokens (decode)
    num_seqs = 1
    seq_lens = torch.tensor([6], dtype=torch.int32, device=device)
    query = torch.randn(1, num_q_heads, head_dim, dtype=torch.bfloat16,
                         device=device)

    # KV cache: unified 5D [num_pages, 2, page_size, num_kv_heads, head_dim]
    num_pages = 2
    kv_cache = torch.zeros(num_pages, 2, page_size, num_kv_heads, head_dim,
                           dtype=torch.uint8, device=device)
    k_float = torch.randn(num_pages, page_size, num_kv_heads, head_dim,
                           device=device).clamp(-10, 10)
    kv_cache[:, 0] = k_float.to(torch.float8_e4m3fn).view(torch.uint8)
    v_float = torch.randn(num_pages, page_size, num_kv_heads, head_dim,
                           device=device).clamp(-10, 10)
    kv_cache[:, 1] = v_float.to(torch.float8_e4m3fn).view(torch.uint8)

    page_table = torch.zeros(1, 2, dtype=torch.int32, device=device)

    # --- Create NVFP4 W_O weights ---
    K = num_q_heads * head_dim  # 6144
    N = hidden_dim              # 5120

    # Random FP4 nibbles (0-15, including sign in bit 3)
    torch.manual_seed(123)
    wo_nibbles = torch.randint(0, 16, (N, K), dtype=torch.uint8, device=device)

    # Pack into uint8 (2 nibbles per byte): low nibble = even, high nibble = odd
    wo_weight = torch.zeros(N, K // 2, dtype=torch.uint8, device=device)
    for k in range(0, K, 2):
        lo = wo_nibbles[:, k] & 0x0F
        hi = wo_nibbles[:, k + 1] & 0x0F
        wo_weight[:, k // 2] = lo | (hi << 4)

    # Scale factors: all 1.0 in FP8 E4M3 (byte 0x38) for simplicity
    K_sf = K // 16  # 384
    wo_scales_linear = torch.full((N, K_sf), 0x38, dtype=torch.uint8,
                                  device=device)
    wo_scales_linear = wo_scales_linear.view(torch.float8_e4m3fn)

    # Swizzle the scales (required by kernel)
    from vllm.model_executor.layers.quantization.utils.nvfp4_utils import (
        swizzle_blockscale,
    )
    wo_scales = swizzle_blockscale(wo_scales_linear)

    wo_global_scale = torch.tensor(1.0, dtype=torch.float32, device=device)

    print("=" * 60)
    print("W_O Fusion Test")
    print("=" * 60)
    print(f"Config: Q={num_q_heads}h KV={num_kv_heads}h hd={head_dim} "
          f"hidden={hidden_dim} K={K}")

    # --- Reference: unfused attention -> dequant -> matmul ---
    # Step 1: run attention only (no W_O)
    attn_out = paged_attention_forward(
        query=query, kv_cache=kv_cache, page_table=page_table,
        seq_lens=seq_lens, scale=scale, k_scale=1.0, v_scale=1.0,
        page_size=page_size,
    )  # [1, 24, 256]
    print(f"Attention output shape: {attn_out.shape}")

    # Step 2: dequant FP4 weights and matmul in PyTorch
    # Unpack nibbles back to float
    kE2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
                          dtype=torch.float32, device=device)
    # Unpack all nibbles
    flat_bytes = wo_weight.flatten()
    lo_nibbles = flat_bytes & 0x0F
    hi_nibbles = (flat_bytes >> 4) & 0x0F
    all_nibbles = torch.stack([lo_nibbles, hi_nibbles], dim=1).flatten()
    # all_nibbles has shape [N * K] in (lo, hi, lo, hi, ...) order
    all_nibbles = all_nibbles.reshape(N, K)

    signs = ((all_nibbles >> 3) & 1).float()
    abs_vals = (all_nibbles & 7).long()
    wo_dequant = kE2M1[abs_vals] * (1.0 - 2.0 * signs)  # apply sign
    # wo_dequant: [N, K] float32, with global_scale=1.0 and block_scale=1.0

    # Reference output
    attn_flat = attn_out.view(1, -1).float()  # [1, K=6144]
    ref_output = attn_flat @ wo_dequant.T      # [1, N=3584]

    # --- Fused: attention + W_O in one kernel ---
    wo_output = torch.zeros(num_seqs, N, dtype=torch.float32, device=device)

    fused_attn_out = paged_attention_forward(
        query=query, kv_cache=kv_cache, page_table=page_table,
        seq_lens=seq_lens, scale=scale, k_scale=1.0, v_scale=1.0,
        page_size=page_size,
        wo_weight=wo_weight,
        wo_scales=wo_scales,
        wo_global_scale=wo_global_scale,
        wo_output=wo_output,
    )

    # --- Compare ---
    fused_f = wo_output.float()
    ref_f = ref_output.float()
    diff = (fused_f - ref_f).abs()

    print(f"\nResults:")
    print(f"  fused[0,:8]  = {fused_f[0,:8].tolist()}")
    print(f"  ref[0,:8]    = {ref_f[0,:8].tolist()}")
    print(f"  max diff     = {diff.max().item():.6f}")
    print(f"  mean diff    = {diff.mean().item():.6f}")

    # Per-output-slice analysis
    for i in range(0, N, N // 4):
        sl = diff[0, i:i+N//4]
        print(f"  rows {i:4d}-{i+N//4:4d}: max={sl.max().item():.6f} "
              f"mean={sl.mean().item():.6f}")

    max_diff = diff.max().item()
    if max_diff < 0.05:
        print(f"\nPASS: max diff = {max_diff:.4f}")
    else:
        print(f"\nFAIL: max diff = {max_diff:.4f}")


if __name__ == "__main__":
    test_cute_kernel()
    print("\n" + "=" * 60 + "\n")
    test_wo_fusion()
