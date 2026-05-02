"""ε epilogue bit-close against Python reference.

Invokes Phase_D_MLP_Kernel with the new emit_epilogue=True kwarg and
compares next_hidden_output to epsilon_epilogue_ref.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "docs/research"))

import pytest
import torch
from importlib import import_module

CUTE_AVAILABLE = True
try:
    from vllm.v1.attention.backends.cute_paged.mlp_kernel import Phase_D_MLP_Kernel
except ImportError:
    CUTE_AVAILABLE = False


@pytest.mark.skipif(not CUTE_AVAILABLE, reason="CUTLASS CuTe DSL not available")
def test_epsilon_epilogue_matches_python_ref():
    # Import the repro harness; file has a hyphen so can't use normal `import`.
    repro = import_module("2026-04-22-phase-e-repro")
    nat, hidden, interm = 4, 5120, 17408

    kernel = Phase_D_MLP_Kernel(hidden_size=hidden, intermediate_size=interm)

    # Inputs for full MLP — we set the MLP weights to zero so mlp_out = 0 exactly.
    # Then ε epilogue reduces to residual + 0 + RMSNorm.
    x = torch.randn(nat, hidden, dtype=torch.bfloat16, device='cuda')
    residual_post = torch.randn(nat, hidden, dtype=torch.bfloat16, device='cuda')
    next_gamma = torch.ones(hidden, dtype=torch.bfloat16, device='cuda')

    # Zero MLP weights (fp4 = 0x00 decodes to +0 in NVFP4)
    gate_fp4 = torch.zeros(interm, hidden // 2, dtype=torch.uint8, device='cuda')
    up_fp4 = torch.zeros(interm, hidden // 2, dtype=torch.uint8, device='cuda')
    down_fp4 = torch.zeros(hidden, interm // 2, dtype=torch.uint8, device='cuda')
    gate_scale = torch.zeros(interm, hidden // 16, dtype=torch.uint8, device='cuda')
    up_scale = torch.zeros(interm, hidden // 16, dtype=torch.uint8, device='cuda')
    down_scale = torch.zeros(hidden, interm // 16, dtype=torch.uint8, device='cuda')

    partial = torch.zeros(nat, 8, hidden, dtype=torch.float32, device='cuda')
    arrival = torch.zeros(nat, 8, dtype=torch.uint32, device='cuda')
    mlp_out = torch.zeros(nat, hidden, dtype=torch.bfloat16, device='cuda')
    next_hidden = torch.zeros(nat, hidden, dtype=torch.bfloat16, device='cuda')

    kernel(
        x, gate_fp4, gate_scale, up_fp4, up_scale, down_fp4, down_scale,
        partial, arrival, mlp_out, nat,
        # NEW kwargs (Task 8):
        residual_post_ln=residual_post,
        next_input_layernorm_gamma=next_gamma,
        next_hidden_output=next_hidden,
        emit_epilogue=True,
        emit_next_layernorm=True,
        rms_eps=1e-6,
    )

    # Reference: mlp_out from zero weights → zero
    mlp_out_ref = torch.zeros_like(mlp_out)
    residual_final_ref, next_hidden_ref = repro.epsilon_epilogue_ref(
        residual_post, mlp_out_ref, next_gamma, eps=1e-6,
    )

    assert torch.allclose(mlp_out, mlp_out_ref, atol=1e-3), (
        f"mlp_out diverged: max {(mlp_out - mlp_out_ref).abs().max().item()}"
    )
    assert torch.allclose(next_hidden, next_hidden_ref, rtol=1e-2, atol=1e-3), (
        f"next_hidden diverged: max {(next_hidden - next_hidden_ref).abs().max().item()}"
    )


@pytest.mark.skipif(not CUTE_AVAILABLE, reason="CUTLASS CuTe DSL not available")
def test_epsilon_epilogue_last_layer_skips_next_layernorm():
    """Last-layer case: emit_next_layernorm=False → next_hidden is just
    residual_final (no RMSNorm, no gamma multiply). Mirrors spec §5.3."""
    repro = import_module("2026-04-22-phase-e-repro")
    nat, hidden, interm = 4, 5120, 17408

    kernel = Phase_D_MLP_Kernel(hidden_size=hidden, intermediate_size=interm)

    x = torch.randn(nat, hidden, dtype=torch.bfloat16, device='cuda')
    residual_post = torch.randn(nat, hidden, dtype=torch.bfloat16, device='cuda')

    # Zero MLP weights → mlp_out = 0
    gate_fp4 = torch.zeros(interm, hidden // 2, dtype=torch.uint8, device='cuda')
    up_fp4 = torch.zeros(interm, hidden // 2, dtype=torch.uint8, device='cuda')
    down_fp4 = torch.zeros(hidden, interm // 2, dtype=torch.uint8, device='cuda')
    gate_scale = torch.zeros(interm, hidden // 16, dtype=torch.uint8, device='cuda')
    up_scale = torch.zeros(interm, hidden // 16, dtype=torch.uint8, device='cuda')
    down_scale = torch.zeros(hidden, interm // 16, dtype=torch.uint8, device='cuda')

    partial = torch.zeros(nat, 8, hidden, dtype=torch.float32, device='cuda')
    arrival = torch.zeros(nat, 8, dtype=torch.uint32, device='cuda')
    mlp_out = torch.zeros(nat, hidden, dtype=torch.bfloat16, device='cuda')
    next_hidden = torch.zeros(nat, hidden, dtype=torch.bfloat16, device='cuda')

    kernel(
        x, gate_fp4, gate_scale, up_fp4, up_scale, down_fp4, down_scale,
        partial, arrival, mlp_out, nat,
        residual_post_ln=residual_post,
        next_input_layernorm_gamma=None,  # last layer has no next layer
        next_hidden_output=next_hidden,
        emit_epilogue=True,
        emit_next_layernorm=False,  # KEY: last-layer path
        rms_eps=1e-6,
    )

    # Reference: mlp_out = 0, next_gamma = None → next_hidden == residual_final
    mlp_out_ref = torch.zeros_like(mlp_out)
    residual_final_ref, next_hidden_ref = repro.epsilon_epilogue_ref(
        residual_post, mlp_out_ref, next_gamma=None, eps=1e-6,
    )

    # Last-layer path is a memcpy — expect bit-close at BF16 precision
    # (reference cast + storage are both BF16; kernel also stores BF16).
    assert torch.equal(next_hidden, residual_final_ref), (
        f"last-layer next_hidden diverged from residual_final: "
        f"max diff {(next_hidden.float() - residual_final_ref.float()).abs().max().item()}"
    )


@pytest.mark.skipif(not CUTE_AVAILABLE, reason="CUTLASS CuTe DSL not available")
def test_phase_0_prologue_matches_rmsnorm_ref():
    """Phase 0 (Task 11): single-CTA input_layernorm prologue.

    Reference: torch RMSNorm(hidden_in + residual_in) * γ.
    Kernel: PhaseE_Beta_Kernel.run_phase_0_only (grid=(1,1,nat), phases 1-4
    are not yet wired — Tasks 12-15 will add them).
    """
    from vllm.v1.attention.backends.cute_paged.phase_e_kernel import (
        PhaseE_Beta_Kernel,
    )

    k = PhaseE_Beta_Kernel(
        hidden_size=5120, intermediate_size=17408,
        num_attn_heads=24, num_kv_heads=4, head_dim=256,
        rms_eps=1e-6,
    )

    nat, hidden = 1, 5120
    hidden_in = torch.randn(nat, hidden, dtype=torch.bfloat16, device='cuda')
    residual_in = torch.randn(nat, hidden, dtype=torch.bfloat16, device='cuda')
    # Non-trivial γ catches scale-broadcast bugs that γ=1 would hide.
    gamma = torch.randn(hidden, dtype=torch.bfloat16, device='cuda')
    normed_out = torch.zeros(nat, hidden, dtype=torch.bfloat16, device='cuda')

    k.run_phase_0_only(hidden_in, residual_in, gamma, normed_out)
    torch.cuda.synchronize()

    # Reference: exact torch RMSNorm math (FP32 accumulator, BF16 cast at end).
    # Qwen3_5RMSNorm uses x * (1 + γ) — see vllm/nvllm/layers/layernorm.py:78
    summed = hidden_in.float() + residual_in.float()
    variance = summed.pow(2).mean(dim=-1, keepdim=True)
    rstd = torch.rsqrt(variance + 1e-6)
    normed_ref = ((summed * rstd) * (1.0 + gamma.float())).to(torch.bfloat16)

    max_abs = (normed_out.float() - normed_ref.float()).abs().max().item()
    assert torch.allclose(normed_out, normed_ref, rtol=1e-2, atol=1e-3), (
        f"Phase 0 RMSNorm diverged: max_abs={max_abs:.3e}; "
        f"kernel[0,:8]={normed_out[0,:8].tolist()}; "
        f"ref[0,:8]={normed_ref[0,:8].tolist()}"
    )


@pytest.mark.skipif(not CUTE_AVAILABLE, reason="CUTLASS CuTe DSL not available")
def test_phase_1_matches_standalone_decode():
    """Phase 1 (Task 12): β-coop Phase 0+1 matches standalone DecodeKernel.

    Reference: standalone DecodeKernel via paged_attention_forward with
      wo_* + rmsnorm_* set (fused Phase A+B+C path).
    Kernel: PhaseE_Beta_Kernel.run_phase_01_only (grid=(1,4,nat), Phase 0
      prologue gated to cta_y==0, Phase 1 attn on all 4 CTAs).

    Two asserts:
      1. attn_input_bf16 (Phase 0 out) matches Python RMSNorm(hidden+res)*γ_in.
      2. attn_output (Phase C out) matches standalone decode's rmsnorm_output.
    """
    from vllm.v1.attention.backends.cute_paged.phase_e_kernel import (
        PhaseE_Beta_Kernel,
    )
    from vllm.v1.attention.backends.cute_paged.kernel import (
        paged_attention_forward,
    )
    from vllm.model_executor.layers.quantization.utils.nvfp4_utils import (
        swizzle_blockscale,
    )

    torch.manual_seed(42)

    # --- Model / config ---
    hidden = 5120
    interm = 17408
    num_q_heads = 24
    num_kv_heads = 4
    head_dim = 256
    group_size = num_q_heads // num_kv_heads  # 6
    page_size = 64
    seq_len = 128   # 2 pages of 64
    nat = 1
    scale = 1.0 / (head_dim ** 0.5)
    device = "cuda"

    # --- Inputs for Phase 0 ---
    hidden_in = torch.randn(nat, hidden, dtype=torch.bfloat16, device=device)
    residual_in = torch.randn(nat, hidden, dtype=torch.bfloat16, device=device)
    input_gamma = torch.randn(hidden, dtype=torch.bfloat16, device=device)
    post_attn_gamma = torch.randn(hidden, dtype=torch.bfloat16, device=device)

    # --- Pre-projected Q (external QKV is not fused in Task 12) ---
    query = torch.randn(nat, num_q_heads, head_dim,
                        dtype=torch.bfloat16, device=device)

    # --- KV cache: [num_pages, 2, page_size, num_kv_heads, head_dim] FP8 ---
    num_pages = 2
    kv_cache = torch.zeros(num_pages, 2, page_size, num_kv_heads, head_dim,
                           dtype=torch.uint8, device=device)
    k_float = torch.randn(num_pages, page_size, num_kv_heads, head_dim,
                          device=device).clamp(-10, 10)
    kv_cache[:, 0] = k_float.to(torch.float8_e4m3fn).view(torch.uint8)
    v_float = torch.randn(num_pages, page_size, num_kv_heads, head_dim,
                          device=device).clamp(-10, 10)
    kv_cache[:, 1] = v_float.to(torch.float8_e4m3fn).view(torch.uint8)

    page_table = torch.zeros(nat, 2, dtype=torch.int32, device=device)
    page_table[0, 0] = 0
    page_table[0, 1] = 1
    seq_lens = torch.tensor([seq_len], dtype=torch.int32, device=device)

    # --- NVFP4 W_O weights (same construction as test_cute_kernel_standalone) ---
    K_dim = num_q_heads * head_dim  # 6144
    N_dim = hidden                   # 5120
    torch.manual_seed(123)
    wo_nibbles = torch.randint(0, 16, (N_dim, K_dim),
                               dtype=torch.uint8, device=device)
    wo_weight = torch.zeros(N_dim, K_dim // 2,
                            dtype=torch.uint8, device=device)
    for k in range(0, K_dim, 2):
        lo = wo_nibbles[:, k] & 0x0F
        hi = wo_nibbles[:, k + 1] & 0x0F
        wo_weight[:, k // 2] = lo | (hi << 4)
    K_sf = K_dim // 16  # 384
    wo_scales_linear = torch.full((N_dim, K_sf), 0x38,
                                   dtype=torch.uint8, device=device)
    wo_scales_linear = wo_scales_linear.view(torch.float8_e4m3fn)
    wo_scales = swizzle_blockscale(wo_scales_linear)
    wo_global_scale = torch.tensor(1.0, dtype=torch.float32, device=device)

    # ================================================================
    # Reference: standalone DecodeKernel via paged_attention_forward
    # with wo_* + rmsnorm_* set (same fused A+B+C path).
    # ================================================================
    # Phase C in DecodeKernel uses rmsnorm_residual as the residual add
    # source (same role as our `residual_in`). NOTE: The reference reads
    # rmsnorm_gamma = post_attn_gamma (NOT input_gamma) because the Phase C
    # RMSNorm is the POST-ATTN norm, not the input norm.
    ref_wo_output = torch.zeros(nat, 4, hidden,   # 4 CTAs (grid_x=1, grid_y=4)
                                dtype=torch.float32, device=device)
    ref_rmsnorm_output = torch.empty(nat, hidden,
                                     dtype=torch.bfloat16, device=device)
    ref_residual_output = torch.empty(nat, hidden,
                                      dtype=torch.bfloat16, device=device)
    ref_arrival_count = torch.zeros(nat, dtype=torch.int32, device=device)

    _ = paged_attention_forward(
        query=query, kv_cache=kv_cache, page_table=page_table,
        seq_lens=seq_lens, scale=scale, k_scale=1.0, v_scale=1.0,
        page_size=page_size,
        wo_weight=wo_weight, wo_scales=wo_scales,
        wo_global_scale=wo_global_scale, wo_output=ref_wo_output,
        rmsnorm_gamma=post_attn_gamma,
        rmsnorm_residual=residual_in,
        rmsnorm_output=ref_rmsnorm_output,
        residual_output=ref_residual_output,
        arrival_count=ref_arrival_count,
        rmsnorm_eps=1e-6,
        padded_num_seqs=nat,
    )
    torch.cuda.synchronize()

    # ================================================================
    # Kernel under test: PhaseE_Beta_Kernel phase 0+1.
    # ================================================================
    k = PhaseE_Beta_Kernel(
        hidden_size=hidden, intermediate_size=interm,
        num_attn_heads=num_q_heads, num_kv_heads=num_kv_heads,
        head_dim=head_dim, rms_eps=1e-6,
    )

    attn_input_bf16 = torch.zeros(nat, hidden,
                                   dtype=torch.bfloat16, device=device)
    attn_output = torch.zeros(nat, hidden,
                              dtype=torch.bfloat16, device=device)

    k.run_phase_01_only(
        hidden_in=hidden_in, residual_in=residual_in,
        input_gamma=input_gamma, post_attn_gamma=post_attn_gamma,
        attn_input_bf16=attn_input_bf16,
        query=query, kv_cache=kv_cache,
        page_table=page_table, seq_lens=seq_lens,
        wo_weight=wo_weight, wo_scales=wo_scales,
        wo_global_scale=wo_global_scale,
        attn_output=attn_output,
        scale=scale, k_scale=1.0, v_scale=1.0,
    )
    torch.cuda.synchronize()

    # --- Assert 1: Phase 0 output matches Python RMSNorm(h+r)*(1+γ_in) ---
    # Qwen3_5RMSNorm uses x * (1 + γ) — see vllm/nvllm/layers/layernorm.py:78
    summed = hidden_in.float() + residual_in.float()
    variance = summed.pow(2).mean(dim=-1, keepdim=True)
    rstd = torch.rsqrt(variance + 1e-6)
    attn_input_ref = ((summed * rstd) * (1.0 + input_gamma.float())).to(
        torch.bfloat16)

    max_abs_p0 = (attn_input_bf16.float() - attn_input_ref.float()).abs().max().item()
    assert torch.allclose(attn_input_bf16, attn_input_ref,
                          rtol=1e-2, atol=1e-3), (
        f"Phase 0 (attn_input_bf16) diverged: max_abs={max_abs_p0:.3e}; "
        f"kernel[0,:8]={attn_input_bf16[0,:8].tolist()}; "
        f"ref[0,:8]={attn_input_ref[0,:8].tolist()}"
    )

    # --- Assert 2: Phase 1 output matches standalone decode fused path ---
    max_abs_p1 = (attn_output.float() - ref_rmsnorm_output.float()).abs().max().item()
    assert torch.allclose(attn_output, ref_rmsnorm_output,
                          rtol=1e-2, atol=1e-3), (
        f"Phase 1 (attn_output post-C RMSNorm) diverged: "
        f"max_abs={max_abs_p1:.3e}; "
        f"kernel[0,:8]={attn_output[0,:8].tolist()}; "
        f"ref[0,:8]={ref_rmsnorm_output[0,:8].tolist()}"
    )


@pytest.mark.skipif(not CUTE_AVAILABLE, reason="CUTLASS CuTe DSL not available")
def test_phase_3_matches_standalone_mlp():
    """Phase 3 (Task 14): β kernel's `run_phase_3_only` MLP matches
    standalone Phase_D_MLP_Kernel legacy path (emit_epilogue=False).

    Inputs: random FP4 nibble weights (same construction as the Phase 1
    test's W_O weights) + UE4M3 scales set to 1.0 (0x38 encoding) so both
    kernels consume identical byte-for-byte buffers.

    Allocate FRESH partial/arrival/output buffers per kernel — the
    arrival counter mutates and neither kernel self-resets it in the
    legacy path (matches Phase_D_MLP_Kernel contract).
    """
    from vllm.v1.attention.backends.cute_paged.phase_e_kernel import (
        PhaseE_Beta_Kernel,
    )

    torch.manual_seed(42)
    nat, hidden, interm = 4, 5120, 17408
    device = "cuda"

    # Random FP4 weights. Mirrors the Phase 1 test pattern
    # (tests/kernels/cute/test_phase_e_epsilon_epilogue.py:234) and the
    # ε epilogue test tensor shapes (line 22) — uint8 storage, 2 nibbles
    # per byte along the K dim.
    gate_fp4 = torch.randint(0, 256, (interm, hidden // 2),
                             dtype=torch.uint8, device=device)
    up_fp4 = torch.randint(0, 256, (interm, hidden // 2),
                           dtype=torch.uint8, device=device)
    down_fp4 = torch.randint(0, 256, (hidden, interm // 2),
                             dtype=torch.uint8, device=device)
    # UE4M3 encoding of 1.0 = 0x38 (matches the Phase 1 test / swizzle_blockscale
    # mocks elsewhere in this file).
    gate_scale = torch.full((interm, hidden // 16), 0x38,
                            dtype=torch.uint8, device=device)
    up_scale = torch.full((interm, hidden // 16), 0x38,
                          dtype=torch.uint8, device=device)
    down_scale = torch.full((hidden, interm // 16), 0x38,
                            dtype=torch.uint8, device=device)

    x = torch.randn(nat, hidden, dtype=torch.bfloat16, device=device)

    # --- Reference: standalone Phase_D_MLP_Kernel -----------------------
    ref_kernel = Phase_D_MLP_Kernel(
        hidden_size=hidden, intermediate_size=interm,
    )
    ref_partial = torch.zeros(nat, ref_kernel.slice_ctas, hidden,
                              dtype=torch.float32, device=device)
    ref_arrival = torch.zeros(nat, ref_kernel.num_k_tiles,
                              dtype=torch.uint32, device=device)
    ref_mlp_out = torch.zeros(nat, hidden,
                              dtype=torch.bfloat16, device=device)

    ref_kernel(
        x, gate_fp4, gate_scale, up_fp4, up_scale, down_fp4, down_scale,
        ref_partial, ref_arrival, ref_mlp_out, nat,
        # legacy path: no ε epilogue
        emit_epilogue=False,
    )
    torch.cuda.synchronize()

    # --- Kernel under test: β kernel Phase 3 only -----------------------
    beta = PhaseE_Beta_Kernel(
        hidden_size=hidden, intermediate_size=interm,
        num_attn_heads=24, num_kv_heads=4, head_dim=256,
    )
    # Same tile config → same SMEM layout → identical math.
    assert beta.slice_ctas == ref_kernel.slice_ctas, (
        f"β kernel slice_ctas={beta.slice_ctas} != "
        f"standalone slice_ctas={ref_kernel.slice_ctas}"
    )
    assert beta.num_k_tiles == ref_kernel.num_k_tiles
    assert beta.tile_s == ref_kernel.tile_s
    assert beta.tile_k == ref_kernel.tile_k

    beta_partial = torch.zeros(nat, beta.slice_ctas, hidden,
                               dtype=torch.float32, device=device)
    beta_arrival = torch.zeros(nat, beta.num_k_tiles,
                               dtype=torch.uint32, device=device)
    beta_mlp_out = torch.zeros(nat, hidden,
                               dtype=torch.bfloat16, device=device)

    beta.run_phase_3_only(
        x=x,
        gate_w_fp4=gate_fp4, gate_w_scale=gate_scale,
        up_w_fp4=up_fp4, up_w_scale=up_scale,
        down_w_fp4=down_fp4, down_w_scale=down_scale,
        mlp_partial_fp32=beta_partial,
        mlp_arrival_count=beta_arrival,
        mlp_output=beta_mlp_out,
        nat=nat,
    )
    torch.cuda.synchronize()

    max_abs = (beta_mlp_out.float() - ref_mlp_out.float()).abs().max().item()
    assert torch.allclose(beta_mlp_out, ref_mlp_out,
                          rtol=1e-2, atol=1e-3), (
        f"Phase 3 β kernel MLP diverged from standalone: "
        f"max_abs={max_abs:.3e}; "
        f"beta[0,:8]={beta_mlp_out[0,:8].tolist()}; "
        f"ref[0,:8]={ref_mlp_out[0,:8].tolist()}"
    )


@pytest.mark.skipif(not CUTE_AVAILABLE, reason="CUTLASS CuTe DSL not available")
@pytest.mark.parametrize("emit_next_ln", [True, False])
def test_phase_4_matches_python_epsilon_ref(emit_next_ln):
    """Phase 4 (Task 15): β kernel's ε epilogue matches Python reference.

    Two paths under test:
      - emit_next_ln=True  → next_hidden = RMSNorm(residual_post + mlp_out) * γ
      - emit_next_ln=False → next_hidden = residual_final (memcpy, last layer)

    Also verifies residual_post is updated in-place to residual_final.
    """
    from vllm.v1.attention.backends.cute_paged.phase_e_kernel import (
        PhaseE_Beta_Kernel,
    )
    repro = import_module("2026-04-22-phase-e-repro")

    torch.manual_seed(42)
    nat, hidden = 4, 5120
    device = "cuda"

    beta = PhaseE_Beta_Kernel(
        hidden_size=hidden, intermediate_size=17408,
        num_attn_heads=24, num_kv_heads=4, head_dim=256,
        rms_eps=1e-6,
    )

    residual_post = torch.randn(nat, hidden, dtype=torch.bfloat16,
                                device=device)
    mlp_out = torch.randn(nat, hidden, dtype=torch.bfloat16, device=device)
    next_hidden = torch.zeros(nat, hidden, dtype=torch.bfloat16,
                              device=device)
    next_gamma = (torch.randn(hidden, dtype=torch.bfloat16, device=device)
                  if emit_next_ln else None)

    # Take snapshots BEFORE kernel mutates residual_post.
    residual_post_snapshot = residual_post.clone()

    beta.run_phase_4_only(
        residual_post_ln=residual_post,
        mlp_output=mlp_out,
        next_input_layernorm_gamma=next_gamma,
        next_hidden_output=next_hidden,
        emit_next_layernorm=emit_next_ln,
    )
    torch.cuda.synchronize()

    # Python reference (snapshot → ref, since kernel mutates in-place).
    residual_final_ref, next_hidden_ref = repro.epsilon_epilogue_ref(
        residual_post_snapshot, mlp_out,
        next_gamma=next_gamma if emit_next_ln else None,
        eps=1e-6,
    )

    # --- Assert 1: residual_post updated in-place to residual_final ---
    max_abs_rf = (residual_post.float()
                  - residual_final_ref.float()).abs().max().item()
    assert torch.allclose(residual_post, residual_final_ref,
                          rtol=1e-2, atol=1e-3), (
        f"residual_final (in-place) diverged: max_abs={max_abs_rf:.3e}; "
        f"kernel[0,:8]={residual_post[0,:8].tolist()}; "
        f"ref[0,:8]={residual_final_ref[0,:8].tolist()}"
    )

    # --- Assert 2: next_hidden matches Python ref ---
    if emit_next_ln:
        # Tolerance: rsqrt.approx.f32 (kernel) vs torch.rsqrt (ref) diverge
        # by ~2 ULPs in FP32 → up to 1 BF16 ULP (2^-5 = 0.031) after the
        # * γ_bf16 stage with random γ. Phase_D's test hides this by using
        # γ=ones so the final multiply is a no-op. atol sized for 1 BF16
        # ULP at magnitude ~1.
        max_abs_nh = (next_hidden.float()
                      - next_hidden_ref.float()).abs().max().item()
        assert torch.allclose(next_hidden, next_hidden_ref,
                              rtol=3e-2, atol=5e-2), (
            f"next_hidden (RMSNorm path) diverged: max_abs={max_abs_nh:.3e}; "
            f"kernel[0,:8]={next_hidden[0,:8].tolist()}; "
            f"ref[0,:8]={next_hidden_ref[0,:8].tolist()}"
        )
    else:
        # Last-layer: next_hidden == residual_final (BF16-exact memcpy).
        assert torch.equal(next_hidden, residual_final_ref), (
            f"last-layer next_hidden != residual_final: "
            f"max diff {(next_hidden.float() - residual_final_ref.float()).abs().max().item()}"
        )


@pytest.mark.skipif(not CUTE_AVAILABLE, reason="CUTLASS CuTe DSL not available")
@pytest.mark.parametrize("total_ctas", [2, 8, 32])
def test_phase_2_grid_barrier_stress(total_ctas):
    """Phase 2 (Task 13): cooperative-launch grid barrier.

    Each CTA writes its block-id (as FP32) into scratch[bx], passes
    through the grid barrier, then CTA 0 sums the whole scratch.
    Expected: total_ctas*(total_ctas-1)/2.

    Also serves as the Task 13 Step 6 deadlock smoke — if the barrier
    never releases the kernel hangs, pytest will eventually time out,
    and we fail loudly.
    """
    from vllm.v1.attention.backends.cute_paged.phase_e_kernel import (
        PhaseE_Beta_Kernel,
    )

    k = PhaseE_Beta_Kernel(
        hidden_size=5120, intermediate_size=17408,
        num_attn_heads=24, num_kv_heads=4, head_dim=256,
    )

    scratch, result = k.run_barrier_stress_debug(total_ctas=total_ctas)
    torch.cuda.synchronize()

    expected = float(total_ctas * (total_ctas - 1) // 2)
    got = result.item()
    assert got == expected, (
        f"grid barrier: expected sum={expected}, got {got}; "
        f"scratch={scratch.tolist()}"
    )
    # Also assert every scratch slot is its own bx — proves Phase 1 writes
    # from all CTAs landed, which is what the barrier enforced.
    expected_scratch = torch.arange(total_ctas, dtype=torch.float32,
                                     device=scratch.device)
    assert torch.equal(scratch, expected_scratch), (
        f"scratch mismatch: got {scratch.tolist()}, "
        f"expected {expected_scratch.tolist()}"
    )


@pytest.mark.skipif(not CUTE_AVAILABLE, reason="CUTLASS CuTe DSL not available")
def test_beta_coop_full_matches_beta_lite():
    """Task 16 pre-synthesis: β-coop unified kernel matches β-lite path.

    β-lite baseline (two-kernel dispatch):
      1. `paged_attention_forward(..., wo_*, rmsnorm_*)` → writes
         rmsnorm_output (= attn output after Phase C) + residual_output.
      2. `Phase_D_MLP_Kernel(..., emit_epilogue=True, emit_next_layernorm=True)`
         → consumes rmsnorm_output as x, writes mlp_output + residual_final
         (in-place on residual_output) + next_hidden_output.

    β-coop unified kernel (one launch):
      `PhaseE_Beta_Kernel.run_beta_coop_full(...)` — grid=(8,8,nat)
      cooperative launch. Same byte-for-byte math paths for attn +
      MLP + ε; only difference is that Phase 1 and Phase 3 share one
      kernel context (SMEM aliased; grid barrier between).

    Tolerance sized for the 1-BF16-ULP approx-rsqrt drift verified in
    Task 15 (memory:feedback_graphs_over_eager; Phase 4 test uses
    rtol=3e-2/atol=5e-2 for the RMSNorm path with random γ). β-coop
    adds no new numerical paths beyond those already covered by the
    Phase 1/3/4 standalone tests, so the same tolerance applies.
    """
    from vllm.v1.attention.backends.cute_paged.phase_e_kernel import (
        PhaseE_Beta_Kernel,
    )
    from vllm.v1.attention.backends.cute_paged.kernel import (
        paged_attention_forward,
    )
    from vllm.v1.attention.backends.cute_paged.mlp_kernel import (
        Phase_D_MLP_Kernel,
    )
    from vllm.model_executor.layers.quantization.utils.nvfp4_utils import (
        swizzle_blockscale,
    )

    torch.manual_seed(42)

    # --- Model / config ---
    hidden = 5120
    interm = 17408
    num_q_heads = 24
    num_kv_heads = 4
    head_dim = 256
    group_size = num_q_heads // num_kv_heads  # 6
    page_size = 64
    seq_len = 128   # 2 pages of 64
    nat = 1
    scale = 1.0 / (head_dim ** 0.5)
    device = "cuda"

    # --- Phase 0 / Phase 1 inputs (mirror test_phase_1_matches_standalone_decode) ---
    hidden_in = torch.randn(nat, hidden, dtype=torch.bfloat16, device=device)
    residual_in = torch.randn(nat, hidden, dtype=torch.bfloat16, device=device)
    input_gamma = torch.randn(hidden, dtype=torch.bfloat16, device=device)
    post_attn_gamma = torch.randn(hidden, dtype=torch.bfloat16, device=device)
    query = torch.randn(nat, num_q_heads, head_dim,
                        dtype=torch.bfloat16, device=device)

    num_pages = 2
    kv_cache = torch.zeros(num_pages, 2, page_size, num_kv_heads, head_dim,
                           dtype=torch.uint8, device=device)
    k_float = torch.randn(num_pages, page_size, num_kv_heads, head_dim,
                          device=device).clamp(-10, 10)
    kv_cache[:, 0] = k_float.to(torch.float8_e4m3fn).view(torch.uint8)
    v_float = torch.randn(num_pages, page_size, num_kv_heads, head_dim,
                          device=device).clamp(-10, 10)
    kv_cache[:, 1] = v_float.to(torch.float8_e4m3fn).view(torch.uint8)

    page_table = torch.zeros(nat, 2, dtype=torch.int32, device=device)
    page_table[0, 0] = 0
    page_table[0, 1] = 1
    seq_lens = torch.tensor([seq_len], dtype=torch.int32, device=device)

    # --- NVFP4 W_O (same construction as phase_1 test) ---
    K_dim = num_q_heads * head_dim
    N_dim = hidden
    torch.manual_seed(123)
    wo_nibbles = torch.randint(0, 16, (N_dim, K_dim),
                               dtype=torch.uint8, device=device)
    wo_weight = torch.zeros(N_dim, K_dim // 2,
                            dtype=torch.uint8, device=device)
    for k in range(0, K_dim, 2):
        lo = wo_nibbles[:, k] & 0x0F
        hi = wo_nibbles[:, k + 1] & 0x0F
        wo_weight[:, k // 2] = lo | (hi << 4)
    K_sf = K_dim // 16
    wo_scales_linear = torch.full((N_dim, K_sf), 0x38,
                                   dtype=torch.uint8, device=device)
    wo_scales_linear = wo_scales_linear.view(torch.float8_e4m3fn)
    wo_scales = swizzle_blockscale(wo_scales_linear)
    wo_global_scale = torch.tensor(1.0, dtype=torch.float32, device=device)

    # --- MLP weights (same construction as phase_3 test) ---
    torch.manual_seed(42)
    gate_fp4 = torch.randint(0, 256, (interm, hidden // 2),
                             dtype=torch.uint8, device=device)
    up_fp4 = torch.randint(0, 256, (interm, hidden // 2),
                           dtype=torch.uint8, device=device)
    down_fp4 = torch.randint(0, 256, (hidden, interm // 2),
                             dtype=torch.uint8, device=device)
    gate_scale = torch.full((interm, hidden // 16), 0x38,
                            dtype=torch.uint8, device=device)
    up_scale = torch.full((interm, hidden // 16), 0x38,
                          dtype=torch.uint8, device=device)
    down_scale = torch.full((hidden, interm // 16), 0x38,
                            dtype=torch.uint8, device=device)

    # C1.5: next_gamma allocation removed — Assert 3 (next_hidden) deleted,
    # so lite_mlp is invoked with emit_next_layernorm=False below and the
    # next-layer γ is no longer consumed in this test.

    # ================================================================
    # β-lite reference: two-kernel dispatch.
    # ================================================================
    # Step 1: attn (wo + rmsnorm fused). Writes rmsnorm_output (attn
    # output after Phase C) + residual_output.
    lite_wo_output = torch.zeros(nat, 4, hidden,
                                  dtype=torch.float32, device=device)
    lite_rmsnorm_output = torch.empty(nat, hidden,
                                       dtype=torch.bfloat16, device=device)
    lite_residual_output = torch.empty(nat, hidden,
                                        dtype=torch.bfloat16, device=device)
    lite_arrival_count = torch.zeros(nat, dtype=torch.int32, device=device)

    _ = paged_attention_forward(
        query=query, kv_cache=kv_cache, page_table=page_table,
        seq_lens=seq_lens, scale=scale, k_scale=1.0, v_scale=1.0,
        page_size=page_size,
        wo_weight=wo_weight, wo_scales=wo_scales,
        wo_global_scale=wo_global_scale, wo_output=lite_wo_output,
        rmsnorm_gamma=post_attn_gamma,
        rmsnorm_residual=residual_in,
        rmsnorm_output=lite_rmsnorm_output,
        residual_output=lite_residual_output,
        arrival_count=lite_arrival_count,
        rmsnorm_eps=1e-6,
        padded_num_seqs=nat,
    )
    torch.cuda.synchronize()

    # Step 2: Phase D MLP + ε epilogue. Consumes rmsnorm_output as x;
    # mutates residual_output (= lite_residual_output) in place.
    lite_mlp = Phase_D_MLP_Kernel(hidden_size=hidden, intermediate_size=interm)
    lite_partial = torch.zeros(nat, lite_mlp.slice_ctas, hidden,
                               dtype=torch.float32, device=device)
    lite_arrival = torch.zeros(nat, lite_mlp.num_k_tiles,
                               dtype=torch.uint32, device=device)
    lite_mlp_out = torch.zeros(nat, hidden,
                               dtype=torch.bfloat16, device=device)
    # C1.5: lite_next_hidden buffer is required scratch — Phase_D_MLP_Kernel
    # asserts `next_hidden_output is not None` whenever emit_epilogue=True
    # (mlp_kernel.py:497-498), even when emit_next_layernorm=False. The
    # buffer is written by the kernel but never read by this test after
    # Assert 3 deletion.
    lite_next_hidden = torch.zeros(nat, hidden,
                                    dtype=torch.bfloat16, device=device)

    lite_mlp(
        lite_rmsnorm_output, gate_fp4, gate_scale, up_fp4, up_scale,
        down_fp4, down_scale,
        lite_partial, lite_arrival, lite_mlp_out, nat,
        residual_post_ln=lite_residual_output,
        next_hidden_output=lite_next_hidden,
        emit_epilogue=True,
        emit_next_layernorm=False,
        rms_eps=1e-6,
    )
    torch.cuda.synchronize()

    # ================================================================
    # β-coop unified kernel under test.
    # ================================================================
    beta = PhaseE_Beta_Kernel(
        hidden_size=hidden, intermediate_size=interm,
        num_attn_heads=num_q_heads, num_kv_heads=num_kv_heads,
        head_dim=head_dim, rms_eps=1e-6,
    )

    coop_attn_input_bf16 = torch.zeros(nat, hidden,
                                        dtype=torch.bfloat16, device=device)
    coop_attn_output = torch.zeros(nat, hidden,
                                    dtype=torch.bfloat16, device=device)
    coop_mlp_output = torch.zeros(nat, hidden,
                                   dtype=torch.bfloat16, device=device)
    # C1.5: Phase 4 deleted — coop_next_hidden no longer used.

    # Persistent β-coop workspace buffers (spec 2026-04-30 §4.3).
    # Per-call alloc fine here — test runs without CUDA graph capture.
    coop_wo_output = torch.zeros(nat, 4, hidden,
                                  dtype=torch.float32, device=device)
    coop_mlp_partial_fp32 = torch.zeros(
        nat, beta.slice_ctas, hidden,
        dtype=torch.float32, device=device,
    )
    coop_mlp_arrival_count = torch.zeros(
        nat, beta.num_k_tiles, dtype=torch.uint32, device=device,
    )
    coop_grid_barrier_i32 = torch.zeros(
        nat, dtype=torch.int32, device=device,
    )
    coop_phase1_arrival_count = torch.zeros(
        nat, dtype=torch.int32, device=device,
    )

    beta.run_beta_coop_full(
        hidden_in=hidden_in,
        residual_in=residual_in,
        input_gamma=input_gamma,
        post_attn_gamma=post_attn_gamma,
        attn_input_bf16=coop_attn_input_bf16,
        query=query,
        kv_cache=kv_cache,
        page_table=page_table,
        seq_lens=seq_lens,
        wo_weight=wo_weight,
        wo_scales=wo_scales,
        wo_global_scale=wo_global_scale,
        attn_output=coop_attn_output,
        gate_w_fp4=gate_fp4, gate_w_scale=gate_scale,
        up_w_fp4=up_fp4, up_w_scale=up_scale,
        down_w_fp4=down_fp4, down_w_scale=down_scale,
        mlp_output=coop_mlp_output,
        scale=scale, k_scale=1.0, v_scale=1.0,
        wo_output=coop_wo_output,
        mlp_partial_fp32=coop_mlp_partial_fp32,
        mlp_arrival_count=coop_mlp_arrival_count,
        grid_barrier_i32=coop_grid_barrier_i32,
        phase1_arrival_count=coop_phase1_arrival_count,
    )
    torch.cuda.synchronize()

    # --- Assert 1: attn_output (Phase 1 Phase C RMSNorm) matches β-lite ---
    max_abs_attn = (coop_attn_output.float()
                    - lite_rmsnorm_output.float()).abs().max().item()
    assert torch.allclose(coop_attn_output, lite_rmsnorm_output,
                          rtol=1e-2, atol=1e-3), (
        f"β-coop attn_output diverged from β-lite: max_abs={max_abs_attn:.3e}; "
        f"coop[0,:8]={coop_attn_output[0,:8].tolist()}; "
        f"lite[0,:8]={lite_rmsnorm_output[0,:8].tolist()}"
    )

    # --- Assert 2: mlp_output matches β-lite ---
    max_abs_mlp = (coop_mlp_output.float()
                   - lite_mlp_out.float()).abs().max().item()
    assert torch.allclose(coop_mlp_output, lite_mlp_out,
                          rtol=5e-2, atol=5e-2), (
        f"β-coop mlp_output diverged from β-lite: max_abs={max_abs_mlp:.3e}; "
        f"coop[0,:8]={coop_mlp_output[0,:8].tolist()}; "
        f"lite[0,:8]={lite_mlp_out[0,:8].tolist()}"
    )
    # C1.5: Assert 3 (next_hidden) deleted — Phase 4 deleted from kernel.
