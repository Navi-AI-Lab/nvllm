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
    summed = hidden_in.float() + residual_in.float()
    variance = summed.pow(2).mean(dim=-1, keepdim=True)
    rstd = torch.rsqrt(variance + 1e-6)
    normed_ref = ((summed * rstd) * gamma.float()).to(torch.bfloat16)

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

    # --- Assert 1: Phase 0 output matches Python RMSNorm(h+r)*γ_in ---
    summed = hidden_in.float() + residual_in.float()
    variance = summed.pow(2).mean(dim=-1, keepdim=True)
    rstd = torch.rsqrt(variance + 1e-6)
    attn_input_ref = ((summed * rstd) * input_gamma.float()).to(
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
