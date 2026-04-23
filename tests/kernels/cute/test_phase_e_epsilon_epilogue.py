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
