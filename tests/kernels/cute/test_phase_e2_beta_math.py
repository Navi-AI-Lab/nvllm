"""Phase E.2 — β kernel math correctness against Qwen3_5RMSNorm reference.

Catches the raw-γ-vs-(1+γ) bug that the existing
test_phase_e_epsilon_epilogue.py missed because its reference harness
had the same bug as the kernel.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "docs/research"))

import pytest
import torch

CUTE_AVAILABLE = True
try:
    from vllm.v1.attention.backends.cute_paged.mlp_kernel import (
        Phase_D_MLP_Kernel,
    )
    from vllm.nvllm.layers.layernorm import Qwen3_5RMSNorm
except ImportError:
    CUTE_AVAILABLE = False


@pytest.mark.skipif(
    not CUTE_AVAILABLE, reason="CUTLASS CuTe DSL not available"
)
def test_beta_lite_epsilon_matches_qwen35_rmsnorm_forward_native():
    """β-lite ε epilogue's next_hidden output must match
    Qwen3_5RMSNorm.forward_native(residual_final, None) for the next
    layer's γ (the "no prior residual" no-residual-add case).

    Pass criterion: torch.allclose(rtol=3e-2, atol=5e-2) BF16.

    Tolerance matches existing test_phase_e_epsilon_epilogue.py line 511
    and is documented there: rsqrt.approx.f32 (kernel) vs torch.rsqrt
    (ref) diverges ~2 FP32 ULPs → up to 1 BF16 ULP (2^-5 ≈ 0.031) after
    γ multiply with random γ. With the raw-γ bug, max_diff was ~4.2
    (order of |residual| × |γ_max|) — regressions fail by >>1 ULP.
    """
    nat, hidden, interm = 4, 5120, 17408
    device = 'cuda'

    # Random trained-range γ (Qwen stores γ such that γ ≈ 0 is the identity
    # because the model does `x * (1 + γ)`).
    next_gamma = (torch.randn(hidden, dtype=torch.bfloat16, device=device)
                  * 0.02)  # typical trained-γ stddev ~0.02

    # Random residual_final (β would have computed this from residual_post +
    # mlp_out; for this test we construct it directly and zero MLP weights).
    residual_final = torch.randn(
        nat, hidden, dtype=torch.bfloat16, device=device
    )

    # Call the staticmethod directly — Qwen3_5RMSNorm.__init__ would need
    # a vLLM config context (CustomOp base class), but the math is a pure
    # staticmethod taking (weight, eps, x), so we bypass module construction.
    ref_next_hidden = Qwen3_5RMSNorm._forward_static_no_residual(
        next_gamma, 1e-6, residual_final
    )

    # Invoke β-lite kernel via Phase_D_MLP_Kernel with zero MLP weights
    # so mlp_out = 0, residual_post = residual_final, ε epilogue runs.
    kernel = Phase_D_MLP_Kernel(
        hidden_size=hidden, intermediate_size=interm
    )
    zero_fp4_shape = (interm, hidden // 2)
    zero_fp4_down = (hidden, interm // 2)
    zero_sc_shape = (interm, hidden // 16)
    zero_sc_down = (hidden, interm // 16)

    gate_fp4 = torch.zeros(*zero_fp4_shape, dtype=torch.uint8, device=device)
    up_fp4 = torch.zeros(*zero_fp4_shape, dtype=torch.uint8, device=device)
    down_fp4 = torch.zeros(*zero_fp4_down, dtype=torch.uint8, device=device)
    gate_sc = torch.zeros(*zero_sc_shape, dtype=torch.uint8, device=device)
    up_sc = torch.zeros(*zero_sc_shape, dtype=torch.uint8, device=device)
    down_sc = torch.zeros(*zero_sc_down, dtype=torch.uint8, device=device)
    partial = torch.zeros(nat, 8, hidden, dtype=torch.float32, device=device)
    arrival = torch.zeros(nat, 8, dtype=torch.uint32, device=device)
    mlp_out = torch.zeros(nat, hidden, dtype=torch.bfloat16, device=device)
    next_hidden = torch.zeros(
        nat, hidden, dtype=torch.bfloat16, device=device
    )
    x = torch.randn(nat, hidden, dtype=torch.bfloat16, device=device)

    kernel(
        x, gate_fp4, gate_sc, up_fp4, up_sc, down_fp4, down_sc,
        partial, arrival, mlp_out, nat,
        residual_post_ln=residual_final,  # residual_post + mlp_out=0 = residual_final
        next_input_layernorm_gamma=next_gamma,
        next_hidden_output=next_hidden,
        emit_epilogue=True,
        emit_next_layernorm=True,
        rms_eps=1e-6,
    )

    max_diff = (next_hidden - ref_next_hidden).abs().max().item()
    assert torch.allclose(
        next_hidden, ref_next_hidden, rtol=3e-2, atol=5e-2
    ), (
        f"β-lite ε epilogue does not match Qwen3_5RMSNorm.forward_native. "
        f"Max diff: {max_diff}. "
        f"Most likely cause: kernel multiplies by raw γ instead of (1+γ) "
        f"at mlp_kernel.py:1502."
    )


@pytest.mark.skipif(
    not CUTE_AVAILABLE, reason="CUTLASS CuTe DSL not available"
)
def test_beta_coop_phase0_matches_qwen35_rmsnorm_forward_native():
    """β-coop Phase 0 input_layernorm prologue must match
    Qwen3_5RMSNorm._forward_static_with_residual — i.e., normed output
    is `(hidden + residual) / rms * (1 + γ)`.

    Targets phase_e_kernel.py:641 (run_phase_0_only). Same tolerance
    rationale as the β-lite test — rsqrt.approx.f32 ULP drift after γ multiply.
    """
    from vllm.v1.attention.backends.cute_paged.phase_e_kernel import (
        PhaseE_Beta_Kernel,
    )

    nat, hidden = 4, 5120
    device = 'cuda'

    gamma = (torch.randn(hidden, dtype=torch.bfloat16, device=device)
             * 0.02)  # trained-γ stddev
    hidden_in = torch.randn(
        nat, hidden, dtype=torch.bfloat16, device=device
    )
    residual_in = torch.randn(
        nat, hidden, dtype=torch.bfloat16, device=device
    )
    normed_out = torch.zeros_like(hidden_in)

    # Reference: Qwen3_5RMSNorm static — returns (normed, residual_post_add).
    ref_normed, _ref_residual = (
        Qwen3_5RMSNorm._forward_static_with_residual(
            gamma, 1e-6, hidden_in, residual_in
        )
    )

    # Kernel: run_phase_0_only writes normed_out only.
    kernel = PhaseE_Beta_Kernel(
        hidden_size=hidden, intermediate_size=17408,
        num_attn_heads=24, num_kv_heads=4, head_dim=256,
        rms_eps=1e-6,
    )
    kernel.run_phase_0_only(hidden_in, residual_in, gamma, normed_out)
    torch.cuda.synchronize()

    max_diff = (normed_out - ref_normed).abs().max().item()
    assert torch.allclose(
        normed_out, ref_normed, rtol=3e-2, atol=5e-2
    ), (
        f"β-coop Phase 0 does not match Qwen3_5RMSNorm.forward_native. "
        f"Max diff: {max_diff}. "
        f"Most likely cause: kernel multiplies by raw γ instead of (1+γ) "
        f"at phase_e_kernel.py:641."
    )


@pytest.mark.skipif(
    not CUTE_AVAILABLE, reason="CUTLASS CuTe DSL not available"
)
def test_beta_coop_phase4_epsilon_matches_qwen35_rmsnorm_forward_native():
    """β-coop Phase 4 ε epilogue must match
    Qwen3_5RMSNorm._forward_static_no_residual on residual_final =
    residual_post + mlp_output. Targets phase_e_kernel.py:2625
    (run_phase_4_only) — and through identical pattern, also exercises
    the same fix at line 4641 (run_beta_coop_full).
    """
    from vllm.v1.attention.backends.cute_paged.phase_e_kernel import (
        PhaseE_Beta_Kernel,
    )

    nat, hidden = 4, 5120
    device = 'cuda'

    next_gamma = (torch.randn(hidden, dtype=torch.bfloat16, device=device)
                  * 0.02)
    residual_post = torch.randn(
        nat, hidden, dtype=torch.bfloat16, device=device
    )
    mlp_output = torch.randn(
        nat, hidden, dtype=torch.bfloat16, device=device
    )
    next_hidden_out = torch.zeros_like(residual_post)

    # Snapshot before kernel mutates residual_post in-place.
    residual_final_ref = residual_post + mlp_output

    # Reference: forward_native on residual_final, no prior residual.
    ref_next_hidden = Qwen3_5RMSNorm._forward_static_no_residual(
        next_gamma, 1e-6, residual_final_ref
    )

    kernel = PhaseE_Beta_Kernel(
        hidden_size=hidden, intermediate_size=17408,
        num_attn_heads=24, num_kv_heads=4, head_dim=256,
        rms_eps=1e-6,
    )
    kernel.run_phase_4_only(
        residual_post_ln=residual_post,
        mlp_output=mlp_output,
        next_input_layernorm_gamma=next_gamma,
        next_hidden_output=next_hidden_out,
        emit_next_layernorm=True,
    )
    torch.cuda.synchronize()

    max_diff = (next_hidden_out - ref_next_hidden).abs().max().item()
    assert torch.allclose(
        next_hidden_out, ref_next_hidden, rtol=3e-2, atol=5e-2
    ), (
        f"β-coop Phase 4 ε does not match Qwen3_5RMSNorm.forward_native. "
        f"Max diff: {max_diff}. "
        f"Most likely cause: kernel multiplies by raw γ instead of (1+γ) "
        f"at phase_e_kernel.py:2625 (and run_beta_coop_full at 4641)."
    )
