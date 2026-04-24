"""Phase F.1 — cute_phase_e_dispatch + cute_phase_e_skip_input_layernorm
custom-op integration tests.

Verifies the real ops live in _mlp_op.py, register cleanly, branch
correctly on impl._phase_e_consumed / impl._phase_e_skip_next_ln, and
fail-loud on missing registry entries.

The skip-op's "runs when flag unset" test stubs the input_layernorm
module so we don't require the vLLM config context (Qwen3_5RMSNorm
constructor calls get_current_vllm_config()). The stub mirrors the
real module's surface: .weight (Parameter), .variance_epsilon (float),
._forward_static_with_residual (static method).
"""
import types

import pytest
import torch
from torch import nn

# Registration happens on import.
from vllm.v1.attention.backends.cute_paged import _mlp_op  # noqa: F401
from vllm.v1.attention.backends.cute_paged._mlp_op import _CUTE_MLP_REGISTRY
from vllm.nvllm.layers.layernorm import Qwen3_5RMSNorm


class _FakeImpl:
    """Minimal stand-in for CutePagedAttentionImpl state."""
    def __init__(self, nat, hidden, device):
        self._phase_e_consumed = False
        self._phase_e_skip_next_ln = False
        self.next_hidden_scratch = torch.full(
            (nat, hidden), 7.0, dtype=torch.bfloat16, device=device
        )
        self.residual_output = torch.full(
            (nat, hidden), 9.0, dtype=torch.bfloat16, device=device
        )


def test_cute_phase_e_dispatch_consumes_when_flag_set():
    device = 'cuda'
    nat, hidden = 4, 5120
    name = "test_layer_consume"
    impl = _FakeImpl(nat, hidden, device)
    impl._phase_e_consumed = True
    _CUTE_MLP_REGISTRY[name] = impl
    try:
        x = torch.randn(nat, hidden, dtype=torch.bfloat16, device=device)
        hidden_out = torch.zeros(
            nat, hidden, dtype=torch.bfloat16, device=device
        )
        residual_out = torch.zeros_like(hidden_out)
        residual_in = torch.randn(
            nat, hidden, dtype=torch.bfloat16, device=device
        )

        torch.ops.vllm.cute_phase_e_dispatch(
            x, hidden_out, residual_out, residual_in, name
        )

        assert torch.allclose(hidden_out, impl.next_hidden_scratch)
        assert torch.allclose(residual_out, impl.residual_output)
        assert impl._phase_e_consumed is False, (
            "consume flag not cleared after consume"
        )
        assert impl._phase_e_skip_next_ln is True, (
            "skip flag for next layer not set — "
            "layer N+1 will double-process"
        )
    finally:
        del _CUTE_MLP_REGISTRY[name]


def test_cute_phase_e_dispatch_fails_loud_on_unknown_layer():
    x = torch.zeros(4, 5120, dtype=torch.bfloat16, device='cuda')
    h = torch.zeros_like(x)
    r = torch.zeros_like(x)
    rin = torch.zeros_like(x)
    with pytest.raises(RuntimeError, match="unregistered"):
        torch.ops.vllm.cute_phase_e_dispatch(
            x, h, r, rin, "nonexistent_layer_dispatch"
        )


def test_cute_phase_e_skip_input_layernorm_skips_when_flag_set():
    device = 'cuda'
    nat, hidden = 4, 5120
    name = "test_layer_skip"
    impl = _FakeImpl(nat, hidden, device)
    impl._phase_e_skip_next_ln = True
    _CUTE_MLP_REGISTRY[name] = impl
    try:
        x = torch.randn(nat, hidden, dtype=torch.bfloat16, device=device)
        residual = torch.randn(
            nat, hidden, dtype=torch.bfloat16, device=device
        )
        out_x = torch.zeros_like(x)
        out_r = torch.zeros_like(residual)

        torch.ops.vllm.cute_phase_e_skip_input_layernorm(
            x, residual, out_x, out_r, name
        )

        # Skip branch: pass-through.
        assert torch.allclose(out_x, x)
        assert torch.allclose(out_r, residual)
        assert impl._phase_e_skip_next_ln is False, (
            "skip flag not cleared after skip"
        )
    finally:
        del _CUTE_MLP_REGISTRY[name]


def test_cute_phase_e_skip_input_layernorm_runs_when_flag_unset():
    """When skip flag is False, must run impl._input_layernorm_module
    via its _forward_static_with_residual staticmethod.

    Stubs the input_ln module to avoid Qwen3_5RMSNorm's vLLM-config
    requirement. The op only touches .weight, .variance_epsilon, and
    ._forward_static_with_residual — surface fully replicable with a
    SimpleNamespace + Parameter + staticmethod ref.
    """
    device = 'cuda'
    nat, hidden = 4, 5120
    name = "test_layer_run"
    impl = _FakeImpl(nat, hidden, device)
    impl._phase_e_skip_next_ln = False

    # Stub input_layernorm: same surface as Qwen3_5RMSNorm without the
    # vLLM config dependency.
    weight_tensor = (
        torch.randn(hidden, dtype=torch.bfloat16, device=device) * 0.02
    )
    stub = types.SimpleNamespace(
        weight=nn.Parameter(weight_tensor),
        variance_epsilon=1e-6,
        _forward_static_with_residual=Qwen3_5RMSNorm._forward_static_with_residual,
    )
    impl._input_layernorm_module = stub
    _CUTE_MLP_REGISTRY[name] = impl
    try:
        x = torch.randn(nat, hidden, dtype=torch.bfloat16, device=device)
        residual = torch.randn(
            nat, hidden, dtype=torch.bfloat16, device=device
        )
        out_x = torch.zeros_like(x)
        out_r = torch.zeros_like(residual)

        torch.ops.vllm.cute_phase_e_skip_input_layernorm(
            x, residual, out_x, out_r, name
        )

        ref_x, ref_r = Qwen3_5RMSNorm._forward_static_with_residual(
            stub.weight.data, stub.variance_epsilon, x, residual
        )
        assert torch.allclose(out_x, ref_x, atol=1e-3, rtol=1e-3)
        assert torch.allclose(out_r, ref_r, atol=1e-3, rtol=1e-3)
    finally:
        del _CUTE_MLP_REGISTRY[name]


def test_cute_phase_e_skip_input_layernorm_fails_loud_on_unknown_layer():
    x = torch.zeros(4, 5120, dtype=torch.bfloat16, device='cuda')
    r = torch.zeros_like(x)
    out_x = torch.zeros_like(x)
    out_r = torch.zeros_like(x)
    with pytest.raises(RuntimeError, match="unregistered"):
        torch.ops.vllm.cute_phase_e_skip_input_layernorm(
            x, r, out_x, out_r, "nonexistent_layer_skip"
        )
