"""Phase F.1 — `cute_phase_e_dispatch` custom-op integration tests.

C1.5 (commit 54da780f3) retired the `cute_phase_e_skip_input_layernorm`
op and the cross-layer LN bake it enabled. Layer `input_layernorm` now
runs unconditionally at every layer entry from Python; this file
exercises only the surviving `cute_phase_e_dispatch` op.

Post-C1.5 consume contract:
  * `hidden_out[:nat]` ← `impl.mlp_output[:nat]` (raw post-MLP hidden;
    the F.1 next-layer LN bake into `next_hidden_scratch` is gone)
  * `residual_out[:nat]` ← `impl.residual_output[:nat]`
  * `impl._phase_e_consumed` flips False after consume
"""
import pytest
import torch

# Registration happens on import.
from vllm.v1.attention.backends.cute_paged import _mlp_op  # noqa: F401
from vllm.v1.attention.backends.cute_paged._mlp_op import _CUTE_MLP_REGISTRY


class _FakeImpl:
    """Minimal stand-in for CutePagedAttentionImpl state."""

    def __init__(self, nat, hidden, device):
        self._phase_e_consumed = False
        self.mlp_output = torch.full(
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

        # Post-C1.5: consume reads impl.mlp_output (NOT next_hidden_scratch
        # — that was the Phase F.1 LN-baked path, retired in C1.5).
        assert torch.allclose(hidden_out, impl.mlp_output)
        assert torch.allclose(residual_out, impl.residual_output)
        assert impl._phase_e_consumed is False, (
            "consume flag not cleared after consume"
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
