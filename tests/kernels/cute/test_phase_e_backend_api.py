"""Unit test: attach_next_input_layernorm stores module ref correctly
and allocates phase_e workspace.

The production path gets `num_hidden_layers` via `get_current_vllm_config()`,
which is only live inside `set_current_vllm_config()` context (entered by
vLLM's model init). These tests construct the impl via `__new__` to skip
`__init__`, so they monkeypatch the config resolver with a stub instead of
standing up a real VllmConfig.
"""
from types import SimpleNamespace

import pytest
import torch

# Every test in this file allocates CUDA tensors. Skip the whole module
# on hosts without CUDA rather than erroring on first allocation.
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA not available"
)


def _stub_cfg(num_hidden_layers: int = 64):
    """Minimal vllm-config stub — only fields attach_next_input_layernorm
    reaches into."""
    return SimpleNamespace(
        model_config=SimpleNamespace(
            hf_text_config=SimpleNamespace(
                num_hidden_layers=num_hidden_layers,
            )
        )
    )


def test_attach_next_input_layernorm_stores_module_ref(monkeypatch):
    from vllm.v1.attention.backends.cute_paged._backend import (
        CutePagedAttentionImpl,
    )
    # Stub get_current_vllm_config so the workspace allocation can resolve
    # num_hidden_layers without a real VllmConfig context. The production
    # import lives inside the method body (`_backend.py: attach_next_input_
    # layernorm`), so the monkeypatch target is the `vllm.config` module
    # attribute — if the import is ever hoisted to module scope, switch the
    # target to `vllm.v1.attention.backends.cute_paged._backend.get_current_vllm_config`.
    monkeypatch.setattr(
        'vllm.config.get_current_vllm_config', lambda: _stub_cfg(64),
    )

    # Minimal impl construction — skip __init__ for this unit test
    impl = CutePagedAttentionImpl.__new__(CutePagedAttentionImpl)
    impl._fusion_attached = True
    impl._fusion_max_num_seqs = 128
    impl._fusion_hidden_dim = 5120

    mock_next_norm = torch.nn.Module()
    mock_next_norm.weight = torch.nn.Parameter(
        torch.ones(5120, dtype=torch.bfloat16, device='cuda'))
    mock_next_norm.variance_epsilon = 1e-6

    impl.attach_next_input_layernorm(mock_next_norm)

    assert impl._next_input_layernorm_module is mock_next_norm, \
        "Module ref not stored"
    assert impl._emit_next_layernorm is True, \
        "emit flag should be True for a real module"
    assert hasattr(impl, 'phase_e_barrier'), \
        "phase_e_barrier not allocated"
    assert impl.phase_e_barrier.shape == (64,), \
        f"barrier shape should be (num_hidden_layers,); got {tuple(impl.phase_e_barrier.shape)}"
    assert impl.phase_e_barrier.dtype == torch.int32
    assert impl.phase_e_barrier.device.type == 'cuda'
    assert (impl.phase_e_barrier == 0).all(), \
        "barrier must be zero-initialized (reset-before-launch protocol)"
    assert hasattr(impl, 'next_hidden_scratch'), \
        "next_hidden_scratch not allocated"
    assert impl.next_hidden_scratch.shape == (128, 5120)
    assert impl.next_hidden_scratch.dtype == torch.bfloat16
    assert impl.next_hidden_scratch.device.type == 'cuda'


def test_attach_next_input_layernorm_none_is_last_layer(monkeypatch):
    """Last decoder layer (63) passes None — `_emit_next_layernorm` flips False."""
    from vllm.v1.attention.backends.cute_paged._backend import (
        CutePagedAttentionImpl,
    )
    monkeypatch.setattr(
        'vllm.config.get_current_vllm_config', lambda: _stub_cfg(64),
    )

    impl = CutePagedAttentionImpl.__new__(CutePagedAttentionImpl)
    impl._fusion_attached = True
    impl._fusion_max_num_seqs = 128
    impl._fusion_hidden_dim = 5120

    impl.attach_next_input_layernorm(None)

    assert impl._next_input_layernorm_module is None
    assert impl._emit_next_layernorm is False, \
        "last-layer flag not set"


def test_attach_next_input_layernorm_requires_attach_fusion_first():
    """Precondition guard: calling before attach_fusion raises AssertionError."""
    from vllm.v1.attention.backends.cute_paged._backend import (
        CutePagedAttentionImpl,
    )
    impl = CutePagedAttentionImpl.__new__(CutePagedAttentionImpl)
    # Note: _fusion_attached is INTENTIONALLY not set — mimics a misordered call.

    with pytest.raises(AssertionError, match="attach_fusion must run first"):
        impl.attach_next_input_layernorm(None)
