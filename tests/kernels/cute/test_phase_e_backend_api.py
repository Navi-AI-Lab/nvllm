"""Unit tests for CutePagedAttentionImpl env-parser and resident-cap probe.

The four `attach_next_input_layernorm` tests that used to live here were
removed when C1.5 (commit 54da780f3) disabled the cross-layer
input_layernorm bake. The `attach_next_input_layernorm` /
`attach_input_layernorm` methods are commented out in `_backend.py` per
the comment-out-kernel-code rule. The next layer's input_LN runs
unconditionally from Python at layer entry (qwen3_5.py:Qwen3_5DecoderLayer.forward).

Keep these surviving tests until the cross-layer pattern returns or the
env-parser/resident-cap probe is renamed.
"""
import pytest
import torch

# Every test in this file allocates CUDA tensors. Skip the whole module
# on hosts without CUDA rather than erroring on first allocation.
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA not available"
)


def test_resident_cap_is_positive_and_bounded(monkeypatch):
    """Smoke: cap probe returns a sane number (not 0, not MAX_INT)."""
    from vllm.v1.attention.backends.cute_paged._backend import (
        CutePagedAttentionImpl,
    )
    impl = CutePagedAttentionImpl.__new__(CutePagedAttentionImpl)

    # Stub the kernel function — use None → probe falls back to
    # SMEM-only occupancy estimate
    dummy_fn = None

    cap = impl._probe_resident_cap(dummy_fn, num_threads=128, smem_bytes=45568)
    num_sms = torch.cuda.get_device_properties(0).multi_processor_count
    assert 0 < cap <= 16 * num_sms, (
        f"cap={cap} num_sms={num_sms} — implausible result"
    )


def test_phase_e_env_config_defaults(monkeypatch):
    """Defaults: fusion=0, path=auto, layers=None."""
    from vllm.v1.attention.backends.cute_paged._backend import (
        _phase_e_env_config,
    )
    for k in ("CUTE_PHASE_E_FUSION", "CUTE_PHASE_E_PATH", "CUTE_PHASE_E_LAYERS"):
        monkeypatch.delenv(k, raising=False)
    cfg = _phase_e_env_config()
    assert cfg.enabled is False
    assert cfg.forced_path == "auto"
    assert cfg.restricted_layers is None


def test_phase_e_env_config_restricted_layers(monkeypatch):
    from vllm.v1.attention.backends.cute_paged._backend import (
        _phase_e_env_config,
    )
    monkeypatch.setenv("CUTE_PHASE_E_FUSION", "1")
    monkeypatch.setenv("CUTE_PHASE_E_LAYERS", "3,7,11")
    cfg = _phase_e_env_config()
    assert cfg.enabled is True
    assert cfg.restricted_layers == {3, 7, 11}


def test_phase_e_env_config_forced_path(monkeypatch):
    from vllm.v1.attention.backends.cute_paged._backend import (
        _phase_e_env_config,
    )
    monkeypatch.setenv("CUTE_PHASE_E_PATH", "lite")
    cfg = _phase_e_env_config()
    assert cfg.forced_path == "lite"
