"""Phase E.1 follow-up #1 — β-coop compile cache shared across instances.

Root cause of the ~6 min cold-start stall: each of the 16 full_attention
layers in Qwen3.5-27B gets its own ``PhaseE_Beta_Kernel`` instance, each
with its own ``self._compiled_phase_coop_full = None``. First request
fires ``cute.compile`` 16 times (~23 s each → ~6 min total) even though
every instance shares identical constexpr config.

Fix: module-level cache keyed by the constexpr config tuple so all
matching instances reuse one compiled kernel.

Related evidence:
    memory: project_phase_e_shipped.md (Phase E.1 follow-up #1)
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from vllm.v1.attention.backends.cute_paged import phase_e_kernel as mod
from vllm.v1.attention.backends.cute_paged.phase_e_kernel import (
    PhaseE_Beta_Kernel,
)


_QWEN35_27B_COOP_CONFIG = dict(
    hidden_size=5120,
    intermediate_size=17408,
    num_attn_heads=40,
    num_kv_heads=8,
    head_dim=128,
    rms_eps=1e-6,
)


@pytest.fixture
def clean_cache():
    """Snapshot + restore the module-level compile cache around each test."""
    cache_name = "_PHASE_E_COOP_FULL_COMPILE_CACHE"
    had_cache = hasattr(mod, cache_name)
    saved = dict(getattr(mod, cache_name, {})) if had_cache else None
    if had_cache:
        getattr(mod, cache_name).clear()
    yield
    if had_cache:
        getattr(mod, cache_name).clear()
        getattr(mod, cache_name).update(saved)


def test_coop_full_cache_dict_exists():
    """The module exposes a process-wide compile cache dict."""
    assert hasattr(mod, "_PHASE_E_COOP_FULL_COMPILE_CACHE"), (
        "Expected module-level _PHASE_E_COOP_FULL_COMPILE_CACHE dict "
        "for sharing cute.compile results across PhaseE_Beta_Kernel "
        "instances (Phase E.1 follow-up #1)."
    )
    assert isinstance(mod._PHASE_E_COOP_FULL_COMPILE_CACHE, dict)


def test_coop_full_compile_key_matches_for_identical_config():
    """Two kernels with identical constexpr config produce identical keys."""
    k1 = PhaseE_Beta_Kernel(**_QWEN35_27B_COOP_CONFIG)
    k2 = PhaseE_Beta_Kernel(**_QWEN35_27B_COOP_CONFIG)
    assert hasattr(k1, "_coop_full_compile_key"), (
        "Expected PhaseE_Beta_Kernel._coop_full_compile_key() helper."
    )
    assert k1._coop_full_compile_key() == k2._coop_full_compile_key()


def test_coop_full_compile_key_differs_for_different_config():
    """Kernels with different hidden_size produce different keys.

    Alt config uses hidden_size=3840 (multiple of tile_k=640 default
    preset) and intermediate_size=13056 (multiple of tile_s=256).
    """
    cfg_a = dict(_QWEN35_27B_COOP_CONFIG)
    cfg_b = dict(_QWEN35_27B_COOP_CONFIG, hidden_size=3840,
                 intermediate_size=13056)
    k_a = PhaseE_Beta_Kernel(**cfg_a)
    k_b = PhaseE_Beta_Kernel(**cfg_b)
    assert k_a._coop_full_compile_key() != k_b._coop_full_compile_key()


def test_coop_full_compile_fires_once_across_instances(
    clean_cache, monkeypatch,
):
    """16 instances with identical config trigger cute.compile exactly once.

    This is the behaviour that kills the ~6 min cold-start stall: on main,
    cute.compile fires 16 times (once per layer attachment). After the
    fix, the module-level cache dedupes across instances.
    """
    call_count = {"n": 0}

    def fake_compile(fn, *args, **kwargs):
        call_count["n"] += 1
        return MagicMock(name=f"compiled_coop_full_{call_count['n']}")

    monkeypatch.setattr(mod.cute, "compile", fake_compile)

    N_LAYERS = 16
    kernels = [
        PhaseE_Beta_Kernel(**_QWEN35_27B_COOP_CONFIG) for _ in range(N_LAYERS)
    ]
    # Dummy compile args — the patched cute.compile ignores them.
    dummy_args = tuple(range(50))
    for k in kernels:
        k._compile_coop_full(*dummy_args)

    assert call_count["n"] == 1, (
        f"Expected 1 cute.compile call across {N_LAYERS} matching "
        f"instances, got {call_count['n']}"
    )


def test_coop_full_compile_fires_per_distinct_config(
    clean_cache, monkeypatch,
):
    """Two instances with different constexpr config each get their own
    compile — the cache keys them apart correctly.
    """
    call_count = {"n": 0}

    def fake_compile(fn, *args, **kwargs):
        call_count["n"] += 1
        return MagicMock(name=f"compiled_coop_full_{call_count['n']}")

    monkeypatch.setattr(mod.cute, "compile", fake_compile)

    k_a = PhaseE_Beta_Kernel(**_QWEN35_27B_COOP_CONFIG)
    k_b = PhaseE_Beta_Kernel(
        **dict(_QWEN35_27B_COOP_CONFIG, hidden_size=3840,
               intermediate_size=13056)
    )
    dummy_args = tuple(range(50))
    k_a._compile_coop_full(*dummy_args)
    k_b._compile_coop_full(*dummy_args)

    assert call_count["n"] == 2


def test_coop_full_compile_populates_instance_attr(
    clean_cache, monkeypatch,
):
    """After the cache hit, ``self._compiled_phase_coop_full`` is the
    cached callable so back-compat readers see the same handle.
    """
    sentinel = MagicMock(name="cached_coop_full")

    def fake_compile(fn, *args, **kwargs):
        return sentinel

    monkeypatch.setattr(mod.cute, "compile", fake_compile)

    k1 = PhaseE_Beta_Kernel(**_QWEN35_27B_COOP_CONFIG)
    k2 = PhaseE_Beta_Kernel(**_QWEN35_27B_COOP_CONFIG)
    dummy_args = tuple(range(50))

    k1._compile_coop_full(*dummy_args)
    k2._compile_coop_full(*dummy_args)

    assert k1._compiled_phase_coop_full is sentinel
    assert k2._compiled_phase_coop_full is sentinel
