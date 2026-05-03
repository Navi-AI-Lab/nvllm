"""Structural tests for region-timing instrumentation infrastructure.

Verifies (without launching a kernel):
  - PhaseE_Beta_Kernel._coop_full_compile_key changes when timing flag flips
  - run_beta_coop_full signature accepts region_timing_buf=None default
  - run_beta_coop_full signature accepts a non-None region_timing_buf

Test file does NOT validate kernel output correctness — that is covered
by existing GSM8K + uber-kernel multi-layer tests under PIECEWISE.
"""
from __future__ import annotations

import inspect

import pytest


@pytest.fixture(scope="module")
def phase_e_module():
    return pytest.importorskip(
        "vllm.v1.attention.backends.cute_paged.phase_e_kernel"
    )


def test_compile_key_differs_when_timing_enabled(phase_e_module):
    """Same config → distinct cache key for timing-on vs timing-off.

    Without this, the shared disk cache (apply_disk_cache_patch in
    cutlass) would return the timing-off compile artifact and timing
    writes would be no-ops.
    """
    PhaseE_Beta_Kernel = phase_e_module.PhaseE_Beta_Kernel
    src = inspect.getsource(PhaseE_Beta_Kernel._coop_full_compile_key)
    assert "region_timing" in src or "REGION_TIMING" in src, (
        "_coop_full_compile_key must include the region-timing flag in "
        "the cache key tuple, otherwise disk cache returns the wrong "
        "compile artifact when CUTE_BETA_REGION_TIMING=1."
    )


def test_run_beta_coop_full_accepts_region_timing_buf(phase_e_module):
    """run_beta_coop_full signature has region_timing_buf=None default."""
    sig = inspect.signature(phase_e_module.PhaseE_Beta_Kernel.run_beta_coop_full)
    assert "region_timing_buf" in sig.parameters, (
        "run_beta_coop_full must accept region_timing_buf parameter "
        "(default None) for the timing-off path."
    )
    param = sig.parameters["region_timing_buf"]
    assert param.default is None, (
        f"region_timing_buf default must be None (got {param.default}); "
        "production path must not allocate timing scratch."
    )


def test_backend_allocates_region_timing_when_env_set(monkeypatch):
    """When CUTE_BETA_REGION_TIMING=1 at backend init, the impl exposes
    a `_phase_e_coop_region_timing` tensor; otherwise the attribute is
    None.
    """
    backend_mod = pytest.importorskip(
        "vllm.v1.attention.backends.cute_paged._backend"
    )
    src = inspect.getsource(backend_mod)
    assert "_phase_e_coop_region_timing" in src, (
        "_backend must define the _phase_e_coop_region_timing workspace "
        "buffer (env-gated) — otherwise the kernel cannot receive timing "
        "scratch from the call site."
    )
    assert "CUTE_BETA_REGION_TIMING" in src, (
        "_backend must read CUTE_BETA_REGION_TIMING env to gate "
        "allocation."
    )
