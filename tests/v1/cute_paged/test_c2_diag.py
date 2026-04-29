# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for vllm.v1.attention.backends.cute_paged._c2_diag.

These tests exercise the comparison primitives with synthetic tensors so
they run on any host (no GPU required, no full vLLM bring-up). The
integration tests (probe wired into qwen3_5.py) are manual serve-cute
runs documented in the C2 diagnostic plan.
"""

from __future__ import annotations

import pytest
import torch

from vllm.v1.attention.backends.cute_paged import _c2_diag


def test_compare_pair_within_tolerance() -> None:
    """Two BF16 tensors within unit-roundoff should compare OK."""
    torch.manual_seed(0)
    a = torch.randn(4, 5120, dtype=torch.bfloat16)
    b = a + 1e-4 * torch.randn_like(a)
    result = _c2_diag._compare_pair(a, b, atol=1e-2, rtol=1e-2)
    assert result["ok"] is True
    assert result["linf"] < 1e-2


def test_compare_pair_diverges_on_large_offset() -> None:
    """Two BF16 tensors with a >atol offset should compare DIVERGED."""
    torch.manual_seed(0)
    a = torch.randn(4, 5120, dtype=torch.bfloat16)
    b = a + 1.0  # constant offset > atol
    result = _c2_diag._compare_pair(a, b, atol=1e-2, rtol=1e-2)
    assert result["ok"] is False
    assert result["linf"] > 0.5


def test_compare_pair_returns_required_keys() -> None:
    """Result dict must contain linf, rel_med, ok keys for log formatting."""
    a = torch.randn(2, 4, dtype=torch.bfloat16)
    result = _c2_diag._compare_pair(a, a.clone(), atol=1e-2, rtol=1e-2)
    assert set(result.keys()) >= {"linf", "rel_med", "ok"}


def test_dump_on_divergence_writes_bundle(tmp_path, monkeypatch) -> None:
    """Dump should write a torch.save bundle with all required keys."""
    monkeypatch.setenv("CUTE_C2_DIAG_DUMP_DIR", str(tmp_path))
    legacy_h = torch.randn(2, 4, dtype=torch.bfloat16)
    legacy_r = torch.randn(2, 4, dtype=torch.bfloat16)
    beta_h = torch.randn(2, 4, dtype=torch.bfloat16)
    beta_r = torch.randn(2, 4, dtype=torch.bfloat16)
    _c2_diag._dump_on_divergence(
        layer_idx=3,
        step_idx=42,
        nat=2,
        atol=1e-2,
        rtol=1e-2,
        legacy_hidden=legacy_h,
        legacy_residual=legacy_r,
        beta_rmsnorm_output=beta_h,
        beta_residual_output=beta_r,
    )
    dump_path = tmp_path / "layer3_step42.pt"
    assert dump_path.exists()
    bundle = torch.load(dump_path)
    assert bundle["layer_idx"] == 3
    assert bundle["step_idx"] == 42
    assert bundle["nat"] == 2
    assert bundle["atol"] == 1e-2
    assert bundle["rtol"] == 1e-2
    assert torch.equal(bundle["legacy_hidden"], legacy_h)
    assert torch.equal(bundle["beta_rmsnorm_output"], beta_h)


def test_dump_on_divergence_creates_dir(tmp_path, monkeypatch) -> None:
    """Dump directory should be created if it does not exist."""
    target = tmp_path / "subdir" / "deeper"
    monkeypatch.setenv("CUTE_C2_DIAG_DUMP_DIR", str(target))
    _c2_diag._dump_on_divergence(
        layer_idx=0,
        step_idx=0,
        nat=1,
        atol=1e-2,
        rtol=1e-2,
        legacy_hidden=torch.zeros(1, 1, dtype=torch.bfloat16),
        legacy_residual=torch.zeros(1, 1, dtype=torch.bfloat16),
        beta_rmsnorm_output=torch.zeros(1, 1, dtype=torch.bfloat16),
        beta_residual_output=torch.zeros(1, 1, dtype=torch.bfloat16),
    )
    assert (target / "layer0_step0.pt").exists()


def test_next_step_idx_increments() -> None:
    """next_step_idx returns monotonically increasing integers from 0."""
    _c2_diag._reset_step_counter_for_test()
    assert _c2_diag.next_step_idx() == 0
    assert _c2_diag.next_step_idx() == 1
    assert _c2_diag.next_step_idx() == 2


def test_reset_step_counter_for_test() -> None:
    """The reset hook returns the counter to 0 (used by tests only)."""
    _c2_diag.next_step_idx()
    _c2_diag.next_step_idx()
    _c2_diag._reset_step_counter_for_test()
    assert _c2_diag.next_step_idx() == 0


def test_assert_no_flashinfer_autotune_disabled_passes(monkeypatch) -> None:
    """When autotune is disabled (default), the assert is a no-op."""

    # Default vllm config has autotune off; passing a stub config is enough.
    class _Stub:
        enable_flashinfer_autotune = False

    _c2_diag.assert_no_flashinfer_autotune(_Stub())  # must not raise


def test_assert_no_flashinfer_autotune_enabled_raises() -> None:
    """When autotune is enabled, the assert raises with a clear message."""

    class _Stub:
        enable_flashinfer_autotune = True

    with pytest.raises(RuntimeError, match="flashinfer autotune"):
        _c2_diag.assert_no_flashinfer_autotune(_Stub())


def test_inject_noise_disabled_returns_unchanged(monkeypatch) -> None:
    """When CUTE_C2_DIAG_INJECT_NOISE is unset, tensor is returned unchanged."""
    monkeypatch.delenv("CUTE_C2_DIAG_INJECT_NOISE", raising=False)
    a = torch.ones(2, 4, dtype=torch.bfloat16)
    out = _c2_diag._inject_noise(a)
    assert torch.equal(out, a)


def test_inject_noise_enabled_adds_offset(monkeypatch) -> None:
    """When CUTE_C2_DIAG_INJECT_NOISE=1.0, the offset is added in-place."""
    monkeypatch.setenv("CUTE_C2_DIAG_INJECT_NOISE", "1.0")
    a = torch.ones(2, 4, dtype=torch.bfloat16)
    out = _c2_diag._inject_noise(a)
    expected = a + 1.0
    assert torch.equal(out, expected)


def test_inject_noise_invalid_value_raises(monkeypatch) -> None:
    """Non-float CUTE_C2_DIAG_INJECT_NOISE values raise loud."""
    monkeypatch.setenv("CUTE_C2_DIAG_INJECT_NOISE", "abc")
    with pytest.raises(ValueError):
        _c2_diag._inject_noise(torch.zeros(1, 1))


def test_inject_noise_empty_string_returns_unchanged(monkeypatch) -> None:
    """Set-but-empty CUTE_C2_DIAG_INJECT_NOISE='' must be treated as unset
    (serve-cute.sh writes empty values via `-e VAR=` for unset host shell vars)."""
    monkeypatch.setenv("CUTE_C2_DIAG_INJECT_NOISE", "")
    a = torch.ones(2, 4, dtype=torch.bfloat16)
    out = _c2_diag._inject_noise(a)
    assert torch.equal(out, a)


def test_compare_and_log_handles_empty_tol_env(monkeypatch) -> None:
    """Set-but-empty CUTE_C2_DIAG_TOL_* must not crash float() — fall back
    to defaults (1e-2). serve-cute.sh writes empty values for unset vars."""
    monkeypatch.setenv("CUTE_C2_DIAG_TOL_ATOL", "")
    monkeypatch.setenv("CUTE_C2_DIAG_TOL_RTOL", "")
    monkeypatch.delenv("CUTE_C2_DIAG_INJECT_NOISE", raising=False)
    a = torch.randn(2, 4, dtype=torch.bfloat16)
    _c2_diag.compare_and_log(
        layer_idx=0,
        step_idx=0,
        nat=2,
        legacy_hidden=a,
        legacy_residual=a,
        beta_rmsnorm_output=a.clone(),
        beta_residual_output=a.clone(),
    )  # must not raise


def test_compare_and_log_ok_path_no_raise(monkeypatch, capsys) -> None:
    """When both pairs match, compare_and_log logs OK and does not raise."""
    monkeypatch.delenv("CUTE_C2_DIAG_INJECT_NOISE", raising=False)
    torch.manual_seed(0)
    legacy_h = torch.randn(2, 4, dtype=torch.bfloat16)
    legacy_r = torch.randn(2, 4, dtype=torch.bfloat16)
    _c2_diag.compare_and_log(
        layer_idx=0,
        step_idx=0,
        nat=2,
        legacy_hidden=legacy_h,
        legacy_residual=legacy_r,
        beta_rmsnorm_output=legacy_h.clone(),
        beta_residual_output=legacy_r.clone(),
    )
    captured = capsys.readouterr()
    combined = captured.out + captured.err
    assert "[C2_DIAG]" in combined
    assert "OK" in combined


def test_compare_and_log_diverged_path_raises(monkeypatch, tmp_path) -> None:
    """When divergence above tolerance, compare_and_log dumps + raises."""
    monkeypatch.delenv("CUTE_C2_DIAG_INJECT_NOISE", raising=False)
    monkeypatch.setenv("CUTE_C2_DIAG_DUMP_DIR", str(tmp_path))
    torch.manual_seed(0)
    legacy_h = torch.randn(2, 4, dtype=torch.bfloat16)
    legacy_r = torch.randn(2, 4, dtype=torch.bfloat16)
    beta_h = legacy_h + 1.0  # large divergence
    beta_r = legacy_r.clone()
    with pytest.raises(RuntimeError, match=r"\[C2_DIAG\] diverged"):
        _c2_diag.compare_and_log(
            layer_idx=3,
            step_idx=42,
            nat=2,
            legacy_hidden=legacy_h,
            legacy_residual=legacy_r,
            beta_rmsnorm_output=beta_h,
            beta_residual_output=beta_r,
        )
    assert (tmp_path / "layer3_step42.pt").exists()


def test_compare_and_log_skips_when_nat_zero(monkeypatch) -> None:
    """nat=0 means empty decode step; skip silently, no error."""
    legacy_h = torch.randn(2, 4, dtype=torch.bfloat16)
    _c2_diag.compare_and_log(
        layer_idx=0,
        step_idx=0,
        nat=0,
        legacy_hidden=legacy_h,
        legacy_residual=legacy_h,
        beta_rmsnorm_output=legacy_h,
        beta_residual_output=legacy_h,
    )  # must not raise
