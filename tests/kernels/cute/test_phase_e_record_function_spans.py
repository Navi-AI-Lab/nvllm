"""Phase E.1 follow-up #3 — per-layer record_function spans around β-coop
and β-lite dispatch in cute_paged._backend.forward().

Context: torch profiler traces of the CuTe backend lump all 16 full_attention
layers together because the β-coop / β-lite call sites are unwrapped. Each
call site should emit a ``torch.profiler.record_function`` span labelled
``"PhaseE_Beta.coop.<layer_name>"`` (resp. ``.lite.``) so profile events in
``chrome://tracing`` attribute time back to a specific decoder layer and
dispatch path.

``record_function`` is a no-op when no profiler is active, so there is no
steady-state cost.

Structural tests (no forward() execution needed) — the forward path is
covered by integration traces; these tests just guard against
regression-by-deletion of the span wrappers.
"""
from __future__ import annotations

import inspect
import re

import pytest


# ---------------------------------------------------------------------------
# Import target module source once per test run.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def backend_src() -> str:
    import vllm.v1.attention.backends.cute_paged._backend as mod
    return inspect.getsource(mod)


# ---------------------------------------------------------------------------
# Import-level sanity: record_function must be imported for the spans
# to work. Either a top-level import or a lazy import inside forward().
# ---------------------------------------------------------------------------


def test_record_function_is_imported(backend_src: str):
    """The module references ``torch.profiler.record_function`` (direct or
    aliased) so the span wrappers have a callable symbol.
    """
    assert (
        "from torch.profiler import record_function" in backend_src
        or "torch.profiler.record_function" in backend_src
    ), (
        "Phase E.1 #3: backend must import record_function from "
        "torch.profiler so β-coop / β-lite spans can be emitted."
    )


# ---------------------------------------------------------------------------
# Per-path span guards.
# ---------------------------------------------------------------------------


def _find_path_block(src: str, path_label: str) -> str:
    """Return the block of source for the ``_use_beta_{path}`` branch.

    We use the ``if _use_beta_{path}:`` landmark and take the next ~100
    lines — a heuristic slice that covers the kernel-launch plus its
    fail-closed except clause without bleeding into the next path.
    """
    marker = f"if _use_beta_{path_label}:"
    start = src.find(marker)
    assert start != -1, f"landmark {marker!r} missing from _backend.py"
    # Walk forward to the next `# --- END ...` or next `if _use_beta_`
    # boundary; return the slice.
    rest = src[start:]
    end_match = re.search(
        r"\n\s*# --- END PHASE E |^\s*if _use_beta_",
        rest[len(marker):],
        flags=re.MULTILINE,
    )
    if end_match is None:
        return rest[:4000]
    return rest[: len(marker) + end_match.start()]


def test_beta_coop_call_wrapped_in_record_function(backend_src: str):
    """The β-coop kernel launch is wrapped in a record_function span whose
    label contains ``PhaseE_Beta.coop`` and the layer name.
    """
    block = _find_path_block(backend_src, "coop")
    assert "run_beta_coop_full" in block, (
        "expected run_beta_coop_full call in β-coop branch — test scoping "
        "broke"
    )
    assert "record_function" in block, (
        "Phase E.1 #3: β-coop launch must be wrapped in "
        "record_function(...) for per-layer attribution in torch profiler "
        "traces."
    )
    # Label must include the path marker.
    assert "PhaseE_Beta.coop" in block, (
        "Phase E.1 #3: β-coop record_function span must be labelled with "
        '"PhaseE_Beta.coop" so the chrome trace distinguishes paths.'
    )
    # Label must interpolate a layer identifier (either _layer_name or
    # layer.layer_name).
    has_layer_interpolation = re.search(
        r"f['\"]PhaseE_Beta\.coop\.\{_layer_name[^}]*\}",
        block,
    )
    assert has_layer_interpolation, (
        "Phase E.1 #3: β-coop record_function label must interpolate "
        "_layer_name so individual layers are distinguishable in traces."
    )


def test_beta_lite_call_wrapped_in_record_function(backend_src: str):
    """The β-lite kernel launch is wrapped in a record_function span whose
    label contains ``PhaseE_Beta.lite`` and the layer name.
    """
    block = _find_path_block(backend_src, "lite")
    assert "_mlp_kernel(" in block, (
        "expected _mlp_kernel call in β-lite branch — test scoping broke"
    )
    assert "record_function" in block, (
        "Phase E.1 #3: β-lite launch must be wrapped in "
        "record_function(...) for per-layer attribution."
    )
    assert "PhaseE_Beta.lite" in block, (
        "Phase E.1 #3: β-lite record_function span must be labelled with "
        '"PhaseE_Beta.lite" to distinguish it from β-coop spans.'
    )
    has_layer_interpolation = re.search(
        r"f['\"]PhaseE_Beta\.lite\.\{_layer_name[^}]*\}",
        block,
    )
    assert has_layer_interpolation, (
        "Phase E.1 #3: β-lite record_function label must interpolate "
        "_layer_name."
    )


def test_span_labels_distinct_between_coop_and_lite(backend_src: str):
    """Regression guard: coop and lite must use distinct path markers so
    traces show one row per path, not two rows smashed together.
    """
    coop_labels = set(re.findall(r"PhaseE_Beta\.coop", backend_src))
    lite_labels = set(re.findall(r"PhaseE_Beta\.lite", backend_src))
    assert coop_labels, "coop span label missing"
    assert lite_labels, "lite span label missing"
