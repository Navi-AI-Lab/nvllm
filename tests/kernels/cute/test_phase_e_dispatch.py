"""Phase E dispatch-predicate unit tests (Task 19 — gate 6.3).

These are env-config + dispatch-rule tests. They run in `.venv` without
needing a live model — they exercise the parsing in `_PhaseEEnvConfig`
and replicate the forward() predicate to verify coop/lite/none routing
for all (forced_path, total_ctas, resident_cap, coop_attached) combos.

End-to-end correctness of each path (β-coop and β-lite) is covered by
test_phase_e_epsilon_epilogue.py::test_beta_coop_full_matches_beta_lite
(same ground-truth — paged_attention_forward + Phase_D_MLP_Kernel).
"""
import os

import pytest

from vllm.v1.attention.backends.cute_paged._backend import (
    _phase_e_env_config,
)


# ---------------------------------------------------------------------------
# _PhaseEEnvConfig parsing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "env,expected",
    [
        # Default (no vars set): disabled, auto, no layer restriction.
        ({}, {"enabled": False, "forced_path": "auto", "restricted_layers": None}),
        # Enabled but auto path.
        ({"CUTE_PHASE_E_FUSION": "1"},
         {"enabled": True, "forced_path": "auto", "restricted_layers": None}),
        # Forced coop.
        ({"CUTE_PHASE_E_FUSION": "1", "CUTE_PHASE_E_PATH": "coop"},
         {"enabled": True, "forced_path": "coop", "restricted_layers": None}),
        # Forced lite.
        ({"CUTE_PHASE_E_FUSION": "1", "CUTE_PHASE_E_PATH": "lite"},
         {"enabled": True, "forced_path": "lite", "restricted_layers": None}),
        # Invalid forced_path falls back to auto (not error).
        ({"CUTE_PHASE_E_FUSION": "1", "CUTE_PHASE_E_PATH": "garbage"},
         {"enabled": True, "forced_path": "auto", "restricted_layers": None}),
        # Case-insensitive forced_path.
        ({"CUTE_PHASE_E_FUSION": "1", "CUTE_PHASE_E_PATH": "COOP"},
         {"enabled": True, "forced_path": "coop", "restricted_layers": None}),
        # Layer restriction CSV.
        ({"CUTE_PHASE_E_FUSION": "1", "CUTE_PHASE_E_LAYERS": "0,5,10"},
         {"enabled": True, "forced_path": "auto",
          "restricted_layers": {0, 5, 10}}),
        # Empty CUTE_PHASE_E_LAYERS → None (not empty set).
        ({"CUTE_PHASE_E_FUSION": "1", "CUTE_PHASE_E_LAYERS": ""},
         {"enabled": True, "forced_path": "auto", "restricted_layers": None}),
        # Malformed CUTE_PHASE_E_LAYERS → None (not crash).
        ({"CUTE_PHASE_E_FUSION": "1", "CUTE_PHASE_E_LAYERS": "abc"},
         {"enabled": True, "forced_path": "auto", "restricted_layers": None}),
    ],
)
def test_phase_e_env_config_parses(monkeypatch, env, expected):
    # Clear phase-E-related vars first so tests are hermetic.
    for k in (
        "CUTE_PHASE_E_FUSION", "CUTE_PHASE_E_PATH", "CUTE_PHASE_E_LAYERS",
    ):
        monkeypatch.delenv(k, raising=False)
    for k, v in env.items():
        monkeypatch.setenv(k, v)

    cfg = _phase_e_env_config()
    assert cfg.enabled == expected["enabled"]
    assert cfg.forced_path == expected["forced_path"]
    assert cfg.restricted_layers == expected["restricted_layers"]


# ---------------------------------------------------------------------------
# Dispatch predicate — replicates the inline logic in _backend.py forward()
# ---------------------------------------------------------------------------

def _dispatch_path(
    forced_path: str,
    total_ctas: int,
    resident_cap: int,
    coop_attached: bool,
    phase_e_active: bool = True,
) -> str:
    """Mirror of _backend.py forward() predicate.

    Returns "coop", "lite", or "none".

    Post-2026-04-26 contract (_backend.py:1396-1406): the resident-cap
    fitness check `total_ctas <= resident_cap` is a HARD gate even under
    `forced_path="coop"` — it was previously bypassed, which produced
    CUDA_ERROR_COOPERATIVE_LAUNCH_TOO_LARGE on multi-seq decode and silent
    gibberish. β-lite picks up coop's rejected case under `auto` and `lite`.
    """
    if not phase_e_active:
        return "none"
    cap_ok = total_ctas <= resident_cap
    use_beta_coop = (
        coop_attached
        and cap_ok
        and forced_path in ("coop", "auto")
    )
    use_beta_lite = (
        not use_beta_coop
        and forced_path in ("lite", "auto")
    )
    if use_beta_coop:
        return "coop"
    if use_beta_lite:
        return "lite"
    return "none"


@pytest.mark.parametrize(
    "forced,total,cap,attached,expected",
    [
        # --- forced="coop" ---
        # Coop when kernel attached AND fits under cap.
        ("coop", 64,   96, True,  "coop"),
        # Forced coop but over cap → none (hard cap: no silent fallback to
        # lite when the user explicitly asked for coop).
        ("coop", 2048, 96, True,  "none"),
        # If user forces coop but kernel not attached, neither path fires
        # — the caller uses the non-Phase-E path.
        ("coop", 64,   96, False, "none"),

        # --- forced="lite" ---
        # Always lite, even if coop would fit.
        ("lite", 64,   96, True,  "lite"),
        ("lite", 64,   96, False, "lite"),

        # --- forced="auto" ---
        # Under cap + attached → coop.
        ("auto", 64,   96, True,  "coop"),
        # At cap → coop (<= boundary).
        ("auto", 96,   96, True,  "coop"),
        # Over cap → lite.
        ("auto", 97,   96, True,  "lite"),
        ("auto", 2048, 96, True,  "lite"),
        # Under cap but coop not attached → lite.
        ("auto", 64,   96, False, "lite"),
    ],
)
def test_dispatch_predicate(forced, total, cap, attached, expected):
    """Dispatch-safety: (forced_path × total_ctas × resident_cap ×
    coop_attached) → correct branch. Replicates _backend.py forward()."""
    assert _dispatch_path(forced, total, cap, attached) == expected


@pytest.mark.parametrize("num_seqs", [1, 2, 4, 16])
def test_auto_never_exceeds_cap(num_seqs):
    """Gate 6.3: in auto mode, β-coop must NOT fire when the full grid
    exceeds the resident cap. Uses GB10's empirical cap ~96 CTAs
    (per memory:reference_cute_cooperative_launch)."""
    CTAS_PER_SEQ = 64
    RESIDENT_CAP_GB10 = 96  # conservative — real probe in impl
    total_ctas = CTAS_PER_SEQ * num_seqs
    path = _dispatch_path(
        forced_path="auto",
        total_ctas=total_ctas,
        resident_cap=RESIDENT_CAP_GB10,
        coop_attached=True,
    )
    if total_ctas <= RESIDENT_CAP_GB10:
        assert path == "coop", (
            f"auto should pick coop when {total_ctas} <= cap={RESIDENT_CAP_GB10}"
        )
    else:
        assert path == "lite", (
            f"auto must pick lite when {total_ctas} > cap={RESIDENT_CAP_GB10}"
        )
