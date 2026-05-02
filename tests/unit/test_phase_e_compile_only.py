"""compile_only=True on run_beta_coop_full primes the cache but skips the launch.

Concrete coverage: signature includes compile_only kwarg with default
False. End-to-end behavior (cache prime + launch skip) is exercised
in-container via the Task 11 precompile workflow — instantiating
PhaseE_Beta_Kernel for a unit test requires the full CUDA + CUTLASS
runtime which isn't available in the host venv.
"""

from __future__ import annotations

import inspect


def test_run_beta_coop_full_has_compile_only_kwarg():
    from vllm.v1.attention.backends.cute_paged.phase_e_kernel import (
        PhaseE_Beta_Kernel,
    )

    sig = inspect.signature(PhaseE_Beta_Kernel.run_beta_coop_full)
    assert "compile_only" in sig.parameters
    assert sig.parameters["compile_only"].default is False
    # phase_e_kernel.py uses `from __future__ import annotations` so the
    # annotation is a string, not the bool class.
    assert sig.parameters["compile_only"].annotation == "bool"


def test_run_beta_coop_full_has_persistent_buffer_kwargs():
    """Five workspace buffers must be required keyword-only params on
    run_beta_coop_full (spec 2026-04-30 §4.3). They sit behind a `*,`
    separator after the trailing defaulted params to satisfy Python's
    no-non-default-after-default rule.
    """
    from vllm.v1.attention.backends.cute_paged.phase_e_kernel import (
        PhaseE_Beta_Kernel,
    )

    sig = inspect.signature(PhaseE_Beta_Kernel.run_beta_coop_full)
    expected = [
        "wo_output",
        "mlp_partial_fp32",
        "mlp_arrival_count",
        "grid_barrier_i32",
        "phase1_arrival_count",
    ]
    for name in expected:
        assert name in sig.parameters, (
            f"missing required kwarg: {name}; spec §4.3 requires it"
        )
        p = sig.parameters[name]
        assert p.default is inspect.Parameter.empty, (
            f"{name} must be required (no default); see spec §4.3 + "
            f"feedback_no_silent_fallbacks"
        )
        assert p.kind == inspect.Parameter.KEYWORD_ONLY, (
            f"{name} must be keyword-only (declared after `*,`); "
            f"required-after-default is a SyntaxError otherwise"
        )
