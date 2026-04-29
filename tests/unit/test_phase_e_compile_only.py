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
