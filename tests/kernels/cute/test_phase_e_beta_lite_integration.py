"""β-lite end-to-end: with CUTE_PHASE_E_FUSION=1 and CUTE_PHASE_E_PATH=lite,
the decoder layer produces bit-identical output to CUTE_PHASE_E_FUSION=0.

This is a SKIP-UNLESS-MODEL-CACHED integration placeholder — full coverage
comes in Task 10 (Docker GSM8K smoke). Task 9 lands the dispatch wiring;
this test verifies the dispatch code path exists at the source level.
"""
import pytest
import torch


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_beta_lite_dispatch_wiring_present_in_source():
    """Source-level guard: the β-lite dispatch path must be wired in
    _backend.py::forward and qwen3_5.py's decoder forward.

    Catches regressions where the dispatch branch is accidentally removed
    or bypassed by a refactor.
    """
    import inspect
    import vllm.v1.attention.backends.cute_paged._backend as be
    import vllm.nvllm.models.qwen3_5 as m

    be_src = inspect.getsource(be)
    # _backend.py must reference the β-lite-specific kwargs that Task 8
    # added to Phase_D_MLP_Kernel
    assert 'emit_epilogue' in be_src, (
        "β-lite dispatch missing from _backend.py forward — expected "
        "Phase_D_MLP_Kernel call to pass emit_epilogue=True"
    )
    assert '_phase_e_env_config' in be_src, (
        "β-lite dispatch not using the env-flag parser helper from Task 6"
    )

    # qwen3_5.py must have a consume-gate for Phase E
    m_src = inspect.getsource(m)
    assert '_phase_e_consumed' in m_src or 'phase_e_consumed' in m_src, (
        "qwen3_5.py decoder layer missing Phase-E consume flag — "
        "the Python copy_ wrap at :405,428 should be skipped when β-lite runs"
    )
