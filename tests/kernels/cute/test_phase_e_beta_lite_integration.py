"""Source-contract test for β-lite MLP-only dispatch wiring.

Renamed from the original β-lite "integration" placeholder — this is a
source-level smoke that the β-lite dispatch points still exist in the
expected files. End-to-end behavior is covered by the GSM8K quality gate
and the Phase E ε-epilogue parity tests (test_phase_e_epsilon_epilogue.py).

Current production state (post-C1.5):
  * `_backend.py` launches β-lite with `emit_epilogue=False` (Phase F.1
    next-layer LN bake was removed in C1.5 — consumers read raw mlp_output
    + residual_output and the next layer's input_LN runs from Python).
  * `qwen3_5.py` decoder forward routes the consume step through the
    opaque op `cute_phase_e_dispatch`, not the original Python
    `_phase_e_consumed` gate.
"""
import inspect

import pytest
import torch


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_beta_lite_emit_epilogue_disabled_in_source():
    """β-lite must launch Phase_D_MLP_Kernel with emit_epilogue=False.

    Locks the C1.5 invariant: the ε epilogue (residual_final + next-layer
    RMSNorm bake) is no longer baked into Phase_D. Flipping this back to
    True would re-introduce the F.1 layer-LN-bake bug.
    """
    import vllm.v1.attention.backends.cute_paged._backend as be
    be_src = inspect.getsource(be)
    assert "emit_epilogue=False" in be_src, (
        "Expected emit_epilogue=False at the β-lite Phase_D launch site — "
        "post-C1.5 the next-layer LN bake is gone and consumers read raw "
        "mlp_output. If you have a reason to re-enable, update this test."
    )
    assert "_phase_e_env_config" in be_src, (
        "β-lite dispatch not using the env-flag parser helper "
        "(_phase_e_env_config / _PHASE_E_ENV)"
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_phase_e_consume_routed_through_opaque_op_in_qwen35():
    """Qwen3_5 decoder forward must consume Phase E results via the
    opaque `cute_phase_e_dispatch` op.

    Pre-F.1 the consume step was a Python `if impl._phase_e_consumed:`
    gate that torch.compile dead-branched under PIECEWISE CUDA graphs;
    the opaque op replaces it so the replay path picks the right branch.
    """
    import vllm.nvllm.models.qwen3_5 as m
    m_src = inspect.getsource(m)
    assert "torch.ops.vllm.cute_phase_e_dispatch" in m_src, (
        "Qwen3_5 decoder forward must call cute_phase_e_dispatch — "
        "regression risk: the Python `_phase_e_consumed` gate would "
        "silently dead-branch under PIECEWISE CUDA graphs."
    )
