"""Source-contract tests for the post-C1.5 Phase E model-side wiring.

C1.5 deleted the `attach_next_input_layernorm` cross-layer binding loop
in `Qwen3_5Model.__init__`. The new invariants this test enforces:

  1. Every decoder layer entry runs `input_layernorm` unconditionally
     (no per-layer `_phase_e_skip_next_ln` gate). Cross-layer LN bake is
     gone — the Phase F.1 skip-op was retired.
  2. The β-coop framework-output route is wired: the decoder's forward
     calls the opaque op `cute_beta_coop_run` (defined in
     `_beta_coop_op.py`) which delegates dispatch to `_backend.forward`.
  3. The MLP consume step runs through the opaque op
     `cute_phase_e_dispatch` (not the original Python `_phase_e_consumed`
     gate, which dead-branched under PIECEWISE CUDA graphs).

Structural tests via `inspect.getsource` on the model module — does NOT
instantiate a model (too heavy for a unit test).
"""
import inspect

import pytest
import torch


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_input_layernorm_runs_unconditionally_at_layer_entry():
    """Layer entry must call `self.input_layernorm(hidden_states, residual)`
    on the non-first-layer branch. C1.5 removed the conditional skip path.
    """
    import vllm.nvllm.models.qwen3_5 as m
    src = inspect.getsource(m)
    assert "hidden_states, residual = self.input_layernorm(" in src, (
        "Qwen3_5DecoderLayer.forward should always run input_layernorm "
        "at layer entry post-C1.5 — the Phase F.1 skip-op was retired."
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_beta_coop_framework_output_route_wired():
    """The model must dispatch β-coop via the opaque op cute_beta_coop_run."""
    import vllm.nvllm.models.qwen3_5 as m
    src = inspect.getsource(m)
    assert "torch.ops.vllm.cute_beta_coop_run" in src, (
        "Qwen3_5 attention forward should call cute_beta_coop_run for the "
        "β-coop framework-output route (defined in _beta_coop_op.py). "
        "A regression would silently route every step through the legacy "
        "Python-projection path."
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_phase_e_consume_uses_opaque_dispatch_op():
    """MLP consume must run through cute_phase_e_dispatch."""
    import vllm.nvllm.models.qwen3_5 as m
    src = inspect.getsource(m)
    assert "torch.ops.vllm.cute_phase_e_dispatch" in src, (
        "Qwen3_5 decoder forward should call cute_phase_e_dispatch — "
        "regression risk: the original Python _phase_e_consumed gate "
        "dead-branched under PIECEWISE CUDA graphs (memory:feedback_"
        "opaque_op_not_enough)."
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_cross_layer_input_layernorm_attach_is_disabled():
    """`attach_next_input_layernorm` must remain a no-op site (commented)
    in _backend.py post-C1.5. The model module must not call it.
    """
    import vllm.nvllm.models.qwen3_5 as m
    m_src = inspect.getsource(m)
    # The call must NOT appear on any live (non-comment) line of the model.
    # Strip comment-only lines, then assert the call site is absent.
    code_lines = [
        ln for ln in m_src.splitlines()
        if ln.strip() and not ln.lstrip().startswith("#")
    ]
    code_text = "\n".join(code_lines)
    assert "attach_next_input_layernorm(" not in code_text, (
        "Qwen3_5Model.__init__ should NOT call attach_next_input_layernorm "
        "post-C1.5 — the cross-layer LN bake was retired."
    )
