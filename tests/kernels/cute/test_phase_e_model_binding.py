"""Binding test: the Qwen3_5Model.__init__ Phase E hook is present AND
reads the correct attribute path to reach the CuTe impl.

Structural test — does NOT instantiate a model (too heavy for a unit
test). Task 20's integration test catches runtime misbehavior; this one
catches the two most common drift modes for this hook:

1. Hook missing from the source file.
2. Hook present but reading the wrong attribute path
   (`layer.self_attn.impl` instead of `layer.self_attn.attn.impl`).
   `memory:feedback_verify_model_class` warns against this exact class
   of bug — the hook would silently become dead code.

Reading `inspect.getsource(Qwen3_5Model.__init__)` does NOT work because
`Qwen3_5Model` is wrapped with `@support_torch_compile`, which replaces
`__init__` with a thin wrapper whose source doesn't include the class
body. We read the module source instead.
"""
import pytest
import torch


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_phase_e_binding_hook_present_in_source():
    """Hook's identifying call must exist in the module source."""
    import inspect
    import vllm.nvllm.models.qwen3_5 as m
    src = inspect.getsource(m)
    assert 'attach_next_input_layernorm' in src, (
        "Qwen3_5Model.__init__ doesn't reference attach_next_input_layernorm"
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_phase_e_binding_reads_attn_impl_not_self_attn_impl():
    """Regression guard: the hook must read `self_attn.attn.impl`, not
    `self_attn.impl`. The `.impl` attribute lives on the inner Attention
    module (cf. existing pattern at self_attn.attn.impl in this file).

    If this assertion fires, the hook is dead code: `getattr(layer.self_attn,
    'impl', None)` returns None for every layer, every call to
    attach_next_input_layernorm is skipped, and Phase E silently never binds.

    We scope the substring check to the hook block itself — the pre-existing
    forward-time code at L361/395 also contains `self_attn.attn.impl` and
    would mask a hook regression without this scoping.
    """
    import inspect
    import vllm.nvllm.models.qwen3_5 as m
    src = inspect.getsource(m)

    # Slice the module source to the hook block bounded by its landmark
    # comment and the next init statement.
    hook_start = src.find('Phase E cross-layer binding')
    hook_end = src.find('self.make_empty_intermediate_tensors', hook_start)
    assert hook_start != -1, "hook landmark comment missing from qwen3_5.py"
    assert hook_end != -1 and hook_end > hook_start, (
        "hook trailing landmark missing from qwen3_5.py"
    )
    hook_block = src[hook_start:hook_end]

    # The hook must use the getattr-chain form. Checking only for the
    # literal call `getattr(<layer>.self_attn, 'attn', ...)` — substring
    # matches in comments would falsely pass, so we look for the actual
    # runtime call pattern instead.
    has_getattr_chain = (
        "getattr(layer.self_attn, 'attn'" in hook_block
        or "getattr(self.self_attn, 'attn'" in hook_block
    )
    assert has_getattr_chain, (
        "Phase E hook doesn't reach impl via a .attn getattr chain — "
        "likely regressed to layer.self_attn.impl which is always None. "
        "See existing pattern at qwen3_5.py:243, 261, 361, 395."
    )
