"""L3 multi-layer test: verifies layer-boundary semantics post-C1.5.

Catches:
- Phase 4 not adding mlp_out (audit Finding 1) — layer N+1's input_LN does the sum
- Per-layer input_layernorm fires unconditionally (no skip-op fall-through)
- F.1 layer-LN bake plumbing (skip-op, attach methods, flags) is gone
- run_beta_coop_full no longer takes Phase 4 / next-LN parameters
- cute_phase_e_dispatch consume branch reads mlp_output, not next_hidden_scratch

Strategy: pure source-text inspection via `inspect.getsource`. The full
kernel-level diff is covered by L4 (gsm8k); this test catches the
structural class. No CUDA, no kernel launch — runs anywhere.
"""
import inspect


def test_qwen35_layer_forward_runs_input_layernorm_unconditionally():
    """qwen3_5.py: input_LN gate must collapse to unconditional run.

    Post-C1.5 the non-first-layer branch of Qwen3_5DecoderLayer.forward
    must call self.input_layernorm(...) directly — no skip-op detour
    via cute_phase_e_skip_input_layernorm.
    """
    from vllm.nvllm.models import qwen3_5
    src = inspect.getsource(qwen3_5.Qwen3_5DecoderLayer.forward)
    assert "cute_phase_e_skip_input_layernorm" not in src, (
        "F.1 skip-op call site still present in layer forward. "
        "Should be deleted in C1.5."
    )
    assert "self.input_layernorm(hidden_states, residual)" in src, (
        "Expected unconditional self.input_layernorm(hidden_states, residual) "
        "call in non-first-layer branch."
    )


def test_no_attach_input_layernorm_loops_in_model_init():
    """qwen3_5.py source must drop attach_*_layernorm loops.

    Both attach_input_layernorm and attach_next_input_layernorm loops
    are gone — the F.1 cross-layer bake plumbing they enabled is gone.

    We grep the file directly because Qwen3_5Model.__init__ is replaced
    by @support_torch_compile, so inspect.getsource(Qwen3_5Model.__init__)
    returns the wrapper, not the class body.
    """
    from vllm.nvllm.models import qwen3_5
    with open(qwen3_5.__file__, "r") as f:
        src = f.read()
    assert "attach_input_layernorm" not in src, (
        "attach_input_layernorm reference still present in qwen3_5.py. "
        "C1.5 must delete the attach loop and any Phase F.1 plumbing. "
        "(check for impl.attach_input_layernorm(...) call in Qwen3_5Model.__init__)"
    )
    assert "attach_next_input_layernorm" not in src, (
        "attach_next_input_layernorm reference still present in qwen3_5.py. "
        "C1.5 must delete the attach loop and any Phase F.1 plumbing. "
        "(check for impl.attach_next_input_layernorm(...) call in Qwen3_5Model.__init__)"
    )


def test_skip_op_deleted():
    """cute_phase_e_skip_input_layernorm op must be deleted entirely.

    Both the impl/fake functions and the direct_register_custom_op
    registration must be gone from _mlp_op.py.
    """
    from vllm.v1.attention.backends.cute_paged import _mlp_op
    src = inspect.getsource(_mlp_op)
    assert 'op_name="cute_phase_e_skip_input_layernorm"' not in src, (
        "cute_phase_e_skip_input_layernorm op still registered. "
        "C1.5 must delete the op registration and the impl/fake functions."
    )


def test_phase_4_deleted_from_run_beta_coop_full():
    """Phase 4 args must be dropped from run_beta_coop_full's signature.

    The kernel returns at the end of Phase 3 (MLP write). The next-layer
    input_LN runs from Python at every layer entry instead of being baked
    into the previous layer's epilogue.
    """
    from vllm.v1.attention.backends.cute_paged import phase_e_kernel
    src = inspect.getsource(
        phase_e_kernel.PhaseE_Beta_Kernel.run_beta_coop_full
    )
    assert "next_input_layernorm_gamma" not in src, (
        "Phase 4 arg next_input_layernorm_gamma still present in "
        "run_beta_coop_full. C1.5 must drop it."
    )
    assert "emit_next_layernorm" not in src, (
        "Phase 4 arg emit_next_layernorm still present in "
        "run_beta_coop_full. C1.5 must drop it."
    )


def test_dispatch_op_consumes_mlp_output_not_next_hidden_scratch():
    """cute_phase_e_dispatch consume branch must read mlp_output.

    Pre-C1.5 the consume branch read impl.next_hidden_scratch (the
    Phase-4-baked next-layer input_LN output). Post-C1.5 it reads
    impl.mlp_output (raw post-MLP hidden) and the next layer's
    input_LN runs from Python.
    """
    from vllm.v1.attention.backends.cute_paged import _mlp_op
    src = inspect.getsource(_mlp_op)
    assert "next_hidden_scratch" not in src, (
        "cute_phase_e_dispatch still references next_hidden_scratch. "
        "C1.5 must update consume branch to read from mlp_output."
    )
    assert "impl.mlp_output[:nat]" in src, (
        "Expected cute_phase_e_dispatch consume branch to read "
        "impl.mlp_output[:nat] for hidden_out. C1.5 must keep this read "
        "active — see _mlp_op.py consume branch."
    )
