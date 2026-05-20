"""L2 structural test: β-coop / β-lite read inputs from the right buffer.

Audit Finding 6 / commit 76b88ba21 fixed an alias bug where β-coop and
β-lite were reading `self.residual_output` (the legacy paged kernel's
post-Phase-C output) as their `residual_in` / `residual_post_ln`, giving
`residual_post_attn = 2*attn_out + h + r`. Post-fix, the correct source
is `self.residual_buf` (the residual mirror from
`qwen3_5.py:Qwen3_5DecoderLayer.forward`).

Post-uber-kernel migration (v0.3.0), the β-coop / β-lite call sites
route residual through locals (`_residual_in_buf`, `_residual_post_ln`)
that default to `self.residual_buf[:nat]` and switch to the framework
`output_residual` tensor when `_framework_output_route` is active. This
test enforces the negative invariant: the buggy alias
`self.residual_output` must NEVER appear as a residual INPUT to β kernels
in the forward source.
"""
import inspect


def test_residual_output_never_used_as_kernel_residual_input():
    """`self.residual_output` is the legacy paged kernel OUTPUT — it must
    not be plumbed back in as `residual_in=` / `residual_post_ln=` to
    β-coop or β-lite. The fix routes those through locals fed from
    `self.residual_buf`.
    """
    from vllm.v1.attention.backends.cute_paged._backend import (
        CutePagedAttentionImpl,
    )
    src = inspect.getsource(CutePagedAttentionImpl.forward)
    assert "residual_in=self.residual_output" not in src, (
        "β-coop launch reads self.residual_output as residual_in — the "
        "alias bug from audit Finding 6 has regressed. See _backend.py "
        "and commit 76b88ba21."
    )
    assert "residual_post_ln=self.residual_output" not in src, (
        "β-lite launch reads self.residual_output as residual_post_ln — "
        "the alias bug from audit Finding 6 has regressed."
    )


def test_residual_buf_is_the_canonical_residual_input_source():
    """`self.residual_buf` must appear as a default for the residual
    input locals — proves the canonical-source fix is still wired.
    """
    from vllm.v1.attention.backends.cute_paged._backend import (
        CutePagedAttentionImpl,
    )
    src = inspect.getsource(CutePagedAttentionImpl.forward)
    # Match common slice forms used in the framework-output route fallbacks.
    has_residual_buf_fallback = (
        "self.residual_buf[:nat]" in src
        or "self.residual_buf[:" in src
    )
    assert has_residual_buf_fallback, (
        "self.residual_buf isn't referenced in forward() — the residual "
        "mirror from qwen3_5.py is no longer consumed; β kernels will "
        "read stale state."
    )
