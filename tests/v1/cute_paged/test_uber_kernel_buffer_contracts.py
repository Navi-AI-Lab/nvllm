"""L2 structural test: verifies β-coop / β-lite read inputs from the right buffer.

Pre-fix: β-coop reads `self.residual_output` (post-Phase-C output of the legacy
paged_attention_forward), causing residual_post_attn = 2*attn_out + h + r.
Post-fix: β-coop reads `self.residual_buf` (post-input-LN residual mirrored
from qwen3_5.py:460), giving residual_post_attn = attn_out + h + r.

Strategy: pure source-text inspection via `inspect.getsource` on
`CutePagedAttentionImpl.forward`. We assert the post-fix wiring is present
(`self.residual_buf`) and the buggy alias (`self.residual_output` as residual
input to β kernels) is absent. No CUDA, no kernel launch — runs anywhere.
"""
import inspect
import pytest


def test_beta_coop_residual_in_sources_from_residual_buf():
    """β-coop's residual_in must source from self.residual_buf, not residual_output."""
    from vllm.v1.attention.backends.cute_paged._backend import (
        CutePagedAttentionImpl,
    )

    src = inspect.getsource(CutePagedAttentionImpl.forward)
    assert "residual_in=self.residual_buf" in src, (
        "Expected β-coop launch to read from self.residual_buf; found a different source. "
        "Check _backend.py:1175 — buffer-aliasing bug may have regressed."
    )
    # Strengthened guard (audit Finding 6 / option b): C1 fixes both occurrences
    # of the alias bug, so `residual_in=self.residual_output` must not appear
    # ANYWHERE in CutePagedAttentionImpl.forward source. The original anchor
    # ("# β-coop") doesn't exist in source, so the guarded form silently passed
    # either way. This bare check fails loudly if the bug regresses.
    assert "residual_in=self.residual_output" not in src, (
        "β-coop call site still reads self.residual_output — the alias bug is back. "
        "See _backend.py:1175 (commit 76b88ba21) and audit Finding 6."
    )


def test_beta_lite_residual_post_ln_sources_from_residual_buf():
    """β-lite has the same alias bug pre-migration (audit Finding 6). Verify fix."""
    from vllm.v1.attention.backends.cute_paged._backend import (
        CutePagedAttentionImpl,
    )

    src = inspect.getsource(CutePagedAttentionImpl.forward)
    assert "residual_post_ln=self.residual_buf" in src, (
        "β-lite still aliases legacy buffer. See audit Finding 6 / _backend.py:1268."
    )
