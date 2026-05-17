# SPDX-License-Identifier: Apache-2.0
"""Regression test for the AOT-load `dummy_cache` placeholder.

The placeholder cache_dir passed to `compiler_manager.initialize_cache`
(`disable_cache=True`) on the AOT-load path used to live under
`<VLLM_CACHE_ROOT>/dummy_cache`. That broke blessed-cache production
serve, which mounts VLLM_CACHE_ROOT :ro — `os.makedirs` raised EROFS
and torch.compile silently fell back to cold compile.

The fix is `tempfile.mkdtemp(prefix="vllm_dummy_cache_")` (always
writable, no dependency on VLLM_CACHE_ROOT being RW). The test pins
that design choice so a future refactor can't silently re-introduce the
RO-mount-incompatible path.
"""
from __future__ import annotations

import inspect

from vllm.compilation import caching


def test_dummy_cache_uses_tempfile_not_vllm_cache_root() -> None:
    src = inspect.getsource(caching.reconstruct_serializable_fn_from_mega_artifact)
    assert 'tempfile.mkdtemp(prefix="vllm_dummy_cache_")' in src, (
        "AOT-load dummy_cache must use tempfile.mkdtemp (RO-mount safe), "
        "not VLLM_CACHE_ROOT/dummy_cache. See "
        "docs/research/2026-04-29-full-graph-spike/ for context."
    )
    assert 'os.path.join(envs.VLLM_CACHE_ROOT, "dummy_cache")' not in src
    assert "vllm_backend._dummy_cache_dir = dummy_cache_dir" in src, (
        "dummy_cache_dir must be pinned on vllm_backend so its lifetime "
        "tracks the backend (avoids /tmp GC while compiler_manager holds "
        "a reference)."
    )
