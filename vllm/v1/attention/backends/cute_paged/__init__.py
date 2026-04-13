# Copyright 2026 Navi Ai Labs
# SPDX-License-Identifier: Apache-2.0
"""CuTe DSL paged attention backend for SM120/SM121 (GB10).

Backend classes are lazily imported from _backend.py to avoid pulling
in vLLM's full dependency chain when only disk_cache or kernel is needed.
"""

# Lazy re-exports: vLLM's registry resolves
# "vllm.v1.attention.backends.cute_paged.CutePagedBackend" via __getattr__.
_LAZY_EXPORTS = {
    "CutePagedBackend",
    "CutePagedAttentionImpl",
    "CutePagedMetadataBuilder",
    "CutePagedMetadata",
}


def __getattr__(name: str):
    if name in _LAZY_EXPORTS:
        from vllm.v1.attention.backends.cute_paged._backend import (
            CutePagedAttentionImpl,
            CutePagedBackend,
            CutePagedMetadata,
            CutePagedMetadataBuilder,
        )
        _exports = {
            "CutePagedBackend": CutePagedBackend,
            "CutePagedAttentionImpl": CutePagedAttentionImpl,
            "CutePagedMetadataBuilder": CutePagedMetadataBuilder,
            "CutePagedMetadata": CutePagedMetadata,
        }
        globals().update(_exports)
        return _exports[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
