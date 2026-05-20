# SPDX-License-Identifier: Apache-2.0
"""Backward-compat re-export.

The reference paged-attention body moved to
``vllm.v1.attention.backends.cute_paged._pytorch_reference`` on 2026-05-19
so the production backend no longer imports from the ``tests`` package
(audit Finding 1.5). Existing test imports stay valid via this shim.
"""
from vllm.v1.attention.backends.cute_paged._pytorch_reference import (  # noqa: F401
    _gather_kv_pages,
    reference_paged_attention,
)
