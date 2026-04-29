# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""β-coop framework-output-buffer dispatch op.

Mirrors `vllm::unified_attention_with_output` (vllm/model_executor/layers/
attention/attention.py:712-760) — a thin custom op that delegates to
`layer.impl.forward(...)` via `get_attention_context(layer_name)`.

Registered as a PIECEWISE splitting boundary in
`vllm/config/compilation.py:_attention_ops` so torch.compile splits the
FX graph at the call. The op body runs as eager Python at runtime,
between captured graph segments, on every decode step.

Phase 2: stub raises NotImplementedError. Phase 3 fills in delegation.

See spec: docs/research/uber_kernel_migration/2026-04-27-beta-coop-rewrite-design.md
See feedback memory: feedback_splitting_op_runtime_dispatch
"""

from __future__ import annotations

import os

import torch

from vllm.utils.torch_utils import direct_register_custom_op

# Per-layer fire counter for empirical replay verification (mirrors
# tests/compile/silly_attention.py:50). Reset by tests.
#
# Phase 6a: gated behind CUTE_BETA_COOP_COUNT=1. Default OFF — the
# dict-get + dict-set ran every full-attn layer × every token despite
# being a debug-only observation point. Local flag (not _DEBUG_FUSION)
# avoids cross-module coupling and keeps the gate narrow: _DEBUG_FUSION
# also enables heavier backend debug paths.
_BETA_COOP_COUNT_FIRES: bool = os.environ.get("CUTE_BETA_COOP_COUNT", "0") == "1"
_BETA_COOP_FIRE_COUNTER: dict[str, int] = {}


def cute_beta_coop_run(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    residual: torch.Tensor,
    attn_input: torch.Tensor,
    gate: torch.Tensor,
    output_rmsnorm: torch.Tensor,
    output_residual: torch.Tensor,
    output_mlp: torch.Tensor,
    layer_name: str,
    kv_cache_dummy_dep: torch.Tensor | None = None,
) -> None:
    """Eager Python body — runs every decode step (splitting boundary).

    Delegates to layer.impl.forward(...) with the framework-output kwargs.
    `_backend.forward` handles β-coop dispatch vs fall-through internally
    based on attn_metadata.is_decode_only and num_seqs vs resident_cap.
    """
    # kv_cache_dummy_dep is the canonical phantom-dep pattern from
    # vllm/model_executor/layers/attention/attention.py:721-726 — provides
    # an explicit data-dependency edge from unified_kv_cache_update to
    # this op so dynamo preserves ordering.
    del kv_cache_dummy_dep

    from vllm.model_executor.layers.attention.attention import get_attention_context

    if _BETA_COOP_COUNT_FIRES:
        _BETA_COOP_FIRE_COUNTER[layer_name] = (
            _BETA_COOP_FIRE_COUNTER.get(layer_name, 0) + 1
        )

    attn_metadata, attn_layer, kv_cache, _ = get_attention_context(layer_name)

    # Reshape q/k/v from 2D [num_tokens, n_heads*head_size] to 3D
    # [num_tokens, n_heads, head_size]. Mirrors canonical
    # Attention.forward at vllm/model_executor/layers/attention/attention.py:451-456.
    # Done at op-body level (outside captured graphs) since we're a
    # splitting boundary; matches the canonical "minimize CPU overhead
    # from non-CUDA-graph regions" rationale.
    #
    # Phase 6a: defensive `dim() == 2` branch. Qwen3_5Attention.forward
    # already passes 3D q/k/v on the framework route, so this skips ~3
    # view ops per full-attn layer × per token. Legacy 2D callers (if
    # any) still work via the fallback branch.
    if query.dim() == 2:
        query = query.view(-1, attn_layer.num_heads, attn_layer.head_size)
    if key is not None and key.dim() == 2:
        key = key.view(-1, attn_layer.num_kv_heads, attn_layer.head_size)
    if value is not None and value.dim() == 2:
        value = value.view(-1, attn_layer.num_kv_heads, attn_layer.head_size_v)

    attn_layer.impl.forward(
        attn_layer,
        query,
        key,
        value,
        kv_cache,
        attn_metadata,
        # `output` is reused as the framework-output rmsnorm buffer
        # (per spec: "existing `output: Tensor` is reused as `output_rmsnorm`").
        # The legacy assertion at _backend.forward L1030 requires output != None.
        output=output_rmsnorm,
        residual=residual,
        attn_input=attn_input,
        gate=gate,
        output_rmsnorm=output_rmsnorm,
        output_residual=output_residual,
        output_mlp=output_mlp,
    )


def cute_beta_coop_run_fake(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    residual: torch.Tensor,
    attn_input: torch.Tensor,
    gate: torch.Tensor,
    output_rmsnorm: torch.Tensor,
    output_residual: torch.Tensor,
    output_mlp: torch.Tensor,
    layer_name: str,
    kv_cache_dummy_dep: torch.Tensor | None = None,
) -> None:
    return None


direct_register_custom_op(
    op_name="cute_beta_coop_run",
    op_func=cute_beta_coop_run,
    mutates_args=["output_rmsnorm", "output_residual", "output_mlp"],
    fake_impl=cute_beta_coop_run_fake,
)
