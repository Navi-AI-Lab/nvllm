# Copyright 2026 Navi Ai Labs
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Phase D1 custom-op wrap for CuTe paged fused MLP.

Wraps the fused MLP dispatch as an opaque `torch.ops.vllm.cute_mlp_forward`
so torch.compile cannot peek into the per-forward `_mlp_fusion_active`
decision and dead-branch one of the two paths. Mirrors the
`unified_attention_with_output` pattern in
`vllm/model_executor/layers/attention/attention.py`.

The outer env var `CUTE_MLP_FUSION` still gates whether `attach_mlp_fusion`
runs at all; this op fixes the *per-step* Python-level `if` that Dynamo
was specializing at trace time.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from vllm.utils.torch_utils import direct_register_custom_op

if TYPE_CHECKING:
    from vllm.v1.attention.backends.cute_paged._backend import (
        CutePagedAttentionImpl,
    )


# Module-level registry. `attach_mlp_fusion` populates one entry per
# fusion-attached MLP layer, keyed by the decoder-supplied layer name.
# The op body reads from this dict at runtime (not at trace time).
_CUTE_MLP_REGISTRY: dict[str, "CutePagedAttentionImpl"] = {}


def _cute_mlp_forward_impl(
    x: torch.Tensor,
    output: torch.Tensor,
    layer_name: str,
) -> None:
    """Runtime body of the fused MLP custom op.

    `x`: [num_tokens, hidden] input (post-rmsnorm hidden states).
    `output`: [num_tokens, hidden] pre-allocated by the caller; mutated in-place.
    `layer_name`: registry key set by `attach_mlp_fusion`.

    Branches on `impl._mlp_fusion_active` (set per-forward by the attention
    impl after the fused kernel launched). If active, copies from
    `impl.mlp_output[:nat]`; otherwise runs the unfused gate_up + silu_mul +
    down math via module refs stashed on `impl` at attach time. Both branches
    live inside the opaque op body, invisible to torch.compile.
    """
    impl = _CUTE_MLP_REGISTRY.get(layer_name)
    if impl is None:
        # Should be unreachable: Qwen3_5MLP.forward only dispatches here when
        # `_cute_layer_name` is set on the module, which `attach_mlp_fusion`
        # sets in the same block where it inserts into this registry. A loud
        # error at the custom-op boundary beats a silent AttributeError from
        # the fallback path or a downstream segfault.
        raise RuntimeError(
            f"cute_mlp_forward called for unregistered layer {layer_name!r}"
        )

    if impl._mlp_fusion_active:
        nat = impl._mlp_fusion_nat
        output[:nat].copy_(impl.mlp_output[:nat])
        return

    # Fallback: fusion-inactive this step (e.g. prefill batch, or the fused
    # kernel hit its fail-closed fallback in _backend.py). Run the unfused
    # math via module refs stashed at attach time. These are standard
    # torch.nn.Modules — calling them from inside an opaque custom op is
    # safe (nothing is being traced; inner ops run eagerly).
    gate_up, _ = impl._mlp_gate_up_proj(x)
    mid = impl._mlp_act_fn(gate_up)
    result, _ = impl._mlp_down_proj(mid)
    output.copy_(result)


def _cute_mlp_forward_fake(
    x: torch.Tensor,
    output: torch.Tensor,
    layer_name: str,
) -> None:
    """Fake impl used during torch.compile symbolic shape analysis.

    Signature must match `_cute_mlp_forward_impl` exactly (arg count + types).
    Returns None because the op mutates `output` in place; no outputs.
    """
    return


direct_register_custom_op(
    op_name="cute_mlp_forward",
    op_func=_cute_mlp_forward_impl,
    mutates_args=["output"],
    fake_impl=_cute_mlp_forward_fake,
)
