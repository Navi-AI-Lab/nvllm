# Copyright 2026 Navi Ai Labs
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opaque custom-op wrap for CuTe paged fused MLP.

`torch.ops.vllm.cute_mlp_forward` owns the full MLP dispatch decision
— per-step fuse/fallback gate, kernel launch, fallback unfused GEMMs —
inside an Inductor-opaque op body. Single call site, single path
visible in the compiled graph: no dual-firing.

Outer env var `CUTE_MLP_FUSION` gates whether `attach_mlp_fusion` runs
at all. When unset, `_cute_layer_name` stays unset on the MLP module
and `Qwen3_5MLP.forward` never dispatches here.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import os

import torch

from vllm.logger import init_logger
from vllm.utils.torch_utils import direct_register_custom_op

# Debug gate — when CUTE_DEBUG_MLP_FUSION=1, the op body logs the four
# inputs to `can_fuse` once per unique (layer_name, nat, can_fuse) tuple.
# Keying on nat + can_fuse (not just layer_name) captures every distinct
# batch shape / gate state the op sees — e.g., warmup (nat=65536,
# can_fuse=False) AND real decode (nat=4, can_fuse=True/False). Zero
# steady-state overhead once each tuple has fired once.
_DEBUG_MLP = os.environ.get("CUTE_DEBUG_MLP_FUSION", "0") == "1"
_DEBUG_SEEN: set[tuple[str, int, bool]] = set()

if TYPE_CHECKING:
    from vllm.v1.attention.backends.cute_paged._backend import (
        CutePagedAttentionImpl,
    )

logger = init_logger(__name__)


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

    `x`: [num_tokens, hidden] post-attn-RMSNorm hidden states. When the
         attention uber-kernel fused RMSNorm (`impl._fusion_active`), this
         is numerically equivalent to `impl.rmsnorm_output[:num_tokens]`.
    `output`: [num_tokens, hidden] pre-allocated by the caller; mutated
              in-place.
    `layer_name`: registry key set by `attach_mlp_fusion`.

    Runs the Phase D fused MLP kernel when the per-step gate passes;
    otherwise runs the unfused `gate_up + silu_mul + down` sequence via
    module refs stashed at attach time. Both branches live inside the
    opaque op body so torch.compile sees only a single opaque call.
    Falls back to the unfused path on any kernel exception
    (fail-closed — one bad step doesn't crash the server).
    """
    impl = _CUTE_MLP_REGISTRY.get(layer_name)
    if impl is None:
        # Should be unreachable: Qwen3_5MLP.forward only dispatches here
        # when `_cute_layer_name` is set, which `attach_mlp_fusion` sets
        # in the same block where it inserts into this registry.
        raise RuntimeError(
            f"cute_mlp_forward called for unregistered layer {layer_name!r}"
        )

    nat = x.shape[0]
    b_mlp  = getattr(impl, "_mlp_fusion_bound", False)
    b_attn = getattr(impl, "_fusion_active", False)
    max_n  = getattr(impl, "_fusion_max_num_seqs", 0)
    # Gate on bound + buffer-safety only. `_fusion_active` is NOT part
    # of the gate because under PIECEWISE CUDA graphs it's False at
    # capture (prefill shape) and the captured graph would bake in the
    # fallback path. `x` is post-RMSNorm regardless of attn fusion, so
    # the kernel doesn't need to know which path produced it.
    can_fuse = b_mlp and nat <= max_n

    if _DEBUG_MLP:
        _dbg_key = (layer_name, nat, can_fuse)
        if _dbg_key not in _DEBUG_SEEN:
            logger.info(
                "[CUTE_DEBUG_MLP_FUSION] layer=%s _mlp_fusion_bound=%s "
                "_fusion_active=%s nat=%d _fusion_max_num_seqs=%d "
                "x.shape=%s can_fuse=%s",
                layer_name, b_mlp, b_attn, nat, max_n, tuple(x.shape), can_fuse,
            )
            _DEBUG_SEEN.add(_dbg_key)

    if can_fuse:
        try:
            # When attention fusion ran, `impl.rmsnorm_output[:nat]` holds
            # the attn uber-kernel's post-RMSNorm buffer — use it directly
            # (avoids capture-vs-replay mismatch in the decoder's
            # `if _fusion_active:` branch). Otherwise `x` is the
            # Python-computed `post_attention_layernorm` output. Both are
            # post-norm BF16; the kernel doesn't distinguish.
            if b_attn:
                x_src = impl.rmsnorm_output[:nat]
            else:
                x_src = x
            impl.mlp_partial_fp32[:nat, :].zero_()
            impl.mlp_arrival_count[:nat].zero_()

            # Kernel writes directly into `output[:nat]` (the op's
            # mutates_args target, which Inductor's downstream piece
            # reads from). `impl.mlp_output` is unused on the fused path
            # but kept allocated for potential debug use.
            #
            # stream=None → Phase_D_MLP_Kernel wraps the current torch
            # stream internally. Passing a torch.cuda.Stream directly
            # fails an internal CuTe DSL isinstance check.
            #
            # `_mlp_gate_up_gs` / `_mlp_down_gs` are NVFP4 weight_global_
            # scale factors cached at attach time (Python floats, no
            # per-step device sync). Without them the kernel output is
            # off by prod(1/wgs); see Phase D2e trace summary.
            impl._mlp_kernel(
                x_src,
                impl._mlp_gate_w,
                impl._mlp_gate_s,
                impl._mlp_up_w,
                impl._mlp_up_s,
                impl._mlp_down_w,
                impl._mlp_down_s,
                impl.mlp_partial_fp32[:nat],
                impl.mlp_arrival_count[:nat],
                output[:nat],
                nat,
                gate_up_global_scale=impl._mlp_gate_up_gs,
                down_global_scale=impl._mlp_down_gs,
            )
            if nat < output.shape[0]:
                output[nat:].zero_()
            return
        except Exception as e:  # noqa: BLE001 — fail closed, log + fallback
            logger.warning(
                "CuTe MLP fusion launch failed (fallback to unfused) "
                "layer=%s nat=%d %s: %r",
                layer_name, nat, type(e).__name__, e,
            )

    # Unfused fallback. Plain torch.nn.Modules; calling them inside an
    # opaque custom op is safe — inner ops run eagerly, nothing is traced.
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

    Signature must match `_cute_mlp_forward_impl` exactly. Returns None
    because the op mutates `output` in place; no outputs.
    """
    return


direct_register_custom_op(
    op_name="cute_mlp_forward",
    op_func=_cute_mlp_forward_impl,
    mutates_args=["output"],
    fake_impl=_cute_mlp_forward_fake,
)


# --- Phase F.1: cute_phase_e_dispatch --------------------------------------
# Opaque replacement for the dead-branching `if _phase_e_consumed:` gate at
# qwen3_5.py:473. Op body reads impl._phase_e_consumed at call time (runtime,
# not trace time), branches to consume β output or delegate to cute_mlp_forward.
#
# Pairs with cute_phase_e_skip_input_layernorm — dispatcher sets
# impl._phase_e_skip_next_ln=True when consumed, skip-op reads it on layer N+1.

def _cute_phase_e_dispatch_impl(
    x: torch.Tensor,
    hidden_out: torch.Tensor,
    residual_out: torch.Tensor,
    residual_in: torch.Tensor,
    layer_name: str,
) -> None:
    impl = _CUTE_MLP_REGISTRY.get(layer_name)
    if impl is None:
        raise RuntimeError(
            f"cute_phase_e_dispatch called for unregistered "
            f"layer {layer_name!r}"
        )
    nat = x.shape[0]

    if getattr(impl, "_phase_e_consumed", False):
        # Consume β output. Fail-loud — no try/except per spec Decision 5.
        hidden_out[:nat].copy_(impl.next_hidden_scratch[:nat])
        residual_out[:nat].copy_(impl.residual_output[:nat])
        impl._phase_e_consumed = False
        impl._phase_e_skip_next_ln = True
        return

    # Not-consumed: β did not run this layer; delegate to regular MLP op.
    torch.ops.vllm.cute_mlp_forward(x, hidden_out, layer_name)
    residual_out.copy_(residual_in)
    impl._phase_e_skip_next_ln = False


def _cute_phase_e_dispatch_fake(
    x: torch.Tensor,
    hidden_out: torch.Tensor,
    residual_out: torch.Tensor,
    residual_in: torch.Tensor,
    layer_name: str,
) -> None:
    return


direct_register_custom_op(
    op_name="cute_phase_e_dispatch",
    op_func=_cute_phase_e_dispatch_impl,
    mutates_args=["hidden_out", "residual_out"],
    fake_impl=_cute_phase_e_dispatch_fake,
)


# --- Phase F.1: cute_phase_e_skip_input_layernorm --------------------------
# Opaque wrap of layer N+1's self.input_layernorm(hidden_states, residual)
# call at qwen3_5.py:386. Reads impl._phase_e_skip_next_ln (set by the
# previous layer's cute_phase_e_dispatch when it consumed β output) and
# either passes through (skip) or runs the module normally.
#
# State flows ACROSS a layer boundary (layer N writes, N+1 reads). Ordering
# is guaranteed by the decoder's sequential forward() calls.

def _cute_phase_e_skip_input_layernorm_impl(
    x: torch.Tensor,
    residual: torch.Tensor,
    out_x: torch.Tensor,
    out_residual: torch.Tensor,
    layer_name: str,
) -> None:
    impl = _CUTE_MLP_REGISTRY.get(layer_name)
    if impl is None:
        raise RuntimeError(
            f"cute_phase_e_skip_input_layernorm called for unregistered "
            f"layer {layer_name!r}"
        )
    nat = x.shape[0]

    if getattr(impl, "_phase_e_skip_next_ln", False):
        # Skip: previous layer's β ε epilogue already applied THIS layer's
        # input_layernorm. Pass-through.
        out_x[:nat].copy_(x[:nat])
        out_residual[:nat].copy_(residual[:nat])
        impl._phase_e_skip_next_ln = False
        return

    # Normal path: run input_layernorm.
    input_ln = getattr(impl, "_input_layernorm_module", None)
    if input_ln is None:
        raise RuntimeError(
            f"cute_phase_e_skip_input_layernorm: layer {layer_name!r} "
            f"has no _input_layernorm_module attached. Check that "
            f"attach_input_layernorm was called at model init."
        )
    ln_out, ln_residual = input_ln._forward_static_with_residual(
        input_ln.weight.data, input_ln.variance_epsilon, x, residual
    )
    out_x[:nat].copy_(ln_out[:nat])
    out_residual[:nat].copy_(ln_residual[:nat])


def _cute_phase_e_skip_input_layernorm_fake(
    x: torch.Tensor,
    residual: torch.Tensor,
    out_x: torch.Tensor,
    out_residual: torch.Tensor,
    layer_name: str,
) -> None:
    return


direct_register_custom_op(
    op_name="cute_phase_e_skip_input_layernorm",
    op_func=_cute_phase_e_skip_input_layernorm_impl,
    mutates_args=["out_x", "out_residual"],
    fake_impl=_cute_phase_e_skip_input_layernorm_fake,
)
