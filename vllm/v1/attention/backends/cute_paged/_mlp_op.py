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
    max_n  = getattr(impl, "_fusion_max_tokens", 0)
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
                "_fusion_active=%s nat=%d _fusion_max_tokens=%d "
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
# Post-C1.5: the consume branch reads impl.mlp_output (raw post-MLP hidden);
# layer N+1's input_layernorm runs from Python at layer entry (no F.1 bake).

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
        hidden_out[:nat].copy_(impl.mlp_output[:nat])
        residual_out[:nat].copy_(impl.residual_output[:nat])
        impl._phase_e_consumed = False
        return

    # Not-consumed: β did not run this layer; delegate to regular MLP op.
    torch.ops.vllm.cute_mlp_forward(x, hidden_out, layer_name)
    residual_out.copy_(residual_in)


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


# --- Phase F.1: cute_phase_e_skip_input_layernorm — DELETED (C1.5) ---------
# The skip-op + bake-into-previous-layer scheme was removed in C1.5 because
# it corrupted the residual stream at non-fusion-active layers (linear-attn
# layers in Qwen3.5's stride-4 pattern do not honor the skip flag). Layer
# input_layernorm now runs unconditionally at every layer entry; β-coop's
# ε epilogue (Phase 4) was deleted in the same commit.
#
# See docs/research/uber_kernel_migration/q4_brainstorm_layer_LN_2026-04-25.md.


# --- 2026-04-26: cute_residual_mirror -----------------------------------------
# Opaque op for the residual mirror copy that qwen3_5.py's decoder forward
# does at layer entry: `impl.residual_buf[:nat].copy_(residual[:nat])`.
#
# Why an op: the prior plain-Python `.copy_()` was inside an `if fusion_could_run:
# try: ... attn_md = get_forward_context().attn_metadata[layer_name] ...`
# block. Under @support_torch_compile (model.forward), dynamo traced the
# get_forward_context lookup (None at trace time) → TypeError → except
# pass. The captured graph then dropped the `.copy_` because (a) the
# inferred trace path always took the except branch and (b) `impl.residual_buf`
# is mutated state torch.compile doesn't track as a graph output. Result at
# runtime: residual_buf stayed at the CUDA-graph-allocator-zeroed value;
# β-coop read zeros; gibberish. (Verified 2026-04-26 via /tmp/nvllm-dumps —
# residual_in absmax=0.0000 across all 16 full-attn layers.)
#
# Wrapping the copy in an opaque custom op makes it a black-box side-effect
# from torch.compile's perspective — it's preserved across graph capture and
# always runs at runtime.

_RES_MIRROR_DIAG_SEEN: set[int] = set()


def _cute_residual_mirror_impl(
    residual_buf: torch.Tensor,
    residual: torch.Tensor,
) -> None:
    """Copy `residual` into `residual_buf` (in-place mutation).

    Direct buffer-passing replaces the prior registry-lookup design:
    `mutates_args=["residual_buf"]` tells torch.compile the op has a
    real side effect on a tracked tensor, so it isn't dead-eliminated.
    """
    nat = residual.shape[0]
    if nat == 0:
        return
    nat = min(nat, residual_buf.shape[0])
    # 2026-04-26 DIAG: one-shot per residual_buf identity. Logs whether
    # the op fires at runtime + the input magnitude. Remove after ship.
    _key = id(residual_buf)
    if _key not in _RES_MIRROR_DIAG_SEEN:
        _RES_MIRROR_DIAG_SEEN.add(_key)
        logger.info(
            "[RES_MIRROR_OP] nat=%d residual_absmax=%.4e "
            "buf_shape=%s buf_pre_absmax=%.4e",
            nat,
            residual.float().abs().max().item(),
            tuple(residual_buf.shape),
            residual_buf.float().abs().max().item(),
        )
    residual_buf[:nat].copy_(residual[:nat])


def _cute_residual_mirror_fake(
    residual_buf: torch.Tensor,
    residual: torch.Tensor,
) -> None:
    return


direct_register_custom_op(
    op_name="cute_residual_mirror",
    op_func=_cute_residual_mirror_impl,
    mutates_args=["residual_buf"],
    fake_impl=_cute_residual_mirror_fake,
)
