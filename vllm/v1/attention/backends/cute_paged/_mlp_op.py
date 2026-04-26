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

# 2026-04-26 (B-fix): attn-consume registry, populated by
# `CutePagedAttentionImpl.attach_fusion`. Same impl object as
# _CUTE_MLP_REGISTRY but keyed by ATTENTION layer name (e.g.
# `language_model.model.layers.3.self_attn.attn`), not the MLP key
# used by cute_phase_e_dispatch. Allows cute_attn_consume and
# cute_post_attn_ln_dispatch to look up the impl and read its
# Python-side flags at runtime — avoids the .item() host-device sync
# on a 0-dim tensor signal (which raises cudaErrorStreamCaptureInvalidated
# under CUDA graph capture, verified 2026-04-26).
_CUTE_ATTN_REGISTRY: dict[str, "CutePagedAttentionImpl"] = {}


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


# --- 2026-04-26: cute_attn_consume + cute_post_attn_ln_dispatch ----------------
# B-fix: replace the dead-eliminated Python `if _fusion_active` consume branch
# at qwen3_5.py:466-476 and the dead-eliminated `if not _fusion_active`
# post_attention_layernorm gate at qwen3_5.py:490-496.
#
# WHY needed: the captured FX graph (verified 2026-04-26 via
# /root/.cache/vllm/torch_compile_cache/<hash>/rank_0_0/backbone/computation_graph.py)
# specialized BOTH gates at trace time on `_fusion_active = False` (the impl's
# __init__ default) — dynamo can't see the runtime mutation that happens inside
# the unified_attention opaque op. Result: the consume copy was DCE'd, the
# legacy Python o_proj + post_attn_LN ALWAYS ran, β-coop's rmsnorm_output /
# residual_output were never read by the captured graph. In dual-fire this
# happened to produce coherent output because paged populated `output` with
# Phase A and the Python pipeline applied o_proj + post_attn_LN over it. In
# solo (paged gated off, β-coop only), `output` stayed uninitialised and
# Python applied o_proj over junk → gibberish.
#
# Fix: route the consume / postln decision through a runtime tensor signal
# (`impl._fusion_active_signal`, 0-dim int32) that's mutated INSIDE the
# unified_attention op (invisible to dynamo's specialization) and read at
# runtime via .item() inside these opaque ops. Both ops always run, dispatch
# at runtime via the signal value:
#   signal == 0 : non-fusion mode (β-coop didn't fire). consume no-ops;
#                 postln applies the fused-residual RMSNorm in-place over
#                 the Python o_proj's wo_out.
#   signal > 0  : fusion mode (β-coop fired with N=signal tokens). consume
#                 copies β-coop's rmsnorm_output → self_attention_output and
#                 residual_output → residual; postln no-ops (β-coop's Phase
#                 1C already produced LN(post_input_LN_residual + wo_out)·γ).
#
# residual_buf and gate_buf are passed to consume as PHANTOM inputs (not
# read inside the body) — their sole purpose is to give the cute_residual_mirror
# and cute_residual_mirror(gate_buf, ...) ops observable downstream readers
# in the captured graph, which prevents dynamo's DCE from dropping them
# (verified empirically that mutates_args alone is NOT sufficient against
# DCE — the ops were dead-eliminated despite mutates_args=["residual_buf"]
# until a downstream reader was added).


def _cute_attn_consume_impl(
    self_attention_output: torch.Tensor,  # mutated [num_tokens, hidden_dim] BF16
    residual: torch.Tensor,                # mutated [num_tokens, hidden_dim] BF16
    rmsnorm_output: torch.Tensor,          # impl.rmsnorm_output [max_num_seqs, hidden_dim] BF16
    residual_output: torch.Tensor,         # impl.residual_output [max_num_seqs, hidden_dim] BF16
    residual_buf: torch.Tensor,            # phantom for cute_residual_mirror dep
    gate_buf: torch.Tensor,                # phantom for gate-mirror dep
    layer_name: str,                       # registry key into _CUTE_ATTN_REGISTRY
) -> None:
    """If β-coop fired this step: copy its outputs into model-side tensors.

    Reads `impl._phase_e_use_beta_coop` (Python attr) at runtime via
    `_CUTE_ATTN_REGISTRY[layer_name]` — no .item() call, no CUDA sync,
    safe under CUDA graph capture. Reset to False at top of impl.forward,
    set to True only on successful β-coop launch — so True ⇔ β-coop wrote
    rmsnorm_output and residual_output for THIS forward call.
    """
    impl = _CUTE_ATTN_REGISTRY.get(layer_name)
    # 2026-04-26 (B-fix v2): gate on `_fusion_bound` (set once at
    # attach_fusion, stable across warmup + runtime) rather than
    # `_phase_e_use_beta_coop` (set per-step inside impl.forward — not
    # consistently True at warmup capture time, so the captured segment
    # would skip the consume kernels and replay would never fill
    # self_attention_output from β-coop's outputs). With _fusion_bound:
    # capture always sees True for fusion-bound full-attn layers,
    # consume kernels always captured. Cost: if β-coop ever fails to
    # fire at runtime (e.g. predicate fails), consume reads stale
    # impl.rmsnorm_output. Mitigated by the predicate hard-gate landed
    # in the prior commit which prevents silent β-coop fallthrough on
    # cooperative-launch-too-large.
    if impl is None or not getattr(impl, "_fusion_bound", False):
        # Non-fusion / non-bound: leave self_attention_output as-is (Python
        # o_proj already wrote it) and residual untouched.
        return
    # Fusion mode: β-coop's Phase 1C produced these. Bound by buffer capacity
    # defensively (matches the original Python consume branch).
    nat = min(self_attention_output.shape[0], rmsnorm_output.shape[0])
    self_attention_output[:nat].copy_(rmsnorm_output[:nat])
    if nat < self_attention_output.shape[0]:
        # Match the prior `if nat < num_tokens: self_attention_output[nat:].zero_()`
        # — keeps unused rows deterministic across decode steps.
        self_attention_output[nat:].zero_()
    residual[:nat].copy_(residual_output[:nat])


def _cute_attn_consume_fake(
    self_attention_output: torch.Tensor,
    residual: torch.Tensor,
    rmsnorm_output: torch.Tensor,
    residual_output: torch.Tensor,
    residual_buf: torch.Tensor,
    gate_buf: torch.Tensor,
    layer_name: str,
) -> None:
    return


direct_register_custom_op(
    op_name="cute_attn_consume",
    op_func=_cute_attn_consume_impl,
    # Both self_attention_output and residual are mutated when fusion fires;
    # the phantom inputs are read-only.
    mutates_args=["self_attention_output", "residual"],
    fake_impl=_cute_attn_consume_fake,
)


def _cute_post_attn_ln_dispatch_impl(
    hidden_states: torch.Tensor,  # mutated [num_tokens, hidden_dim] BF16
    residual: torch.Tensor,        # mutated [num_tokens, hidden_dim] BF16
    weight: torch.Tensor,          # post_attention_layernorm.weight [hidden_dim] BF16
    rmsnorm_eps: float,
    layer_name: str,               # registry key into _CUTE_ATTN_REGISTRY
) -> None:
    """If β-coop did NOT fire: apply fused-residual post_attention_layernorm.

    Mirrors `_forward_static_with_residual` in vllm/nvllm/layers/layernorm.py:
        combined = hidden_states + residual
        residual = combined
        x = combined.float()
        var = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(var + eps)
        x = x * (1.0 + weight.float())
        hidden_states = x.to(combined.dtype)

    When β-coop fired, its Phase 1C already produced this exact output into
    hidden_states via cute_attn_consume above, and residual already holds
    residual_post_attn — skip to avoid double-LN.

    Reads `impl._phase_e_use_beta_coop` (Python attr) — no .item() needed,
    CUDA-graph-safe. See cute_attn_consume docstring for the gate semantics.
    """
    impl = _CUTE_ATTN_REGISTRY.get(layer_name)
    # See cute_attn_consume docstring above for why we gate on _fusion_bound
    # rather than _phase_e_use_beta_coop. Symmetric: when consume fires,
    # post_attn_LN must skip; when consume no-ops, post_attn_LN must apply.
    if impl is not None and getattr(impl, "_fusion_bound", False):
        # Fusion mode: β-coop already did post_attn_LN. Skip.
        return
    # Non-fusion mode: replicate _forward_static_with_residual in-place.
    combined = hidden_states + residual
    residual.copy_(combined)
    x = combined.float()
    var = x.pow(2).mean(dim=-1, keepdim=True)
    x = x * torch.rsqrt(var + rmsnorm_eps)
    x = x * (1.0 + weight.float())
    hidden_states.copy_(x.to(combined.dtype))


def _cute_post_attn_ln_dispatch_fake(
    hidden_states: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    rmsnorm_eps: float,
    layer_name: str,
) -> None:
    return


direct_register_custom_op(
    op_name="cute_post_attn_ln_dispatch",
    op_func=_cute_post_attn_ln_dispatch_impl,
    mutates_args=["hidden_states", "residual"],
    fake_impl=_cute_post_attn_ln_dispatch_fake,
)
