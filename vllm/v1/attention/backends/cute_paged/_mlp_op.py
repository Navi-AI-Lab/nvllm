# Copyright 2026 Navi Ai Labs
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Phase D2 custom-op wrap for CuTe paged fused MLP.

Owns the full MLP dispatch decision inside an opaque
`torch.ops.vllm.cute_mlp_forward`: the per-step gate, the kernel launch,
the fallback unfused GEMMs. torch.compile treats the op body as a black
box, so neither the fused path nor the unfused fallback leaks into the
compiled graph.

Design rationale (verified in
`benchmarks/nvllm/traces/cute_paged_mlp_fusion/2026-04-18-phase-d1-custom-op/summary.md`):

- Phase D1 moved the gate into this op but still launched the kernel as
  a side effect from `_backend.py::forward`. Dynamo still lifted the
  fallback GEMMs into the compiled graph because the fallback's
  `_mlp_gate_up_proj / _mlp_down_proj` calls were reachable from traced
  Python.
- Phase D2 moves the launch inside this op and deletes the attention-side
  side effect. Single call site → single path visible to Inductor.
- `direct_register_custom_op(..., mutates_args=["output"])` produces an
  opaque op that Inductor cannot decompose. Precondition verified in
  `/tmp/phase_d2_opaque_test.py` (compiled graph contains only the op
  call; no addmm / extern_kernels.mm / linear).

The outer env var `CUTE_MLP_FUSION` still gates whether
`attach_mlp_fusion` runs at all; when unset, `_cute_layer_name` stays
unset on the MLP module and `Qwen3_5MLP.forward` never dispatches here.
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
    # Phase D2b: removed `impl._fusion_active` from the gate. The attribute
    # is set per-step inside attention.forward, but under PIECEWISE CUDA
    # graphs the op body Python runs only at *capture* time; at capture,
    # `_fusion_active` is False (warmup uses prefill-shaped metadata), so
    # the captured graph bakes in the fallback's unfused GEMMs and never
    # fires the kernel at replay. The `_fusion_active` bit was only needed
    # when the pre-D2 kernel read `impl.rmsnorm_output`; in D2 we pass `x`
    # directly, and `x` is post-RMSNorm in both fused (`attn uber-kernel
    # wrote rmsnorm_output`) and unfused (`post_attention_layernorm` ran
    # eagerly) paths — the kernel doesn't care which. Buffer-safety gate
    # (`nat <= max_n`) remains; `_mlp_fusion_bound` catches quant/shape
    # mismatches at resolve time.
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
            # Input-source selection (Phase D2c correctness fix):
            # When attention fusion is active at *runtime*, the attention
            # uber-kernel writes a known-good post-RMSNorm buffer at
            # `impl.rmsnorm_output[:nat]`. Reading from it directly matches
            # D1's working behavior and is immune to the
            # capture-vs-replay mismatch in the decoder's
            # `if _fusion_active:` Python branch (see investigation notes in
            # `benchmarks/nvllm/traces/cute_paged_mlp_fusion/2026-04-18-phase-d2-op-body-move/`).
            # When attention fusion is inactive, `x` is the Python-computed
            # post-`post_attention_layernorm` hidden states — also post-norm,
            # also correct.
            if b_attn:
                x_src = impl.rmsnorm_output[:nat]
            else:
                x_src = x
            # Zero per-step mutable buffers. Python-side zero_() is
            # stream-ordered before the kernel launch (same guarantee as
            # the pre-D2 attention-side launch relied on).
            impl.mlp_partial_fp32[:nat, :].zero_()
            impl.mlp_arrival_count[:nat].zero_()

            # Phase D2e: kernel writes directly into the op's `output`
            # argument (`output[:nat]`) instead of into the persistent
            # `impl.mlp_output` buffer followed by a copy. Two reasons:
            # (1) eliminates a same-size redundant BF16 copy per layer
            # per step — small but measurable over 16 × 127 steps;
            # (2) `output` is the tensor Inductor's downstream piece
            # reads from, so writing to it directly removes one layer
            # of indirection that could interact with CUDA-graph capture
            # (the suspected D2d gibberish root cause). `impl.mlp_output`
            # is now unused on the fused path — kept allocated for
            # potential debug/ref use but not read by anyone.
            #
            # Pass stream=None → Phase_D_MLP_Kernel wraps the current
            # torch stream internally. Passing a torch.cuda.Stream
            # directly fails an internal CuTe DSL isinstance check.
            # Phase D2e fix: pass weight_global_scale factors. Without
            # these the kernel returns `true_output × (1/wgs)`, producing
            # the D2 gibberish that D1 accidentally hid via dead-branching.
            # Cached on `impl` at attach time — no per-step device sync.
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
