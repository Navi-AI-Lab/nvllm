# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Copyright 2025 The vLLM team.
# Copyright 2025 The Qwen Team.
# Copyright 2025 The HuggingFace Inc. team.
# All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Inference-only Qwen3.5 Series compatible with HuggingFace weights."""

import os
import sys
import time as _cute_time
import typing
from collections.abc import Callable, Iterable
from itertools import islice

import torch
from torch import nn

# Side-effect import: registers torch.ops.vllm.cute_beta_coop_run.
# Mirrors vllm/nvllm/layers/mlp.py:21 pattern. Importing here ensures
# the op exists at torch.compile trace time even for the attached-fusion
# branch.
import vllm.v1.attention.backends.cute_paged._beta_coop_op  # noqa: F401
import vllm.v1.attention.backends.cute_paged._wo_output_reset_op  # noqa: F401

# C2 diag env load (vLLM's EngineCore subprocess strips most of pid-1's env;
# source the gate config from a sentinel file written by serve-cute.sh).
# No-op when the file is absent (production behavior unchanged).
# See docs/research/uber_kernel_migration/2026-04-26-c2-diagnostic-plan.md.
_C2_ENV_FILE = "/tmp/c2_diag/ENV"
if os.path.isfile(_C2_ENV_FILE):
    with open(_C2_ENV_FILE) as _c2_f:
        for _c2_ln in _c2_f:
            if "=" in _c2_ln and (
                _c2_ln.startswith("CUTE_C2_")
                or _c2_ln.startswith("CUTE_WO_SPLIT=")
            ):
                _c2_k, _c2_v = _c2_ln.strip().split("=", 1)
                if _c2_v:  # skip empty values so we don't shadow real env
                    os.environ.setdefault(_c2_k, _c2_v)
    del _c2_f, _c2_ln, _c2_k, _c2_v
del _C2_ENV_FILE

# Per-step instrumentation. Env-gated (CUTE_DEBUG_TIMING=1) so production
# stays untouched. CUTE_DEBUG_TIMING_BUDGET caps total log lines to avoid
# 65k-token spam at long contexts.
_CUTE_DEBUG_TIMING = os.environ.get("CUTE_DEBUG_TIMING", "0") == "1"
_CUTE_DEBUG_TIMING_BUDGET = int(os.environ.get("CUTE_DEBUG_TIMING_BUDGET", "200"))
_CUTE_DEBUG_TIMING_STATE = {"emitted": 0}


def _cute_tlog(msg: str) -> None:
    sys.stderr.write(f"[CUTE_TIMING] {msg}\n")
    sys.stderr.flush()


from vllm.compilation.decorators import support_torch_compile
from vllm.config import VllmConfig
from vllm.distributed import (
    get_pp_group,
    get_tensor_model_parallel_world_size,
)
from vllm.logger import init_logger
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.fused_moe import SharedFusedMoE
from vllm.model_executor.layers.linear import (
    QKVParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.mamba.gdn_linear_attn import GatedDeltaNetAttention
from vllm.model_executor.layers.mamba.mamba_utils import (
    MambaStateCopyFunc,
    MambaStateCopyFuncCalculator,
    MambaStateDtypeCalculator,
    MambaStateShapeCalculator,
)
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader,
    maybe_remap_kv_scale_name,
)
from vllm.model_executor.models.interfaces import (
    EagleModelMixin,
    HasInnerState,
    IsHybrid,
    MixtureOfExperts,
    MultiModalEmbeddings,
    SupportsEagle3,
    SupportsLoRA,
    SupportsPP,
    _require_is_multimodal,
)
from vllm.model_executor.models.qwen3_next import (
    Qwen3NextSparseMoeBlock,
    QwenNextMixtureOfExperts,
)
from vllm.model_executor.models.qwen3_vl import (
    Qwen3_VisionTransformer,
    Qwen3VLDummyInputsBuilder,
    Qwen3VLForConditionalGeneration,
    Qwen3VLMultiModalProcessor,
    Qwen3VLProcessingInfo,
)
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    PPMissingLayer,
    _merge_multimodal_embeddings,
    extract_layer_index,
    is_pp_missing_parameter,
    make_empty_intermediate_tensors_factory,
    make_layers,
    maybe_prefix,
)
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.nvllm.layers.layernorm import Qwen3_5RMSNorm
from vllm.nvllm.layers.mlp import Qwen3_5MLP
from vllm.sequence import IntermediateTensors
from vllm.transformers_utils.configs.qwen3_5 import (
    Qwen3_5Config,
    Qwen3_5TextConfig,
)
from vllm.transformers_utils.configs.qwen3_5_moe import (
    Qwen3_5MoeConfig,
    Qwen3_5MoeTextConfig,
)

logger = init_logger(__name__)


class Qwen3_5ProcessingInfo(Qwen3VLProcessingInfo):
    def get_hf_config(self):
        return self.ctx.get_hf_config(Qwen3_5Config)


class Qwen3_5MoeProcessingInfo(Qwen3VLProcessingInfo):
    def get_hf_config(self):
        return self.ctx.get_hf_config(Qwen3_5MoeConfig)


class Qwen3_5Attention(nn.Module):
    """Qwen3.5 attention block.

    Inlined copy of Qwen3NextAttention as of fusion-ship commit 37cceaa6c,
    with the fusion side-channel (`_fusion_active` write, `fusion_active` arg)
    removed. Impl owns all fusion state; this class only unconditionally writes
    `gate_buf` when `attn_output_gate=True` and leaves the decision to fuse
    to `CutePagedAttentionImpl.forward`.
    """

    def __init__(
        self,
        config,
        model_config,
        cache_config,
        quant_config,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = config.num_attention_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = config.num_key_value_heads
        if self.total_num_kv_heads >= tp_size:
            assert self.total_num_kv_heads % tp_size == 0
        else:
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.head_dim = config.head_dim or (self.hidden_size // self.num_heads)
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.dual_chunk_attention_config = getattr(
            config, "dual_chunk_attention_config", None
        )
        self.attn_output_gate = getattr(config, "attn_output_gate", True)

        self.qkv_proj = QKVParallelLinear(
            config.hidden_size,
            self.head_dim,
            self.total_num_heads * (1 + self.attn_output_gate),
            self.total_num_kv_heads,
            bias=getattr(config, "qkv_bias", False),
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )

        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            config.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )

        self.rotary_emb = get_rope(
            head_size=self.head_dim,
            max_position=config.max_position_embeddings,
            rope_parameters=config.rope_parameters,
            dual_chunk_attention_config=self.dual_chunk_attention_config,
        )

        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.attn",
            **{
                "layer_idx": extract_layer_index(prefix),
                "dual_chunk_attention_config": self.dual_chunk_attention_config,
            }
            if self.dual_chunk_attention_config
            else {},
        )

        self.q_norm = Qwen3_5RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Qwen3_5RMSNorm(self.head_dim, eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        output: torch.Tensor,
        hidden_states: torch.Tensor,
        # NEW (Phase 3): framework-output route. When all three non-None,
        # β-coop / fall-through writes through these (output is reused
        # as output_rmsnorm).
        residual: torch.Tensor | None = None,
        output_residual: torch.Tensor | None = None,
        output_mlp: torch.Tensor | None = None,
    ):
        qkv, _ = self.qkv_proj(hidden_states)

        if self.attn_output_gate:
            q_gate, k, v = qkv.split(
                [self.q_size * 2, self.kv_size, self.kv_size], dim=-1
            )
            orig_shape = q_gate.shape[:-1]
            q_gate = q_gate.view(*orig_shape, self.num_heads, -1)
            q, gate = torch.chunk(q_gate, 2, dim=-1)
            q = q.reshape(*orig_shape, -1)
            gate = gate.reshape(*orig_shape, -1)
        else:
            q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
            gate = None

        q = self.q_norm(q.view(-1, self.num_heads, self.head_dim)).view(
            -1, self.num_heads * self.head_dim
        )
        k = self.k_norm(k.view(-1, self.num_kv_heads, self.head_dim)).view(
            -1, self.num_kv_heads * self.head_dim
        )

        q, k = self.rotary_emb(positions, q, k)

        # Unconditionally mirror `gate` into impl's persistent buffer so that
        # CUDA-graph replay sees a stable copy and impl.forward() can choose
        # to read it when fusion is active. When fusion is disabled the copy
        # is a cheap one-off BF16 memcpy; it avoids the old model->impl flag
        # side-channel that was flagged as fragile.
        if gate is not None:
            # 2026-04-26: gate_buf mirror via the same opaque op as
            # residual_buf — the prior plain-Python .copy_() inside
            # try/except (which protected the trace-time-failing
            # `attn_metadata[...]` lookup) was being dead-eliminated by
            # @support_torch_compile dynamo. Same root cause, same fix.
            impl = self.attn.impl
            gate_buf = getattr(impl, "gate_buf", None)
            if gate_buf is not None:
                torch.ops.vllm.cute_residual_mirror(gate_buf, gate)

        # ---- β-coop framework-output route (Phase 3+) ----
        # _beta_coop_framework_output_bound is set in
        # _resolve_mlp_weights when ALL three pre-conditions hold
        # (attn fusion + MLP fusion + coop kernel attached).
        # See feedback_splitting_op_runtime_dispatch.
        impl = self.attn.impl
        _use_framework_output = (
            getattr(impl, "_beta_coop_framework_output_bound", False)
            and residual is not None
            and output_residual is not None
            and output_mlp is not None
        )
        if _use_framework_output:
            # FIX #2 (2026-04-27, friend's analysis): reshape Q/K/V to 3D
            # BEFORE unified_kv_cache_update. This mirrors canonical
            # Attention.forward (vllm/model_executor/layers/attention/
            # attention.py:451-468): query/key/value are reshaped to 3D
            # `[num_tokens, n_heads, head_size]` BEFORE the kv-cache update
            # op AND before the attention op. Our prior version called
            # unified_kv_cache_update with 2D K/V — even if it didn't crash,
            # it bypasses the canonical contract and could write the cache
            # at the wrong stride. The reshape inside _beta_coop_op.py is
            # now redundant (Q/K/V already 3D when the op receives them).
            inner_attn = self.attn  # the framework-side `Attention` instance
            q3d = q.view(-1, inner_attn.num_heads, inner_attn.head_size)
            k3d = k.view(-1, inner_attn.num_kv_heads, inner_attn.head_size)
            v3d = v.view(-1, inner_attn.num_kv_heads, inner_attn.head_size_v)

            # KV-cache update (cute_paged backend has
            # forward_includes_kv_cache_update = False at _backend.py:124,
            # so caller must do this).
            #
            # PHASE 4 BISECT FIX (2026-04-28, friend's analysis):
            # Canonical Attention.forward dispatches `unified_kv_cache_update`
            # via use_direct_call (attention.py:458-487). On CUDA-opaque-attn
            # platforms (our case, cute_paged is opaque), use_direct_call=False
            # → must call via `torch.ops.vllm.unified_kv_cache_update`.
            # The prior unconditional direct Python call was being DCE'd by
            # torch.compile (no graph-op marker for the side effect),
            # corrupting KV cache for every layer of every forward → byte-
            # identical "rome?" gibberish across all Phase 4 configurations.
            if inner_attn.use_direct_call:
                from vllm.model_executor.layers.attention.attention import (
                    unified_kv_cache_update,
                )
                kv_cache_dummy_dep = unified_kv_cache_update(
                    k3d, v3d, inner_attn.layer_name
                )
            else:
                kv_cache_dummy_dep = torch.ops.vllm.unified_kv_cache_update(
                    k3d, v3d, inner_attn.layer_name
                )
            # `attn_input` for β-coop is the post-input-LN hidden_states
            # (already computed by the caller before this forward).
            torch.ops.vllm.cute_beta_coop_run(
                q3d,
                k3d,
                v3d,
                residual,
                hidden_states,  # attn_input
                gate if self.attn_output_gate else torch.empty(0, device=q.device, dtype=torch.bfloat16),
                output,         # → output_rmsnorm
                output_residual,
                output_mlp,
                inner_attn.layer_name,
                kv_cache_dummy_dep=kv_cache_dummy_dep,
            )
            return  # Decoder layer reads `output`, `output_residual`, `output_mlp`.

        # ---- Legacy path (unchanged) ----
        attn_output = self.attn(q, k, v)

        # Apply gate + o_proj in Python when the kernel did not fuse them.
        # `impl._fusion_active` is managed entirely inside impl.forward based
        # on the per-forward decode+boundary check, NOT set from this method.
        impl = self.attn.impl
        if getattr(impl, "_fusion_active", False):
            # Kernel wrote wo_output / rmsnorm_output / residual_output.
            # DecoderLayer.forward reads those directly; this class leaves
            # `output` untouched so the decoder layer can branch on the
            # same flag and copy from impl buffers.
            return

        if self.attn_output_gate and gate is not None:
            gate = torch.sigmoid(gate)
            attn_output = attn_output * gate
        output[:], _ = self.o_proj(attn_output)


class Qwen3_5DecoderLayer(nn.Module):
    """Self-contained Qwen3.5 decoder layer.

    No longer subclasses Qwen3NextDecoderLayer. Fusion state lives on impl;
    this layer calls `impl.attach_fusion(self)` once in __init__.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        layer_type: str,
        prefix: str = "",
    ) -> None:
        super().__init__()

        config = vllm_config.model_config.hf_text_config
        model_config = vllm_config.model_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config

        self.layer_type = layer_type
        self.layer_idx = extract_layer_index(prefix)
        self.prefix = prefix  # needed for MTP opt-out in attach_fusion

        if self.layer_type == "linear_attention":
            self.linear_attn = GatedDeltaNetAttention(
                config=config,
                vllm_config=vllm_config,
                prefix=f"{prefix}.linear_attn",
                gqa_interleaved_layout=False,
                create_in_proj_qkvz=vllm_config.lora_config is None,
            )
        elif self.layer_type == "full_attention":
            self.self_attn = Qwen3_5Attention(
                config,
                model_config=model_config,
                cache_config=cache_config,
                quant_config=quant_config,
                prefix=f"{prefix}.self_attn",
            )
        else:
            raise ValueError(f"Invalid layer_type {self.layer_type}")

        # MLP dispatch on model_type (copied from current child, NOT parent).
        if config.model_type == "qwen3_5_moe_text":
            self.mlp = Qwen3NextSparseMoeBlock(
                vllm_config=vllm_config,
                prefix=f"{prefix}.mlp",
            )
        elif config.model_type == "qwen3_5_text":
            self.mlp = Qwen3_5MLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                prefix=f"{prefix}.mlp",
            )
        else:
            raise ValueError(f"Invalid model_type {config.model_type}")

        self.input_layernorm = Qwen3_5RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_attention_layernorm = Qwen3_5RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

        self.layer_scale = getattr(config, "layer_scale", False)
        if self.layer_scale:
            self.attn_layer_scale = torch.nn.Parameter(
                torch.zeros(1, 1, config.hidden_size),
            )
            self.ffn_layer_scale = torch.nn.Parameter(
                torch.zeros(1, 1, config.hidden_size),
            )

        # Declare fusion intent once. Impl owns state; all gating + rebinding
        # happens inside CutePagedAttentionImpl.attach_fusion() +
        # _resolve_fusion_weights(). Pass `self` so impl reads o_proj,
        # post_attention_layernorm, sizes, and prefix off the live module.
        if self.layer_type == "full_attention":
            try:
                from vllm.v1.attention.backends.cute_paged._backend import (
                    CutePagedAttentionImpl,
                )

                impl = self.self_attn.attn.impl
                if isinstance(impl, CutePagedAttentionImpl):
                    impl.attach_fusion(self)
                    # Phase D: attach MLP fusion after attn fusion is in
                    # place. Only dense Qwen3_5MLP is supported; MoE
                    # (Qwen3NextSparseMoeBlock) falls through to the
                    # unfused path.
                    if isinstance(self.mlp, Qwen3_5MLP):
                        impl.attach_mlp_fusion(
                            self.mlp, layer_name=f"{self.prefix}.mlp"
                        )
            except (ImportError, AttributeError):
                pass

    def forward(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
        positions: torch.Tensor = None,
        **kwargs: object,
    ):
        # CUTE_DEBUG_TIMING checkpoints: only when env-gated, only during
        # decode (num_tokens==1), and only until budget exhausted.
        _ct_log = (
            _CUTE_DEBUG_TIMING
            and hidden_states.shape[0] == 1
            and _CUTE_DEBUG_TIMING_STATE["emitted"] < _CUTE_DEBUG_TIMING_BUDGET
        )
        _ct_marks: list[str] = []
        _ct_t = _cute_time.perf_counter() if _ct_log else 0.0

        def _ct_mark(label: str) -> None:
            nonlocal _ct_t
            if not _ct_log:
                return
            _now = _cute_time.perf_counter()
            _ct_marks.append(f"{label}={(_now - _ct_t) * 1e6:.1f}us")
            _ct_t = _now

        if residual is None:
            # First-layer case: no residual to add.
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
            _ct_mark("input_ln_first")
        else:
            # C1.5: Phase F.1 skip-op deleted. The previous layer's β-coop
            # kernel ends at Phase 3 (no input_LN bake), so every layer
            # entry runs input_layernorm unconditionally.
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
            _ct_mark("input_ln")

        # Impl decides fusion per-forward. We mirror residual into impl's
        # persistent buffer unconditionally when fusion could run (full
        # attention + CuTe impl) so graph-capture sees stable pointers.
        num_tokens = hidden_states.shape[0]
        nat = num_tokens
        impl = None
        if self.layer_type == "full_attention":
            impl = self.self_attn.attn.impl
            # 2026-04-26: residual mirror via opaque custom op. The prior
            # plain-Python .copy_(residual) was inside a try/except whose
            # protected lookup `get_forward_context().attn_metadata[...]`
            # threw at torch.compile trace time. dynamo concluded the
            # try body was always-caught dead code and the captured
            # graph dropped the .copy_. At runtime residual_buf stayed at
            # the CUDA-graph-allocator-zeroed value → β-coop read zeros
            # → gibberish.
            #
            # The opaque op preserves the side effect across graph capture.
            # `residual_buf` is a declared mutates_args so torch.compile
            # tracks the mutation as a real side effect (the prior op
            # version with `mutates_args=[]` was still dead-eliminated).
            if getattr(impl, "_fusion_bound", False):
                torch.ops.vllm.cute_residual_mirror(impl.residual_buf, residual)
        _ct_mark("residual_mirror")

        # ---- Phase 4 framework-output route gate (trace-static only) ----
        # Phase 3 had this gate also check `_framework_decode_only` (Python
        # read of attn_metadata.is_decode_only at this point) and
        # `nat <= _fusion_max_num_seqs`. Both are runtime-only signals that
        # torch.compile bakes to False at trace time → captured graph never
        # entered the framework-output route, only the legacy fall-through
        # ran. Phase 4 makes the gate trace-static so the graph emits
        # `torch.ops.vllm.cute_beta_coop_run` unconditionally. The op's
        # eager body (splitting boundary at compilation.py:722) handles
        # runtime dispatch — β-coop, β-lite, or full-legacy fallback —
        # inside _backend.forward (writer-invariant per Edit 4 below).
        #
        # PHASE 3 ORIGINAL (commented per feedback_comment_not_delete; may
        # be re-enabled if Phase 4 architecture is rolled back):
        # _framework_decode_only = False
        # if self.layer_type == "full_attention" and impl is not None:
        #     try:
        #         from vllm.forward_context import get_forward_context
        #
        #         _attn_md = get_forward_context().attn_metadata
        #         if isinstance(_attn_md, dict):
        #             _attn_md = _attn_md.get(self.self_attn.attn.layer_name)
        #         _framework_decode_only = bool(
        #             getattr(_attn_md, "is_decode_only", False)
        #         )
        #     except (RuntimeError, KeyError, AttributeError, TypeError):
        #         _framework_decode_only = False
        # _framework_output_route = (
        #     self.layer_type == "full_attention"
        #     and impl is not None
        #     and getattr(impl, "_beta_coop_framework_output_bound", False)
        #     and _framework_decode_only
        #     and nat <= getattr(impl, "_fusion_max_num_seqs", 0)
        # )
        _framework_output_route = (
            self.layer_type == "full_attention"
            and impl is not None
            and getattr(impl, "_beta_coop_framework_output_bound", False)
        )
        if _framework_output_route:
            output_residual = torch.empty_like(residual)
            output_mlp = torch.empty_like(residual)
        else:
            output_residual = None
            output_mlp = None

        self_attention_output = torch.empty_like(hidden_states)
        _ct_mark("attn_out_alloc")

        if self.layer_type == "linear_attention":
            self.linear_attn(
                hidden_states=hidden_states,
                output=self_attention_output,
            )
            hidden_states = self_attention_output
            _ct_mark("linear_attn")
        elif self.layer_type == "full_attention":
            self.self_attn(
                hidden_states=hidden_states,
                output=self_attention_output,
                positions=positions,
                residual=residual if _framework_output_route else None,
                output_residual=output_residual,
                output_mlp=output_mlp,
            )
            _ct_mark("self_attn")
            # Phase 3 (2026-04-27): when framework-output route is active,
            # the kernel wrote output_rmsnorm (= self_attention_output) and
            # output_residual directly. Skip the legacy `impl.X` consume —
            # it would copy stale impl-attribute data (impl.rmsnorm_output
            # is unwritten under the framework-output route) over the good
            # framework values, corrupting them until rebound at line ~577.
            # Functionally harmless but wastes compute and obscures intent.
            if (
                impl is not None
                and getattr(impl, "_fusion_active", False)
                and not _framework_output_route
            ):
                # Kernel already did gate*attn, W_O GEMV, residual+RMSNorm.
                self_attention_output[:nat].copy_(impl.rmsnorm_output[:nat])
                if nat < num_tokens:
                    self_attention_output[nat:].zero_()
                residual[:nat].copy_(impl.residual_output[:nat])
                hidden_states = self_attention_output
                _ct_mark("attn_consume")
            else:
                hidden_states = self_attention_output
                _ct_mark("attn_legacy")
        else:
            raise ValueError("Invalid layer_type")

        if self.layer_scale:
            if len(hidden_states.shape) == 2:
                hidden_states = hidden_states * (
                    self.attn_layer_scale.to(hidden_states.dtype)[0] + 1
                )
            else:
                hidden_states = hidden_states * (
                    self.attn_layer_scale.to(hidden_states.dtype) + 1
                )

        if _framework_output_route:
            # β-coop / fall-through wrote through self_attention_output
            # (= output_rmsnorm), output_residual, output_mlp.
            # Decoder layer consumes them directly — no Python
            # post-attn-LN, no Python MLP.
            hidden_states = output_mlp
            residual = output_residual
            return hidden_states, residual

        # ---- Legacy path (unchanged below) ----
        if not getattr(impl, "_fusion_active", False):
            hidden_states, residual = self.post_attention_layernorm(
                hidden_states, residual
            )
            _ct_mark("post_attn_ln")
        else:
            _ct_mark("post_attn_skip")

        # --- C2 DIAG (env-gated, per spec
        # docs/research/uber_kernel_migration/2026-04-26-c2-diagnostic-spec.md)
        # Compares β-coop's outputs (impl.rmsnorm_output/residual_output)
        # against the legacy post-attn-LN outputs (hidden_states/residual)
        # in dual-fire mode under PIECEWISE+graphs. Off by default.
        #
        # ARCHITECTURAL LIMIT (2026-04-27, results doc):
        # `direct_register_custom_op` runs the op's Python body once at
        # graph capture, then captured CUDA ops replay on every step —
        # the body never re-runs during steady-state decode. Combined
        # with the impl's required capture-skip (host-sync inside the op
        # body raises `cudaErrorStreamCaptureInvalidated`), this means
        # the diag fundamentally cannot fire under captured PIECEWISE
        # decode. It can only fire during eager warmup iters, where
        # β-coop's own fire-gate (is_decode_only + cooperative-fitness)
        # often gates it off — yielding spurious `legacy vs 0` verdicts.
        # Use `CUTE_DUMP_TENSORS=1` (in _backend.py) for offline forensics
        # of real β-coop outputs. Tightening the gate with
        # `_phase_e_consumed` was attempted and reverted: under fullgraph
        # compile the attribute is False at trace time → Dynamo DCE'd
        # the entire diag block. The current loose gate at least lets
        # the eager warmup paths fire (uninformative but doesn't crash).
        if (
            os.getenv("CUTE_C2_DIAG") == "1"
            and impl is not None
            and getattr(impl, "_fusion_bound", False)
            and self.layer_type == "full_attention"
            and nat > 0
        ):
            from vllm.v1.attention.backends.cute_paged import _c2_diag

            step_idx = (
                _c2_diag.next_step_idx()
                if self.layer_idx == 0
                else max(0, _c2_diag._STEP_COUNTER - 1)
            )
            # Custom-op call (positional args). The op is opaque to Dynamo
            # via the registered fake_impl returning None — see
            # _c2_diag.py for the registration + rationale.
            torch.ops.vllm.cute_c2_diag_compare(
                self.layer_idx,
                step_idx,
                nat,
                hidden_states,
                residual,
                impl.rmsnorm_output,
                impl.residual_output,
            )

        # Phase E β-lite consume. When the CuTe backend launched the
        # β-lite dispatch inside its forward, the MLP kernel's ε epilogue
        # already produced:
        #   - residual_final  -> impl.residual_output (overwritten in-kernel)
        #   - next_hidden     -> impl.next_hidden_scratch (pre-RMSNorm'd by
        #                        the next decoder layer's input_layernorm
        #                        gamma, or a plain residual_final memcpy
        #                        for the last layer)
        #
        # Phase F.1 (2026-04-24): the consume gate used to be a Python
        # `if getattr(impl, "_phase_e_consumed", False):` block that
        # dead-branched under PIECEWISE CUDA graphs (see
        # memory:feedback_opaque_op_not_enough, project_phase_e_phantom_speedup).
        # Replaced with the cute_phase_e_dispatch opaque op below — op body
        # reads impl._phase_e_consumed at runtime (not trace time), so the
        # replay path picks the right branch regardless of capture-time state.
        # Original Python gate kept commented for history:
        # ORIGINAL (pre-F.1):
        #     if getattr(impl, "_phase_e_consumed", False):
        #         hidden_states = impl.next_hidden_scratch[:nat]
        #         residual = impl.residual_output[:nat]
        #         impl._phase_e_consumed = False
        #         return hidden_states, residual
        #     hidden_states = self.mlp(hidden_states)

        # Phase F.1 opaque dispatch. Attach-state gate is init-time
        # constant (trace-safe); runtime branch happens inside the op body.
        _mlp_layer_name = getattr(self.mlp, "_cute_layer_name", None)
        if _mlp_layer_name is not None:
            hidden_out = torch.empty_like(hidden_states)
            residual_out = torch.empty_like(residual)
            _ct_mark("mlp_alloc")
            torch.ops.vllm.cute_phase_e_dispatch(
                hidden_states,
                hidden_out,
                residual_out,
                residual,
                _mlp_layer_name,
            )
            hidden_states, residual = hidden_out, residual_out
            _ct_mark("mlp_op")
        else:
            hidden_states = self.mlp(hidden_states)
            _ct_mark("mlp_legacy")

        if self.layer_scale:
            if len(hidden_states.shape) == 2:
                hidden_states = hidden_states * (
                    self.ffn_layer_scale.to(hidden_states.dtype)[0] + 1
                )
            else:
                assert len(hidden_states.shape) == len(self.ffn_layer_scale.shape), (
                    f"shape must be the same {len(hidden_states.shape)}, "
                    f"{len(self.ffn_layer_scale.shape)}"
                )
                hidden_states = hidden_states * (
                    self.ffn_layer_scale.to(hidden_states.dtype) + 1
                )

        if _ct_log:
            torch.cuda.synchronize()
            _now = _cute_time.perf_counter()
            _ct_marks.append(f"sync_end={(_now - _ct_t) * 1e6:.1f}us")
            _CUTE_DEBUG_TIMING_STATE["emitted"] += 1
            _cute_tlog(
                f"step_emit={_CUTE_DEBUG_TIMING_STATE['emitted']} "
                f"layer={self.layer_type} prefix={self.prefix} " + " ".join(_ct_marks)
            )

        return hidden_states, residual


@support_torch_compile(
    dynamic_arg_dims={
        "input_ids": 0,
        # positions is of shape (3, seq_len) if mrope is enabled for qwen2-vl,
        # otherwise (seq_len, ).
        "positions": -1,
        "intermediate_tensors": 0,
        "inputs_embeds": 0,
    }
)
class Qwen3_5Model(nn.Module, EagleModelMixin):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()

        config: Qwen3_5TextConfig | Qwen3_5MoeTextConfig = (
            vllm_config.model_config.hf_text_config
        )
        parallel_config = vllm_config.parallel_config

        eplb_config = parallel_config.eplb_config
        self.num_redundant_experts = eplb_config.num_redundant_experts

        self.config = config
        self.enable_lora = vllm_config.lora_config is not None

        self.vocab_size = config.vocab_size

        self.embed_tokens = VocabParallelEmbedding(
            self.vocab_size,
            config.hidden_size,
        )

        def get_layer(prefix: str):
            return Qwen3_5DecoderLayer(
                vllm_config,
                layer_type=config.layer_types[extract_layer_index(prefix)],
                prefix=prefix,
            )

        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers, get_layer, prefix=f"{prefix}.layers"
        )

        # C1.5: Phase F.1 cross-layer binding loops (per-layer + next-layer
        # LN bake) deleted. The skip-op they enabled (cute_phase_e_skip_*)
        # was permanently retired in C1.5 along with β-coop's Phase 4
        # epilogue — every layer now runs input_layernorm unconditionally
        # at layer entry from Python (see Qwen3_5DecoderLayer.forward).
        # The corresponding attach_*  methods on CutePagedAttentionImpl
        # are commented-out (not deleted) in _backend.py per the
        # comment-out-kernel-code rule.

        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["hidden_states", "residual"], config.hidden_size
        )

        if get_pp_group().is_last_rank:
            self.norm = Qwen3_5RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        else:
            self.norm = PPMissingLayer()

        self.aux_hidden_state_layers: tuple[int, ...] = ()

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors | tuple[torch.Tensor, list[torch.Tensor]]:
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = self.embed_input_ids(input_ids)
            residual = None
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]

        aux_hidden_states = self._maybe_add_hidden_state([], 0, hidden_states, residual)
        for layer_idx, layer in enumerate(
            islice(self.layers, self.start_layer, self.end_layer),
            start=self.start_layer,
        ):
            hidden_states, residual = layer(
                positions=positions,
                hidden_states=hidden_states,
                residual=residual,
            )
            self._maybe_add_hidden_state(
                aux_hidden_states, layer_idx + 1, hidden_states, residual
            )

        if not get_pp_group().is_last_rank:
            return IntermediateTensors(
                {"hidden_states": hidden_states, "residual": residual}
            )
        hidden_states, _ = self.norm(hidden_states, residual)
        if aux_hidden_states:
            return hidden_states, aux_hidden_states
        return hidden_states

    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
        return SharedFusedMoE.make_expert_params_mapping(
            self,
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=getattr(self.config, "num_experts", 0),
            num_redundant_experts=self.num_redundant_experts,
        )

    def load_fused_expert_weights(
        self,
        name: str,
        params_dict: dict,
        loaded_weight: torch.Tensor,
        shard_id: str,
        num_experts: int,
    ) -> bool:
        param = params_dict[name]
        weight_loader = typing.cast(Callable[..., bool], param.weight_loader)
        loaded_local_expert = False
        for expert_id in range(num_experts):
            curr_expert_weight = loaded_weight[expert_id]
            success = weight_loader(
                param,
                curr_expert_weight,
                name,
                shard_id,
                expert_id,
                return_success=True,
            )
            if success:
                loaded_local_expert = True

        return loaded_local_expert

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            # self attention
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            # mlp
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
            ("in_proj_ba", "in_proj_b", 0),
            ("in_proj_ba", "in_proj_a", 1),
        ]

        if self.enable_lora:
            stacked_params_mapping.extend(
                [
                    ("in_proj_qkv", "in_proj_qkv", (0, 1, 2)),
                    ("in_proj_z", "in_proj_z", 0),
                ]
            )
        else:
            stacked_params_mapping.extend(
                [
                    ("in_proj_qkvz", "in_proj_qkv", (0, 1, 2)),
                    ("in_proj_qkvz", "in_proj_z", 3),
                ]
            )

        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()
        expert_params_mapping = self.get_expert_mapping()
        is_fused_expert = False
        fused_expert_params_mapping = [
            ("experts.w13_weight", "experts.gate_up_proj", 0, "w1"),
            ("experts.w2_weight", "experts.down_proj", 0, "w2"),
        ]
        num_experts = (
            self.config.num_experts if hasattr(self.config, "num_experts") else 0
        )
        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue

            if name.startswith("mtp."):
                continue

            # Remapping the name of FP8 kv-scale.
            if name.endswith("scale"):
                name = maybe_remap_kv_scale_name(name, params_dict)
                if name is None:
                    continue

            for param_name, weight_name, shard_id in stacked_params_mapping:
                if "experts.gate_up_proj" in name or "experts.down_proj" in name:
                    is_fused_expert = True
                    expert_params_mapping = fused_expert_params_mapping

                if weight_name not in name:
                    continue

                if "mlp.experts" in name:
                    continue

                name = name.replace(weight_name, param_name)
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                # Skip layers on other devices.
                if is_pp_missing_parameter(name, self):
                    continue
                # name = apply_attn_prefix(name, params_dict)
                if name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = param.weight_loader
                if param_name == "in_proj_z" and self.enable_lora:
                    weight_loader(param, loaded_weight)
                else:
                    weight_loader(param, loaded_weight, shard_id)
                break
            else:
                is_expert_weight = False
                for mapping in expert_params_mapping:
                    param_name, weight_name, expert_id, shard_id = mapping
                    if weight_name not in name:
                        continue
                    is_expert_weight = True
                    name_mapped = name.replace(weight_name, param_name)
                    # Skip layers on other devices.
                    if is_pp_missing_parameter(name_mapped, self):
                        continue
                    if is_fused_expert:
                        # qwen3.5 no need to transpose
                        # loaded_weight = loaded_weight.transpose(-1, -2)
                        if "experts.gate_up_proj" in name:
                            loaded_weight = loaded_weight.chunk(2, dim=-2)
                            success_w1 = self.load_fused_expert_weights(
                                name_mapped,
                                params_dict,
                                loaded_weight[0],
                                "w1",
                                num_experts,
                            )
                            success_w3 = self.load_fused_expert_weights(
                                name_mapped,
                                params_dict,
                                loaded_weight[1],
                                "w3",
                                num_experts,
                            )
                            success = success_w1 and success_w3
                        else:
                            # down_proj
                            success = self.load_fused_expert_weights(
                                name_mapped,
                                params_dict,
                                loaded_weight,
                                shard_id,
                                num_experts,
                            )
                        if success:
                            name = name_mapped
                            break
                    else:
                        # Skip loading extra bias for GPTQ models.
                        if (
                            name_mapped.endswith(".bias")
                            or name_mapped.endswith("_bias")
                        ) and name_mapped not in params_dict:
                            continue
                        param = params_dict[name_mapped]
                        weight_loader = param.weight_loader
                        success = weight_loader(
                            param,
                            loaded_weight,
                            name_mapped,
                            shard_id=shard_id,
                            expert_id=expert_id,
                            return_success=True,
                        )
                    if success:
                        name = name_mapped
                        break
                else:
                    if is_expert_weight:
                        # We've checked that this is an expert weight
                        # However it's not mapped locally to this rank
                        # So we simply skip it
                        continue
                    # Skip loading extra bias for GPTQ models.
                    if name.endswith(".bias") and name not in params_dict:
                        continue
                    if is_pp_missing_parameter(name, self):
                        continue
                    if name not in params_dict:
                        logger.warning_once(
                            f"Parameter {name} not found in params_dict, skip loading"
                        )
                        continue
                    param = params_dict[name]
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    weight_loader(param, loaded_weight)
            loaded_params.add(name)
        return loaded_params


class Qwen3_5ForCausalLMBase(
    nn.Module,
    HasInnerState,
    IsHybrid,
    SupportsEagle3,
    SupportsLoRA,
    SupportsPP,
):
    packed_modules_mapping = {
        "qkv_proj": [
            "q_proj",
            "k_proj",
            "v_proj",
        ],
        "gate_up_proj": ["gate_proj", "up_proj"],
        # GDN fused projections.
        "in_proj_qkvz": ["in_proj_qkv", "in_proj_z"],
        "in_proj_ba": ["in_proj_b", "in_proj_a"],
    }

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        config = vllm_config.model_config.hf_text_config
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        cache_config = vllm_config.cache_config

        scheduler_config = vllm_config.scheduler_config
        if cache_config.mamba_cache_mode == "all":
            raise NotImplementedError(
                "Qwen3.5 currently does not support 'all' prefix caching, "
                "please use '--mamba-cache-mode=align' instead"
            )
        self.quant_config = vllm_config.quant_config

        super().__init__()
        self.config = config
        self.scheduler_config = scheduler_config
        self.model = Qwen3_5Model(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )

        # When LoRA is enabled, GDN uses separate in_proj_qkv and in_proj_z
        # instead of merged in_proj_qkvz; pack mapping must match.
        if vllm_config.lora_config:
            base = getattr(Qwen3_5ForCausalLMBase, "packed_modules_mapping", {})
            self.packed_modules_mapping = {k: list(v) for k, v in base.items()}
            self.packed_modules_mapping.pop("in_proj_qkvz", None)
            self.packed_modules_mapping["in_proj_qkv"] = ["in_proj_qkv"]
            self.packed_modules_mapping["in_proj_z"] = ["in_proj_z"]

        if get_pp_group().is_last_rank:
            if config.tie_word_embeddings:
                self.lm_head = self.model.embed_tokens
            else:
                self.lm_head = ParallelLMHead(
                    config.vocab_size,
                    config.hidden_size,
                    prefix=maybe_prefix(prefix, "lm_head"),
                )
        else:
            self.lm_head = PPMissingLayer()

        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def set_aux_hidden_state_layers(self, layers: tuple[int, ...]) -> None:
        self.model.aux_hidden_state_layers = layers

    def get_eagle3_aux_hidden_state_layers(self) -> tuple[int, ...]:
        num_layers = len(self.model.layers)
        return (2, num_layers // 2, num_layers - 3)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ):
        hidden_states = self.model(
            input_ids, positions, intermediate_tensors, inputs_embeds
        )

        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        return self.logits_processor(self.lm_head, hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(
            self,
            skip_prefixes=["mtp."],
        )
        return loader.load_weights(weights)

    @classmethod
    def get_mamba_state_dtype_from_config(
        cls,
        vllm_config: "VllmConfig",
    ) -> tuple[torch.dtype, torch.dtype]:
        return MambaStateDtypeCalculator.gated_delta_net_state_dtype(
            vllm_config.model_config.dtype,
            vllm_config.cache_config.mamba_cache_dtype,
            vllm_config.cache_config.mamba_ssm_cache_dtype,
        )

    @classmethod
    def get_mamba_state_shape_from_config(
        cls, vllm_config: "VllmConfig"
    ) -> tuple[tuple[int, int], tuple[int, int]]:
        parallel_config = vllm_config.parallel_config
        hf_config = vllm_config.model_config.hf_text_config
        tp_size = parallel_config.tensor_parallel_size
        num_spec = (
            vllm_config.speculative_config.num_speculative_tokens
            if vllm_config.speculative_config
            else 0
        )
        return MambaStateShapeCalculator.gated_delta_net_state_shape(
            tp_size,
            hf_config.linear_num_key_heads,
            hf_config.linear_num_value_heads,
            hf_config.linear_key_head_dim,
            hf_config.linear_value_head_dim,
            hf_config.linear_conv_kernel_dim,
            num_spec,
        )

    @classmethod
    def get_mamba_state_copy_func(
        cls,
    ) -> tuple[MambaStateCopyFunc, MambaStateCopyFunc]:
        return MambaStateCopyFuncCalculator.gated_delta_net_state_copy_func()


class Qwen3_5ForCausalLM(Qwen3_5ForCausalLMBase):
    pass


class Qwen3_5MoeForCausalLM(Qwen3_5ForCausalLMBase, QwenNextMixtureOfExperts):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__(vllm_config=vllm_config, prefix=prefix)

        # set MoE hyperparameters
        self.set_moe_parameters()

    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
        return self.model.get_expert_mapping()


########################################################
# Qwen3_5-Dense
########################################################


@MULTIMODAL_REGISTRY.register_processor(
    Qwen3VLMultiModalProcessor,
    info=Qwen3_5ProcessingInfo,
    dummy_inputs=Qwen3VLDummyInputsBuilder,
)
class Qwen3_5ForConditionalGeneration(Qwen3VLForConditionalGeneration, IsHybrid):
    # Qwen3.5 does not support multimodal pruning (EVS).
    supports_multimodal_pruning = False

    packed_modules_mapping = Qwen3VLForConditionalGeneration.packed_modules_mapping | {
        "in_proj_qkvz": ["in_proj_qkv", "in_proj_z"],
        "in_proj_ba": ["in_proj_b", "in_proj_a"],
    }

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "model"):
        # protocols have not __init__ method, so we need to use nn.Module.__init__
        nn.Module.__init__(self)
        self.update_packed_mapping(enable_lora=vllm_config.lora_config is not None)
        config: Qwen3_5Config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        multimodal_config = vllm_config.model_config.multimodal_config

        self.config = config
        self.multimodal_config = multimodal_config
        self.use_data_parallel = multimodal_config.mm_encoder_tp_mode == "data"
        # Qwen3.5 does not support multimodal pruning (EVS).
        self.is_multimodal_pruning_enabled = False

        with self._mark_tower_model(vllm_config, {"image", "video"}):
            self.visual = Qwen3_VisionTransformer(
                config.vision_config,
                norm_eps=getattr(config, "rms_norm_eps", 1e-6),
                quant_config=quant_config,
                prefix=maybe_prefix(prefix, "visual"),
            )

        with self._mark_language_model(vllm_config):
            self.language_model = Qwen3_5ForCausalLM(
                vllm_config=vllm_config, prefix=maybe_prefix(prefix, "language_model")
            )

        self.make_empty_intermediate_tensors = (
            self.language_model.make_empty_intermediate_tensors
        )

    def update_packed_mapping(self, enable_lora: bool):
        # When LoRA is enabled, GDN uses separate in_proj_qkv and in_proj_z
        if enable_lora:
            base = getattr(
                Qwen3_5ForConditionalGeneration, "packed_modules_mapping", {}
            )
            self.packed_modules_mapping = {k: list(v) for k, v in base.items()}
            self.packed_modules_mapping.pop("in_proj_qkvz", None)
            self.packed_modules_mapping["in_proj_qkv"] = ["in_proj_qkv"]
            self.packed_modules_mapping["in_proj_z"] = ["in_proj_z"]

    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: MultiModalEmbeddings | None = None,
        *,
        is_multimodal: torch.Tensor | None = None,
    ) -> torch.Tensor:
        inputs_embeds = self._embed_text_input_ids(
            input_ids,
            self.language_model.embed_input_ids,
            is_multimodal=is_multimodal,
        )

        if multimodal_embeddings is None or len(multimodal_embeddings) == 0:
            return inputs_embeds

        is_multimodal = _require_is_multimodal(is_multimodal)

        inputs_embeds = _merge_multimodal_embeddings(
            inputs_embeds=inputs_embeds,
            multimodal_embeddings=multimodal_embeddings,
            is_multimodal=is_multimodal,
        )

        return inputs_embeds

    def recompute_mrope_positions(self, *args, **kwargs):
        raise NotImplementedError(
            "Qwen3.5 does not support multimodal pruning (EVS). "
            "recompute_mrope_positions should never be called."
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ) -> torch.Tensor | IntermediateTensors:
        """Run forward pass for Qwen3.5.

        Args:
            input_ids: Flattened (concatenated) input_ids corresponding to a
                batch.
            positions: Flattened (concatenated) position ids corresponding to a
                batch.
                **NOTE**: If mrope is enabled (default setting for Qwen3VL
                opensource models), the shape will be `(3, seq_len)`,
                otherwise it will be `(seq_len,).
            intermediate_tensors: Intermediate tensors from previous pipeline
                stages.
            inputs_embeds: Pre-computed input embeddings.
            **kwargs: Additional keyword arguments including:
                - pixel_values: Pixel values to be fed to a model.
                    `None` if no images are passed.
                - image_grid_thw: Tensor `(n_images, 3)` of image 3D grid in
                    LLM. `None` if no images are passed.
                - pixel_values_videos: Pixel values of videos to be fed to a
                    model. `None` if no videos are passed.
                - video_grid_thw: Tensor `(n_videos, 3)` of video 3D grid in
                    LLM. `None` if no videos are passed.
        """

        if intermediate_tensors is not None:
            inputs_embeds = None

        hidden_states = self.language_model.model(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
        )

        return hidden_states

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(
            self,
            skip_prefixes=["mtp."],
        )
        return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)

    @classmethod
    def get_mamba_state_dtype_from_config(
        cls,
        vllm_config: "VllmConfig",
    ) -> tuple[torch.dtype, torch.dtype]:
        return MambaStateDtypeCalculator.gated_delta_net_state_dtype(
            vllm_config.model_config.dtype,
            vllm_config.cache_config.mamba_cache_dtype,
            vllm_config.cache_config.mamba_ssm_cache_dtype,
        )

    @classmethod
    def get_mamba_state_shape_from_config(
        cls, vllm_config: "VllmConfig"
    ) -> tuple[tuple[int, int], tuple[int, int]]:
        parallel_config = vllm_config.parallel_config
        hf_config = vllm_config.model_config.hf_text_config
        tp_size = parallel_config.tensor_parallel_size
        num_spec = (
            vllm_config.speculative_config.num_speculative_tokens
            if vllm_config.speculative_config
            else 0
        )
        return MambaStateShapeCalculator.gated_delta_net_state_shape(
            tp_size,
            hf_config.linear_num_key_heads,
            hf_config.linear_num_value_heads,
            hf_config.linear_key_head_dim,
            hf_config.linear_value_head_dim,
            hf_config.linear_conv_kernel_dim,
            num_spec,
        )

    @classmethod
    def get_mamba_state_copy_func(cls) -> tuple[MambaStateCopyFunc, MambaStateCopyFunc]:
        return MambaStateCopyFuncCalculator.gated_delta_net_state_copy_func()


########################################################
# Qwen3_5-MoE
########################################################


class Qwen3_5_MoeMixtureOfExperts(MixtureOfExperts):
    def update_physical_experts_metadata(
        self,
        num_physical_experts: int,
        num_local_physical_experts: int,
    ) -> None:
        assert self.num_local_physical_experts == num_local_physical_experts
        self.num_physical_experts = num_physical_experts
        self.num_local_physical_experts = num_local_physical_experts
        self.num_redundant_experts = num_physical_experts - self.num_logical_experts
        for layer in self.language_model.model.layers:
            if isinstance(layer.mlp, Qwen3NextSparseMoeBlock):
                moe = layer.mlp
                moe.n_local_physical_experts = num_local_physical_experts
                moe.n_physical_experts = num_physical_experts
                moe.n_redundant_experts = self.num_redundant_experts
                moe.experts.update_expert_map()

    def set_moe_parameters(self):
        self.expert_weights = []

        self.moe_layers = []
        example_moe = None
        for layer in self.language_model.model.layers:
            if isinstance(layer, Qwen3_5DecoderLayer) and isinstance(
                layer.mlp, Qwen3NextSparseMoeBlock
            ):
                example_moe = layer.mlp
                self.moe_layers.append(layer.mlp.experts)

        if example_moe is None:
            raise RuntimeError(
                "No Qwen3_5 layer found in the language_model.model.layers."
            )

        # Set MoE hyperparameters
        self.num_moe_layers = len(self.moe_layers)
        self.num_expert_groups = 1
        self.num_shared_experts = 0
        self.num_logical_experts = example_moe.n_logical_experts
        self.num_physical_experts = example_moe.n_physical_experts
        self.num_local_physical_experts = example_moe.n_local_physical_experts
        self.num_routed_experts = example_moe.n_routed_experts
        self.num_redundant_experts = example_moe.n_redundant_experts


@MULTIMODAL_REGISTRY.register_processor(
    Qwen3VLMultiModalProcessor,
    info=Qwen3_5MoeProcessingInfo,
    dummy_inputs=Qwen3VLDummyInputsBuilder,
)
class Qwen3_5MoeForConditionalGeneration(
    Qwen3_5ForConditionalGeneration, Qwen3_5_MoeMixtureOfExperts
):
    # For MoE LoRA weights loading
    is_3d_moe_weight: bool = True

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "model"):
        # protocols have not __init__ method, so we need to use nn.Module.__init__
        nn.Module.__init__(self)
        self.update_packed_mapping(enable_lora=vllm_config.lora_config is not None)
        config: Qwen3_5MoeConfig = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        multimodal_config = vllm_config.model_config.multimodal_config

        self.config = config
        self.multimodal_config = multimodal_config
        self.use_data_parallel = multimodal_config.mm_encoder_tp_mode == "data"
        # Qwen3.5 does not support multimodal pruning (EVS).
        self.is_multimodal_pruning_enabled = False

        with self._mark_tower_model(vllm_config, {"image", "video"}):
            self.visual = Qwen3_VisionTransformer(
                config.vision_config,
                norm_eps=getattr(config, "rms_norm_eps", 1e-6),
                quant_config=quant_config,
                prefix=maybe_prefix(prefix, "visual"),
            )

        with self._mark_language_model(vllm_config):
            self.language_model = Qwen3_5MoeForCausalLM(
                vllm_config=vllm_config, prefix=maybe_prefix(prefix, "language_model")
            )

        self.make_empty_intermediate_tensors = (
            self.language_model.make_empty_intermediate_tensors
        )

        # set MoE hyperparameters
        self.set_moe_parameters()
