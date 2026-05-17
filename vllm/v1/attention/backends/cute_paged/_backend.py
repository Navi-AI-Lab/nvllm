# Copyright 2026 Navi Ai Labs
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CuTe DSL paged attention backend classes for SM120/SM121 (GB10).

Custom attention kernel using CuTe Python DSL with FP8 MMA for QK,
BF16 MMA for PV, and CpAsync for paged KV loads. Targets NVIDIA GB10
(DGX Spark) with owned KV page layout optimized for SM121 SMEM budget.

See: docs/superpowers/specs/2026-04-10-cute-paged-attention-design.md
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

import torch
from torch.profiler import record_function

from vllm.logger import init_logger
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionImpl,
    AttentionMetadata,
    AttentionMetadataBuilder,
    AttentionType,
    CommonAttentionMetadata,
)

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.config.cache import CacheDType
    from vllm.platforms.interface import DeviceCapability
    from vllm.v1.kv_cache_interface import AttentionSpec

logger = init_logger(__name__)

# Set CUTE_DEBUG_FUSION=1 to enable per-call diff vs Python-dequant W_O ref.
_DEBUG_FUSION = os.environ.get("CUTE_DEBUG_FUSION", "0") == "1"

# Phase 6a hot-path cache: CUTE_DUMP_TENSORS read once at module import.
# Disables per-forward env parsing in the β-coop branch.
_CUTE_DUMP_TENSORS: bool = os.environ.get("CUTE_DUMP_TENSORS", "0") == "1"

# Phase 6a: gate framework-route shape/dtype/contiguity asserts and the
# one-shot phase3 diagnostic behind this flag. Default OFF in production
# — the asserts ran every full-attn forward × every token despite never
# firing (a successful framework route always satisfies them).
_VERIFY_FRAMEWORK_OUTPUTS: bool = os.environ.get("CUTE_VERIFY_FW", "0") == "1"

# β-coop region-timing instrumentation gate. When set, allocates a
# (num_ctas, _REGION_TIMING_NUM_REGIONS, 2) u64 scratch tensor and
# plumbs it to run_beta_coop_full as `region_timing_buf`. Production
# path is unchanged: env unset → buffer is None → kernel sees the
# timing-off compile path. See plan
# docs/superpowers/plans/2026-05-02-beta-region-breakdown.md (Task 4).
_REGION_TIMING_ENABLED = (
    os.environ.get("CUTE_BETA_REGION_TIMING", "0") == "1"
)
# B' instrumentation (2026-05-16): 13 -> 16 (R13 prologue_pre_r0,
# R14 epilogue_post_r10, R15 phase3_3d_last_cta_gather).
# C' instrumentation (2026-05-16): 16 -> 19 (R16/R17/R18 per-call
# accumulated sums of R7/R8/R9 across the slice loop).
# C' rework (same day): 19 -> 20 (+R19 phase3_post_loop_atomic;
# R9 brackets fixed + renamed phase3_3c_fc2_last_iter).
_REGION_TIMING_NUM_REGIONS = 20

# CuTe DSL disk cache — runtime hookup. Without this call, the env vars
# B12X_CUTE_COMPILE_DISK_CACHE and B12X_CUTE_COMPILE_CACHE_DIR are inert
# at serve time. apply_disk_cache_patch() monkey-patches
# CompileCallable._compile so that subsequent cute.compile() calls
# consult the on-disk cache before invoking NVRTC.
#
# Historical note: build-time warmup was retired with the FULL+blessed AOT
# cache work (see docs/research/2026-04-29-full-graph-spike/); the runtime
# now consults the disk cache directly via this hook.
if os.environ.get("B12X_CUTE_COMPILE_DISK_CACHE", "0") == "1":
    from vllm.v1.attention.backends.cute_paged.disk_cache import (
        apply_disk_cache_patch,
    )
    apply_disk_cache_patch(
        cache_dir=os.environ.get(
            "B12X_CUTE_COMPILE_CACHE_DIR", "/opt/vllm/kernel_cache",
        )
    )


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------


@dataclass
class CutePagedMetadata(AttentionMetadata):
    """Per-batch metadata for CuTe paged attention."""

    num_actual_tokens: int
    slot_mapping: torch.Tensor

    # Batch composition
    num_decodes: int
    num_decode_tokens: int
    num_prefills: int
    num_prefill_tokens: int

    # Sequence info
    seq_lens: torch.Tensor  # [num_seqs] int32 on device
    query_start_loc: torch.Tensor  # [num_seqs + 1] int32 on device
    max_query_len: int
    max_seq_len: int

    # Page table
    block_table: torch.Tensor  # [num_seqs, max_blocks_per_seq] int32

    # Flags
    is_decode_only: bool


# ---------------------------------------------------------------------------
# Phase E env-flag parser
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _PhaseEEnvConfig:
    enabled: bool
    forced_path: str  # "auto" | "coop" | "lite"
    restricted_layers: set[int] | None


def _phase_e_env_config() -> _PhaseEEnvConfig:
    """Parse Phase E env flags. Call once per forward for dispatch decisions.

    CUTE_PHASE_E_FUSION={0,1}    — default 0 (OFF)
    CUTE_PHASE_E_PATH={auto,coop,lite}  — default 'auto'
    CUTE_PHASE_E_LAYERS=<csv>    — optional debug restriction, None = all
    """
    enabled = os.environ.get("CUTE_PHASE_E_FUSION", "0") == "1"
    forced_path = os.environ.get("CUTE_PHASE_E_PATH", "auto").lower()
    if forced_path not in {"auto", "coop", "lite"}:
        forced_path = "auto"
    layers_str = os.environ.get("CUTE_PHASE_E_LAYERS", "").strip()
    restricted_layers: set[int] | None = None
    if layers_str:
        try:
            restricted_layers = {
                int(x.strip()) for x in layers_str.split(",") if x.strip()
            }
        except ValueError:
            restricted_layers = None
    return _PhaseEEnvConfig(
        enabled=enabled,
        forced_path=forced_path,
        restricted_layers=restricted_layers,
    )


# Phase 6a hot-path cache: snapshot Phase E env once at module import.
# Per spec — eliminates per-forward env parsing + set construction
# (~16 forward calls / token at steady state). Runtime mutation of
# CUTE_PHASE_E_FUSION / CUTE_PHASE_E_PATH / CUTE_PHASE_E_LAYERS is no
# longer honored after import. Same contract as _DEBUG_FUSION (L42).
_PHASE_E_ENV: _PhaseEEnvConfig = _phase_e_env_config()


# Spike/debug guard — when set, β-coop failures raise instead of falling
# through to the legacy split path. Intentionally hostile to recovery; not
# a serving safety mode. See:
#   docs/superpowers/specs/2026-04-29-full-and-piecewise-cute-spike-design.md
_PHASE_E_FALLBACK_RAISE: bool = (
    os.environ.get("CUTE_PHASE_E_FALLBACK_RAISE", "0") == "1"
)
if _PHASE_E_FALLBACK_RAISE:
    logger.warning(
        "CUTE_PHASE_E_FALLBACK_RAISE=1 — β-coop fallback is fail-fast. "
        "Any β-coop failure will raise rather than fall through to legacy. "
        "Spike/debug guard only; do NOT enable in serving."
    )


# ---------------------------------------------------------------------------
# Backend
# ---------------------------------------------------------------------------


class CutePagedBackend(AttentionBackend):
    """CuTe DSL paged attention backend for SM120/SM121."""

    accept_output_buffer: bool = True
    forward_includes_kv_cache_update: bool = False

    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.bfloat16]
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [
        "fp8",
        "fp8_e4m3",
    ]

    @staticmethod
    def get_name() -> str:
        return "CUTE_PAGED"

    @staticmethod
    def get_impl_cls() -> type[CutePagedAttentionImpl]:
        return CutePagedAttentionImpl

    @staticmethod
    def get_builder_cls() -> type[CutePagedMetadataBuilder]:
        return CutePagedMetadataBuilder

    @classmethod
    def supports_head_size(cls, head_size: int) -> bool:
        return head_size in (128, 256)

    @classmethod
    def supports_compute_capability(
        cls,
        capability: DeviceCapability,
    ) -> bool:
        return capability.major == 12

    @classmethod
    def supports_kv_cache_dtype(
        cls,
        kv_cache_dtype: CacheDType | None,
    ) -> bool:
        return kv_cache_dtype in ("fp8", "fp8_e4m3")

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int]:
        return [64]

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        if block_size != 64:
            raise ValueError(
                f"CutePagedAttention requires block_size=64, got {block_size}"
            )
        if head_size not in (128, 256):
            raise ValueError(
                f"CutePagedAttention requires head_size 128 or 256, got {head_size}"
            )
        # Dim 1 = 2 for K/V split (matches FlashInfer convention)
        return (num_blocks, 2, 64, num_kv_heads, head_size)

    @staticmethod
    def get_kv_cache_stride_order(
        include_num_layers_dimension: bool = False,
    ) -> tuple[int, ...]:
        # Row-major (identity), matching FlashInfer NHD layout
        # Shape: (num_blocks, 2, page_size, num_kv_heads, head_dim)
        if include_num_layers_dimension:
            return (1, 0, 2, 3, 4, 5)
        return (0, 1, 2, 3, 4)


# ---------------------------------------------------------------------------
# Attention Implementation
# ---------------------------------------------------------------------------


class CutePagedAttentionImpl(AttentionImpl[CutePagedMetadata]):
    """CuTe DSL paged attention forward pass."""

    can_return_lse_for_decode: bool = False

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int | None = None,
        alibi_slopes: list[float] | None = None,
        sliding_window: int | None = None,
        kv_cache_dtype: str = "auto",
        logits_soft_cap: float | None = None,
        attn_type: str = AttentionType.DECODER,
        kv_sharing_target_layer_name: str | None = None,
    ) -> None:
        if sliding_window is not None:
            raise ValueError("CutePagedAttention does not support sliding window")
        if logits_soft_cap is not None:
            raise ValueError("CutePagedAttention does not support logits_soft_cap")
        if attn_type != AttentionType.DECODER:
            raise ValueError(
                f"CutePagedAttention only supports DECODER, got {attn_type}"
            )

        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_kv_heads or num_heads
        self.kv_cache_dtype = kv_cache_dtype
        self.num_queries_per_kv = self.num_heads // self.num_kv_heads
        self.kv_sharing_target_layer_name = kv_sharing_target_layer_name
        self.alibi_slopes = alibi_slopes
        self.sliding_window = sliding_window
        self.logits_soft_cap = logits_soft_cap

        logger.info(
            "CutePagedAttention initialized: %d Q heads, %d KV heads, "
            "head_dim=%d, GQA ratio=%d",
            self.num_heads,
            self.num_kv_heads,
            self.head_size,
            self.num_queries_per_kv,
        )

        # Fusion state is owned exclusively by this impl (spec § Impl side).
        # Buffers are allocated later in attach_fusion() with sizes passed
        # by the model, NOT read from get_current_vllm_config() — avoids the
        # hf_config vs hf_text_config fragility (code-review I1).
        self._fusion_bound = False
        self._fusion_active = False
        self._fusion_attached = False  # set by attach_fusion

        # Phase D MLP fusion state
        self._mlp_fusion_bound = False
        self._mlp_fusion_active = False
        self._mlp_fusion_nat = 0
        self._mlp_attached = False
        self._mlp_kernel = None
        self._mlp_module = None
        self._mlp_num_k_tiles = 0

        # Phase E β-coop framework-output-buffer route. Set True at end of
        # _resolve_mlp_weights() ONLY when all three pre-conditions hold:
        # _fusion_bound, _mlp_fusion_bound, _phase_e_coop_kernel is not None.
        # See feedback_splitting_op_runtime_dispatch + spec § Q3.
        self._beta_coop_framework_output_bound: bool = False

        # Phase E β-lite dispatch state (task 9). `_phase_e_consumed` is read
        # by the Python decoder wrapper (`vllm/nvllm/models/qwen3_5.py`) to
        # decide whether to skip the post-MLP residual `copy_` and the
        # Python `next_input_layernorm` pass — when β-lite runs the MLP
        # kernel's ε epilogue produces both in-kernel. Always defined so
        # the wrapper can use a bare `getattr(..., False)` without None
        # ambiguity (see memory:feedback_kwargs_get_none_default).
        self._phase_e_consumed = False
        self._phase_e_use_beta_lite = False
        # Phase E β-coop state (Task 16). When attach_mlp_fusion succeeds
        # AND CUTE_PHASE_E_FUSION=1 the β-coop kernel is constructed here;
        # dispatch chooses β-coop vs β-lite at forward() time via
        # _use_beta_coop predicate.
        self._phase_e_coop_kernel = None
        self._phase_e_use_beta_coop = False
        # β-coop region-timing scratch (Task 4 of beta-region-breakdown plan).
        # Allocated below in the same try-block as _phase_e_coop_wo_output
        # only when CUTE_BETA_REGION_TIMING=1; otherwise stays None and the
        # run_beta_coop_full kwarg defaults to None → timing-off compile path.
        self._phase_e_coop_region_timing: torch.Tensor | None = None
        # Tracks slice size of the most recent β-coop launch
        # (nat * slice_ctas * num_k_tiles). Used by the sentinel-file
        # dump to write only populated rows (not the max_num_seqs slab).
        # 0 means no β-coop launch has happened yet.
        self._phase_e_coop_region_timing_last_ctas: int = 0
        # --- C1.5: Phase F.1 layer-LN bake plumbing disabled. ----------------
        # Skip-flag + this-layer LN module ref were used by the opaque
        # cute_phase_e_skip_input_layernorm op (now no-op'd in _mlp_op.py).
        # Layer input_layernorm runs unconditionally at every layer entry
        # post-C1.5; per-step LN bake is gone. Kept commented in case the
        # skip-op pattern is needed for future Phase B/C kernel debugging.
        # self._phase_e_skip_next_ln = False
        # self._input_layernorm_module = None
        # ---------------------------------------------------------------------
        # β-lite (kept through C3) still reads these fields via getattr with
        # defaults; initialise here so the getattr lookups resolve cleanly
        # even after attach_next_input_layernorm is retired.
        self._next_input_layernorm_module = None
        self._emit_next_layernorm = False

    def _preallocate_fusion_buffers(
        self,
        max_num_seqs: int,
        hidden_dim: int,
        q_size: int,
        total_ctas_per_seq: int,
        device: str | torch.device,
    ) -> None:
        """Allocate persistent fusion I/O buffers.

        Called during __init__ (before forward) so allocations don't
        interfere with vLLM V1's pre-allocated memory pool.

        `total_ctas_per_seq` sizes the new per-CTA-slot staging axis in
        `wo_output` — each attention CTA writes its Phase-B partial into
        its own slot, and the last-arriving CTA per seq sums the slots
        in fixed (cta_idx) order before Phase C reads the collapsed
        row. This replaces the pre-2026-04-20 cross-CTA `atomicAdd_f32`
        into a 2-D buffer, whose arrival-order non-determinism flipped
        knife-edge argmax tokens on the Opus-distilled model (audit
        commit 16475223f).
        """
        # wo_output layout: [max_num_seqs, total_ctas_per_seq, hidden_dim].
        # Slot 0 holds the final summed value after the last-CTA gather
        # (Phase B.5 — see kernel.py); Phase C reads from slot 0.
        self.wo_output = torch.zeros(
            max_num_seqs, total_ctas_per_seq, hidden_dim,
            dtype=torch.float32, device=device,
        )
        self._fusion_total_ctas_per_seq = total_ctas_per_seq
        self.rmsnorm_output = torch.empty(
            max_num_seqs, hidden_dim, dtype=torch.bfloat16, device=device
        )
        self.residual_output = torch.empty(
            max_num_seqs, hidden_dim, dtype=torch.bfloat16, device=device
        )
        self.arrival_count = torch.zeros(max_num_seqs, dtype=torch.int32, device=device)
        self.gate_buf = torch.empty(
            max_num_seqs, q_size, dtype=torch.bfloat16, device=device
        )
        self.residual_buf = torch.empty(
            max_num_seqs, hidden_dim, dtype=torch.bfloat16, device=device
        )

        # Phase D MLP fusion buffers. Shape-defining axes (`slice_ctas`
        # for `mlp_partial_fp32`, `num_k_tiles` for `mlp_arrival_count`)
        # are both kernel-side constants resolved inside
        # `attach_mlp_fusion` once the `Phase_D_MLP_Kernel` instance is
        # constructed — so both are allocated lazily there, not here.
        self.mlp_output = torch.empty(
            max_num_seqs, hidden_dim, dtype=torch.bfloat16, device=device
        )
        # C1.5: next_hidden_scratch was previously allocated lazily inside
        # attach_next_input_layernorm. β-lite (kept through C3) still
        # references self.next_hidden_scratch[:nat] in its launch site,
        # so keep the allocation but hoist it here next to the other
        # persistent fusion buffers. Deletes after C3 with β-lite.
        self.next_hidden_scratch = torch.empty(
            max_num_seqs, hidden_dim, dtype=torch.bfloat16, device=device
        )
        self.mlp_partial_fp32 = None
        self.mlp_arrival_count = None

    def attach_fusion(self, parent_layer: torch.nn.Module) -> None:
        """Declare fusion intent. Called once per layer from the model
        `__init__` (see `vllm/nvllm/models/qwen3_5.py:Qwen3_5DecoderLayer`).

        Stores MODULE refs (not tensor refs) to o_proj and
        post_attention_layernorm — NVFP4's `process_weights_after_loading`
        REPLACES `weight_global_scale` with a new Parameter, so any tensor
        captured here would go stale (code-review C1).

        Pre-allocates persistent fusion buffers synchronously from sizes
        read off parent_layer. This replaces the old
        `get_current_vllm_config()` fallback that could silently defer
        allocation past CUDA-graph capture (code-review I1).
        """
        # MTP opt-out (spec "MTP handling"; code-review G3). MTP draft
        # layers run with different batch shapes, and the fused kernel's
        # layout assumptions aren't verified for the spec-decode path.
        prefix = getattr(parent_layer, "prefix", "")
        if "mtp" in prefix:
            logger.debug("CuTe fusion: skipping MTP layer %s", prefix or "<no-prefix>")
            return

        # Resolve sizes explicitly — no reliance on hf_config attr name.
        self_attn = parent_layer.self_attn
        q_size = self_attn.num_heads * self_attn.head_dim
        hidden_dim = self_attn.hidden_size
        num_q_heads = self_attn.num_heads
        num_kv_heads = self_attn.num_kv_heads
        group_size = max(num_q_heads // max(num_kv_heads, 1), 1)

        try:
            from vllm.config import get_current_vllm_config

            cfg = get_current_vllm_config()
            max_num_seqs = cfg.scheduler_config.max_num_seqs
        except Exception as e:
            logger.error(
                "CuTe fusion: attach_fusion cannot resolve max_num_seqs; "
                "fusion disabled for layer %s. Error: %s",
                prefix,
                e,
            )
            return

        # Total CTAs per seq = num_q_tiles * num_kv_heads. The attention
        # grid is (num_q_tiles, num_kv_heads, num_seqs); for our CuTe
        # paged backend both the decode (cta_q=16) and prefill (cta_q=64)
        # kernels launch with num_q_tiles = ceil(group_size / cta_q). The
        # decode kernel has the smaller cta_q, so it produces the larger
        # num_q_tiles; we use decode-cta_q=16 here as an upper-bound on
        # the per-CTA staging slot count. 2026-04-20 deterministic-
        # reduction fix — see audit commit 16475223f.
        _CTA_Q_DECODE = 16
        num_q_tiles_max = max(
            (group_size + _CTA_Q_DECODE - 1) // _CTA_Q_DECODE, 1
        )
        total_ctas_per_seq = num_q_tiles_max * max(num_kv_heads, 1)

        # Store module refs, NOT tensor refs.
        self._o_proj_module = self_attn.o_proj
        self._post_norm_module = parent_layer.post_attention_layernorm
        self._attn_output_gate = bool(self_attn.attn_output_gate)
        self._fusion_prefix = prefix
        self._fusion_max_num_seqs = max_num_seqs
        self._fusion_hidden_dim = hidden_dim
        self._fusion_q_size = q_size
        self._fusion_num_q_heads = num_q_heads
        self._fusion_num_kv_heads = num_kv_heads
        self._fusion_total_ctas_per_seq = total_ctas_per_seq

        # Allocate buffers ONCE. Subsequent attach calls (should not happen
        # under single-instantiation, but defensive) are no-ops for
        # buffer allocation so CUDA-graph pointers stay stable (H3).
        if not hasattr(self, "wo_output"):
            self._preallocate_fusion_buffers(
                max_num_seqs, hidden_dim, q_size,
                total_ctas_per_seq, "cuda",
            )

        self._fusion_attached = True
        logger.info(
            "CuTe fusion attached: layer=%s max_num_seqs=%d hidden_dim=%d "
            "q_size=%d attn_output_gate=%s",
            prefix,
            max_num_seqs,
            hidden_dim,
            q_size,
            self._attn_output_gate,
        )

    # ------------------------------------------------------------------ #
    # C1.5: attach_next_input_layernorm + attach_input_layernorm DISABLED #
    # ------------------------------------------------------------------ #
    # The Phase F.1 layer-LN bake (ε epilogue → next-layer input_LN baked
    # into previous layer's β-coop kernel) was removed in C1.5. Layer
    # input_layernorm now runs unconditionally at every layer entry; the
    # cross-layer module-attach scheme is no longer needed. The
    # `next_hidden_scratch` allocation moved into _preallocate_fusion_buffers
    # (above), `_resident_cap` probing moved into attach_mlp_fusion. β-lite
    # (kept through C3) reads `_next_input_layernorm_module` /
    # `_emit_next_layernorm` via getattr-with-defaults, which now resolve
    # to the inert __init__ defaults — β-lite's ε then takes the
    # last-layer (residual memcpy) branch.
    #
    # Methods kept commented in case the cross-layer pattern is needed
    # again during Phase B/C kernel debugging.
    #
    # def attach_next_input_layernorm(
    #     self, next_input_layernorm_module: torch.nn.Module | None
    # ) -> None:
    #     """Bind the next decoder layer's input_layernorm module for the
    #     Phase E β kernel's ε epilogue. Called from
    #     `Qwen3_5Model.__init__` post-hook (see vllm/nvllm/models/qwen3_5.py).
    #
    #     Pass `None` for the last decoder layer (index 63 in Qwen3.5-27B):
    #     ε epilogue then omits the next-layer norm and writes residual
    #     straight to residual_output. See spec §5.3.
    #
    #     Stores the MODULE ref (not the tensor) — NVFP4's
    #     process_weights_after_loading replaces Parameters, so a tensor
    #     captured here would go stale. Mirrors attach_fusion pattern.
    #     """
    #     assert getattr(self, '_fusion_attached', False), (
    #         "attach_next_input_layernorm: attach_fusion must run first"
    #     )
    #     self._next_input_layernorm_module = next_input_layernorm_module
    #     self._emit_next_layernorm = next_input_layernorm_module is not None
    #
    #     # Kill-switch: refuse to enable β if host memory is tight.
    #     # Runs BEFORE workspace allocation so a refusal doesn't leave
    #     # ~1.25 MiB of half-attached tensors on self.
    #     # CUTE_BETA_MIN_FREE_GB is plumbed via serve scripts
    #     # (commit 5c000a09d); default 8 GiB preserves KV + CUDA graph
    #     # headroom on GB10 (128 GiB unified).
    #     min_free_gb = float(os.environ.get("CUTE_BETA_MIN_FREE_GB", "8"))
    #     free_bytes, _total = torch.cuda.mem_get_info()
    #     if free_bytes < min_free_gb * (1024 ** 3):
    #         free_gb = free_bytes / (1024 ** 3)
    #         raise RuntimeError(
    #             f"CUTE_BETA_MIN_FREE_GB={min_free_gb} GiB threshold not met: "
    #             f"only {free_gb:.1f} GiB free. "
    #             f"Lower threshold or free memory before enabling Phase E."
    #         )
    #
    #     # Allocate Phase E workspace once. Qwen3_5Model.__init__ runs
    #     # inside vLLM's set_current_vllm_config() context, so
    #     # get_current_vllm_config() is always available on the real
    #     # serving path. Unit tests that bypass __init__ must stub this.
    #     if not hasattr(self, 'phase_e_barrier'):
    #         from vllm.config import get_current_vllm_config
    #         cfg = get_current_vllm_config()
    #         num_layers = cfg.model_config.hf_text_config.num_hidden_layers
    #         self.phase_e_barrier = torch.zeros(
    #             num_layers, dtype=torch.int32, device='cuda',
    #         )
    #         self.next_hidden_scratch = torch.empty(
    #             self._fusion_max_num_seqs, self._fusion_hidden_dim,
    #             dtype=torch.bfloat16, device='cuda',
    #         )
    #
    #     # Probe resident-CTA cap once (SMEM-bound estimate; a real
    #     # cuOccupancy probe re-runs post kernel compile in the launch
    #     # path). 45568 B matches the β kernel SMEM footprint (spec §6).
    #     self._resident_cap = self._probe_resident_cap(
    #         kernel_fn=None, num_threads=128, smem_bytes=45568
    #     )
    #     logger.info(
    #         "CuTe Phase E: resident_cap=%d (num_seqs_coop_max=%d)",
    #         self._resident_cap, max(1, self._resident_cap // 64)
    #     )
    #
    #     logger.info(
    #         "CuTe Phase E next-input-layernorm attached: "
    #         "emit_next_layernorm=%s num_layers_barrier=%d "
    #         "scratch_shape=(%d, %d)",
    #         self._emit_next_layernorm,
    #         self.phase_e_barrier.numel(),
    #         self._fusion_max_num_seqs,
    #         self._fusion_hidden_dim,
    #     )
    #
    # def attach_input_layernorm(
    #     self, input_layernorm_module: torch.nn.Module | None,
    # ) -> None:
    #     """Phase F.1: Attach THIS layer's input_layernorm module so the
    #     opaque cute_phase_e_skip_input_layernorm op can invoke it at
    #     call time (when the skip flag is unset).
    #
    #     Mirror of attach_next_input_layernorm; the two are paired.
    #     Stores the MODULE ref (not the tensor) — NVFP4's
    #     process_weights_after_loading replaces Parameters, so a tensor
    #     captured here would go stale.
    #     """
    #     self._input_layernorm_module = input_layernorm_module
    # ------------------------------------------------------------------ #

    def _probe_resident_cap(
        self, kernel_fn, num_threads: int, smem_bytes: int
    ) -> int:
        """Probe cooperative-launch cap. Returns occupancy_per_SM * num_sms.

        When kernel_fn is None (pre-compile), returns a conservative
        default based on SMEM occupancy alone: floor(smem_per_sm / smem_bytes).
        """
        num_sms = torch.cuda.get_device_properties(0).multi_processor_count
        if kernel_fn is None:
            # Pre-compile conservative fallback — SMEM-only
            smem_per_sm = torch.cuda.get_device_properties(
                0
            ).shared_memory_per_multiprocessor
            occ = max(1, smem_per_sm // max(smem_bytes, 1))
            return int(occ) * int(num_sms)
        # Real probe against compiled kernel
        try:
            from cuda.bindings import driver as cuda_driver
            ok, occ = cuda_driver.cuOccupancyMaxActiveBlocksPerMultiprocessor(
                kernel_fn, int(num_threads), int(smem_bytes)
            )
            if ok == cuda_driver.CUresult.CUDA_SUCCESS:
                return int(occ) * int(num_sms)
        except Exception as e:
            logger.warning(
                "CuTe Phase E: cuOccupancy probe failed (%s); "
                "using SMEM-only fallback", e
            )
        # Fallback to SMEM-only
        smem_per_sm = torch.cuda.get_device_properties(
            0
        ).shared_memory_per_multiprocessor
        occ = max(1, smem_per_sm // max(smem_bytes, 1))
        return int(occ) * int(num_sms)

    def _resolve_fusion_weights(self) -> None:
        """Bind current NVFP4 weight tensors off the stored o_proj / post_norm
        module refs. Called from `process_weights_after_loading` on EVERY
        invocation — supports live weight reload at
        `vllm/model_executor/model_loader/reload/layerwise.py:215-284`
        (code-review C2).

        No short-circuit on `_fusion_bound=True`. Overwrites strong refs so
        the next forward reads the NEW Parameter identity NVFP4 installed.
        """
        if not getattr(self, "_fusion_attached", False):
            # attach_fusion() was never called (MTP, BF16, non-full-attention,
            # or attach_fusion hit an early return).
            return

        o_proj = self._o_proj_module
        post_norm = self._post_norm_module

        # The "is this NVFP4?" gate — matches current behavior at
        # `vllm/model_executor/models/qwen3_next.py:484` (code-review H2).
        # A BF16 / FP8 serve lacks weight_global_scale — skip silently.
        if not hasattr(o_proj, "weight_global_scale"):
            logger.warning(
                "CuTe fusion: o_proj weights not NVFP4 (or not loaded) for "
                "layer %s; fusion disabled this call.",
                self._fusion_prefix,
            )
            self._fusion_bound = False
            return

        # Read tensor refs FRESH every call (code-review C1, C2).
        self.wo_weight = o_proj.weight
        self.wo_scales = o_proj.weight_scale
        self.wo_global_scale = o_proj.weight_global_scale
        self.rmsnorm_gamma = post_norm.weight
        self.rmsnorm_eps = post_norm.variance_epsilon

        # Phase D2 diagnostic: when CUTE_ATTN_FUSION=0, skip marking
        # attention as bound — `_fusion_active` will stay False, the
        # attention kernel takes its non-fused path, and Python handles
        # gate+o_proj + post_attention_layernorm for every step. Used to
        # isolate MLP-fusion correctness from attention-fusion state
        # handoff under CUDA-graph capture/replay.
        #
        # Default re-flipped 2026-04-20 back to "1" (fusion on): the
        # deterministic-reduction kernel fix (per-CTA slot + fixed-order
        # gather — see audit commit 16475223f and Phase B diff-vs-ref
        # evidence at benchmarks/nvllm/traces/phase_a_fused_reduction_fix/
        # 2026-04-20/debug_fusion/) replaces the cross-CTA atomicAdd_f32
        # that produced per-request non-determinism. Phase B kernel
        # output now matches a Python `attn @ W_O.T` reference to BF16
        # cast precision (1200/1200 close=True, diff max <= 0.0002).
        # Set CUTE_ATTN_FUSION=0 only for isolation diagnostics; the
        # entire fused codepath is intentionally kept inline (do not
        # delete) so future diag flips are a one-env-var toggle.
        if os.environ.get("CUTE_ATTN_FUSION", "1") != "1":
            logger.info(
                "CuTe fusion: CUTE_ATTN_FUSION=%s (set 0 to isolate "
                "attention; default 1 since 2026-04-20 after the "
                "deterministic-reduction kernel fix); "
                "layer=%s stays unbound (Python handles post-attn math).",
                os.environ.get("CUTE_ATTN_FUSION", "1"),
                self._fusion_prefix,
            )
            self._fusion_bound = False
            return

        self._fusion_bound = True
        logger.info(
            "CuTe fusion resolved: layer=%s wo_weight=%s rmsnorm_gamma=%s",
            self._fusion_prefix,
            list(self.wo_weight.shape),
            list(self.rmsnorm_gamma.shape),
        )

    def attach_mlp_fusion(
        self,
        mlp_module: torch.nn.Module,
        layer_name: str,
    ) -> None:
        """Declare MLP fusion intent (Phase D). Called from
        `Qwen3_5DecoderLayer.__init__` immediately after `attach_fusion(self)`.

        Resolves shape info from `gate_up_proj` + `down_proj`, instantiates
        `Phase_D_MLP_Kernel`, allocates `mlp_arrival_count` (sized by
        num_k_tiles), and wires Phase D2 custom-op dispatch: stashes
        module refs on `self` for the fallback unfused path (run inside
        the op body), registers `self` under `layer_name` in
        `_CUTE_MLP_REGISTRY`, and sets `mlp_module._cute_layer_name =
        layer_name` so `Qwen3_5MLP.forward` routes through
        `torch.ops.vllm.cute_mlp_forward`. The op body itself owns the
        per-step gate and kernel launch (see `_mlp_op.py`).

        `layer_name` (e.g. `"model.layers.N.mlp"`) must be unique per MLP
        instance — supplied by the decoder at the call site.

        MTP opt-out mirrors `attach_fusion` — uses the already-set
        `_fusion_prefix` for the MTP check since Qwen3_5MLP has no prefix.
        """
        from vllm.v1.attention.backends.cute_paged.mlp_kernel import (
            Phase_D_MLP_Kernel,
        )

        # Phase D remains opt-in while kernel perf is still being tuned.
        # Phase D2 resolves the dual-firing issue (launch + fallback now
        # live inside the opaque op body; see _mlp_op.py), but the kernel
        # itself is still tiled for prefill-sized work and over-launches
        # for small decode batches. Keep default OFF until Phase D2 kernel
        # tuning closes the perf gap; set CUTE_MLP_FUSION=1 to opt in.
        if os.environ.get("CUTE_MLP_FUSION", "0") != "1":
            logger.info(
                "CuTe MLP fusion: disabled (default); set CUTE_MLP_FUSION=1 "
                "to opt in (kernel tuning in progress; see Phase D2 trace "
                "summary)."
            )
            return

        # MTP opt-out — same pattern as attach_fusion (code-review G3)
        prefix = getattr(self, "_fusion_prefix", "")
        if "mtp" in prefix:
            logger.debug("CuTe MLP fusion: skipping MTP layer %s", prefix)
            return

        # attach_fusion must have run first (buffers allocated, max_num_seqs set)
        if not getattr(self, "_fusion_attached", False):
            logger.warning(
                "CuTe MLP fusion: attach_fusion() must be called before "
                "attach_mlp_fusion(); skipping."
            )
            return

        self._mlp_module = mlp_module

        # Resolve shapes from gate_up_proj / down_proj. The fused MLP kernel
        # consumes gate and up as SEPARATE weight tensors; we split the
        # MergedColumnParallelLinear stacked weight at _resolve_mlp_weights().
        gate_up = mlp_module.gate_up_proj
        down = mlp_module.down_proj

        # Output per-TP-partition surfaces vary across linear classes; try the
        # per-partition attr first, fall back to the unqualified one.
        hidden_size = getattr(
            down, "output_size_per_partition", getattr(down, "output_size", None)
        )
        intermediate_size = getattr(
            down, "input_size_per_partition", getattr(down, "input_size", None)
        )
        if hidden_size is None or intermediate_size is None:
            logger.warning(
                "CuTe MLP fusion: could not resolve hidden/intermediate sizes "
                "from down_proj=%s; skipping.",
                type(down).__name__,
            )
            return

        # Phase D3a: tile values come from _tile_presets (sibling module).
        # The presets and resolver live OUTSIDE mlp_kernel.py because adding
        # runtime Python to that file perturbs CuTe DSL JIT compilation
        # enough to break FP4 decode numerics. See _tile_presets.py module
        # docstring for the investigation evidence.
        from vllm.v1.attention.backends.cute_paged._tile_presets import (
            resolve_tile_preset_from_env,
        )
        tile_s, tile_k, slice_ctas = resolve_tile_preset_from_env()

        try:
            self._mlp_kernel = Phase_D_MLP_Kernel(
                hidden_size=hidden_size,
                intermediate_size=intermediate_size,
                tile_s=tile_s,
                tile_k=tile_k,
                slice_ctas=slice_ctas,
            )
        except (AssertionError, ValueError) as e:
            logger.warning(
                "CuTe MLP fusion: kernel shape mismatch "
                "(hidden=%d, intermediate=%d, tile_s=%d, tile_k=%d, "
                "slice_ctas=%d): %s. Fusion disabled.",
                hidden_size, intermediate_size, tile_s, tile_k, slice_ctas, e,
            )
            return

        self._mlp_num_k_tiles = self._mlp_kernel.num_k_tiles
        self._mlp_slice_ctas = self._mlp_kernel.slice_ctas

        # Allocate fusion buffers now that the kernel's shape-defining
        # axes are known. 3-D `mlp_partial_fp32` layout [max_num_seqs,
        # slice_ctas, hidden_dim]: each slice-CTA writes into its own
        # `bx` slot (2026-04-20 deterministic-reduction fix — see
        # audit commit 16475223f). 2-D arrival counter layout
        # [max_num_seqs, num_k_tiles]; kernel consumes via data_ptr so
        # the byte layout is (`token_idx * num_k_tiles + k_tile_id`).
        max_num_seqs = self._fusion_max_num_seqs
        hidden_dim = self._fusion_hidden_dim
        self.mlp_partial_fp32 = torch.zeros(
            max_num_seqs, self._mlp_slice_ctas, hidden_dim,
            dtype=torch.float32, device="cuda",
        )
        self.mlp_arrival_count = torch.zeros(
            max_num_seqs, self._mlp_num_k_tiles,
            dtype=torch.int32, device="cuda",
        )

        # Phase E β-coop kernel (Task 16). Gated on CUTE_PHASE_E_FUSION=1
        # so the env flag disables it without any attach-time work. Phase 0
        # of the unified kernel consumes a (hidden_in, residual_in,
        # input_gamma) triple that the external model code doesn't pass
        # through to the attn backend — Phase 0's output (attn_input_bf16)
        # is a side-channel for a future QKV-fusion step and isn't consumed
        # by the current layer's attn path. We allocate throwaway scratch
        # buffers and a dummy input_gamma=ones so Phase 0 runs harmlessly;
        # the real work happens in Phases 1-4. Future: hoist input_LN into
        # the kernel by having this layer's Phase 0 write the next layer's
        # pre-QKV input (requires model-side plumbing).
        if os.environ.get("CUTE_PHASE_E_FUSION", "0") == "1":
            try:
                from vllm.v1.attention.backends.cute_paged.phase_e_kernel import (  # noqa: E501
                    PhaseE_Beta_Kernel,
                )
                self._phase_e_coop_kernel = PhaseE_Beta_Kernel(
                    hidden_size=hidden_size,
                    intermediate_size=intermediate_size,
                    num_attn_heads=self.num_heads,
                    num_kv_heads=self.num_kv_heads,
                    head_dim=self.head_size,
                    rms_eps=1e-6,  # overridden per-call from next-LN module
                    tile_s=tile_s,
                    tile_k=tile_k,
                    slice_ctas=slice_ctas,
                )
                # Phase 0 throwaway buffers (output ignored; inputs dummied).
                self._phase_e_coop_attn_input_scratch = torch.empty(
                    max_num_seqs, hidden_dim,
                    dtype=torch.bfloat16, device="cuda",
                )
                self._phase_e_coop_input_gamma = torch.ones(
                    hidden_dim, dtype=torch.bfloat16, device="cuda",
                )
                # Persistent β-coop workspace buffers (spec 2026-04-30
                # §4.1, §4.2). Hoisted from per-call torch.zeros inside
                # run_beta_coop_full to fix the FULL graph capture
                # replay-divergence bug (vLLM #35175 analog). Allocated
                # inside this `try:` so an OOM trips the except handler
                # that nulls _phase_e_coop_kernel.
                self._phase_e_coop_wo_output = torch.zeros(
                    max_num_seqs,
                    self.num_kv_heads * self._phase_e_coop_kernel.wo_split,
                    hidden_dim,
                    dtype=torch.float32, device="cuda",
                )
                self._phase_e_coop_mlp_partial_fp32 = torch.zeros(
                    max_num_seqs, self._mlp_slice_ctas, hidden_dim,
                    dtype=torch.float32, device="cuda",
                )
                self._phase_e_coop_mlp_arrival_count = torch.zeros(
                    max_num_seqs, self._mlp_num_k_tiles,
                    dtype=torch.uint32, device="cuda",
                )
                self._phase_e_coop_grid_barrier_i32 = torch.zeros(
                    max_num_seqs, dtype=torch.int32, device="cuda",
                )
                self._phase_e_coop_phase1_arrival_count = torch.zeros(
                    max_num_seqs, dtype=torch.int32, device="cuda",
                )
                # Task 6: pre-W_O arrival counter — producers (bx==0 attn
                # CTAs) atomic_add 1 after attn_output is written;
                # consumers (bx>0 W_O CTAs, only at wo_split>1) spin-wait
                # until the counter reaches num_kv_heads. At wo_split=1 the
                # consumer mask is empty and no CTA reads this counter, so
                # the increment to num_kv_heads is harmless. Reset by host
                # zero_() before each launch (Task 6 chose host-zero_
                # approach over kernel atomic-subtract for symmetry with
                # mlp_arrival_count.zero_() that already runs at every
                # launch).
                self._phase_e_coop_pre_wo_arrival_count = torch.zeros(
                    max_num_seqs, dtype=torch.int32, device="cuda",
                )
                if _REGION_TIMING_ENABLED:
                    # Per-CTA region timing scratch. Layout:
                    #   (num_ctas, num_regions, 2) u64 — entry+exit ticks.
                    # See vllm/v1/attention/backends/cute_paged/region_timing.py
                    # for reducer + region taxonomy. Last-launch only for v1
                    # (caller does not increment a sample index; each launch
                    # overwrites the previous launch's data).
                    num_ctas = (
                        self._phase_e_coop_kernel.slice_ctas
                        * self._phase_e_coop_kernel.num_k_tiles
                        * max_num_seqs
                    )
                    self._phase_e_coop_region_timing = torch.zeros(
                        num_ctas, _REGION_TIMING_NUM_REGIONS, 2,
                        dtype=torch.int64, device="cuda",
                    )
                    logger.warning(
                        "[β-coop region timing] enabled; scratch shape=%s "
                        "(%d bytes). Last-launch-only.",
                        tuple(self._phase_e_coop_region_timing.shape),
                        self._phase_e_coop_region_timing.numel() * 8,
                    )
                else:
                    self._phase_e_coop_region_timing = None
                # C1.5: probe resident-CTA cap here (was previously inside
                # attach_next_input_layernorm). 45568 B matches the β kernel
                # SMEM footprint (spec §6). Read by the β-coop dispatch
                # predicate at L1150.
                self._resident_cap = self._probe_resident_cap(
                    kernel_fn=None, num_threads=128, smem_bytes=45568
                )
                logger.info(
                    "CuTe Phase E β-coop kernel attached: hidden=%d "
                    "intermediate=%d num_q_heads=%d num_kv_heads=%d "
                    "head_dim=%d resident_cap=%d",
                    hidden_size, intermediate_size,
                    self.num_heads, self.num_kv_heads, self.head_size,
                    self._resident_cap,
                )
            except Exception as e:  # noqa: BLE001 — fail-closed
                logger.warning(
                    "CuTe Phase E β-coop kernel construction failed: "
                    "%s. β-coop dispatch disabled; β-lite still available.",
                    e,
                )
                self._phase_e_coop_kernel = None

        # Phase D (pre-D1): exposed impl directly on the module; Qwen3_5MLP.forward
        # did `getattr(self, "_cute_impl", None)` + per-step `_mlp_fusion_active`
        # check, which torch.compile dead-branched at trace time. Left commented
        # for reference; the Phase D1 custom-op path below replaces it.
        # mlp_module._cute_impl = self

        # Phase D1: stash unfused-path module refs on self for the fallback
        # branch inside `cute_mlp_forward` (prefill batches, fail-closed
        # fusion). These are plain torch.nn.Modules; calling them inside an
        # opaque custom op is safe — inner ops run eagerly, nothing is traced.
        self._mlp_gate_up_proj = mlp_module.gate_up_proj
        self._mlp_act_fn = mlp_module.act_fn
        self._mlp_down_proj = mlp_module.down_proj

        # Phase D1: register impl under the decoder-supplied layer name so
        # the custom op's runtime body can look it up. Import inside the
        # method so the _mlp_op module's direct_register_custom_op side
        # effect runs exactly once, at first attach.
        from vllm.v1.attention.backends.cute_paged._mlp_op import (
            _CUTE_MLP_REGISTRY,
        )
        _CUTE_MLP_REGISTRY[layer_name] = self
        mlp_module._cute_layer_name = layer_name

        self._mlp_attached = True
        logger.info(
            "CuTe MLP fusion attached: layer=%s hidden=%d interm=%d "
            "num_k_tiles=%d tile_s=%d tile_k=%d slice_ctas=%d op_key=%s",
            prefix, hidden_size, intermediate_size, self._mlp_num_k_tiles,
            self._mlp_kernel.tile_s, self._mlp_kernel.tile_k,
            self._mlp_kernel.slice_ctas, layer_name,
        )

    def _resolve_mlp_weights(self) -> None:
        """Bind current NVFP4 gate_up + down weight tensors off the stored
        MLP module ref. Called from `process_weights_after_loading` alongside
        `_resolve_fusion_weights`. Splits the MergedColumnParallelLinear
        stacked gate_up weight into separate gate/up tensors for the kernel.
        """
        if not getattr(self, "_mlp_attached", False):
            return

        mlp = self._mlp_module
        gate_up = mlp.gate_up_proj
        down = mlp.down_proj

        # NVFP4 gate: if weight_global_scale missing, model is BF16/FP8 —
        # disable fusion this call. Mirrors _resolve_fusion_weights pattern.
        if not hasattr(gate_up, "weight_global_scale"):
            logger.warning(
                "CuTe MLP fusion: gate_up_proj weights not NVFP4 (or not "
                "loaded); MLP fusion disabled this call."
            )
            self._mlp_fusion_bound = False
            self._beta_coop_framework_output_bound = False
            return
        if not hasattr(down, "weight_global_scale"):
            logger.warning(
                "CuTe MLP fusion: down_proj weights not NVFP4; disabled."
            )
            self._mlp_fusion_bound = False
            self._beta_coop_framework_output_bound = False
            return

        # Split stacked [2*interm, hidden/2] FP4 weights into separate
        # gate and up views. MergedColumnParallelLinear stacks gate in rows
        # [0, interm) and up in rows [interm, 2*interm) on the first axis.
        interm = self._mlp_kernel.intermediate_size
        gate_up_w = gate_up.weight
        gate_up_s = gate_up.weight_scale

        if gate_up_w.shape[0] != 2 * interm:
            logger.warning(
                "CuTe MLP fusion: gate_up_proj first dim = %d != 2*interm=%d; "
                "disabled.",
                gate_up_w.shape[0], 2 * interm,
            )
            self._mlp_fusion_bound = False
            self._beta_coop_framework_output_bound = False
            return

        self._mlp_gate_w = gate_up_w[:interm]
        self._mlp_up_w = gate_up_w[interm : 2 * interm]
        # NVFP4 scales are stored as torch.float8_e4m3fn; the kernel
        # consumes them via byte pointers (UE4M3 interpretation happens
        # inside the kernel), so reinterpret as uint8 to satisfy the
        # kernel-side dtype assertion. Same memory, no copy.
        self._mlp_gate_s = gate_up_s[:interm].view(torch.uint8)
        self._mlp_up_s = gate_up_s[interm : 2 * interm].view(torch.uint8)
        self._mlp_down_w = down.weight
        self._mlp_down_s = down.weight_scale.view(torch.uint8)
        # NVFP4 dequant = fp4 × block_scale × weight_global_scale.
        # `.item()` sync happens once at attach; forwards pass the
        # cached Python floats (no per-step device sync). gate and up
        # share one scale via MergedColumnParallelLinear.
        self._mlp_gate_up_gs = float(
            gate_up.weight_global_scale.to(torch.float32).item()
        )
        self._mlp_down_gs = float(
            down.weight_global_scale.to(torch.float32).item()
        )

        self._mlp_fusion_bound = True
        # All three β-coop framework-output prerequisites resolved.
        # This is the stable post-weight-load flag the decoder layer
        # branches on.
        # PHASE 3 DIAG (2026-04-27): env override `CUTE_PHASE3_DIAG_DISABLE_FW=1`
        # forces framework-output route off so we can test the LEGACY
        # paged+β-lite path with CUTE_PHASE_E_FUSION=1 without changing
        # any other behavior. Apples-to-apples comparison vs framework-output.
        if os.environ.get("CUTE_PHASE3_DIAG_DISABLE_FW") == "1":
            self._beta_coop_framework_output_bound = False
            logger.warning(
                "[PHASE3_DIAG] framework-output route DISABLED via "
                "CUTE_PHASE3_DIAG_DISABLE_FW=1 (legacy paged+β-lite test path)"
            )
        else:
            self._beta_coop_framework_output_bound = (
                self._fusion_bound
                and self._mlp_fusion_bound
                and getattr(self, "_phase_e_coop_kernel", None) is not None
            )
        logger.debug(
            "CuTe MLP fusion resolved: gate_w=%s up_w=%s down_w=%s",
            list(self._mlp_gate_w.shape),
            list(self._mlp_up_w.shape),
            list(self._mlp_down_w.shape),
        )

    # --- DISABLED 2026-04-17 (Phase B own-the-stack refactor) ---
    # Replaced by `attach_fusion(parent_layer)` + `_resolve_fusion_weights()`.
    # Kept commented (not deleted) until Tier-3 GSM8K 8/8 validates the new
    # path. Remove in a follow-up commit once the refactor is proven.
    # --- DISABLED block start ---
    # def bind_fusion_weights(
    #     self,
    #     wo_weight: torch.Tensor,
    #     wo_scales: torch.Tensor,
    #     wo_global_scale: torch.Tensor,
    #     rmsnorm_gamma: torch.Tensor,
    #     rmsnorm_eps: float,
    #     max_num_seqs: int,
    # ) -> None:
    #     """Bind static fusion weights and allocate persistent I/O buffers."""
    #     self.wo_weight = wo_weight
    #     self.wo_scales = wo_scales
    #     self.wo_global_scale = wo_global_scale
    #     self.rmsnorm_gamma = rmsnorm_gamma
    #     self.rmsnorm_eps = rmsnorm_eps
    #     hidden_dim = rmsnorm_gamma.shape[0]
    #     q_size = self.num_heads * self.head_size
    #     if not hasattr(self, "wo_output"):
    #         self._preallocate_fusion_buffers(
    #             max_num_seqs, hidden_dim, q_size, wo_weight.device
    #         )
    #     self._fusion_bound = True
    #     logger.info(
    #         "CuTe fusion bound: hidden_dim=%d, q_size=%d, max_seqs=%d, "
    #         "wo_weight=%s, rmsnorm_gamma=%s",
    #         hidden_dim, q_size, max_num_seqs,
    #         list(wo_weight.shape), list(rmsnorm_gamma.shape),
    #     )
    # --- DISABLED block end ---

    def forward(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: CutePagedMetadata,
        output: torch.Tensor | None = None,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
        # NEW (Phase 3): framework-output-buffer route. When all three
        # are non-None, β-coop / fall-through writes through these.
        residual: torch.Tensor | None = None,
        attn_input: torch.Tensor | None = None,
        gate: torch.Tensor | None = None,
        output_rmsnorm: torch.Tensor | None = None,
        output_residual: torch.Tensor | None = None,
        output_mlp: torch.Tensor | None = None,
    ) -> torch.Tensor:
        assert output is not None, "Output tensor must be provided"

        if attn_metadata is None:
            return output.fill_(0)

        # Phase 3 framework-output route is active when ALL six new kwargs
        # are non-None (caller is the new cute_beta_coop_run op).
        _framework_output_route = (
            residual is not None and attn_input is not None and gate is not None
            and output_rmsnorm is not None and output_residual is not None
            and output_mlp is not None
        )
        if _framework_output_route and _VERIFY_FRAMEWORK_OUTPUTS:
            # Phase 6a: gated behind CUTE_VERIFY_FW=1. The asserts and
            # phase3 diagnostic ran every full-attn forward × every token
            # in production despite never firing on a healthy framework
            # route. Set CUTE_VERIFY_FW=1 to re-enable for debugging.
            assert output_rmsnorm.shape == output_residual.shape == output_mlp.shape == residual.shape, (
                f"Framework output shape mismatch: rmsnorm={output_rmsnorm.shape} "
                f"residual={output_residual.shape} mlp={output_mlp.shape} "
                f"input residual={residual.shape}"
            )
            assert output_rmsnorm.dtype == torch.bfloat16
            # Contiguity asserts — paged kernel uses raw data_ptr() arithmetic
            # and assumes contiguous BF16. Non-contiguous tensors silently
            # read/write wrong memory.
            assert residual.is_contiguous(), (
                f"residual not contiguous: stride={residual.stride()} shape={residual.shape}"
            )
            assert output_rmsnorm.is_contiguous(), "output_rmsnorm not contiguous"
            assert output_residual.is_contiguous(), "output_residual not contiguous"
            assert output_mlp.is_contiguous(), "output_mlp not contiguous"
            assert gate.is_contiguous(), (
                f"gate not contiguous: stride={gate.stride()} shape={gate.shape}"
            )
            # One-shot diagnostic: dump first 4 elements of each input/output
            # at the FIRST forward call only (per impl). Reveals stale data,
            # NaN, or value mismatches vs the legacy self.X buffers.
            if not getattr(self, "_phase3_diag_done", False):
                self._phase3_diag_done = True
                _layer_name_dbg = getattr(layer, "layer_name", "<unknown>")
                logger.warning(
                    "[PHASE3_DIAG] layer=%s nat=%d residual[0,:4]=%s "
                    "gate[0,:4]=%s self.residual_buf[0,:4]=%s "
                    "self.gate_buf[0,:4]=%s",
                    _layer_name_dbg,
                    attn_metadata.num_actual_tokens,
                    residual[0, :4].float().tolist(),
                    gate[0, :4].float().tolist(),
                    self.residual_buf[0, :4].float().tolist(),
                    self.gate_buf[0, :4].float().tolist() if hasattr(self, "gate_buf") else None,
                )

        k_scale = getattr(layer, "_k_scale_float", 1.0)
        v_scale = getattr(layer, "_v_scale_float", 1.0)

        # Per-forward gating lives entirely inside impl (spec § Per-forward
        # gating). Fusion activates only for decode batches whose
        # num_actual_tokens fits the pre-allocated buffers — prevents
        # out-of-range writes if an unusually large decode batch arrives
        # (code-review A3).
        num_actual_tokens = attn_metadata.num_actual_tokens
        is_decode_only = getattr(attn_metadata, "is_decode_only", False)
        fits_buffer = num_actual_tokens <= getattr(self, "_fusion_max_num_seqs", 0)
        self._fusion_active = self._fusion_bound and is_decode_only and fits_buffer
        use_fusion = self._fusion_active
        # --- PHASE D2 DISABLED (commented, not deleted — Phase B/C debug may
        # need this reset back) ---
        # Pre-D2, the MLP fusion launch was an attention-side side effect
        # (see the disabled block after the attention kernel below), and
        # this line reset the per-step flag. Phase D2 moves the launch
        # into `torch.ops.vllm.cute_mlp_forward`, so no reset is needed.
        # self._mlp_fusion_active = False
        # --- END PHASE D2 DISABLED ---
        if _DEBUG_FUSION:
            logger.info(
                "[CUTE_DEBUG_FUSION] layer=%s bound=%s active=%s use_fusion=%s",
                getattr(layer, "layer_name", "<layer>"),
                self._fusion_bound,
                self._fusion_active,
                use_fusion,
            )
        wo_weight = self.wo_weight if use_fusion else None
        wo_scales = self.wo_scales if use_fusion else None
        wo_global_scale = self.wo_global_scale if use_fusion else None
        wo_output = self.wo_output if use_fusion else None
        rmsnorm_gamma = self.rmsnorm_gamma if use_fusion else None
        rmsnorm_residual = self.residual_buf if use_fusion else None
        rmsnorm_output = self.rmsnorm_output if use_fusion else None
        residual_output = self.residual_output if use_fusion else None
        arrival_count = self.arrival_count if use_fusion else None
        rmsnorm_eps = self.rmsnorm_eps if use_fusion else None
        gate_buf = self.gate_buf if use_fusion else None

        # Zero accumulation buffers before kernel launch.
        # Python-side zero_() is ordered before the kernel by CUDA
        # stream semantics; self-zero inside the kernel races across
        # CTAs that all launch concurrently.
        # 2026-04-20 deterministic-reduction fix: wo_output is now
        # shape [max_num_seqs, total_ctas_per_seq, hidden_dim]. Each
        # CTA writes its Phase-B partial into its own `cta_idx` slot
        # (plain FP32 store). Zero-init keeps unused slots (beyond the
        # runtime total_ctas for this launch) at 0 so the Phase B.5
        # gather can sum up to the allocation bound without reading
        # garbage. arrival_count still tracks "all CTAs arrived" for
        # the last-CTA gather gate (audit commit 16475223f).
        if use_fusion:
            self.wo_output.zero_()
            self.arrival_count.zero_()

        from vllm.v1.attention.backends.cute_paged.kernel import (
            paged_attention_forward,
        )

        # For graph-safe dispatch: padded batch size for grid.z
        num_seqs = len(attn_metadata.seq_lens)
        padded_num_seqs = num_seqs  # graph capture overrides via metadata

        # PHASE 4 EDIT 3 (2026-04-28): hoisted unified β-coop predicate +
        # collapsed skip rule. Replaces the prior parallel pair
        # (_will_fire_beta_coop_pre / _use_beta_coop) with one computation.
        # The duplicate computation block at the original site (~L1261-1340)
        # is commented out below. New invariant:
        #   _skip_paged = _use_beta_coop and not _framework_output_route
        # When framework-output route is active, paged ALWAYS runs to
        # guarantee writer-invariant on output_rmsnorm/output_residual:
        #   - β-coop except → β-lite needs filled buffers (else stale)
        #   - _layer_allowed=False → no β fires (else stale)
        #   - prefill / oversize → β disabled (handled in Edit 4 fallback)
        # PHASE 4 EDIT 5 (re-flipped 2026-04-28 PM after KV-update fix).
        # Bisect history: initial flip to False produced "rome?" gibberish
        # (root cause: KV-update DCE in qwen3_5.py:328, NOT β-coop).
        # Friend's KV canonical-dispatch fix landed at qwen3_5.py:328
        # restored Phase 3 "Paris" baseline with route always-ON. Now
        # re-enabling β-coop to validate the full Phase 4 stack.
        # Rollback path (proven coherent under PIECEWISE+graphs):
        # _phase3_force_fallthrough = True
        _phase3_force_fallthrough = False
        # Per-step Phase E flag resets (was at original L1261-1262 + new
        # _phase_e_use_beta_coop reset per friend's audit).
        self._phase_e_consumed = False
        self._phase_e_use_beta_coop = False
        self._phase_e_use_beta_lite = False
        _phase_e_env = _PHASE_E_ENV
        # `layer.layer_name` is the canonical identifier; layer_idx isn't
        # populated, so extract via dotted-name helper.
        _layer_idx: int | None = None
        _layer_name = getattr(layer, "layer_name", None)
        if _layer_name is not None:
            try:
                from vllm.model_executor.models.utils import extract_layer_index
                _layer_idx = extract_layer_index(_layer_name)
            except Exception:
                _layer_idx = None
        # C1.5: live "Phase E plumbing wired" sentinel — either kernel
        # attached implies the path can run.
        _phase_e_attached = (
            getattr(self, "_phase_e_coop_kernel", None) is not None
            or getattr(self, "_mlp_fusion_bound", False)
        )
        _layer_allowed = (
            _phase_e_env.restricted_layers is None
            or (_layer_idx is not None
                and _layer_idx in _phase_e_env.restricted_layers)
        )
        # INVARIANT: β-lite reads `self.residual_output` below, populated by
        # the attention uber-kernel only when use_fusion=True. Keep
        # `use_fusion` in this AND.
        _phase_e_active = (
            _phase_e_env.enabled
            and is_decode_only
            and _phase_e_attached
            and _layer_allowed
            and use_fusion  # INVARIANT — do not remove
            and getattr(self, "_mlp_fusion_bound", False)
            and num_actual_tokens <= getattr(self, "_fusion_max_num_seqs", 0)
        )
        # 64 = CTAs-per-seq in the attn grid (num_q_tiles=1 × num_kv_heads=4
        # × slice_ctas=8 × num_k_tiles=8 = 64 per seq).
        _CTAS_PER_SEQ = 64
        _total_ctas = _CTAS_PER_SEQ * num_seqs
        _resident_cap = getattr(self, "_resident_cap", 0)
        _coop_attached = getattr(self, "_phase_e_coop_kernel", None) is not None
        # 2026-04-26: cooperative-launch fitness (64*num_seqs <= _resident_cap)
        # is a HARD gate even in forced-coop mode. Previously bypassed when
        # forced_path=="coop", causing CUDA_ERROR_COOPERATIVE_LAUNCH_TOO_LARGE
        # on multi-seq decode and silent gibberish.
        _use_beta_coop = (
            not _phase3_force_fallthrough
            and _phase_e_active
            and _coop_attached
            and _total_ctas <= _resident_cap
            and _phase_e_env.forced_path in ("coop", "auto")
        )
        _use_beta_lite = (
            _phase_e_active
            and not _use_beta_coop
            and (
                _phase_e_env.forced_path == "lite"
                or _phase_e_env.forced_path == "auto"
            )
        )
        # PHASE 5 EDIT (2026-04-28): drop `and not _framework_output_route`
        # from skip rule. Paged is now skipped whenever β-coop will fire,
        # regardless of route. The β-coop except handler below explicitly
        # re-runs paged into framework buffers BEFORE β-lite fallback so
        # the writer-invariant still holds when β-coop raises.
        # Phase 4 (commented for rollback):
        # _skip_paged = _use_beta_coop and not _framework_output_route
        _skip_paged = _use_beta_coop
        # PHASE 3 ORIGINAL `_will_fire_beta_coop_pre` (commented per
        # feedback_comment_not_delete — may be re-enabled if the
        # framework-output writer-invariant changes):
        # _phase_e_env_pre = _phase_e_env_config()
        # _will_fire_beta_coop_pre = (
        #     not _phase3_force_fallthrough
        #     and _phase_e_env_pre.enabled
        #     and is_decode_only
        #     and use_fusion
        #     and getattr(self, "_phase_e_coop_kernel", None) is not None
        #     and getattr(self, "_mlp_fusion_bound", False)
        #     and num_actual_tokens <= getattr(self, "_fusion_max_num_seqs", 0)
        #     and (64 * num_seqs) <= getattr(self, "_resident_cap", 0)
        #     and _phase_e_env_pre.forced_path in ("coop", "auto")
        # )

        # PHASE 5 EDIT (2026-04-28): paged call extracted to a closure so it
        # can be invoked from BOTH the normal path AND the β-coop except
        # handler (writer-invariant on β-coop kernel failure when route
        # active). Closure captures forward-local kwargs.
        def _run_paged() -> torch.Tensor:
            paged_rmsnorm_output = (
                output_rmsnorm if _framework_output_route else rmsnorm_output
            )
            paged_residual_output = (
                output_residual if _framework_output_route else residual_output
            )
            paged_rmsnorm_residual = (
                residual if _framework_output_route else rmsnorm_residual
            )
            paged_gate_buf = (
                gate if _framework_output_route else gate_buf
            )
            return paged_attention_forward(
                query=query[:num_actual_tokens],
                kv_cache=kv_cache,
                page_table=attn_metadata.block_table,
                seq_lens=attn_metadata.seq_lens,
                scale=self.scale,
                k_scale=k_scale,
                v_scale=v_scale,
                page_size=64,
                query_start_loc=attn_metadata.query_start_loc,
                wo_weight=wo_weight,
                wo_scales=wo_scales,
                wo_global_scale=wo_global_scale,
                wo_output=wo_output,
                rmsnorm_gamma=rmsnorm_gamma,
                rmsnorm_residual=paged_rmsnorm_residual,
                rmsnorm_output=paged_rmsnorm_output,
                residual_output=paged_residual_output,
                arrival_count=arrival_count,
                rmsnorm_eps=rmsnorm_eps,
                gate_buf=paged_gate_buf,
                padded_num_seqs=padded_num_seqs,
            )

        if _skip_paged:
            result = None
            # Mark snapshots stale so the BETA_DIFF harness skips below.
            self._debug_paged_res = None
        else:
            result = _run_paged()

            # --- BETA_DIFF harness: snapshot paged's outputs so we can diff
            # against β-coop's overwrite later. Gated on CUTE_DEBUG_FUSION=1.
            # Only fires when paged actually ran (else clause).
            # See memory:project_beta_coop_residual_solo_bug for protocol.
            if _DEBUG_FUSION and use_fusion and is_decode_only:
                self._debug_paged_wo = self.wo_output.detach().clone()
                self._debug_paged_rms = self.rmsnorm_output.detach().clone()
                self._debug_paged_res = self.residual_output.detach().clone()

        # --- DEBUG: fusion diagnostic (CUTE_DEBUG_FUSION=1) ---
        # Compares kernel's impl.wo_output (Phase B GEMV) against a Python
        # reference computed from the kernel's own Phase A output (`result`)
        # and a one-time-dequantized W_O. Proves whether Phase B is faithful.
        # Skip when paged was gated off (result is None).
        if _DEBUG_FUSION and use_fusion and result is not None:
            self._debug_fusion_diff(
                result=result,
                num_actual_tokens=num_actual_tokens,
                layer_name=getattr(layer, "layer_name", "<layer>"),
            )
        # --- END DEBUG ---

        # --- PHASE E β-lite dispatch (task 9) ---------------------------
        # When CUTE_PHASE_E_FUSION=1 AND (forced_path=lite OR auto-chose
        # lite because 64*num_seqs > resident_cap) AND next-input-layernorm
        # was attached, launch Phase_D_MLP_Kernel with emit_epilogue=True
        # so the MLP kernel's ε epilogue does residual_add + next RMSNorm
        # in-kernel. Sets `self._phase_e_consumed=True` so the Python
        # decoder wrapper in `qwen3_5.py` skips its post-MLP residual
        # copy_ and next_input_layernorm pass (both done in-kernel).
        #
        # Default OFF: when CUTE_PHASE_E_FUSION=0 (the default), this
        # entire branch is bypassed and `_phase_e_consumed` stays False —
        # the legacy Phase D path (via `torch.ops.vllm.cute_mlp_forward`)
        # runs unchanged.
        #
        # Task 10 will verify this end-to-end with GSM8K under Docker.
        # Task 9's deliverable is the dispatch path + source-level test.
        # PHASE 4 EDIT 3 (2026-04-28): predicate computation MOVED UP to the
        # paged-skip site (see ~L1162). The originals below are commented
        # out per feedback_comment_not_delete; they are referenced only by
        # the dispatch branches that follow. All variables (_phase_e_env,
        # _layer_idx, _layer_name, _phase_e_attached, _layer_allowed,
        # _phase_e_active, _CTAS_PER_SEQ, _total_ctas, _resident_cap,
        # _coop_attached, _use_beta_coop, _use_beta_lite) are now defined
        # by the hoisted block. The per-step Phase E flag resets are also
        # moved to the hoist site.
        #
        # PHASE 3 ORIGINAL (commented):
        # self._phase_e_consumed = False
        # self._phase_e_use_beta_lite = False
        # _phase_e_env = _phase_e_env_config()
        # _layer_idx: int | None = None
        # _layer_name = getattr(layer, "layer_name", None)
        # if _layer_name is not None:
        #     try:
        #         from vllm.model_executor.models.utils import extract_layer_index
        #         _layer_idx = extract_layer_index(_layer_name)
        #     except Exception:
        #         _layer_idx = None
        # _phase_e_attached = (
        #     getattr(self, "_phase_e_coop_kernel", None) is not None
        #     or getattr(self, "_mlp_fusion_bound", False)
        # )
        # _layer_allowed = (
        #     _phase_e_env.restricted_layers is None
        #     or (_layer_idx is not None
        #         and _layer_idx in _phase_e_env.restricted_layers)
        # )
        # _phase_e_active = (
        #     _phase_e_env.enabled
        #     and is_decode_only
        #     and _phase_e_attached
        #     and _layer_allowed
        #     and use_fusion
        #     and getattr(self, "_mlp_fusion_bound", False)
        #     and num_actual_tokens <= getattr(self, "_fusion_max_num_seqs", 0)
        # )
        # _CTAS_PER_SEQ = 64
        # _total_ctas = _CTAS_PER_SEQ * num_seqs
        # _resident_cap = getattr(self, "_resident_cap", 0)
        # _coop_attached = getattr(self, "_phase_e_coop_kernel", None) is not None
        # _use_beta_coop = (
        #     not _phase3_force_fallthrough
        #     and _phase_e_active
        #     and _coop_attached
        #     and _total_ctas <= _resident_cap
        #     and _phase_e_env.forced_path in ("coop", "auto")
        # )
        # _use_beta_lite = (
        #     _phase_e_active
        #     and not _use_beta_coop
        #     and (
        #         _phase_e_env.forced_path == "lite"
        #         or _phase_e_env.forced_path == "auto"
        #     )
        # )
        if _use_beta_coop:
            try:
                nat = num_actual_tokens
                # C1.5: Phase 4 (ε epilogue) deleted. β-coop no longer takes
                # next-LN gamma / next_hidden_output / emit_next_layernorm.
                # Locals + kwargs kept commented for Phase B/C debug recovery.
                # _next_ln = getattr(
                #     self, '_next_input_layernorm_module', None
                # )
                # _emit_next = getattr(self, '_emit_next_layernorm', False)
                # if _emit_next and _next_ln is not None:
                #     _next_gamma = _next_ln.weight
                #     _rms_eps = float(_next_ln.variance_epsilon)
                # else:
                #     _next_gamma = None
                #     _rms_eps = 1e-6

                # β-coop reads its workspace buffers from persistent
                # impl attributes (self._phase_e_coop_*) — separate from
                # β-lite's self.mlp_partial_fp32 / self.mlp_arrival_count
                # because β-coop's slice_ctas / num_k_tiles can differ.
                # Counter zero_() before launch happens inside
                # run_beta_coop_full at phase_e_kernel.py:3036-3038.
                # Spec: docs/superpowers/specs/2026-04-30-beta-coop-persistent-buffers-design.md
                # C1.5: rms_eps no longer read by β-coop (Phase 4 deleted),
                # but the kernel still has the attribute so leave default.
                # self._phase_e_coop_kernel.rms_eps = _rms_eps
                # Phase E.1 #3: per-layer NVTX span for torch-profiler
                # attribution. No-op when no profiler is active.
                # β-coop launch — use framework outputs when available,
                # else legacy self.X scratch. Phase 5 cleanup deletes
                # the self.X path entirely.
                _attn_output_buf = (
                    output_rmsnorm[:nat] if _framework_output_route
                    else self.rmsnorm_output[:nat]
                )
                _residual_output_buf = (
                    output_residual[:nat] if _framework_output_route
                    else self.residual_output[:nat]
                )
                _mlp_output_buf = (
                    output_mlp[:nat] if _framework_output_route
                    else self.mlp_output[:nat]
                )
                _residual_in_buf = (
                    residual[:nat] if _framework_output_route
                    else self.residual_buf[:nat]
                )
                _gate_buf = (
                    gate[:nat] if _framework_output_route
                    else self.gate_buf[:nat]
                )
                # v2 captured pre-launch reset of wo_output
                # (spec docs/superpowers/specs/2026-04-30-beta-coop-persistent-buffers-v2-design.md §4.3):
                # zeroes [:nat] before Phase 1 atomic_add re-accumulates.
                torch.ops.vllm.cute_paged_reset_wo_output(
                    self._phase_e_coop_wo_output, nat
                )
                # B' (2026-05-16): zero the populated slice of
                # region_timing_buf before each β-coop launch. Without
                # this, dynamic_single regions (R12 phase1_gather_reduce,
                # R15 phase3_3d_last_cta_gather) only overwrite the ONE
                # CTA elected this launch — and the host reducer's
                # `delta > 0` nonzero filter then picks up every CTA
                # that was ever elected across the buf's lifetime,
                # inflating n_active over time. All-CTA regions
                # (R0..R11, R13, R14) overwrite every active row per
                # launch and don't have this issue, but zeroing them is
                # harmless. Gated on timing-enabled so production pays
                # nothing.
                if self._phase_e_coop_region_timing is not None:
                    self._phase_e_coop_region_timing[
                        : nat
                        * self._phase_e_coop_kernel.slice_ctas
                        * self._phase_e_coop_kernel.num_k_tiles
                    ].zero_()
                with record_function(
                    f"PhaseE_Beta.coop.{_layer_name}"
                ):
                    self._phase_e_coop_kernel.run_beta_coop_full(
                        # Phase 0 inputs (dummy — output side-channel for future
                        # QKV-fusion; not consumed by this layer's attn path).
                        hidden_in=_attn_output_buf,  # placeholder; β-coop ignores
                        residual_in=_residual_in_buf,
                        input_gamma=self._phase_e_coop_input_gamma,
                        post_attn_gamma=self.rmsnorm_gamma,
                        attn_input_bf16=self._phase_e_coop_attn_input_scratch[:nat],
                        # Phase 1 inputs:
                        query=query[:nat],
                        kv_cache=kv_cache,
                        page_table=attn_metadata.block_table,
                        seq_lens=attn_metadata.seq_lens,
                        wo_weight=self.wo_weight,
                        wo_scales=self.wo_scales,
                        wo_global_scale=self.wo_global_scale,
                        attn_output=_attn_output_buf,
                        # Phase 3 inputs (MLP):
                        gate_w_fp4=self._mlp_gate_w,
                        gate_w_scale=self._mlp_gate_s,
                        up_w_fp4=self._mlp_up_w,
                        up_w_scale=self._mlp_up_s,
                        down_w_fp4=self._mlp_down_w,
                        down_w_scale=self._mlp_down_s,
                        mlp_output=_mlp_output_buf,
                        # C1.5: Phase 4 (ε) inputs disabled — kernel returns
                        # at end of Phase 3. Per-layer input_LN now runs
                        # from Python at every layer entry.
                        # next_input_layernorm_gamma=_next_gamma,
                        # next_hidden_output=self.next_hidden_scratch[:nat],
                        scale=self.scale,
                        k_scale=k_scale,
                        v_scale=v_scale,
                        gate_up_global_scale=self._mlp_gate_up_gs,
                        down_global_scale=self._mlp_down_gs,
                        # emit_next_layernorm=_emit_next,
                        # Caller-supplied residual_output so self.residual_output
                        # reflects residual_post_attn (Phase-1 Phase-C output).
                        residual_output=_residual_output_buf,
                        # C2: Qwen3.5 attn output gate — buffer was filled by
                        # qwen3_5.py:267 from the q_proj's gate slice. Mirrors
                        # the paged kernel's `gate_buf=` plumbing.
                        gate_buf=_gate_buf,
                        # Persistent workspace buffers (spec §4.3):
                        wo_output=self._phase_e_coop_wo_output[:nat],
                        mlp_partial_fp32=self._phase_e_coop_mlp_partial_fp32[:nat],
                        mlp_arrival_count=self._phase_e_coop_mlp_arrival_count[:nat],
                        grid_barrier_i32=self._phase_e_coop_grid_barrier_i32[:nat],
                        phase1_arrival_count=self._phase_e_coop_phase1_arrival_count[:nat],
                        # Task 6: pre-W_O arrival counter (dormant at
                        # wo_split=1 — consumer mask `bx>0 && bx<wo_split`
                        # is empty so no CTA spins, R11 buffer rows stay
                        # zero, host nonzero filter drops them).
                        pre_wo_arrival_count=self._phase_e_coop_pre_wo_arrival_count[:nat],
                        # Task 4 plumb: env-gated region-timing scratch (or
                        # None when CUTE_BETA_REGION_TIMING is unset; see
                        # _phase_e_coop_region_timing init above).
                        # Friend-review fix: persistent buffer is sized for
                        # max_num_seqs (e.g. 256 = 8*8*4); kernel validates
                        # against per-call nat (e.g. 64 = 8*8*1). Slice
                        # mirrors the wo_output[:nat] / mlp_partial[:nat]
                        # pattern above. Track the last-used slice size so
                        # the sentinel dump only writes the populated rows.
                        region_timing_buf=(
                            None
                            if self._phase_e_coop_region_timing is None
                            else self._phase_e_coop_region_timing[
                                : nat
                                * self._phase_e_coop_kernel.slice_ctas
                                * self._phase_e_coop_kernel.num_k_tiles
                            ]
                        ),
                    )
                # Track the slice size used for the most recent β-coop
                # launch — the sentinel dump (end of forward) must dump
                # only this many rows, not the full max_num_seqs buffer.
                if self._phase_e_coop_region_timing is not None:
                    self._phase_e_coop_region_timing_last_ctas = (
                        nat
                        * self._phase_e_coop_kernel.slice_ctas
                        * self._phase_e_coop_kernel.num_k_tiles
                    )
                self._phase_e_consumed = True
                self._phase_e_use_beta_coop = True
                # 2026-04-26: ENV-GATED dump for off-line math verification.
                # CUTE_DUMP_TENSORS=1 enables; bounded to first 3 decode
                # steps × 16 full-attn layers so disk doesn't bloat. Files
                # land in /tmp/nvllm-dumps/layer{N}_step{S}_{name}.pt.
                # See ~/jupyterlab/beta_coop_kernel_dump_compare.ipynb.
                if _CUTE_DUMP_TENSORS:
                    _dump_dir = "/tmp/nvllm-dumps"
                    os.makedirs(_dump_dir, exist_ok=True)
                    _step_counter = getattr(self, "_dump_step_counter", 0)
                    if _step_counter < 3 * 16:
                        _layer_segs = getattr(
                            layer, "layer_name", "<layer>").split(".")
                        _layer_digits = [
                            p for p in _layer_segs if p.isdigit()]
                        _layer_idx = int(_layer_digits[0]) \
                            if _layer_digits else -1
                        _base = (f"{_dump_dir}/layer{_layer_idx}_"
                                 f"step{_step_counter // 16}")
                        torch.save(
                            self.residual_buf[:nat].detach().clone(),
                            f"{_base}_residual_in.pt")
                        torch.save(
                            query[:nat].detach().clone(),
                            f"{_base}_query.pt")
                        torch.save(
                            self.gate_buf[:nat].detach().clone(),
                            f"{_base}_gate.pt")
                        torch.save(
                            self.residual_output[:nat].detach().clone(),
                            f"{_base}_residual_out.pt")
                        torch.save(
                            self.rmsnorm_output[:nat].detach().clone(),
                            f"{_base}_rmsnorm_out.pt")
                        self._dump_step_counter = _step_counter + 1
                # --- BETA_DIFF harness: diff β-coop's overwrite vs paged.
                # See memory:project_beta_coop_residual_solo_bug for protocol.
                if (_DEBUG_FUSION and is_decode_only
                    and getattr(self, "_debug_paged_res", None) is not None):
                    nat_dbg = num_actual_tokens
                    wo_diff = (
                        self.wo_output[:nat_dbg]
                        - self._debug_paged_wo[:nat_dbg]
                    ).abs()
                    rms_diff = (
                        self.rmsnorm_output[:nat_dbg].float()
                        - self._debug_paged_rms[:nat_dbg].float()
                    ).abs()
                    res_diff = (
                        self.residual_output[:nat_dbg].float()
                        - self._debug_paged_res[:nat_dbg].float()
                    ).abs()
                    # Also dump the raw β-coop residual_output[0, :8] and
                    # the corresponding paged value, so we can eyeball
                    # whether sentinel landed.
                    res_b = self.residual_output[0, :8].float().tolist()
                    res_p = self._debug_paged_res[0, :8].float().tolist()
                    logger.info(
                        "[BETA_DIFF] layer=%s nat=%d "
                        "wo:max=%.4e mean=%.4e | "
                        "rms:max=%.4e mean=%.4e | "
                        "res:max=%.4e mean=%.4e | "
                        "res_beta[0,:8]=%s | res_paged[0,:8]=%s",
                        getattr(layer, "layer_name", "<layer>"), nat_dbg,
                        wo_diff.max().item(), wo_diff.mean().item(),
                        rms_diff.max().item(), rms_diff.mean().item(),
                        res_diff.max().item(), res_diff.mean().item(),
                        res_b, res_p,
                    )
            except Exception as e:  # noqa: BLE001 — fail-closed, fall through to β-lite
                if _PHASE_E_FALLBACK_RAISE:
                    raise
                logger.warning(
                    "CuTe Phase E β-coop launch failed (falling back to "
                    "β-lite) layer=%s nat=%d %s: %r",
                    getattr(layer, "layer_name", "<layer>"),
                    num_actual_tokens, type(e).__name__, e,
                )
                self._phase_e_consumed = False
                self._phase_e_use_beta_coop = False
                # PHASE 5 EDIT (2026-04-28, friend's audit): paged was
                # skipped at top (_skip_paged=True because _use_beta_coop
                # =True). β-coop raised before populating its output
                # buffers. Re-run paged now so β-lite reads valid input:
                # - framework route: paged writes output_rmsnorm/
                #   output_residual that β-lite reads as MLP input
                # - non-route (e.g. CUTE_PHASE3_DIAG_DISABLE_FW=1
                #   rollback): paged writes self.rmsnorm_output/
                #   self.residual_output that β-lite reads
                # Guard is `_skip_paged and use_fusion` (not just route)
                # so non-framework paths get the same writer-invariant.
                # Re-zero wo_output/arrival_count first since β-coop
                # may have partially mutated them before raising.
                # β-coop's own wo/MLP scratch is internal to
                # run_beta_coop_full; β-lite re-zeros mlp_partial_fp32/
                # mlp_arrival_count itself, so no extra reset needed.
                if _skip_paged and use_fusion:
                    self.wo_output.zero_()
                    self.arrival_count.zero_()
                    result = _run_paged()
                # Retry via β-lite on this forward.
                _use_beta_lite = True
        if _use_beta_lite:
            try:
                # Zero the MLP per-step buffers BEFORE the kernel launch
                # (Python-side zero_ is stream-ordered before the kernel).
                nat = num_actual_tokens
                self.mlp_partial_fp32[:nat, :].zero_()
                self.mlp_arrival_count[:nat].zero_()

                # Next-input-layernorm gamma + eps from the module attached
                # by attach_next_input_layernorm (Task 3). When the module
                # is None this is the last decoder layer — ε epilogue
                # skips the norm and does a residual_final memcpy to
                # next_hidden_scratch (emit_next_layernorm=False).
                _next_ln = getattr(
                    self, '_next_input_layernorm_module', None
                )
                _emit_next = getattr(self, '_emit_next_layernorm', False)
                if _emit_next and _next_ln is not None:
                    _next_gamma = _next_ln.weight
                    _rms_eps = float(_next_ln.variance_epsilon)
                else:
                    _next_gamma = None
                    _rms_eps = 1e-6

                # β-lite launch — uses framework outputs when available.
                # When the framework-output route is active, paged
                # attention has already written output_rmsnorm and
                # output_residual; β-lite reads output_rmsnorm as MLP
                # input and writes output_mlp.
                _mlp_in = (
                    output_rmsnorm[:nat] if _framework_output_route
                    else self.rmsnorm_output[:nat]
                )
                _mlp_out = (
                    output_mlp[:nat] if _framework_output_route
                    else self.mlp_output[:nat]
                )
                _residual_post_ln = (
                    output_residual[:nat] if _framework_output_route
                    else self.residual_buf[:nat]
                )
                # Phase E.1 #3: per-layer NVTX span for torch-profiler
                # attribution. No-op when no profiler is active.
                with record_function(
                    f"PhaseE_Beta.lite.{_layer_name}"
                ):
                    self._mlp_kernel(
                        _mlp_in,
                        self._mlp_gate_w,
                        self._mlp_gate_s,
                        self._mlp_up_w,
                        self._mlp_up_s,
                        self._mlp_down_w,
                        self._mlp_down_s,
                        self.mlp_partial_fp32[:nat],
                        self.mlp_arrival_count[:nat],
                        # Reuse mlp_output as the Phase-D MLP output surface.
                        # C1.5 consumes raw MLP output and the post-attn
                        # residual separately; next layer input_LN performs
                        # the residual + MLP accumulation.
                        _mlp_out,
                        nat,
                        gate_up_global_scale=self._mlp_gate_up_gs,
                        down_global_scale=self._mlp_down_gs,
                        # ε epilogue inputs (Task 8 kwargs):
                        residual_post_ln=_residual_post_ln,
                        next_input_layernorm_gamma=_next_gamma,
                        next_hidden_output=self.next_hidden_scratch[:nat],
                        # C1.5 contract: β-lite is MLP-only. The epilogue
                        # mutates residual_post_ln in place, but current
                        # consumers read raw mlp_output + residual_output.
                        emit_epilogue=False,
                        emit_next_layernorm=_emit_next,
                        rms_eps=_rms_eps,
                    )
                # FIX #3 (2026-04-27, friend's analysis): in framework-output
                # mode, the decoder layer reads output_mlp/output_residual
                # DIRECTLY (returns before reaching cute_phase_e_dispatch).
                # `_phase_e_consumed=True` here would poison any later
                # non-framework fallback that still calls cute_phase_e_dispatch
                # — that op would try to consume from impl.mlp_output, which
                # in framework-output mode was NOT written (β-lite wrote
                # output_mlp instead). Keep the flag False on framework path.
                if not _framework_output_route:
                    self._phase_e_consumed = True
                self._phase_e_use_beta_lite = True
            except Exception as e:  # noqa: BLE001 — fail-closed, log & fall back
                logger.warning(
                    "CuTe Phase E β-lite launch failed (fallback to "
                    "legacy Phase D MLP path) layer=%s nat=%d %s: %r",
                    getattr(layer, "layer_name", "<layer>"),
                    num_actual_tokens, type(e).__name__, e,
                )
                self._phase_e_consumed = False
                self._phase_e_use_beta_lite = False
        # --- END PHASE E β-lite dispatch ---

        # --- PHASE 4 EDIT 4: writer-invariant fallback ------------------
        # Required when framework-output route is active but no β path
        # produced output_mlp (and possibly not output_rmsnorm/output_residual
        # either). Covers:
        #   - prefill / oversize batch (use_fusion=False; paged returned
        #     raw 2D `result`, didn't touch framework buffers)
        #   - decode where _phase_e_active=False (e.g. _layer_allowed=False
        #     or _phase_e_attached=False; paged ran fused → output_rmsnorm
        #     and output_residual are valid, only output_mlp missing)
        #   - β-coop and β-lite both raised (rare)
        # Uses modules stashed on impl by attach_fusion + attach_mlp_fusion:
        #   _o_proj_module, _post_norm_module, _attn_output_gate,
        #   _mlp_gate_up_proj, _mlp_act_fn, _mlp_down_proj.
        if (
            _framework_output_route
            and not self._phase_e_use_beta_coop
            and not self._phase_e_use_beta_lite
        ):
            nat = num_actual_tokens
            if use_fusion:
                # paged ran fused → output_rmsnorm + output_residual valid.
                # Only output_mlp missing.
                _mlp_in = output_rmsnorm[:nat]
            else:
                # paged returned 3D raw attn `[nat, num_q_heads, head_dim]`
                # into `result`. Reshape to 2D `[nat, q_size]` before the
                # gate multiply (gate is 2D `[nat, q_size]` from
                # qwen3_5.py:264) and o_proj input.
                attn_2d = result.reshape(nat, -1)
                if self._attn_output_gate and gate is not None:
                    gate_sigmoid = torch.sigmoid(gate[:nat])
                    attn_2d = attn_2d * gate_sigmoid
                wo_out, _ = self._o_proj_module(attn_2d)
                # Fused-residual post-attn LN: returns (LN(x+r)*(1+γ), x+r).
                # Per feedback_layer_output_contract, residual_post_attn = x+r,
                # NOT residual_final. The next layer's input_LN will re-fuse.
                out_x, out_residual = self._post_norm_module(
                    wo_out, residual[:nat]
                )
                output_rmsnorm[:nat].copy_(out_x)
                output_residual[:nat].copy_(out_residual)
                _mlp_in = output_rmsnorm[:nat]
            # MLP fallback into output_mlp[:nat]. Stashed unfused modules
            # are safe under graph capture because this entire block is
            # executed inside the cute_beta_coop_run splitting-op body
            # (eager Python between captured pieces).
            gate_up, _ = self._mlp_gate_up_proj(_mlp_in)
            mid = self._mlp_act_fn(gate_up)
            mlp_out, _ = self._mlp_down_proj(mid)
            output_mlp[:nat].copy_(mlp_out)
        # --- END PHASE 4 EDIT 4 -----------------------------------------

        # --- PHASE D2 DISABLED (commented, not deleted — Phase B/C debug may
        # need this attention-side launch path back) ---
        # Pre-D2 design: launch the Phase D MLP kernel as a side effect AFTER
        # the attention uber-kernel wrote rmsnorm_output, set _mlp_fusion_active
        # + _mlp_fusion_nat, and let Qwen3_5MLP.forward read impl.mlp_output.
        #
        # Verdict (see benchmarks/nvllm/traces/cute_paged_mlp_fusion/
        # 2026-04-18-phase-d1-custom-op/summary.md): this produced dual-firing.
        # The compiled graph also contained the fallback unfused GEMMs because
        # the fallback was reachable from traced Python. Phase D2 moves this
        # launch inside torch.ops.vllm.cute_mlp_forward (_mlp_op.py) — single
        # call site, single path visible to Inductor. Kept commented so the
        # Phase B/C kernel-math debug harness can swap back easily.
        #
        # if (
        #     getattr(self, "_mlp_fusion_bound", False)
        #     and use_fusion  # attention fusion succeeded (rmsnorm_output valid)
        #     and fits_buffer
        # ):
        #     try:
        #         # Zero per-step mutable buffers. Must precede kernel launch
        #         # (Python-side zero_ is stream-ordered before kernel).
        #         self.mlp_partial_fp32[:num_actual_tokens, :].zero_()
        #         self.mlp_arrival_count[:num_actual_tokens].zero_()
        #
        #         # Pass stream=None so Phase_D_MLP_Kernel wraps the current
        #         # torch stream as a CUstream internally; passing a
        #         # torch.cuda.Stream directly fails an internal CuTe DSL
        #         # `isinstance(arg, _cext.ir.Value)` assertion.
        #         self._mlp_kernel(
        #             self.rmsnorm_output[:num_actual_tokens],
        #             self._mlp_gate_w,
        #             self._mlp_gate_s,
        #             self._mlp_up_w,
        #             self._mlp_up_s,
        #             self._mlp_down_w,
        #             self._mlp_down_s,
        #             self.mlp_partial_fp32[:num_actual_tokens],
        #             self.mlp_arrival_count[:num_actual_tokens],
        #             self.mlp_output[:num_actual_tokens],
        #             num_actual_tokens,
        #         )
        #         self._mlp_fusion_active = True
        #         self._mlp_fusion_nat = num_actual_tokens
        #     except Exception as e:  # noqa: BLE001 — fail closed, log and fall back
        #         logger.warning(
        #             "CuTe MLP fusion launch failed (fallback to unfused) "
        #             "nat=%d %s: %r",
        #             num_actual_tokens, type(e).__name__, e,
        #         )
        #         self._mlp_fusion_active = False
        # --- END PHASE D2 DISABLED ---

        # C2: when paged_attention_forward was gated off (β-coop fired
        # alone), `result` is None. β-coop wrote its outputs into
        # self.rmsnorm_output / self.mlp_output / self.residual_output
        # which the consume branch reads directly — `output` is the
        # framework's unified attn-output buffer, not consumed in the
        # fusion path. Skip the copy_ in that case.
        #
        # Phase 3 (2026-04-27): when the framework-output route is active,
        # paged writes directly to output_rmsnorm (== output) via the
        # rmsnorm_output kwarg. The legacy 3D `result` (per-head attn
        # output) doesn't match the 2D `output_rmsnorm` shape and is also
        # redundant — skip the copy on this route.
        if result is not None and not _framework_output_route:
            output[:num_actual_tokens].copy_(result)

        # Region-timing buffer dump: triggered by host writing a sentinel
        # file; we check + delete + dump per call. Cheap when sentinel
        # is absent (one os.path.exists per forward).
        # Friend-review fixes: (a) gate on _phase_e_use_beta_coop so we
        # only dump after a real β-coop launch (not a β-lite fallback or
        # a non-fusion layer); (b) dump only the populated slice
        # (last_ctas), not the full max_num_seqs buffer.
        if (
            self._phase_e_coop_region_timing is not None
            and self._phase_e_use_beta_coop
            and self._phase_e_coop_region_timing_last_ctas > 0
            and os.path.exists("/tmp/.dump_region_timings")
        ):
            try:
                import numpy as np
                last_ctas = self._phase_e_coop_region_timing_last_ctas
                buf = (
                    self._phase_e_coop_region_timing[:last_ctas]
                    .detach()
                    .cpu()
                    .numpy()
                )
                out_path = "/root/.cache/vllm/region_timings.npy"
                os.makedirs(os.path.dirname(out_path), exist_ok=True)
                np.save(out_path, buf)
                # B' (2026-05-16): also drain the host-side launch-wall
                # CUDA-event pairs accumulated by run_beta_coop_full and
                # save them alongside region_timings.npy. The reducer
                # uses the last value as host_launch_wall_us to populate
                # wall_minus_regions_us. drain_launch_walls_us() syncs
                # the stream first; safe inside this dump path because
                # we're past the cute kernel launch for this step.
                walls_us = (
                    self._phase_e_coop_kernel.drain_launch_walls_us()
                )
                if walls_us:
                    walls_path = "/root/.cache/vllm/host_launch_walls.npy"
                    np.save(
                        walls_path,
                        np.asarray(walls_us, dtype=np.float64),
                    )
                    logger.warning(
                        "[β-coop region timing] dumped %d bytes "
                        "(shape=%s) to %s + %d host launch wall(s) "
                        "to %s (last_wall_us=%.3f)",
                        buf.nbytes, buf.shape, out_path,
                        len(walls_us), walls_path, walls_us[-1],
                    )
                else:
                    logger.warning(
                        "[β-coop region timing] dumped %d bytes "
                        "(shape=%s) to %s (no host launch walls "
                        "drained — queue was empty)",
                        buf.nbytes, buf.shape, out_path,
                    )
            finally:
                # Always clear the sentinel so we don't dump every step.
                try:
                    os.remove("/tmp/.dump_region_timings")
                except FileNotFoundError:
                    pass
        return output

    def _debug_fusion_diff(
        self,
        result: torch.Tensor,
        num_actual_tokens: int,
        layer_name: str,
    ) -> None:
        """One-shot per-call diagnostic: compare kernel wo_output to ref."""
        # Dequant W_O lazily on first call, then cache on self.
        if not hasattr(self, "_wo_dq_cached"):
            W = self.wo_weight  # [N, K/2] uint8 NVFP4 packed
            S_sw = self.wo_scales  # [N, K_sf] fp8_e4m3fn (swizzled!)
            GS = self.wo_global_scale.item()

            # Invert the CUTLASS swizzle to recover logical [N, K/16] scales.
            # Our swizzle layout is [M/128, K/4, 32, 4, 4]; inverse permute (0,4,3,1,2).
            N, K_half = W.shape
            K = K_half * 2
            num_k_groups = K // 16
            num_m_tiles = (N + 127) // 128
            num_k_tiles = (num_k_groups + 3) // 4
            if (
                S_sw.shape[0] == N
                and S_sw.shape[1] == num_k_groups
                and num_m_tiles * 128 == N
                and num_k_tiles * 4 == num_k_groups
            ):
                # Swizzled 5D layout: (m_tile, k_tile, m_inner=32, m_mid=4, k_inner=4).
                # Recover (m_tile, m_mid, m_inner, k_tile, k_inner) so reshape
                # yields M = m_tile*128 + m_mid*32 + m_inner in C order.
                S_sw_view = S_sw.view(num_m_tiles, num_k_tiles, 32, 4, 4)
                S_unswizzled = S_sw_view.permute(0, 3, 2, 1, 4).contiguous()
                S_unswizzled = S_unswizzled.view(N, num_k_groups).to(torch.float32)
            else:
                # Fall back: treat as logical already (diagnostic best-effort).
                S_unswizzled = S_sw.to(torch.float32).view(N, num_k_groups)

            # FP4 E2M1 LUT (matches kernel _fp4_nibble_to_f32)
            lut = torch.tensor(
                [
                    0.0,
                    0.5,
                    1.0,
                    1.5,
                    2.0,
                    3.0,
                    4.0,
                    6.0,
                    -0.0,
                    -0.5,
                    -1.0,
                    -1.5,
                    -2.0,
                    -3.0,
                    -4.0,
                    -6.0,
                ],
                dtype=torch.float32,
                device=W.device,
            )
            low_nib = (W & 0x0F).to(torch.int64)
            high_nib = ((W >> 4) & 0x0F).to(torch.int64)
            nib = torch.empty(N, K, dtype=torch.int64, device=W.device)
            nib[:, 0::2] = low_nib
            nib[:, 1::2] = high_nib
            W_fp = lut[nib]
            sf_expanded = S_unswizzled.repeat_interleave(16, dim=1)
            self._wo_dq_cached = (W_fp * sf_expanded * GS).contiguous()
            logger.info(
                "[CUTE_DEBUG_FUSION] layer=%s cached W_O dq: shape=%s absmax=%.4f",
                layer_name,
                list(self._wo_dq_cached.shape),
                self._wo_dq_cached.abs().max().item(),
            )

        W_dq = self._wo_dq_cached  # [N, K]
        nat = int(num_actual_tokens)
        attn = result[:nat].reshape(nat, -1).float()  # [nat, K]
        ref = attn @ W_dq.T  # [nat, N]

        # 2026-04-20 deterministic-reduction fix: wo_output is now
        # [max_num_seqs, total_ctas_per_seq, hidden_dim]; the summed
        # value lives in slot 0 after Phase B.5 gather (audit 16475223f).
        kernel_out = self.wo_output[:nat, 0].float()
        diff = (kernel_out - ref).abs()
        logger.info(
            "[CUTE_DEBUG_FUSION] layer=%s nat=%d phaseB  "
            "ref: absmax=%.4f mean=%.4e  "
            "kernel: absmax=%.4f mean=%.4e  "
            "diff: max=%.4f mean=%.4e  close=%s",
            layer_name,
            nat,
            ref.abs().max().item(),
            ref.mean().item(),
            kernel_out.abs().max().item(),
            kernel_out.mean().item(),
            diff.max().item(),
            diff.mean().item(),
            bool(torch.allclose(kernel_out, ref, rtol=1e-2, atol=1e-2)),
        )

        # --- Phase C reference: residual add + RMSNorm ---
        residual_in = self.residual_buf[:nat].float()  # BF16 → F32
        new_residual_ref = residual_in + kernel_out  # f32
        gamma = self.rmsnorm_gamma.float()
        eps = float(self.rmsnorm_eps)
        var = new_residual_ref.pow(2).mean(dim=-1, keepdim=True)
        inv_rms = torch.rsqrt(var + eps)
        hidden_ref = new_residual_ref * inv_rms * gamma  # f32

        hidden_kernel = self.rmsnorm_output[:nat].float()
        res_kernel = self.residual_output[:nat].float()
        h_diff = (hidden_kernel - hidden_ref).abs()
        r_diff = (res_kernel - new_residual_ref).abs()
        logger.info(
            "[CUTE_DEBUG_FUSION] layer=%s nat=%d phaseC  "
            "hidden_ref_absmax=%.4f hidden_kernel_absmax=%.4f h_max_diff=%.4f  "
            "res_ref_absmax=%.4f res_kernel_absmax=%.4f r_max_diff=%.4f  "
            "close_h=%s close_r=%s",
            layer_name,
            nat,
            hidden_ref.abs().max().item(),
            hidden_kernel.abs().max().item(),
            h_diff.max().item(),
            new_residual_ref.abs().max().item(),
            res_kernel.abs().max().item(),
            r_diff.max().item(),
            bool(torch.allclose(hidden_kernel, hidden_ref, rtol=2e-2, atol=2e-2)),
            bool(torch.allclose(res_kernel, new_residual_ref, rtol=2e-2, atol=2e-2)),
        )

    def do_kv_cache_update(
        self,
        layer: torch.nn.Module,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> None:
        """Write K/V to the paged cache via vLLM's C++ op."""
        if self.kv_sharing_target_layer_name is not None:
            return
        torch.ops._C_cache_ops.reshape_and_cache_flash(
            key,
            value,
            kv_cache[:, 0],
            kv_cache[:, 1],
            slot_mapping,
            self.kv_cache_dtype,
            layer._k_scale,
            layer._v_scale,
        )

    def _assert_spike_invariant(self) -> None:
        """Spike post-condition: when FULL_AND_PIECEWISE + Phase E ON +
        max_num_seqs=1, _beta_coop_framework_output_bound MUST be True.
        Otherwise the FULL graph would emit the legacy fall-through path.
        Outside the spike config this is a no-op. Spec §2.4."""
        try:
            from vllm.config import CUDAGraphMode  # local import — avoid cycles
            from vllm.config import get_current_vllm_config
            _vllm_config = get_current_vllm_config()
            _cg_mode = _vllm_config.compilation_config.cudagraph_mode
            _max_seqs = _vllm_config.scheduler_config.max_num_seqs
            _spike_config = (
                _cg_mode == CUDAGraphMode.FULL_AND_PIECEWISE
                and _PHASE_E_ENV.enabled
                and _max_seqs == 1
            )
        except Exception:  # noqa: BLE001 — config introspection best-effort
            _spike_config = False
        if _spike_config and not self._beta_coop_framework_output_bound:
            raise AssertionError(
                "Spike config (FULL_AND_PIECEWISE + CUTE_PHASE_E_FUSION=1 "
                "+ max_num_seqs=1) requires β-coop framework-output route "
                "bound after process_weights_after_loading. Got "
                "_beta_coop_framework_output_bound=False. The FULL graph "
                "would otherwise emit the legacy fall-through path. Spec: "
                "2026-04-29-full-and-piecewise-cute-spike-design.md §2.4."
            )

    def process_weights_after_loading(self, act_dtype: torch.dtype) -> None:
        """Invoked by vLLM's weight loader for each Attention module AFTER
        all quant methods have processed weights (swizzle, pad, invert GS).
        This is the last safe opportunity to bind fusion weights before
        torch.compile traces the forward pass — and it fires a SECOND time
        on live weight reload (see `layerwise.py:215-284`), so re-resolving
        on every call is a correctness requirement (code-review C2).
        """
        self._resolve_fusion_weights()
        self._resolve_mlp_weights()
        # Spike-only post-condition: covers every early-return path in
        # both resolvers in one place. Outside the spike config,
        # _beta_coop_framework_output_bound=False is legal.
        self._assert_spike_invariant()


# ---------------------------------------------------------------------------
# Metadata Builder
# ---------------------------------------------------------------------------


class CutePagedMetadataBuilder(
    AttentionMetadataBuilder[CutePagedMetadata],
):
    """Builds per-batch metadata for CuTe paged attention."""

    _cudagraph_support: ClassVar[AttentionCGSupport] = (
        AttentionCGSupport.UNIFORM_SINGLE_TOKEN_DECODE
    )

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ) -> None:
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        self.block_size = kv_cache_spec.block_size
        logger.info(
            "CutePagedMetadataBuilder: block_size=%d, layers=%d",
            self.block_size,
            len(layer_names),
        )

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> CutePagedMetadata:
        num_reqs = common_attn_metadata.num_reqs
        num_actual_tokens = common_attn_metadata.num_actual_tokens
        max_query_len = common_attn_metadata.max_query_len
        max_seq_len = common_attn_metadata.max_seq_len

        # Determine batch composition
        query_start_loc = common_attn_metadata.query_start_loc
        query_start_loc_cpu = common_attn_metadata.query_start_loc_cpu
        seq_lens = common_attn_metadata.seq_lens

        # Count prefill vs decode requests
        # Decode: query_len == 1, Prefill: query_len > 1
        query_lens_cpu = query_start_loc_cpu[1:] - query_start_loc_cpu[:-1]
        num_decodes = int((query_lens_cpu == 1).sum().item())
        num_prefills = num_reqs - num_decodes
        num_decode_tokens = num_decodes
        num_prefill_tokens = num_actual_tokens - num_decode_tokens

        return CutePagedMetadata(
            num_actual_tokens=num_actual_tokens,
            slot_mapping=common_attn_metadata.slot_mapping,
            num_decodes=num_decodes,
            num_decode_tokens=num_decode_tokens,
            num_prefills=num_prefills,
            num_prefill_tokens=num_prefill_tokens,
            seq_lens=seq_lens,
            query_start_loc=query_start_loc,
            max_query_len=max_query_len,
            max_seq_len=max_seq_len,
            block_table=common_attn_metadata.block_table_tensor,
            is_decode_only=(num_prefills == 0),
        )

    def build_for_cudagraph_capture(
        self,
        common_attn_metadata: CommonAttentionMetadata,
    ) -> CutePagedMetadata:
        """Override for CUDA graph capture.

        Fills seq_lens with 1 so every CTA exercises the full code path
        (one page load, one QK dot, etc.) during capture. Padding slots
        produce ignored results.
        """
        attn_metadata = self.build(0, common_attn_metadata)
        # All slots get seq_len=1: fast capture, full code path exercised
        attn_metadata.seq_lens.fill_(1)
        attn_metadata.is_decode_only = True
        return attn_metadata
