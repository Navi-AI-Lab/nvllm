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
        # Phase F.1: skip-flag set by cute_phase_e_dispatch when consuming
        # β output; read by cute_phase_e_skip_input_layernorm on layer N+1.
        self._phase_e_skip_next_ln = False
        # Phase F.1: this layer's own input_layernorm module, attached at
        # model-init post-processing (Qwen3_5Model.__init__). Used by the
        # opaque skip-op when the skip flag is unset.
        self._input_layernorm_module = None

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

    def attach_next_input_layernorm(
        self, next_input_layernorm_module: torch.nn.Module | None
    ) -> None:
        """Bind the next decoder layer's input_layernorm module for the
        Phase E β kernel's ε epilogue. Called from
        `Qwen3_5Model.__init__` post-hook (see vllm/nvllm/models/qwen3_5.py).

        Pass `None` for the last decoder layer (index 63 in Qwen3.5-27B):
        ε epilogue then omits the next-layer norm and writes residual
        straight to residual_output. See spec §5.3.

        Stores the MODULE ref (not the tensor) — NVFP4's
        process_weights_after_loading replaces Parameters, so a tensor
        captured here would go stale. Mirrors attach_fusion pattern.
        """
        assert getattr(self, '_fusion_attached', False), (
            "attach_next_input_layernorm: attach_fusion must run first"
        )
        self._next_input_layernorm_module = next_input_layernorm_module
        self._emit_next_layernorm = next_input_layernorm_module is not None

        # Kill-switch: refuse to enable β if host memory is tight.
        # Runs BEFORE workspace allocation so a refusal doesn't leave
        # ~1.25 MiB of half-attached tensors on self.
        # CUTE_BETA_MIN_FREE_GB is plumbed via serve scripts
        # (commit 5c000a09d); default 8 GiB preserves KV + CUDA graph
        # headroom on GB10 (128 GiB unified).
        min_free_gb = float(os.environ.get("CUTE_BETA_MIN_FREE_GB", "8"))
        free_bytes, _total = torch.cuda.mem_get_info()
        if free_bytes < min_free_gb * (1024 ** 3):
            free_gb = free_bytes / (1024 ** 3)
            raise RuntimeError(
                f"CUTE_BETA_MIN_FREE_GB={min_free_gb} GiB threshold not met: "
                f"only {free_gb:.1f} GiB free. "
                f"Lower threshold or free memory before enabling Phase E."
            )

        # Allocate Phase E workspace once. Qwen3_5Model.__init__ runs
        # inside vLLM's set_current_vllm_config() context, so
        # get_current_vllm_config() is always available on the real
        # serving path. Unit tests that bypass __init__ must stub this.
        if not hasattr(self, 'phase_e_barrier'):
            from vllm.config import get_current_vllm_config
            cfg = get_current_vllm_config()
            num_layers = cfg.model_config.hf_text_config.num_hidden_layers
            self.phase_e_barrier = torch.zeros(
                num_layers, dtype=torch.int32, device='cuda',
            )
            self.next_hidden_scratch = torch.empty(
                self._fusion_max_num_seqs, self._fusion_hidden_dim,
                dtype=torch.bfloat16, device='cuda',
            )

        # Probe resident-CTA cap once (SMEM-bound estimate; a real
        # cuOccupancy probe re-runs post kernel compile in the launch
        # path). 45568 B matches the β kernel SMEM footprint (spec §6).
        self._resident_cap = self._probe_resident_cap(
            kernel_fn=None, num_threads=128, smem_bytes=45568
        )
        logger.info(
            "CuTe Phase E: resident_cap=%d (num_seqs_coop_max=%d)",
            self._resident_cap, max(1, self._resident_cap // 64)
        )

        logger.info(
            "CuTe Phase E next-input-layernorm attached: "
            "emit_next_layernorm=%s num_layers_barrier=%d "
            "scratch_shape=(%d, %d)",
            self._emit_next_layernorm,
            self.phase_e_barrier.numel(),
            self._fusion_max_num_seqs,
            self._fusion_hidden_dim,
        )

    def attach_input_layernorm(
        self, input_layernorm_module: torch.nn.Module | None,
    ) -> None:
        """Phase F.1: Attach THIS layer's input_layernorm module so the
        opaque cute_phase_e_skip_input_layernorm op can invoke it at
        call time (when the skip flag is unset).

        Mirror of attach_next_input_layernorm; the two are paired.
        Stores the MODULE ref (not the tensor) — NVFP4's
        process_weights_after_loading replaces Parameters, so a tensor
        captured here would go stale.
        """
        self._input_layernorm_module = input_layernorm_module

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
                logger.info(
                    "CuTe Phase E β-coop kernel attached: hidden=%d "
                    "intermediate=%d num_q_heads=%d num_kv_heads=%d "
                    "head_dim=%d",
                    hidden_size, intermediate_size,
                    self.num_heads, self.num_kv_heads, self.head_size,
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
            return
        if not hasattr(down, "weight_global_scale"):
            logger.warning(
                "CuTe MLP fusion: down_proj weights not NVFP4; disabled."
            )
            self._mlp_fusion_bound = False
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
    ) -> torch.Tensor:
        assert output is not None, "Output tensor must be provided"

        if attn_metadata is None:
            return output.fill_(0)

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

        result = paged_attention_forward(
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
            rmsnorm_residual=rmsnorm_residual,
            rmsnorm_output=rmsnorm_output,
            residual_output=residual_output,
            arrival_count=arrival_count,
            rmsnorm_eps=rmsnorm_eps,
            gate_buf=gate_buf,
            padded_num_seqs=padded_num_seqs,
        )

        # --- DEBUG: fusion diagnostic (CUTE_DEBUG_FUSION=1) ---
        # Compares kernel's impl.wo_output (Phase B GEMV) against a Python
        # reference computed from the kernel's own Phase A output (`result`)
        # and a one-time-dequantized W_O. Proves whether Phase B is faithful.
        if _DEBUG_FUSION and use_fusion:
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
        self._phase_e_consumed = False
        self._phase_e_use_beta_lite = False
        _phase_e_env = _phase_e_env_config()
        # `layer.layer_name` is the canonical identifier on vllm's Attention;
        # `layer.layer_idx` is not populated, so extract the int index from
        # the dotted name via `extract_layer_index` (same helper used elsewhere
        # in the codebase).
        _layer_idx: int | None = None
        _layer_name = getattr(layer, "layer_name", None)
        if _layer_name is not None:
            try:
                from vllm.model_executor.models.utils import extract_layer_index
                _layer_idx = extract_layer_index(_layer_name)
            except Exception:
                _layer_idx = None
        _phase_e_attached = hasattr(self, '_next_input_layernorm_module')
        _layer_allowed = (
            _phase_e_env.restricted_layers is None
            or (_layer_idx is not None
                and _layer_idx in _phase_e_env.restricted_layers)
        )
        # INVARIANT: β-lite reads `self.residual_output` below, which is only
        # populated by the attention uber-kernel when `use_fusion=True`. Keep
        # `use_fusion` in this AND; removing it would silently feed stale
        # residual data from the previous step into the ε epilogue.
        _phase_e_active = (
            _phase_e_env.enabled
            and is_decode_only
            and _phase_e_attached
            and _layer_allowed
            and use_fusion  # INVARIANT above — do not remove
            and getattr(self, "_mlp_fusion_bound", False)
            and num_actual_tokens <= getattr(self, "_fusion_max_num_seqs", 0)
        )
        # 64 = CTAs-per-seq in the attn grid (num_q_tiles=1 × num_kv_heads=4
        # × slice_ctas=8 × num_k_tiles=8 = 64 per seq for the β grid).
        _CTAS_PER_SEQ = 64
        _total_ctas = _CTAS_PER_SEQ * num_seqs
        _resident_cap = getattr(self, "_resident_cap", 0)
        # Task 16: β-coop dispatch. β-coop requires the unified kernel
        # attached in attach_mlp_fusion (CUTE_PHASE_E_FUSION=1 at attach
        # time). forced_path="coop" always routes here; "auto" routes here
        # when the full grid fits the resident cap for a single cooperative
        # launch (otherwise β-lite's two-kernel path handles it).
        _coop_attached = getattr(self, "_phase_e_coop_kernel", None) is not None
        _use_beta_coop = _phase_e_active and _coop_attached and (
            _phase_e_env.forced_path == "coop"
            or (
                _phase_e_env.forced_path == "auto"
                and _total_ctas <= _resident_cap
            )
        )
        _use_beta_lite = (
            _phase_e_active
            and not _use_beta_coop
            and (
                _phase_e_env.forced_path == "lite"
                or _phase_e_env.forced_path == "auto"
            )
        )
        if _use_beta_coop:
            try:
                nat = num_actual_tokens
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

                # run_beta_coop_full allocates its own internal MLP partial/
                # arrival/grid-barrier buffers — unlike β-lite we do NOT
                # zero self.mlp_partial_fp32 / self.mlp_arrival_count.
                # Future optimization: hoist those into pre-allocated
                # per-impl buffers (task-21 perf work).
                self._phase_e_coop_kernel.rms_eps = _rms_eps
                # Phase E.1 #3: per-layer NVTX span for torch-profiler
                # attribution. No-op when no profiler is active.
                with record_function(
                    f"PhaseE_Beta.coop.{_layer_name}"
                ):
                    self._phase_e_coop_kernel.run_beta_coop_full(
                        # Phase 0 inputs (dummy — output side-channel for future
                        # QKV-fusion; not consumed by this layer's attn path).
                        hidden_in=self.rmsnorm_output[:nat],
                        residual_in=self.residual_output[:nat],
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
                        attn_output=self.rmsnorm_output[:nat],
                        # Phase 3 inputs (MLP):
                        gate_w_fp4=self._mlp_gate_w,
                        gate_w_scale=self._mlp_gate_s,
                        up_w_fp4=self._mlp_up_w,
                        up_w_scale=self._mlp_up_s,
                        down_w_fp4=self._mlp_down_w,
                        down_w_scale=self._mlp_down_s,
                        mlp_output=self.mlp_output[:nat],
                        # Phase 4 inputs (ε):
                        next_input_layernorm_gamma=_next_gamma,
                        next_hidden_output=self.next_hidden_scratch[:nat],
                        scale=self.scale,
                        k_scale=k_scale,
                        v_scale=v_scale,
                        gate_up_global_scale=self._mlp_gate_up_gs,
                        down_global_scale=self._mlp_down_gs,
                        emit_next_layernorm=_emit_next,
                        # Caller-supplied residual_output so self.residual_output
                        # reflects residual_final (matches β-lite post-forward state).
                        residual_output=self.residual_output[:nat],
                    )
                self._phase_e_consumed = True
                self._phase_e_use_beta_coop = True
            except Exception as e:  # noqa: BLE001 — fail-closed, fall through to β-lite
                logger.warning(
                    "CuTe Phase E β-coop launch failed (falling back to "
                    "β-lite) layer=%s nat=%d %s: %r",
                    getattr(layer, "layer_name", "<layer>"),
                    num_actual_tokens, type(e).__name__, e,
                )
                self._phase_e_consumed = False
                self._phase_e_use_beta_coop = False
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

                # Phase E.1 #3: per-layer NVTX span for torch-profiler
                # attribution. No-op when no profiler is active.
                with record_function(
                    f"PhaseE_Beta.lite.{_layer_name}"
                ):
                    self._mlp_kernel(
                        self.rmsnorm_output[:nat],
                        self._mlp_gate_w,
                        self._mlp_gate_s,
                        self._mlp_up_w,
                        self._mlp_up_s,
                        self._mlp_down_w,
                        self._mlp_down_s,
                        self.mlp_partial_fp32[:nat],
                        self.mlp_arrival_count[:nat],
                        # Reuse mlp_output as the Phase-D MLP output surface
                        # (epilogue then consumes it for residual+norm).
                        self.mlp_output[:nat],
                        nat,
                        gate_up_global_scale=self._mlp_gate_up_gs,
                        down_global_scale=self._mlp_down_gs,
                        # ε epilogue inputs (Task 8 kwargs):
                        residual_post_ln=self.residual_output[:nat],
                        next_input_layernorm_gamma=_next_gamma,
                        next_hidden_output=self.next_hidden_scratch[:nat],
                        emit_epilogue=True,
                        emit_next_layernorm=_emit_next,
                        rms_eps=_rms_eps,
                    )
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

        output[:num_actual_tokens].copy_(result)
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
