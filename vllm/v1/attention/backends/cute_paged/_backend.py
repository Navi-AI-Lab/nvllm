# Copyright 2026 Navi Ai Labs
# SPDX-License-Identifier: Apache-2.0
"""CuTe DSL paged attention backend classes for SM120/SM121 (GB10).

Custom attention kernel using CuTe Python DSL with FP8 MMA for QK,
BF16 MMA for PV, and CpAsync for paged KV loads. Targets NVIDIA GB10
(DGX Spark) with owned KV page layout optimized for SM121 SMEM budget.

See: docs/superpowers/specs/2026-04-10-cute-paged-attention-design.md
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

import torch

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
    seq_lens: torch.Tensor          # [num_seqs] int32 on device
    query_start_loc: torch.Tensor   # [num_seqs + 1] int32 on device
    max_query_len: int
    max_seq_len: int

    # Page table
    block_table: torch.Tensor       # [num_seqs, max_blocks_per_seq] int32

    # Flags
    is_decode_only: bool


# ---------------------------------------------------------------------------
# Backend
# ---------------------------------------------------------------------------

class CutePagedBackend(AttentionBackend):
    """CuTe DSL paged attention backend for SM120/SM121."""

    accept_output_buffer: bool = True
    forward_includes_kv_cache_update: bool = False

    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.bfloat16]
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [
        "fp8", "fp8_e4m3",
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
        cls, capability: DeviceCapability,
    ) -> bool:
        return capability.major == 12

    @classmethod
    def supports_kv_cache_dtype(
        cls, kv_cache_dtype: CacheDType | None,
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
            raise ValueError(
                "CutePagedAttention does not support sliding window"
            )
        if logits_soft_cap is not None:
            raise ValueError(
                "CutePagedAttention does not support logits_soft_cap"
            )
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
            self.num_heads, self.num_kv_heads,
            self.head_size, self.num_queries_per_kv,
        )

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

        # Profiling pass: no computation needed
        if attn_metadata is None:
            return output.fill_(0)

        # Extract descale factors from the attention layer
        k_scale = getattr(layer, "_k_scale_float", 1.0)
        v_scale = getattr(layer, "_v_scale_float", 1.0)

        from vllm.v1.attention.backends.cute_paged.kernel import (
            paged_attention_forward,
        )

        # kv_cache shape: [num_pages, 2, 64, num_kv_heads, head_dim] uint8
        # Dim 1: 0=K, 1=V (FlashInfer convention)
        # Kernel uses raw _ld_global_b32 with stride-aware addressing —
        # K/V byte offsets computed from kv_cache.stride(1). No copy needed.
        num_actual_tokens = attn_metadata.num_actual_tokens

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
        )

        # Kernel returns 3D [num_actual_tokens, num_heads, head_dim],
        # matching the output buffer shape from vLLM's attention layer.
        output[:num_actual_tokens].copy_(result)
        return output

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
        pass


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
            self.block_size, len(layer_names),
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
        query_lens_cpu = (
            query_start_loc_cpu[1:] - query_start_loc_cpu[:-1]
        )
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
