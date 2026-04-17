# Copyright 2026 Navi Ai Labs
# SPDX-License-Identifier: Apache-2.0
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

from vllm.logger import init_logger

# Set CUTE_DEBUG_FUSION=1 to enable per-call diff vs Python-dequant W_O ref.
_DEBUG_FUSION = os.environ.get("CUTE_DEBUG_FUSION", "0") == "1"
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

        self._fusion_bound = False
        self._fusion_active = False

        # Pre-allocate fusion buffers during init so they don't
        # interfere with vLLM V1's memory pool during forward.
        # Uses vllm_config to get max_num_seqs and hidden_dim.
        try:
            from vllm.config import get_current_vllm_config
            cfg = get_current_vllm_config()
            max_num_seqs = cfg.scheduler_config.max_num_seqs
            hidden_dim = cfg.model_config.hf_config.hidden_size
            q_size = self.num_heads * self.head_size
            self._preallocate_fusion_buffers(
                max_num_seqs, hidden_dim, q_size, "cuda")
        except Exception:
            pass  # Will allocate lazily in bind_fusion_weights

    def _preallocate_fusion_buffers(
        self,
        max_num_seqs: int,
        hidden_dim: int,
        q_size: int,
        device: str | torch.device,
    ) -> None:
        """Allocate persistent fusion I/O buffers.

        Called during __init__ (before forward) so allocations don't
        interfere with vLLM V1's pre-allocated memory pool.
        """
        self.wo_output = torch.zeros(
            max_num_seqs, hidden_dim, dtype=torch.float32, device=device)
        self.rmsnorm_output = torch.empty(
            max_num_seqs, hidden_dim, dtype=torch.bfloat16, device=device)
        self.residual_output = torch.empty(
            max_num_seqs, hidden_dim, dtype=torch.bfloat16, device=device)
        self.arrival_count = torch.zeros(
            max_num_seqs, dtype=torch.int32, device=device)
        self.gate_buf = torch.empty(
            max_num_seqs, q_size, dtype=torch.bfloat16, device=device)
        self.residual_buf = torch.empty(
            max_num_seqs, hidden_dim, dtype=torch.bfloat16, device=device)

    def bind_fusion_weights(
        self,
        wo_weight: torch.Tensor,
        wo_scales: torch.Tensor,
        wo_global_scale: torch.Tensor,
        rmsnorm_gamma: torch.Tensor,
        rmsnorm_eps: float,
        max_num_seqs: int,
    ) -> None:
        """Bind static fusion weights and allocate persistent I/O buffers.

        Called once from the model layer after weight loading. Replaces
        the per-forward side-channel set/clear pattern. All buffer
        addresses are stable — safe for CUDA graph capture and replay.

        Args:
            wo_weight: NVFP4 packed weights [N, K/2] uint8
            wo_scales: Per-block scales [N, K_sf] fp8
            wo_global_scale: Scalar scale [1] fp32 (kernel reads via ld.global)
            rmsnorm_gamma: LayerNorm weight [hidden_dim] bf16
            rmsnorm_eps: LayerNorm epsilon (e.g. 1e-6)
            max_num_seqs: Maximum batch size for buffer allocation
        """
        # Static weights (bound once, never change)
        self.wo_weight = wo_weight
        self.wo_scales = wo_scales
        self.wo_global_scale = wo_global_scale
        self.rmsnorm_gamma = rmsnorm_gamma
        self.rmsnorm_eps = rmsnorm_eps

        hidden_dim = rmsnorm_gamma.shape[0]
        q_size = self.num_heads * self.head_size  # num_heads * head_dim

        # Persistent I/O buffers are pre-allocated during __init__ via
        # _preallocate_fusion_buffers() so they don't interfere with
        # vLLM V1's memory pool during the first forward pass.
        # If not yet allocated (e.g. __init__ didn't have config), do it now.
        if not hasattr(self, 'wo_output'):
            self._preallocate_fusion_buffers(
                max_num_seqs, hidden_dim, q_size, wo_weight.device)

        self._fusion_bound = True

        logger.info(
            "CuTe fusion bound: hidden_dim=%d, q_size=%d, max_seqs=%d, "
            "wo_weight=%s, rmsnorm_gamma=%s",
            hidden_dim, q_size, max_num_seqs,
            list(wo_weight.shape), list(rmsnorm_gamma.shape),
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

        if attn_metadata is None:
            return output.fill_(0)

        k_scale = getattr(layer, "_k_scale_float", 1.0)
        v_scale = getattr(layer, "_v_scale_float", 1.0)

        # Fusion requires both: weights bound AND model layer opted in.
        # _fusion_bound = weights/buffers allocated (set once at init).
        # _fusion_active = model layer says "this forward is fused" (per-call).
        use_fusion = self._fusion_bound and self._fusion_active
        if _DEBUG_FUSION:
            logger.info(
                "[CUTE_DEBUG_FUSION] layer=%s bound=%s active=%s use_fusion=%s",
                getattr(layer, "layer_name", "<layer>"),
                self._fusion_bound, self._fusion_active, use_fusion,
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
        # Must happen before any CTA's atomicAdd — Python-side zero_()
        # is ordered before the kernel by CUDA stream semantics.
        # (Self-zero inside the kernel races across KV-head CTAs.)
        if use_fusion:
            self.wo_output.zero_()
            self.arrival_count.zero_()

        from vllm.v1.attention.backends.cute_paged.kernel import (
            paged_attention_forward,
        )

        num_actual_tokens = attn_metadata.num_actual_tokens

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
            W = self.wo_weight           # [N, K/2] uint8 NVFP4 packed
            S_sw = self.wo_scales        # [N, K_sf] fp8_e4m3fn (swizzled!)
            GS = self.wo_global_scale.item()

            # Invert the CUTLASS swizzle to recover logical [N, K/16] scales.
            # Our swizzle layout is [M/128, K/4, 32, 4, 4]; inverse permute (0,4,3,1,2).
            N, K_half = W.shape
            K = K_half * 2
            num_k_groups = K // 16
            num_m_tiles = (N + 127) // 128
            num_k_tiles = (num_k_groups + 3) // 4
            if S_sw.shape[0] == N and S_sw.shape[1] == num_k_groups \
                    and num_m_tiles * 128 == N and num_k_tiles * 4 == num_k_groups:
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
                [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
                 -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
                dtype=torch.float32, device=W.device,
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
                layer_name, list(self._wo_dq_cached.shape),
                self._wo_dq_cached.abs().max().item(),
            )

        W_dq = self._wo_dq_cached            # [N, K]
        nat = int(num_actual_tokens)
        attn = result[:nat].reshape(nat, -1).float()  # [nat, K]
        ref = attn @ W_dq.T                           # [nat, N]

        kernel_out = self.wo_output[:nat].float()
        diff = (kernel_out - ref).abs()
        logger.info(
            "[CUTE_DEBUG_FUSION] layer=%s nat=%d phaseB  "
            "ref: absmax=%.4f mean=%.4e  "
            "kernel: absmax=%.4f mean=%.4e  "
            "diff: max=%.4f mean=%.4e  close=%s",
            layer_name, nat,
            ref.abs().max().item(), ref.mean().item(),
            kernel_out.abs().max().item(), kernel_out.mean().item(),
            diff.max().item(), diff.mean().item(),
            bool(torch.allclose(kernel_out, ref, rtol=1e-2, atol=1e-2)),
        )

        # --- Phase C reference: residual add + RMSNorm ---
        residual_in = self.residual_buf[:nat].float()         # BF16 → F32
        new_residual_ref = residual_in + kernel_out            # f32
        gamma = self.rmsnorm_gamma.float()
        eps = float(self.rmsnorm_eps)
        var = new_residual_ref.pow(2).mean(dim=-1, keepdim=True)
        inv_rms = torch.rsqrt(var + eps)
        hidden_ref = new_residual_ref * inv_rms * gamma        # f32

        hidden_kernel = self.rmsnorm_output[:nat].float()
        res_kernel = self.residual_output[:nat].float()
        h_diff = (hidden_kernel - hidden_ref).abs()
        r_diff = (res_kernel - new_residual_ref).abs()
        logger.info(
            "[CUTE_DEBUG_FUSION] layer=%s nat=%d phaseC  "
            "hidden_ref_absmax=%.4f hidden_kernel_absmax=%.4f h_max_diff=%.4f  "
            "res_ref_absmax=%.4f res_kernel_absmax=%.4f r_max_diff=%.4f  "
            "close_h=%s close_r=%s",
            layer_name, nat,
            hidden_ref.abs().max().item(), hidden_kernel.abs().max().item(),
            h_diff.max().item(),
            new_residual_ref.abs().max().item(), res_kernel.abs().max().item(),
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
        # Invoked by vLLM's weight loader for each Attention module AFTER
        # quant methods have processed weights (swizzle, pad, invert GS).
        # This is the last safe opportunity to bind fusion state before
        # torch.compile traces the forward pass.
        cb = getattr(self, "_fusion_bind_callback", None)
        if cb is not None:
            cb()


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

    def build_for_cudagraph_capture(
        self, common_attn_metadata: CommonAttentionMetadata,
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
