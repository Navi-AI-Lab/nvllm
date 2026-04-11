"""Tests for CutePagedBackend selection and fallback.

These tests require vLLM's full dependency chain (zmq, etc.) and will
be skipped when running outside the Docker container.
"""
import pytest
import torch

try:
    from vllm.v1.attention.backends.cute_paged._backend import (
        CutePagedBackend,
        CutePagedAttentionImpl,
    )
    from vllm.platforms.interface import DeviceCapability
    _HAS_VLLM_DEPS = True
except ImportError:
    _HAS_VLLM_DEPS = False

pytestmark = pytest.mark.skipif(
    not _HAS_VLLM_DEPS,
    reason="Requires full vLLM dependencies (run inside Docker container)",
)


class TestBackendGuards:
    def test_supports_sm121(self):
        assert CutePagedBackend.supports_compute_capability(
            DeviceCapability(12, 1)
        )

    def test_supports_sm120(self):
        assert CutePagedBackend.supports_compute_capability(
            DeviceCapability(12, 0)
        )

    def test_rejects_sm100(self):
        assert not CutePagedBackend.supports_compute_capability(
            DeviceCapability(10, 0)
        )

    def test_supports_head_128(self):
        assert CutePagedBackend.supports_head_size(128)

    def test_rejects_head_64(self):
        assert not CutePagedBackend.supports_head_size(64)

    def test_supports_fp8_kv(self):
        assert CutePagedBackend.supports_kv_cache_dtype("fp8_e4m3")
        assert CutePagedBackend.supports_kv_cache_dtype("fp8")

    def test_rejects_bf16_kv(self):
        assert not CutePagedBackend.supports_kv_cache_dtype("bfloat16")

    def test_block_sizes(self):
        assert CutePagedBackend.get_supported_kernel_block_sizes() == [64]

    def test_kv_cache_shape(self):
        shape = CutePagedBackend.get_kv_cache_shape(
            num_blocks=100, block_size=64, num_kv_heads=8, head_size=128
        )
        assert shape == (100, 64, 8, 128)

    def test_kv_cache_shape_rejects_wrong_block(self):
        with pytest.raises(ValueError):
            CutePagedBackend.get_kv_cache_shape(
                num_blocks=100, block_size=16, num_kv_heads=8, head_size=128
            )


class TestImplInit:
    def test_rejects_sliding_window(self):
        with pytest.raises(ValueError, match="sliding window"):
            CutePagedAttentionImpl(
                num_heads=32, head_size=128, scale=0.088,
                num_kv_heads=8, sliding_window=4096,
            )

    def test_rejects_logits_soft_cap(self):
        with pytest.raises(ValueError, match="logits_soft_cap"):
            CutePagedAttentionImpl(
                num_heads=32, head_size=128, scale=0.088,
                num_kv_heads=8, logits_soft_cap=30.0,
            )

    def test_profiling_pass_returns_zeros(self):
        impl = CutePagedAttentionImpl(
            num_heads=32, head_size=128, scale=0.088, num_kv_heads=8,
        )
        output = torch.ones(1, 32, 128, dtype=torch.bfloat16, device="cuda")
        result = impl.forward(
            layer=None, query=None, key=None, value=None,
            kv_cache=None, attn_metadata=None, output=output,
        )
        assert (result == 0).all()
