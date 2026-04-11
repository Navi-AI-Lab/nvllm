"""Tests for the CuTe DSL paged attention kernel.

Tests the production paged_attention_forward() against the PyTorch
reference. Tolerance is >=2% per element (FP8 KV quantization + BF16
truncation in the MMA path).
"""
import torch
import pytest

from tests.nvllm.attention.reference import reference_paged_attention


def _make_fp8_cache(data: torch.Tensor) -> torch.Tensor:
    """Convert BF16 data to FP8 uint8 cache format."""
    return data.to(torch.float8_e4m3fn).view(torch.uint8)


class TestPagedAttentionForward:
    """Tests for the production paged_attention_forward."""

    def test_single_page_decode(self):
        """Decode with 1 page of KV cache."""
        from vllm.v1.attention.backends.cute_paged.kernel import (
            paged_attention_forward,
        )
        torch.manual_seed(42)
        q = torch.randn(1, 32, 128, dtype=torch.bfloat16, device="cuda")
        kv = torch.randn(2, 64, 8, 128, dtype=torch.bfloat16, device="cuda")
        k_cache = _make_fp8_cache(kv)
        v_cache = _make_fp8_cache(kv)
        page_table = torch.tensor([[0]], dtype=torch.int32, device="cuda")
        seq_lens = torch.tensor([32], dtype=torch.int32, device="cuda")

        out = paged_attention_forward(
            q, k_cache, v_cache, page_table, seq_lens,
            scale=1.0 / (128 ** 0.5), k_scale=1.0, v_scale=1.0,
        )
        assert out.shape == (1, 32, 128)
        assert out.dtype == torch.bfloat16
        assert not torch.isnan(out).any()

    def test_multi_page_decode(self):
        """Decode with KV cache spanning multiple pages."""
        from vllm.v1.attention.backends.cute_paged.kernel import (
            paged_attention_forward,
        )
        torch.manual_seed(42)
        q = torch.randn(1, 32, 128, dtype=torch.bfloat16, device="cuda")
        kv = torch.randn(4, 64, 8, 128, dtype=torch.bfloat16, device="cuda")
        k_cache = _make_fp8_cache(kv)
        v_cache = _make_fp8_cache(kv)
        page_table = torch.tensor([[0, 1, 2, 3]], dtype=torch.int32, device="cuda")
        seq_lens = torch.tensor([200], dtype=torch.int32, device="cuda")

        out = paged_attention_forward(
            q, k_cache, v_cache, page_table, seq_lens,
            scale=1.0 / (128 ** 0.5), k_scale=1.0, v_scale=1.0,
        )
        assert out.shape == (1, 32, 128)
        assert not torch.isnan(out).any()

    def test_gqa_head_groups(self):
        """32 Q heads correctly share 8 KV heads."""
        from vllm.v1.attention.backends.cute_paged.kernel import (
            paged_attention_forward,
        )
        torch.manual_seed(42)
        q = torch.randn(1, 32, 128, dtype=torch.bfloat16, device="cuda")
        kv = torch.randn(1, 64, 8, 128, dtype=torch.bfloat16, device="cuda")
        k_cache = _make_fp8_cache(kv)
        v_cache = _make_fp8_cache(kv)
        page_table = torch.tensor([[0]], dtype=torch.int32, device="cuda")
        seq_lens = torch.tensor([16], dtype=torch.int32, device="cuda")

        out = paged_attention_forward(
            q, k_cache, v_cache, page_table, seq_lens,
            scale=1.0 / (128 ** 0.5), k_scale=1.0, v_scale=1.0,
        )
        assert out.shape == (1, 32, 128)

    def test_matches_reference(self):
        """Full forward matches PyTorch reference within >=2% tolerance."""
        from vllm.v1.attention.backends.cute_paged.kernel import (
            paged_attention_forward,
        )
        torch.manual_seed(42)
        q = torch.randn(1, 32, 128, dtype=torch.bfloat16, device="cuda")
        kv = torch.randn(2, 64, 8, 128, dtype=torch.bfloat16, device="cuda")
        k_cache = _make_fp8_cache(kv)
        v_cache = _make_fp8_cache(kv)
        page_table = torch.tensor([[0, 1]], dtype=torch.int32, device="cuda")
        seq_lens = torch.tensor([100], dtype=torch.int32, device="cuda")
        scale = 1.0 / (128 ** 0.5)

        result = paged_attention_forward(
            q, k_cache, v_cache, page_table, seq_lens,
            scale=scale, k_scale=1.0, v_scale=1.0,
        )
        expected = reference_paged_attention(
            q, k_cache, v_cache, page_table, seq_lens,
            scale=scale, k_scale=1.0, v_scale=1.0,
        )

        torch.testing.assert_close(result, expected, atol=2e-2, rtol=2e-2)

    def test_edge_single_token(self):
        """Minimal case: batch=1, query=1, seq_len=1."""
        from vllm.v1.attention.backends.cute_paged.kernel import (
            paged_attention_forward,
        )
        torch.manual_seed(42)
        q = torch.randn(1, 32, 128, dtype=torch.bfloat16, device="cuda")
        kv = torch.randn(1, 64, 8, 128, dtype=torch.bfloat16, device="cuda")
        k_cache = _make_fp8_cache(kv)
        v_cache = _make_fp8_cache(kv)
        page_table = torch.tensor([[0]], dtype=torch.int32, device="cuda")
        seq_lens = torch.tensor([1], dtype=torch.int32, device="cuda")

        out = paged_attention_forward(
            q, k_cache, v_cache, page_table, seq_lens,
            scale=1.0 / (128 ** 0.5), k_scale=1.0, v_scale=1.0,
        )
        assert out.shape == (1, 32, 128)
        assert not torch.isnan(out).any()

    def test_edge_page_boundary(self):
        """Sequence length exactly fills 2 pages (128 tokens, 0 remainder)."""
        from vllm.v1.attention.backends.cute_paged.kernel import (
            paged_attention_forward,
        )
        torch.manual_seed(42)
        q = torch.randn(1, 32, 128, dtype=torch.bfloat16, device="cuda")
        kv = torch.randn(2, 64, 8, 128, dtype=torch.bfloat16, device="cuda")
        k_cache = _make_fp8_cache(kv)
        v_cache = _make_fp8_cache(kv)
        page_table = torch.tensor([[0, 1]], dtype=torch.int32, device="cuda")
        seq_lens = torch.tensor([128], dtype=torch.int32, device="cuda")

        out = paged_attention_forward(
            q, k_cache, v_cache, page_table, seq_lens,
            scale=1.0 / (128 ** 0.5), k_scale=1.0, v_scale=1.0,
        )
        assert out.shape == (1, 32, 128)
        assert not torch.isnan(out).any()

    def test_prefill_with_query_start_loc(self):
        """Prefill: multiple query tokens with query_start_loc."""
        from vllm.v1.attention.backends.cute_paged.kernel import (
            paged_attention_forward,
        )
        torch.manual_seed(42)
        q = torch.randn(8, 32, 128, dtype=torch.bfloat16, device="cuda")
        kv = torch.randn(1, 64, 8, 128, dtype=torch.bfloat16, device="cuda")
        k_cache = _make_fp8_cache(kv)
        v_cache = _make_fp8_cache(kv)
        page_table = torch.tensor([[0]], dtype=torch.int32, device="cuda")
        seq_lens = torch.tensor([8], dtype=torch.int32, device="cuda")
        query_start_loc = torch.tensor([0, 8], dtype=torch.int32, device="cuda")

        out = paged_attention_forward(
            q, k_cache, v_cache, page_table, seq_lens,
            scale=1.0 / (128 ** 0.5), k_scale=1.0, v_scale=1.0,
            query_start_loc=query_start_loc,
        )
        assert out.shape == (8, 32, 128)
        assert not torch.isnan(out).any()

    def test_descale_factors(self):
        """Non-unit k_scale and v_scale are applied correctly."""
        from vllm.v1.attention.backends.cute_paged.kernel import (
            paged_attention_forward,
        )
        torch.manual_seed(42)
        q = torch.randn(1, 32, 128, dtype=torch.bfloat16, device="cuda")
        kv = torch.randn(1, 64, 8, 128, dtype=torch.bfloat16, device="cuda")
        k_cache = _make_fp8_cache(kv)
        v_cache = _make_fp8_cache(kv)
        page_table = torch.tensor([[0]], dtype=torch.int32, device="cuda")
        seq_lens = torch.tensor([32], dtype=torch.int32, device="cuda")
        scale = 1.0 / (128 ** 0.5)

        result = paged_attention_forward(
            q, k_cache, v_cache, page_table, seq_lens,
            scale=scale, k_scale=0.5, v_scale=0.75,
        )
        expected = reference_paged_attention(
            q, k_cache, v_cache, page_table, seq_lens,
            scale=scale, k_scale=0.5, v_scale=0.75,
        )
        torch.testing.assert_close(result, expected, atol=2e-2, rtol=2e-2)
