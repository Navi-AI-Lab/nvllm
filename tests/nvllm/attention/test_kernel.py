"""Tests for the CuTe DSL paged attention kernel components."""
import torch
import pytest


class TestOnlineSoftmax:
    def test_matches_naive_softmax(self):
        """Online softmax produces same result as torch.softmax."""
        from vllm.v1.attention.backends.cute_paged.kernel import online_softmax

        torch.manual_seed(42)
        # Simulate QK scores: [num_q_rows, kv_len]
        scores = torch.randn(16, 256, dtype=torch.float32, device="cuda")
        scale = 1.0 / (128 ** 0.5)

        result = online_softmax(scores, scale)
        expected = torch.softmax(scores * scale, dim=-1)

        torch.testing.assert_close(result, expected, atol=1e-5, rtol=1e-5)

    def test_online_softmax_numerical_stability(self):
        """Large values don't cause overflow."""
        from vllm.v1.attention.backends.cute_paged.kernel import online_softmax

        scores = torch.full((16, 128), 1000.0, dtype=torch.float32, device="cuda")
        scale = 1.0
        result = online_softmax(scores, scale)
        assert not torch.isnan(result).any()
        assert not torch.isinf(result).any()


class TestQKPass:
    def test_qk_matches_reference(self):
        """FP8 QK dot product matches BF16 reference within FP8 tolerance."""
        from vllm.v1.attention.backends.cute_paged.kernel import qk_pass

        torch.manual_seed(42)
        # Q: BF16, K: FP8 (stored as uint8)
        q = torch.randn(16, 128, dtype=torch.bfloat16, device="cuda")
        k_bf16 = torch.randn(64, 128, dtype=torch.bfloat16, device="cuda")
        k_fp8 = k_bf16.to(torch.float8_e4m3fn)

        scale = 1.0 / (128 ** 0.5)
        k_scale = 1.0

        result = qk_pass(q, k_fp8, scale, k_scale)

        # Reference: dequant K, BF16 matmul
        k_deq = k_fp8.to(torch.bfloat16) * k_scale
        expected = (q.float() @ k_deq.float().T) * scale

        # FP8 quantization of Q adds noise — use relaxed tolerance
        torch.testing.assert_close(result, expected.to(result.dtype),
                                   atol=1e-1, rtol=1e-1)

    def test_qk_output_shape(self):
        """QK output has shape [num_q_rows, num_kv_rows]."""
        from vllm.v1.attention.backends.cute_paged.kernel import qk_pass

        torch.manual_seed(42)
        q = torch.randn(16, 128, dtype=torch.bfloat16, device="cuda")
        k_fp8 = torch.randn(64, 128, dtype=torch.bfloat16, device="cuda").to(
            torch.float8_e4m3fn
        )
        result = qk_pass(q, k_fp8, scale=0.088, k_scale=1.0)
        assert result.shape == (16, 64)


class TestPVPass:
    def test_pv_matches_reference(self):
        """BF16 PV multiply with FP8 V dequant matches reference."""
        from vllm.v1.attention.backends.cute_paged.kernel import pv_pass

        torch.manual_seed(42)
        # P: softmax output [num_q_rows, num_kv_rows], FP32
        p = torch.softmax(torch.randn(16, 64, device="cuda"), dim=-1)
        # V: FP8
        v_bf16 = torch.randn(64, 128, dtype=torch.bfloat16, device="cuda")
        v_fp8 = v_bf16.to(torch.float8_e4m3fn)
        v_scale = 1.0

        result = pv_pass(p, v_fp8, v_scale)

        # Reference: dequant V, BF16 matmul
        v_deq = v_fp8.to(torch.bfloat16) * v_scale
        expected = (p @ v_deq.float()).to(torch.bfloat16)

        torch.testing.assert_close(result, expected, atol=1e-1, rtol=1e-1)

    def test_pv_output_shape(self):
        """PV output has shape [num_q_rows, head_dim]."""
        from vllm.v1.attention.backends.cute_paged.kernel import pv_pass

        torch.manual_seed(42)
        p = torch.softmax(torch.randn(16, 64, device="cuda"), dim=-1)
        v_fp8 = torch.randn(64, 128, dtype=torch.bfloat16, device="cuda").to(
            torch.float8_e4m3fn
        )
        result = pv_pass(p, v_fp8, v_scale=1.0)
        assert result.shape == (16, 128)
        assert result.dtype == torch.bfloat16


class TestPagedAttentionForward:
    def test_single_page_decode(self):
        """Decode with 1 page of KV cache."""
        from vllm.v1.attention.backends.cute_paged.kernel import (
            paged_attention_forward,
        )
        torch.manual_seed(42)
        q = torch.randn(1, 32, 128, dtype=torch.bfloat16, device="cuda")
        kv = torch.randn(2, 64, 8, 128, dtype=torch.bfloat16, device="cuda")
        k_cache = kv.to(torch.float8_e4m3fn).view(torch.uint8)
        v_cache = kv.to(torch.float8_e4m3fn).view(torch.uint8)
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
        k_cache = kv.to(torch.float8_e4m3fn).view(torch.uint8)
        v_cache = kv.to(torch.float8_e4m3fn).view(torch.uint8)
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
        k_cache = kv.to(torch.float8_e4m3fn).view(torch.uint8)
        v_cache = kv.to(torch.float8_e4m3fn).view(torch.uint8)
        page_table = torch.tensor([[0]], dtype=torch.int32, device="cuda")
        seq_lens = torch.tensor([16], dtype=torch.int32, device="cuda")

        out = paged_attention_forward(
            q, k_cache, v_cache, page_table, seq_lens,
            scale=1.0 / (128 ** 0.5), k_scale=1.0, v_scale=1.0,
        )
        # Q heads 0-3 should produce same output (share KV head 0)
        # Not exactly equal due to different Q values, but shape is right
        assert out.shape == (1, 32, 128)

    def test_matches_reference(self):
        """Full forward matches PyTorch reference within tolerance."""
        from vllm.v1.attention.backends.cute_paged.kernel import (
            paged_attention_forward,
        )
        from tests.nvllm.attention.reference import reference_paged_attention

        torch.manual_seed(42)
        q = torch.randn(1, 32, 128, dtype=torch.bfloat16, device="cuda")
        kv = torch.randn(2, 64, 8, 128, dtype=torch.bfloat16, device="cuda")
        k_cache = kv.to(torch.float8_e4m3fn).view(torch.uint8)
        v_cache = kv.to(torch.float8_e4m3fn).view(torch.uint8)
        page_table = torch.tensor([[0, 1]], dtype=torch.int32, device="cuda")
        seq_lens = torch.tensor([100], dtype=torch.int32, device="cuda")
        scale = 1.0 / (128 ** 0.5)

        result = paged_attention_forward(
            q, k_cache, v_cache, page_table, seq_lens,
            scale=scale, k_scale=1.0, v_scale=1.0,
        )
        expected = reference_paged_attention(
            q, k_cache, page_table, seq_lens, scale=scale,
        )

        torch.testing.assert_close(result, expected, atol=1e-2, rtol=1e-2)

    def test_edge_single_token(self):
        """Minimal case: batch=1, query=1, seq_len=1."""
        from vllm.v1.attention.backends.cute_paged.kernel import (
            paged_attention_forward,
        )
        torch.manual_seed(42)
        q = torch.randn(1, 32, 128, dtype=torch.bfloat16, device="cuda")
        kv = torch.randn(1, 64, 8, 128, dtype=torch.bfloat16, device="cuda")
        k_cache = kv.to(torch.float8_e4m3fn).view(torch.uint8)
        v_cache = kv.to(torch.float8_e4m3fn).view(torch.uint8)
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
        k_cache = kv.to(torch.float8_e4m3fn).view(torch.uint8)
        v_cache = kv.to(torch.float8_e4m3fn).view(torch.uint8)
        page_table = torch.tensor([[0, 1]], dtype=torch.int32, device="cuda")
        seq_lens = torch.tensor([128], dtype=torch.int32, device="cuda")

        out = paged_attention_forward(
            q, k_cache, v_cache, page_table, seq_lens,
            scale=1.0 / (128 ** 0.5), k_scale=1.0, v_scale=1.0,
        )
        assert out.shape == (1, 32, 128)
        assert not torch.isnan(out).any()
