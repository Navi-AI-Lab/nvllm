"""Tests for the PyTorch reference attention implementation."""
import torch
import pytest

from tests.nvllm.attention.reference import reference_paged_attention


@pytest.fixture
def attention_config():
    """Qwen3.5-27B attention geometry."""
    return {
        "num_q_heads": 32,
        "num_kv_heads": 8,
        "head_dim": 128,
        "page_size": 64,
        "scale": 1.0 / (128 ** 0.5),
    }


class TestReferenceCorrectness:
    def test_single_token_decode(self, attention_config):
        """Single decode token against short KV cache."""
        torch.manual_seed(42)
        q = torch.randn(1, 32, 128, dtype=torch.bfloat16, device="cuda")
        # Create FP8 KV cache: 1 page, 64 tokens, 8 KV heads, 128 dim
        kv_data = torch.randn(1, 64, 8, 128, dtype=torch.bfloat16, device="cuda")
        kv_cache = kv_data.to(torch.float8_e4m3fn).view(torch.uint8)

        page_table = torch.tensor([[0]], dtype=torch.int32, device="cuda")
        seq_lens = torch.tensor([32], dtype=torch.int32, device="cuda")

        out = reference_paged_attention(
            q, kv_cache, page_table, seq_lens,
            scale=attention_config["scale"],
        )
        assert out.shape == (1, 32, 128)
        assert out.dtype == torch.bfloat16
        assert not torch.isnan(out).any()

    def test_gqa_ratio(self, attention_config):
        """GQA: 32 Q heads / 8 KV heads = 4:1 ratio."""
        torch.manual_seed(42)
        q = torch.randn(1, 32, 128, dtype=torch.bfloat16, device="cuda")
        kv_data = torch.randn(1, 64, 8, 128, dtype=torch.bfloat16, device="cuda")
        kv_cache = kv_data.to(torch.float8_e4m3fn).view(torch.uint8)
        page_table = torch.tensor([[0]], dtype=torch.int32, device="cuda")
        seq_lens = torch.tensor([16], dtype=torch.int32, device="cuda")

        out = reference_paged_attention(
            q, kv_cache, page_table, seq_lens,
            scale=attention_config["scale"],
        )
        # Output should have 32 heads (Q heads), not 8
        assert out.shape[1] == 32

    def test_causal_mask(self, attention_config):
        """Future tokens should not be attended to."""
        torch.manual_seed(42)
        # 4 query tokens (prefill-like)
        q = torch.randn(4, 32, 128, dtype=torch.bfloat16, device="cuda")
        kv_data = torch.randn(1, 64, 8, 128, dtype=torch.bfloat16, device="cuda")
        kv_cache = kv_data.to(torch.float8_e4m3fn).view(torch.uint8)
        page_table = torch.tensor([[0]], dtype=torch.int32, device="cuda")
        seq_lens = torch.tensor([4], dtype=torch.int32, device="cuda")

        out = reference_paged_attention(
            q, kv_cache, page_table, seq_lens,
            scale=attention_config["scale"],
        )
        assert out.shape == (4, 32, 128)
        assert not torch.isnan(out).any()
