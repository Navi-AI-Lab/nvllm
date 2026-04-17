# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tier-1 host-side smoke tests for nvllm Phase C owned layer classes.

Run:  .venv/bin/python notebooks/nvllm/layers_smoke_tests.py
Runs in ~5 s on CPU. No GPU required.

Tests:
  1. Qwen3_5RMSNorm block-level equivalence vs upstream GemmaRMSNorm(5120).
  2. Qwen3_5RMSNorm head-dim equivalence vs upstream GemmaRMSNorm(256).
  3. Qwen3_5RMSNorm fused-residual equivalence vs upstream GemmaRMSNorm.
  4. CustomOp registry: 'qwen3_5_rms_norm' exists and does not clobber
     'gemma_rms_norm'.

Notes:
  - Qwen3_5MLP vs Qwen2MoeMLP equivalence test is SKIPPED here because
    MergedColumnParallelLinear / RowParallelLinear require an initialized
    distributed backend (torch.distributed). Shape-only smoke test is
    included. Full correctness is covered by Tier-3 GSM8K 8/8.
"""

import sys

import torch

from vllm.config import VllmConfig, set_current_vllm_config
from vllm.model_executor.custom_op import op_registry
from vllm.model_executor.layers.layernorm import GemmaRMSNorm
from vllm.nvllm.layers.layernorm import Qwen3_5RMSNorm
from vllm.nvllm.layers.mlp import Qwen3_5MLP


def _seed_weights(rms_upstream: GemmaRMSNorm, rms_owned: Qwen3_5RMSNorm) -> None:
    """Copy identical weights into both norms so outputs can be compared."""
    torch.manual_seed(42)
    w = torch.randn_like(rms_upstream.weight.data)
    rms_upstream.weight.data.copy_(w)
    rms_owned.weight.data.copy_(w)


def test_rms_block_level_equivalence() -> None:
    """Block-level (hidden_size=5120) equivalence — no residual."""
    hidden = 5120
    upstream = GemmaRMSNorm(hidden, eps=1e-6)
    owned = Qwen3_5RMSNorm(hidden, eps=1e-6)
    _seed_weights(upstream, owned)

    torch.manual_seed(0)
    x = torch.randn(8, hidden, dtype=torch.bfloat16)
    y_up = upstream(x.clone())
    y_own = owned(x.clone())

    torch.testing.assert_close(y_own, y_up, rtol=0, atol=0)
    print("  [PASS] test_rms_block_level_equivalence")


def test_rms_head_dim_equivalence() -> None:
    """Head-dim (hidden_size=256) equivalence — the q_norm / k_norm shape."""
    hidden = 256
    upstream = GemmaRMSNorm(hidden, eps=1e-6)
    owned = Qwen3_5RMSNorm(hidden, eps=1e-6)
    _seed_weights(upstream, owned)

    torch.manual_seed(1)
    x = torch.randn(8, 24, hidden, dtype=torch.bfloat16)
    y_up = upstream(x.clone())
    y_own = owned(x.clone())

    torch.testing.assert_close(y_own, y_up, rtol=0, atol=0)
    print("  [PASS] test_rms_head_dim_equivalence")


def test_rms_fused_residual_equivalence() -> None:
    """Block-level fused-add-residual forward equivalence."""
    hidden = 5120
    upstream = GemmaRMSNorm(hidden, eps=1e-6)
    owned = Qwen3_5RMSNorm(hidden, eps=1e-6)
    _seed_weights(upstream, owned)

    torch.manual_seed(2)
    x = torch.randn(8, hidden, dtype=torch.bfloat16)
    r = torch.randn(8, hidden, dtype=torch.bfloat16)

    y_up, r_up = upstream(x.clone(), r.clone())
    y_own, r_own = owned(x.clone(), r.clone())

    torch.testing.assert_close(y_own, y_up, rtol=0, atol=0)
    torch.testing.assert_close(r_own, r_up, rtol=0, atol=0)
    print("  [PASS] test_rms_fused_residual_equivalence")


def test_customop_registry_no_collision() -> None:
    """Both 'qwen3_5_rms_norm' and 'gemma_rms_norm' must coexist."""
    assert "qwen3_5_rms_norm" in op_registry, "owned CustomOp name missing"
    assert "gemma_rms_norm" in op_registry, "upstream CustomOp name was clobbered"
    assert op_registry["qwen3_5_rms_norm"] is Qwen3_5RMSNorm
    assert op_registry["gemma_rms_norm"] is GemmaRMSNorm
    print("  [PASS] test_customop_registry_no_collision")


def test_mlp_class_surface() -> None:
    """Qwen3_5MLP constructor surface matches spec (no expert_gate / reduce_results)."""
    import inspect

    params = list(inspect.signature(Qwen3_5MLP.__init__).parameters)
    assert "expert_gate" not in params, "expert_gate should have been dropped"
    assert "reduce_results" not in params, "reduce_results should have been dropped"
    # Expected surface:
    assert params == [
        "self",
        "hidden_size",
        "intermediate_size",
        "hidden_act",
        "quant_config",
        "prefix",
    ], f"unexpected Qwen3_5MLP signature: {params}"
    print("  [PASS] test_mlp_class_surface")


def main() -> int:
    print("Running Tier-1 smoke tests for nvllm Phase C owned layers:")
    tests = [
        test_rms_block_level_equivalence,
        test_rms_head_dim_equivalence,
        test_rms_fused_residual_equivalence,
        test_customop_registry_no_collision,
        test_mlp_class_surface,
    ]
    # CustomOp.__init__ calls get_current_vllm_config(); wrap the whole run
    # in a bare VllmConfig context so module-import-time instantiations work
    # without standing up a full engine. Same pattern as tests/conftest.py
    # `default_vllm_config` fixture.
    failed = 0
    with set_current_vllm_config(VllmConfig()):
        for fn in tests:
            try:
                fn()
            except Exception as exc:  # noqa: BLE001
                print(f"  [FAIL] {fn.__name__}: {exc}")
                failed += 1
    total = len(tests)
    print(f"\nResult: {total - failed}/{total} passed.")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
