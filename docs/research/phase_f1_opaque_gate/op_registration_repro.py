"""Phase F.1 op-registration repro — verify direct_register_custom_op
mechanics before adding the real ops to _mlp_op.py.

Tests:
- fake_impl signature matches real op signature
- mutates_args correctly pins outputs (no unnecessary copies)
- Nested torch.ops.* call from inside custom-op body works under
  torch.compile
- str layer_name threads correctly through registry lookup

Run: .venv/bin/python docs/research/phase_f1_opaque_gate/op_registration_repro.py
Expected: prints "ALL CHECKS PASSED"; exits 0.

Per memory:feedback_kernel_repro_before_rebuild — catch op-registration
bugs in seconds, not a 30-minute Docker rebuild.
"""
import torch
from vllm.utils.torch_utils import direct_register_custom_op

# Module-level registry mimicking _CUTE_MLP_REGISTRY
_TEST_REGISTRY: dict[str, dict] = {}


def _dummy_sub_impl(x: torch.Tensor, out: torch.Tensor, name: str) -> None:
    """Mimics cute_mlp_forward — writes x * 2 into out."""
    state = _TEST_REGISTRY[name]
    state["sub_fires"] += 1
    out.copy_(x * 2)


def _dummy_sub_fake(x, out, name):
    return None


direct_register_custom_op(
    op_name="_test_dummy_sub",
    op_func=_dummy_sub_impl,
    mutates_args=["out"],
    fake_impl=_dummy_sub_fake,
)


def _dummy_dispatch_impl(
    x: torch.Tensor,
    hidden_out: torch.Tensor,
    residual_out: torch.Tensor,
    residual_in: torch.Tensor,
    name: str,
) -> None:
    """Mimics cute_phase_e_dispatch — branches on runtime flag,
    can call another custom op (nested dispatch test).
    """
    state = _TEST_REGISTRY[name]
    state["dispatch_fires"] += 1
    if state["consumed"]:
        # Consumed branch: copy from "scratch".
        hidden_out.copy_(state["scratch_hidden"])
        residual_out.copy_(state["scratch_residual"])
        state["consumed"] = False
        return
    # Not-consumed: call the sibling op (nested dispatch).
    torch.ops.vllm._test_dummy_sub(x, hidden_out, name)
    residual_out.copy_(residual_in)


def _dummy_dispatch_fake(x, hidden_out, residual_out, residual_in, name):
    return None


direct_register_custom_op(
    op_name="_test_dummy_dispatch",
    op_func=_dummy_dispatch_impl,
    mutates_args=["hidden_out", "residual_out"],
    fake_impl=_dummy_dispatch_fake,
)


def run_checks() -> None:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    _TEST_REGISTRY["L0"] = {
        "consumed": False,
        "scratch_hidden": torch.full((4, 8), 7.0, device=device),
        "scratch_residual": torch.full((4, 8), 9.0, device=device),
        "sub_fires": 0,
        "dispatch_fires": 0,
    }

    x = torch.full((4, 8), 3.0, device=device)
    residual_in = torch.full((4, 8), 5.0, device=device)
    hidden_out = torch.empty_like(x)
    residual_out = torch.empty_like(residual_in)

    # Check 1: Not-consumed branch fires sub-op nested.
    torch.ops.vllm._test_dummy_dispatch(
        x, hidden_out, residual_out, residual_in, "L0"
    )
    assert _TEST_REGISTRY["L0"]["dispatch_fires"] == 1
    assert _TEST_REGISTRY["L0"]["sub_fires"] == 1
    assert torch.allclose(hidden_out, x * 2), (
        f"not-consumed branch failed: got {hidden_out}, expected {x*2}"
    )
    assert torch.allclose(residual_out, residual_in), (
        "residual passthrough failed"
    )

    # Check 2: Consumed branch does NOT fire sub-op.
    _TEST_REGISTRY["L0"]["consumed"] = True
    torch.ops.vllm._test_dummy_dispatch(
        x, hidden_out, residual_out, residual_in, "L0"
    )
    assert _TEST_REGISTRY["L0"]["dispatch_fires"] == 2
    assert _TEST_REGISTRY["L0"]["sub_fires"] == 1, (
        "consumed branch fired sub-op — nested dispatch bled through"
    )
    assert torch.allclose(hidden_out, _TEST_REGISTRY["L0"]["scratch_hidden"])
    assert torch.allclose(
        residual_out, _TEST_REGISTRY["L0"]["scratch_residual"]
    )

    # Check 3: Flag cleared after consume.
    assert _TEST_REGISTRY["L0"]["consumed"] is False

    # Check 4: Under torch.compile, both branches still work.
    @torch.compile(fullgraph=True, dynamic=False)
    def compiled_fn(x, hidden_out, residual_out, residual_in, name):
        torch.ops.vllm._test_dummy_dispatch(
            x, hidden_out, residual_out, residual_in, name
        )
        return hidden_out, residual_out

    # Not-consumed under compile
    _TEST_REGISTRY["L0"]["consumed"] = False
    h, r = compiled_fn(x, hidden_out, residual_out, residual_in, "L0")
    assert torch.allclose(h, x * 2)
    # Consumed under compile — flag set AFTER trace
    _TEST_REGISTRY["L0"]["consumed"] = True
    h, r = compiled_fn(x, hidden_out, residual_out, residual_in, "L0")
    assert torch.allclose(h, _TEST_REGISTRY["L0"]["scratch_hidden"]), (
        "COMPILED consumed branch did not fire — opaque-op is not opaque"
    )

    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    run_checks()
