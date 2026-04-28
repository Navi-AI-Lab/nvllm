"""Phase 1 verification harness for β-coop framework-output rewrite.

Empirically proves a custom op registered via direct_register_custom_op AND
added to _attention_ops splitting list runs eager Python on every decode
step (not once at capture). Mirrors tests/compile/silly_attention.py +
tests/compile/fullgraph/test_simple.py.

If this fails, the splitting-op-as-runtime-dispatch premise is wrong; do
not proceed to Phase 2.
"""

import pytest
import torch

from vllm.compilation.counter import compilation_counter
from vllm.config import (
    CompilationConfig,
    CompilationMode,
    CUDAGraphMode,
    VllmConfig,
)
from vllm.forward_context import BatchDescriptor, set_forward_context
from vllm.utils.torch_utils import direct_register_custom_op

# --- Test-local op (NOT registered into vllm:: namespace) -------------------
_test_lib = torch.library.Library("test_beta_skel", "FRAGMENT")
_global_counter: int = 0


def get_global_counter() -> int:
    return _global_counter


def reset_global_counter() -> None:
    global _global_counter
    _global_counter = 0


def skeleton_op(x: torch.Tensor, out: torch.Tensor) -> None:
    global _global_counter
    _global_counter += 1
    out.copy_(x + 1)


def skeleton_op_fake(x: torch.Tensor, out: torch.Tensor) -> None:
    return


direct_register_custom_op(
    op_name="skeleton_op",
    op_func=skeleton_op,
    mutates_args=["out"],
    fake_impl=skeleton_op_fake,
    target_lib=_test_lib,
)


# --- Module under test ------------------------------------------------------
class SkeletonModel(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out1 = torch.empty_like(x)
        torch.ops.test_beta_skel.skeleton_op(x, out1)
        out2 = torch.empty_like(out1)
        torch.ops.test_beta_skel.skeleton_op(out1, out2)
        return out2


# --- Test --------------------------------------------------------------------
def test_skeleton_counter_advances_per_replay():
    """Counter must advance N×2 (two op calls) per N replays per shape."""
    vllm_config = VllmConfig(
        compilation_config=CompilationConfig(
            mode=CompilationMode.VLLM_COMPILE,
            splitting_ops=["test_beta_skel::skeleton_op"],
            cudagraph_mode=CUDAGraphMode.PIECEWISE,
            cudagraph_capture_sizes=[1, 2],
        ),
    )

    model = SkeletonModel().cuda()
    model = torch.compile(model, fullgraph=False, dynamic=False)

    # Warm up + capture for both shapes.
    model(torch.randn(2).cuda())
    with set_forward_context(
        None, vllm_config=vllm_config,
        cudagraph_runtime_mode=CUDAGraphMode.PIECEWISE,
        batch_descriptor=BatchDescriptor(num_tokens=2),
    ):
        model(torch.randn(2).cuda())
    with set_forward_context(
        None, vllm_config=vllm_config,
        cudagraph_runtime_mode=CUDAGraphMode.PIECEWISE,
        batch_descriptor=BatchDescriptor(num_tokens=1),
    ):
        model(torch.randn(1).cuda())

    # Replay N=3 times per shape; expect counter to advance 3 * 2 = 6 per shape.
    reset_global_counter()
    for _ in range(3):
        with set_forward_context(
            None, vllm_config=vllm_config,
            cudagraph_runtime_mode=CUDAGraphMode.PIECEWISE,
            batch_descriptor=BatchDescriptor(num_tokens=2),
        ):
            model(torch.zeros(2).cuda())
    assert get_global_counter() == 6, (
        f"Expected counter=6 after 3 replays of size-2; got {get_global_counter()}. "
        "If this is much smaller (e.g. 2), the op body ran only at capture — "
        "splitting-op registration is failing."
    )

    reset_global_counter()
    for _ in range(3):
        with set_forward_context(
            None, vllm_config=vllm_config,
            cudagraph_runtime_mode=CUDAGraphMode.PIECEWISE,
            batch_descriptor=BatchDescriptor(num_tokens=1),
        ):
            model(torch.zeros(1).cuda())
    assert get_global_counter() == 6, (
        f"Expected counter=6 after 3 replays of size-1; got {get_global_counter()}."
    )
