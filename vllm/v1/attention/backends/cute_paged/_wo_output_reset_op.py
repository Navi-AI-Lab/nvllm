# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""β-coop wo_output captured pre-launch reset op.

Zeros `_phase_e_coop_wo_output[:nat]` via a captured cudaMemsetAsync
graph node before each `run_beta_coop_full` launch. Solves the v1
"stale content at stable address" failure under FULL_AND_PIECEWISE
(see docs/superpowers/specs/2026-04-30-beta-coop-persistent-buffers-v2-design.md).

libcudart loading is lazy (first op call) so module import never hard-
fails on libcudart absence; β-coop callers see a clear RuntimeError
naming all candidates if the lookup fails.

Per feedback_op_body_capture_only: op body runs once at FULL-graph
capture; the cudaMemsetAsync issued on the current capture stream
becomes a graph node and replays at every FULL-graph replay.
The call runs inside the eager body of the existing cute_beta_coop_run
splitting boundary, so Dynamo/FX does not trace through or DCE it.
mutates_args keeps the op schema honest if it is ever traced elsewhere.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import os

import torch

from vllm.utils.torch_utils import direct_register_custom_op

_LOG_ENABLED: bool = os.environ.get("CUTE_WO_RESET_LOG", "0") == "1"
_LOGGED_PAIRS: set[tuple[int, int]] = set()

_libcudart: ctypes.CDLL | None = None
_cudaMemsetAsync = None  # type: ignore[var-annotated]
_cudaGetErrorString = None  # type: ignore[var-annotated]


def _ensure_libcudart_loaded() -> None:
    """Lazy bind on first op call. find_library → soname-12 → soname.

    Failure mode: clear RuntimeError naming all candidates, raised
    only when β-coop is exercised (not at module import).
    """
    global _libcudart, _cudaMemsetAsync, _cudaGetErrorString
    if _libcudart is not None:
        return
    candidates = [
        ctypes.util.find_library("cudart"),
        "libcudart.so.12",
        "libcudart.so",
    ]
    last_err: Exception | None = None
    for cand in candidates:
        if cand is None:
            continue
        try:
            _libcudart = ctypes.CDLL(cand)
            break
        except OSError as e:
            last_err = e
    if _libcudart is None:
        raise RuntimeError(
            f"cute_paged_reset_wo_output: failed to load libcudart "
            f"(tried {candidates!r}): {last_err}"
        )
    _cudaMemsetAsync = _libcudart.cudaMemsetAsync
    _cudaMemsetAsync.argtypes = [
        ctypes.c_void_p,   # devPtr
        ctypes.c_int,      # value (byte)
        ctypes.c_size_t,   # count (bytes)
        ctypes.c_void_p,   # cudaStream_t
    ]
    _cudaMemsetAsync.restype = ctypes.c_int  # cudaError_t
    _cudaGetErrorString = _libcudart.cudaGetErrorString
    _cudaGetErrorString.argtypes = [ctypes.c_int]
    _cudaGetErrorString.restype = ctypes.c_char_p


def cute_paged_reset_wo_output(
    wo_output: torch.Tensor,
    nat: int,
) -> None:
    """Zero `wo_output[:nat]` in-place via captured cudaMemsetAsync.

    Preconditions enforce the expected layout — a future buffer-shape
    change must not turn into a silent byte memset of the wrong region
    (per feedback_no_silent_fallbacks).
    """
    assert wo_output.is_cuda, "wo_output must be CUDA"
    assert wo_output.dtype == torch.float32, (
        f"wo_output dtype must be float32, got {wo_output.dtype}"
    )
    assert wo_output.dim() == 3, (
        f"wo_output must be 3D [max_num_seqs, num_kv_heads*wo_split, hidden], "
        f"got {tuple(wo_output.shape)}"
    )
    assert wo_output.is_contiguous(), (
        "wo_output must be contiguous (slice [:nat] depends on dim-0 "
        "stride)"
    )
    assert 0 <= nat <= wo_output.shape[0], (
        f"nat={nat} out of range for wo_output.shape[0]="
        f"{wo_output.shape[0]}"
    )
    if nat == 0:
        return  # legal under PIECEWISE no-op paths

    if _LOG_ENABLED:
        key = (wo_output.data_ptr(), nat)
        if key not in _LOGGED_PAIRS:
            _LOGGED_PAIRS.add(key)
            print(
                f"[CUTE_WO_RESET] data_ptr={key[0]:#x} nat={nat} "
                f"shape={tuple(wo_output.shape)}",
                flush=True,
            )

    _ensure_libcudart_loaded()
    nbytes = (
        nat * wo_output.shape[1] * wo_output.shape[2]
        * wo_output.element_size()
    )
    stream_handle = int(torch.cuda.current_stream().cuda_stream)
    err = _cudaMemsetAsync(
        wo_output.data_ptr(), 0, nbytes, stream_handle
    )
    if err != 0:
        msg = _cudaGetErrorString(err)
        msg_str = msg.decode("utf-8", errors="replace") if msg else "?"
        raise RuntimeError(
            f"cudaMemsetAsync failed: cudaError={err} ({msg_str}) "
            f"nbytes={nbytes} nat={nat} stream={stream_handle:#x}"
        )


def cute_paged_reset_wo_output_fake(
    wo_output: torch.Tensor,
    nat: int,
) -> None:
    return None


direct_register_custom_op(
    op_name="cute_paged_reset_wo_output",
    op_func=cute_paged_reset_wo_output,
    mutates_args=["wo_output"],
    fake_impl=cute_paged_reset_wo_output_fake,
)
