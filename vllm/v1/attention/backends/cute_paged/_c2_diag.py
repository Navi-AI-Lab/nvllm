# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""C2 diagnostic probe — β-coop vs legacy comparison harness.

Spec: docs/research/uber_kernel_migration/2026-04-26-c2-diagnostic-spec.md.

Env-gated. When CUTE_C2_DIAG=1, compares β-coop's outputs (impl.rmsnorm_output,
impl.residual_output) against the legacy Python path's outputs (hidden_states,
residual after post_attention_layernorm) at every full-attn layer. On first
divergence above tolerance, dumps a forensics bundle and raises RuntimeError.

The module is import-safe regardless of CUTE_C2_DIAG setting; the call site in
qwen3_5.py guards every entry point with `os.getenv("CUTE_C2_DIAG") == "1"`.

Call-site constraint: `compare_and_log` does ~6 host-device syncs per call
(`.item()` inside `_compare_pair`). It MUST be invoked outside CUDA-graph-
captured regions, or capture will fail with cudaErrorStreamCaptureInvalidated
(see memory:feedback_item_breaks_cuda_graphs). The qwen3_5.py call site lives
in eager Python between captured segments — keep it there.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import torch


def _compare_pair(
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    atol: float,
    rtol: float,
) -> dict:
    """Compare two BF16 tensors element-wise.

    Returns a dict with:
      - linf: float, max absolute difference
      - rel_med: float, median |a-b| / (|a| + 1e-9)
      - ok:    bool, True iff every element within (atol + rtol * |a|)

    Computes everything in FP32 to avoid BF16-roundoff perturbing the stats.
    """
    a32 = a.float()
    b32 = b.float()
    diff = (a32 - b32).abs()
    linf = diff.max().item()
    rel = diff / (a32.abs() + 1e-9)
    rel_med = rel.median().item()
    tol = atol + rtol * a32.abs()
    ok = bool((diff <= tol).all().item())
    return {"linf": linf, "rel_med": rel_med, "ok": ok}


def _dump_on_divergence(
    *,
    layer_idx: int,
    step_idx: int,
    nat: int,
    atol: float,
    rtol: float,
    legacy_hidden: torch.Tensor,
    legacy_residual: torch.Tensor,
    beta_rmsnorm_output: torch.Tensor,
    beta_residual_output: torch.Tensor,
) -> Path:
    """Write a torch.save bundle for offline forensics. Returns the path."""
    dump_dir = Path(os.getenv("CUTE_C2_DIAG_DUMP_DIR") or "/tmp/c2_diag")
    dump_dir.mkdir(parents=True, exist_ok=True)
    dump_path = dump_dir / f"layer{layer_idx}_step{step_idx}.pt"
    bundle = {
        "layer_idx": layer_idx,
        "step_idx": step_idx,
        "nat": nat,
        "atol": atol,
        "rtol": rtol,
        "legacy_hidden": legacy_hidden[:nat].detach().clone().cpu(),
        "legacy_residual": legacy_residual[:nat].detach().clone().cpu(),
        "beta_rmsnorm_output": beta_rmsnorm_output[:nat].detach().clone().cpu(),
        "beta_residual_output": beta_residual_output[:nat].detach().clone().cpu(),
    }
    torch.save(bundle, dump_path)
    return dump_path


_STEP_COUNTER: int = 0


def next_step_idx() -> int:
    """Return a monotonically increasing step index (0, 1, 2, ...).

    Imperfect: only used for log readability. The caller is expected to
    invoke this once per layer-0 call so per-step grouping in stderr is
    legible. If linear-attn-only layers fire (rare in production), the
    counter may skip; that's fine — the layer index in the same log line
    disambiguates.
    """
    global _STEP_COUNTER
    idx = _STEP_COUNTER
    _STEP_COUNTER += 1
    return idx


def _reset_step_counter_for_test() -> None:
    """Reset the module-level step counter. Tests only."""
    global _STEP_COUNTER
    _STEP_COUNTER = 0


def assert_no_flashinfer_autotune(kernel_config) -> None:
    """Refuse to run if flashinfer autotune is enabled.

    Per memory:feedback_flashinfer_autotune_sm120, autotune on SM120 can
    cause the host to hard-reboot during kernel selection. The C2
    diagnostic must never trigger this. serve-cute already bakes
    --kernel-config '{"enable_flashinfer_autotune":false}' (commit
    2b21f3450); this assert is a belt-and-suspenders check.

    Caller passes the live kernel_config object (e.g., from
    vllm.config.get_current_vllm_config().kernel_config).
    """
    if getattr(kernel_config, "enable_flashinfer_autotune", False):
        raise RuntimeError(
            "[C2_DIAG] refuses to run with flashinfer autotune enabled "
            "— host reboot risk on SM120. Pass "
            "--kernel-config '{\"enable_flashinfer_autotune\":false}' "
            "or unset the env var that enabled it."
        )


def _inject_noise(t: torch.Tensor) -> torch.Tensor:
    """Add a constant offset to t when CUTE_C2_DIAG_INJECT_NOISE is set.

    Used by Phase-3 self-test: verifies the probe halts when divergence
    is forced, ensuring the comparison and dump paths actually fire on
    real divergence (not just rejected by tolerance).

    Returns t unchanged when the env var is unset. Raises ValueError on
    non-float values (don't silently fall back per feedback_no_silent_fallbacks).
    """
    raw = os.getenv("CUTE_C2_DIAG_INJECT_NOISE")
    if not raw:  # handles both unset (None) and set-but-empty ("")
        return t
    offset = float(raw)  # raises ValueError on non-float per spec
    return t + offset


def compare_and_log(
    *,
    layer_idx: int,
    step_idx: int,
    nat: int,
    legacy_hidden: torch.Tensor,
    legacy_residual: torch.Tensor,
    beta_rmsnorm_output: torch.Tensor,
    beta_residual_output: torch.Tensor,
) -> None:
    """Compare β-coop's outputs vs the legacy path's outputs.

    Reads tolerances from CUTE_C2_DIAG_TOL_ATOL / _RTOL (defaults 1e-2).
    Logs one stderr line per call. On first divergence above tolerance,
    dumps a forensics bundle to CUTE_C2_DIAG_DUMP_DIR and raises
    RuntimeError. nat=0 (empty decode) is skipped silently.

    Wrapped via torch.ops.vllm.cute_c2_diag_compare (registered below):
    vLLM compiles the decoder forward end-to-end with fullgraph=True
    under cudagraph_mode=PIECEWISE, so this call site is reached during
    Dynamo trace. compare_and_log itself cannot be traced — Dynamo
    cannot reason about `tensor[:nat]` when `nat` is dynamic and operand
    shapes don't match (legacy_hidden has prefill-dim s18, impl.rmsnorm
    is max-num-seqs-sized). The custom op pattern (with explicit
    fake_impl returning None) keeps the call in the FX graph as opaque
    without symbolic-introspecting the body. The body runs in eager at
    runtime with real tensors. `@torch._dynamo.disable` would be cleaner
    but is rejected under fullgraph=True per upstream
    pytorch/pytorch#167927; `torch.compiler.allow_in_graph` is also
    insufficient (Dynamo still fake-executes the call).
    """
    if nat == 0:
        return

    # `or "1e-2"` handles both unset (None) and set-but-empty ("") env;
    # plain `getenv(name, default)` returns "" not the default when the var
    # is set to "" (which serve-cute.sh does via `-e VAR=""` for unset vars).
    atol = float(os.getenv("CUTE_C2_DIAG_TOL_ATOL") or "1e-2")
    rtol = float(os.getenv("CUTE_C2_DIAG_TOL_RTOL") or "1e-2")

    # Self-test injection (CUTE_C2_DIAG_INJECT_NOISE=1.0 forces divergence).
    beta_h = _inject_noise(beta_rmsnorm_output[:nat])

    h = _compare_pair(legacy_hidden[:nat], beta_h, atol=atol, rtol=rtol)
    r = _compare_pair(
        legacy_residual[:nat],
        beta_residual_output[:nat],
        atol=atol,
        rtol=rtol,
    )

    verdict = "OK" if (h["ok"] and r["ok"]) else "DIVERGED"
    print(
        f"[C2_DIAG] step={step_idx} L={layer_idx} nat={nat}  "
        f"hidden  L∞={h['linf']:.2e} rel_med={h['rel_med']:.2e}  "
        f"residual  L∞={r['linf']:.2e} rel_med={r['rel_med']:.2e}  "
        f"{verdict}",
        file=sys.stderr,
        flush=True,
    )

    if not (h["ok"] and r["ok"]):
        dump_path = _dump_on_divergence(
            layer_idx=layer_idx,
            step_idx=step_idx,
            nat=nat,
            atol=atol,
            rtol=rtol,
            legacy_hidden=legacy_hidden,
            legacy_residual=legacy_residual,
            beta_rmsnorm_output=beta_rmsnorm_output,
            beta_residual_output=beta_residual_output,
        )
        raise RuntimeError(
            f"[C2_DIAG] diverged: layer={layer_idx} step={step_idx} "
            f"hidden L∞={h['linf']:.2e} residual L∞={r['linf']:.2e}  "
            f"dump={dump_path}"
        )


# --- Custom op wrapper for use inside torch.compile / fullgraph -----------
# Direct register the diag as a vLLM custom op (mirrors cute_residual_mirror
# in _mlp_op.py). This makes it Dynamo-opaque: the FX graph records the call
# but never fake-executes the body. At runtime the real impl runs in eager.
#
# mutates_args=["legacy_hidden"] is a defensive declaration (we don't actually
# mutate any input — but per memory:feedback_mutates_args_not_dce_safe, ops
# without declared mutation can be DCE'd if return value is unused).
# legacy_hidden is the model's hidden_states, which IS read by downstream
# layers, so this declaration anchors the op in the graph.
#
# Capture-skip: under cudagraph_mode=PIECEWISE, the op call may land inside
# a captured graph segment. compare_and_log uses .item() (host-device sync)
# which would invalidate capture per memory:feedback_item_breaks_cuda_graphs.
# Skip when capturing to avoid cudaErrorStreamCaptureInvalidated. (Side
# effect: diag won't fire at captured batch sizes [1,2,4,8] during the
# warmup capture phase, but DOES fire at runtime replay since replay just
# replays kernels — the op-call FX node is part of the captured kernel
# launch sequence, not re-executed at the Python level.)


def _cute_c2_diag_compare_impl(
    layer_idx: int,
    step_idx: int,
    nat: int,
    legacy_hidden: torch.Tensor,
    legacy_residual: torch.Tensor,
    beta_rmsnorm_output: torch.Tensor,
    beta_residual_output: torch.Tensor,
) -> None:
    """Real impl: skip during prefill or graph capture; else run compare_and_log.

    Prefill skip: `impl.rmsnorm_output` and `impl.residual_output` are
    pre-allocated at max-num-seqs (decode-only) shape. During prefill warmup
    or prefill steps, `nat` can exceed that size (up to max-model-len=65536).
    β-coop doesn't run during prefill (it's decode-only), so comparing
    legacy[:nat] against beta[:nat] when nat > beta.shape[0] is meaningless
    AND raises RuntimeError on the shape mismatch in `_compare_pair`'s
    elementwise subtract. Skip silently — the diag is decode-only.

    Capture skip: see capture-skip comment above the registration.
    """
    if nat > beta_rmsnorm_output.shape[0]:
        return  # prefill: β-coop didn't run, comparison invalid
    if torch.cuda.is_current_stream_capturing():
        return
    compare_and_log(
        layer_idx=layer_idx,
        step_idx=step_idx,
        nat=nat,
        legacy_hidden=legacy_hidden,
        legacy_residual=legacy_residual,
        beta_rmsnorm_output=beta_rmsnorm_output,
        beta_residual_output=beta_residual_output,
    )


def _cute_c2_diag_compare_fake(
    layer_idx: int,
    step_idx: int,
    nat: int,
    legacy_hidden: torch.Tensor,
    legacy_residual: torch.Tensor,
    beta_rmsnorm_output: torch.Tensor,
    beta_residual_output: torch.Tensor,
) -> None:
    """Fake impl for Dynamo trace: returns None without inspecting body."""
    return


from vllm.utils.torch_utils import direct_register_custom_op  # noqa: E402

direct_register_custom_op(
    op_name="cute_c2_diag_compare",
    op_func=_cute_c2_diag_compare_impl,
    mutates_args=["legacy_hidden"],
    fake_impl=_cute_c2_diag_compare_fake,
)
