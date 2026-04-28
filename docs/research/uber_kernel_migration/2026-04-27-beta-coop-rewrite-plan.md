# β-coop framework-output-buffer rewrite — implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rewire β-coop's outputs from Python-attribute side-channel buffers (`impl.rmsnorm_output / .residual_output / .mlp_output`) into framework-supplied tensors that the layer pre-allocates and consumes as graph-tracked nodes, registered as a PIECEWISE splitting op for runtime dispatch — eliminating the DCE / consume-gate / Python-round-trip class of bugs that produced gibberish output (`这种现象 × 256`) under PIECEWISE+CUDA graphs.

**Architecture:** Single new thin-wrapper op `vllm::cute_beta_coop_run` (mirrors `unified_attention_with_output`); registered in `_attention_ops` so torch.compile splits the graph at the call. Op body runs eager Python every decode step, delegates to a refactored `_backend.forward(...)` that takes framework output kwargs and dispatches β-coop or fall-through (paged in fusion mode + β-lite MLP). Both paths write to caller-owned tensors; no `cute_residual_mirror`, no `cute_phase_e_dispatch`, no `if _fusion_active` consume gate.

**Tech Stack:** Python 3.12, PyTorch 2.10, vLLM V1 (CuTe paged attention backend), CUDA graphs (PIECEWISE), pytest, nsys. Tests run inside the nvllm Docker container via `.venv/bin/python -m pytest`. Build flag: `TORCH_CUDA_ARCH_LIST="12.0"`. Target hardware: NVIDIA DGX Spark (SM120), 27B model for kernel debugging per CLAUDE.md.

**Branch:** `feat/uber-kernel-migration` HEAD `7d429f1b7`. Savepoint: `feat/pre-beta-coop-rewrite-savepoint` at `7d429f1b7`. File backups at `/tmp/c2_rewrite_backup/{_backend,_mlp_op,kernel,qwen3_5}.py.pre-rewrite`.

**Spec:** [`docs/research/uber_kernel_migration/2026-04-27-beta-coop-rewrite-design.md`](2026-04-27-beta-coop-rewrite-design.md).

---

## Pre-flight (Step 0)

- [ ] **Step 0.1: Verify branch + savepoint**

Run:
```bash
git status --short
git rev-parse --short HEAD
git rev-parse --short feat/pre-beta-coop-rewrite-savepoint
```
Expected: clean tree, HEAD `7d429f1b7`, savepoint `7d429f1b7`. If working tree dirty, stash before proceeding.

- [ ] **Step 0.2: Verify file backups still exist**

Run:
```bash
ls -la /tmp/c2_rewrite_backup/
```
Expected: four `.pre-rewrite` files (`_backend.py`, `_mlp_op.py`, `kernel.py`, `qwen3_5.py`). If missing, recreate per spec rollback section.

- [ ] **Step 0.3: Verify nvllm container is stopped**

Run:
```bash
docker rm -f nvllm 2>&1 | tail -1 || true
```
Expected: empty or `nvllm` removed. Frees ~50 GB unified memory.

---

## Phase 1 — Counter-pattern verification harness (no production code, no rebuild)

**Goal:** Empirically prove the splitting-op pattern (`_attention_ops` registration → eager Python dispatch every step between captured pieces) works on our hardware before touching production code. If this gate fails, our entire architectural premise is wrong — STOP and re-investigate.

**Pre-conditions:** existing `nvllm:gb10` image at `7d429f1b7`. Container can launch via `bash scripts/serve-cute.sh`.

### Task 1: Counter-pattern test scaffold

**Files:**
- Create: `tests/v1/cute_paged/test_beta_coop_skeleton.py`

- [ ] **Step 1.1: Write the test op + counter**

Create `tests/v1/cute_paged/test_beta_coop_skeleton.py` with:

```python
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
```

- [ ] **Step 1.2: Launch container with the existing image**

Run:
```bash
docker rm -f nvllm 2>&1 | tail -1 || true
bash scripts/serve-cute.sh
```
Wait for `Container started: nvllm`. The script enters serve mode; we'll exec into it for tests.

- [ ] **Step 1.3: Run the counter test inside the container**

Run:
```bash
docker exec nvllm /opt/venv/bin/python -m pytest \
  /app/nvllm/tests/v1/cute_paged/test_beta_coop_skeleton.py -v -s 2>&1 | tail -30
```
Note: the test file landed in the host repo; the container's `/app/nvllm` is baked at image build, so we need to copy the new test file into the container first OR mount it.

Alternative if `/app/nvllm` doesn't have the test:
```bash
docker cp tests/v1/cute_paged/test_beta_coop_skeleton.py nvllm:/app/nvllm/tests/v1/cute_paged/test_beta_coop_skeleton.py
docker exec nvllm /opt/venv/bin/python -m pytest \
  /app/nvllm/tests/v1/cute_paged/test_beta_coop_skeleton.py -v -s 2>&1 | tail -30
```

Expected output:
```
tests/v1/cute_paged/test_beta_coop_skeleton.py::test_skeleton_counter_advances_per_replay PASSED
```

If FAIL with `counter=2` (or some small constant): splitting-op registration didn't take effect. Body ran only at capture. **STOP — do not proceed to Phase 2.** Investigate `splitting_ops` config plumbing.

If PASS: the splitting-op pattern works on this hardware. Proceed to Phase 2.

- [ ] **Step 1.4: Stop container**

Run:
```bash
docker rm -f nvllm 2>&1 | tail -1
```

**Phase 1 gate:** counter test passes with N×2 = 6 per shape per 3 replays.

**Rollback if Phase 1 fails:** delete `tests/v1/cute_paged/test_beta_coop_skeleton.py`. No production code touched yet. Re-investigate the architectural premise per the audit doc.

---

## Phase 2 — Op registration stub (rebuild #1)

**Goal:** Create the `cute_beta_coop_run` op as a registered no-op stub. Verify it appears at `torch.ops.vllm.cute_beta_coop_run` and that `_attention_ops` registration takes effect.

**Pre-conditions:** Phase 1 gate passed.

### Task 2: Op registration

**Files:**
- Create: `vllm/v1/attention/backends/cute_paged/_beta_coop_op.py`
- Modify: `vllm/config/compilation.py:713-727` (add op to `_attention_ops`)
- Modify: `vllm/nvllm/models/qwen3_5.py` top imports (add side-effect import)

- [ ] **Step 2.1: Create the new op module**

Create `vllm/v1/attention/backends/cute_paged/_beta_coop_op.py` with:

```python
"""β-coop framework-output-buffer dispatch op.

Mirrors `vllm::unified_attention_with_output` (vllm/model_executor/layers/
attention/attention.py:712-760) — a thin custom op that delegates to
`layer.impl.forward(...)` via `get_attention_context(layer_name)`.

Registered as a PIECEWISE splitting boundary in
`vllm/config/compilation.py:_attention_ops` so torch.compile splits the
FX graph at the call. The op body runs as eager Python at runtime,
between captured graph segments, on every decode step.

Phase 2: stub raises NotImplementedError. Phase 3 fills in delegation.

See spec: docs/research/uber_kernel_migration/2026-04-27-beta-coop-rewrite-design.md
See feedback memory: feedback_splitting_op_runtime_dispatch
"""

from __future__ import annotations

import torch

from vllm.utils.torch_utils import direct_register_custom_op

# Per-layer fire counter for empirical replay verification (mirrors
# tests/compile/silly_attention.py:50). Reset by tests; in production
# remains a debug observation point.
_BETA_COOP_FIRE_COUNTER: dict[str, int] = {}


def cute_beta_coop_run(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    residual: torch.Tensor,
    attn_input: torch.Tensor,
    gate: torch.Tensor,
    output_rmsnorm: torch.Tensor,
    output_residual: torch.Tensor,
    output_mlp: torch.Tensor,
    layer_name: str,
    kv_cache_dummy_dep: torch.Tensor | None = None,
) -> None:
    """Phase 2 stub. Phase 3 will delegate to layer.impl.forward."""
    del kv_cache_dummy_dep
    raise NotImplementedError(
        "cute_beta_coop_run is a Phase 2 stub; not yet wired to "
        "_backend.forward (Phase 3)."
    )


def cute_beta_coop_run_fake(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    residual: torch.Tensor,
    attn_input: torch.Tensor,
    gate: torch.Tensor,
    output_rmsnorm: torch.Tensor,
    output_residual: torch.Tensor,
    output_mlp: torch.Tensor,
    layer_name: str,
    kv_cache_dummy_dep: torch.Tensor | None = None,
) -> None:
    return None


direct_register_custom_op(
    op_name="cute_beta_coop_run",
    op_func=cute_beta_coop_run,
    mutates_args=["output_rmsnorm", "output_residual", "output_mlp"],
    fake_impl=cute_beta_coop_run_fake,
)
```

- [ ] **Step 2.2: Add op to `_attention_ops` splitting list**

Modify `vllm/config/compilation.py` at `_attention_ops: ClassVar[list[str]] = [...]` (currently around line 713-727). Add after the existing `vllm::linear_attention` entry:

```python
    _attention_ops: ClassVar[list[str]] = [
        "vllm::unified_attention",
        "vllm::unified_attention_with_output",
        "vllm::unified_mla_attention",
        "vllm::unified_mla_attention_with_output",
        "vllm::mamba_mixer2",
        "vllm::mamba_mixer",
        "vllm::short_conv",
        "vllm::linear_attention",
        "vllm::cute_beta_coop_run",   # ← ADD THIS
        "vllm::plamo2_mamba_mixer",
        # ... rest unchanged
```

Verify the exact list order and existing entries with `grep -n '_attention_ops' vllm/config/compilation.py` before editing.

- [ ] **Step 2.3: Add side-effect import in qwen3_5.py**

Modify `vllm/nvllm/models/qwen3_5.py`. Find the existing import block near the top (look for the existing `from vllm.model_executor...` lines). Add:

```python
# Side-effect import: registers torch.ops.vllm.cute_beta_coop_run.
# Mirrors vllm/nvllm/layers/mlp.py:21 pattern. Importing here ensures
# the op exists at torch.compile trace time even for the attached-fusion
# branch.
import vllm.v1.attention.backends.cute_paged._beta_coop_op  # noqa: F401
```

Place it alongside the existing imports (any position before any `@support_torch_compile`-decorated class).

- [ ] **Step 2.4: Trigger rebuild #1 in tmux**

Per `feedback_delegate_builds`: builds in tmux, not subagents. Per `feedback_no_poll`: background task with completion notification.

Run:
```bash
tmux kill-session -t phase2-rebuild 2>/dev/null || true
tmux new-session -d -s phase2-rebuild "docker build -f docker/Dockerfile.gb10 -t nvllm:gb10 . 2>&1 | tee /tmp/nvllm-build.log; echo BUILD_DONE_RC=\$? >> /tmp/nvllm-build.log"
echo "Build started in tmux session phase2-rebuild"
```

Then wait via background task:
```bash
until grep -q '^BUILD_DONE_RC=' /tmp/nvllm-build.log 2>/dev/null; do sleep 30; done
rc=$(grep '^BUILD_DONE_RC=' /tmp/nvllm-build.log | cut -d= -f2)
echo "BUILD_FINISHED rc=$rc"
if [ "$rc" != "0" ]; then
  echo "---LAST 60 LINES---"
  tail -60 /tmp/nvllm-build.log
  exit 1
fi
docker images nvllm:gb10 --format 'IMAGE_TS={{.CreatedAt}}'
```
Use `run_in_background: true`. Wait for notification. Expect `BUILD_FINISHED rc=0` with a fresh image timestamp.

- [ ] **Step 2.5: Verify op visible in baked image**

Run:
```bash
docker run --rm --entrypoint bash nvllm:gb10 -lc "
/opt/venv/bin/python -c '
import torch
from vllm.v1.attention.backends.cute_paged import _beta_coop_op
print(\"OP:\", torch.ops.vllm.cute_beta_coop_run)
print(\"FAKE_IMPL_RUNS:\", end=\" \")
import torch._subclasses.fake_tensor as ft
with ft.FakeTensorMode():
    q = torch.empty(1, 1, dtype=torch.bfloat16)
    rest = [torch.empty(1, 1, dtype=torch.bfloat16) for _ in range(8)]
    out = torch.ops.vllm.cute_beta_coop_run(q, *rest, \"test_layer\")
    print(out)
print(\"GREP_ATTENTION_OPS:\")
import vllm.config.compilation as cc
assert \"vllm::cute_beta_coop_run\" in cc.CompilationConfig._attention_ops, \"NOT IN _attention_ops!\"
print(\"  YES — in _attention_ops\")
'"
```

Expected:
```
OP: vllm.cute_beta_coop_run
FAKE_IMPL_RUNS: None
GREP_ATTENTION_OPS:
  YES — in _attention_ops
```

If anything fails: investigate registration. Likely candidates: missing side-effect import, typo in op name, missing entry in `_attention_ops`.

**Phase 2 gate:** op visible at `torch.ops.vllm.cute_beta_coop_run`; fake impl returns `None`; `_attention_ops` includes the new op.

**Rollback if Phase 2 fails:** revert the three file edits via:
```bash
cp /tmp/c2_rewrite_backup/qwen3_5.py.pre-rewrite vllm/nvllm/models/qwen3_5.py
git checkout vllm/config/compilation.py
rm -f vllm/v1/attention/backends/cute_paged/_beta_coop_op.py
```
Then re-investigate before retry.

---

## Phase 3 — Refactor `_backend.forward` + layer rewrite, fall-through forced (rebuild #2)

**Goal:** Wire `cute_beta_coop_run`'s body to delegate to a refactored `_backend.forward(...)` that takes framework output kwargs. Force fall-through (β-coop disabled). Production behavior must match the existing `CUTE_PHASE_E_FUSION=0` baseline (coherent output, GSM8K-50 ≥ 90%).

**Pre-conditions:** Phase 2 gate passed.

### Task 3: Add `_beta_coop_framework_output_bound` flag

**Files:**
- Modify: `vllm/v1/attention/backends/cute_paged/_backend.py:884-960` (add flag init at end of `_resolve_mlp_weights`)
- Modify: `vllm/v1/attention/backends/cute_paged/_backend.py:252+` (add flag init in `__init__`)

- [ ] **Step 3.1: Initialize flag in `__init__`**

Modify `_backend.py:252+`. After the existing `self._fusion_bound = False` and `self._mlp_fusion_bound = False` (at lines 252, 257), add:

```python
        # Phase E β-coop framework-output-buffer route. Set True at end of
        # _resolve_mlp_weights() ONLY when all three pre-conditions hold:
        # _fusion_bound, _mlp_fusion_bound, _phase_e_coop_kernel is not None.
        # See feedback_splitting_op_runtime_dispatch + spec § Q3.
        self._beta_coop_framework_output_bound: bool = False
```

- [ ] **Step 3.2: Set flag at end of `_resolve_mlp_weights`**

Modify `_backend.py` inside `_resolve_mlp_weights` (around line 884-958). Find the line `self._mlp_fusion_bound = True` (currently line 950). After that line, add:

```python
        # All three β-coop framework-output prerequisites resolved.
        # This is the stable post-weight-load flag the decoder layer
        # branches on.
        self._beta_coop_framework_output_bound = (
            self._fusion_bound
            and self._mlp_fusion_bound
            and getattr(self, "_phase_e_coop_kernel", None) is not None
        )
```

Also verify: at every early-return inside `_resolve_mlp_weights` that sets `self._mlp_fusion_bound = False` (currently lines 904, 910, 926), explicitly add `self._beta_coop_framework_output_bound = False` to ensure the flag never stays True across re-resolution failures.

- [ ] **Step 3.3: Run a quick import smoke test**

Run:
```bash
docker run --rm --entrypoint bash nvllm:gb10 -lc "
/opt/venv/bin/python -c '
from vllm.v1.attention.backends.cute_paged._backend import CutePagedAttentionImpl
print(\"OK\")
'"
```
Expected: `OK` (this just verifies the new flag init doesn't introduce a syntax error). If it fails, fix and re-run before continuing.

### Task 4: Refactor `_backend.forward` signature with output kwargs

**Files:**
- Modify: `vllm/v1/attention/backends/cute_paged/_backend.py:993-1004` (signature)
- Modify: `vllm/v1/attention/backends/cute_paged/_backend.py:1267-1310` (β-coop launch site — wire framework outputs)
- Modify: `vllm/v1/attention/backends/cute_paged/_backend.py:1410-1450` (β-lite launch site — wire framework outputs, eliminate side-channels)

- [ ] **Step 4.1: Extend `_backend.forward` signature**

Modify the signature at line 993-1004. New signature:

```python
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
        # NEW (Phase 3): framework-output-buffer route. When all three
        # are non-None, β-coop / fall-through writes through these.
        residual: torch.Tensor | None = None,
        attn_input: torch.Tensor | None = None,
        gate: torch.Tensor | None = None,
        output_rmsnorm: torch.Tensor | None = None,
        output_residual: torch.Tensor | None = None,
        output_mlp: torch.Tensor | None = None,
    ) -> torch.Tensor:
```

Add precondition assertions immediately after the signature:

```python
        # Phase 3 framework-output route is active when ALL six new kwargs
        # are non-None (caller is the new cute_beta_coop_run op).
        _framework_output_route = (
            residual is not None and attn_input is not None and gate is not None
            and output_rmsnorm is not None and output_residual is not None
            and output_mlp is not None
        )
        if _framework_output_route:
            assert output_rmsnorm.shape == output_residual.shape == output_mlp.shape == residual.shape, (
                f"Framework output shape mismatch: rmsnorm={output_rmsnorm.shape} "
                f"residual={output_residual.shape} mlp={output_mlp.shape} "
                f"input residual={residual.shape}"
            )
            assert output_rmsnorm.dtype == torch.bfloat16
```

The legacy path (when `_framework_output_route=False`) is preserved unchanged for now — Phase 3 introduces the framework-output route alongside the legacy path; Phase 5 deletes the legacy path.

- [ ] **Step 4.2: Wire β-coop launch to framework outputs**

Modify the β-coop call at `_backend.py:1267-1310`. Replace `attn_output=self.rmsnorm_output[:nat]` and `residual_output=self.residual_output[:nat]` and `mlp_output=self.mlp_output[:nat]` with framework-output equivalents conditioned on the route:

```python
                # β-coop launch — use framework outputs when available,
                # else legacy self.X scratch. Phase 5 cleanup deletes
                # the self.X path entirely.
                _attn_output_buf = (
                    output_rmsnorm[:nat] if _framework_output_route
                    else self.rmsnorm_output[:nat]
                )
                _residual_output_buf = (
                    output_residual[:nat] if _framework_output_route
                    else self.residual_output[:nat]
                )
                _mlp_output_buf = (
                    output_mlp[:nat] if _framework_output_route
                    else self.mlp_output[:nat]
                )
                _residual_in_buf = (
                    residual[:nat] if _framework_output_route
                    else self.residual_buf[:nat]
                )
                _gate_buf = (
                    gate[:nat] if _framework_output_route
                    else self.gate_buf[:nat]
                )
                with record_function(f"PhaseE_Beta.coop.{_layer_name}"):
                    self._phase_e_coop_kernel.run_beta_coop_full(
                        hidden_in=_attn_output_buf,  # placeholder; β-coop ignores
                        residual_in=_residual_in_buf,
                        input_gamma=self._phase_e_coop_input_gamma,
                        post_attn_gamma=self.rmsnorm_gamma,
                        attn_input_bf16=self._phase_e_coop_attn_input_scratch[:nat],
                        query=query[:nat],
                        kv_cache=kv_cache,
                        page_table=attn_metadata.block_table,
                        seq_lens=attn_metadata.seq_lens,
                        wo_weight=self.wo_weight,
                        wo_scales=self.wo_scales,
                        wo_global_scale=self.wo_global_scale,
                        attn_output=_attn_output_buf,
                        gate_w_fp4=self._mlp_gate_w,
                        gate_w_scale=self._mlp_gate_s,
                        up_w_fp4=self._mlp_up_w,
                        up_w_scale=self._mlp_up_s,
                        down_w_fp4=self._mlp_down_w,
                        down_w_scale=self._mlp_down_s,
                        mlp_output=_mlp_output_buf,
                        scale=self.scale,
                        k_scale=k_scale,
                        v_scale=v_scale,
                        gate_up_global_scale=self._mlp_gate_up_gs,
                        down_global_scale=self._mlp_down_gs,
                        residual_output=_residual_output_buf,
                        gate_buf=_gate_buf,
                    )
                self._phase_e_consumed = True
                self._phase_e_use_beta_coop = True
```

- [ ] **Step 4.3: Wire β-lite launch to framework outputs**

Modify the β-lite call at `_backend.py:1410-1450`. Replace the existing `self.rmsnorm_output[:nat]` (input), `self.mlp_output[:nat]` (output), `self.residual_buf[:nat]` (residual_post_ln) with framework equivalents:

```python
                # β-lite launch — uses framework outputs when available.
                # When the framework-output route is active, paged
                # attention has already written output_rmsnorm and
                # output_residual; β-lite reads output_rmsnorm as MLP
                # input and writes output_mlp.
                _mlp_in = (
                    output_rmsnorm[:nat] if _framework_output_route
                    else self.rmsnorm_output[:nat]
                )
                _mlp_out = (
                    output_mlp[:nat] if _framework_output_route
                    else self.mlp_output[:nat]
                )
                _residual_post_ln = (
                    output_residual[:nat] if _framework_output_route
                    else self.residual_buf[:nat]
                )
                with record_function(f"PhaseE_Beta.lite.{_layer_name}"):
                    self._mlp_kernel(
                        _mlp_in,
                        self._mlp_gate_w,
                        self._mlp_gate_s,
                        self._mlp_up_w,
                        self._mlp_up_s,
                        self._mlp_down_w,
                        self._mlp_down_s,
                        self.mlp_partial_fp32[:nat],
                        self.mlp_arrival_count[:nat],
                        _mlp_out,
                        nat,
                        gate_up_global_scale=self._mlp_gate_up_gs,
                        down_global_scale=self._mlp_down_gs,
                        residual_post_ln=_residual_post_ln,
                        next_input_layernorm_gamma=_next_gamma,
                        next_hidden_output=self.next_hidden_scratch[:nat],
                        emit_epilogue=True,
                        emit_next_layernorm=_emit_next,
                        rms_eps=_rms_eps,
                    )
```

The `self.next_hidden_scratch` stays as impl-internal scratch — never consumed by Python, so DCE doesn't apply. Phase 4 (deleted ε epilogue per `project_own_the_stack` C1.5) doesn't actually use it but the kernel signature still requires it.

- [ ] **Step 4.4: Wire paged path to framework outputs**

Modify the paged_attention_forward call at `_backend.py:1105-1127` (the `else` branch when β-coop won't fire). Replace the rmsnorm_output / residual_output with framework equivalents when route is active:

```python
            paged_rmsnorm_output = (
                output_rmsnorm if _framework_output_route else rmsnorm_output
            )
            paged_residual_output = (
                output_residual if _framework_output_route else residual_output
            )
            paged_rmsnorm_residual = (
                residual if _framework_output_route else rmsnorm_residual
            )
            paged_gate_buf = (
                gate if _framework_output_route else gate_buf
            )
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
                rmsnorm_residual=paged_rmsnorm_residual,
                rmsnorm_output=paged_rmsnorm_output,
                residual_output=paged_residual_output,
                arrival_count=arrival_count,
                rmsnorm_eps=rmsnorm_eps,
                gate_buf=paged_gate_buf,
                padded_num_seqs=padded_num_seqs,
            )
```

- [ ] **Step 4.5: Hard-wire `_use_beta_coop = False` for Phase 3 forced fall-through**

This is the Phase-3-only change. Modify `_use_beta_coop` at `_backend.py:1223-1228`:

```python
        # PHASE 3 ONLY: force fall-through to validate layer-side rewrite
        # against existing PIECEWISE+graphs baseline. Phase 4 restores
        # the live gate.
        _use_beta_coop = False  # PHASE_3_FORCE_FALLTHROUGH — REVERT IN PHASE 4
        # _use_beta_coop = (
        #     _phase_e_active
        #     and _coop_attached
        #     and _total_ctas <= _resident_cap
        #     and _phase_e_env.forced_path in ("coop", "auto")
        # )
```

Per `feedback_comment_not_delete`: comment out the original gate, don't delete it. Phase 4 restores it.

### Task 5: Implement op body delegation

**Files:**
- Modify: `vllm/v1/attention/backends/cute_paged/_beta_coop_op.py` (replace stub with real delegation)

- [ ] **Step 5.1: Replace stub body with delegation**

Replace the entire `cute_beta_coop_run` function body in `_beta_coop_op.py`:

```python
def cute_beta_coop_run(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    residual: torch.Tensor,
    attn_input: torch.Tensor,
    gate: torch.Tensor,
    output_rmsnorm: torch.Tensor,
    output_residual: torch.Tensor,
    output_mlp: torch.Tensor,
    layer_name: str,
    kv_cache_dummy_dep: torch.Tensor | None = None,
) -> None:
    """Eager Python body — runs every decode step (splitting boundary).

    Delegates to layer.impl.forward(...) with the framework-output kwargs.
    `_backend.forward` handles β-coop dispatch vs fall-through internally
    based on attn_metadata.is_decode_only and num_seqs vs resident_cap.
    """
    # kv_cache_dummy_dep is the canonical phantom-dep pattern from
    # vllm/model_executor/layers/attention/attention.py:721-726 — provides
    # an explicit data-dependency edge from unified_kv_cache_update to
    # this op so dynamo preserves ordering.
    del kv_cache_dummy_dep

    from vllm.model_executor.layers.attention.attention import get_attention_context

    _BETA_COOP_FIRE_COUNTER[layer_name] = (
        _BETA_COOP_FIRE_COUNTER.get(layer_name, 0) + 1
    )

    attn_metadata, attn_layer, kv_cache, _ = get_attention_context(layer_name)
    attn_layer.impl.forward(
        attn_layer,
        query,
        key,
        value,
        kv_cache,
        attn_metadata,
        residual=residual,
        attn_input=attn_input,
        gate=gate,
        output_rmsnorm=output_rmsnorm,
        output_residual=output_residual,
        output_mlp=output_mlp,
    )
```

The fake impl stays as-is (returns None).

### Task 6: Modify `Qwen3_5Attention.forward`

**Files:**
- Modify: `vllm/nvllm/models/qwen3_5.py:235-300` (signature + body)

- [ ] **Step 6.1: Extend signature**

Modify `Qwen3_5Attention.forward`. New signature:

```python
    def forward(
        self,
        positions: torch.Tensor,
        output: torch.Tensor,
        hidden_states: torch.Tensor,
        # NEW (Phase 3): framework-output route. When all three non-None,
        # β-coop / fall-through writes through these (output is reused
        # as output_rmsnorm).
        residual: torch.Tensor | None = None,
        output_residual: torch.Tensor | None = None,
        output_mlp: torch.Tensor | None = None,
    ):
```

- [ ] **Step 6.2: Add trace-time branch**

After the existing QKV+q_norm+k_norm+rotary+gate compute (around line 295, just before the existing `if self.attn_output_gate and gate is not None:` block), insert the new dispatch:

```python
        # ---- β-coop framework-output route (Phase 3+) ----
        # _beta_coop_framework_output_bound is set in
        # _resolve_mlp_weights when ALL three pre-conditions hold
        # (attn fusion + MLP fusion + coop kernel attached).
        # See feedback_splitting_op_runtime_dispatch.
        impl = self.attn.impl
        _use_framework_output = (
            getattr(impl, "_beta_coop_framework_output_bound", False)
            and residual is not None
            and output_residual is not None
            and output_mlp is not None
        )
        if _use_framework_output:
            # KV-cache update (cute_paged backend has
            # forward_includes_kv_cache_update = False at _backend.py:124,
            # so caller must do this — mirrors attention.py:466-475).
            from vllm.model_executor.layers.attention.attention import (
                unified_kv_cache_update,
            )
            kv_cache_dummy_dep = unified_kv_cache_update(
                k, v, self.attn.layer_name
            )
            # `attn_input` for β-coop is the post-input-LN hidden_states
            # (already computed by the caller before this forward).
            torch.ops.vllm.cute_beta_coop_run(
                q,
                k,
                v,
                residual,
                hidden_states,  # attn_input
                gate if self.attn_output_gate else torch.empty(0, device=q.device, dtype=torch.bfloat16),
                output,         # → output_rmsnorm
                output_residual,
                output_mlp,
                self.attn.layer_name,
                kv_cache_dummy_dep=kv_cache_dummy_dep,
            )
            return  # Decoder layer reads `output`, `output_residual`, `output_mlp`.

        # ---- Legacy path (unchanged) ----
        # [existing self.attn(q, k, v) call + post-attn gate/o_proj]
```

The legacy path (everything from the existing `if self.attn_output_gate and gate is not None:` block at line 295 through the end of forward) stays unchanged.

### Task 7: Modify `Qwen3_5DecoderLayer.forward`

**Files:**
- Modify: `vllm/nvllm/models/qwen3_5.py:440-510` (decoder layer)

- [ ] **Step 7.1: Pre-allocate framework outputs and pass to attention**

In `Qwen3_5DecoderLayer.forward`, after the existing `input_layernorm` call and before `self.self_attn(...)`:

```python
        impl = self.self_attn.attn.impl if self.layer_type == "full_attention" else None
        _framework_output_route = (
            self.layer_type == "full_attention"
            and impl is not None
            and getattr(impl, "_beta_coop_framework_output_bound", False)
        )
        if _framework_output_route:
            output_residual = torch.empty_like(residual)
            output_mlp = torch.empty_like(residual)
        else:
            output_residual = None
            output_mlp = None

        self_attention_output = torch.empty_like(hidden_states)
        if self.layer_type == "full_attention":
            self.self_attn(
                hidden_states=hidden_states,
                output=self_attention_output,
                positions=positions,
                residual=residual if _framework_output_route else None,
                output_residual=output_residual,
                output_mlp=output_mlp,
            )
        else:
            self.linear_attn(
                hidden_states=hidden_states,
                output=self_attention_output,
            )
            hidden_states = self_attention_output
```

- [ ] **Step 7.2: Consume framework outputs (skip post-attn-LN + MLP when route active)**

In the same forward, replace the existing block (around lines 480-510 — `if impl is not None and getattr(impl, "_fusion_active", False):` consume + `if not _fusion_active: post_attention_layernorm`) with:

```python
        if _framework_output_route:
            # β-coop / fall-through wrote through output, output_residual,
            # output_mlp. Decoder layer consumes them directly — no Python
            # post-attn-LN, no Python MLP.
            hidden_states = output_mlp
            residual = output_residual
            return hidden_states, residual

        # ---- Legacy path (unchanged) ----
        # [existing post_attention_layernorm + MLP blocks]
```

Per `feedback_comment_not_delete`: don't delete the legacy block; this `if _framework_output_route: ... return` short-circuits before it. The legacy code below remains live for non-fused layers.

### Task 8: Rebuild + verify fall-through baseline

- [ ] **Step 8.1: Trigger rebuild #2 in tmux**

Run:
```bash
tmux kill-session -t phase3-rebuild 2>/dev/null || true
tmux new-session -d -s phase3-rebuild "docker build -f docker/Dockerfile.gb10 -t nvllm:gb10 . 2>&1 | tee /tmp/nvllm-build.log; echo BUILD_DONE_RC=\$? >> /tmp/nvllm-build.log"
echo "Build started in tmux session phase3-rebuild"
```

Then wait via background task:
```bash
until grep -q '^BUILD_DONE_RC=' /tmp/nvllm-build.log 2>/dev/null; do sleep 30; done
rc=$(grep '^BUILD_DONE_RC=' /tmp/nvllm-build.log | cut -d= -f2)
echo "BUILD_FINISHED rc=$rc"
if [ "$rc" != "0" ]; then echo "---LAST 60 LINES---"; tail -60 /tmp/nvllm-build.log; exit 1; fi
docker images nvllm:gb10 --format 'IMAGE_TS={{.CreatedAt}}'
```
Use `run_in_background: true`. Wait for notification. Expect `BUILD_FINISHED rc=0`.

- [ ] **Step 8.2: Launch container with `CUTE_PHASE_E_FUSION=1` (β-coop forced off via Phase 3 hard-wire)**

Run:
```bash
docker rm -f nvllm 2>&1 | tail -1 || true
CUTE_PHASE_E_FUSION=1 bash scripts/serve-cute.sh
```

The Phase 3 `_use_beta_coop = False` hard-wire forces fall-through regardless of `CUTE_PHASE_E_FUSION`.

- [ ] **Step 8.3: Wait for /v1/models or engine death (background task)**

Run:
```bash
until curl -sf http://localhost:8000/v1/models >/dev/null 2>&1; do
  if ! docker ps --filter name=nvllm --format '{{.Names}}' | grep -q '^nvllm$'; then
    echo "ENGINE_DIED"
    docker logs --tail 80 nvllm 2>&1
    exit 1
  fi
  sleep 5
done
echo "API_UP"
```
Use `run_in_background: true`. Wait for notification.

- [ ] **Step 8.4: Coherence probe (256 tokens)**

Run:
```bash
curl -sS http://localhost:8000/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"default","prompt":"The capital of France is ","max_tokens":256,"temperature":0.0}' \
  > /tmp/phase3_completion.json
.venv/bin/python -c "
import json
j = json.load(open('/tmp/phase3_completion.json'))
print(j['choices'][0]['text'][:300])
"
```
Expected: starts with `Paris.` and continues coherently. If gibberish (`这种现象` loop), the layer-side rewrite has a bug — fix before proceeding.

- [ ] **Step 8.5: GSM8K-50 quality gate**

Run:
```bash
.venv/bin/python scripts/gsm8k_eval_50.py 2>&1 | tail -20
```
Expected: accuracy ≥ 90% (per `feedback_post_quant_sanity`). Runtime: ~5-10 minutes.

If accuracy drops: review steps 4.4 (paged path output wiring) + steps 7 (decoder layer consume). The output_residual or output_rmsnorm is likely being written wrong, or the decoder layer is consuming the wrong tensor.

- [ ] **Step 8.6: Stop container**

Run:
```bash
docker rm -f nvllm 2>&1 | tail -1
```

**Phase 3 gate (must pass ALL):**
- Engine starts under `CUTE_PHASE_E_FUSION=1` (Phase 3 forces fall-through internally).
- Coherent output on `"The capital of France is "` probe → starts with `Paris.`
- GSM8K-50 ≥ 90%.

**Rollback if Phase 3 fails:**
```bash
git checkout vllm/v1/attention/backends/cute_paged/_backend.py
git checkout vllm/nvllm/models/qwen3_5.py
git checkout vllm/v1/attention/backends/cute_paged/_beta_coop_op.py
git checkout vllm/config/compilation.py
# rebuild via Step 2.4 procedure
```
Or revert to savepoint:
```bash
git reset --hard feat/pre-beta-coop-rewrite-savepoint
# rebuild
```

---

## Phase 4 — β-coop dispatch enabled (rebuild #3) — moment of truth

**Goal:** Restore live `_use_beta_coop` gate. With `CUTE_PHASE_E_FUSION=1 MAX_NUM_SEQS=1`, β-coop must fire at runtime decode (num_seqs=1 fits cooperative cap). Output must be coherent — gibberish here means the architecture is correct but β-coop's kernel has graph-replay quirks (separate kernel-level investigation).

**Pre-conditions:** Phase 3 gate passed.

### Task 9: Restore live β-coop gate

**Files:**
- Modify: `vllm/v1/attention/backends/cute_paged/_backend.py:1223-1228`

- [ ] **Step 9.1: Restore live gate**

Find the `_use_beta_coop = False  # PHASE_3_FORCE_FALLTHROUGH` line from Step 4.5. Replace with the original gate (uncommented):

```python
        _use_beta_coop = (
            _phase_e_active
            and _coop_attached
            and _total_ctas <= _resident_cap
            and _phase_e_env.forced_path in ("coop", "auto")
        )
```

- [ ] **Step 9.2: Rebuild #3 in tmux**

Run:
```bash
tmux kill-session -t phase4-rebuild 2>/dev/null || true
tmux new-session -d -s phase4-rebuild "docker build -f docker/Dockerfile.gb10 -t nvllm:gb10 . 2>&1 | tee /tmp/nvllm-build.log; echo BUILD_DONE_RC=\$? >> /tmp/nvllm-build.log"
echo "Build started"
```

Wait via background task:
```bash
until grep -q '^BUILD_DONE_RC=' /tmp/nvllm-build.log 2>/dev/null; do sleep 30; done
rc=$(grep '^BUILD_DONE_RC=' /tmp/nvllm-build.log | cut -d= -f2)
echo "BUILD_FINISHED rc=$rc"
if [ "$rc" != "0" ]; then echo "---LAST 60 LINES---"; tail -60 /tmp/nvllm-build.log; exit 1; fi
```
Use `run_in_background: true`.

### Task 10: Verify β-coop runtime correctness

- [ ] **Step 10.1: Launch with β-coop enabled**

Run:
```bash
docker rm -f nvllm 2>&1 | tail -1 || true
MAX_NUM_SEQS=1 CUTE_PHASE_E_FUSION=1 bash scripts/serve-cute.sh
```

- [ ] **Step 10.2: Wait for /v1/models or halt (background task)**

Run:
```bash
until curl -sf http://localhost:8000/v1/models >/dev/null 2>&1; do
  if ! docker ps --filter name=nvllm --format '{{.Names}}' | grep -q '^nvllm$'; then
    echo "ENGINE_DIED"
    docker logs --tail 80 nvllm 2>&1
    exit 1
  fi
  sleep 5
done
echo "API_UP"
docker logs nvllm 2>&1 | grep -E 'CUDA_ERROR|cudaErrorStreamCaptureInvalidated' | head -5 || echo "NO_CAPTURE_ERRORS"
```
Use `run_in_background: true`.

Expected: `API_UP` + `NO_CAPTURE_ERRORS`.

- [ ] **Step 10.3: 256-token coherence probe**

Run:
```bash
curl -sS http://localhost:8000/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"default","prompt":"The capital of France is ","max_tokens":256,"temperature":0.0}' \
  > /tmp/phase4_completion.json
.venv/bin/python -c "
import json
j = json.load(open('/tmp/phase4_completion.json'))
txt = j['choices'][0]['text']
print(txt[:400])
assert '这种现象' not in txt, 'GIBBERISH detected — β-coop graph-replay bug persists'
assert txt.strip().startswith('Paris'), f'Bad first token: {txt[:50]!r}'
print('COHERENCE_OK')
"
```
Expected: `COHERENCE_OK`.

If `这种现象` appears: the architectural rewrite is correct (Phase 3 fall-through proved that), but β-coop's cooperative-launch + atomic-counter spin-wait has CUDA-graph replay quirks (suspect #2 from `2026-04-26-consume-gate-dce-and-graph-capture.md`). Trigger kernel-level debug session:
- Capture nsys trace at decode steps; verify kernel actually launches with non-zero residual_in.
- Run β-coop kernel in a standalone `torch.cuda.graph(...)` test (no vLLM, no PIECEWISE).
- Investigate `gridDependencyAcquire` / cooperative-launch interactions with graph capture.

Stop here pending kernel investigation. Do NOT attempt another rewrite cycle on the same architecture.

- [ ] **Step 10.4: Counter verification**

The `_BETA_COOP_FIRE_COUNTER` is module-state inside the EngineCore subprocess. Read it via a docker exec:

```bash
docker exec nvllm /opt/venv/bin/python -c "
from vllm.v1.attention.backends.cute_paged._beta_coop_op import _BETA_COOP_FIRE_COUNTER
print('LAYERS_FIRED:', sorted(_BETA_COOP_FIRE_COUNTER.keys()))
print('FIRES_PER_LAYER:', {k: v for k, v in sorted(_BETA_COOP_FIRE_COUNTER.items())})
total = sum(_BETA_COOP_FIRE_COUNTER.values())
print(f'TOTAL_FIRES: {total}')
"
```

Expected:
- 16 fusion-bound layers fired (numbered for full-attn positions in stride-4 pattern: layers 3,7,11,...,63).
- Per-layer count ≈ 256 (one per generated token).
- TOTAL_FIRES ≈ 256 × 16 = ~4096.

Significantly fewer fires per layer means dispatch isn't reaching β-coop on every step (debug paths: `_fusion_bound`, `_phase_e_active`, cooperative-fitness gate).

- [ ] **Step 10.5: GSM8K-50 with β-coop active**

Run:
```bash
MAX_NUM_SEQS=1 CUTE_PHASE_E_FUSION=1 .venv/bin/python scripts/gsm8k_eval_50.py 2>&1 | tail -20
```
Expected: accuracy ≥ 90%. Runtime: ~5-15 min.

- [ ] **Step 10.6: Capture nsys trace per AGENTS.md §4**

Per `feedback_nsys_privileged` and AGENTS.md §4:
```bash
mkdir -p /tmp/nsys_phase4
docker run --rm --gpus all --privileged \
  --network host \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  -v /tmp/nsys_phase4:/tmp/nsys_phase4 \
  -e MAX_NUM_SEQS=1 -e CUTE_PHASE_E_FUSION=1 \
  --entrypoint bash nvllm:gb10 -lc "
    nsys profile --trace=cuda,nvtx \
      --output=/tmp/nsys_phase4/baseline.nsys-rep \
      --capture-range=cudaProfilerApi --capture-range-end=stop \
      /opt/venv/bin/python -m vllm.entrypoints.openai.api_server \
      --model ig1/Qwen3.5-27B-NVFP4 --max-num-seqs 1 \
      --kernel-config '{\"enable_flashinfer_autotune\":false}' \
      --compilation-config '{\"cudagraph_mode\":\"PIECEWISE\"}' &
    SERVER_PID=\$!
    sleep 60  # let server start; not polling per feedback_no_poll
    curl -s http://localhost:8000/v1/completions -H 'Content-Type: application/json' \
      -d '{\"model\":\"default\",\"prompt\":\"Tell me a story.\",\"max_tokens\":64,\"temperature\":0.0}' > /dev/null
    kill \$SERVER_PID
"
ls -la /tmp/nsys_phase4/
```

Inspect the trace for β-coop kernel launches:
```bash
docker run --rm -v /tmp/nsys_phase4:/data --entrypoint bash nvllm:gb10 -lc "
  /opt/nvidia/nsight-systems/2024.6.1/host-linux-x64/nsys stats --report cuda_kern_exec_sum /data/baseline.nsys-rep 2>&1 | grep -iE 'beta_coop|run_beta|phase_e' | head -10
"
```
Expected: at least one β-coop kernel name appearing (the cooperative-launch entry for the fused attn+mlp).

- [ ] **Step 10.7: Stop container**

```bash
docker rm -f nvllm 2>&1 | tail -1
```

**Phase 4 gate (must pass ALL):**
- Engine starts; no `cudaErrorStreamCaptureInvalidated`.
- Coherent 256-token completion (no `这种现象` loop).
- `_BETA_COOP_FIRE_COUNTER` shows ~256 fires per fusion-bound layer.
- GSM8K-50 ≥ 90%.
- nsys trace shows β-coop kernel launching at decode.

**Rollback if Phase 4 gibberish:** revert Step 9.1 to `_use_beta_coop = False`. Engine returns to Phase 3 fall-through behavior (proven coherent). Investigate β-coop kernel-level graph-replay quirks separately.

---

## Phase 5 — Cleanup + nsys evidence + commit (rebuild #4)

**Goal:** Retire the deprecated DCE-fragile ops and the dead self.X scratch, write nsys evidence, update memory, request user approval, commit.

**Pre-conditions:** Phase 4 gate passed (all four bullets).

### Task 11: Retire deprecated ops

**Files:**
- Modify: `vllm/v1/attention/backends/cute_paged/_mlp_op.py:266-311` (cute_residual_mirror → NotImplementedError stub)
- Modify: `vllm/v1/attention/backends/cute_paged/_mlp_op.py:182-216` (cute_phase_e_dispatch — delete or stub)
- Modify: `vllm/nvllm/models/qwen3_5.py:482-490` (delete copy/consume block)
- Modify: `vllm/v1/attention/backends/cute_paged/_backend.py:325-330` (delete self.rmsnorm_output / residual_output / mlp_output allocations)

- [ ] **Step 11.1: Replace cute_residual_mirror impl with NotImplementedError**

In `_mlp_op.py`, replace the `_cute_residual_mirror_impl` body (around line 269-296) with:

```python
def _cute_residual_mirror_impl(
    residual_buf: torch.Tensor,
    residual: torch.Tensor,
) -> None:
    """RETIRED 2026-04-27 (β-coop framework-output rewrite).

    Per the rewrite, β-coop reads `residual` directly as a graph-tracked
    op input via cute_beta_coop_run; no mirror copy needed.

    Stub raises so anything still calling this is loud, not silent.
    Phase 6 (next cleanup cycle) deletes the registration entirely.
    """
    raise NotImplementedError(
        "cute_residual_mirror is RETIRED. β-coop now reads residual "
        "directly via cute_beta_coop_run framework-output route."
    )
```

The fake impl and registration stay so `torch.ops.vllm.cute_residual_mirror` still resolves; only the impl raises if called.

- [ ] **Step 11.2: Stub cute_phase_e_dispatch the same way**

In `_mlp_op.py:182-216` (the `cute_phase_e_dispatch` op body), replace the impl body with `raise NotImplementedError("cute_phase_e_dispatch retired by β-coop framework-output rewrite")`. Same pattern as Step 11.1.

- [ ] **Step 11.3: Delete the layer-side copy/consume block**

In `vllm/nvllm/models/qwen3_5.py`, find the existing block at lines 480-490 (the `if impl is not None and getattr(impl, "_fusion_active", False):` consume block + the post-attn-LN dispatch). Replace with a single comment marker:

```python
        # ---- RETIRED 2026-04-27 (β-coop framework-output rewrite) ----
        # The framework-output route (handled above by _framework_output_route)
        # writes through caller-owned tensors. The legacy consume block
        # is no longer needed; layers either take the framework route OR
        # the canonical unified_attention_with_output path (above).
        # See feedback_splitting_op_runtime_dispatch +
        # 2026-04-27-beta-coop-rewrite-design.md.
```

(Per `feedback_comment_not_delete`: leave as comment, don't `git rm`.)

- [ ] **Step 11.4: Delete dead self.X allocations**

In `_backend.py:325-330`, the allocations:
```python
        self.rmsnorm_output = torch.empty(...)
        self.residual_output = torch.empty(...)
```
become dead state in the framework-output route. Comment them out per `feedback_comment_not_delete`:

```python
        # RETIRED 2026-04-27 (β-coop framework-output rewrite):
        # output buffers come from caller now via framework-output route.
        # Kept commented for one cycle in case a Phase D2 path needs them.
        # self.rmsnorm_output = torch.empty(
        #     max_num_seqs, hidden_dim, dtype=torch.bfloat16, device=device
        # )
        # self.residual_output = torch.empty(
        #     max_num_seqs, hidden_dim, dtype=torch.bfloat16, device=device
        # )
```

Same for `self.mlp_output` if it's nearby.

Also remove the legacy-fallback ternary expressions added in Steps 4.2-4.4 — those should now unconditionally use framework outputs since the framework-output route is the only live path.

### Task 12: Rebuild + final verification

- [ ] **Step 12.1: Rebuild #4 in tmux**

Run:
```bash
tmux kill-session -t phase5-rebuild 2>/dev/null || true
tmux new-session -d -s phase5-rebuild "docker build -f docker/Dockerfile.gb10 -t nvllm:gb10 . 2>&1 | tee /tmp/nvllm-build.log; echo BUILD_DONE_RC=\$? >> /tmp/nvllm-build.log"
echo "Build started"
```

Wait via background task:
```bash
until grep -q '^BUILD_DONE_RC=' /tmp/nvllm-build.log 2>/dev/null; do sleep 30; done
rc=$(grep '^BUILD_DONE_RC=' /tmp/nvllm-build.log | cut -d= -f2)
echo "BUILD_FINISHED rc=$rc"
if [ "$rc" != "0" ]; then echo "---LAST 60 LINES---"; tail -60 /tmp/nvllm-build.log; exit 1; fi
```

- [ ] **Step 12.2: Final GSM8K-50**

Launch + run:
```bash
docker rm -f nvllm 2>&1 | tail -1 || true
MAX_NUM_SEQS=1 CUTE_PHASE_E_FUSION=1 bash scripts/serve-cute.sh
```
Wait API up (background task per Step 8.3 pattern), then:
```bash
.venv/bin/python scripts/gsm8k_eval_50.py 2>&1 | tail -20
```
Expected: ≥ 90%.

- [ ] **Step 12.3: Capture final nsys trace**

Per AGENTS.md §4. Use the nsys procedure from Step 10.6, output to `benchmarks/nvllm/traces/beta_coop_framework_output/2026-04-27-shipped/baseline.nsys-rep`:

```bash
mkdir -p benchmarks/nvllm/traces/beta_coop_framework_output/2026-04-27-shipped
# [nsys command from Step 10.6, output to the new path]
ls -la benchmarks/nvllm/traces/beta_coop_framework_output/2026-04-27-shipped/
```

Expected: a `.nsys-rep` file present, 5-20 MB.

- [ ] **Step 12.4: Write nsys summary**

Create `benchmarks/nvllm/traces/beta_coop_framework_output/2026-04-27-shipped/summary.md` with:
- Commit hash (will be filled at commit time)
- Model: `ig1/Qwen3.5-27B-NVFP4`
- Config: `--max-num-seqs 1 --kv-cache-dtype fp8_e4m3 --max-model-len 65536 --compilation-config '{"cudagraph_mode":"PIECEWISE"}'`
- Kernel duration table extracted via:
  ```bash
  /opt/nvidia/nsight-systems/2024.6.1/host-linux-x64/nsys stats \
    --report cuda_kern_exec_sum baseline.nsys-rep | head -40
  ```
- "How to reproduce" section with the exact docker run commands.

- [ ] **Step 12.5: Stop container**

```bash
docker rm -f nvllm 2>&1 | tail -1
```

### Task 13: Update memory

**Files:**
- Modify: `~/.claude/projects/-home-natfii-docker-nvllm/memory/project_phase_e_beta_math_bug.md` (mark resolved)
- Create: `~/.claude/projects/-home-natfii-docker-nvllm/memory/feedback_framework_output_buffer_pattern.md` (new lesson)
- Modify: `~/.claude/projects/-home-natfii-docker-nvllm/memory/MEMORY.md` (index)

- [ ] **Step 13.1: Mark β-coop math bug resolved**

Update `project_phase_e_beta_math_bug.md` description and body to RESOLVED, citing the new commit (will fill SHA at commit time) and pointing to the spec + design + plan docs.

- [ ] **Step 13.2: Write `feedback_framework_output_buffer_pattern.md`**

Capture the lesson: when a fused kernel needs to produce multiple downstream-consumed outputs under PIECEWISE+CUDA graphs, write through caller-supplied tensors via `mutates_args` declared on a splitting-op-registered custom op. Do NOT use `impl.*` Python attributes — they're invisible to dynamo. Cross-reference `feedback_splitting_op_runtime_dispatch` (the underlying mechanism), `feedback_mutates_args_not_dce_safe` (sister failure mode), `feedback_op_body_capture_only` (the contrasting in-graph trap).

- [ ] **Step 13.3: Update MEMORY.md index**

Add the new feedback memory + update `project_phase_e_beta_math_bug` description.

### Task 14: Confirm + commit

- [ ] **Step 14.1: Final pre-commit verification**

Run:
```bash
git status --short
git diff feat/pre-beta-coop-rewrite-savepoint..HEAD --stat
git log --oneline -5
```
Confirm all changes are intentional and no stray files (e.g. `.pyc`, build artifacts).

- [ ] **Step 14.2: ASK USER for commit approval**

Per `feedback_commits`: do NOT commit without explicit user approval. Present:
- Files changed (from `git diff --stat`)
- Proposed commit message (single squashed commit per `feedback_commits` precedent for big rewrites)
- nsys trace path
- Memory updates

Wait for user "yes commit" before proceeding.

- [ ] **Step 14.3: Commit (after approval)**

```bash
git add vllm/config/compilation.py \
        vllm/v1/attention/backends/cute_paged/_backend.py \
        vllm/v1/attention/backends/cute_paged/_beta_coop_op.py \
        vllm/v1/attention/backends/cute_paged/_mlp_op.py \
        vllm/nvllm/models/qwen3_5.py \
        tests/v1/cute_paged/test_beta_coop_skeleton.py \
        docs/research/uber_kernel_migration/2026-04-27-beta-coop-rewrite-design.md \
        docs/research/uber_kernel_migration/2026-04-27-beta-coop-rewrite-plan.md \
        benchmarks/nvllm/traces/beta_coop_framework_output/2026-04-27-shipped/

git commit -m "$(cat <<'EOF'
feat(cute): β-coop framework-output-buffer rewrite — splitting-op runtime dispatch

Rewires β-coop's outputs from Python-attribute side-channel buffers
(impl.rmsnorm_output / residual_output / mlp_output) into framework-supplied
tensors that the layer pre-allocates and consumes as graph-tracked nodes.
The new vllm::cute_beta_coop_run op is registered in
vllm/config/compilation.py:_attention_ops as a PIECEWISE splitting boundary
— body runs eager Python every decode step between captured pieces,
mirroring vllm::unified_attention_with_output's production pattern.

Fixes the architectural trap that produced gibberish output (这种现象 × 256)
under PIECEWISE+CUDA graphs:
  - cute_residual_mirror DCE'd from captured FX graph (mutates_args alone
    not DCE-safe; readers were Python-attribute, invisible to dynamo)
  - β-coop read zeros from impl.residual_buf, computed attn_out + 0,
    cascade through 64 layers → gibberish
  - consume gate `if _fusion_active` specialized to always-else at trace
    time

The new op gets all inputs as explicit tensor args, writes all outputs
through caller-owned tensors. _backend.forward refactored to take output
kwargs; both β-coop and fall-through (paged in fusion mode + β-lite Phase
D MLP) write through the same three framework outputs. β-lite call site
also rewired to eliminate the same self.* side-channel pattern.

Retired:
  - cute_residual_mirror (NotImplementedError stub; removed next cycle)
  - cute_phase_e_dispatch (NotImplementedError stub)
  - Layer-side copy/consume block in qwen3_5.py
  - Dead self.rmsnorm_output / .residual_output / .mlp_output buffer
    allocations in _backend.py (commented per feedback_comment_not_delete)

Verification:
  - Phase 1 counter test: skeleton splitting op fires N×2 per N replays
    per shape (proves runtime dispatch model, not capture-only).
  - Phase 3 fall-through baseline: GSM8K-50 ≥ 90% with β-coop forced off.
  - Phase 4 β-coop active: GSM8K-50 ≥ 90%, coherent 256-token output,
    _BETA_COOP_FIRE_COUNTER ~256 per fusion-bound layer.
  - Phase 5 final: GSM8K-50 ≥ 90%, nsys trace at
    benchmarks/nvllm/traces/beta_coop_framework_output/2026-04-27-shipped/.

Spec: docs/research/uber_kernel_migration/2026-04-27-beta-coop-rewrite-design.md
Plan: docs/research/uber_kernel_migration/2026-04-27-beta-coop-rewrite-plan.md
Refs: feedback_splitting_op_runtime_dispatch, feedback_mutates_args_not_dce_safe,
      feedback_op_body_capture_only, feedback_no_silent_fallbacks,
      project_phase_e_beta_math_bug (RESOLVED).

Savepoint preserved at branch feat/pre-beta-coop-rewrite-savepoint.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

- [ ] **Step 14.4: Verify commit landed**

```bash
git log --oneline -3
git diff feat/pre-beta-coop-rewrite-savepoint..HEAD --stat | tail -5
```

**Phase 5 gate (must pass ALL):**
- GSM8K-50 ≥ 90% (final).
- nsys trace + summary committed.
- Memory updated.
- Commit reviewed and approved by user.
- `git log` shows the new commit on top of `7d429f1b7`.

**Rollback if Phase 5 fails:**
```bash
git reset --hard feat/pre-beta-coop-rewrite-savepoint
# OR per-file via /tmp/c2_rewrite_backup/
```

---

## Done state

After all five phases:
- β-coop produces coherent output under PIECEWISE+CUDA graphs at `MAX_NUM_SEQS=1, CUTE_PHASE_E_FUSION=1`.
- All input/output tensors flow through graph-tracked op args; no Python-attribute round-trip.
- `cute_residual_mirror` and `cute_phase_e_dispatch` retired.
- nsys evidence committed.
- Memory + spec + plan docs all reference each other.
- Savepoint branch `feat/pre-beta-coop-rewrite-savepoint` preserved for rollback safety.

The deeper kernel-level question (β-coop's cooperative-launch + atomic-counter spin-wait under graph replay) is decoupled: if Phase 4 gibberish appears, this rewrite is structurally correct (proven by Phase 3) and the kernel needs separate investigation.
