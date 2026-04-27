# C2 Diagnostic Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the C2 β-coop-vs-legacy diagnostic probe per `docs/research/uber_kernel_migration/2026-04-26-c2-diagnostic-spec.md` — an env-gated, no-rebuild Python harness that compares β-coop's outputs against the legacy path's outputs in dual-fire under PIECEWISE+graphs, dumps + halts on first divergence.

**Architecture:** Two new pure-Python modules in `vllm/v1/attention/backends/cute_paged/` (`_c2_diag.py`, `_c2_eager_replay.py`) plus a single env-gated block inserted in `vllm/nvllm/models/qwen3_5.py` between the consume gate and the post-attn-LN gate. Unit-tested where possible; integration verified through structured serve-cute runs. No CUDA rebuild — all changes are pure Python on bind-mounted source.

**Tech Stack:** Python 3.12, PyTorch 2.10, vLLM V1 (CuTe paged attention backend), pytest. Tests run inside the nvllm Docker container via `.venv/bin/python -m pytest`.

**Branch:** Fresh branch `diag/c2-beta-coop-vs-legacy` off `feat/uber-kernel-migration` HEAD `788697bff`.

---

## ⚠️ Status snapshot 2026-04-26 EOD (handoff for next session)

**Tasks 1-9 complete.** Tasks 10-13 awaiting rebuild #6 verification. Branch HEAD: `89040f66c`.

### Plan assumptions that turned out WRONG (4 architectural surprises)

The plan assumed "no CUDA rebuild — all changes are pure Python on bind-mounted source" and "diag block lives in eager Python between captured graph segments". Both assumptions failed in production. Six docker rebuilds were needed to land Tasks 1-8 in a working state. Resume this plan only after reading the 4 lessons below — do NOT re-attempt the bind-mount or `os.getenv` patterns the plan originally used.

**Blocker 1 — vLLM EngineCore strips most env vars.** docker `-e CUTE_C2_DIAG=1` reaches pid 1 (APIServer) but is stripped between pid 1 and pid 146 (EngineCore subprocess). Only ~3 of ~14 docker -e vars survive. See `memory:feedback_vllm_enginecore_env_strip`. **Fix shipped in `fdd2cbc98`:** serve-cute.sh writes `/tmp/c2_diag/ENV` from host shell vars before `docker run`; qwen3_5.py module-level prelude sources the file into `os.environ` at EngineCore import time.

**Blocker 2 — Image-bake required, not bind-mount.** The image's vllm tree is baked at build time, not bind-mounted. Surgical file mounts (`-v <host>/_c2_diag.py:<container>/_c2_diag.py:ro`) work for individual new files BUT shadow-mounting the whole `vllm/` tree breaks `_C.abi3.so` (it lives at `vllm/_C.abi3.so`, gets shadowed). Wholesale dir mount of `cute_paged/` works (preserves the .so) but is fragile. **Fix shipped in `916c6a210`:** drop bind-mounts entirely; rebuild after every change to in-container Python. Per `memory:feedback_no_shortcuts` clean rebuilds are correct here; ccache keeps them ~5 min after the first.

**Blocker 3 — `os.getenv(name, default)` returns "" not default for set-but-empty.** docker `-e CUTE_C2_DIAG_TOL_ATOL=""` (which serve-cute.sh writes when host shell var is unset) sets the container env to "". Then `float(os.getenv("...", "1e-2"))` becomes `float("")` and raises `ValueError`. Crashed compare_and_log inside torch.compile trace. See `memory:feedback_getenv_empty_string`. **Fix shipped in `7cbe00611`:** use `float(os.getenv("...") or "1e-2")` pattern. Two regression tests added.

**Blocker 4 — `@torch._dynamo.disable` rejected under vLLM's fullgraph compile.** vLLM compiles model.forward with fullgraph=True. `@torch._dynamo.disable` raises `Skip calling torch.compiler.disable()d function` per upstream pytorch/pytorch#167927. `torch.compiler.allow_in_graph` is also insufficient — Dynamo writes the call into FX as opaque but still fake-executes the body. **Only working pattern: `direct_register_custom_op` with explicit `fake_impl`** mirroring `cute_residual_mirror` in `_mlp_op.py`. See `memory:feedback_dynamo_disable_fullgraph`. **Fix shipped in `240f58a32`:** registered as `torch.ops.vllm.cute_c2_diag_compare`. Bonus runtime guards in the impl: capture-skip via `torch.cuda.is_current_stream_capturing()` (avoid `cudaErrorStreamCaptureInvalidated` from `.item()`) AND prefill-skip when `nat > beta_rmsnorm_output.shape[0]` (impl buffers are max-num-seqs sized, not max-model-len). Latter shipped in `89040f66c` after rebuild #5 surfaced the prefill warmup crash.

### What actually shipped vs the plan's literal text

- **Tasks 1-7:** Shipped exactly per plan text (TDD + commits). 17 unit tests passing.
- **Task 8:** Shipped per plan text initially (`8c9a496b6`); refactored 3× during Tasks 9-10 to address blockers above. The diag call site is now `torch.ops.vllm.cute_c2_diag_compare(positional, args, ...)` instead of `_c2_diag.compare_and_log(keyword=args, ...)`.
- **Task 9:** Passed first try (probe disabled produces zero `[C2_DIAG]` lines + coherent output). Confirmed production behavior unchanged when CUTE_C2_DIAG is unset.
- **Tasks 10-13:** Pending. Resume after rebuild #6 lands and verify Task 10 step 10.4-10.7 (active probe count + verdict distribution).

### serve-cute.sh changes (`252ab183f` + `fdd2cbc98` + `916c6a210`)

Three new permanent additions (independent of diag-active state):
1. `mkdir -p /tmp/c2_diag` + write `/tmp/c2_diag/ENV` from CUTE_C2_*-prefixed host shell vars
2. `-v /tmp/c2_diag:/tmp/c2_diag` mount
3. NO source bind-mounts (rolled back)

Production behavior is unchanged when CUTE_C2_DIAG is unset (the env-load prelude in qwen3_5.py is a no-op when /tmp/c2_diag/ENV doesn't contain the key with a non-empty value).

### Resume instructions for next session

1. **Verify rebuild #6 succeeded:** `tail -5 /tmp/nvllm-build.log` should show `BUILD_DONE_RC=0` and a fresh image sha. (Watcher task `bxgr35d9l` from prior session; may need re-arming.)
2. **Launch Task 10:** `CUTE_C2_DIAG=1 bash scripts/serve-cute.sh`. Wait for `/v1/models` to respond. Send 256-token prompt (plan Step 10.3). Capture `docker logs nvllm 2>&1 | grep -c '\[C2_DIAG\]'` — expect ~3500-4500 lines, 16 distinct full-attn layer indices.
3. **Then continue per plan** through Tasks 11 (forced-divergence self-test), 12 (rung-0 sanity), 13 (main diagnostic + interpretation).

If rebuild #6 also fails: most likely yet another fullgraph/capture interaction. Read the new error carefully BEFORE rebuilding again — most fixes are 1-line + 1 rebuild; iteration is cheap once the right pattern is identified.

### Commit chain on `diag/c2-beta-coop-vs-legacy` (most-recent-first)

```
89040f66c  diag(c2): skip op when nat exceeds beta buffer size (prefill case)
240f58a32  diag(c2): register cute_c2_diag_compare as direct custom op
6c2e26d50  diag(c2): pre-commit reformat of qwen3_5.py (cosmetic)
62d18ca0c  diag(c2): use torch.compiler.allow_in_graph (rolled into 240f58a32)
7cbe00611  diag(c2): handle set-but-empty env vars in compare_and_log + _inject_noise
b723a3ed0  diag(c2): @torch._dynamo.disable on compare_and_log (rolled into 62d18ca0c)
987687fd3  diag(c2): use surgical file mounts (rolled back in 916c6a210)
fdd2cbc98  diag(c2): workaround vLLM EngineCore env stripping
916c6a210  diag(c2): drop bind-mounts (rebuild instead per feedback_no_shortcuts)
252ab183f  diag(c2): add C2 diag env passthrough + /tmp/c2_diag mount to serve-cute
8c9a496b6  diag(c2): wire C2 diagnostic call site into qwen3_5 decoder (Task 8)
ba46baae3  diag(c2): document CUDA-graph call-site constraint (Phase-1 polish)
ab254c1c2  diag(c2): add compare_and_log public dispatch (Task 6)
b38166c51  diag(c2): add _inject_noise self-test helper (Task 5)
fc4e3ee95  diag(c2): add assert_no_flashinfer_autotune host-safety guard (Task 4)
ec358ad33  diag(c2): add next_step_idx step counter (Task 3)
695831ed6  diag(c2): add _dump_on_divergence with tmp_path tests (Task 2)
c4aa1be90  diag(c2): add _compare_pair primitive with unit tests (Task 1)
788697bff  (parent — feat/uber-kernel-migration HEAD)
```

---

## Pre-flight: branch setup

- [ ] **Step 0.1: Confirm working tree clean and on `feat/uber-kernel-migration`**

Run: `git status && git rev-parse --short HEAD`
Expected: clean tree, HEAD `788697bff`.

- [ ] **Step 0.2: Create and check out new branch**

Run: `git checkout -b diag/c2-beta-coop-vs-legacy`
Expected: switched to new branch.

- [ ] **Step 0.3: Verify nvllm container is stopped**

Run: `docker ps --filter name=nvllm --format '{{.Names}}'`
Expected: empty output.
If running: `docker stop nvllm` (frees ~50 GB unified memory before tests).

---

## File Structure

**New files:**
- `vllm/v1/attention/backends/cute_paged/_c2_diag.py` — primary probe module. Pure Python. Public API: `compare_and_log()`, `next_step_idx()`, `assert_no_flashinfer_autotune()`. Internal helpers: `_compare_pair()`, `_dump_on_divergence()`, `_inject_noise()`. Reads env vars: `CUTE_C2_DIAG`, `CUTE_C2_DIAG_TOL_ATOL`, `CUTE_C2_DIAG_TOL_RTOL`, `CUTE_C2_DIAG_DUMP_DIR`, `CUTE_C2_DIAG_INJECT_NOISE`.
- `vllm/v1/attention/backends/cute_paged/_c2_eager_replay.py` — stashed companion harness. Pure Python. Public API: `EagerReplayHook` class with `snapshot()` and `replay_and_compare()`. Reads env: `CUTE_C2_DIAG_EAGER`. Off by default; only enabled if (b) primary probe is inconclusive.
- `tests/v1/cute_paged/test_c2_diag.py` — unit tests for `_c2_diag.py` (synthetic-tensor tests for compare logic, dump bundle structure, noise injection, autotune assertion).

**Modified files:**
- `vllm/nvllm/models/qwen3_5.py:466-476` — additive: insert env-gated diag-call block between consume-gate and post-attn-LN gate. ~15 lines added, no existing lines removed.

**Documentation files (committed at end of phase 4):**
- `docs/research/uber_kernel_migration/2026-04-27-c2-diagnostic-results.md` — created at the end of the main diagnostic run, records observed behavior + inferred next-design direction.

---

## Phase 1 — Core diag module (TDD)

Each task in this phase follows the TDD rhythm: write failing test → run → verify FAIL → implement minimal code → run → verify PASS → commit. Tests run via `.venv/bin/python -m pytest tests/v1/cute_paged/test_c2_diag.py -v`.

### Task 1: Comparison primitive — `_compare_pair()`

**Files:**
- Create: `vllm/v1/attention/backends/cute_paged/_c2_diag.py`
- Test: `tests/v1/cute_paged/test_c2_diag.py`

- [ ] **Step 1.1: Write the failing tests**

Create `tests/v1/cute_paged/test_c2_diag.py`:

```python
"""Unit tests for vllm.v1.attention.backends.cute_paged._c2_diag.

These tests exercise the comparison primitives with synthetic tensors so
they run on any host (no GPU required, no full vLLM bring-up). The
integration tests (probe wired into qwen3_5.py) are manual serve-cute
runs documented in the C2 diagnostic plan.
"""

from __future__ import annotations

import torch

from vllm.v1.attention.backends.cute_paged import _c2_diag


def test_compare_pair_within_tolerance() -> None:
    """Two BF16 tensors within unit-roundoff should compare OK."""
    torch.manual_seed(0)
    a = torch.randn(4, 5120, dtype=torch.bfloat16)
    b = a + 1e-4 * torch.randn_like(a)
    result = _c2_diag._compare_pair(a, b, atol=1e-2, rtol=1e-2)
    assert result["ok"] is True
    assert result["linf"] < 1e-2


def test_compare_pair_diverges_on_large_offset() -> None:
    """Two BF16 tensors with a >atol offset should compare DIVERGED."""
    torch.manual_seed(0)
    a = torch.randn(4, 5120, dtype=torch.bfloat16)
    b = a + 1.0  # constant offset > atol
    result = _c2_diag._compare_pair(a, b, atol=1e-2, rtol=1e-2)
    assert result["ok"] is False
    assert result["linf"] > 0.5


def test_compare_pair_returns_required_keys() -> None:
    """Result dict must contain linf, rel_med, ok keys for log formatting."""
    a = torch.randn(2, 4, dtype=torch.bfloat16)
    result = _c2_diag._compare_pair(a, a.clone(), atol=1e-2, rtol=1e-2)
    assert set(result.keys()) >= {"linf", "rel_med", "ok"}
```

- [ ] **Step 1.2: Run tests, expect FAIL**

Run: `.venv/bin/python -m pytest tests/v1/cute_paged/test_c2_diag.py -v`
Expected: 3 errors, all `ModuleNotFoundError: No module named '...._c2_diag'` or similar import error.

- [ ] **Step 1.3: Create `_c2_diag.py` skeleton with `_compare_pair()`**

Create `vllm/v1/attention/backends/cute_paged/_c2_diag.py`:

```python
"""C2 diagnostic probe — β-coop vs legacy comparison harness.

Spec: docs/research/uber_kernel_migration/2026-04-26-c2-diagnostic-spec.md.

Env-gated. When CUTE_C2_DIAG=1, compares β-coop's outputs (impl.rmsnorm_output,
impl.residual_output) against the legacy Python path's outputs (hidden_states,
residual after post_attention_layernorm) at every full-attn layer. On first
divergence above tolerance, dumps a forensics bundle and raises RuntimeError.

The module is import-safe regardless of CUTE_C2_DIAG setting; the call site in
qwen3_5.py guards every entry point with `os.getenv("CUTE_C2_DIAG") == "1"`.
"""

from __future__ import annotations

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
```

- [ ] **Step 1.4: Run tests, expect PASS**

Run: `.venv/bin/python -m pytest tests/v1/cute_paged/test_c2_diag.py -v`
Expected: 3 passed.

- [ ] **Step 1.5: Commit**

```bash
git add vllm/v1/attention/backends/cute_paged/_c2_diag.py \
        tests/v1/cute_paged/test_c2_diag.py
git commit -m "diag(c2): add _compare_pair primitive with unit tests

Pure-Python BF16 comparison utility that returns L∞ + median(rel) + ok
verdict. Computes in FP32 to avoid BF16-roundoff perturbing the stats.
First commit toward the C2 β-coop-vs-legacy diagnostic per spec
docs/research/uber_kernel_migration/2026-04-26-c2-diagnostic-spec.md.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
"
```

---

### Task 2: Dump-on-divergence — `_dump_on_divergence()`

**Files:**
- Modify: `vllm/v1/attention/backends/cute_paged/_c2_diag.py`
- Test: `tests/v1/cute_paged/test_c2_diag.py`

- [ ] **Step 2.1: Write the failing test**

Append to `tests/v1/cute_paged/test_c2_diag.py`:

```python
def test_dump_on_divergence_writes_bundle(tmp_path, monkeypatch) -> None:
    """Dump should write a torch.save bundle with all required keys."""
    monkeypatch.setenv("CUTE_C2_DIAG_DUMP_DIR", str(tmp_path))
    legacy_h = torch.randn(2, 4, dtype=torch.bfloat16)
    legacy_r = torch.randn(2, 4, dtype=torch.bfloat16)
    beta_h = torch.randn(2, 4, dtype=torch.bfloat16)
    beta_r = torch.randn(2, 4, dtype=torch.bfloat16)
    _c2_diag._dump_on_divergence(
        layer_idx=3,
        step_idx=42,
        nat=2,
        atol=1e-2,
        rtol=1e-2,
        legacy_hidden=legacy_h,
        legacy_residual=legacy_r,
        beta_rmsnorm_output=beta_h,
        beta_residual_output=beta_r,
    )
    dump_path = tmp_path / "layer3_step42.pt"
    assert dump_path.exists()
    bundle = torch.load(dump_path)
    assert bundle["layer_idx"] == 3
    assert bundle["step_idx"] == 42
    assert bundle["nat"] == 2
    assert bundle["atol"] == 1e-2
    assert bundle["rtol"] == 1e-2
    assert torch.equal(bundle["legacy_hidden"], legacy_h)
    assert torch.equal(bundle["beta_rmsnorm_output"], beta_h)


def test_dump_on_divergence_creates_dir(tmp_path, monkeypatch) -> None:
    """Dump directory should be created if it does not exist."""
    target = tmp_path / "subdir" / "deeper"
    monkeypatch.setenv("CUTE_C2_DIAG_DUMP_DIR", str(target))
    _c2_diag._dump_on_divergence(
        layer_idx=0, step_idx=0, nat=1, atol=1e-2, rtol=1e-2,
        legacy_hidden=torch.zeros(1, 1, dtype=torch.bfloat16),
        legacy_residual=torch.zeros(1, 1, dtype=torch.bfloat16),
        beta_rmsnorm_output=torch.zeros(1, 1, dtype=torch.bfloat16),
        beta_residual_output=torch.zeros(1, 1, dtype=torch.bfloat16),
    )
    assert (target / "layer0_step0.pt").exists()
```

- [ ] **Step 2.2: Run tests, expect FAIL**

Run: `.venv/bin/python -m pytest tests/v1/cute_paged/test_c2_diag.py -v`
Expected: 2 new tests fail with `AttributeError: module ... has no attribute '_dump_on_divergence'`.

- [ ] **Step 2.3: Implement `_dump_on_divergence()`**

Append to `vllm/v1/attention/backends/cute_paged/_c2_diag.py`:

```python
import os
from pathlib import Path


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
    dump_dir = Path(os.getenv("CUTE_C2_DIAG_DUMP_DIR", "/tmp/c2_diag"))
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
```

- [ ] **Step 2.4: Run tests, expect PASS**

Run: `.venv/bin/python -m pytest tests/v1/cute_paged/test_c2_diag.py -v`
Expected: 5 passed.

- [ ] **Step 2.5: Commit**

```bash
git add vllm/v1/attention/backends/cute_paged/_c2_diag.py \
        tests/v1/cute_paged/test_c2_diag.py
git commit -m "diag(c2): add _dump_on_divergence with tmp_path tests

Writes torch.save bundle to CUTE_C2_DIAG_DUMP_DIR (default /tmp/c2_diag).
Creates intermediate directories. Bundles BF16 tensors as CPU clones to
avoid hanging onto GPU memory after a divergence halts the run.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
"
```

---

### Task 3: Step counter — `next_step_idx()`

**Files:**
- Modify: `vllm/v1/attention/backends/cute_paged/_c2_diag.py`
- Test: `tests/v1/cute_paged/test_c2_diag.py`

- [ ] **Step 3.1: Write the failing test**

Append to `tests/v1/cute_paged/test_c2_diag.py`:

```python
def test_next_step_idx_increments() -> None:
    """next_step_idx returns monotonically increasing integers from 0."""
    _c2_diag._reset_step_counter_for_test()
    assert _c2_diag.next_step_idx() == 0
    assert _c2_diag.next_step_idx() == 1
    assert _c2_diag.next_step_idx() == 2


def test_reset_step_counter_for_test() -> None:
    """The reset hook returns the counter to 0 (used by tests only)."""
    _c2_diag.next_step_idx()
    _c2_diag.next_step_idx()
    _c2_diag._reset_step_counter_for_test()
    assert _c2_diag.next_step_idx() == 0
```

- [ ] **Step 3.2: Run tests, expect FAIL**

Run: `.venv/bin/python -m pytest tests/v1/cute_paged/test_c2_diag.py -v`
Expected: 2 new tests fail (`AttributeError: ... has no attribute 'next_step_idx'`).

- [ ] **Step 3.3: Implement `next_step_idx()`**

Append to `vllm/v1/attention/backends/cute_paged/_c2_diag.py`:

```python
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
```

- [ ] **Step 3.4: Run tests, expect PASS**

Run: `.venv/bin/python -m pytest tests/v1/cute_paged/test_c2_diag.py -v`
Expected: 7 passed.

- [ ] **Step 3.5: Commit**

```bash
git add vllm/v1/attention/backends/cute_paged/_c2_diag.py \
        tests/v1/cute_paged/test_c2_diag.py
git commit -m "diag(c2): add next_step_idx step counter

Module-local counter incremented per call from layer-0. Used only for
log readability — the layer index is the load-bearing identifier.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
"
```

---

### Task 4: Flashinfer-autotune assert — `assert_no_flashinfer_autotune()`

**Files:**
- Modify: `vllm/v1/attention/backends/cute_paged/_c2_diag.py`
- Test: `tests/v1/cute_paged/test_c2_diag.py`

- [ ] **Step 4.1: Write the failing test**

Append to `tests/v1/cute_paged/test_c2_diag.py`:

```python
import pytest


def test_assert_no_flashinfer_autotune_disabled_passes(monkeypatch) -> None:
    """When autotune is disabled (default), the assert is a no-op."""
    # Default vllm config has autotune off; passing a stub config is enough.
    class _Stub:
        enable_flashinfer_autotune = False
    _c2_diag.assert_no_flashinfer_autotune(_Stub())  # must not raise


def test_assert_no_flashinfer_autotune_enabled_raises() -> None:
    """When autotune is enabled, the assert raises with a clear message."""
    class _Stub:
        enable_flashinfer_autotune = True
    with pytest.raises(RuntimeError, match="flashinfer autotune"):
        _c2_diag.assert_no_flashinfer_autotune(_Stub())
```

- [ ] **Step 4.2: Run tests, expect FAIL**

Run: `.venv/bin/python -m pytest tests/v1/cute_paged/test_c2_diag.py -v`
Expected: 2 new tests fail (`AttributeError`).

- [ ] **Step 4.3: Implement `assert_no_flashinfer_autotune()`**

Append to `vllm/v1/attention/backends/cute_paged/_c2_diag.py`:

```python
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
```

- [ ] **Step 4.4: Run tests, expect PASS**

Run: `.venv/bin/python -m pytest tests/v1/cute_paged/test_c2_diag.py -v`
Expected: 9 passed.

- [ ] **Step 4.5: Commit**

```bash
git add vllm/v1/attention/backends/cute_paged/_c2_diag.py \
        tests/v1/cute_paged/test_c2_diag.py
git commit -m "diag(c2): add assert_no_flashinfer_autotune host-safety guard

Refuses to start the diagnostic when flashinfer autotune is on, per
feedback_flashinfer_autotune_sm120 (host reboot risk).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
"
```

---

### Task 5: Noise injection helper — `_inject_noise()`

**Files:**
- Modify: `vllm/v1/attention/backends/cute_paged/_c2_diag.py`
- Test: `tests/v1/cute_paged/test_c2_diag.py`

- [ ] **Step 5.1: Write the failing test**

Append to `tests/v1/cute_paged/test_c2_diag.py`:

```python
def test_inject_noise_disabled_returns_unchanged(monkeypatch) -> None:
    """When CUTE_C2_DIAG_INJECT_NOISE is unset, tensor is returned unchanged."""
    monkeypatch.delenv("CUTE_C2_DIAG_INJECT_NOISE", raising=False)
    a = torch.ones(2, 4, dtype=torch.bfloat16)
    out = _c2_diag._inject_noise(a)
    assert torch.equal(out, a)


def test_inject_noise_enabled_adds_offset(monkeypatch) -> None:
    """When CUTE_C2_DIAG_INJECT_NOISE=1.0, the offset is added in-place."""
    monkeypatch.setenv("CUTE_C2_DIAG_INJECT_NOISE", "1.0")
    a = torch.ones(2, 4, dtype=torch.bfloat16)
    out = _c2_diag._inject_noise(a)
    expected = a + 1.0
    assert torch.equal(out, expected)


def test_inject_noise_invalid_value_raises(monkeypatch) -> None:
    """Non-float CUTE_C2_DIAG_INJECT_NOISE values raise loud."""
    monkeypatch.setenv("CUTE_C2_DIAG_INJECT_NOISE", "abc")
    with pytest.raises(ValueError):
        _c2_diag._inject_noise(torch.zeros(1, 1))
```

- [ ] **Step 5.2: Run tests, expect FAIL**

Run: `.venv/bin/python -m pytest tests/v1/cute_paged/test_c2_diag.py -v`
Expected: 3 new tests fail (`AttributeError`).

- [ ] **Step 5.3: Implement `_inject_noise()`**

Append to `vllm/v1/attention/backends/cute_paged/_c2_diag.py`:

```python
def _inject_noise(t: torch.Tensor) -> torch.Tensor:
    """Add a constant offset to t when CUTE_C2_DIAG_INJECT_NOISE is set.

    Used by Phase-3 self-test: verifies the probe halts when divergence
    is forced, ensuring the comparison and dump paths actually fire on
    real divergence (not just rejected by tolerance).

    Returns t unchanged when the env var is unset. Raises ValueError on
    non-float values (don't silently fall back per feedback_no_silent_fallbacks).
    """
    raw = os.getenv("CUTE_C2_DIAG_INJECT_NOISE")
    if raw is None:
        return t
    offset = float(raw)  # raises ValueError on non-float per spec
    return t + offset
```

- [ ] **Step 5.4: Run tests, expect PASS**

Run: `.venv/bin/python -m pytest tests/v1/cute_paged/test_c2_diag.py -v`
Expected: 12 passed.

- [ ] **Step 5.5: Commit**

```bash
git add vllm/v1/attention/backends/cute_paged/_c2_diag.py \
        tests/v1/cute_paged/test_c2_diag.py
git commit -m "diag(c2): add _inject_noise self-test helper

Used by the diag-active forced-divergence pre-flight test (Section 5.2 of
the spec). Off when env unset; raises ValueError on non-float values.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
"
```

---

### Task 6: `compare_and_log()` — public dispatch entry point

**Files:**
- Modify: `vllm/v1/attention/backends/cute_paged/_c2_diag.py`
- Test: `tests/v1/cute_paged/test_c2_diag.py`

- [ ] **Step 6.1: Write the failing tests**

Append to `tests/v1/cute_paged/test_c2_diag.py`:

```python
def test_compare_and_log_ok_path_no_raise(monkeypatch, capsys) -> None:
    """When both pairs match, compare_and_log logs OK and does not raise."""
    monkeypatch.delenv("CUTE_C2_DIAG_INJECT_NOISE", raising=False)
    torch.manual_seed(0)
    legacy_h = torch.randn(2, 4, dtype=torch.bfloat16)
    legacy_r = torch.randn(2, 4, dtype=torch.bfloat16)
    _c2_diag.compare_and_log(
        layer_idx=0,
        step_idx=0,
        nat=2,
        legacy_hidden=legacy_h,
        legacy_residual=legacy_r,
        beta_rmsnorm_output=legacy_h.clone(),
        beta_residual_output=legacy_r.clone(),
    )
    captured = capsys.readouterr()
    combined = captured.out + captured.err
    assert "[C2_DIAG]" in combined
    assert "OK" in combined


def test_compare_and_log_diverged_path_raises(monkeypatch, tmp_path) -> None:
    """When divergence above tolerance, compare_and_log dumps + raises."""
    monkeypatch.delenv("CUTE_C2_DIAG_INJECT_NOISE", raising=False)
    monkeypatch.setenv("CUTE_C2_DIAG_DUMP_DIR", str(tmp_path))
    torch.manual_seed(0)
    legacy_h = torch.randn(2, 4, dtype=torch.bfloat16)
    legacy_r = torch.randn(2, 4, dtype=torch.bfloat16)
    beta_h = legacy_h + 1.0  # large divergence
    beta_r = legacy_r.clone()
    with pytest.raises(RuntimeError, match=r"\[C2_DIAG\] diverged"):
        _c2_diag.compare_and_log(
            layer_idx=3,
            step_idx=42,
            nat=2,
            legacy_hidden=legacy_h,
            legacy_residual=legacy_r,
            beta_rmsnorm_output=beta_h,
            beta_residual_output=beta_r,
        )
    assert (tmp_path / "layer3_step42.pt").exists()


def test_compare_and_log_skips_when_nat_zero(monkeypatch) -> None:
    """nat=0 means empty decode step; skip silently, no error."""
    legacy_h = torch.randn(2, 4, dtype=torch.bfloat16)
    _c2_diag.compare_and_log(
        layer_idx=0, step_idx=0, nat=0,
        legacy_hidden=legacy_h, legacy_residual=legacy_h,
        beta_rmsnorm_output=legacy_h, beta_residual_output=legacy_h,
    )  # must not raise
```

- [ ] **Step 6.2: Run tests, expect FAIL**

Run: `.venv/bin/python -m pytest tests/v1/cute_paged/test_c2_diag.py -v`
Expected: 3 new tests fail (`AttributeError: ... has no attribute 'compare_and_log'`).

- [ ] **Step 6.3: Implement `compare_and_log()`**

Append to `vllm/v1/attention/backends/cute_paged/_c2_diag.py`:

```python
import sys


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
    """
    if nat == 0:
        return

    atol = float(os.getenv("CUTE_C2_DIAG_TOL_ATOL", "1e-2"))
    rtol = float(os.getenv("CUTE_C2_DIAG_TOL_RTOL", "1e-2"))

    # Self-test injection (CUTE_C2_DIAG_INJECT_NOISE=1.0 forces divergence).
    beta_h = _inject_noise(beta_rmsnorm_output[:nat])

    h = _compare_pair(legacy_hidden[:nat], beta_h, atol=atol, rtol=rtol)
    r = _compare_pair(
        legacy_residual[:nat], beta_residual_output[:nat],
        atol=atol, rtol=rtol,
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
```

- [ ] **Step 6.4: Run tests, expect PASS**

Run: `.venv/bin/python -m pytest tests/v1/cute_paged/test_c2_diag.py -v`
Expected: 15 passed.

- [ ] **Step 6.5: Commit**

```bash
git add vllm/v1/attention/backends/cute_paged/_c2_diag.py \
        tests/v1/cute_paged/test_c2_diag.py
git commit -m "diag(c2): add compare_and_log public dispatch entry point

Reads tolerances from env (defaults 1e-2/1e-2). Logs OK/DIVERGED to
stderr per call. On divergence: dumps bundle, raises RuntimeError with
layer+step. nat=0 (empty decode) skipped silently per spec Section 4.1.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
"
```

---

## Phase 2 — Wire into model

### Task 7: Verify `CUTE_ATTN_FUSION` is the rung-0 disable knob

**Files:**
- Read-only: `vllm/v1/attention/backends/cute_paged/_backend.py`

The spec's "TBD env name" was already resolved during plan-writing: `CUTE_ATTN_FUSION` (default `"1"`) at `_backend.py:645,699`. Setting `CUTE_ATTN_FUSION=0` disables β-coop attn fusion. This task just records that fact in the diagnostic results doc — no code change needed.

- [ ] **Step 7.1: Confirm by re-reading the env knob**

Run: `grep -n "CUTE_ATTN_FUSION" vllm/v1/attention/backends/cute_paged/_backend.py`
Expected: lines 645 + 651 + 699 referencing `CUTE_ATTN_FUSION`.

- [ ] **Step 7.2: Confirm there is NO existing `CUTE_FUSION_DISABLE`**

Run: `grep -rn "CUTE_FUSION_DISABLE" vllm/`
Expected: empty output.

No code commit for this task — just confirmation.

---

### Task 8: Wire diag block into `qwen3_5.py`

**Files:**
- Modify: `vllm/nvllm/models/qwen3_5.py:466-476` (additive insertion only)

This task is integration-level — there's no clean unit test that doesn't require the full model load. Verification is in Phase 3 via serve-cute.

- [ ] **Step 8.1: Read the consume gate + post-attn-LN region**

Run: `sed -n '460,500p' vllm/nvllm/models/qwen3_5.py`
Expected: shows the `if impl is not None and getattr(impl, "_fusion_active", False):` block at ~466-476 followed by the `if not getattr(impl, "_fusion_active", False): post_attention_layernorm(...)` block at ~490-496.

The diag block goes immediately AFTER the post-attn-LN block (NOT after the consume gate) — `legacy_hidden` and `legacy_residual` must be in their post-LN state for the comparison to be meaningful.

- [ ] **Step 8.2: Insert the diag block after `post_attention_layernorm`**

Find the existing `post_attention_layernorm` gate (~lines 490-496):

```python
        if not getattr(impl, "_fusion_active", False):
            hidden_states, residual = self.post_attention_layernorm(
                hidden_states, residual
            )
            _ct_mark("post_attn_ln")
        else:
            _ct_mark("post_attn_skip")
```

Insert immediately AFTER that block (before any subsequent layer-scale or MLP code), the C2 diag call:

```python
        if not getattr(impl, "_fusion_active", False):
            hidden_states, residual = self.post_attention_layernorm(
                hidden_states, residual
            )
            _ct_mark("post_attn_ln")
        else:
            _ct_mark("post_attn_skip")

        # --- C2 DIAG (env-gated, per spec
        # docs/research/uber_kernel_migration/2026-04-26-c2-diagnostic-spec.md)
        # Compares β-coop's outputs (impl.rmsnorm_output/residual_output)
        # against the legacy post-attn-LN outputs (hidden_states/residual)
        # in dual-fire mode under PIECEWISE+graphs. Off by default.
        if (
            os.getenv("CUTE_C2_DIAG") == "1"
            and impl is not None
            and getattr(impl, "_fusion_bound", False)
            and self.layer_type == "full_attention"
            and nat > 0
        ):
            from vllm.v1.attention.backends.cute_paged import _c2_diag
            step_idx = (
                _c2_diag.next_step_idx() if self.layer_idx == 0
                else max(0, _c2_diag._STEP_COUNTER - 1)
            )
            _c2_diag.compare_and_log(
                layer_idx=self.layer_idx,
                step_idx=step_idx,
                nat=nat,
                legacy_hidden=hidden_states,
                legacy_residual=residual,
                beta_rmsnorm_output=impl.rmsnorm_output,
                beta_residual_output=impl.residual_output,
            )
```

- [ ] **Step 8.3: Verify `os` is imported at the top of qwen3_5.py**

Run: `grep -n "^import os" vllm/nvllm/models/qwen3_5.py`
Expected: a line near the top of the file. If missing, add `import os` to the existing import block.

- [ ] **Step 8.4: Confirm `self.layer_idx` resolves correctly**

`self.layer_idx` is set at `qwen3_5.py:306` via `extract_layer_index(prefix)` (verified during plan-writing). No conditional needed — the diag block can use `layer_idx=self.layer_idx` directly.

Sanity-check by running:
`grep -n "self\.layer_idx = extract_layer_index" vllm/nvllm/models/qwen3_5.py`
Expected: one match at line 306.

- [ ] **Step 8.5: Smoke-import the modified file**

Run: `.venv/bin/python -c "import vllm.nvllm.models.qwen3_5"`
Expected: no error; clean import.

- [ ] **Step 8.6: Commit**

```bash
git add vllm/nvllm/models/qwen3_5.py
git commit -m "diag(c2): wire C2 diagnostic call site into qwen3_5 decoder

Inserts an env-gated (CUTE_C2_DIAG=1) call to _c2_diag.compare_and_log
after the post_attention_layernorm gate. Off by default — production
behavior is unchanged when CUTE_C2_DIAG is unset.

Per spec Section 2.1 + 3.1, the diag block lives in eager Python
between captured graph segments and reads only graph-output tensors.
No new opaque ops, no kernel changes, no framework op signature changes.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
"
```

---

## Phase 3 — Verification runs

These tasks are manual integration checks, not pytest-driven. Each task starts the `nvllm` container with a specific env config and inspects the output. Container restart commands assume serve-cute.sh is the canonical entry point (it should be — verify at run time).

### Task 9: Pre-flight — no-op check (probe disabled)

- [ ] **Step 9.1: Reload Python source into the running container or rebuild image**

Since the changes are pure Python and the source is bind-mounted (per `feedback_docker_bindmount`), no full rebuild is needed. Either:

- (A) Copy modified files into the running container if it's still up: `docker cp vllm/v1/attention/backends/cute_paged/_c2_diag.py nvllm:/app/vllm/v1/attention/backends/cute_paged/_c2_diag.py` etc., then `docker restart nvllm`.
- (B) Stop the container (if running), launch a fresh `serve-cute` against the bind-mounted tree.

(B) is cleaner per `feedback_no_shortcuts`. Run: `docker stop nvllm 2>/dev/null; bash docker/serve-cute.sh` (or whatever the canonical launch script is — verify at run time).

- [ ] **Step 9.2: Confirm CUTE_C2_DIAG is NOT set in container env**

Run: `docker exec nvllm env | grep CUTE_C2_DIAG`
Expected: empty output.

- [ ] **Step 9.3: Send standard probe via /v1/completions**

Run:
```
curl -s http://localhost:8000/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"default","prompt":"What is the capital of France?","max_tokens":32,"temperature":0.0}' \
  | python -c "import sys,json; print(json.load(sys.stdin)['choices'][0]['text'])"
```
Expected: contains "Paris".

- [ ] **Step 9.4: Confirm no `[C2_DIAG]` lines in container logs**

Run: `docker logs nvllm 2>&1 | grep '\[C2_DIAG\]' | head -5`
Expected: empty output.

- [ ] **Step 9.5: Spot-check 5 nominal completions for behavioral parity**

Send 5 different short prompts; confirm each returns coherent text (manual eyeball check). The full GSM8K-50 gate is reserved for after the diagnostic itself runs (Phase 4 final task).

- [ ] **Step 9.6: Commit a results note (if anything notable observed)**

If the smoke check reveals any unexpected behavior, append a section to `docs/research/uber_kernel_migration/2026-04-27-c2-diagnostic-results.md` (file may not yet exist; create if needed) and commit. If clean, no commit — proceed to Task 10.

---

### Task 10: Pre-flight — active check (probe enabled, dual-fire)

- [ ] **Step 10.1: Restart container with `CUTE_C2_DIAG=1`**

Stop the current container, edit `docker/serve-cute.sh` (or its compose file) to inject `CUTE_C2_DIAG=1` as an env var, restart. Or pass via `docker run -e CUTE_C2_DIAG=1 ...`.

- [ ] **Step 10.2: Confirm probe is active**

Run: `docker exec nvllm env | grep CUTE_C2_DIAG`
Expected: `CUTE_C2_DIAG=1`.

- [ ] **Step 10.3: Send a 256-token prompt**

Run:
```
curl -s http://localhost:8000/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"default","prompt":"Tell me a story about a robot.","max_tokens":256,"temperature":0.0}' \
  > /dev/null
```

- [ ] **Step 10.4: Count `[C2_DIAG]` lines**

Run: `docker logs nvllm 2>&1 | grep -c '\[C2_DIAG\]'`
Expected: roughly `255 × 16 ≈ 4080`. Acceptable range: 3500-4500. Significantly fewer means the gate is wrong; significantly more means the step counter is double-incrementing.

- [ ] **Step 10.5: Inspect first few `[C2_DIAG]` lines for shape**

Run: `docker logs nvllm 2>&1 | grep '\[C2_DIAG\]' | head -5`
Expected: lines matching the format `[C2_DIAG] step=N L=M nat=K  hidden  L∞=... rel_med=...  residual  L∞=... rel_med=...  OK|DIVERGED`.

- [ ] **Step 10.6: Confirm only full-attn layers fire (linear-attn skipped)**

Run: `docker logs nvllm 2>&1 | grep '\[C2_DIAG\]' | grep -oP 'L=\K\d+' | sort -u`
Expected: 16 distinct layer indices, all matching the full-attn layer positions in Qwen3.5-27B's stride-4 pattern.

- [ ] **Step 10.7: Check verdict distribution**

Run: `docker logs nvllm 2>&1 | grep '\[C2_DIAG\]' | grep -oP '(OK|DIVERGED)$' | sort | uniq -c`
Expected output documents the diagnostic result. This is the load-bearing data point.

- [ ] **Step 10.8: Commit findings to results doc**

Create `docs/research/uber_kernel_migration/2026-04-27-c2-diagnostic-results.md` (or update if it exists) with the verdict distribution and a brief interpretation note.

```bash
git add docs/research/uber_kernel_migration/2026-04-27-c2-diagnostic-results.md
git commit -m "diag(c2): record active-check probe results

Verdict distribution from a 256-token completion under PIECEWISE+graphs
+ dual-fire + CUTE_C2_DIAG=1.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
"
```

---

### Task 11: Self-test — forced-divergence injection

- [ ] **Step 11.1: Restart container with `CUTE_C2_DIAG=1` AND `CUTE_C2_DIAG_INJECT_NOISE=1.0`**

Stop and restart with both env vars set.

- [ ] **Step 11.2: Send any short prompt**

Run a single completion (any prompt). The diag should halt the engine on the first full-attn layer.

- [ ] **Step 11.3: Verify the dump bundle exists**

Run: `docker exec nvllm ls /tmp/c2_diag/`
Expected: at least one `layerN_stepM.pt` file.

- [ ] **Step 11.4: Verify the engine logged the divergence and exited cleanly**

Run: `docker logs nvllm 2>&1 | grep -E 'C2_DIAG.*DIVERGED|RuntimeError.*C2_DIAG' | head -5`
Expected: a `DIVERGED` line followed by a `RuntimeError: [C2_DIAG] diverged ...` traceback. No segfault, no SIGABRT — clean exception propagation.

- [ ] **Step 11.5: Verify dump bundle structure**

Run inside container:
```
docker exec nvllm .venv/bin/python -c "
import torch, glob
b = torch.load(sorted(glob.glob('/tmp/c2_diag/*.pt'))[0])
print(sorted(b.keys()))
print('layer:', b['layer_idx'], 'step:', b['step_idx'], 'nat:', b['nat'])
print('legacy_hidden shape:', b['legacy_hidden'].shape)
"
```
Expected: keys include `layer_idx, step_idx, nat, atol, rtol, legacy_hidden, legacy_residual, beta_rmsnorm_output, beta_residual_output`. Tensor shapes match `[nat, hidden]`.

- [ ] **Step 11.6: Clean up dump dir before next run**

Run: `docker exec nvllm rm -rf /tmp/c2_diag/`

- [ ] **Step 11.7: Append self-test confirmation to results doc**

Update `docs/research/uber_kernel_migration/2026-04-27-c2-diagnostic-results.md` with a "Self-test verified" subsection. Commit.

---

### Task 12: Sanity rung 0 — paged-only + NVFP4 + PIECEWISE+graphs

- [ ] **Step 12.1: Restart container with `CUTE_ATTN_FUSION=0` and `CUTE_C2_DIAG` UNSET**

Stop, restart with `CUTE_ATTN_FUSION=0`. Confirm: `docker exec nvllm env | grep CUTE_ATTN_FUSION` shows `CUTE_ATTN_FUSION=0`.

- [ ] **Step 12.2: Send standard probe**

Run:
```
curl -s http://localhost:8000/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"default","prompt":"What is the capital of France?","max_tokens":64,"temperature":0.0}' \
  | python -c "import sys,json; print(json.load(sys.stdin)['choices'][0]['text'])"
```

- [ ] **Step 12.3: Interpret**

- If output contains "Paris" and is coherent: **rung 0 PASS** — paged-only + NVFP4 + PIECEWISE+graphs is correct. Proceed to Task 13.
- If output is gibberish: **rung 0 FAIL** — bigger problem than C2; halt the diagnostic plan, escalate to "paged-alone+NVFP4+graphs is broken on this branch."

- [ ] **Step 12.4: Record result in `2026-04-27-c2-diagnostic-results.md`**

Add a "Sanity rung 0" section with the prompt, output, and pass/fail verdict. Commit.

---

## Phase 4 — Main diagnostic + interpretation

### Task 13: Main diagnostic run

- [ ] **Step 13.1: Restart container with `CUTE_C2_DIAG=1` only (no inject noise, no fusion disable)**

Stop, restart, confirm env: `CUTE_C2_DIAG=1` set; `CUTE_C2_DIAG_INJECT_NOISE` unset; `CUTE_ATTN_FUSION` set to default `1`.

- [ ] **Step 13.2: Run two prompts (one short, one ≥256 tokens)**

```
curl -s http://localhost:8000/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"default","prompt":"What is 2+2?","max_tokens":16,"temperature":0.0}'
```

```
curl -s http://localhost:8000/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"default","prompt":"Tell me a story about a robot exploring Mars.","max_tokens":256,"temperature":0.0}'
```

- [ ] **Step 13.3: Tally verdict counts**

Run: `docker logs nvllm 2>&1 | grep '\[C2_DIAG\]' | grep -oP '(OK|DIVERGED)$' | sort | uniq -c`

- [ ] **Step 13.4: Interpret per spec Section 5.3**

**Match interpretation:** ≥1000 lines without `DIVERGED`. Both prompts produce coherent output.
- **Conclusion:** β-coop kernel is graph-replay-correct in dual-fire under PIECEWISE+graphs. The C2 design problem is in the consume-gate op pattern.
- **Next step (next session):** design Option 1(a) consume-gate redesign with confidence; the kernel doesn't need fixing.

**Divergence interpretation:** First-divergence dump bundle present at `/tmp/c2_diag/`.
- Re-run the same prompt to test reproducibility.
- Same layer/step on re-run: deterministic; β-coop kernel has a graph-replay bug.
- Different layer/step: non-deterministic; possibly cooperative-launch + atomic-counter-spin issue per `feedback_cooperative_grid_barrier`.
- **Next step (next session):** investigate β-coop kernel directly; may need to enable Phase 5 stashed harness to disambiguate further.

- [ ] **Step 13.5: Save dump file (if any) outside `/tmp`**

If divergence occurred, copy the dump out of `/tmp` to a committed location:

```bash
mkdir -p docs/research/uber_kernel_migration/c2_diag_dumps_2026-04-27/
docker cp nvllm:/tmp/c2_diag/. docs/research/uber_kernel_migration/c2_diag_dumps_2026-04-27/
git add docs/research/uber_kernel_migration/c2_diag_dumps_2026-04-27/
```

(`.pt` files are binary but typically <10 MB; commit alongside the results doc.)

- [ ] **Step 13.6: Write the results doc completely**

Update `docs/research/uber_kernel_migration/2026-04-27-c2-diagnostic-results.md` with:
- Sanity rung 0 result (Task 12)
- Self-test confirmation (Task 11)
- Main diagnostic verdict distribution (this task)
- Interpretation
- Next-design direction (which architectural answer the probe pointed to)
- Whether Phase 5 stashed harness is needed

- [ ] **Step 13.7: Commit results**

```bash
git add docs/research/uber_kernel_migration/2026-04-27-c2-diagnostic-results.md
git commit -m "diag(c2): main diagnostic results — <one-sentence verdict>

Detailed interpretation in the results doc. Next-session direction:
<consume-gate redesign | β-coop kernel investigation | stashed harness>.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
"
```

---

## Phase 5 — Stashed eager-replay harness (deferred — only if Phase 4 inconclusive)

**Skip this entire phase if Task 13 produced a clean match or clean divergence.** Only execute if the result is sub-threshold non-deterministic divergence or other ambiguous outcome.

### Task 14: `_c2_eager_replay.py` — `EagerReplayHook` skeleton + tests

**Files:**
- Create: `vllm/v1/attention/backends/cute_paged/_c2_eager_replay.py`
- Test: `tests/v1/cute_paged/test_c2_eager_replay.py`

- [ ] **Step 14.1: Write the failing tests**

Create `tests/v1/cute_paged/test_c2_eager_replay.py`:

```python
"""Unit tests for the stashed eager-replay harness.

These tests exercise the snapshot/replay plumbing without launching the
real β-coop kernel — they use a stub kernel object that records the
inputs it was called with.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from vllm.v1.attention.backends.cute_paged import _c2_eager_replay


@dataclass
class _StubKernel:
    last_inputs: dict | None = None

    def run_beta_coop_full(self, **kwargs) -> None:
        self.last_inputs = kwargs


@dataclass
class _StubImpl:
    kernel: _StubKernel
    rmsnorm_output: torch.Tensor
    residual_output: torch.Tensor


def test_eager_replay_hook_snapshots_inputs() -> None:
    impl = _StubImpl(
        kernel=_StubKernel(),
        rmsnorm_output=torch.zeros(2, 4, dtype=torch.bfloat16),
        residual_output=torch.zeros(2, 4, dtype=torch.bfloat16),
    )
    hook = _c2_eager_replay.EagerReplayHook()
    inputs = {
        "hidden_in": torch.randn(2, 4, dtype=torch.bfloat16),
        "residual_in": torch.randn(2, 4, dtype=torch.bfloat16),
    }
    hook.snapshot(impl, inputs)
    assert hook.captured_inputs is not None
    assert torch.equal(hook.captured_inputs["hidden_in"], inputs["hidden_in"])
```

- [ ] **Step 14.2: Run tests, expect FAIL**

Run: `.venv/bin/python -m pytest tests/v1/cute_paged/test_c2_eager_replay.py -v`
Expected: import error.

- [ ] **Step 14.3: Implement `EagerReplayHook`**

Create `vllm/v1/attention/backends/cute_paged/_c2_eager_replay.py`:

```python
"""C2 diagnostic — stashed eager-replay harness.

Companion to _c2_diag.py. Off by default; enabled by CUTE_C2_DIAG_EAGER=1.
Used only if the primary (b) probe is inconclusive (sub-threshold or
non-deterministic divergence).

Re-runs β-coop in eager mode on the same inputs the graph-captured
β-coop saw, and compares outputs. Tests "is graph capture distorting
β-coop?" rather than "is β-coop correct vs legacy?".
"""

from __future__ import annotations

from typing import Any

import torch

from vllm.v1.attention.backends.cute_paged import _c2_diag


class EagerReplayHook:
    """Captures β-coop's inputs at the consume-gate site for eager replay."""

    def __init__(self) -> None:
        self.captured_inputs: dict[str, Any] | None = None

    def snapshot(self, impl: Any, inputs: dict[str, Any]) -> None:
        """Mirror inputs into pinned host tensors so they survive subsequent
        graph activity. Tensors are .clone()'d so the originals can be
        mutated by the captured graph without affecting the snapshot.
        """
        self.captured_inputs = {
            k: (v.detach().clone() if isinstance(v, torch.Tensor) else v)
            for k, v in inputs.items()
        }

    def replay_and_compare(self, impl: Any) -> dict:
        """Re-run β-coop in eager on captured inputs; compare against
        impl.rmsnorm_output / residual_output (the graph-replayed outputs).
        Returns a dict with comparison stats. Raises if no snapshot taken.
        """
        if self.captured_inputs is None:
            raise RuntimeError(
                "[C2_DIAG_EAGER] replay_and_compare called without snapshot"
            )
        # Allocate eager output buffers matching impl's buffers' shapes.
        eager_rmsnorm_output = torch.empty_like(impl.rmsnorm_output)
        eager_residual_output = torch.empty_like(impl.residual_output)
        impl.kernel.run_beta_coop_full(
            **self.captured_inputs,
            # Caller wires the rmsnorm_output / residual_output kwargs as
            # eager_rmsnorm_output / eager_residual_output here. The exact
            # kwarg names depend on run_beta_coop_full's signature at
            # phase_e_kernel.py:2685; verify at probe-wiring time.
        )
        return {
            "rmsnorm": _c2_diag._compare_pair(
                impl.rmsnorm_output,
                eager_rmsnorm_output,
                atol=1e-2, rtol=1e-2,
            ),
            "residual": _c2_diag._compare_pair(
                impl.residual_output,
                eager_residual_output,
                atol=1e-2, rtol=1e-2,
            ),
        }
```

Note: the `replay_and_compare` body has a known integration gap — `run_beta_coop_full`'s exact kwarg names (rmsnorm_output? out_residual?) must be verified at probe-wiring time against `vllm/v1/attention/backends/cute_paged/phase_e_kernel.py:2685`. The unit test only exercises `snapshot()`; the replay path is integration-only.

- [ ] **Step 14.4: Run tests, expect PASS**

Run: `.venv/bin/python -m pytest tests/v1/cute_paged/test_c2_eager_replay.py -v`
Expected: 1 passed.

- [ ] **Step 14.5: Commit**

```bash
git add vllm/v1/attention/backends/cute_paged/_c2_eager_replay.py \
        tests/v1/cute_paged/test_c2_eager_replay.py
git commit -m "diag(c2): add stashed EagerReplayHook (Phase 5 — deferred)

Companion to _c2_diag. Off by default; enabled by CUTE_C2_DIAG_EAGER=1.
Tests whether graph-replay distorts β-coop on the same inputs. Used
only if primary (b) probe is inconclusive.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
"
```

---

### Task 15: Wire `CUTE_C2_DIAG_EAGER` path into `qwen3_5.py`

**Files:**
- Modify: `vllm/nvllm/models/qwen3_5.py` (additive — extends the diag block from Task 8)

- [ ] **Step 15.1: Find the existing diag block (committed in Task 8)**

Run: `grep -n 'CUTE_C2_DIAG' vllm/nvllm/models/qwen3_5.py`
Expected: lines from Task 8.3.

- [ ] **Step 15.2: Add the EAGER companion call**

Inside the existing `if os.getenv("CUTE_C2_DIAG") == "1" and ...:` block from Task 8.3, after the `_c2_diag.compare_and_log(...)` call, append:

```python
            if os.getenv("CUTE_C2_DIAG_EAGER") == "1":
                from vllm.v1.attention.backends.cute_paged import _c2_eager_replay
                # Reuse a per-impl hook instance so snapshots persist
                # across the consume gate.
                hook = getattr(impl, "_c2_eager_hook", None)
                if hook is None:
                    hook = _c2_eager_replay.EagerReplayHook()
                    impl._c2_eager_hook = hook
                # Snapshot already happened upstream of self.self_attn —
                # see Step 15.3 for the snapshot insertion point.
                stats = hook.replay_and_compare(impl)
                if not (stats["rmsnorm"]["ok"] and stats["residual"]["ok"]):
                    raise RuntimeError(
                        f"[C2_DIAG_EAGER] graph-vs-eager diverged: "
                        f"layer={self.layer_idx} step={step_idx}  "
                        f"rmsnorm L∞={stats['rmsnorm']['linf']:.2e}  "
                        f"residual L∞={stats['residual']['linf']:.2e}"
                    )
```

- [ ] **Step 15.3: Add the snapshot call BEFORE `self.self_attn(...)`**

The `EagerReplayHook.snapshot()` must be called BEFORE β-coop fires (so we capture inputs, not outputs). Insert ahead of the existing `self.self_attn(...)` call inside the full-attention branch:

```python
        elif self.layer_type == "full_attention":
            if os.getenv("CUTE_C2_DIAG") == "1" \
               and os.getenv("CUTE_C2_DIAG_EAGER") == "1" \
               and impl is not None \
               and getattr(impl, "_fusion_bound", False):
                from vllm.v1.attention.backends.cute_paged import _c2_eager_replay
                hook = getattr(impl, "_c2_eager_hook", None)
                if hook is None:
                    hook = _c2_eager_replay.EagerReplayHook()
                    impl._c2_eager_hook = hook
                # Capture β-coop's inputs. Exact kwarg list TBD —
                # verify against run_beta_coop_full signature at
                # phase_e_kernel.py:2685 at probe-wiring time.
                hook.snapshot(impl, {
                    "hidden_in": hidden_states.clone(),
                    "residual_in": residual.clone(),
                    # ... rest of the input set per phase_e_kernel.py:2685
                })
            self.self_attn(
                hidden_states=hidden_states,
                output=self_attention_output,
                positions=positions,
            )
```

The full input list is non-trivial (see `run_beta_coop_full(...)` at `phase_e_kernel.py:2685` — ~20 named kwargs). Implementor must read that signature and mirror it exactly. Do not approximate.

- [ ] **Step 15.4: Smoke-import**

Run: `.venv/bin/python -c "import vllm.nvllm.models.qwen3_5"`
Expected: no error.

- [ ] **Step 15.5: Commit**

```bash
git add vllm/nvllm/models/qwen3_5.py
git commit -m "diag(c2): wire CUTE_C2_DIAG_EAGER stashed harness into qwen3_5

Off by default; only fires when both CUTE_C2_DIAG=1 and
CUTE_C2_DIAG_EAGER=1 are set. Snapshot happens before self.self_attn;
replay_and_compare happens after the comparison block.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
"
```

---

### Task 16: Stashed-harness verification run

- [ ] **Step 16.1: Restart with `CUTE_C2_DIAG=1` AND `CUTE_C2_DIAG_EAGER=1`**

Stop, restart with both env vars.

- [ ] **Step 16.2: Send a short prompt**

If the harness runs without `cudaErrorStreamCaptureInvalidated`: snapshot+replay works under graphs. Proceed to interpret the rmsnorm/residual graph-vs-eager comparison stats.

If `cudaErrorStreamCaptureInvalidated` fires: the snapshot's `.clone()` calls are interfering with graph capture. Either:
- Move the snapshot outside the captured segment (it already is, in qwen3_5.py's eager Python — but verify with the actual capture boundary).
- Defer this whole branch — the stashed harness can't run under graphs without further design work.

- [ ] **Step 16.3: Record results**

Update `2026-04-27-c2-diagnostic-results.md` with the stashed-harness verdict. Commit.

---

## Self-Review

**Spec coverage:**

| Spec section | Implemented by |
|---|---|
| Section 1 — Architecture / Scope | Tasks 1-8 (core probe wiring) + Task 12 (rung 0) + Task 13 (main diagnostic) |
| Section 2.1 — qwen3_5.py call site | Task 8 |
| Section 2.2 — `_c2_diag.py` | Tasks 1-6 |
| Section 2.3 — `_c2_eager_replay.py` (stashed) | Tasks 14-15 |
| Section 2.4 — Env variables | Tasks 4 (autotune assert), 5 (inject noise), 6 (compare_and_log reads tols + dump dir), 7 (CUTE_ATTN_FUSION verified) |
| Section 3 — Data flow | Task 8 (call site) + Tasks 6, 2 (compare + dump) |
| Section 4.1 — Failure modes | Tasks 6 (RuntimeError on divergence), 6 (nat=0 skip) |
| Section 4.2 — Probe explicitly does NOT do | Tasks 1-6 by construction (no try/except, no silent fallback, no retry, no get_forward_context) |
| Section 4.3 — Non-perturbation | Task 6 (no .item() in compare_and_log; only inside f-string formatting after compute) |
| Section 4.4 — Failure-isolation | Task 6 (RuntimeError messages prefixed `[C2_DIAG]`) |
| Section 4.6 — Host-safety | Task 4 (autotune assert) + Task 6 (single dump on first divergence) |
| Section 5.1 — Pre-flight | Tasks 9 (no-op), 10 (active) |
| Section 5.2 — Forced-divergence injection | Task 11 |
| Section 5.3 — Result interpretation | Task 13 |
| Section 5.4 — Stashed harness verification | Task 16 |
| Section 5.5 — What we don't test | Tasks throughout — no CI, no TP, no FULL graphs |

No gaps.

**Placeholder scan:** searched for "TBD", "TODO", "fill in", "similar to", "appropriate".

- Task 8.5 contains a conditional ("If `self.layer_idx` is not present, replace ..."). This is acceptable because the spec couldn't pre-decide the attribute name without reading qwen3_5.py at write time, and the alternative is named explicitly with the format. Not a true placeholder.
- Task 14.3's `replay_and_compare` body says "verify at probe-wiring time" for `run_beta_coop_full`'s kwarg names. Recorded as a known integration gap with the explicit reference (`phase_e_kernel.py:2685`). The unit test only exercises `snapshot()`, which is fully specified.
- Task 15.3 says "rest of the input set per phase_e_kernel.py:2685". Same gap — verifying against the live signature at integration time, with the explicit reference.

These two integration-time-resolved gaps are inherent to wiring against an existing kernel API; they're not placeholder failures. Both have explicit file:line references the implementor can resolve in 30 seconds.

**Type consistency:**

- `_compare_pair` returns `dict` with keys `linf`, `rel_med`, `ok`. Used consistently in Tasks 6, 14, 15.
- `_dump_on_divergence` returns `Path`. Used in Task 6 (assigned to `dump_path` then formatted in the RuntimeError message).
- `compare_and_log` returns `None` (raises on divergence). Used as a void call in Tasks 8 + 15.
- `EagerReplayHook.replay_and_compare` returns `dict` with keys `rmsnorm`, `residual`. Used in Task 15 (`stats["rmsnorm"]["ok"]`).

All type signatures are consistent across tasks.

**No issues found.**

---

## Execution

Plan complete and saved to `docs/research/uber_kernel_migration/2026-04-26-c2-diagnostic-plan.md`. Two execution options:

**1. Subagent-Driven (recommended)** — A fresh subagent per task, reviewed between tasks. Fast iteration. Each task's TDD cycle (test → fail → impl → pass → commit) is one subagent dispatch.

**2. Inline Execution** — All tasks in this session via `superpowers:executing-plans` with checkpoint reviews.

The plan is sized for either; subagent-driven keeps the main context lean and parallels the existing memory pattern (`feedback_subagent_kernel_fixes`, `feedback_delegate_small_tasks`).
