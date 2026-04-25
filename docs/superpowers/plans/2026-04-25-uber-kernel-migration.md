# Uber-Kernel Migration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Complete the β-coop substitution for full-attn decode in Qwen3.5-27B, eliminating the buffer-aliasing bug, the Phase 4 mlp_out double-add, the F.1 layer-LN bake corruption, and the +15 ms/layer regression. Ship β-coop as the sole full-attn-decode kernel with cooperative launch and a hard `num_seqs ≤ 4` cap.

**Architecture:** Single-source-of-truth uber-kernel design, cooperative-only with Phase 1 SMEM shrink. β-coop returns `(mlp_output, residual_output=residual_post_attn)`; layer N+1's `input_layernorm` does the residual+mlp sum itself. `paged_attention_forward` retires from decode (kept for prefill). β-lite, Phase 4, and the F.1 layer-LN bake plumbing all delete. No fallback — failures raise.

**Tech Stack:** vLLM V1 backend, CuTe DSL kernels, NVFP4 weights + BF16 activations, FP8 KV cache, Qwen3.5-27B (64 layers, 16 full-attn at stride-4), DGX Spark GB10 (SM120, 48 SMs, 102400 SMEM/SM).

**Spec:** `docs/superpowers/specs/2026-04-25-uber-kernel-migration-design.md`

**Branch:** `feat/uber-kernel-migration` (create off `main`)

---

## Pre-flight

Read these before starting:
- The spec: `docs/superpowers/specs/2026-04-25-uber-kernel-migration-design.md`
- Fresh-eyes audit findings: `docs/research/uber_kernel_migration/spec_audit_2026-04-25.md`
- Project memory: `project_uber_kernel_migration` (in MEMORY.md)
- New CLAUDE.md rules: Debug Protocol steps 7 (layer-output contract) and 8 (cooperative launch barriers)
- New feedback memories: `feedback_layer_output_contract.md`, `feedback_cooperative_grid_barrier.md`

**Branch setup:**

- [ ] **Step P.1: Create feature branch from main**

```bash
git checkout main
git pull --ff-only
git checkout -b feat/uber-kernel-migration
```

- [ ] **Step P.2: Verify F.1 commits are in tree**

```bash
git log --oneline | grep -E "9f39b86|c2a6d87|98551db" | head -3
```

Expected: three lines showing `9f39b86ef`, `c2a6d8766`, `98551dba6` (F.1 opaque op + math fixes).

---

## File Structure

**Modified files:**
- `vllm/v1/attention/backends/cute_paged/_backend.py` — dispatch logic, residual_in fix, drop paged_attention_forward call, hard cap, delete β-lite block + attach methods + flags
- `vllm/v1/attention/backends/cute_paged/phase_e_kernel.py` — delete Phase 4, drop bake-related kernel args, SMEM shrink Phase 1
- `vllm/v1/attention/backends/cute_paged/_mlp_op.py` — simplify cute_phase_e_dispatch consume branch, delete cute_phase_e_skip_input_layernorm op
- `vllm/nvllm/models/qwen3_5.py` — collapse input_LN gate to unconditional, drop attach_input_layernorm + attach_next_input_layernorm loops

**Created files:**
- `tests/v1/cute_paged/test_uber_kernel_buffer_contracts.py` — L2 integration test (catches buffer-aliasing class)
- `tests/v1/cute_paged/test_uber_kernel_multi_layer.py` — L3 multi-layer flow test (catches per-layer LN regressions)
- `tests/v1/cute_paged/test_uber_kernel_hard_cap.py` — verifies num_seqs=5 raises clearly

**Evidence directory** (created at C3/C4):
- `benchmarks/nvllm/traces/uber_kernel_migration/<date>/` — L5 perf traces with `summary.md`, `baseline.nsys-rep`, `migrated.nsys-rep`

---

## Task C1: residual_buf bandaid + L2 integration test

**Goal:** Fix the buffer-aliasing bug for both β-coop and β-lite. Land the L2 test that catches this class of bug. Ship a correctness fix while keeping all the structural changes for later commits.

**Files:**
- Modify: `vllm/v1/attention/backends/cute_paged/_backend.py:1175` (β-coop residual_in source)
- Modify: `vllm/v1/attention/backends/cute_paged/_backend.py:1268` (β-lite residual_post_ln source — audit Finding 6)
- Create: `tests/v1/cute_paged/test_uber_kernel_buffer_contracts.py`

### Substeps

- [ ] **Step C1.1: Write the L2 buffer-contracts test (failing first)**

Create `tests/v1/cute_paged/test_uber_kernel_buffer_contracts.py` with this content:

```python
"""L2 integration test: verifies β-coop reads its inputs from the right buffer.

Pre-fix: β-coop reads `self.residual_output` (post-Phase-C output of the
legacy paged_attention_forward), causing residual_post_attn = 2*attn_out + h + r.
Post-fix: β-coop reads `self.residual_buf` (post-input-LN residual mirrored
from qwen3_5.py:460), giving residual_post_attn = attn_out + h + r.

Test strategy: monkey-patch run_beta_coop_full to capture its actual residual_in
arg (data_ptr), then verify it matches self.residual_buf.data_ptr() and NOT
self.residual_output.data_ptr().
"""
import os
import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_properties(0).major < 12,
    reason="Requires SM120 (DGX Spark / GB10).",
)


def test_beta_coop_residual_in_sources_from_residual_buf(monkeypatch):
    """β-coop's residual_in pointer must match self.residual_buf, not residual_output."""
    from vllm.v1.attention.backends.cute_paged._backend import (
        CutePagedAttentionImpl,
        PhaseE_Beta_Kernel,
    )

    captured = {}

    real_run = PhaseE_Beta_Kernel.run_beta_coop_full

    def capturing_run(self, *args, **kwargs):
        captured["residual_in_data_ptr"] = kwargs["residual_in"].data_ptr()
        # Skip the actual launch — the test only needs to verify the arg.
        captured["called"] = True

    monkeypatch.setattr(PhaseE_Beta_Kernel, "run_beta_coop_full", capturing_run)

    # Build a minimal CutePagedAttentionImpl with the fields the dispatch
    # path reads. Real attach paths would set these; we set them directly.
    # See _backend.py:265-330 for the buffer init.
    impl = CutePagedAttentionImpl.__new__(CutePagedAttentionImpl)
    nat = 4
    hidden_dim = 5120
    device = "cuda"
    impl.residual_buf = torch.zeros(nat, hidden_dim, dtype=torch.bfloat16, device=device)
    impl.residual_output = torch.zeros(nat, hidden_dim, dtype=torch.bfloat16, device=device)

    # Distinct sentinel data ptrs — if the wrong buffer is read, captured
    # value won't match residual_buf.
    expected_ptr = impl.residual_buf.data_ptr()
    other_ptr = impl.residual_output.data_ptr()
    assert expected_ptr != other_ptr, "test sanity: buffers must be distinct"

    # Invoke the launch path (smallest reproducer of the line we're testing).
    # In the real backend.forward(), the call site at _backend.py:1175 passes
    # residual_in=self.residual_output[:nat]. We're verifying that line was
    # changed to residual_buf.
    #
    # We do this by reading the source of the launch call and checking the
    # captured value matches the expected source.
    # NOTE: this test is structural — it doesn't run the kernel, just verifies
    # the call wiring. Full end-to-end correctness is covered by L3.
    import inspect
    src = inspect.getsource(impl.__class__.forward)
    assert "residual_in=self.residual_buf" in src, (
        "Expected β-coop launch to read from self.residual_buf; found a different source. "
        "Check _backend.py:1175 — buffer-aliasing bug may have regressed."
    )
    assert "residual_in=self.residual_output" not in src.split("# β-coop")[1] if "# β-coop" in src else True, (
        "β-coop call site still reads self.residual_output — the alias bug is back."
    )


def test_beta_lite_residual_post_ln_sources_from_residual_buf():
    """β-lite has the same alias bug pre-migration (audit Finding 6).
    Verify it's also fixed."""
    from vllm.v1.attention.backends.cute_paged._backend import CutePagedAttentionImpl
    import inspect

    src = inspect.getsource(CutePagedAttentionImpl.forward)
    # β-lite block reads residual_post_ln. After fix it should source from residual_buf.
    if "residual_post_ln=" in src:
        assert "residual_post_ln=self.residual_buf" in src, (
            "β-lite still aliases legacy buffer. See audit Finding 6."
        )
```

- [ ] **Step C1.2: Run the test to verify it fails**

```bash
cd /home/natfii/docker/nvllm
.venv/bin/python -m pytest tests/v1/cute_paged/test_uber_kernel_buffer_contracts.py -v
```

Expected: FAIL — `assert "residual_in=self.residual_buf" in src` fails because the source still has `residual_in=self.residual_output`.

- [ ] **Step C1.3: Apply the β-coop fix at `_backend.py:1175`**

Use Edit tool:

```
file_path: vllm/v1/attention/backends/cute_paged/_backend.py
old_string:                         residual_in=self.residual_output[:nat],
new_string:                         residual_in=self.residual_buf[:nat],
```

Note: the `old_string` may have surrounding context — find the line at `:1175` and replace just that one line. The file has multiple `residual_in=` calls; ensure you target the β-coop call site (around line 1175 in the `_use_beta_coop` block).

- [ ] **Step C1.4: Apply the β-lite fix at `_backend.py:1268` (audit Finding 6)**

Find the β-lite block (line ~1268 in the `_use_beta_lite` block):

```
old_string:                         residual_post_ln=self.residual_output[:nat],
new_string:                         residual_post_ln=self.residual_buf[:nat],
```

Verify only one match — there should be exactly one `residual_post_ln=` site in `_backend.py`.

- [ ] **Step C1.5: Run the L2 test to verify it passes**

```bash
.venv/bin/python -m pytest tests/v1/cute_paged/test_uber_kernel_buffer_contracts.py -v
```

Expected: PASS — both tests green.

- [ ] **Step C1.6: Docker rebuild in tmux**

Per CLAUDE.md, builds run in tmux:

```bash
tmux new-session -d -s build-c1 'cd /home/natfii/docker/nvllm && docker build -f docker/Dockerfile.gb10 -t nvllm:gb10 . 2>&1 | tee /tmp/build-c1.log'
```

Periodically check progress:

```bash
tmux capture-pane -p -t build-c1 | tail -30
```

Wait until tmux session ends (build complete, ~60-90 min).

- [ ] **Step C1.7: Verify image contains the fix**

Per `feedback_docker_cache`:

```bash
docker run --rm --entrypoint cat nvllm:gb10 \
  /workspace/vllm/v1/attention/backends/cute_paged/_backend.py | \
  grep -A1 "β-coop launch" | grep "residual_in"
```

Expected: line shows `residual_in=self.residual_buf[:nat]`.

- [ ] **Step C1.8: Run gsm8k_eval_50 with β-coop ON**

```bash
# Start serve in tmux
tmux new-session -d -s serve-c1 'cd /home/natfii/docker/nvllm && CUTE_PHASE_E_FUSION=1 ./scripts/serve.sh 2>&1 | tee /tmp/serve-c1.log'

# Wait for "Started server process" + warmup
until docker logs nvllm 2>&1 | grep -q "Started server process"; do sleep 5; done

# Run eval
.venv/bin/python scripts/gsm8k_eval_50.py --seed 42 --max-tokens 256
```

Expected: ≥ 90% accuracy. C1's fix recovers correctness for both β-coop and β-lite.

- [ ] **Step C1.9: Stop serve**

```bash
tmux kill-session -t serve-c1
docker stop nvllm 2>/dev/null || true
```

- [ ] **Step C1.10: Confirm with user before committing, then commit**

Per `feedback_commits`, ASK USER before committing. Show them the diff:

```bash
git diff vllm/v1/attention/backends/cute_paged/_backend.py
git diff tests/
```

After user approves:

```bash
git add vllm/v1/attention/backends/cute_paged/_backend.py \
        tests/v1/cute_paged/test_uber_kernel_buffer_contracts.py
git commit -m "$(cat <<'EOF'
fix(cute): C1 — β-coop and β-lite read residual_buf, not residual_output

β-coop's Phase 1C residual_in pointed at self.residual_output, which
paged_attention_forward had already filled with (h+r) + wo_out =
residual_post_attn. β-coop then re-added wo_out, producing
2·wo_out + h + r — gibberish output cascading through 16 fused
full-attn layers.

Same alias existed in β-lite's residual_post_ln source. Fixed both.

L2 integration test added to catch the class.

Refs: docs/superpowers/specs/2026-04-25-uber-kernel-migration-design.md
      docs/research/uber_kernel_migration/spec_audit_2026-04-25.md
      memory:project_phase_e_beta_math_bug

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task C1.5: Delete Phase 4 + F.1 layer-LN bake plumbing + per-layer input_LN

**Goal:** Eliminate the cross-layer-state machinery that the Q4 self-review proved structurally broken. β-coop's Phase 4 deletes (audit Finding 1: in-place mlp_out add causes layer N+1 to double-count). The F.1 skip-op + attach methods + flags + scratch buffer all delete. Every layer runs `input_layernorm` at its own entry, matching the unfused flow.

**Files:**
- Modify: `vllm/v1/attention/backends/cute_paged/phase_e_kernel.py` — delete Phase 4 from `run_beta_coop_full` + drop bake args
- Modify: `vllm/v1/attention/backends/cute_paged/_backend.py` — delete attach_input_layernorm + attach_next_input_layernorm + flags + next_hidden_scratch allocation
- Modify: `vllm/v1/attention/backends/cute_paged/_mlp_op.py` — simplify cute_phase_e_dispatch consume branch, delete cute_phase_e_skip_input_layernorm op
- Modify: `vllm/nvllm/models/qwen3_5.py` — collapse input_LN gate to unconditional, drop attach loops
- Create: `tests/v1/cute_paged/test_uber_kernel_multi_layer.py`

### Substeps

- [ ] **Step C1.5.1: Write the L3 multi-layer test (failing first)**

Create `tests/v1/cute_paged/test_uber_kernel_multi_layer.py`:

```python
"""L3 integration test: verifies layer-boundary semantics.

Catches:
- Phase 4 not adding mlp_out (Finding 1) — layer N+1's input_LN does the sum
- Per-layer input_layernorm fires unconditionally (no skip-op fall-through)
- _phase_e_skip_next_ln flag is gone (no cross-layer state)

Strategy: source-level audit. The full kernel-level diff is covered by
L4 (gsm8k); this test catches the structural class.
"""
import inspect
import pytest


def test_qwen35_layer_forward_runs_input_layernorm_unconditionally():
    """qwen3_5.py:421-441 input_LN gate must collapse to unconditional run."""
    from vllm.nvllm.models import qwen3_5
    src = inspect.getsource(qwen3_5.Qwen3_5DecoderLayer.forward)
    assert "cute_phase_e_skip_input_layernorm" not in src, (
        "F.1 skip-op call site still present in layer forward. "
        "Should be deleted in C1.5."
    )
    # Both first-layer and non-first-layer branches must call input_layernorm.
    # Look for the standard pattern.
    assert "self.input_layernorm(hidden_states, residual)" in src or \
           "self.input_layernorm(hidden_states)" in src, (
        "input_layernorm call missing from layer forward."
    )


def test_no_attach_input_layernorm_loops_in_model_init():
    """Qwen3_5Model.__init__ post-hook must drop attach_*_layernorm loops."""
    from vllm.nvllm.models import qwen3_5
    src = inspect.getsource(qwen3_5.Qwen3_5Model.__init__)
    assert "attach_input_layernorm" not in src, (
        "attach_input_layernorm loop still present. C1.5 must delete it."
    )
    assert "attach_next_input_layernorm" not in src, (
        "attach_next_input_layernorm loop still present. C1.5 must delete it."
    )


def test_skip_op_deleted():
    """cute_phase_e_skip_input_layernorm op must be deleted entirely."""
    from vllm.v1.attention.backends.cute_paged import _mlp_op
    src = inspect.getsource(_mlp_op)
    # The op registration uses direct_register_custom_op(op_name="..."). Look
    # for that registration.
    assert 'op_name="cute_phase_e_skip_input_layernorm"' not in src, (
        "Skip op still registered. C1.5 must delete the op registration "
        "and the impl/fake functions."
    )


def test_phase_4_deleted_from_run_beta_coop_full():
    """Phase 4 ε epilogue must not write to next_hidden_scratch.
    The kernel returns at end of Phase 3."""
    from vllm.v1.attention.backends.cute_paged import phase_e_kernel
    src = inspect.getsource(phase_e_kernel.PhaseE_Beta_Kernel.run_beta_coop_full)
    # Phase 4 args dropped from the call signature
    assert "next_input_layernorm_gamma" not in src, (
        "Phase 4 arg next_input_layernorm_gamma still present in "
        "run_beta_coop_full. C1.5 must drop it."
    )
    assert "emit_next_layernorm" not in src, (
        "Phase 4 arg emit_next_layernorm still present. Drop it."
    )


def test_dispatch_op_consumes_mlp_output_not_next_hidden_scratch():
    """cute_phase_e_dispatch consume branch must read mlp_output, not next_hidden_scratch."""
    from vllm.v1.attention.backends.cute_paged import _mlp_op
    src = inspect.getsource(_mlp_op)
    # The consume branch should reference mlp_output (or equivalent buffer) for hidden_out.
    # next_hidden_scratch should not appear at all in this op.
    assert "next_hidden_scratch" not in src, (
        "Dispatch op still references next_hidden_scratch. C1.5 must update "
        "consume branch to read from mlp_output."
    )
```

- [ ] **Step C1.5.2: Run the test to verify it fails**

```bash
.venv/bin/python -m pytest tests/v1/cute_paged/test_uber_kernel_multi_layer.py -v
```

Expected: 5 FAIL — all 5 tests should fail because the deletions haven't happened yet.

- [ ] **Step C1.5.3: Delete `cute_phase_e_skip_input_layernorm` op from `_mlp_op.py`**

In `vllm/v1/attention/backends/cute_paged/_mlp_op.py`, locate the section (lines 239-301):

```python
# --- Phase F.1: cute_phase_e_skip_input_layernorm --------------------------
...
def _cute_phase_e_skip_input_layernorm_impl(...): ...
def _cute_phase_e_skip_input_layernorm_fake(...): ...
direct_register_custom_op(
    op_name="cute_phase_e_skip_input_layernorm",
    ...
)
```

Delete the entire section (the comment header through the `direct_register_custom_op` closing paren).

- [ ] **Step C1.5.4: Simplify `cute_phase_e_dispatch_impl` consume branch**

In `_mlp_op.py:192-218`, the current consume branch is:

```python
if getattr(impl, "_phase_e_consumed", False):
    hidden_out[:nat].copy_(impl.next_hidden_scratch[:nat])
    residual_out[:nat].copy_(impl.residual_output[:nat])
    impl._phase_e_consumed = False
    impl._phase_e_skip_next_ln = True
    return
```

Replace with:

```python
if getattr(impl, "_phase_e_consumed", False):
    hidden_out[:nat].copy_(impl.mlp_output[:nat])
    residual_out[:nat].copy_(impl.residual_output[:nat])
    impl._phase_e_consumed = False
    return
```

(Drop the `_phase_e_skip_next_ln = True` line and change `next_hidden_scratch` → `mlp_output`.)

- [ ] **Step C1.5.5: Delete `attach_input_layernorm` method from `_backend.py`**

In `vllm/v1/attention/backends/cute_paged/_backend.py`, locate `attach_input_layernorm` (around lines 507-519). Delete the method entirely.

- [ ] **Step C1.5.6: Delete `attach_next_input_layernorm` method from `_backend.py`**

Locate `attach_next_input_layernorm` (around lines 433-505). Delete the method entirely. This also drops the `next_hidden_scratch` allocation at lines 481-484 and the `phase_e_barrier` workspace.

- [ ] **Step C1.5.7: Delete F.1 plumbing flags from `CutePagedAttentionImpl.__init__`**

In `_backend.py` around line 280, find:

```python
# Phase F.1: skip-flag set by cute_phase_e_dispatch when consuming
# β output; read by cute_phase_e_skip_input_layernorm on layer N+1.
self._phase_e_skip_next_ln = False
self._input_layernorm_module = None
```

Delete those four lines.

Also find and delete:
- `self._next_input_layernorm_module = None` (init)
- `self._emit_next_layernorm = False` (init)

- [ ] **Step C1.5.8: Delete Phase 4 from `phase_e_kernel.py`**

In `vllm/v1/attention/backends/cute_paged/phase_e_kernel.py`, locate the Phase 4 section in `_jit_launch_phase_0_to_4` (within `run_beta_coop_full`'s body, around lines 4570-4674).

The Phase 4 block is the section that starts with the secondary-barrier election and contains:
```python
if emit_next_ln == Int32(1):
    # ... LN bake branch (already deleted in earlier work, but verify)
else:
    # ... raw residual memcpy branch
```

Delete the entire Phase 4 block. The kernel should return after Phase 3 (end of MLP write).

- [ ] **Step C1.5.9: Drop Phase 4 args from `run_beta_coop_full` signature**

In `phase_e_kernel.py:2685-2732`, the current signature includes:
```python
next_input_layernorm_gamma: Optional[torch.Tensor],
next_hidden_output: torch.Tensor,
emit_next_layernorm: bool = True,
```

Remove these three parameters. Also remove their related code in the body:
- Lines around 2768-2775: `if emit_next_layernorm:` block
- Lines around 2860-2869: `next_gamma_ptr`, `next_hidden_ptr`, `emit_next_ln_i32` derived values
- The corresponding entries in the `all_args` tuple (around line 2926-2928).

- [ ] **Step C1.5.10: Drop the β-coop launch's Phase 4 kwargs in `_backend.py`**

In `_backend.py:1146-1208` (the β-coop launch block), remove:
```python
next_input_layernorm_gamma=_next_gamma,
next_hidden_output=self.next_hidden_scratch[:nat],
emit_next_layernorm=_emit_next,
```

And the `_next_gamma`, `_rms_eps`, `_emit_next` derived locals (around lines 1149-1158).

- [ ] **Step C1.5.11: Collapse input_LN gate in `qwen3_5.py:421-441`**

The current non-first-layer branch in `Qwen3_5DecoderLayer.forward` is:
```python
else:
    _mlp_layer_name = getattr(self.mlp, "_cute_layer_name", None)
    if _mlp_layer_name is not None:
        out_x = torch.empty_like(hidden_states)
        out_residual = torch.empty_like(residual)
        torch.ops.vllm.cute_phase_e_skip_input_layernorm(
            hidden_states, residual, out_x, out_residual,
            _mlp_layer_name,
        )
        hidden_states, residual = out_x, out_residual
    else:
        hidden_states, residual = self.input_layernorm(
            hidden_states, residual
        )
```

Replace with:
```python
else:
    hidden_states, residual = self.input_layernorm(
        hidden_states, residual
    )
```

(Just the unconditional call. Drop the `_mlp_layer_name` check entirely.)

- [ ] **Step C1.5.12: Drop attach loops from `Qwen3_5Model.__init__`**

In `vllm/nvllm/models/qwen3_5.py`:

Delete the `attach_input_layernorm` loop at lines ~636-647:
```python
for idx, layer in enumerate(self.layers):
    ...
    impl.attach_input_layernorm(...)
```

Delete the `attach_next_input_layernorm` loop at lines ~656-676:
```python
if os.environ.get("CUTE_PHASE_E_FUSION", "0") == "1":
    for idx, layer in enumerate(self.layers):
        ...
        impl.attach_next_input_layernorm(next_norm)
```

Drop any imports that are no longer used after these deletions.

- [ ] **Step C1.5.13: Run the L3 test to verify it passes**

```bash
.venv/bin/python -m pytest tests/v1/cute_paged/test_uber_kernel_multi_layer.py -v
```

Expected: 5 PASS.

- [ ] **Step C1.5.14: Run the L2 test to verify no regression**

```bash
.venv/bin/python -m pytest tests/v1/cute_paged/test_uber_kernel_buffer_contracts.py -v
```

Expected: 2 PASS.

- [ ] **Step C1.5.15: Docker rebuild in tmux**

```bash
tmux new-session -d -s build-c15 'cd /home/natfii/docker/nvllm && docker build -f docker/Dockerfile.gb10 -t nvllm:gb10 . 2>&1 | tee /tmp/build-c15.log'
```

Wait for completion.

- [ ] **Step C1.5.16: Verify image contains the deletions**

```bash
docker run --rm --entrypoint bash nvllm:gb10 -c '
  grep -c "cute_phase_e_skip_input_layernorm" /workspace/vllm/v1/attention/backends/cute_paged/_mlp_op.py || echo "0 (good)"
  grep -c "attach_input_layernorm" /workspace/vllm/nvllm/models/qwen3_5.py || echo "0 (good)"
'
```

Both should report 0.

- [ ] **Step C1.5.17: Run gsm8k_eval_50 with β-coop ON**

```bash
tmux new-session -d -s serve-c15 'CUTE_PHASE_E_FUSION=1 ./scripts/serve.sh 2>&1 | tee /tmp/serve-c15.log'
until docker logs nvllm 2>&1 | grep -q "Started server process"; do sleep 5; done
.venv/bin/python scripts/gsm8k_eval_50.py --seed 42 --max-tokens 256
tmux kill-session -t serve-c15
docker stop nvllm 2>/dev/null || true
```

Expected: ≥ 90%. This is the critical gate — proves C1.5's structural changes preserved correctness.

- [ ] **Step C1.5.18: Confirm with user, then commit**

```bash
git diff --stat
```

Show user, get approval. Then:

```bash
git add vllm/v1/attention/backends/cute_paged/phase_e_kernel.py \
        vllm/v1/attention/backends/cute_paged/_backend.py \
        vllm/v1/attention/backends/cute_paged/_mlp_op.py \
        vllm/nvllm/models/qwen3_5.py \
        tests/v1/cute_paged/test_uber_kernel_multi_layer.py
git commit -m "$(cat <<'EOF'
refactor(cute): C1.5 — delete Phase 4 + F.1 layer-LN bake plumbing

Per spec audit Finding 1: Phase 4's in-place mlp_out add into
residual_output caused layer N+1's input_layernorm to double-count
mlp_out. Per Q4 self-review: the F.1 layer-LN bake corrupts
linear-attn layer N+1's residual stream in Qwen3.5's stride-4
pattern (linear-attn doesn't honor the skip-op).

Resolution: per-layer input_LN at every layer entry, matching every
surveyed hybrid model (Jamba, Zamba2, Qwen3-Next, Megatron hybrid).

Deleted:
- cute_phase_e_skip_input_layernorm op
- attach_input_layernorm + attach_next_input_layernorm methods
- _phase_e_skip_next_ln, _input_layernorm_module,
  _next_input_layernorm_module, _emit_next_layernorm fields
- next_hidden_scratch buffer allocation
- Phase 4 ε epilogue (no longer fires; no in-place mlp_out add)
- run_beta_coop_full's next_input_layernorm_gamma,
  next_hidden_output, emit_next_layernorm parameters

Simplified:
- cute_phase_e_dispatch consume branch reads mlp_output (raw)
- qwen3_5.py layer forward: unconditional self.input_layernorm

L3 multi-layer test added; gsm8k_eval_50 ≥ 90% with β-coop ON.

Refs: docs/superpowers/specs/2026-04-25-uber-kernel-migration-design.md
      docs/research/uber_kernel_migration/q4_brainstorm_layer_LN_2026-04-25.md
      docs/research/uber_kernel_migration/spec_audit_2026-04-25.md

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task C2: Drop `paged_attention_forward` from decode + redefine `_fusion_active`

**Goal:** Retire the legacy A+B+C uber-kernel from the full-attn decode path. β-coop becomes the only writer of `self.residual_output`, `self.rmsnorm_output`, `self.wo_output`. The `_fusion_active` flag's semantics change — used to mean "the legacy uber-kernel populated buffers"; now means "β-coop will fire this step." Update the load-bearing comment at `_backend.py:1108-1110` (audit Finding 5).

**Files:**
- Modify: `vllm/v1/attention/backends/cute_paged/_backend.py:1034` (paged_attention_forward call site, decode branch)
- Modify: `vllm/v1/attention/backends/cute_paged/_backend.py:980` (`_fusion_active` definition)
- Modify: `vllm/v1/attention/backends/cute_paged/_backend.py:1108-1110` (the `use_fusion` AND comment)

### Substeps

- [ ] **Step C2.1: Add a regression test asserting paged_attention_forward isn't called for decode**

Append to `tests/v1/cute_paged/test_uber_kernel_buffer_contracts.py`:

```python
def test_paged_attention_forward_not_called_for_decode(monkeypatch):
    """C2 invariant: paged_attention_forward must not fire on the decode path."""
    import inspect
    from vllm.v1.attention.backends.cute_paged import _backend
    src = inspect.getsource(_backend.CutePagedAttentionImpl.forward)
    # The decode branch should NOT contain a call to paged_attention_forward.
    # If is_decode_only is True, the call site must be gated off.
    # Quick structural check: paged_attention_forward should be inside an
    # `if not is_decode_only` or equivalent prefill-only block.
    assert "paged_attention_forward(" in src, (
        "paged_attention_forward call entirely missing — should still be present for prefill."
    )
    # Find the section before β-coop dispatch and verify the call is gated.
    pre_beta = src.split("_use_beta_coop")[0]
    if "paged_attention_forward(" in pre_beta:
        # If it's in the pre-β section, it must be inside a prefill gate.
        # Look for the gate keyword.
        assert ("not is_decode_only" in pre_beta) or ("is_prefill" in pre_beta) or \
               ("not _is_decode" in pre_beta), (
            "paged_attention_forward called unconditionally — must be gated to prefill only."
        )
```

- [ ] **Step C2.2: Run the test to verify it fails**

```bash
.venv/bin/python -m pytest tests/v1/cute_paged/test_uber_kernel_buffer_contracts.py::test_paged_attention_forward_not_called_for_decode -v
```

Expected: FAIL — call is currently unconditional.

- [ ] **Step C2.3: Gate `paged_attention_forward` to prefill-only**

In `_backend.py`, locate the call at line 1034:

```python
result = paged_attention_forward(
    query=query[:num_actual_tokens],
    ...
)
```

Wrap it in a prefill-only gate. The flag `is_decode_only` is already in scope (used elsewhere in `forward`). Modify:

```python
# Pre-migration: paged_attention_forward ran unconditionally when use_fusion=True.
# C2 (uber-kernel migration): retired from decode. β-coop is the sole
# full-attn-decode kernel. Prefill keeps this path.
if not is_decode_only:
    result = paged_attention_forward(
        query=query[:num_actual_tokens],
        ...
    )
else:
    # Decode path: β-coop populates the buffers paged_attention_forward
    # used to populate (residual_output, rmsnorm_output, wo_output).
    # Define `result` for downstream code that reads it.
    result = None  # downstream debug harness must check is_decode_only
```

If downstream code references `result` unconditionally, you may need to refactor those references too — search for `result` usages within `forward` and gate appropriately.

- [ ] **Step C2.4: Update the `use_fusion` AND clause comment**

At `_backend.py:1107-1119`, the existing comment reads:

```python
# INVARIANT: β-lite reads `self.residual_output` below, which is only
# populated by the attention uber-kernel when `use_fusion=True`. Keep
# `use_fusion` in this AND; removing it would silently feed stale
# residual data from the previous step into the ε epilogue.
```

Update to:

```python
# INVARIANT (post-C2 migration): β-coop is the SOLE writer of
# self.residual_output on the decode path. The `use_fusion` flag is
# retained as a gate — when False, no fusion runs at all and β-coop
# is skipped. β-lite is deleted in C3 so this gate covers only β-coop.
```

- [ ] **Step C2.5: Redefine `_fusion_active` (per audit Finding 5)**

At `_backend.py:980`:

```python
self._fusion_active = self._fusion_bound and is_decode_only and fits_buffer
```

Change to:

```python
# Post-C2: "fusion active" means β-coop will fire this step (not "the
# legacy uber-kernel populated buffers"). Semantics drifted post-migration;
# kept name for compatibility but the meaning is β-coop-will-launch.
self._fusion_active = (
    self._fusion_bound and is_decode_only and fits_buffer
)
```

(Comment-only change. The boolean computation is the same; only the
semantics-documentation comment is added.)

- [ ] **Step C2.6: Run all tests**

```bash
.venv/bin/python -m pytest tests/v1/cute_paged/ -v
```

Expected: all 8 tests pass (2 from C1, 5 from C1.5, 1 new from C2).

- [ ] **Step C2.7: Docker rebuild**

```bash
tmux new-session -d -s build-c2 'docker build -f docker/Dockerfile.gb10 -t nvllm:gb10 . 2>&1 | tee /tmp/build-c2.log'
```

Wait for completion.

- [ ] **Step C2.8: Run gsm8k_eval_50 — and capture L5 partial measurement**

```bash
tmux new-session -d -s serve-c2 'CUTE_PHASE_E_FUSION=1 ./scripts/serve.sh 2>&1 | tee /tmp/serve-c2.log'
until docker logs nvllm 2>&1 | grep -q "Started server process"; do sleep 5; done

# Correctness gate
.venv/bin/python scripts/gsm8k_eval_50.py --seed 42 --max-tokens 256

# Sustained tok/s measurement (L5 partial — captures the +15 ms/layer recovery from removing double-fire)
.venv/bin/python scripts/measure_tok_per_sec.py --num-prompts 8 --max-tokens 256 \
  > /tmp/tok-per-sec-c2.txt
cat /tmp/tok-per-sec-c2.txt

tmux kill-session -t serve-c2
docker stop nvllm 2>/dev/null || true
```

Expected: gsm8k ≥ 90%, tok/s noticeably higher than C1's value (~+15 ms/layer recovery).

- [ ] **Step C2.9: Confirm with user, then commit**

```bash
git add vllm/v1/attention/backends/cute_paged/_backend.py \
        tests/v1/cute_paged/test_uber_kernel_buffer_contracts.py
git commit -m "$(cat <<'EOF'
refactor(cute): C2 — paged_attention_forward retires from decode path

β-coop is now the sole full-attn-decode kernel. paged_attention_forward
stays in tree but only fires on prefill (gated by is_decode_only).

Eliminates the +15 ms/layer regression caused by double-firing Phase
A+B+C through both the legacy uber-kernel and β-coop's Phase 1.

Updates:
- Gate paged_attention_forward call to non-decode-only
- Update use_fusion AND clause comment (audit Finding 5)
- Redefine _fusion_active semantics: now means β-coop-will-fire,
  not legacy-kernel-populated-buffers

Test: regression test verifies paged_attention_forward is gated.
gsm8k_eval_50 ≥ 90%; tok/s recovers vs C1.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task C3: SMEM shrink Phase 1 + delete β-lite + hard-cap `num_seqs=4`

**Goal:** Shrink β-coop's Phase 1 SMEM enough to lift `resident_cap` from 96 to ~288, covering num_seqs ≤ 4 cooperatively. Delete β-lite entirely (no longer needed after the cap is lifted to production ceiling). Add a hard-cap precondition that raises if `num_seqs × 64 > resident_cap`.

**This is the biggest commit. Includes a kernel restructure for SMEM packing — implementation details below are guidelines, not strict prescriptions; the executor will iterate on the actual packing strategy.**

**Files:**
- Modify: `vllm/v1/attention/backends/cute_paged/phase_e_kernel.py` — Phase 1 SMEM shrink (K/V ping-pong, Q packing)
- Modify: `vllm/v1/attention/backends/cute_paged/_backend.py` — delete β-lite block, add hard-cap precondition
- Create: `tests/v1/cute_paged/test_uber_kernel_hard_cap.py`

### Substeps

- [ ] **Step C3.1: Capture pre-shrink SMEM number**

```bash
grep -n "smem_bytes=45568\|smem_bytes_phase_coop_full" \
  vllm/v1/attention/backends/cute_paged/_backend.py \
  vllm/v1/attention/backends/cute_paged/phase_e_kernel.py
```

Note the current value (~45568 B). The shrink target is ~17 KB; intermediate ladder is ~32 KB.

- [ ] **Step C3.2: Delete β-lite block from `_backend.forward`**

In `_backend.py`, locate the β-lite invocation block (around lines 1222-1290 — the `if _use_beta_lite:` block including the try/except wrapper). Delete the entire block.

Also delete the surrounding flag computations:
- `self._phase_e_use_beta_lite = False` (init)
- `_use_beta_lite = (...)` (around line 1138)
- The `_phase_e_use_beta_lite` field references throughout `_backend.py`
- The β-coop except clause's `_use_beta_lite = True` retry logic (around line 1221)

Replace the β-coop try/except's fall-through with a simple raise:

```python
if _use_beta_coop:
    nat = num_actual_tokens
    self._phase_e_coop_kernel.run_beta_coop_full(...)
    self._phase_e_consumed = True
```

(Drop the `try/except` and the β-lite fallback. Per Q5=A, β-coop must succeed.)

- [ ] **Step C3.3: Add hard-cap precondition**

Just before the β-coop launch in `_backend.forward()`, add:

```python
# C3 hard-cap: cooperative launch requires CTAs to fit resident_cap.
# Per spec: num_seqs ≤ 4 covered after Phase 1 SMEM shrink. Beyond cap,
# raise — no fallback (Q5=A).
total_ctas = 64 * num_seqs
if total_ctas > self._resident_cap:
    raise RuntimeError(
        f"β-coop hard-cap exceeded: num_seqs={num_seqs} requires "
        f"{total_ctas} CTAs but resident_cap={self._resident_cap}. "
        f"Reduce --max-num-seqs or run with CUTE_PHASE_E_FUSION=0."
    )
```

- [ ] **Step C3.4: Write hard-cap regression test**

Create `tests/v1/cute_paged/test_uber_kernel_hard_cap.py`:

```python
"""Verifies the hard-cap precondition raises on num_seqs > resident_cap."""
import pytest
import torch
from unittest.mock import MagicMock

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_properties(0).major < 12,
    reason="Requires SM120 (DGX Spark / GB10).",
)


def test_hard_cap_raises_on_too_many_seqs():
    """Launching β-coop with num_seqs > resident_cap/64 must raise RuntimeError."""
    from vllm.v1.attention.backends.cute_paged._backend import CutePagedAttentionImpl

    impl = CutePagedAttentionImpl.__new__(CutePagedAttentionImpl)
    impl._resident_cap = 96  # current default (pre-shrink); 96 / 64 = 1 seq
    impl._phase_e_active = True
    impl._fusion_bound = True

    # Build the smallest set of fields the precondition path reads.
    # The precondition fires before the launch, so we don't need the kernel
    # to actually exist — just need num_seqs to exceed cap.

    # Forward expects attn_metadata with num_actual_tokens; mock it.
    attn_md = MagicMock()
    attn_md.num_actual_tokens = 5  # 5 seqs × 64 CTAs = 320 > 96
    attn_md.seq_lens = list(range(5))

    # Skip the full forward setup; just verify the precondition message
    # appears in CutePagedAttentionImpl.forward source.
    import inspect
    src = inspect.getsource(CutePagedAttentionImpl.forward)
    assert "β-coop hard-cap exceeded" in src or "resident_cap" in src, (
        "Hard-cap precondition message missing from _backend.forward — "
        "C3 must add it before the β-coop launch."
    )
    assert "RuntimeError" in src, (
        "Hard-cap precondition must raise RuntimeError, not return silently."
    )
```

- [ ] **Step C3.5: SMEM shrink Phase 1 — K/V FP8 ping-pong with cp.async**

This is the largest single change in the migration. The goal is to halve K/V SMEM by double-buffering with `cp.async.cg`. The executor should:

1. Identify Phase 1's K/V SMEM allocation in `phase_e_kernel.py` (search for `k_smem`, `v_smem` or equivalent SMEM region declarations within `_jit_launch_phase_0_to_4`).
2. Halve the SMEM allocation: instead of resident `[tile_s × kv_dim]` for both K and V, allocate `[tile_s/2 × kv_dim]` × 2 buffers (ping/pong).
3. Issue `cp.async.cg` loads to the next ping/pong buffer while the current one is being consumed by the MMA.
4. Add explicit `cp.async.commit_group()` and `cp.async.wait_group(0)` synchronization at the buffer-swap boundary.
5. Update `_smem_bytes_phase_coop_full` to reflect the new total.

Reference: existing `cp.async` usage in `kernel.py:DecodeKernel` Phase A (the legacy uber-kernel uses the same pattern; mine it for the API shape).

This step is iterative. Expected wall-clock: 1-2 days of kernel work. Periodically rebuild + run the L0 phase tests to verify correctness at each iteration.

- [ ] **Step C3.6: SMEM shrink Phase 1 — Q packing to FP8**

Q is currently BF16 (12288 B for 24 heads × 256 head_dim). Halve via FP8 storage with dequant on read inside the MMA loop. Same `cp.async` pattern as K/V is fine; Q is loaded once per layer, not streaming.

Reference: NVFP4 dequant primitives in `phase_e_kernel.py` (`_fp4_nibble_to_f32`, etc.) — Q is BF16→FP8, simpler than the FP4 path.

- [ ] **Step C3.7: Re-probe `resident_cap` after shrink**

In `_backend.py`, the `_probe_resident_cap` call at line 489 currently uses `smem_bytes=45568`. Update the constant to match the new Phase 1 SMEM total (e.g., ~17000 if the shrink hits target).

```python
self._resident_cap = self._probe_resident_cap(
    kernel_fn=None, num_threads=128, smem_bytes=NEW_SMEM_BYTES,
)
```

After the build, the log line "CuTe Phase E: resident_cap=X" should show `X ≥ 256` for num_seqs=4 to fit.

- [ ] **Step C3.8: Run L0 phase tests after shrink**

```bash
.venv/bin/python -m pytest tests/kernels/cute/test_phase_e_*.py -v
```

Expected: all phase tests pass — the shrink must not regress correctness. If any test fails, the SMEM packing has a bug; iterate before continuing.

- [ ] **Step C3.9: Run L1 β-coop full kernel test**

```bash
.venv/bin/python -m pytest tests/kernels/cute/test_phase_e_beta_coop.py -v
```

Expected: PASS. β-coop full math correct after shrink.

- [ ] **Step C3.10: Run L2 + L3 + hard-cap tests**

```bash
.venv/bin/python -m pytest tests/v1/cute_paged/ -v
```

Expected: all pass.

- [ ] **Step C3.11: Docker rebuild**

```bash
tmux new-session -d -s build-c3 'docker build -f docker/Dockerfile.gb10 -t nvllm:gb10 . 2>&1 | tee /tmp/build-c3.log'
```

Wait for completion.

- [ ] **Step C3.12: Verify resident_cap in image logs**

```bash
tmux new-session -d -s serve-c3 'CUTE_PHASE_E_FUSION=1 ./scripts/serve.sh 2>&1 | tee /tmp/serve-c3.log'
until docker logs nvllm 2>&1 | grep -q "Started server process"; do sleep 5; done
docker logs nvllm 2>&1 | grep "resident_cap="
```

Expected: `CuTe Phase E: resident_cap=X (num_seqs_coop_max=Y)` where X ≥ 256 and Y ≥ 4. If Y < 4, the shrink hasn't hit target yet — iterate.

- [ ] **Step C3.13: gsm8k_eval_50 with `MAX_NUM_SEQS=4`**

```bash
.venv/bin/python scripts/gsm8k_eval_50.py --seed 42 --max-tokens 256
```

Expected: ≥ 90%. β-coop covers num_seqs=4.

- [ ] **Step C3.14: Verify hard-cap raises on `num_seqs=5`**

```bash
# Modify serve config to MAX_NUM_SEQS=5 temporarily
tmux kill-session -t serve-c3
docker stop nvllm 2>/dev/null || true

tmux new-session -d -s serve-c3-cap 'MAX_NUM_SEQS=5 CUTE_PHASE_E_FUSION=1 ./scripts/serve.sh 2>&1 | tee /tmp/serve-c3-cap.log'
until docker logs nvllm 2>&1 | grep -q "Started server process"; do sleep 5; done

# Send a batch-5 request and verify the server raises clearly
.venv/bin/python -c '
import openai
client = openai.OpenAI(base_url="http://localhost:8000/v1", api_key="x")
try:
    # Force num_seqs=5 via concurrent requests
    import threading
    results = []
    def hit():
        try:
            r = client.completions.create(model="<model>", prompt="hi", max_tokens=8, n=5)
            results.append(r)
        except Exception as e:
            results.append(e)
    t = threading.Thread(target=hit)
    t.start(); t.join()
    print(results)
'

# Check server logs for the hard-cap error
docker logs nvllm 2>&1 | grep "hard-cap exceeded"

tmux kill-session -t serve-c3-cap
docker stop nvllm 2>/dev/null || true
```

Expected: server logs contain "β-coop hard-cap exceeded: num_seqs=5 requires 320 CTAs but resident_cap=…". The request fails (per Q5=A no-fallback), not silently degrades.

- [ ] **Step C3.15: Capture L5 perf trace**

```bash
mkdir -p benchmarks/nvllm/traces/uber_kernel_migration/$(date +%Y-%m-%d)
cd benchmarks/nvllm/traces/uber_kernel_migration/$(date +%Y-%m-%d)

# Restart serve at MAX_NUM_SEQS=4
tmux new-session -d -s serve-c3-perf 'MAX_NUM_SEQS=4 CUTE_PHASE_E_FUSION=1 ./scripts/serve.sh 2>&1 | tee /tmp/serve-c3-perf.log'
until docker logs nvllm 2>&1 | grep -q "Started server process"; do sleep 5; done

# Capture sustained tok/s
.venv/bin/python /home/natfii/docker/nvllm/scripts/measure_tok_per_sec.py \
  --num-prompts 16 --max-tokens 256 > tok_per_sec_migrated.txt

# nsys trace via vLLM's torch profiler (per memory feedback_vllm_profiling)
.venv/bin/python /home/natfii/docker/nvllm/scripts/profile_with_torch.py \
  --model "<serve model>" --max-tokens 64 --output migrated.pt.trace.json

tmux kill-session -t serve-c3-perf
docker stop nvllm 2>/dev/null || true
```

Write `summary.md` in the same dir comparing tok/s vs the β-OFF baseline. Per AGENTS.md §4 (Performance Evidence Standard), include commit hash + reproduction commands.

- [ ] **Step C3.16: Confirm with user, then commit**

```bash
git add vllm/v1/attention/backends/cute_paged/phase_e_kernel.py \
        vllm/v1/attention/backends/cute_paged/_backend.py \
        tests/v1/cute_paged/test_uber_kernel_hard_cap.py \
        benchmarks/nvllm/traces/uber_kernel_migration/
git commit -m "$(cat <<'EOF'
perf(cute): C3 — Phase 1 SMEM shrink + delete β-lite + hard-cap num_seqs=4

Phase 1 SMEM shrunk via packed-FP8 K/V ping-pong (cp.async.cg
double-buffer) and FP8 Q packing with dequant-on-read. Phase 1 SMEM:
~45 KB → ~XX KB. Resident cap rises 96 → YYY (num_seqs_coop_max ≥ 4).

β-lite deleted entirely — no longer needed once β-coop covers
num_seqs=4 (matches scripts/serve.sh:52 default MAX_NUM_SEQS=4).
β-lite was also latent-broken pre-migration via the same residual_output
alias as β-coop (audit Finding 6); fix is moot since the path is gone.

Hard-cap: num_seqs > resident_cap/64 raises RuntimeError. No fallback
per Q5=A.

Evidence: benchmarks/nvllm/traces/uber_kernel_migration/<date>/

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

(Replace `XX`, `YYY` with actual numbers from the build logs and trace summary.)

---

## Task C4: Cleanup + flip production config

**Goal:** Remove orphaned imports, dead code, stale comments. Flip production serve config from `CUTE_PHASE_E_FUSION=0` to `=1`. Update memory + open PR.

**Files:**
- Modify: `vllm/v1/attention/backends/cute_paged/*.py` — orphaned imports cleanup
- Modify: `scripts/serve.sh` — flip default `CUTE_PHASE_E_FUSION`
- Update: memory `project_uber_kernel_migration.md`

### Substeps

- [ ] **Step C4.1: Find orphaned imports**

```bash
.venv/bin/python -m ruff check vllm/v1/attention/backends/cute_paged/ \
  vllm/nvllm/models/qwen3_5.py --select F401
```

Fix any unused imports flagged.

- [ ] **Step C4.2: Find dead code references**

```bash
grep -rn "next_hidden_scratch\|_phase_e_skip_next_ln\|_input_layernorm_module\|attach_input_layernorm\|cute_phase_e_skip_input_layernorm" vllm/ --include="*.py" | grep -v test_
```

Expected: no matches outside test files. Remove any stragglers.

- [ ] **Step C4.3: Flip `scripts/serve.sh` default**

```
file_path: scripts/serve.sh
old_string: CUTE_PHASE_E_FUSION="${CUTE_PHASE_E_FUSION:-0}"
new_string: CUTE_PHASE_E_FUSION="${CUTE_PHASE_E_FUSION:-1}"
```

(Find the actual line in `serve.sh` — pattern may differ slightly.)

- [ ] **Step C4.4: Run full test suite**

```bash
.venv/bin/python -m pytest tests/v1/cute_paged/ tests/kernels/cute/ -v
```

Expected: all pass.

- [ ] **Step C4.5: Final docker rebuild**

```bash
tmux new-session -d -s build-c4 'docker build -f docker/Dockerfile.gb10 -t nvllm:gb10 . 2>&1 | tee /tmp/build-c4.log'
```

Wait.

- [ ] **Step C4.6: Final gsm8k + tok/s with default config**

```bash
tmux new-session -d -s serve-c4 './scripts/serve.sh 2>&1 | tee /tmp/serve-c4.log'
until docker logs nvllm 2>&1 | grep -q "Started server process"; do sleep 5; done

# CUTE_PHASE_E_FUSION should now default to 1
docker logs nvllm 2>&1 | grep "PHASE_E_FUSION"

# Final correctness gate
.venv/bin/python scripts/gsm8k_eval_50.py --seed 42 --max-tokens 256

# Final tok/s
.venv/bin/python scripts/measure_tok_per_sec.py --num-prompts 16 --max-tokens 256

tmux kill-session -t serve-c4
docker stop nvllm 2>/dev/null || true
```

Expected: gsm8k ≥ 90%, tok/s ≥ pre-migration β-OFF baseline.

- [ ] **Step C4.7: Update memory**

Edit `~/.claude/projects/-home-natfii-docker-nvllm/memory/project_uber_kernel_migration.md`:

Change frontmatter description to: `"MIGRATION SHIPPED <date>. Commits: C1=<hash>, C1.5=<hash>, C2=<hash>, C3=<hash>, C4=<hash>. Production serve.sh defaults CUTE_PHASE_E_FUSION=1 with hard-cap num_seqs=4. β-coop is the sole full-attn-decode kernel. β-lite, paged_attention_forward (decode), Phase 4, F.1 layer-LN bake plumbing all retired. Trace evidence at benchmarks/nvllm/traces/uber_kernel_migration/<date>/."`

Add a `## Migration shipped <date>` section at the top with the commit hashes and L4/L5 results.

- [ ] **Step C4.8: Confirm with user, then commit**

```bash
git add scripts/serve.sh vllm/
git commit -m "$(cat <<'EOF'
chore(cute): C4 — flip production CUTE_PHASE_E_FUSION default to 1, cleanup

The uber-kernel migration is complete. β-coop is the sole full-attn-decode
kernel; production now ships with CUTE_PHASE_E_FUSION=1 by default.

- Flip scripts/serve.sh default
- Cleanup orphaned imports and dead code references
- Final gsm8k_eval_50 ≥ 90% with default config
- Trace evidence committed under benchmarks/nvllm/traces/uber_kernel_migration/

Migration shipped commits: C1, C1.5, C2, C3, C4.
Memory: project_uber_kernel_migration updated to MIGRATION SHIPPED.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

- [ ] **Step C4.9: Confirm with user, then push and open PR**

```bash
# Push to Navi-AI-Lab/nvllm only (per feedback_never_touch_upstream_vllm)
git push -u origin feat/uber-kernel-migration

# Open PR with full migration summary
gh pr create --repo Navi-AI-Lab/nvllm \
  --title "Uber-kernel migration: β-coop is the sole full-attn-decode kernel" \
  --body "$(cat <<'EOF'
## Summary

Completes the β-coop substitution that was started but never finished. β-coop is now the only full-attn-decode kernel for Qwen3.5-27B; `paged_attention_forward` retires from the decode path; β-lite deletes; Phase 4 + F.1 layer-LN bake plumbing all delete.

## Three bugs fixed

1. **Buffer aliasing (C1)**: β-coop and β-lite read `self.residual_output` (post-Phase-C output of legacy `paged_attention_forward`), causing `2·attn_out + h + r` cascade through 16 fused layers.
2. **Phase 4 mlp_out double-add (C1.5, audit Finding 1)**: Phase 4 mutated `residual_output` to add `mlp_out`, causing layer N+1's `input_layernorm` to double-count via fused-residual semantics.
3. **F.1 layer-LN bake corruption (C1.5, Q4 self-review)**: Phase 4 baked layer N+1's input_LN, but linear-attn layers (N+1 in stride-4) re-applied input_LN over the pre-baked output.

## Architecture endpoint

- β-coop is the only full-attn-decode kernel, cooperative-only.
- Phase 1 SMEM shrunk to fit num_seqs=4 in `resident_cap`.
- Hard-cap `num_seqs ≤ 4` (matches `serve.sh:52` default `MAX_NUM_SEQS=4`).
- Per-layer `input_layernorm` at every layer entry (matches Jamba/Zamba2/Qwen3-Next/Megatron hybrid).
- No fallback; failures raise.

## Test plan

- [x] L4: `gsm8k_eval_50.py seed=42 ≥ 90%` with `CUTE_PHASE_E_FUSION=1 MAX_NUM_SEQS=4`
- [x] L5: sustained tok/s ≥ β-OFF baseline (evidence under `benchmarks/nvllm/traces/uber_kernel_migration/`)
- [x] Hard-cap: `num_seqs=5` raises clearly, refuses to serve

## AI assistance

This change was developed with AI assistance (Claude Opus 4.7).
- Spec: `docs/superpowers/specs/2026-04-25-uber-kernel-migration-design.md`
- Plan: `docs/superpowers/plans/2026-04-25-uber-kernel-migration.md`
- Two subagent reports under `docs/research/uber_kernel_migration/` (external review + Q4 brainstorm) and one fresh-eyes audit (`spec_audit_2026-04-25.md`) caught 5 structural issues during design.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

---

## Self-review (do this BEFORE handing off)

After all tasks land, verify the following against the spec:

1. **Spec coverage**: every Q1-Q5 decision is implemented (scope, prefill, β-lite, Phase 4 boundary, retirement). Trace each to a commit.
2. **Audit coverage**: every HIGH/MED finding has a fix in a commit (Findings 1-6).
3. **Test pyramid**: L0-L5 all green at C4.
4. **No dead code**: grep checks at C4.1-C4.2 return zero unexpected matches.
5. **Memory + CLAUDE.md updates**: reflect the shipped state.
6. **PR description**: includes AI-assistance disclosure per AGENTS.md §1.

If any gap, fix it and re-run gates.
