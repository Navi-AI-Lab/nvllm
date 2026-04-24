# Phase E.2 + F.1 — β Kernel Math Correctness + Opaque-Gate Refactor

**Date:** 2026-04-24 (revised post fresh-eyes audit)
**Target:** Fix latent β kernel math bug (Phase E.2), then unmask β outputs
via opaque-gate refactor (Phase F.1).
**Model:** `ig1/Qwen3.5-27B-NVFP4` (non-distilled)
**Hardware:** NVIDIA DGX Spark (GB10, SM120/121), 128 GB unified

## Revision history

- **2026-04-24 initial:** Phase F.1 only (opaque-gate refactor). Framed
  as a Phase F narrow slice to fix the β-lite "2× MLP firing" regression.
- **2026-04-24 revised (post spec-reviewer audit):** Audit caught a
  latent β kernel math bug that the F.1 fix would activate. Scope
  expanded to include Phase E.2 math fix + reference-diff gate as a
  **hard prerequisite** for F.1. Findings #2, #6, #10, #11 from the
  audit addressed; #8, #12, #13 acknowledged.

## Context

Phase D shipped at commit `2ffda1fb8` (D2e weight_global_scale fix).
Phase E β kernels shipped at commit `bc9037955` (Task 17 β-coop GSM8K
8/8 PIECEWISE). Both are on `main`.

**Discovery via fresh-eyes audit (2026-04-24):** Phase E's reported
"51.7% speedup" at `memory:project_phase_e_shipped` measures **β kernel
launch latency, not end-to-end decode wall time**. The β kernels run
but their output is orphaned by a dead-branched Python gate; the
legacy full path fires alongside and produces the actual decode output
GSM8K reads. See `memory:project_phase_e_phantom_speedup` for full
mechanism. Phase F.1 fixes the dead-branch, which unmasks a second
latent bug: the β kernels have wrong RMSNorm math (multiply by raw γ
instead of `(1 + γ)`). See `memory:project_phase_e_beta_math_bug`.

This spec now covers both:
- **Phase E.2** — fix β kernel math correctness + add reference-diff
  verification harness. Required for any correct consumption of β
  output.
- **Phase F.1** — opaque-gate refactor to unmask β's output under
  PIECEWISE CUDA graphs. Depends on E.2.

Together these are **still the first slice of Phase F** (FULL-graph
enablement). Remaining 9 blockers from `memory:project_cudagraph_blockers`
deferred to next session per scope decision.

## Motivation

Evidence committed 2026-04-24 at
`benchmarks/nvllm/traces/phase_e_1/2026-04-24-baseline-matched/summary.md`
shows β-lite at matched concurrency (`num_seqs=8`) is **+62.8% slower
per full_attn layer decode step** than the pre-Phase-E baseline:

```
baseline_matched (FUSION=0):  DecodeKernel + Phase_D_MLP × 1 = 121,537 μs
β-lite           (FUSION=1):  DecodeKernel + Phase_D_MLP × 2 = 197,886 μs
                                                               ────────
                                                               +76,349 μs/layer/step
```

Per-call MLP mean is 13.5% faster (90,408 vs 104,499 μs), but
`Phase_D_MLP_Kernel` fires **twice** per full_attn layer per decode
step (trace: `n_calls = 2016` vs baseline's `1008`).

### Root cause

In `vllm/nvllm/models/qwen3_5.py:473-481`:

```python
if getattr(impl, "_phase_e_consumed", False):
    hidden_states = impl.next_hidden_scratch[:nat]
    residual = impl.residual_output[:nat]
    impl._phase_e_consumed = False
    return hidden_states, residual

hidden_states = self.mlp(hidden_states)
```

This `if` is a **Python-conditional gate evaluated at trace time under
`@support_torch_compile`** (see `Qwen3_5Model` decorator). At
torch.compile trace time `_phase_e_consumed=False`, so the `if` branch
dead-code-eliminates. At PIECEWISE replay, `self.mlp(hidden_states)`
fires **unconditionally** — even after β-lite/β-coop already produced
the correct outputs into `impl.next_hidden_scratch` and
`impl.residual_output`. The β kernel output is discarded; the fallback
MLP kernel fires again, writing the identical-shape output (but
recomputed from scratch).

The TODO at `qwen3_5.py:459-468` predicted this exactly:

> Task 10 MUST test first with --debug (--enforce-eager) to validate
> β-lite math. Before enabling PIECEWISE, refactor this gate into an
> opaque custom op (mirror torch.ops.vllm.cute_mlp_forward pattern
> from _mlp_op.py).

This spec is that refactor. Also applies to β-coop — same gate, same
bug (β-coop trace's `Phase_D_MLP_Kernel: 5040` n_calls may be the same
redundant-fire pattern; the opaque-gate fix settles it either way).

### Latent β kernel math bug (discovered during audit)

`Qwen3_5RMSNorm.forward_native` at `vllm/nvllm/layers/layernorm.py:53,
78` multiplies by `(1.0 + weight.float())`. The class docstring at
`:28-30` is explicit: *"Identical semantics to GemmaRMSNorm: x × (1 + w)
instead of x × w"*.

Both β kernels multiply by **raw γ**, missing the `+ 1`:

- β-lite ε epilogue: `vllm/v1/attention/backends/cute_paged/
  mlp_kernel.py:1502` — `out_f32 = normed_round * gamma_f32`
- β-coop Phase 0 prologue: `vllm/v1/attention/backends/cute_paged/
  phase_e_kernel.py:641` — `normed = (h_f32 + r_f32) * inv_rms_val *
  gamma_f32`

No pre-adjustment of γ happens anywhere in the codebase (grepped for
`weight + 1`, `1 + weight`, `gamma + 1`, etc. — only Python RMSNorm
forward paths contain the `+ 1`).

β kernel output is off by factor `γ/(1+γ)` from the Python reference.

**Today this manifests as nothing** — the consume branch at
`qwen3_5.py:473` is dead-branched, so β's wrong output is orphaned.
**Phase F.1's opaque-gate fix would activate the consume branch and
feed this wrong output to layer N+1.** GSM8K 8/8 currently passes only
because legacy Python path computes the right answer; it would break.

Additional problem: β-lite's ε epilogue and β-coop's Phase 4 epilogue
compute `next_hidden = RMSNorm(residual_final) × γ_{N+1}` intending
for layer N+1 to skip its input_layernorm. But there is **no skip
mechanism** at `qwen3_5.py:386` — the next layer's forward runs
`self.input_layernorm(hidden_states, residual)` unconditionally,
which would then double-process the already-processed tensor.

### Expected win (after E.2 math fix + F.1 opaque gate)

Under PIECEWISE, eliminating the second MLP fire (F.1) and getting
correct β output (E.2):

```
β-lite fixed:  DecodeKernel + Phase_D_MLP × 1 = 107,478 μs/layer/step
vs baseline:                                    121,537 μs/layer/step
                                                ──────────
                                                −14,059 μs (−11.6%)
```

β-lite returns to net win vs baseline instead of net regression. Plus
foundation for FULL graphs where Python-conditional branches *must* be
opaque. **AND** Phase E's performance numbers finally reflect
end-to-end decode latency instead of kernel launch latency (phantom
speedup resolved).

## Scope

**In scope (this spec):**
- **Phase E.2 (prerequisite):** Fix β kernel math at `mlp_kernel.py:1502`
  and `phase_e_kernel.py:641` — multiply by `(1 + γ)`, not raw `γ`.
  Audit `phase_e_kernel.py` Phase 4 epilogue for the same pattern and
  fix if present.
- **Phase E.2 (prerequisite):** Add `CUTE_DEBUG_FUSION`-style
  per-layer reference-diff harness comparing β kernel output to Python
  `Qwen3_5RMSNorm.forward_native` reference. Pass criterion:
  `torch.allclose(kernel_out, py_out, atol=1e-2, rtol=0)` at BF16 on
  every full_attn layer for one forward pass. Harness must be
  env-gated per `memory:feedback_keep_debug_harnesses` — keep, don't
  strip.
- **Phase E.2 (prerequisite):** Add `_skip_input_layernorm` mechanism
  so layer N+1 knows to bypass its `input_layernorm` when the consume
  branch activates. Implement as a second opaque custom op (or fold
  into `cute_phase_e_dispatch` — see Design Decisions below).
- **Phase F.1:** Fix `_phase_e_consumed` dead-branch via one new
  opaque custom op. Covers both β-lite and β-coop paths. PIECEWISE
  validation and evidence.

**Out of scope (deferred to next session and beyond):**
- The other 9 CUDA-graph blockers in `memory:project_cudagraph_blockers.md`.
  Phase F.2 will re-audit all 10 blockers against current main (D+E
  shipped state) and ship them individually.
- FULL graph mode itself — E.2 + F.1 are necessary-not-sufficient.
  FULL-mode gibberish persists until all blockers addressed.
- `_fusion_active` Python branch at `qwen3_5.py:423, 445` — **same bug
  class as `_phase_e_consumed`** per audit Finding #13. Phase B/C
  attention fusion output (`impl.rmsnorm_output`, `impl.residual_output`)
  may ALSO be phantom at PIECEWISE replay — `project_cute_paged_bench`
  wins need re-verification. Deferred to Phase F.N.
- K/V ping-pong SMEM shrink (was Phase E.1 #2) — deferred to next
  session after E.2+F.1 ships.
- PTX-diff safety check of sibling CuTe kernels post-edit (audit
  Finding #8). High-signal but adds infra; deferred unless the first
  GSM8K smoke regresses unexpectedly.

## Design Decisions

### Decision 1: Scope = Phase E.2 + F.1 coupled
**Chosen:** Ship E.2 (β kernel math fix + reference-diff gate + skip
mechanism) as a hard prerequisite for F.1 (opaque gate). Same commit
series; one evidence bundle.
**Why:** F.1 activates a branch that exposes E.2's latent bug. Shipping
F.1 without E.2 = correctness regression. Shipping E.2 without F.1 =
no visible effect (branch still dead). They have to land together.

### Decision 2: Op boundary = new sibling op
**Chosen:** B1 — add `torch.ops.vllm.cute_phase_e_dispatch` as sibling
to `cute_mlp_forward` in `_mlp_op.py`. Do NOT extend `cute_mlp_forward`.
**Why:** Single-responsibility; pins existing `cute_mlp_forward` fake_impl
signature (no retrace ripple); bisectable revert if win doesn't
materialize.

### Decision 3: Validation = TDD with reference-diff gate + evidence bundle
**Chosen:** C-A (extended) — β kernel reference diff FIRST, then Python
repro for op mechanics, then Docker rebuild, then GSM8K 8/8 smoke,
then torch-profiler trace with kernel-count AND numerical equivalence
pass criteria.
**Why:** The repro is cheap (~10 min) and catches op-registration
mistakes before a 30-50 min rebuild. The reference diff is cheaper
still (~5 min in Jupyter) and catches the math bug before burning any
rebuild cycle. Per `feedback_kernel_repro_before_rebuild`,
`feedback_no_shortcuts`, `feedback_debug_math_live`.

### Decision 4: Skip mechanism = second opaque custom op
**Chosen:** Add a sibling op `cute_phase_e_skip_input_layernorm` that
wraps layer N+1's `self.input_layernorm(hidden_states, residual)` call
site at `qwen3_5.py:386`. Op body checks `impl._phase_e_skip_next_ln`
(set by β kernel via `cute_phase_e_dispatch` when consuming) and
either passes through (skip) or runs `input_layernorm` normally.
**Why (vs folding into dispatch op):** Same dead-branch hazard applies
to `qwen3_5.py:386` — if we add a Python `if _skip_next_ln:` gate
there, it dead-branches too. Must be opaque. Two small ops with one
responsibility each, same pattern as `cute_mlp_forward`.
**Coupling risk:** `_phase_e_skip_next_ln` is a per-impl flag set by
layer N's dispatch and read by layer N+1's skip op. State flows across
a layer boundary — verify layer ordering in profile trace before
shipping.

### Decision 5: Fail-loud on consume-branch errors (audit Finding #10)
**Chosen:** Remove the try/except fall-through in the consume branch.
If the consume copy raises, raise `RuntimeError` up the stack — do
NOT silently degrade to `cute_mlp_forward`.
**Why:** The consume branch's output contract differs from
`cute_mlp_forward`'s (residual semantics). Silent fallback would produce
inconsistent contracts across layers — exactly the bug class
`memory:feedback_bare_assert_hides_bugs` warns against. If β fails
mid-copy, we want loud failure and a post-mortem, not silent wrong
output. The `_backend.py:1256-1264` β-lite/β-coop dispatch can still
fail-closed at the attention-backend level (leaves `_phase_e_consumed
=False`); that's the right layer for soft-fail, not the op body.

## Architecture

One new opaque custom op replaces one Python-conditional block:

```
Before (qwen3_5.py:473-481):           After:
─────────────────────────────────      ──────────────────────────────────
if _phase_e_consumed:                  hidden_out = torch.empty_like(hs)
    hs = impl.next_hidden_scratch       residual_out = torch.empty_like(r)
    r  = impl.residual_output           torch.ops.vllm.cute_phase_e_dispatch(
    return hs, r                            hs, hidden_out, residual_out, r, layer_name
hs = self.mlp(hs)                       )
                                        hs, r = hidden_out, residual_out
```

The op body branches on `impl._phase_e_consumed` at **runtime** (opaque
to torch.compile). PT2 sees one call per layer; kernel fires exactly
once per decode step regardless of which branch is taken.

Both β-lite and β-coop set `_phase_e_consumed=True` (`_backend.py:1188`
and `:1254`), so both benefit from this single fix.

## Components

### Added: `_cute_phase_e_dispatch_impl` in `_mlp_op.py`

Registration mirrors the existing `cute_mlp_forward` pattern — uses
`direct_register_custom_op` from `vllm.utils.torch_utils` (not a
decorator; plain function call at module load). Reuses the existing
`_CUTE_MLP_REGISTRY` dict because both ops key on the same per-impl
attach record:

```python
direct_register_custom_op(
    op_name="cute_phase_e_dispatch",
    op_func=_cute_phase_e_dispatch_impl,
    mutates_args=["hidden_out", "residual_out"],
    fake_impl=_cute_phase_e_dispatch_fake,
)
```

Op body (simplified, fail-loud per Decision 5 / audit Finding #10):

```python
def _cute_phase_e_dispatch_impl(
    x, hidden_out, residual_out, residual_in, layer_name,
):
    impl = _CUTE_MLP_REGISTRY.get(layer_name)
    if impl is None:
        raise RuntimeError(
            f"cute_phase_e_dispatch called for unregistered "
            f"layer {layer_name!r}"
        )
    nat = x.shape[0]

    if getattr(impl, "_phase_e_consumed", False):
        # Consume β output. Fail-loud — no try/except.
        hidden_out[:nat].copy_(impl.next_hidden_scratch[:nat])
        residual_out[:nat].copy_(impl.residual_output[:nat])
        impl._phase_e_consumed = False
        # Signal to next layer's cute_phase_e_skip_input_layernorm op
        # that input_layernorm is already applied (β's ε epilogue did it).
        impl._phase_e_skip_next_ln = True
        return

    # NOT-consumed: β didn't run this layer. Delegate to regular MLP op.
    # residual is pass-through.
    torch.ops.vllm.cute_mlp_forward(x, hidden_out, layer_name)
    residual_out.copy_(residual_in)
    impl._phase_e_skip_next_ln = False


def _cute_phase_e_dispatch_fake(
    x, hidden_out, residual_out, residual_in, layer_name,
):
    # Shape/dtype pinned by mutates_args declaration. No-op.
    return None
```

### Added: `_cute_phase_e_skip_input_layernorm_impl` (Decision 4)

Second sibling op, wraps layer N+1's `self.input_layernorm` call site
so the skip decision is opaque to PT2.

```python
def _cute_phase_e_skip_input_layernorm_impl(
    x, residual, out_x, out_residual, layer_name,
):
    """Wraps self.input_layernorm(x, residual) with opaque skip gate.

    Consumes impl._phase_e_skip_next_ln set by the PREVIOUS layer's
    cute_phase_e_dispatch (cross-layer state flow).
    """
    impl = _CUTE_MLP_REGISTRY.get(layer_name)
    if impl is None:
        raise RuntimeError(
            f"cute_phase_e_skip_input_layernorm called for unregistered "
            f"layer {layer_name!r}"
        )
    nat = x.shape[0]

    if getattr(impl, "_phase_e_skip_next_ln", False):
        # Prior layer's β kernel already applied THIS layer's input_layernorm
        # in its ε epilogue. Pass through.
        out_x[:nat].copy_(x[:nat])
        out_residual[:nat].copy_(residual[:nat])
        impl._phase_e_skip_next_ln = False
        return

    # Normal path: run input_layernorm.
    # Note: `input_layernorm_module` is a per-impl attribute set at
    # attach time (similar to attach_next_input_layernorm).
    ln_out, ln_residual = impl._input_layernorm_module(x, residual)
    out_x[:nat].copy_(ln_out[:nat])
    out_residual[:nat].copy_(ln_residual[:nat])
```

Decoder call site at `qwen3_5.py:386` becomes:
```python
_mlp_layer_name = getattr(self.mlp, "_cute_layer_name", None)
if _mlp_layer_name is not None:
    out_x = torch.empty_like(hidden_states)
    out_r = torch.empty_like(residual)
    torch.ops.vllm.cute_phase_e_skip_input_layernorm(
        hidden_states, residual, out_x, out_r, _mlp_layer_name
    )
    hidden_states, residual = out_x, out_r
else:
    hidden_states, residual = self.input_layernorm(hidden_states, residual)
```

Requires new `attach_input_layernorm` method on `CutePagedAttentionImpl`
(mirror of `attach_next_input_layernorm`).

### Modified: `vllm/nvllm/models/qwen3_5.py` lines 473-481

The decoder `forward()` serves both `linear_attention` and
`full_attention` layer types. Linear_attn layers never have MLP
fusion attached (see `__init__` gate at lines 355-373 —
`attach_mlp_fusion` only runs for `full_attention`). So the
dispatcher op only needs to run where `self.mlp._cute_layer_name`
is set (set by `attach_mlp_fusion`); otherwise the legacy
`self.mlp(...)` path is correct and we keep it.

The attach-state check is a **Python attribute lookup that returns
the same value for the life of the module** — not a runtime-mutable
flag like `_phase_e_consumed` was. torch.compile treats it as a
compile-time constant per module instance; each attached /
not-attached instance gets its own guarded compile. That's safe
(it's exactly how `self.layer_type == "full_attention"` branches
already work in this file).

Replace the `if _phase_e_consumed / return` block + `self.mlp(...)`
call with:

```python
# Phase E β-coop / β-lite consume gate moved into opaque custom op
# (see docs/superpowers/specs/2026-04-24-phase-f1-opaque-gate-refactor-design.md).
# Attach-state branch is init-time constant → trace-safe.
_mlp_layer_name = getattr(self.mlp, "_cute_layer_name", None)
if _mlp_layer_name is not None:
    hidden_out = torch.empty_like(hidden_states)
    residual_out = torch.empty_like(residual)
    torch.ops.vllm.cute_phase_e_dispatch(
        hidden_states, hidden_out, residual_out, residual,
        _mlp_layer_name,
    )
    hidden_states, residual = hidden_out, residual_out
else:
    hidden_states = self.mlp(hidden_states)
```

`layer_scale` handling (lines 483-495) and final `return` stay
as-is. The long TODO comment block at lines 450-472 is replaced by
the one-line reference above — the TODO is now resolved and its
comment rotted.

### Unchanged

- `cute_mlp_forward` op (and its `_cute_mlp_forward_fake` signature).
- `Qwen3_5MLP.forward` in `vllm/nvllm/layers/mlp.py` (still calls
  `cute_mlp_forward` directly; the new dispatcher does not replace it
  — it calls it).
- `Phase_D_MLP_Kernel`, `PhaseE_Beta_Kernel`, `DecodeKernel`, all
  CuTe kernels.
- `_backend.py` β-lite and β-coop dispatch blocks (lines 1117-1265).
  They still launch β kernels and set `_phase_e_consumed=True`.
- Linear_attn layer path.

## Data flow

**Op inputs:**
| Param | Shape | Purpose |
|---|---|---|
| `x` | [nat, hidden_dim] | post-attn, post-post_attn_layernorm hidden_states (MLP input when NOT consumed) |
| `hidden_out` | [nat, hidden_dim] | caller-allocated output (mutates_args) |
| `residual_out` | [nat, hidden_dim] | caller-allocated output (mutates_args) |
| `residual_in` | [nat, hidden_dim] | decoder's residual (passthrough source when NOT consumed) |
| `layer_name` | str | `_LAYER_REGISTRY` key |

**State read at runtime (opaque to PT2 trace):**
- `impl._phase_e_consumed : bool` — branch predicate
- Consumed: `impl.next_hidden_scratch[:nat]`, `impl.residual_output[:nat]`
- Not-consumed: whatever `cute_mlp_forward` reads (`impl._mlp_kernel`,
  weight buffers, etc.)

**State written:**
- `hidden_out`, `residual_out` (mutated via `.copy_`)
- `impl._phase_e_consumed = False` (cleared to prevent stale reuse)

**`torch.empty_like` note:** consistent with existing `Qwen3_5MLP.forward`
(`output = torch.empty_like(x)` at `mlp.py:77`). Graph-safe under
PIECEWISE (shapes are static per capture size). Will need to move to
persistent impl-owned buffers for FULL graphs — that's blocker #2 on
the 10-blockers list, deferred to Phase F.2+.

## Error handling

**Fail-loud philosophy (revised post-audit Finding #10):** exceptions
inside the op body are NOT caught. If the consume branch fails
mid-copy, raise up the stack and crash the request. Silent degradation
to `cute_mlp_forward` would produce a different output contract
(residual = `residual_post_ln` vs `residual_final`), giving inconsistent
semantics across the 16 full_attn layers — exactly the bug class
`memory:feedback_bare_assert_hides_bugs` warns against. If β fails,
we want to see it, not smear it across layers.

The **attention-backend-level fail-closed** at `_backend.py:1256-1264`
stays — if the β kernel launch itself throws, the backend logs and
leaves `_phase_e_consumed=False`, so the consume branch never activates
for that forward. That's the correct soft-fail layer.

**Hard errors (uncaught):**
- `layer_name` not in `_CUTE_MLP_REGISTRY` → `RuntimeError`.
  Programming bug, same as `cute_mlp_forward` today.
- Any exception during consume-branch `.copy_` → propagates.
  User-visible failure; likely indicates a kernel bug or shape drift
  that needs investigation, not a fallback.

**Calling `cute_mlp_forward` from inside the dispatcher:** legal under
vLLM's dispatch; nested opaque ops stay opaque. Keeps one source of
truth for the MLP launch path — when `cute_mlp_forward` evolves, the
dispatcher inherits.

## Testing ladder

Four layers, each gates the next. Revised per audit Findings #6, #11.

### Layer 0 — β kernel reference diff (prerequisite for E.2 math fix)

Before any integration testing, verify the β kernels produce output
matching the Python reference. Run in Jupyter on Spark (per
`memory:reference_jupyter_spark`, `memory:feedback_debug_math_live`).

- Construct synthetic inputs: `residual_final` (BF16, random), γ (BF16,
  random trained-range), ε = 1e-6.
- Python reference: `Qwen3_5RMSNorm.forward_native(...)` — produces
  `out_py = RMSNorm(residual_final) × (1 + γ)`.
- β-lite kernel: launch `Phase_D_MLP_Kernel` with only the ε epilogue
  active (emit_epilogue=1), capture `next_hidden_scratch`.
- β-coop kernel: launch Phase 0 only (`run_phase_0`), capture normed.

**Pass criterion:**
```python
torch.allclose(kernel_out.float(), py_out.float(),
               atol=1e-2, rtol=0)  # BF16-appropriate
```

If fails: the `(1 + γ)` fix in `mlp_kernel.py:1502` and
`phase_e_kernel.py:641` hasn't landed (or has a typo). Fix and retry
BEFORE any rebuild.

Commit the harness under `docs/research/phase_e2_beta_math/` (per
`memory:feedback_keep_debug_harnesses` — don't strip).

### Layer 1 — Python repro for op-registration mechanics (`/tmp/phase_f1_repro.py`)

~50-line standalone script, no nvllm, no Docker, no GPU model. Revised
per audit Finding #11: tests op MECHANICS, not dead-branch (which is
already proven by the phase_e_1 trace).

- Register a dummy `cute_phase_e_dispatch`-shaped op via
  `direct_register_custom_op` with `mutates_args=["hidden_out",
  "residual_out"]`.
- Register matching `fake_impl` with same signature.
- Compile a toy model that calls the op via `torch.ops.vllm.*`.
- Verify:
  - `mutates_args` correctly pins outputs (no unnecessary copies).
  - `fake_impl` signature matches exactly (no type mismatch).
  - Nested op call (real op body calls `torch.ops.vllm.cute_mlp_forward`
    mock) works under `torch.compile`.
  - `layer_name: str` threads correctly to registry lookup.

**Pass criterion:** compilation + run with no signature errors, no
unexpected memory copies. Runs in seconds.

### Layer 2 — Docker rebuild + GSM8K 8/8 PIECEWISE smoke

- Rebuild `nvllm:gb10` in tmux (per `memory:feedback_delegate_builds`).
- Serve `ig1/Qwen3.5-27B-NVFP4`, PIECEWISE, `max_num_seqs=8`,
  `CUTE_PHASE_E_FUSION=1`, `CUTE_PHASE_E_PATH=lite`.
- Run GSM8K 8/8 `/no_think` via `/v1/completions` (per
  `memory:feedback_eval_completions`).
- Repeat with `PATH=coop` at `max_num_seqs=1`.

**Pass criterion:** 8/8 correct on both legs. Same bar Phase E Task 17
cleared — but NOTE that Phase E Task 17 passed even WITH the math bug
(β output was orphaned). So 8/8 here is necessary but not sufficient.
Layer 3's numerical equivalence is the real correctness gate.

### Layer 3 — Kernel-count + numerical equivalence (definitive evidence)

Two pass criteria, both required. Revised per audit Finding #6.

**Criterion A (kernel count, performance gate):**
Re-run `capture_baseline_matched.sh`-style leg with fix enabled:
- `CUTE_PHASE_E_FUSION=1`, `PATH=lite`, `num_seqs=8`, `concurrent=8`,
  `max_tokens=64`, `active_iterations=200`, PIECEWISE graphs.

```
Phase_D_MLP_Kernel n_calls: 2016 → 1008 (exact 2× halving)
```

**Criterion B (numerical equivalence, correctness gate):**
Element-wise compare layer N+1 input tensors between
(a) `CUTE_PHASE_E_FUSION=0` (legacy path) and
(b) `CUTE_PHASE_E_FUSION=1` (with E.2+F.1 fix).

Same prompts, same seed, same PIECEWISE graph mode, same
`max_tokens`. Dump `hidden_states` and `residual` at layer N+1 entry
(hook at `qwen3_5.py:forward` post-input_layernorm) for all 16
full_attn layers for one decode step.

```python
torch.allclose(legacy_hidden, beta_hidden, atol=1e-2, rtol=0)
torch.allclose(legacy_residual, beta_residual, atol=1e-2, rtol=0)
```

Must hold for ALL 16 full_attn layers. This is the gate that would
have caught the math bug if it existed in Phase E as shipped. (It
would have caught it; Phase E didn't have this test.)

**Secondary metrics** (recorded in summary.md, informational only):
- `Phase_D_MLP_Kernel` per-call mean μs
- Per-full-attn-layer decode cost vs baseline_matched
- β-coop leg re-trace at `num_seqs=1`: `Phase_D_MLP_Kernel` n_calls
  should drop significantly from 5040 — confirms β-coop was also
  phantom-firing (audit Finding #12). **This is also the evidence to
  update `project_phase_e_shipped.md` with true end-to-end numbers.**

### Evidence bundle

`benchmarks/nvllm/traces/phase_f/2026-04-24-opaque-gate/`:
| File | Notes |
|---|---|
| `beta_lite_fixed.pt.trace.json.gz` | local-only (gitignored) |
| `beta_lite_fixed_kernels.csv` | committed |
| `beta_lite_fixed_serve.log` | committed |
| `beta_coop_fixed_kernels.csv` | committed |
| `beta_coop_fixed_serve.log` | committed |
| `summary.md` | committed — before/after kernel counts, per-call μs, per-layer-step cost, vs `baseline_matched` |

New gitignore rule:
```
benchmarks/nvllm/traces/phase_f/**/*.pt.trace.json.gz
!benchmarks/nvllm/traces/phase_f/**/*.csv
!benchmarks/nvllm/traces/phase_f/**/*.log
!benchmarks/nvllm/traces/phase_f/**/*.md
!benchmarks/nvllm/traces/phase_f/**/*.txt
!benchmarks/nvllm/traces/phase_f/**/*.json
```

## Success criteria

E.2 + F.1 is done when ALL of:
1. **Layer 0**: β kernel reference diff passes (`torch.allclose` at
   `atol=1e-2` BF16) for both β-lite ε epilogue and β-coop Phase 0.
   Harness committed under `docs/research/phase_e2_beta_math/`.
2. **Layer 1**: `/tmp/phase_f1_repro.py` passes (op registration +
   fake_impl + mutates_args + nested op call work under torch.compile).
3. **Layer 2**: `ig1/Qwen3.5-27B-NVFP4` GSM8K 8/8 PIECEWISE under
   `PATH=lite` AND `PATH=coop`.
4. **Layer 3 Criterion A**: torch-profiler trace shows
   `Phase_D_MLP_Kernel n_calls: 2016 → 1008` at `num_seqs=8`.
5. **Layer 3 Criterion B**: layer-by-layer `torch.allclose` between
   legacy path (FUSION=0) and β path (FUSION=1) at every full_attn
   layer entry, `atol=1e-2` BF16.
6. Summary committed under
   `benchmarks/nvllm/traces/phase_f/2026-04-24-opaque-gate/` with
   before/after kernel counts AND numerical-equivalence evidence.
7. Memory updated:
   - `project_phase_e_shipped.md` superseded with true end-to-end
     numbers (phantom speedup resolved).
   - `project_phase_e_phantom_speedup.md` and
     `project_phase_e_beta_math_bug.md` flagged as RESOLVED.
   - New `project_phase_f_inflight.md` tracks remaining 9 blockers.

## Rollback plan

If any test layer fails and root cause can't be resolved same session:
- Revert the two commits (new op registration, decoder call-site change).
- β-lite returns to its documented 2× firing regression.
- β-coop returns to its documented state.
- Zero impact on Phase D, Phase E shipped state, linear_attn path, or
  any other code path.

## Non-goals

- Fixing other Python-conditional branches in `qwen3_5.py` (e.g.
  `_fusion_active` at lines 423, 445). Same bug class; deferred to
  Phase F.N. **NOTE (audit Finding #13):** this may mean Phase B/C
  attention fusion is ALSO phantom today, and the wins reported in
  `memory:project_cute_paged_bench` may not reflect end-to-end. Phase
  F.N should treat it the same way as this spec treats Phase E —
  verify kernel output matches Python reference, then opaque-gate the
  decoder call site.
- Replacing `torch.empty_like` with persistent buffers. That's an
  FULL-graph dynamic-allocation blocker; deferred to Phase F.2+.
- Enabling `FULL_AND_PIECEWISE` CUDA graph mode. E.2+F.1 are
  necessary-not-sufficient; 9 other blockers remain.
- SMEM shrink for β-coop at `num_seqs=2` (Phase E.1 #2). Deferred to
  next session after E.2+F.1 ships.
- PTX-diff safety check of sibling CuTe kernels post-edit (audit
  Finding #8). Noted but deferred unless Layer 2 GSM8K smoke produces
  unexpected numerical drift.

## Future work (Phase F.2+)

After F.1 ships, next session brainstorms scope B (broad re-audit of
10-blockers list vs current main). Remaining items from
`memory:project_cudagraph_blockers.md` (may already be partially
addressed by Phase D/E commits — needs fresh audit):

1. `wo_global_scale.item()` device-to-host sync
2. `torch.empty_like(query)` dynamic allocation
3. `grid.z = num_seqs` non-uniform shape
4. `query.contiguous().view(-1)` implicit copy
5. Python branching on `wo_weight is not None` ← **same class as F.1**
6. Side-channel `_wo_weight` set/clear cycle
7. `_arrival_count_buf` lazy growth
8. `torch.zeros(num_tokens, hidden_dim)` fresh alloc
9. Output gate (sigmoid·gate) fusion
10. `build_for_cudagraph_capture` override

## Audit findings disposition

All 14 findings from the 2026-04-24 spec-reviewer audit disposed of:

| # | Severity | Status |
|---|---|---|
| 1 | positive | confirmed + kept as spec's core premise |
| 2a/b | **critical** | **resolved — added E.2 math fix + skip mechanism + Layer-0 reference diff** |
| 3 | minor | noted — β-lite failure path doesn't re-introduce double-fire, fail-loud per Decision 5 |
| 4 | minor-positive | kept — `_cute_layer_name` gate is trace-safe by init-time constancy |
| 5 | minor | noted — invariant `_cute_layer_name` set once per instance; comment at attach site (implementation detail) |
| 6 | major | **resolved — Layer 3 Criterion B: layer-by-layer numerical equivalence** |
| 7 | nitpick | acknowledged — blocker #2 refers to attention kernel, not MLP; noted in scope |
| 8 | major | **acknowledged — PTX-diff deferred unless Layer 2 regresses unexpectedly** |
| 9 | minor | accepted tradeoff — nested custom-op dispatch overhead; monitor, don't optimize prematurely |
| 10 | major | **resolved — Decision 5 fail-loud; try/except removed from op body** |
| 11 | minor | **resolved — Layer 1 repro revised to test op mechanics, not dead-branch** |
| 12 | major | **acknowledged — phantom Phase E confirmed in memory; resolved as secondary outcome of Layer 3 Criterion A+B** |
| 13 | major | **acknowledged — same bug class at `_fusion_active`; deferred to Phase F.N with explicit note that `project_cute_paged_bench` wins may also be phantom** |
| 14 | positive | kept — validation ladder, scope discipline, mirror-pattern citation strongest aspects |

## References

- Pre-fix evidence: `benchmarks/nvllm/traces/phase_e_1/2026-04-24-baseline-matched/summary.md`
- Phase E ship: `benchmarks/nvllm/traces/phase_e/2026-04-23-initial/summary.md`
- Root-cause TODO: `vllm/nvllm/models/qwen3_5.py:459-468`
- Mirror pattern: `vllm/v1/attention/backends/cute_paged/_mlp_op.py`
- β math bug locations: `vllm/v1/attention/backends/cute_paged/mlp_kernel.py:1502`,
  `vllm/v1/attention/backends/cute_paged/phase_e_kernel.py:641`
- Python RMSNorm reference: `vllm/nvllm/layers/layernorm.py:53, 78`
- Blocker list: `memory:project_cudagraph_blockers.md`
- Audit memories:
  `memory:project_phase_e_phantom_speedup.md`,
  `memory:project_phase_e_beta_math_bug.md`,
  `memory:feedback_audit_before_code.md`
- Related patterns: `memory:feedback_opaque_op_not_enough`,
  `memory:project_cute_not_capturing`
