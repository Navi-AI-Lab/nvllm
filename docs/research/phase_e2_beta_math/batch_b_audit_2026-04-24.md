# Batch B audit — raw-γ scope expansion finding

**Date:** 2026-04-24
**Author:** Claude (controller, not committed yet)
**Status:** PAUSED — awaiting user decision on scope

## Context

Batch A (commit `98551dba6`) fixed the β-lite ε epilogue at `mlp_kernel.py:1502`
to use `(1 + γ)` instead of raw γ, matching `Qwen3_5RMSNorm` semantics.

Plan Batch B (Tasks 4–6) calls for:
- Task 5: fix `phase_e_kernel.py:641` (β-coop Phase 0 prologue)
- Task 6: audit Phase 4 ε epilogue, fix if same pattern present

Audit ran. Found **6 additional raw-γ multiply sites** beyond the plan's targets,
spanning two source files, plus one bad reference in an existing test.

## All raw-γ multiply sites found

| File | Line | Function | Path | Pattern |
|---|---|---|---|---|
| `mlp_kernel.py` | 1502 | β-lite ε epilogue | **PRODUCTION** | ✅ FIXED Batch A |
| `phase_e_kernel.py` | 641 | `run_phase_0_only` | TEST-ONLY | Plan Task 5 target |
| `phase_e_kernel.py` | 853 | `run_phase_01_only` Phase 0 | TEST-ONLY | NEW (out of plan) |
| `phase_e_kernel.py` | 1543 | `run_phase_01_only` Phase C | TEST-ONLY | NEW (out of plan) |
| `phase_e_kernel.py` | 2625 | `run_phase_4_only` ε epilogue | TEST-ONLY | Plan Task 6 target (one of two) |
| `phase_e_kernel.py` | 3274 | **`run_beta_coop_full` Phase 0** | **PRODUCTION** | NEW (out of plan) |
| `phase_e_kernel.py` | 3942 | **`run_beta_coop_full` Phase C** | **PRODUCTION** | NEW (out of plan) |
| `phase_e_kernel.py` | 4641 | **`run_beta_coop_full` Phase 4 ε** | **PRODUCTION** | Plan Task 6 target (other) |
| `kernel.py` | 1921 | standalone DecodeKernel Phase C | **PRODUCTION** | NEW (out of plan, separate file) |

## Why production has been working despite these bugs

The PIECEWISE-dead-branched gates (`_phase_e_consumed` at qwen3_5.py:473,
`_fusion_active` at qwen3_5.py:423/445) orphan the kernel outputs at replay
time. The Python fallback recomputes correctly via `Qwen3_5RMSNorm.forward_native`
(which uses `x * (1 + weight)`). This matches `memory:project_phase_e_phantom_speedup`.

**Implication for Phase F.1 (next session of work):** once F.1 lands the opaque
ops and the kernel outputs ARE consumed at replay, every uncorrected raw-γ site
will start producing wrong output → GSM8K Layer 2 smoke (Task 15) will fail
unless ALL production sites are fixed.

## Bad reference also found

`tests/kernels/cute/test_phase_e_epsilon_epilogue.py:156`:
```python
normed_ref = ((summed * rstd) * gamma.float()).to(torch.bfloat16)  # raw γ
```

Same pattern as the bug we fixed in `2026-04-22-phase-e-repro.py:32`. This is
why `test_phase_0_prologue_matches_rmsnorm_ref` passes today — both kernel
and ref share the bug. Once kernel is fixed, ref must be fixed too.

## Two options for the user

### Option A — Strict literal plan (do nothing else)

Fix only the lines the plan names: 641 (Task 5), 2625 + 4641 (Task 6 = "Phase 4
audit"). Leave 853, 1543, 3274, 3942, kernel.py:1921 untouched.

**Outcome:** test_phase_4_matches_python_epsilon_ref[True] passes; Batch B
commits as planned. **But Phase F.1 GSM8K (Task 15) will fail** because
β-coop production at lines 3274/3942 still uses raw γ.

### Option B — Spec-driven scope (recommended)

The spec says "fix γ math in *both β kernels*". β-coop's production path is
`run_beta_coop_full` (line 2681) — that function has THREE raw-γ sites
(3274, 3942, 4641). All three must be (1+γ) for β-coop to be correct under
F.1's opaque-gate consumption.

Plus all test-only sites (641, 853, 1543, 2625) for consistency, and the bad
reference at test_phase_e_epsilon_epilogue.py:156.

**Out-of-spec but related:** kernel.py:1921 (standalone DecodeKernel Phase C
post-attn rmsnorm) — this is what β-lite calls. Same bug but in a separate
file outside spec's "both β kernels" scope. Currently masked by `_fusion_active`
dead-branch (qwen3_5.py:423/445 — same bug class). Probably needs its own
issue, separate from E.2.

### Recommended ordering for Option B

1. Write new failing tests in `test_phase_e2_beta_math.py` for β-coop Phase 0
   (against `Qwen3_5RMSNorm._forward_static_with_residual`) and β-coop Phase 4 ε
   (against `_forward_static_no_residual`).
2. Fix all 7 phase_e_kernel.py sites + bad ref in test_phase_e_epsilon_epilogue.py.
3. Re-run pytest tests/kernels/cute/test_phase_e2_beta_math.py +
   test_phase_e_epsilon_epilogue.py — expect ALL green.
4. Commit as "fix(cute): Phase E.2 #2 — β-coop kernels use (1+γ) (all phases)".
5. Defer kernel.py:1921 to a separate F.2-or-later commit.

## What's currently uncommitted

Nothing. Working tree is clean post-Batch-A commit. No edits in progress.

## Recommendation

Option B (spec-driven scope). The plan's literal targets are an undercount
of the actual bug surface. F.1 will surface the rest as serving failures
otherwise. But this exceeds the strict letter of the Batch B task description,
so I'm holding for explicit user authorization before proceeding.
