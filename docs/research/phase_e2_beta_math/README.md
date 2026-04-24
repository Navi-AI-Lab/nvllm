# Phase E.2 — β Kernel Math Reference Diff

## What this is

Cross-reference tests at `tests/kernels/cute/test_phase_e2_beta_math.py`
that compare β-lite, β-coop Phase 0, and β-coop Phase 4 kernel outputs
against `Qwen3_5RMSNorm` — the model's actual RMSNorm semantics
(`x * (1 + γ)`, not raw `γ`).

The original test at `tests/kernels/cute/test_phase_e_epsilon_epilogue.py`
passed-against-wrong-reference because the reference harness at
`docs/research/2026-04-22-phase-e-repro.py` had the same raw-γ bug as
the kernel. Both shared the wrong math, so `torch.allclose` passed.

The discovery came from a fresh-eyes audit (spec-reviewer agent,
2026-04-24 session). See:
- `memory:project_phase_e_beta_math_bug`
- `memory:project_phase_e_phantom_speedup`
- Spec: `docs/superpowers/specs/2026-04-24-phase-f1-opaque-gate-refactor-design.md` (Phase E.2 section)

## Audit-driven scope expansion

The plan's Tasks 4-6 named 3 raw-γ sites (`phase_e_kernel.py:641`, the
two Phase 4 ε epilogues). A fresh audit during Batch B execution found
**6 additional sites** of the same bug:

- `phase_e_kernel.py:855, 1547` — `run_phase_01_only` debug kernel
- `phase_e_kernel.py:3281, 3952, 4648` — `run_beta_coop_full` PRODUCTION paths
- `kernel.py:1922` — standalone `DecodeKernel` Phase C post-attn rmsnorm

All are fixed in commit `c2a6d8766` (Phase E.2 #2). Also fixes two bad
refs in `test_phase_e_epsilon_epilogue.py` that mirrored the kernel bug.
Full write-up at `batch_b_audit_2026-04-24.md`.

## How to run

```bash
# From repo root, in .venv:
.venv/bin/python -m pytest tests/kernels/cute/test_phase_e2_beta_math.py -v
```

Three tests should pass:
- `test_beta_lite_epsilon_matches_qwen35_rmsnorm_forward_native` (β-lite ε)
- `test_beta_coop_phase0_matches_qwen35_rmsnorm_forward_native` (β-coop Phase 0)
- `test_beta_coop_phase4_epsilon_matches_qwen35_rmsnorm_forward_native` (β-coop Phase 4 ε)

These are part of the normal test suite and should remain green forever
— if they fail, β kernels have drifted from `Qwen3_5RMSNorm` semantics
and must be re-audited.

The companion file `tests/kernels/cute/test_phase_e_epsilon_epilogue.py`
contains 11 more tests covering Phase 0/1/2/3/4 individually plus
β-coop full vs β-lite equivalence. All 14 tests across both files must
stay green.

## What failure looks like

Assertion error citing `max_diff`, with a hint about which kernel line
likely regressed (each test names the suspected source line). Follow
the hint — don't blindly relax the tolerance or update the test's
expected value. If the diff is order-`|γ_max × normed_max|` (~1.0 to
~5.0 for typical inputs), the kernel almost certainly reverted to raw γ.

## Tolerance rationale

All three new tests use `rtol=3e-2, atol=5e-2` (BF16). The kernel uses
PTX `rsqrt.approx.f32`; the reference uses `torch.rsqrt`. These diverge
by ~2 FP32 ULPs → up to 1 BF16 ULP (2^-5 ≈ 0.031) after the `* (1+γ)`
multiply with random γ. Tolerance is tight enough to catch any
semantic regression (a raw-γ revert produces max_diff ≫ 1) but loose
enough to absorb the documented approx-rsqrt drift.
