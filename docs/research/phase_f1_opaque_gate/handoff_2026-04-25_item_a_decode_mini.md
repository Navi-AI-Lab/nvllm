---
title: Phase F.1 Item A — `decode-mini` tile preset wins, memory audit, registry dedup (2026-04-25)
date: 2026-04-25
predecessor: handoff_2026-04-24_session_end_v2.md
---

# Phase F.1 Item A — End-of-session handoff (2026-04-25)

Caps the 2026-04-25 session: Item A from `project_fusion_debug_plan`
landed on `fix/phase-d-mlp-decode-tile-preset` (now merged to `main`).

## TL;DR

| Finding | Status |
|---|---|
| MLP fusion tok/s under PIECEWISE | **1.90 → 8.45 tok/s (+345%)** with new `decode-mini` (64, 640, 8) tile preset |
| GSM8K-50 seed=42 gate | **49/50 (98.0%)** — single Q0 miss is a thinking-mode arithmetic slip |
| Item B math (β `(1+γ)` fix) | **Already shipped 2026-04-24** in `98551dba6` + `c2a6d8766` — the audit re-discovered this. Plan B-math commit is OBSOLETE. |
| β-coop ON gibberish | Root cause is NOT the math (math is right post-fix). Different bug — see `project_phase_e_phantom_speedup` revised. Item B reframed as "β residual diagnosis." |
| Tech debt | Removed duplicate `_TILE_PRESETS` registry from `mlp_kernel.py` (35 lines) — `_tile_presets.py` is now the single source of truth |
| Memory audit | 14 memory entries refreshed against current code — multiple stale file paths, line numbers, and status claims fixed |

## Headline perf table

PIECEWISE 256-token e2e on `ig1/Qwen3.5-27B-NVFP4`, MLP-only fusion
(`CUTE_MLP_FUSION=1`, all other CUTE_*_FUSION=0):

| Preset             | (tile_s, tile_k, slice_ctas) | wall    | tok/s | Δ vs legacy |
|--------------------|------------------------------|---------|-------|-------------|
| prefill-legacy     | (256, 640,  8)               | 134.74s | 1.90  | base        |
| decode-balanced    | (128, 640, 16)               |  91.53s | 2.80  | +47%        |
| decode-small       | ( 64, 640, 32)               |  96.94s | 2.64  | +39%        |
| decode-narrow-grid | (256,1280,  8)               | 145.78s | 1.76  | −7%         |
| decode-mini ⭐      | ( 64, 640,  8)               |  30.30s | 8.45  | **+345%**   |
| decode-32          | ( 32, 640,  4)               |  30.36s | 8.43  | +344%       |
| decode-micro       | ( 16, 640,  2)               |  30.26s | 8.46  | +345%       |

Pattern: holding total CTAs at 2176 (decode-balanced winner level)
while shrinking tile_s and slice_ctas in lockstep is what unlocks
the win. decode-mini, decode-32, and decode-micro all tie at ~8.45
tok/s — the bottleneck has shifted off the MLP kernel entirely.
decode-mini selected as the registry's new default (smallest
deviation from `decode-balanced`).

## Commits shipped

```
33f694703  refactor(cute): remove duplicate tile-preset registry in mlp_kernel.py
dd72fbb85  perf(cute): Phase D MLP decode-mini tile preset — +345% PIECEWISE tok/s
2a39c2ec2  docs: Phase F.1 — bisection baseline + CUTE_DEBUG_TIMING instrumentation
```

## Evidence

```
docs/research/phase_f1_opaque_gate/run_logs/
  tile_sweep_piecewise_20260425_075336/   ← 4-preset PIECEWISE sweep
  tile_sweep_micro_20260425_085056/       ← 3 narrower presets (constant CTA)
  gsm8k_decode_mini_20260425_090855/      ← GSM8K-50 gate (49/50 = 98.0%)
```

Each run dir contains `run.sh` (or `sweep.sh`), `completion.json` /
`eval_result.json`, and `findings.md`. The first sweep also saved
`timing_lines.txt` from yesterday's CUTE_DEBUG_TIMING instrumentation —
ultimately not useful here because PIECEWISE drops side-effecting
Python instrumentation (a discovery worth its own line in the lessons
learned column).

## What this does and does not unblock

**Unblocks:**
- Anyone opt-ing into `CUTE_MLP_FUSION=1` now gets ~4.4× faster MLP path.
- Item C (ATTN tile audit) can use the same probe-shape methodology:
  PIECEWISE 256-token e2e wall-clock on the shipped image, no rebuild.

**Does not unblock — production serve config still uses
`CUTE_MLP_FUSION=0`** because:
- Item B (β residual diagnosis) outstanding — math is fixed but β-coop
  ON still produces gibberish. Cause is somewhere in Phase 1/2/3, sync,
  or op-glue. See `project_phase_e_phantom_speedup` for the working
  hypothesis (incomplete uber-kernel migration: β-coop reads regular
  path's output, double-counting attn_out).
- Item C (ATTN fusion ~5× cuBLAS) outstanding.

## Memory audit (2026-04-25)

Background audit-agent flagged 19 stale claims across the auto-memory.
Verified against live code; high-impact updates landed in:

- `project_fusion_debug_plan` — Order revised; Item B-math marked obsolete
  with commit hashes; Item B replaced by "β residual diagnosis"
- `project_phase_e_beta_math_bug` — REWRITTEN; both `(1+γ)` fixes shipped
  2026-04-24 morning; remaining gibberish has different root cause
- `project_fused_path_perf_collapse` — same correction propagated
- `project_phase_d_inflight` — D3a abandonment narrative revised:
  source-hash sensitivity hypothesis was falsified 2026-04-20; real
  cause was atomicAdd non-determinism (fixed in `4af48a62c`)
- `project_strategy_priorities` — Qwen3.5-27B layer count corrected:
  48 linear + 16 full = **64 layers** (was 24+8=32)
- `project_phase_e_d25_brainstorm`, `project_phase_e_phantom_speedup`,
  `project_phase_e_shipped` — `qwen3_5.py:473` dead-branched-gate
  references softened (replaced by `cute_phase_e_dispatch` opaque op
  in F.1 commit `9f39b86ef`); `_acquire_fence` now exists in code
- 7 misc memories — line-number drifts replaced with grep-pattern
  locators per `feedback_pinned_code_refs`
- 3 memories with stale `scripts/run_qwen35_27b.sh` references —
  current scripts under `scripts/local/`

## Lessons logged

1. **PIECEWISE drops side-effecting Python instrumentation** — yesterday's
   `CUTE_DEBUG_TIMING` checkpoints fired in `--enforce-eager` but went
   silent in PIECEWISE. End-to-end tok/s is the only graph-safe metric.
2. **Bind-mount + `__pycache__` interaction**: Docker bind mount of the
   source overrides the file content but Python's `.pyc` cache check
   uses the source's mtime; if the in-image .pyc is newer than the
   bind-mounted source's mtime, the .pyc shadows. Source-mtime > .pyc-
   mtime is the safe state. (We didn't actually hit this in the new
   sweep, but it explains why early debug runs misled.)
3. **Audit before designing**: the planning memory had Item A pointed
   at `vllm/nvllm/cute/_tile_presets.py:127` — that path doesn't exist.
   The actual location is `vllm/v1/attention/backends/cute_paged/
   _tile_presets.py:101`. Per `feedback_pinned_code_refs`, all
   newly-written memory entries now use grep-pattern locators.

## Next session

`project_fusion_debug_plan` order is now:

1. ✅ **A — decode-mini tile preset** (this session)
2. ⏳ **B — β residual diagnosis**: write a reference-diff harness
   against `Qwen3_5RMSNorm.forward_native` capturing β-coop kernel
   partial outputs (Phase 0 → 1 → 2 → 3 → 4) until first divergence.
3. ⏳ **C — ATTN fusion tile audit**: same probe-shape methodology as
   Item A but for the W_O+RMSNorm baked path in `kernel.py`.

C is independent and can run in parallel with B.
