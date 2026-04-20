# Phase D `cute.compile` + `Constexpr` refactor — audit artifacts

Evidence directory for the refactor described in
`docs/superpowers/specs/2026-04-20-phase-d-cute-compile-constexpr-design.md`.

## Gate 0 — Jupyter pre-flight (pre-build, fast)

```bash
.venv/bin/python docs/research/phase_d_cute_compile_audit/jupyter_preflight.py
```

Expected output after the refactor: `RESULT: ALL GATES PASSED`.

Runs 4 sub-gates against the Phase 3b small-dim test config
(`hidden=128, interm=128, tile_s=64, tile_k=32, slice_ctas=2`):

- **0a** — `Phase_D_MLP_Kernel` imports without DSL type errors
- **0b** — constructs + eager `cute.compile` succeeds
- **0c** — two distinct Constexpr tuples produce two distinct PTX files
  in `~/.cache/cutlass_dsl/` (thesis in miniature)
- **0d** — compiled kernel runs on small tensors, output finite

If any gate fails, iterate on `mlp_kernel.py` / `_backend.py` **outside
Docker** until it passes — this saves ~30-50 min per Docker rebuild
cycle.

## Gate 4 — PTX dumps (post-build, per-preset)

Four subdirectories, one per preset from `_tile_presets.py`:
`prefill-legacy/`, `decode-balanced/`, `decode-small/`,
`decode-narrow-grid/`. Each contains `kernel.ptx` and `kernel.mlir.ir`
extracted from `~/.cache/cutlass_dsl/` after a
`CUTE_MLP_KEEP_PTX=1 CUTE_MLP_TILE=<preset>` run. Expected: 4 distinct
MD5 hashes.

## Gate 6 — sibling-edit stability (post-ship thesis test)

`sibling_edit_stability/before_gsm8k.log` — post-refactor-pre-noop
GSM8K on default preset (expected 8/8).

`sibling_edit_stability/after_gsm8k.log` — post-noop-comment GSM8K on
default preset (expected also 8/8; proves cache key is source-stable).

## Pre-flight transcripts

`preflight_output/gate0_transcript.txt` — the final passing pre-flight
run captured before Docker build (gate 0 evidence for ship summary).
