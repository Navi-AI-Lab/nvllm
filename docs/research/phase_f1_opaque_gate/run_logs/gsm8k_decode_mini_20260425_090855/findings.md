# Phase F.1 Item A — Phase D MLP `decode-mini` tile preset

## Headline

Flipping the default `Phase_D_MLP_Kernel` tile preset from `prefill-legacy`
(256, 640, 8) → `decode-mini` (64, 640, 8) recovers **+345% PIECEWISE
end-to-end tok/s** on Qwen3.5-27B-NVFP4 with MLP fusion alone, and
passes the GSM8K-50 seed=42 correctness gate at **49/50 (98.0%)**.

## Method

Three sequential PIECEWISE 256-token e2e measurements on `nvllm:gb10`,
`ig1/Qwen3.5-27B-NVFP4`, `--max-num-seqs 1`, MLP-only fusion config
(`CUTE_MLP_FUSION=1` everything else `=0`), no `--enforce-eager`. Tile
preset selected at server start via `CUTE_MLP_TILE=<name>` env var; new
presets injected via bind-mount of `_tile_presets.py` (no Docker rebuild).

## Sweep 1 — existing registry (4 presets)

Evidence: `../tile_sweep_piecewise_20260425_075336/`

| Preset             | (tile_s, tile_k, slice_ctas) | total CTAs | wall    | tok/s | Δ vs legacy |
|--------------------|------------------------------|------------|---------|-------|-------------|
| prefill-legacy     | (256, 640,  8)               | 544        | 134.74s | 1.90  | (baseline)  |
| **decode-balanced**| (128, 640, 16)               | 2176       | 91.53s  | 2.80  | **+47%**    |
| decode-small       | ( 64, 640, 32)               | 8704       | 96.94s  | 2.64  | +39%        |
| decode-narrow-grid | (256,1280,  8)               | 544        | 145.78s | 1.76  | −7%         |

`decode-balanced` wins. `decode-small` over-saturates at ~5.7 waves
(GB10 ≈ 1500 max resident CTAs). `decode-narrow-grid` regresses —
halving `num_k_tiles` via 2× `tile_k` is a net loss.

## Sweep 2 — constant-CTA-total micro presets

Evidence: `../tile_sweep_micro_20260425_085056/`

Hypothesis: hold total CTAs at 2176 (= balanced winner) but shrink
`tile_s` + `slice_ctas` proportionally. Smaller tile_s reduces
shared-mem and register pressure per CTA → potentially better
occupancy at the same parallelism budget.

| Preset       | (tile_s, tile_k, slice_ctas) | total CTAs | wall   | tok/s | Δ vs balanced |
|--------------|------------------------------|------------|--------|-------|---------------|
| **decode-mini**  | (64, 640, 8)             | 2176       | 30.30s | **8.45** | **+202%**  |
| decode-32    | (32, 640, 4)                 | 2176       | 30.36s | 8.43  | +201%         |
| decode-micro | (16, 640, 2)                 | 2176       | 30.26s | 8.46  | +202%         |

All three statistically tied at ~8.45 tok/s. The MLP kernel is no
longer the bottleneck at this configuration.

`decode-mini` selected as the new default — slice_ctas=8 is the
smallest deviation from the validated `decode-balanced` config and
least likely to surface edge cases.

## Sweep 3 — GSM8K-50 correctness gate

Evidence: `eval_result.json`

```
N:        50
Correct:  49
Accuracy: 98.0%  (gate: ≥ 90%)
Wall:     1048s (17m 28s, ~21s/q avg)
```

Single miss is Q0 — model arrived at the right approach but performed
`$430+$750 = $1180` and `$300+$700 = $1000`, summing to $2180 vs gold
$2280. Pure thinking-mode arithmetic slip inside `<think>`, not a
kernel-level math break.

## What changed

`vllm/v1/attention/backends/cute_paged/_tile_presets.py`:
- New entry `"decode-mini": (64, 640, 8)`
- `_DEFAULT_PRESET_NAME` flipped from `"prefill-legacy"` → `"decode-mini"`

The two redundant micro candidates (`decode-32`, `decode-micro`) were
not committed — they tie with `decode-mini` and add no marginal value;
the audit trail lives in this `findings.md` and the sweep run logs.

## Reproduction

```bash
# Sweep 1 (4-preset)
docs/research/phase_f1_opaque_gate/run_logs/tile_sweep_piecewise_20260425_075336/sweep.sh

# Sweep 2 (micro)
docs/research/phase_f1_opaque_gate/run_logs/tile_sweep_micro_20260425_085056/sweep.sh

# Sweep 3 (GSM8K gate)
docs/research/phase_f1_opaque_gate/run_logs/gsm8k_decode_mini_20260425_090855/run.sh
```

## Out of scope (next session)

This commit lands `decode-mini` as the new MLP-fusion default but does
not flip production serve config to `CUTE_MLP_FUSION=1`. Per
`project_fused_path_perf_collapse`, the remaining blockers are Item C
(ATTN fusion still ~5× cuBLAS) and Item B (β-coop residual diagnosis —
math is fixed but β-coop ON still produces gibberish, root cause TBD).
