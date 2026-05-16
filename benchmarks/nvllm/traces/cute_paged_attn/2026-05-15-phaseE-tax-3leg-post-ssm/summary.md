# Phase-E-tax screening — 3-leg torch profiler — post-SSM-zero-on-realloc

## Verdict

**Phase 4 stays dead. Verdict reproduced within run-to-run noise (~1–2%).**

The SSM zero-on-realloc patch (PR #13, 2026-05-15) did NOT change the kernel
cost model that resolved against Phase 4 on 2026-05-02. The patch fires at
request-realloc boundaries via `torch.index_fill_` outside the decode hot
path; per-kernel μs and per-token aggregates all reproduce within ~1–2% of
the 2026-05-02 prior. `lower8` still wins decisively over `all-beta`; the
β kernel is no cheaper, so adding it to more layers still costs more than
the DecodeKernel call it would replace.

The `feedback_phase4_dead` memory needs no update. The path to re-opening
Phase 4 remains the same: make the β kernel cheaper first (NVFP4 GEMV
K-parallel reduction), then revisit fusion atop a cheaper β.

## Provenance

| Field | Value |
| :---- | :---- |
| Date | 2026-05-15 (started) / 2026-05-16 (completed) |
| Commit | `67619835b` (main; PR #13 squash-merge of SSM zero-on-realloc series) |
| Image ID | `sha256:b7ede5c875760fe253b356190ebac3e88bb4fddb215da31588818ecffa7cdd28` (`nvllm:gb10-ssm`) |
| Image built | 2026-05-15T21:37:25-04:00 (fresh build off `main` HEAD with SSM patch baked in; verified `MambaBlockZeroer` and `attn_groups_list = list(...)` materialization both present in image) |
| Model | `ig1/Qwen3.5-27B-NVFP4` |
| Hardware | NVIDIA DGX Spark (GB10, SM120/SM121), 128 GB unified LPDDR5x |
| Host driver | 590.48.01 |
| Runner | `docs/research/2026-05-02-phaseE-tax-3leg/run_3leg.sh` (OUT_ROOT made env-overridable in this commit; otherwise unchanged from 2026-05-02) |
| Suite wall | 2026-05-15 21:37 → 2026-05-16 04:53 EDT (~7h 16min) |
| Suite exit | `BENCH_RC=0` (`=== ok:true — all attempted legs completed`) |

**What changed from 2026-05-02:** the SSM zero-on-realloc patch is now baked
into the image. No other code changes.

## Leg configuration

Identical to 2026-05-02 (no env changes). For completeness:

| Leg | `CUTE_PHASE_E_FUSION` | `CUTE_PHASE_E_LAYERS` | β-coop layers | Profile timed | GSM8K n |
| :--- | :--- | :--- | :--- | ---: | ---: |
| `lower8` | `1` | `0..7` | 3, 7 (2L) | 10 | 50 |
| `phaseE-off` | `0` | `0..7` (irrelevant) | none | 10 | 50 |
| `all-beta` | `1` | `0..15` | 3, 7, 11, 15 (4L) | 4 | 50 |

All legs: `--max-num-seqs 1`, `--max-model-len 16384`, `--kv-cache-dtype fp8_e4m3`,
`--attention-backend CUTE_PAGED`, `--gpu-memory-utilization 0.65`,
disk-cached cute compile (`B12X_CUTE_COMPILE_DISK_CACHE=1`).

## Kernel duration A/B vs 2026-05-02

### Custom CuTe kernels

Per-call mean μs is the most-direct kernel-cost metric. All deltas
within ~1–2% (run-to-run noise band for this hardware).

| Leg | Kernel | n_calls | 2026-05-02 mean_us | 2026-05-15 mean_us | Δ% |
| :--- | :--- | ---: | ---: | ---: | ---: |
| `lower8` | DecodeKernel | 35700 | 17088.491 | 17128.171 | +0.23% |
| `lower8` | PhaseE_Beta_Kernel | 5100 | 40635.606 | 41311.311 | +1.66% |
| `phaseE-off` | DecodeKernel | 40800 | 17040.088 | 17268.251 | +1.34% |
| `phaseE-off` | Phase_D_MLP_Kernel | 40800 | 23931.397 | 24150.127 | +0.91% |
| `all-beta` | DecodeKernel | 12240 | 17106.305 | 17135.967 | +0.17% |
| `all-beta` | PhaseE_Beta_Kernel | 4080 | 40829.264 | 41461.362 | +1.55% |

`n_calls` is identical across runs (same timed-iter count + same model layer
counts), so the A/B is direct.

### Top GEMM/GEMV (per leg, by total_ms)

| Leg | Kernel | n_calls | 2026-05-02 mean_us | 2026-05-15 mean_us | Δ% |
| :--- | :--- | ---: | ---: | ---: | ---: |
| `lower8` | gemvx (bf16) | 369760 | 391.545 | 393.516 | +0.50% |
| `lower8` | NVFP4 GEMM (variant A) | 358280 | 315.065 | 318.528 | +1.10% |
| `phaseE-off` | gemvx (bf16) | 369760 | 393.817 | 395.209 | +0.35% |
| `phaseE-off` | NVFP4 GEMM (variant A) | 286880 | 313.318 | 318.545 | +1.67% |

NVFP4 GEMM is the workhorse weight×activation kernel; its +1% drift is
consistent with the overall measurement-noise band.

## Aggregate per-token kernel time (decode + MLP + β)

Calculated as `mean_us × layers_per_token / 1000` for the dominant kernels:

| Leg | 2026-05-02 ms/tok | 2026-05-15 ms/tok | Δ% |
| :--- | ---: | ---: | ---: |
| `lower8`     (14 decode × 17.1 + 2 β × 41.3)  | 320 | 322.4 | +0.8% |
| `phaseE-off` (16 decode × 17.3 + 16 mlp × 24.2) | 656 | 662.7 | +1.0% |
| `all-beta`   (12 decode × 17.1 + 4 β × 41.5)   | 369 | 371.5 | +0.7% |

**Order:** `lower8` (322 ms/tok) ≪ `all-beta` (372 ms/tok) ≪ `phaseE-off` (663 ms/tok).
Identical ordering to 2026-05-02; relative gaps preserved.

## GSM8K gate

| Leg | 2026-05-15 correct/50 | errors | floor (≥30) | ship (≥47) | 2026-05-02 |
| :--- | ---: | ---: | :---: | :---: | :---: |
| `lower8` | 46 | 1 (Q45 timeout) | PASS | FAIL by 1 | 47/50 |
| `phaseE-off` | 4 | 46 (timeouts) | FAIL | FAIL | 2/50 |
| `all-beta` | 47 | 1 (timeout) | PASS | PASS | 47/50 |

The ±1–2 question deltas are within single-run noise. Notably:

- `lower8` Q45 timed out at 180.1s (read-deadline) with `completion_tokens=0`,
  `output_len=90` chars before the timeout fired — same long-output question
  that fell just-inside the 180s window on 2026-05-02. Not a numerics
  regression; deterministic-borderline answer length × timeout-band variance.
- `phaseE-off` reproduces the 2026-05-02 result of "most prompts timeout
  because the legacy DecodeKernel + Phase_D_MLP fallback is ~2× slower than
  β-coop". Numerics not broken — the two questions that finished returned
  correct answers (4/50 here, 2/50 prior).

## Conclusion vs `memory:feedback_phase4_dead`

**No update needed.** Verdict reproduced:

> β kernel cost per added layer ≈ ~40.7 ms; replacing a ~17 ms decode call
> with a ~40.7 ms β call is a net regression.

Post-SSM numbers are 17.1 μs DecodeKernel and 41.3–41.5 μs PhaseE_Beta_Kernel
— same shape, same gate. Phase 4 stays dead. The path to re-opening it
remains: cheaper β via NVFP4 GEMV K-parallel reduction first.

## How to reproduce

```bash
# Fresh image off main HEAD with SSM patch baked in.
docker build -f docker/Dockerfile.gb10 -t nvllm:gb10-ssm .

# Run the 3-leg sweep against a fresh output dir.
OUT_ROOT=$PWD/benchmarks/nvllm/traces/cute_paged_attn/2026-05-15-phaseE-tax-3leg-post-ssm \
NVLLM_IMAGE=nvllm:gb10-ssm \
  bash docs/research/2026-05-02-phaseE-tax-3leg/run_3leg.sh
```

Wall: ~7–8 h (3 legs × 2 boots × cold-load + bench, plus the phaseE-off
gsm8k phase grinds for ~2.3 h at 180s/timeout × 46 timeouts). Per-leg detail
in the runner script header.
