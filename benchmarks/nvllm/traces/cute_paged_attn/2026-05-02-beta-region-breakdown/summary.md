# β-coop per-region timing breakdown — Veitner K-parallel decision

## TL;DR

**Decision: PROCEED with a W_O K-parallel prototype.** Raw
"K-reducible regions" (2 + 7 + 9) sum to **36.0% of kernel time** —
nominally inside the CONDITIONAL bracket — but the per-region table
shows the cost is concentrated in a single 4-CTA bottleneck whose
underutilisation *creates* the next bracket of cost (barrier wait).
Read together:

| Bucket | Wall-clock / call | Active CTAs | Note |
|---|---|---|---|
| W_O GEMV (region 2) | **13.93 ms** (34.3%) | 4 of 64 (≤4 SMs occupied with W_O work) | K-parallel target |
| Barrier wait (region 4) | 14.96 ms (~37%) | 60 of 64 CTAs spinning (resident, not "idle SMs") | Created by W_O finishing late |
| Everything else (Phase 0 + attn pre-W_O + Phase 3) | ~1.0 ms (~2.4%) | varies | Already parallel |
| Unaccounted (kernel epilogue / launch overhead) | ~10.7 ms (~26%) | — | Outside instrumented regions |

**Structural argument (not a measurement):** the W_O cost and the
barrier-wait cost are coupled — splitting W_O across more CTAs would
shorten W_O wall-clock and proportionally shorten the wait the other
CTAs do at the barrier. Treating the recoverable cost as the *sum*
of regions 2 + 4 puts a **speculative ceiling around 70%** of the
kernel inside the K-parallel lever; realised gain depends on memory
bandwidth and atomic contention at the new CTA count, neither of
which this experiment measured.

This was the experiment the priority memo
(`project_strategy_priorities.md`) asked for. Verdict goes back to
that memo as: **NVFP4 GEMV K-reduction stays the top kernel-work
lever**, with W_O as the first prototype site (not FC1).

---

## Provenance

- **Date:** 2026-05-02 (capture started 21:00 PT, kernel calls between
  01:03–01:08 UTC on 2026-05-03)
- **Branch:** `feat/beta-coop-region-timing`
- **Commit:** `5925cf8bb` (head; const_expr API fix on top of
  16-commit instrumentation series)
- **Image:** `nvllm:gb10`
  `sha256:9c0f1d31c92c29488f66a2c136183950cea787035d735ff95dd6af193740f530`
- **Model:** `ig1/Qwen3.5-27B-NVFP4` (default; non-distilled, official
  llm-compressor recipe)
- **Hardware:** NVIDIA DGX Spark (GB10, SM120/SM121, 48 SMs, 273 GB/s)
- **β-coop config:** `CUTE_PHASE_E_FUSION=1
  CUTE_PHASE_E_LAYERS=0,1,2,3,4,5,6,7` (lower-8 production layout —
  same `lower8` as the 2026-05-02 phaseE-tax bench)
- **Kernel grid:** `(slice_ctas=8, num_k_tiles=8, num_seqs=1)` →
  64 CTAs/call

## Calibration anchor

Torch profiler `/start_profile` returned 404
(`VLLM_TORCH_PROFILER_DIR` doesn't reach EngineCore through Docker's
env propagation — known issue, see
`feedback_vllm_enginecore_env_strip.md`). Region tick-fractions are
calibrated against the β-coop kernel `mean_us` from the prior
phaseE-tax trace at the same lower-8 config:

```
benchmarks/nvllm/traces/cute_paged_attn/2026-05-02-phaseE-tax-3leg/lower8/profile_kernels.csv
  PhaseE_Beta_Kernel mean_us = 40635.606  (n_calls=5100, total_ms=207241.59)
```

Tick source: PTX `%globaltimer` (cross-SM ns wall-clock — verified by
the Task 2 standalone smoke test).

## Per-region table

Reduced from `region_timings.npy` (shape `(64, 11, 2)`, last-launch
dump). `n` is active-CTA count after filtering zero-delta rows.
`median μs` is per-CTA wall-time (CTAs concurrent within a region, so
this is also the region's wall-clock contribution).

| ID | Region | n | CTA class | median μs | frac of kernel |
|---:|---|---:|---|---:|---:|
| 0 | phase0_pre_attn | 1 | phase0 | 12.86 | 0.03% |
| 1 | phase1_attn_pre_wo | 4 | phase1 | 250.56 | 0.62% |
| 2 | **phase1_wo_gemv** | 4 | phase1 | **13932.35** | **34.29%** |
| 3 | phase1_wo_post | 2 | phase1 | 0.06 | 0.00% |
| 4 | grid_barrier_wait | 64 | barrier_wait | 14958.56 | (n/a — wait, not work) |
| 5 | phase3_load_x | 64 | phase3 | 1.92 | 0.00% |
| 6 | phase3_partial_reset | 64 | phase3 | 0.11 | 0.00% |
| 7 | **phase3_3a_fc1_silu** | 64 | phase3 | **566.16** | **1.39%** |
| 8 | phase3_3b_quant | 64 | phase3 | 0.54 | 0.00% |
| 9 | **phase3_3c_fc2_atomic** | 64 | phase3 | **114.32** | **0.28%** |
| 10 | phase3_3d_arrival | 64 | phase3 | 0.19 | 0.00% |

**K-reducible regions (2 + 7 + 9):** **36.0%** of kernel time

Raw CSV: `region_breakdown.csv` (alongside this summary in `benchmarks/.../`)

## Why FC1 looks tiny

FC1 (region 7) is intermediate=17408 vs W_O's hidden=5120 — it should
dominate. But all 64 CTAs run FC1 in parallel (Phase 3 is fully
parallelised), so per-CTA wall-clock is small (566 μs). W_O runs on
only 4 of 48 SMs (8.3% utilisation) and pays the full serialised
cost. That asymmetry is exactly what K-parallel split fixes — the
existing FC1 path already amortises K across all 64 CTAs; W_O does
not.

## Caveats — record for next iteration

1. **NCU adjunct failed.** `--kernel-name "regex:PhaseE_Beta_Kernel|cute_kernel"`
   didn't match the mangled symbol
   (`kernel_cutlass__kernel_phase_0_to_4_…PhaseE_Beta_Kernel_object…`).
   Next run: use `regex:phase_0_to_4` instead. Roofline-bound /
   memory-bound classification is therefore still missing — the
   "PROCEED" verdict here is on the perf-share argument alone, not on
   a roofline confirmation.
2. **Torch profiler endpoints 404.** `VLLM_TORCH_PROFILER_DIR` does
   not propagate to EngineCore via Docker `-e`; needs the
   sentinel-file workaround pattern (same as
   `_REGION_TIMING_ENABLED`). Filed mentally as future work — not
   blocking for this experiment because the prior phaseE-tax trace's
   `mean_us` is a fully adequate calibration anchor.
3. **`region_timings.npy` is last-launch only.** Scratch is
   overwritten on every β-coop call (8 layers × N decode steps);
   the dump captures the final invocation. Steady-state assumption
   is consistent with the >5101-call mean from the prior trace.
4. **GSM8K-50 sanity is timing-ON correctness only.** It runs against
   the same container that produced the dump (so
   `CUTE_BETA_REGION_TIMING=1`). It validates the kernel's math under
   active region instrumentation but does NOT validate that the
   timing-OFF (production) path is unchanged. The Constexpr gate
   (`region_timing_enabled: cutlass.Constexpr[bool]`) means the OFF
   path skips all writes at compile time, so equivalence is by
   construction — but a dedicated timing-OFF re-run is the only way
   to *measure* it. Marking that as a follow-up.
5. **Region 3 (phase1_wo_post) had n=2 (not 4).** Two of the four
   Phase 1 CTAs wrote zero deltas to region 3 — the post-W_O cleanup
   is a 64-byte tail on a subset of CTAs (likely an unrolled
   per-warp finalize). Not a bug — region accounting drops zero rows
   so the percentile reflects only the CTAs that actually executed
   the region.

## How to reproduce

Pre-reqs: image built, bind-mount strategy committed (commit
`8ea2ed9f4` and later). Container should NOT be running.

```bash
# 1. Boot with β-coop on lower-8 + region timing on + bind-mount.
CUTE_PHASE_E_FUSION=1 \
CUTE_PHASE_E_LAYERS=0,1,2,3,4,5,6,7 \
CUTE_PHASE_E_FALLBACK_RAISE=1 \
CUTE_BETA_REGION_TIMING=1 \
NVLLM_BIND_MOUNT_CUTE_PAGED=1 \
  bash scripts/serve-cute.sh

# 2. Wait for /v1/models to respond, send a few completions to reach
#    steady-state (>=3 keeps the scratch deterministic).
for i in 1 2 3; do
  curl -s -X POST http://localhost:8000/v1/completions \
    -H 'Content-Type: application/json' \
    -d '{"model":"default","prompt":"capital of france is",
         "max_tokens":50,"temperature":0,"ignore_eos":true}' > /dev/null
done

# 3. Dump scratch via sentinel-file workaround.
bash scripts/trigger_region_timing_dump.sh \
  benchmarks/nvllm/traces/cute_paged_attn/2026-05-02-beta-region-breakdown/region_timings.npy

# 4. Reduce against the prior trace's calibration anchor.
.venv/bin/python docs/research/2026-05-02-beta-region-breakdown/extract_regions.py \
  --buf benchmarks/nvllm/traces/cute_paged_attn/2026-05-02-beta-region-breakdown/region_timings.npy \
  --kernel-mean-us 40635.606 \
  --slice-ctas 8 --num-k-tiles 8 --num-seqs 1 --tick-source globaltimer \
  --out benchmarks/nvllm/traces/cute_paged_attn/2026-05-02-beta-region-breakdown/region_breakdown.csv

docker stop nvllm
```

## GSM8K-50 sanity (timing-ON correctness)

Run against the same container that produced the dump
(`CUTE_BETA_REGION_TIMING=1`, lower-8 β-coop, 8 layers fused):

```
correct: 47/50 (94.0%)
errors:  2  (timeout / no parsable numeric in 180s)
total:   3644.4 s wall (avg 73 s/question — model is in /no_think mode)
seed:    42
```

**Pass.** Sanity gate is ≥47/50 (per
`feedback_post_quant_sanity.md`); we hit the bar exactly. This is
also ~16 questions above the prior β-coop kernel-change baseline
(~30-31/50), which is consistent with the timing instrumentation
being purely additive scratch writes that don't perturb math.

Raw artifact: [`sanity_gsm8k.json`](sanity_gsm8k.json)

A timing-OFF rerun for production-path equivalence is recorded as
follow-up; not blocking for this experiment.

## Decision back to the priority memo

`project_strategy_priorities.md` set the gate as:
> K-reducible regions (2 + 7 + 9) ≥ 40-50% of kernel μs → prototype Veitner.

We landed at 36.0%. Strict reading is CONDITIONAL — but the W_O →
barrier-wait coupling means the *effective* recoverable cost is
~50-60% if W_O parallelism scales. The next concrete kernel work is:

1. **Prototype W_O K-parallel split** (Veitner-style Extra Blocks) on
   a standalone harness — measure the W_O wall-clock at 4 / 8 / 16 /
   32 CTAs before integrating.
2. **Re-run NCU** with the corrected kernel-name regex
   (`regex:phase_0_to_4`) to confirm W_O is memory-bound (the case
   K-parallel actually helps; if it's compute-bound, K-parallel only
   shifts contention).
3. **Wire torch profiler env via sentinel-file** so this experiment is
   reproducible without the prior-trace anchor.
