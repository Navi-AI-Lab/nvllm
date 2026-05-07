# W_O K-parallel Production Soak — Qwen3.5-27B-NVFP4

**Date:** 2026-05-04 (primary) → 2026-05-07 (supplementary + writeup)
**Run commit:** `5b8fc399f2c981da86c6ae4455bb4f9adc347da8`
**Branch:** `evidence/wo-split-prod-soak`
**Image:** `nvllm:gb10` (`sha256:9c0f1d31c92c29488f66a2c136183950cea787035d735ff95dd6af193740f530`)
**Hardware:** DGX Spark (GB10, SM120/SM121 — 48 SMs, 273 GB/s peak DRAM, 100 KB SMEM/SM)
**Model:** `ig1/Qwen3.5-27B-NVFP4` (NVFP4 weights, FP8 KV)
**Serving config:** PIECEWISE cudagraphs · CUTE_PAGED attention · `max_model_len=65536` · `max_num_seqs=4` · `gpu_memory_utilization=0.70` · FlashInfer autotune disabled
**Sweep:** `CUTE_WO_SPLIT ∈ {1, 2, 4, 8}` (production-evidenced set)
**Scope:** end-to-end serving validation that K-parallel W_O preserves quality and improves wall/TPOT under realistic prompt mix. Kernel-level μs claims live in the standalone harness (cited below).

## Bottom line

- **wo8 remains keep opt-in, not default.** All non-baseline arms passed parity gates; wall improved by 2.4% → 3.4%, and p95 TPOT improved by 17.7 → 25.5 ms. The gains are real but modest at production batch sizes, so PIECEWISE + wo_split=1 stays the default until larger ROI motivates a flag flip.
- **Kernel-level W_O claim (controlled harness):** 13754 μs → 1639 μs, **8.39× speedup**, NCU classification flips from latency-limited to memory-bound (8.06% → 55.95% of peak DRAM bandwidth), bit-exact correctness across all variants. ([harness summary](../../cute_paged_attn/2026-05-03-w-o-k-parallel-harness/summary.md), commit `46ad9bbc`)
- **Production decoder cross-check (wo8 serving npy):** `phase1_wo_gemv` runs at 32 active CTAs (= 8× the 4-CTA baseline), median 2359.8 μs per kernel call, p99 2468.9 μs — confirming the optimized path is active in real serving and operating in the post-saturation regime predicted by the harness.
- **Quality:** GSM8K parity within 1/50 across all four arms (wo1=48, wo2=47, wo4=48, wo8=47); coherence checks pass on longdecode replays. Quality gate: cleared.

## End-to-end serving evidence

5 ShareGPT replays + 5 longdecode replays + 2-concurrent probe + GSM8K-50 per arm. Bounds: `sharegpt_max_tokens=128`, `longdecode_max_tokens=2048`, `seed=42`.

| arm | gsm8k | replays | wall mean (s) | wall stddev | tpot p50 (ms) | tpot p95 (ms) | tpot p99 (ms) | longdecode p95 |
|-----|-------|---------|---------------|-------------|---------------|---------------|---------------|----------------|
| wo1 | 48/50 | 5 | 8104.75 | 6.74 | 467.98 | 510.73 | 530.38 | 518.54 |
| wo2 | 47/50 | 5 | 7910.47 | 3.39 | 450.43 | 493.07 | 511.96 | 500.75 |
| wo4 | 48/50 | 5 | 7829.37 | 4.22 | 443.63 | 486.66 | 506.36 | 494.26 |
| wo8 | 47/50 | 5 | 7833.98 | 1.98 | 441.94 | 485.21 | 504.69 | 491.69 |

### Pairwise vs baseline (wo1)

`wall improvement` is `(wo1 - arm) / wo1`; positive = arm is faster.

| arm | wall improvement | tpot p95 Δ (ms) | gsm8k Δ |
|-----|------------------|-----------------|---------|
| wo2 | +2.4% | -17.66 | -1 |
| wo4 | +3.4% | -24.07 |  0 |
| wo8 | +3.3% | -25.52 | -1 |

Wall-time gains plateau after wo4. p95 TPOT continues to tighten through wo8 (smaller spread, lower ceiling) and wo8's wall-time stddev is the lowest of all arms (1.98 s vs 6.74 s baseline) — production wall is more stable, even when the mean is statistically tied with wo4.

### Verdicts

- **wo1** — baseline
- **wo2** — keep opt-in (wall +2.4%, tpot p95 -17.66 ms; baseline noise stddev 0.95 ms)
- **wo4** — keep opt-in (wall +3.4%, tpot p95 -24.07 ms)
- **wo8** — keep opt-in (wall +3.3%, tpot p95 -25.52 ms, lowest wall stddev)

`wo_split=1` remains the production default. All three optimized arms are exposed via `CUTE_WO_SPLIT` for opt-in.

## Kernel-level W_O claim (controlled harness)

The kernel-level speedup claim is grounded in [`2026-05-03-w-o-k-parallel-harness`](../../cute_paged_attn/2026-05-03-w-o-k-parallel-harness/summary.md), not in this soak's serving traces. The harness exercises the W_O kernel in isolation with bit-exact correctness gates against per-variant FP32 references, and produces the NCU classification needed to defend the optimization model.

| wo_split | Total W_O CTAs | Mean elapsed (μs) | Effective GB/s (logical) | Speedup vs wo_split=1 |
|---:|---:|---:|---:|---:|
| 1 | 4 | 13754 | 1.49 | 1.00× |
| 2 | 8 | 5176 | 4.47 | 2.66× |
| 4 | 16 | 2693 | 10.6 | 5.11× |
| 8 | 32 | **1639** | **24.0** | **8.39×** |

NCU classification at the endpoints (per `2026-05-03-w-o-k-parallel-harness`):

| Section | wo_split=1 (4 W_O CTAs) | wo_split=8 (32 W_O CTAs) |
|---|---:|---:|
| Max DRAM Bandwidth (% peak) | 8.06% | **55.95%** |
| No-eligible cycles | 98.33% | 91.86% |
| Eligible Warps/Scheduler | 0.02 | 0.08 |
| Verdict | latency-limited | memory-bound |

The classification flip (latency-limited → memory-bound) is the structural argument for the optimization being real and the slope being durable, not just numerical luck on a single shape. Achieved occupancy is identical between endpoints (8.33%) — the speedup comes from putting more independent in-flight work on the memory subsystem, not from changing the launch geometry. The raw NCU reports are tracked at [`../../cute_paged_attn/2026-05-03-w-o-k-parallel-harness/ncu/wo_split_1/kernel.ncu-rep`](../../cute_paged_attn/2026-05-03-w-o-k-parallel-harness/ncu/wo_split_1/kernel.ncu-rep) and [`../../cute_paged_attn/2026-05-03-w-o-k-parallel-harness/ncu/wo_split_8/kernel.ncu-rep`](../../cute_paged_attn/2026-05-03-w-o-k-parallel-harness/ncu/wo_split_8/kernel.ncu-rep).

## Production decoder cross-check (wo8 serving npy)

The `wo_split=8` arm of this soak captured an in-decoder region-timing dump during the supplementary profiler pass on 2026-05-07. Source: [`wo8/supplementary/sharegpt_region_timings.npy`](wo8/supplementary/sharegpt_region_timings.npy) (13440 B, shape `(64, 13, 2)` int64, globaltimer ticks).

Per-region medians (μs), single β-coop kernel call, computed directly from the npy:

| region | active CTAs | median μs | p99 μs | mean μs |
|---|---:|---:|---:|---:|
| `phase0_pre_attn` | 1 | 13.1 | 13.1 | 13.1 |
| `phase1_attn_pre_wo` | 4 | 229.9 | 230.0 | 226.6 |
| `phase1_wo_gemv` | **32** | **2359.8** | **2468.9** | **2274.5** |
| `phase1_wo_post` | 32 | 0.0 | 0.2 | 0.0 |
| `grid_barrier_wait` | 64 | 1809.6 | 2868.5 | 1613.5 |
| `phase3_load_x` | 64 | 1.7 | 1.8 | 1.7 |
| `phase3_partial_reset` | 64 | 0.1 | 0.2 | 0.1 |
| `phase3_3a_fc1_silu` | 64 | 566.8 | 615.9 | 568.3 |
| `phase3_3b_quant` | 64 | 0.5 | 0.6 | 0.6 |
| `phase3_3c_fc2_atomic` | 64 | 116.5 | 176.2 | 116.9 |
| `phase3_3d_arrival` | 64 | 0.1 | 2.6 | 0.3 |
| `phase4_residual` | 28 | 230.3 | 230.4 | 230.3 |
| `phaseE_post` | 1 | 162.9 | 162.9 | 162.9 |

**What this proves:** `phase1_wo_gemv` activates at **32 CTAs** in production decoding — i.e. the 8× K-parallel split is engaged in real serving, not just in the harness. The median is ~1.4× the harness's 1639 μs because production decoder context adds upstream gather work and barrier contention; serving-shape numbers are not 1:1 comparable to the standalone harness.

`grid_barrier_wait` at 1809.6 μs median (p99 2868.5 μs) remains the largest single component after the W_O reduction; the cooperative-launch grid barrier is the natural next target if a future iteration aims to recover more time inside the β-coop kernel.

## Known limitations & failed attempts

These are profiling/tooling limitations, not correctness or quality concerns. Documented for honesty about what was attempted vs what is in evidence.

1. **No same-run wo_split=1 serving region npy.** The 2026-05-07 attempt to capture a baseline `wo_split=1` region npy under matched serving conditions ([`wo1_region_pass.sh`](../../../../../docs/research/2026-05-04-wo-split-prod-soak/wo1_region_pass.sh)) failed to produce the dump: docker logs confirm the `[β-coop region timing] dumped …` line that fires at +26 s under `wo_split=8` does **not fire on the `wo_split=1` code path** in this serving config. The auto-dump hook is gated on something that wo_split=1 doesn't satisfy. Patching the backend to recover this evidence would change the code under measurement; we did not do so. Surviving wo1 supplementary artifacts (sharegpt outputs only) are at [`wo1/supplementary_2026-05-07/`](wo1/supplementary_2026-05-07/).

2. **Torch profiler `stop_profile` can hard-reboot the host.** Reproduced on wo4 supplementary (2026-05-05) and wo1 supplementary (2026-05-07) — `POST /stop_profile` hangs during kineto flush, the SoC OOMs, and the host hard-resets. Bounds (`limit_requests=4`, `max_prompt_chars=5500`) were sufficient against the host crash *during replay* but did not prevent the post-replay flush failure. wo8 supplementary survived without rebooting on 2026-05-07 but the engine still died mid-`stop_profile` (`AsyncLLM EngineDeadError`); kineto trace was lost. Net: torch profiler kineto traces under serving + region timing are not currently producible reliably on this configuration. Fix is out of scope for this soak.

3. **Region-timing extract tool error.** `extract_regions.py` fails on an empty trace dir (`IsADirectoryError: '.'`) when no kineto trace was flushed. The wo8 region CSV was instead computed directly from the npy via the inline breakdown in this writeup. Tool fix not done; treat as a known glue-script limitation.

## How to reproduce

End-to-end soak (primary + supplementary, all four arms):

```bash
cd /home/natfii/docker/nvllm
git checkout 5b8fc399f
WO_SPLITS="1,2,4,8" bash docs/research/2026-05-04-wo-split-prod-soak/runner.sh
```

Direct per-region breakdown from a region_timings.npy:

```bash
.venv/bin/python - <<'PY'
import numpy as np
buf = np.load("benchmarks/nvllm/traces/wo_split_prod_soak/2026-05-04-soak/wo8/supplementary/sharegpt_region_timings.npy")
d = buf[:,:,1] - buf[:,:,0]
d[buf[:,:,0] == 0] = -1  # mask non-executors
for r in range(13):
    a = d[:,r]; a = a[a >= 0]
    if a.size: print(f"r{r}: n={a.size:3d} med={np.median(a)/1000:.2f}us p99={np.percentile(a,99)/1000:.2f}us")
PY
```

Failed wo1 attempt (2026-05-07):

```bash
bash docs/research/2026-05-04-wo-split-prod-soak/wo1_region_pass.sh
# Expected: produces wo1/supplementary_2026-05-07/sharegpt_region_timings.npy
# Actual:   replay completes; stop_profile hangs; host reboots; npy never written.
```

## Files

- [`metadata.json`](metadata.json) — run config (last write was wo8 metadata; per-arm commits captured in primary docker.log)
- [`runner.log`](runner.log) — full primary + supplementary log stream (1500+ lines)
- [`summary.md`](summary.md) — auto-generated per-arm aggregate table (this writeup is the human-readable companion)
- `wo{1,2,4,8}/primary/` — GSM8K + 5×ShareGPT + 5×longdecode + 2-concurrent per arm, with `*_DONE` markers
- `wo8/supplementary/sharegpt_region_timings.npy` — production decoder W_O timing evidence (only npy that survived)
- `wo{1,4,8}/supplementary*/` — partial supplementary artifacts (sharegpt outputs only on wo1/wo4)
- [`../../../../../docs/research/2026-05-04-wo-split-prod-soak/`](../../../../../docs/research/2026-05-04-wo-split-prod-soak/) — runner scripts + replay tooling

## Cross-references

- [`2026-05-03-w-o-k-parallel-harness/summary.md`](../../cute_paged_attn/2026-05-03-w-o-k-parallel-harness/summary.md) — controlled kernel-level baseline (8.39× speedup, NCU classification)
- [`2026-05-02-beta-region-breakdown/region_breakdown.csv`](../../cute_paged_attn/2026-05-02-beta-region-breakdown/region_breakdown.csv) — pre-K-parallel production region timings (W_O at 13932 μs / 4 CTAs)
- PR #6 (W_O K-parallel implementation), PR #7 (W_O K-parallel validation) — code under measurement
