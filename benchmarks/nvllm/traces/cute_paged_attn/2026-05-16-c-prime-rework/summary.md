# β-coop C' rework: per-call accumulators close 98% of the B' 24 ms gap

**Commit:** `cb9d1327f` (PR #17 merged) + uncommitted C' rework patch on `phase_e_kernel.py`
**Host kernel SHA256:** `dcc72dd38ebc840af197a3ef6091e9c3ac13c53779e08c643a7f93a458e7ab14`
**Image:** `nvllm:gb10-cprime` (base `nvllm:gb10-bprime`, C' kernel applied via read-only bind-mount over editable install — no C++/CUDA rebuild required for this Python CuTe DSL source edit)
**Model:** `ig1/Qwen3.5-27B-NVFP4`
**Hardware:** DGX Spark (GB10, SM120/121, 48 SMs, 128 GB unified)
**Backend:** CuTe Paged, FP8 E4M3 KV, PIECEWISE cudagraphs
**Config:** `CUTE_PHASE_E_FUSION=1 CUTE_PHASE_E_LAYERS=3,7 CUTE_WO_SPLIT=8`, `max_num_seqs=1`, decode `max_tokens=64`, 200-req timed window
**Profiler:** torch profiler (vLLM V1 EngineCore — nsys cannot follow spawn boundary per `feedback_vllm_profiling`)

## Purpose

PR #17 (B' instrumentation) reported a ~23 ms `wall_minus_regions_us` gap per β-coop call — host CUDA-event wall ≈ 27.5 ms, but the sum of per-CTA region medians only accounted for ~5 ms. Three hypotheses for the gap:
- (a) per-CTA medians underweight wall-clock when CTAs run regions concurrently
- (b) R7-R9 brackets capture ONE iteration of the Phase-3 K-tile slice loop, so sum-of-medians undercounts by the loop trip count
- (c) uninstrumented kernel work between R-region exits and entries (barrier waits, register/SMEM spills, scheduling gaps)

The C' rework adds per-call accumulators R16/R17/R18 (sum-across-slice-loop counterparts of R7/R8/R9) plus R19 (post-loop atomic-arrival region that was previously absorbed by the old R9 hybrid exit). Bumps region buffer from 16 → 20.

**Validated via read-only bind mount over editable install; no C++/CUDA rebuild required for this Python CuTe source edit.** Bind-mount provenance recorded in `capture.log` shows `host_sha256 == ctr_sha256 == dcc72dd3…` on both timing_on and timing_off legs.

## Headline: loop-trip undercount dominates — gap collapses from 23 ms → 720 μs

Per-iter brackets (R7/R8/R9) capture one slice-loop iteration; per-call accumulators (R16/R17/R18) sum across the full ~34-iteration loop. The per-call sums absorb 22.6 ms of the 23 ms B' gap.

### Four-row comparison

| Metric | Median (μs) | Notes |
|---|---:|---|
| R7+R8+R9 (per-iter, last iter only for R9) | **682.0** | B' brackets — capture 1 of ~34 slice-loop iterations |
| R16+R17+R18 (per-call accumulators) | **23,324.5** | C' rework — sum across full slice loop |
| R19 (post-loop _threadfence + atomic_add_u32) | **0.4** | C' rework — was absorbed by old R9 hybrid exit |
| `wall_minus_regions_us` (residual after C' regions) | **719.9** | Down from ~23,000 in B' |
| Host CUDA-event wall (per-call) | **27,492.8** | timing_on leg, n=100 |

R16/R17/R18 / R7/R8/R9 ratios all ≈ 34× — the slice loop trip count is consistent across Phase-3 subregions, confirming hypothesis (b) is the dominant cause.

### Per-region breakdown (timing_on leg)

| ID | Region | n_active | Class | Median μs |
|---:|---|---:|---|---:|
| 0  | phase0_pre_attn               | 1  | phase0           | 12.3 |
| 1  | phase1_attn_pre_wo            | 4  | phase1           | 250.9 |
| 2  | phase1_wo_gemv                | 32 | phase1           | 2,336.0 |
| 3  | phase1_wo_post                | 14 | phase1           | 0.05 |
| 4  | grid_barrier_wait             | 64 | barrier_wait     | 1,718.2 *(excluded from work sum)* |
| 5  | phase3_load_x                 | 64 | phase3           | 1.9 |
| 6  | phase3_partial_reset          | 64 | phase3           | 0.06 |
| 7  | phase3_3a_fc1_silu (per-iter) | 64 | phase3           | 567.9 |
| 8  | phase3_3b_quant (per-iter)    | 64 | phase3           | 0.58 |
| 9  | phase3_3c_fc2_last_iter       | 64 | phase3           | 113.5 |
| 10 | phase3_3d_arrival             | 64 | phase3           | 0.1 |
| 11 | phase1_pre_wo_wait            | 28 | barrier_wait     | 251.0 *(excluded from work sum)* |
| 12 | phase1_gather_reduce          | 1  | dynamic_single   | 162.9 |
| 13 | prologue_pre_r0               | 64 | kernel_boundary  | 0.06 |
| 14 | epilogue_post_r10             | 15 | kernel_boundary  | 0.03 |
| 15 | phase3_3d_last_cta_gather     | 8  | dynamic_single   | 1.8 |
| **16** | **phase3_3a_fc1_silu_per_call** | 64 | phase3 | **19,349.9** |
| **17** | **phase3_3b_quant_per_call**    | 64 | phase3 | **19.4** |
| **18** | **phase3_3c_fc2_per_call**      | 64 | phase3 | **3,955.2** |
| **19** | **phase3_post_loop_atomic**     | 64 | phase3 | **0.4** |
| — | Σ work regions (excl. R4, R11) | — | — | **26,773.0** |

## Instrumentation tax — verdict NEGLIGIBLE

| Metric | timing_on | timing_off | Δ (tax) |
|---|---:|---:|---:|
| CPU `cudaLaunchKernelExC` API | 6.66 μs | 7.77 μs | -1.10 μs |
| GPU β-coop kernel duration (correlation-paired) | 27,567.20 μs | 27,559.98 μs | **+7.23 μs (+0.03%)** |
| Outer `PhaseE_Beta.coop.*` NVTX (CPU wall) | 376.48 μs | 398.39 μs | -21.91 μs |
| Host CUDA-event wall (timing_on only) | 27,492.83 μs | — | — |

Kernel-dur delta is +7 μs against an std of ~118 μs on both legs — **NEGLIGIBLE**. The B' + C' R-region writes (now 20 regions, per-call accumulators included) plus host CUDA-event recording do **not** perturb the kernel. C' measurements reflect production cost.

## Where the 720 μs residual lives

After C' rework, ~720 μs is still unaccounted vs the host wall. Candidates (not yet bisected):

- **(a) per-CTA median underweight**: even per-call accumulators take medians per CTA before summing; if any CTA is slower the median underestimates wall.
- **(c) uninstrumented inter-region scheduling gaps**: small gaps between exit of R16/R17/R18 and entry of the next region within the slice loop iteration cadence.
- **R10 phase3_3d_arrival (0.1 μs median, 2.64 μs p99)**: tail variance possible but median too small to explain 720 μs.

Bottom line: the 720 μs is a 2.6% slice of the 27.5 ms per-call budget — not a useful next-bet target. The ~98% gap closure means the B' "23 ms unaccounted" framing is fully resolved as a measurement artifact, not real overhead. **Next-bet candidates pivot back to the R16/R17/R18 regions themselves** (they dominate the budget — Phase 3a FC1+SiLU alone is 19.3 ms / 70% of kernel time).

## How to reproduce

```bash
# Bind-mount provenance + warmup + timed window (~70 min wall, both legs):
bash docs/research/launch_overhead_profile/capture.sh full

# Analyze (writes per_call.csv + region_breakdown.csv + summary.md):
.venv/bin/python docs/research/launch_overhead_profile/analyze.py \
    --out-dir benchmarks/nvllm/traces/cute_paged_attn/2026-05-16-c-prime-rework \
    --timed-n 100
```

NOTE: analyze.py's auto-written summary.md uses the B' template (no R16-R19 rows); this file overrides it with the C'-rework framing. Re-running analyze.py will overwrite this file — back it up first if you want to keep both views.

Bind-mount overlay (in `capture.sh`):
```bash
docker run -d \
  -v "$REPO_ROOT/vllm/v1/attention/backends/cute_paged/phase_e_kernel.py:/app/nvllm/vllm/v1/attention/backends/cute_paged/phase_e_kernel.py:ro" \
  ...
```

Per-leg provenance block in `capture.log` confirms `host_sha256 == ctr_sha256` before each timed window starts.

## Caveats

- **`region_timings.npy` is one snapshot.** B' sentinel-file dump is one-shot per touch; host launch walls are the full per-call queue (timed-window slice taken in analyzer); region buf is the last β-coop call before the drain trigger.
- **β-coop solo-only at coop launch.** `max_num_seqs=1`; `num_seqs=2` is the next blocked target (`project_num_seqs_2_target`) but out of scope for this trace.
- **C' source patch is uncommitted at trace time.** The kernel file is bind-mounted live (provenance hash verified per leg). Commit + PR after user review.

## Verdict against decision rule

User's pre-bench decision rule:
> if R16-R18 eats most of 24 ms → loop-trip undercount dominates
> if barely moves → CTA-concurrency/inter-region gaps
> partial close → ranked split

**Result: R16+R17+R18 = 23.3 ms absorbed, residual 720 μs (2.6%)** → loop-trip undercount is decisively the dominant cause. The B' "24 ms gap" was an instrumentation artifact of per-iter bracketing under a ~34-trip slice loop, not an in-kernel hot spot to optimize. The remaining attention pivots back inside the per-call accumulators themselves (especially R16 = 19.3 ms FC1+SiLU = 70% of per-call budget).
