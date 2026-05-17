# β-coop launch-overhead profile (2026-05-16)

**Commit:** `1953ebbb0` (PR #17 merged)  
**Image:** `nvllm:gb10-bprime` sha256:`4fccbd915044a8f5f7db8268b0ec645323eb3d7063fd66233e64b1882e7c2539` (clean rebuild 2026-05-16)  
**Model:** `ig1/Qwen3.5-27B-NVFP4`  
**Hardware:** DGX Spark (GB10, SM120/121, 48 SMs, 128 GB unified)  
**Backend:** CuTe Paged, FP8 E4M3 KV, PIECEWISE cudagraphs  
**Config:** `CUTE_PHASE_E_FUSION=1 CUTE_PHASE_E_LAYERS=3,7 CUTE_WO_SPLIT=8`, `max_num_seqs=1`, decode `max_tokens=64`  
**Profiler:** torch profiler (vLLM V1 EngineCore — nsys cannot follow spawn boundary per `feedback_vllm_profiling`)  
**Window:** 15-req warmup outside profiler, then profiler-active window auto-bounded by `active_iterations=200` (~3 decode reqs captured per leg). `--timed 100` for `timing_on` (host launch walls need samples), `--timed 20` for `timing_off` (control needs only profiler-window samples). `record_shapes=False with_stack=False`.

## Purpose

Convert the B' single-call diagnostic (PR #17, ~23 ms `wall_minus_regions_us` per β-coop call) into a steady-state AGENTS.md §4 perf claim, AND identify whether the unaccounted budget is CPU launch overhead, GPU scheduling, or in-kernel time.

Two legs bound the instrumentation tax: `timing_on` has B' R-region writes + host CUDA-event recording (`CUTE_BETA_REGION_TIMING=1`); `timing_off` is the production control (`CUTE_BETA_REGION_TIMING=0`, `--timed 20` — only profiler-window samples needed since walls aren't captured).

## Headline finding — the 24 ms is IN-KERNEL, not launch overhead

Torch profiler kernel events show `kernel_dur ≈ host_event_wall`. The CPU `cudaLaunchKernelExC` API takes ~7 μs; the OUTER NVTX wrapping the launch is ~400 μs (includes KV update + Python bookkeeping). So the 24 ms `wall_minus_regions_us` budget is **in-kernel time that the R-regions don't cover** — either CTA-concurrency-amortized regions (per-CTA medians summed underweight wall-clock), or uninstrumented kernel work (barrier waits, register/SMEM spills, scheduling gaps between R-region exits and entries).

This **revises the B' memory framing** of "87% cooperative-launch/grid/tail overhead" — the cost lever is inside the kernel, not the launch path. Persistent kernel / smaller cooperative grid / batched launches will not help; the next-bet candidates are R-region coverage expansion (instrument the gaps) and an SMEM-spill / register-pressure audit.

## Per-call breakdown (median across timed window)

| Metric | timing_on | timing_off | Δ (tax) |
|---|---:|---:|---:|
| CPU `cudaLaunchKernelExC` API duration | 6.86 μs | 6.10 μs | +0.76 μs |
| GPU β-coop kernel duration (correlation-paired) | 27376.18 μs | 27464.36 μs | -88.18 μs |
| Outer `PhaseE_Beta.coop.*` NVTX range (CPU wall) | 378.21 μs | 374.65 μs | +3.56 μs |
| Host CUDA-event wall (B' instrumentation) | 27436.16 μs | — μs | — μs |

**Note on `queue_gap`:** the per-call CSV reports `queue_gap_us` (CPU launch-return → GPU kernel-start), but at our workload heavy β-coop kernels (~27 ms each) saturate the device stream so this measures **device backlog**, not scheduling latency. Excluded from the summary table to avoid misinterpretation.

**Region breakdown (timing_on leg only — `region_timings.npy` is a one-call snapshot, walls are timed-window medians):**

| Metric | Value |
|---|---:|
| Sum of per-CTA work-region medians | 3449.90 μs |
| `wall_minus_regions_us` (B' diagnostic) | **23986.26 μs** |

## Instrumentation tax — verdict

`kernel_dur` Δ = **-88 μs (-0.32%)** on timing_on vs timing_off; both legs have std ≈ 116 μs so this is **NEGLIGIBLE**.

The B' R-region writes + host CUDA-event recording do **not** perturb the β-coop kernel. The `wall_minus_regions_us` figure reflects production cost, not measurement artifact, and is safe to use as the perf-claim denominator.

## How to reproduce

The two legs were captured separately to keep wall-clock manageable (β-coop runs at ~2.5 tok/s, so each timed window is rate-limited):

```bash
# Leg 1 — timing_on (B' R-region writes + host CUDA-event recording).
# --timed 100 captures enough host launch walls for stable median.
bash docs/research/launch_overhead_profile/capture.sh smoke

# Leg 2 — timing_off (CUTE_BETA_REGION_TIMING=0 production control).
# --timed 20 since only profiler-window samples drive kernel_dur comparison.
bash docs/research/launch_overhead_profile/capture.sh control

# Analysis — pairs outer PhaseE_Beta.coop NVTX with cudaLaunchKernelExC,
# reduces B' region_timings.npy + host_launch_walls.npy, emits summary.md.
.venv/bin/python docs/research/launch_overhead_profile/analyze.py \
    --out-dir benchmarks/nvllm/traces/cute_paged_attn/2026-05-16-launch-overhead-profile \
    --timed-n 100
```

For a single bundled run (~70 min): `bash docs/research/launch_overhead_profile/capture.sh full`.

## Caveats

- **Torch profiler fallback called out.** vLLM V1 EngineCore is spawned; nsys cannot follow without an explicit `cudaProfilerApi` hook (not installed). Torch profiler captures CPU + GPU events with correlation IDs, which gives us the same per-call breakdown nsys would, but without grid-residency / SM-level detail.
- **`region_timings.npy` is one snapshot.** B' sentinel-file dump is one-shot per touch. host launch walls are the full per-call queue (timed-window slice taken in analyzer); region buf is the last β-coop call before the drain trigger.
- **wall_minus_regions_us interpretation.** Per the headline, the unaccounted budget is **in-kernel time** (kernel_dur ≈ host_event_wall), not pre-launch CPU or post-launch tail. Candidate causes: (a) per-CTA medians underweight wall-clock when CTAs run regions concurrently; (b) **R-region entry/exit likely bracket ONE iteration of inner K-tile loops, not the full per-call sum** — sum-of-medians undercounts by the loop trip count (Phase 3 R7-R9 in particular run inside the FC1/FC2 K-tile loop); (c) uninstrumented kernel work between R-region exits and the next R-region entry (barrier waits, register/SMEM spills). The B' instrumentation does not currently distinguish (a)/(b)/(c) — that's the next investigation.
- **β-coop solo-only at coop launch.** `max_num_seqs=1`; `num_seqs=2` is the next blocked target (`project_num_seqs_2_target`) but out of scope for this trace.
