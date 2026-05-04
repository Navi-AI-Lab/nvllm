# W_O K-parallel Validation Harness — NCU Classification + Sweep Evidence

**Date:** 2026-05-03
**Commit:** `46ad9bbc578c18c4df2c0c2ad064560f42d70377`
**Image:** `nvllm:gb10` (`sha256:9c0f1d31c92c29488f66a2c136183950cea787035d735ff95dd6af193740f530`)
**Hardware:** DGX Spark (GB10, SM120/SM121 — 48 SMs, 273 GB/s peak DRAM, 100 KB SMEM/SM)
**Source:** `docs/research/2026-05-03-w-o-k-parallel-harness/`
**Mirror of run scripts:** [`run_harness.py`](../../../../../docs/research/2026-05-03-w-o-k-parallel-harness/run_harness.py), [`run_sweep.sh`](../../../../../docs/research/2026-05-03-w-o-k-parallel-harness/run_sweep.sh), [`README.md`](../../../../../docs/research/2026-05-03-w-o-k-parallel-harness/README.md)

## Purpose

Validate, in isolation from the production decoder, that adding K-dimension parallelism to the β-coop W_O GEMV produces the speedup predicted by PR #6's per-region timing breakdown ("36% raw K-reducible; W_O is the bottleneck site, not FC1"). PR #6's matrix gate framed this as: "*Conditional. Pursue only if NCU shows memory-bound classification AND no other low-hanging fruit.*" This harness produces that classification.

## Bottom line

- **Bit-exact correctness** at wo_split ∈ {1, 2, 4, 8} against per-variant FP32 reference (`max_abs=0.0`, `max_rel=0.0`, AUTHORITATIVE gate via `reference_split_order(wo_split)`).
- **8.39× speedup** at 8× CTAs (32 W_O CTAs vs 4 baseline).
- **NCU classification:** baseline endpoint is **latency-limited**, scaled endpoint is **memory-bound**.
- The W_O lever is real; the slope is robust evidence for moving forward.

## NCU run log (provenance)

First NCU attempt aborted after **5h31m** due to unfiltered profiling of non-target kernels (cute.compile internals, torch init, etc.). The combination of `--target-processes all` and no `--kernel-name` filter generated a 2.7 GB report file that NCU was still post-processing when killed. Rerun used:

- `--kernel-name regex:wo_kernel_body` (matches the SASS-mangled symbol `kernel_cutlass__wo_kernel_body_________________0`)
- `--launch-count 1`
- dropped `--target-processes all` (harness uses `os.execvp`, no fork)
- kept `--replay-mode application` (cooperative-launch grid barrier deadlocks under kernel-replay)

Both endpoints completed in ~60 seconds each. Report sizes: 324 KB / 525 KB. Aborted attempt's metadata preserved at `ncu/ncu_unfiltered_aborted/wo_split_1/`; the 2.7 GB partial `.ncu-rep` was discarded as unparseable.

## Sweep result

50 launches per variant; mean computed excluding launch_idx=0 (warmup).

| wo_split | Total W_O CTAs | Mean elapsed (μs) | Effective GB/s (logical) | Speedup vs wo_split=1 |
|---:|---:|---:|---:|---:|
| 1 | 4 | 13754 | 1.49 | 1.00× |
| 2 | 8 | 5176 | 4.47 | 2.66× |
| 4 | 16 | 2693 | 10.6 | 5.11× |
| 8 | 32 | **1639** | **24.0** | **8.39×** |

Scaling is super-linear from 1→2 (2.66× speedup at 2× CTAs) and slightly sub-linear thereafter, consistent with the kernel transitioning out of latency-limited territory toward bandwidth-saturated. The "logical" GB/s is `(payload + scratch) / elapsed`; the actual DRAM throughput per NCU is below.

## NCU classification: wo_split=1 vs wo_split=8

Both runs profile the same kernel symbol with the same launch shape — only the W_O gate (`bx < wo_split`) controls how many of the 32 grid blocks do W_O work. The remaining blocks are upstream gather CTAs and run regardless of `wo_split`.

**Kernel:** `kernel_cutlass__wo_kernel_body_________________0`
**Grid:** `(8, 4, 1)` = 32 blocks
**Block:** `(128, 1, 1)` = 128 threads
**Threads launched:** 4096

| Section | Metric | wo_split=1 (4 W_O CTAs) | wo_split=8 (32 W_O CTAs) | Δ |
|---|---|---:|---:|---:|
| Memory | **Max Bandwidth (% peak DRAM)** | **8.06%** | **55.95%** | +47.9 pp |
| Memory | Mem Busy | 7.91% | 57.67% | +49.8 pp |
| Memory | L1/TEX Hit Rate | 46.4% | 50.7% | +4.3 pp |
| Memory | L2 Hit Rate | 97.3% | 97.6% | +0.3 pp |
| Memory | Mem Pipes Busy | 1.76% | 2.34% | +0.6 pp |
| Compute | SM Busy | 1.17% | 6.51% | 5.6× |
| Compute | Issue Slots Busy | 1.12% | 5.43% | 4.8× |
| Compute | Executed IPC Active | 0.07 | 0.33 | 4.7× |
| Scheduler | **No Eligible (% cycles)** | **98.33%** | **91.86%** | -6.5 pp |
| Scheduler | Eligible Warps/Scheduler | 0.02 | 0.08 | 4.0× |
| Scheduler | Active Warps/Scheduler | 1.00 | 0.99 | flat |
| Scheduler | One-or-More Eligible | 1.67% | 8.14% | 4.9× |
| Occupancy | Achieved Occupancy | 8.33% | 8.33% | flat |
| Occupancy | Theoretical Occupancy | 66.67% | 66.67% | flat |
| Occupancy | Achieved Active Warps/SM | 4.00 | 4.00 | flat |
| Launch | Registers/Thread | 58 | 60 | +2 |
| Launch | Waves/SM | 0.08 | 0.08 | flat |

### Verdict — wo_split=1: latency-limited

8% peak DRAM bandwidth, 1% compute pipeline utilization, **98% of cycles with no eligible warp**. The memory subsystem and compute pipelines are both idle; warps are stalled on dependencies (memory latency, sync). 4 active W_O CTAs on 48 SMs cannot generate enough in-flight work to hide DRAM latency. Achieved occupancy (8.33%) is far below theoretical (66.67%), driven by the kernel's compulsory wait pattern (cooperative grid barrier + chained K-reduction in W_O), not by occupancy limits per se.

### Verdict — wo_split=8: memory-bound

**56% peak DRAM bandwidth**, 6% compute pipelines, 92% no-eligible. Active and theoretical occupancy unchanged from wo_split=1 (because the grid is unchanged), but **eligible warps per scheduler quadrupled** (0.02 → 0.08) — the W_O parallelism puts enough independent work in flight to saturate the memory subsystem during active cycles. Bandwidth is now the dominant bottleneck.

### What this means for the matrix gate

PR #6's matrix gate said: "*Conditional. Pursue only if NCU shows memory-bound classification.*" The baseline endpoint *fails* that gate (latency-limited, not memory-bound). The scaled endpoint *passes* it (memory-bound). The implication: adding W_O K-parallelism is the mechanism that converts the kernel from latency-limited to memory-bound. The slope is real; the lever works.

There is still ~44% of peak DRAM bandwidth unclaimed at wo_split=8, suggesting wo_split=16 or further sharding could push closer to bandwidth saturation — at the cost of more scratch traffic (which already grows linearly with wo_split via slot writes + gather reads).

## Open audit: production parity gap

Per-W_O execution time in the production β-coop kernel (per PR #6's region breakdown) is materially faster per call than wo_split=1 here. **This is recorded as a follow-up audit item, not a conclusion.** Multiple differences need accounting before any "Nx overhead" claim is defensible:

- **Denominator.** PR #6's region time is per-layer over 40 layers in the production decoder. Active-token count, KV-head replication regime, batch shape, prefill-vs-decode mix — all may shift the per-call equivalent. The harness measures one isolated step.
- **Launch shape.** Production launches the full β-coop fused kernel (Phase D + W_O + reduction in one cooperative grid). This harness launches a stripped W_O microkernel-only path. Setup, dispatch, and shared-state costs differ.
- **Cache state.** Production benefits from prior-layer warmth on weights/scales; harness uses freshly-allocated tensors each launch (the high L2 hit rate suggests the weights fit in L2, but the cold first-touch path is paid every launch here).
- **Cooperative-launch overhead.** This harness uses `cooperative=True` to satisfy the grid-wide atomic-counter spin-wait barrier. Per-launch overhead from cooperative launch is a known cost we have not isolated.

The sweep slope (8.39× at 8× CTAs, with NCU showing the latency-bound → memory-bound transition) is valid evidence for the W_O lever regardless of the parity gap. **Resolving the absolute-timing mismatch is a separate denominator + launch-shape audit and should not be folded into the slope decision.**

## How to reproduce

```bash
# Sweep (no NCU) — produces the variant_*cta_scratchpad/ artifacts
bash docs/research/2026-05-03-w-o-k-parallel-harness/run_sweep.sh

# Single NCU classification run (example: wo_split=8)
mkdir -p benchmarks/nvllm/traces/cute_paged_attn/2026-05-03-w-o-k-parallel-harness/ncu/wo_split_8
docker run --rm --gpus all --privileged \
  -v /home/natfii/docker/nvllm:/work \
  -v /home/natfii/docker/nvllm:/app/nvllm \
  -v /tmp/cute_harness_cache_v3:/tmp/cute_harness_cache_v3 \
  --entrypoint ncu \
  nvllm:gb10 \
  --kernel-name regex:wo_kernel_body \
  --launch-count 1 \
  --replay-mode application \
  --section MemoryWorkloadAnalysis --section ComputeWorkloadAnalysis \
  --section LaunchStats --section Occupancy --section SchedulerStats \
  --csv \
  --log-file /work/benchmarks/nvllm/traces/cute_paged_attn/2026-05-03-w-o-k-parallel-harness/ncu/wo_split_8/ncu_stdout.log \
  --export /work/benchmarks/nvllm/traces/cute_paged_attn/2026-05-03-w-o-k-parallel-harness/ncu/wo_split_8/kernel.ncu-rep \
  /opt/venv/bin/python /work/docs/research/2026-05-03-w-o-k-parallel-harness/run_harness.py \
    --wo-split 8 --launches 1 \
    --out /work/benchmarks/nvllm/traces/cute_paged_attn/2026-05-03-w-o-k-parallel-harness/ncu/wo_split_8

# Inspect a report
docker run --rm -v /home/natfii/docker/nvllm:/work --entrypoint ncu nvllm:gb10 \
  --import /work/benchmarks/nvllm/traces/cute_paged_attn/2026-05-03-w-o-k-parallel-harness/ncu/wo_split_8/kernel.ncu-rep \
  --csv | head -60
```

## Artifacts

| Path | Contents |
|---|---|
| `variant_4cta_scratchpad/` | wo_split=1 sweep (config.json, timing.csv ×50, 3 correctness JSONs) |
| `variant_8cta_scratchpad/` | wo_split=2 sweep |
| `variant_16cta_scratchpad/` | wo_split=4 sweep |
| `variant_32cta_scratchpad/` | wo_split=8 sweep |
| `ncu/wo_split_1/` | NCU run on baseline endpoint (filtered, ~60s) |
| `ncu/wo_split_8/` | NCU run on scaled endpoint (filtered, ~60s) |
| `ncu/ncu_unfiltered_aborted/wo_split_1/` | Aborted first attempt — metadata only; 2.7 GB partial `.ncu-rep` was discarded |

## Next steps

1. **Decision (held in priority memo):** integrate W_O K-parallel into production β-coop, or hold pending the parity-gap audit. The slope evidence supports moving forward; the absolute-timing parity is unresolved.
2. **Parity-gap audit:** isolate the denominator (per-region wall time per active token) and launch-shape costs in production vs harness before drawing conclusions on absolute overhead.
3. **If integrating:** design the v2 production path (cache-key tagging, dispatch by `num_active_tokens`, reuse of barrier primitive *without* sharing slot id with upstream Phase-D) — out of scope for this harness; covered in §1b of the harness README.
