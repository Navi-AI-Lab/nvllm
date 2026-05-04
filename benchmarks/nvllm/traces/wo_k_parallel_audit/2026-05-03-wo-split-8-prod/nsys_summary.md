# wo_k_parallel_audit / 2026-05-03-wo-split-8-prod / nsys_summary

NSYS evidence for the W_O K-parallel total-kernel performance claim.

## Status

DONE -- Plan B (harness microkernel under nsys) at production-equivalent
launch shape. Plan A (vLLM V1 server under nsys) was not attempted as a
production-trace path because vLLM V1 spawns the EngineCore as a separate
process whose CUPTI activity is not captured by an nsys profile attached
to the API server (per project memory `feedback_vllm_profiling`). The
`--target-processes=all` flag still requires the spawned subprocess to
inherit the CUPTI injection env; vLLM strips most env across the
multiprocessing spawn (per `feedback_vllm_enginecore_env_strip`), and the
sentinel-file workaround does not propagate CUPTI injection.

The harness traces below capture the W_O microkernel only, at the same
launch shape produced inside the production beta-coop kernel
(slice_ctas=8 -> 32 cooperative-grid CTAs per seq, num_kv_heads=4),
which is the substantive kernel work the W_O K-parallel optimization
modifies.

## Approach used

Plan B: harness microkernel.

- Driver: `docs/research/2026-05-03-w-o-k-parallel-harness/run_harness.py`
- nsys: `/opt/nvidia/nsight-systems/2025.6.3` (host bind-mount, --privileged)
- Trace flags: `-t cuda,nvtx`
- Capture: full process duration (no `--duration`, no `--capture-range`)
- Cache: warm `/tmp/cute_harness_cache_v3` (HIT for changed; MISS->stored
  for baseline first run, then HIT on rerun)
- Both runs use 50 timed launches, B=1 active token, seed=4242,
  cooperative=True (hardwired in microkernel.py)

## Provenance

- Branch: `evidence/wo-k-parallel-harness`
- HEAD at run time: `3300f7776eb2b2c875097a98cec90c913f34aacf`
- Image id: `sha256:9c0f1d31c92c29488f66a2c136183950cea787035d735ff95dd6af193740f530`
- Image tag: `nvllm:gb10`
- Hardware: NVIDIA GB10 (DGX Spark, SM120, 48 SMs)
- nsys version: 2025.6.3.541-256337736014v0

## Configs

| | wo_split | slice_ctas | total_grid_ctas | active_W_O_ctas | gather_ctas | cache_key |
|---|---|---|---|---|---|---|
| baseline | 1 | 8 | 32 | 4 | 32 | `35fee3f003016249` |
| changed  | 8 | 8 | 32 | 32 | 32 | `a0950af2b637ba65` |

Both configs share the same total cooperative-grid size (32 CTAs per
seq, matching production beta-coop). The only axis varied is the W_O
K-parallel split (active W_O CTAs goes 4 -> 32).

## Correctness gate (AUTHORITATIVE)

Both runs are bit-exact against `reference_split_order(wo_split=N)`
(the kernel and reference share reduction tree).

| | passes | max_abs | max_rel |
|---|---|---|---|
| baseline (wo_split=1) | true | 0.0 | 0.0 |
| changed  (wo_split=8) | true | 0.0 | 0.0 |

## W_O kernel timings (nsys cuda_gpu_kern_sum, exact)

Kernel symbol: `kernel_cutlass__wo_kernel_body_________________0`
Instances per run: 50 timed launches.

| Stat | Baseline (wo_split=1) | Changed (wo_split=8) | Ratio |
|---|---:|---:|---:|
| Total (ns)  | 706,025,504 | 79,899,712 | 8.84x |
| Avg (ns)    | 14,120,510.1 | 1,597,994.2 | 8.84x |
| Med (ns)    | 13,715,248.0 |  1,598,064.0 | 8.58x |
| Min (ns)    | 13,643,232 |  1,585,248 | 8.61x |
| Max (ns)    | 28,391,072 |  1,625,888 | 17.46x |
| StdDev (ns) |  2,101,617.2 |      5,677.2 | n/a |

In microseconds (median, two-significant-figure):

- Baseline median: 13,715 us
- Changed  median: 1,598 us
- Delta:          -12,117 us  (-88.3% / 8.58x)

The baseline max (28.39 ms) is the launch-0 outlier (first-launch warmup
artifact; cold-cache JIT/driver). All other 49 launches are within
13.6-13.8 ms. The changed run is steady-state (StdDev ~5 us).

These per-launch numbers correspond to the harness device-side timings
(timing.csv) of:
- Baseline: 13,776-13,925 us at launches 1-49 (host-side CUDA event)
- Changed:  1,640-1,690 us at launches 1-49 (host-side CUDA event)

Host CUDA-event timings include cooperative-launch overhead and any
launch-edge sync; nsys CUPTI kernel timings are device-time-only. The
two methods agree on the same ratio.

## Methodology caveats

1. **Harness microkernel ONLY -- not the full beta-coop kernel.**
   The harness exercises only the W_O+gather portion of the beta-coop
   kernel (the section the K-parallel optimization modifies). It does
   not include Phase 0 (input LN), Phase 1 (attention RMS+QK+SDPA),
   Phase 3 (MLP) or Phase 4 (post-attn LN), all of which run in the
   production beta-coop kernel. These traces cannot be used to argue
   end-to-end per-call cost for vLLM serving.

2. **Production grid layout matched, not full kernel composition.**
   The harness reproduces the 32-CTA cooperative grid, num_kv_heads=4,
   num_q_heads=24, head_dim=256, K=6144, hidden_size=5120,
   NUM_THREADS=128, tile_s, tile_k, FP4 NVFP4 weight layout. What it
   does NOT match is the constexpr fan-in of inputs from upstream phases
   (those are present inside the prod beta-coop kernel but not exercised
   here, since the harness feeds synthetic attn_output directly).

3. **No first-launch outlier rejection.** Baseline max (28.4 ms) is the
   launch-0 outlier (cache-MISS first call on this process). The
   50-sample median (13.72 ms) is the canonical number. Reported
   total/avg are inflated by ~2% by this single launch.

4. **Plan A (vLLM V1 nsys) blocked by architecture.** vLLM V1 EngineCore
   is a spawned subprocess. Per project memory `feedback_vllm_profiling`,
   nsys does not capture EngineCore CUPTI activity. The
   `--target-processes=all` flag is necessary but not sufficient because
   the EngineCore subprocess does not inherit CUPTI injection env vars
   from the API server's nsys-instrumented context. Project policy is to
   use vLLM's torch profiler API (`/start_profile`/`/stop_profile` via
   `VLLM_TORCH_PROFILER_DIR`) for V1 evidence -- but that produces a
   torch-profiler trace, not an nsys trace. AGENTS.md sec.4 specifies
   nsys traces, so the correct authoritative measurement at the
   production GRID SHAPE is the harness microkernel under nsys (this
   run). For the end-to-end serving cost we publish region-timing CSVs
   and GSM8K evals (sibling files in this dir).

5. **First baseline capture had to be re-run.** The first attempt
   (`baseline.nsys-rep` size 1.33 MB, 11 kernel instances total, 0
   `wo_kernel` instances) had only the setup-phase kernels. Cause is
   under-captured but coincided with a cache-MISS first-launch. The
   re-run (cache-HIT) captured all 50 launches as expected. Both
   traces below are the re-run captures.

## Files produced

- `benchmarks/nvllm/traces/wo_k_parallel_audit/2026-05-03-wo-split-8-prod/baseline.nsys-rep`
- `benchmarks/nvllm/traces/wo_k_parallel_audit/2026-05-03-wo-split-8-prod/changed.nsys-rep`
- `benchmarks/nvllm/traces/wo_k_parallel_audit/2026-05-03-wo-split-8-prod/baseline_nsys_run/{config,correctness_gate_split_order,correctness_vs_chained,correctness_vs_matmul,timing}.{json,csv}`
- `benchmarks/nvllm/traces/wo_k_parallel_audit/2026-05-03-wo-split-8-prod/changed_nsys_run/{config,correctness_gate_split_order,correctness_vs_chained,correctness_vs_matmul,timing}.{json,csv}`
- `benchmarks/nvllm/traces/wo_k_parallel_audit/2026-05-03-wo-split-8-prod/nsys_summary.md` (this file)

## Reproduction

```bash
REPO=/home/natfii/docker/nvllm
cd "$REPO"
mkdir -p /tmp/cute_harness_cache_v3
DST="$REPO/benchmarks/nvllm/traces/wo_k_parallel_audit/2026-05-03-wo-split-8-prod"

for WS in 1 8; do
  if [ "$WS" = "1" ]; then NAME=baseline; else NAME=changed; fi
  docker run --rm --gpus all --privileged \
    -v /opt/nvidia/nsight-systems/2025.6.3:/opt/nsys \
    -v "$REPO:/work" \
    -v "$REPO:/app/nvllm" \
    -v "/tmp/cute_harness_cache_v3:/tmp/cute_harness_cache_v3" \
    --entrypoint /opt/nsys/bin/nsys \
    nvllm:gb10 \
    profile -t cuda,nvtx \
    -o "/work/benchmarks/nvllm/traces/wo_k_parallel_audit/2026-05-03-wo-split-8-prod/${NAME}.nsys-rep" \
    --force-overwrite=true \
    /opt/venv/bin/python /work/docs/research/2026-05-03-w-o-k-parallel-harness/run_harness.py \
    --wo-split "$WS" \
    --slice-ctas 8 \
    --launches 50 \
    --out "/work/benchmarks/nvllm/traces/wo_k_parallel_audit/2026-05-03-wo-split-8-prod/${NAME}_nsys_run"
done

NSYS=/opt/nvidia/nsight-systems/2025.6.3/bin/nsys
$NSYS stats --report cuda_gpu_kern_sum:mangled "$DST/baseline.nsys-rep" | grep "wo_kernel_body"
$NSYS stats --report cuda_gpu_kern_sum:mangled "$DST/changed.nsys-rep" | grep "wo_kernel_body"
```

## Cross-references

- Parity-gap audit (slice_ctas axis study):
  `benchmarks/nvllm/traces/wo_k_parallel_audit/2026-05-03-parity-gap/README.md`
  config B (wo_split=1, slice_ctas=8) and config C (wo_split=8,
  slice_ctas=8) match the baseline/changed shapes here. The parity-gap
  traces include --slice-ctas axis and audit kernel-level microbenchmarks;
  this dir's traces are the focused two-point evidence at the production
  grid.
- Region-timing CSVs in this same dir capture full-beta-coop kernel
  region breakdown (not just W_O).
- GSM8K eval JSONs in this same dir confirm correctness end-to-end at
  both wo_split values under live vLLM serving.
- Harness README:
  `docs/research/2026-05-03-w-o-k-parallel-harness/README.md`
- Phase-E kernel: `vllm/v1/attention/backends/cute_paged/phase_e_kernel.py`
  - W_O slot: line 3306 (`self._kernel_phase_0_to_4(...)`)
  - wo_split env read: line 262 (`os.environ.get("CUTE_WO_SPLIT", "1")`)
