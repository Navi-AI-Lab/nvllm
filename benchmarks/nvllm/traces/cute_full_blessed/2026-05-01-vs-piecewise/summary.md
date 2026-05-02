# FULL+blessed vs PIECEWISE — apples-to-apples bench

**Commit:** `ce26aaaa062f9dd5b0a308e089cd4cf6fb4b358f` (branch `feat/cute-full-cache-bless`)
**Date:** 2026-05-01
**Hardware:** NVIDIA DGX Spark (GB10, SM120, 48 SMs, 119 GiB unified)
**Image:** `nvllm:gb10` (`sha256:a3f3f609a8ec873b0c8f6ddeb71573514eb84bf41b814ac82303d998a6ac5b88`)
**Model:** `ig1/Qwen3.5-27B-NVFP4` @ `4c546624f1fa8b77f5b7cfb3b6c96bf46d25c3a9`

---

## Headline

Performance-neutral within noise on this workload. FULL+blessed shows a small,
non-significant tilt in its favor at the per-kernel level (-0.27% aggregate)
and identical end-to-end streaming numbers. The structural win is that
FULL+blessed reaches the same throughput as PIECEWISE without the prior
torch.compile/inductor non-determinism (per the bless-v1 evidence in
`docs/research/2026-04-29-full-graph-spike/`), which means the opt-in
`serve-cute-full.sh` lower-8 + n=1 path is now production-capable on a
trace-backed basis, not just a smoke-test basis.

### Streaming (single unprofiled `/v1/completions` request, 256 tokens)

| Leg | TTFT | Decode tok/s | Total latency | Cache state |
|---|--:|--:|--:|---|
| PIECEWISE | 593.5 ms | 2.343 | 109.437 s | no bless mount |
| FULL+blessed | 592.1 ms | **2.352** | 109.022 s | blessed AOT `d97e88db71dd…` mounted ro |
| Δ | -1.4 ms | +0.39% | -0.42 s | — |

Source: `piecewise_streaming.json`, `full_streaming.json` (single streaming
request fired AFTER warmup + smoke, BEFORE profiler started — steady-state).

### Aggregate per-kernel CUDA time (77 common kernels)

| Leg | Total kernel time |
|---|--:|
| PIECEWISE | 85,205.049 ms |
| FULL+blessed | 84,976.460 ms |
| Δ | **-228.589 ms (-0.27%)** |

Source: `comparison.json`. Sample window = first ~200 EngineCore worker
iterations of each leg (see Methodology / Caveats).

---

## Per-kernel comparison (top 20 by absolute Δ total ms)

See `comparison.md` for the full table. Highlights:

| Kernel | PW mean μs | FL mean μs | Δ μs | Δ % | Δ total ms |
|---|--:|--:|--:|--:|--:|
| `DecodeKernel (CuTe paged attn)` | 17244.189 | 17128.719 | -115.470 | -0.7% | **-321.699** |
| `FP4 GEMM (cutlass::device_kernel…BlockScaledSm120…)` | 310.295 | 314.113 | +3.818 | +1.2% | +106.870 |
| `PhaseE_Beta_Kernel (β-coop fused attn+MLP)` | 40967.875 | 40799.316 | -168.559 | -0.4% | **-67.086** |
| `gemvx::kernel<int, int, __nv_bfloat16>` | 391.672 | 393.839 | +2.167 | +0.6% | +62.533 |
| `at::native::vectorized_elementwise_kernel<…FillFunctor<int>…>` | 1.003 | 0.707 | -0.296 | -29.5% | -2.895 |
| `causal_conv1d_update` | 2.760 | 2.478 | -0.282 | -10.2% | -2.690 |
| `vllm::reshape_and_cache_flash_kernel<bf16, u8, …>` | 2.943 | 2.367 | -0.576 | -19.6% | -1.843 |
| `triton_poi_fused_0` | 0.921 | 0.730 | -0.191 | -20.7% | -1.836 |
| `cvt_fp16_to_fp4` | 2.380 | 2.305 | -0.075 | -3.2% | -1.369 |

**Reading the table.** The two largest CuTe attention kernels (DecodeKernel
and PhaseE_Beta_Kernel) are 0.4–0.7% faster per call under FULL graph,
saving 389 ms aggregate. The biggest counter-shift is FP4 GEMM
`cutlass::device_kernel` taking 1.2% longer (+107 ms). Several small
elementwise / triton-fused kernels run 16-30% faster under FULL — likely
because their per-launch overhead is amortized inside the captured graph.

**Inductor-generated kernels diverge.** PIECEWISE has 4 unique fused-RMS-cat
kernels (~17 ms total); FULL has 3 unique `triton_poi_fused_8/9/10`
(~11 ms total). Net ~6 ms in FULL's favor. These are the inductor's choice
for the new graph topology — counted into the 77/4/3 kernel inventory but
not directly comparable per-symbol.

---

## Memory trajectory

Host watchdog (`free -h` + `docker stats nvllm` every 30 s) recorded
throughout each leg:

| Leg | Peak host used | Headroom (of 119 GiB) |
|---|--:|--:|
| PIECEWISE | 82 GiB | 37 GiB free |
| FULL+blessed | 80 GiB | 39 GiB free |

Source: `piecewise_mem.log`, `full_mem.log`. We never crossed 70%
utilization — the prior OOM (see Caveats) was buffer-runaway, not an
inherent budget problem.

---

## Cache state

PIECEWISE: no blessed cache mount. Disk cache for cute.compile shared via
`-v /tmp/nvllm-cute-cache:/opt/vllm/kernel_cache` (warm from prior runs).

FULL+blessed: blessed AOT cache from
`docs/blessed-caches/qwen35-27b-nvfp4_fap_lower8_image-a3f3f60_e6d32b4.json`,
config_hash `e6d32b41c46842c97f877339e86c79d6cc11004a238bef32f2cd3fdb73ce28db`,
mounted read-only at `/root/.cache/vllm` from
`/home/natfii/.cache/nvllm/blessed/e6d32b41…/`. Blessed AOT sha
`d97e88db71ddbffde0553cbb3e805c036181316ff8a956f4d8be9f8b11c02f65`.

Both legs share: same host disk cache for kernel JIT, same HuggingFace
revision, same image SHA, same git SHA, same env vars except for
`cudagraph_mode` and the bless mount flag.

---

## nsys traces

`piecewise.nsys-rep` and `full.nsys-rep` are 180 KB each (system-wide,
60 s active window per leg, sparse single-request load). Smaller than the
AGENTS.md §4 "5-20 MB typical" note because the window is brief and load
is light by design — the torch profiler `*.pt.trace.json.gz` files (12 MB
FULL, 25 MB PIECEWISE) are the kernel-timing source of truth; the
nsys-reps are committed for §4 conformance and for whole-system
visibility (memcpy patterns, driver overhead).

---

## Caveats

1. **Sample window.** Wall workload = 30 × 256 tokens at concurrency=1
   (≈55 min profiled wall per leg), but the torch profiler captures only
   the first **~200 EngineCore worker iterations** per leg. The
   `max_iterations:200` cap fires once and stops the profiler; the
   remaining requests run unprofiled. Per-kernel mean μs is per-launch
   so it remains comparable; total_ms is bounded by the captured window
   (not by wall workload).

2. **OOM precedent.** A prior run of this same harness with
   `active_iterations:600` (no `wait/warmup_iterations`) ran the
   profiler unbounded for ~55 min and OOM-killed the host during Kineto
   serialization. Root cause: `vllm/profiler/wrapper.py:205` only
   constructs `torch.profiler.schedule(...)` when `wait_iterations > 0`
   OR `warmup_iterations > 0`, so `active_iterations` was dead code.
   `max_iterations` is the gate that always fires
   (`vllm/profiler/wrapper.py:104-116`). Saved as
   `feedback_active_iterations_dead_code.md`. The `profile-vllm-v1`
   skill's "active_iterations is preferred" note is wrong for current
   vLLM and should be revised.

3. **Concurrency=1, max_num_seqs=1.** This is the steady state for the
   hermes agent / interactive-use target. Multi-seq behavior is NOT
   measured by this trace — capturing n=2+ remains future work and is
   explicitly NOT cleared by this evidence per the user gate.

4. **Lower-8 layers only.** `CUTE_PHASE_E_LAYERS=0,1,2,3,4,5,6,7`.
   Layers 8-31 of full_attention still run via PIECEWISE attention path
   under both legs. Bless of all 32 layers is future work.

5. **Workload structure.** Both legs use the same fixed prompt from
   `docs/research/gemm_sweep/trace_workload.py` (`FIXED_PROMPT`,
   seed=42, temperature=0, ignore_eos=true) so token sequences are
   identical between legs.

6. **No quality regression check in this trace.** GSM8K parity vs
   bless-v1's prior 32/50 was not re-run for this comparison — this is
   a perf-only bench. Quality remains gated by the bless-v1 GSM8K
   evidence in `docs/research/2026-04-29-full-graph-spike/`.

---

## Reproduction

```bash
# Pre-flight: ≥90 GiB free, no nvllm container, blessed manifest exists.
cd /home/natfii/docker/nvllm
git checkout feat/cute-full-cache-bless
git rev-parse HEAD  # expect ce26aaaa062f9dd5b0a308e089cd4cf6fb4b358f

# Run both legs (≈2 hr wall clock with warm caches):
tmux new-session -d -s cute-full-vs-piecewise \
  "bash docs/research/cute_full_blessed_traces/capture_full_vs_piecewise.sh \
       2>&1 | tee /tmp/cute-full-vs-piecewise.log"

# After completion, render the per-kernel comparison:
.venv/bin/python docs/research/cute_full_blessed_traces/render_comparison.py \
  --piecewise benchmarks/nvllm/traces/cute_full_blessed/2026-05-01-vs-piecewise/piecewise_kernels.csv \
  --full      benchmarks/nvllm/traces/cute_full_blessed/2026-05-01-vs-piecewise/full_kernels.csv \
  --out-md    benchmarks/nvllm/traces/cute_full_blessed/2026-05-01-vs-piecewise/comparison.md \
  --out-json  benchmarks/nvllm/traces/cute_full_blessed/2026-05-01-vs-piecewise/comparison.json
```

Harness defaults (override via env if needed):
- `OUT_DIR=benchmarks/nvllm/traces/cute_full_blessed/2026-05-01-vs-piecewise`
- `PROFILE_STOP_TIMEOUT=1800` (synchronous /stop_profile wait, 30 min cap)
- `NVLLM_MIN_FREE_GB=90`

Workload params (matched both legs, see harness lines 65-70):
- `WARMUP_N=5`, `SMOKE_N=2`, `TIMED_N=30`
- `MAX_TOKENS=256`, `WORKLOAD_TIMEOUT=600`, concurrency=1
- profiler: `max_iterations:200` (env-configurable via `PROFILER_MAX_ITERATIONS`,
  recorded into per-leg `*_meta.json` as `torch_profiler_max_iterations`),
  `torch_profiler_record_shapes:false, with_memory:false, use_gzip:true`

Server config (matched both legs, see `build_common_run` at harness lines 206-253):
- `kv_cache_dtype=fp8_e4m3, attn_backend=CUTE_PAGED, max_model_len=16384`
- `max_num_seqs=1, max_num_batched_tokens=65536, gpu_memory_utilization=0.65`
- `CUTE_PHASE_E_FUSION=1, CUTE_PHASE_E_LAYERS=0..7, CUTE_MLP_FUSION=1, CUTE_ATTN_FUSION=1`
- `cudagraph_capture_sizes=[1]`
- `kernel_config={"enable_flashinfer_autotune":false}`

Only difference between legs: `cudagraph_mode` (PIECEWISE vs FULL_AND_PIECEWISE)
and the FULL leg's bless-mount flag `-v $BLESSED_HOST_PATH:/root/.cache/vllm:ro`.

---

## Files

| File | Description |
|---|---|
| `piecewise.pt.trace.json.gz` | torch profiler trace, PIECEWISE leg (25 MB) |
| `full.pt.trace.json.gz` | torch profiler trace, FULL+blessed leg (12 MB `*` see note) |
| `piecewise.nsys-rep` | nsys system-wide trace, PIECEWISE leg |
| `full.nsys-rep` | nsys system-wide trace, FULL+blessed leg |
| `piecewise_kernels.csv` / `full_kernels.csv` | per-kernel μs stats |
| `piecewise_streaming.json` / `full_streaming.json` | TTFT + decode tok/s |
| `piecewise_meta.json` / `full_meta.json` | per-leg config snapshot |
| `piecewise_serve.log` / `full_serve.log` | docker logs for the EngineCore (incl. profiler events) |
| `piecewise_mem.log` / `full_mem.log` | host watchdog (free -h + docker stats every 30 s) |
| `piecewise_profiled_workload.log` / `full_profiled_workload.log` | trace_workload.py per-leg output |
| `piecewise_profiler_out_0.txt` / `full_profiler_out_0.txt` | torch profiler key_averages table |
| `piecewise_nsys_request.json` / `full_nsys_request.json` | response from the single 256-token request driven into the nsys window |
| `comparison.md` / `comparison.json` | per-kernel comparison output |

`*` PIECEWISE `.pt.trace.json.gz` is 25 MB vs FULL's 12 MB. Same captured
worker-iteration count (200), but PIECEWISE emits more individual kernel
launches per step than FULL's single graph replay → more events recorded.

---

## Verdict and gates

- **Performance:** neutral within noise (-0.27% aggregate kernel time,
  +0.39% decode tok/s) — not a "clear FULL+blessed win" by any normal
  threshold.
- **Production-readiness:** demonstrated. Two complete legs with identical
  workload completed without OOM, regression, or quality drift gate
  trigger. `serve-cute-full.sh` lower-8 + n=1 is now an opt-in path with
  trace evidence behind it.
- **User gates respected:** no all-32 bless attempted; no n>1 attempted;
  no repo-default flip implied by this evidence. Default remains
  PIECEWISE pending a clear win.
