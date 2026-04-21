# Phase A — Stream-K × CUDA graphs verdict

**Date:** 2026-04-21
**Commit:** `a4806469b` (branch `feat/gemm-sweep-2026-04-21`)
**Model:** ig1/Qwen3.5-27B-NVFP4 (`4c546624f1fa8b77f5b7cfb3b6c96bf46d25c3a9`)
**Hardware:** DGX Spark (GB10, SM120 / 121)
**Image:** nvllm:gb10 (sha256:629836e332bbf3462f4f3b09e66e696da604fc5bfe00aa75867e2954959833d2)

## Verdict

**WIN SURVIVES.** Stream-K is **+11.3% faster** than the pre-Stream-K M256
config for decode-M NVFP4 GEMMs under full-piecewise CUDA graphs on SM120.

Threshold from plan: WIN SURVIVES at ≥+10%. 11.33% lands just above the bar;
slightly below the original +12.7% eager-mode claim in memory `project_cutlass_tuning`,
consistent with CUDA-graph capture overhead plus torch-profiler overhead.

**Action:** close the pending CUDA-graphs test flag on Stream-K. Proceed to
Phase B (M=17–256 sweep).

## Measurement

Both runs: identical workload (30 warmup + 30 timed at concurrency 4,
max_tokens=256, fixed prompt, ignore_eos=True), identical serve config
(triton_attn, `--kv-cache-dtype auto`, `--max-num-seqs 4`, PIECEWISE CUDA
graphs), identical profiler config (torch + CUPTI, active_iterations=600).
Only difference: `NVLLM_FP4_GEMM_DISABLE_STREAMK` env var (unset for A.3,
=1 for A.4 — forces `mp2<=16` to fall through to the M256 config).

### Isolating the dispatch target

CUTLASS emits **distinct mangled symbols** for the Stream-K and M256
instantiations. The diff against the two traces:

| Trace | Mangled symbol contains | n | mean (μs) | p50 (μs) | p95 (μs) | total (ms) |
|---|---|---|---|---|---|---|
| A.3 Stream-K (M≤16 path) | `ILi2ELi3E...StreamKScheduler...` tile `<128,_,256>` | **326,400** | **289.560** | 233.587 | 440.221 | 94,512.48 |
| A.3 Stream-K (M=17-256 prefill) | `ILi4ELi3E...vEEE` tile `<128,_,128>` | 2,240 | 346.362 | 289.688 | 542.789 | 775.85 |
| A.4 Baseline (all M≤256) | `ILi4ELi3E...vEEE` tile `<128,_,128>` | **328,800** | **326.542** | 278.529 | 497.123 | 107,366.99 |

A.3 has **two** GemmUniversal device-kernel symbols (Stream-K for small M,
M256 for prefill chunks). A.4 collapses both into the single M256 symbol.

### Delta

Comparing the M≤16 decode path:

- A.3 Stream-K branch: mean **289.56 μs**
- A.4 baseline (same calls redirected to M256 config): mean **326.54 μs**
- **Delta: (326.54 − 289.56) / 326.54 = +11.33% faster with Stream-K**

Prefill-chunk calls (n≈2,240 in A.3) are comparable between runs (346 μs
vs the M256 population mean 326 μs — small enough to ignore for this
verdict; Phase B will sweep M=17–256 configs).

## NVTX M-tag smoke test

**Failed to land** — zero `fp4_gemm_bf16 M=...` NVTX annotations in either
trace's `user_annotation` category. The `__has_include(<nvtx3/nvToolsExt.h>)`
guard in the C++ source must have fallen back to no-op at compile time
(the CUDA toolkit's nvtx3 header isn't in the include path used by the
CUTLASS extension build). This was explicitly called out as a best-effort
punt-to-follow-up in the spec; downstream M-bucketing for Phase B will
use shape-implied M (the distinct mangled symbols already differentiate
small-M vs mid-batch anyway).

## How to reproduce

```bash
# Stream-K leg (production default path):
bash docs/research/gemm_sweep/capture_phase_a_trace.sh streamk

# Baseline leg (Stream-K disabled via runtime env var):
bash docs/research/gemm_sweep/capture_phase_a_trace.sh baseline
```

Each leg takes ~15 min (server warmup + 260s workload + 2 min CUPTI
flush + teardown). Raw `rank0.*.pt.trace.json.gz` files are ~170 MB
each and **kept local-only** (gitignored per `.gitignore` surgical pattern);
`*_graphs_kernels.txt` contains the committed per-kernel summaries.

## Notes

- **Profiler-only comparison.** Throughput numbers here (0.13 req/s at
  concurrency 4) are dominated by torch-profiler instrumentation overhead
  and are NOT representative of production throughput. The verdict rests
  entirely on per-kernel μs.
- **Hybrid architecture.** 48/64 Qwen3.5 layers are `linear_attention`
  (SSM) running BF16 via CUTLASS `cutlass_80_wmma_tensorop_bf16_*` kernels
  (top-10 by total ms in both runs). Those are outside the NVFP4 dispatcher
  and so unaffected by the Stream-K gate. Only the 16 `full_attention`
  layers (qkv/o_proj) and all 64 MLP layers (gate_up/down_proj) hit the
  FP4 GEMM dispatcher.
- **Env-var gate remains in production code.** `NVLLM_FP4_GEMM_DISABLE_STREAMK`
  is additive (default behavior unchanged when env unset). Leaving it in
  gives future sessions a cheap escape hatch if a Stream-K regression shows
  up without needing a rebuild.

## Files

- `streamk_graphs_kernels.txt` — A.3 (Stream-K) per-kernel μs table
- `baseline_graphs_kernels.txt` — A.4 (pre-Stream-K baseline) per-kernel μs table
- `decode_streamk.txt` — A.3 container log
- `decode_baseline.txt` — A.4 container log
- `baseline_search.txt` — A.1 pre-capture search notes (STREAM_K_BASELINE_FOUND=no → A.4 required)
- `streamk_graphs.pt.trace.json.gz` (local-only, 182 MB) — A.3 raw profiler artifact
- `baseline_graphs.pt.trace.json.gz` (local-only, 170 MB) — A.4 raw profiler artifact
