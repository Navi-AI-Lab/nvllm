# Phase E.1 #4 — matched-concurrency baseline for β-lite — 2026-04-24

**Commit:** `612c96cb9` (tip of `main` — capture-script ship)
**Model:** `ig1/Qwen3.5-27B-NVFP4` (non-distilled)
**Hardware:** NVIDIA DGX Spark (GB10, SM120/121), 128 GB unified
**Image:** `nvllm:gb10`
**Backend:** `CUTE_PAGED`, `fp8_e4m3` KV cache, PIECEWISE CUDA graphs

## TL;DR

At matched concurrency (num_seqs=8, concurrent=8), **β-lite is ~63%
slower per full_attention layer decode step than the pre-Phase-E baseline**:

```
baseline_matched (FUSION=0):  DecodeKernel (17,038 μs) + Phase_D_MLP (104,499 μs × 1) = 121,537 μs
β-lite           (FUSION=1):  DecodeKernel (17,070 μs) + Phase_D_MLP ( 90,408 μs × 2) = 197,886 μs
                                                                       ─────────────────────────
                                                                       +76,349 μs / layer / step
```

β-lite's per-call MLP mean is 13.5% faster, but it fires `Phase_D_MLP_Kernel`
**twice** per full_attn layer per decode step (two-kernel fallback —
`n_calls = 2016` vs baseline's `1008` with identical DecodeKernel count).
Per-layer wall time is +76,349 µs despite per-launch being cheaper.

**Implication for Phase E.1 #2 (SMEM shrink):** the user's day-to-day workload
(Hermes agent + interactive = num_seqs ≥ 2) currently takes the β-lite path.
Shrinking β-coop SMEM until `resident_cap ≥ 128` isn't just "leave speedup
on the table" — it's **undoing a regression** vs the pre-E baseline for
the Phase-E-covered layers.

## Leg configuration

| Parameter | baseline_matched | β-lite (existing, 2026-04-23) |
|---|---|---|
| `CUTE_PHASE_E_FUSION` | 0 (Phase E disabled) | 1 |
| `CUTE_PHASE_E_PATH` | auto (moot, FUSION=0) | lite |
| `max_num_seqs` | 8 | 8 |
| client concurrent | 8 | 8 |
| warmup / timed | 4 / 5 | 4 / 5 |
| `max_tokens` | 64 | 64 |
| profiler `active_iterations` | 200 | 200 |
| `torch_profiler_record_shapes` | false | false |

Everything else identical: same model, same PIECEWISE graphs, same FP8 KV,
same `--gpu-memory-utilization 0.70`, same profiler window.

## Kernel-duration comparison (per-call mean, μs — no rounding)

### Phase E target kernels (16 full_attn layers, stride 4: indices 3, 7, …, 63)

| Kernel | baseline_matched | β-lite | Δ (per call) |
|---|---|---|---|
| `Phase_D_MLP_Kernel` (n=1008 vs 2016) | 104,499.102 | 90,408.545 | −13.5% |
| `DecodeKernel` (CuTe paged attention) | 17,038.461 | 17,069.767 | +0.18% |

### Phase E-inactive kernels (linear_attn FLA GDN path)

| Kernel | baseline_matched | β-lite | Δ |
|---|---|---|---|
| `fused_recurrent_gated_delta_rule_packed_decode_kernel` | 94.221 | 95.443 | +1.3% |
| `_causal_conv1d_update_kernel` | 4.187 | 4.209 | +0.5% |
| `cvt_fp16_to_fp4` | 2.142 | 2.155 | +0.6% |

Identical within noise — confirms Phase E enabling doesn't perturb
linear_attn layers and confirms FUSION=0 truly disabled β.

### Firing pattern (n_calls in the same 200-iteration profiler window)

| Kernel | baseline_matched | β-lite | ratio |
|---|---|---|---|
| `DecodeKernel` | 1008 | 1008 | 1.0× |
| `Phase_D_MLP_Kernel` | 1008 | 2016 | **2.0×** |

Same DecodeKernel count across legs ⇒ same number of full_attn decode
steps captured. β-lite's 2× MLP count is the two-kernel fallback
expressing its extra launch (Phase_D split into two halves when the
cooperative grid can't fit).

## Headline finding

Per Phase-E-covered layer per decode step (attention + MLP):

```
baseline_matched:  17,038 + 104,499        = 121,537 μs
β-lite          :  17,070 +  90,408 × 2    = 197,886 μs
                                             ────────────
                                             +76,349 μs  (+62.8%)
```

Applied to 16 full_attn layers per decode step, β-lite currently adds
**~1.22 ms of kernel wall time per decode step** vs the pre-E baseline
at num_seqs=8. The β-coop path (num_seqs=1 only today) replaces both
kernels with a single 42,934 μs fused launch — β-coop is the regime
where Phase E wins, and β-lite is a hold-harmless fallback that
currently isn't hold-harmless.

## Caveats

1. **β-lite per-call MLP (90,408 μs) is genuinely faster** than
   baseline's single-kernel MLP (104,499 μs). The regression is purely
   from the 2× firing count. Any change that fires β-lite exactly once
   per layer would make β-lite faster than baseline at num_seqs=8 — but
   that's β-coop by definition, gated on SMEM.

2. **Kernel-symbol naming is shared.** Both legs report
   `Phase_D_MLP_Kernel` in the trace CSV because the `@cutlass.jit`
   closure name is identical; baseline's is the pre-E single-phase
   variant and β-lite's is the two-phase variant. The FUSION=0 gate
   selected the right one (verified by absent `CuTe MLP fusion
   attached` lines in `baseline_matched_serve.log` and present ones
   in `beta_lite_serve.log`).

3. **torch profiler overhead ~15× on CuTe backend** (0.7 tok/s vs
   ~11 tok/s unprofiled). Absolute μs are kernel-wall accurate (CUPTI
   records device timing regardless of host-side overhead), but total
   throughput here isn't representative of production.

4. **`active_iterations=200` captures the same engine-step count in
   both legs** — that's why DecodeKernel n_calls matches exactly.
   `total_ms` is therefore directly comparable here (unlike the
   2026-04-23 β-coop vs β-lite comparison, where workload sizes
   differed).

## How to reproduce

```bash
# Capture baseline_matched leg (same env β-lite was captured under,
# CUTE_PHASE_E_FUSION=0):
bash docs/research/phase_e_traces/capture_baseline_matched.sh

# Extract per-kernel CSV:
.venv/bin/python docs/research/gemm_sweep/extract_e2e_kernels.py \
  --trace benchmarks/nvllm/traces/phase_e_1/2026-04-24-baseline-matched/baseline_matched.pt.trace.json.gz \
  --config baseline_matched \
  --out benchmarks/nvllm/traces/phase_e_1/2026-04-24-baseline-matched/baseline_matched_kernels.csv

# β-lite CSV (reused from Phase E initial bundle):
#   benchmarks/nvllm/traces/phase_e/2026-04-23-initial/beta_lite_kernels.csv
```

## Artifact index

| File | Bytes | Committed | Notes |
|---|---|---|---|
| `baseline_matched.pt.trace.json.gz` | 12,577,451 | no (local-only) | needs `.gitignore` entry under `phase_e_1/` |
| `baseline_matched_kernels.csv` | 31,160 | **yes** | 67 kernels |
| `baseline_matched_serve.log` | 66,627 | **yes** | EngineCore stdout/stderr, confirms FUSION=0 |
| `baseline_matched_mem.log` | 7,011 | **yes** | host + docker mem every 30 s |
| `profiler_out_0.txt` | 20,655 | **yes** | vLLM profiler stdout tail |

## Next

- **Phase E.1 #2 (SMEM shrink)** — priority raised by this evidence.
  Target `resident_cap ≥ 128` so num_seqs=2 takes the β-coop path and
  avoids the β-lite 2× MLP firing. Candidates:
    - Q-FP8 packing (−9% SMEM) — clean first step
    - K/V ping-pong (−36% SMEM) — bigger, restructures Phase B
    - stack both if single tweak isn't enough
- **Phase E.1 #5 (nsys `cudaProfilerApi` hook)** — would give
  register/SMEM/occupancy numbers needed to validate a shrink design.
