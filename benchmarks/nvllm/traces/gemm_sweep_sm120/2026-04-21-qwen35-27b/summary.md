# NVFP4 GEMM Sweep — Qwen3.5-27B-NVFP4 — 2026-04-21

**Commit:** `7f6a7956e`
**Branch:** `feat/gemm-sweep-2026-04-21`
**Model:** ig1/Qwen3.5-27B-NVFP4 (snapshot `4c546624f1fa8b77f5b7cfb3b6c96bf46d25c3a9`)
**Hardware:** DGX Spark (GB10, SM120 / 121)
**Image:** nvllm:gb10 (post-commit `c83583721` — includes `NVLLM_FP4_GEMM_CONFIG_M256` runtime dispatcher)

## Phase A verdict (recap)

Stream-K × CUDA graphs: **WIN SURVIVES, +11.3% faster** than pre-Stream-K M256
at decode M. See sibling summary:
[`../gemm_stream_k_cudagraph/2026-04-21/summary.md`](../../gemm_stream_k_cudagraph/2026-04-21/summary.md).

## Phase B sweep stats

- **Nominal grid:** 12 MmaTiles × 4 schedules × 2 schedulers = 96 configs
- **Compile-legal after pre-skip:** 20 configs (see `gen_configs.py` skip reasons)
- **Microbench dataset:** 4 shapes × 10 M values × 21 configs = **840 rows** (`microbench.csv`)
- **Shortlisted for E2E:** top-3 per (shape × M-bucket), deduped → **12 configs** (`shortlist.json`)
- **E2E validation:** **12/12** configs passed GSM8K 8/8 (zero correctness regressions), 12/12 traces captured
- **E2E kernel rollup:** 949 per-(config × kernel_symbol) rows in `e2e_kernels.csv`
- **Total sweep wall time:** Phase A ~2 hr + Phase B ~6 hr = **~8 hr**

## Per-(shape × M-bucket) winners (microbench-ranked, E2E-validated)

Top-1 microbench config per bucket. Each entry is also in the per-bucket top-3
of `shortlist.json` and has `gsm8k_sanity/*.json` = 8/8 → green-lit for the
future dispatch table.

| Shape | M=1-8 | M=16-32 | M=64-128 | M=192-256 |
|---|---|---|---|---|
| `qkv_proj` | `Cfg_128x256x128_Auto_Pers` (66.34 μs) | `Cfg_128x256x128_Auto_Pers` (63.65 μs) | `smoke_M256` (43.26 μs) | `Cfg_128x128x128_TmaWSPing_Pers` (80.86 μs) |
| `o_proj` | `Cfg_128x256x128_TmaWSCoop_Pers` (56.16 μs) | `Cfg_128x256x128_Auto_Pers` (52.61 μs) | `Cfg_128x128x256_TmaWSPing_Pers` (36.00 μs) | `Cfg_256x128x128_TmaWSCoop_Pers` (47.52 μs) |
| `gate_up_proj` | `Cfg_128x128x256_Auto_Pers` (430.43 μs) | `Cfg_128x128x256_Auto_Pers` (444.67 μs) | `Cfg_128x128x256_TmaWSCoop_Pers` (463.62 μs) | `Cfg_128x128x256_Auto_Pers` (500.83 μs) |
| `down_proj` | `Cfg_128x128x256_TmaWSCoop_Pers` (227.42 μs) | `Cfg_128x128x256_Auto_Pers` (229.31 μs) | `Cfg_128x128x256_Auto_Pers` (232.64 μs) | `Cfg_128x128x256_Auto_Pers` (247.84 μs) |

(Cell entries: top-1 config from shortlist + min μs from microbench.)

## E2E aggregate winner

Ranked by aggregate `total_ms` of all `BlockScaled` NVFP4 GEMM kernels across the
fixed 60-iteration, concurrency-4 workload (profiler-active iterations = 600;
see notebook for the per-kernel symbol decomposition). `smoke_M256` idx=11 is
the Tuneable-template self-check (same tile as production M256 config):

| Rank | Config | Total ms (all FP4) | Δ vs smoke_M256 |
|---|---|---|---|
| 1 | `Cfg_128x128x128_TmaWSCoop_Pers` | 97,698.6 | **−0.11%** |
| 2 | `Cfg_256x128x128_TmaWSCoop_Pers` | 97,746.8 | −0.06% |
| 3 | `smoke_M256` (self-check) | 97,801.7 | ±0.00% |
| 4 | `Cfg_128x128x256_TmaWSCoop_Pers` | 97,900.2 | +0.10% |
| 5 | `Cfg_128x256x128_Auto_Pers` | 97,933.8 | +0.14% |
| 6 | `Cfg_128x256x128_TmaWSCoop_Pers` | 98,041.2 | +0.24% |
| 7 | `Cfg_256x128x128_Auto_Pers` | 98,084.3 | +0.29% |
| 8 | `Cfg_128x128x256_Auto_Pers` | 98,167.7 | +0.37% |
| 9 | `Cfg_128x256x128_TmaWSCoop_SK` | 98,172.4 | +0.38% |
| 10 | `Cfg_128x128x256_TmaWSPing_Pers` | 98,181.4 | +0.39% |
| 11 | `Cfg_128x128x256_TmaWSCoop_SK` | 98,269.3 | +0.48% |
| 12 | `Cfg_128x128x128_TmaWSPing_Pers` | 98,310.7 | +0.52% |

**Tuneable-self-check:** `smoke_M256` idx=11 routes through
`Fp4GemmSm120Tuneable<ShortlistCfg_smoke, OutType>` with tile `<128,128,128>` +
`KernelScheduleAuto` + persistent scheduler — byte-identical instantiation to
production `sm120_fp4_config_M256`. It lands **within ±0.11%** of the adjacent
`Cfg_128x128x128_TmaWSCoop_Pers` (delta driven by scheduler choice, not
template indirection). **Conclusion:** the Tuneable template introduces zero
material overhead vs the production template — safe to carry forward.

**Caveat on small deltas.** The E2E workload is decode-dominated (M=1 per
request × concurrency 4 → Stream-K M≤16 path carries ~99% of NVFP4 GEMM calls).
The M=17–256 mid-batch band these configs exercise is rare in this workload;
microbench (above) is the authoritative signal for the mid-batch dispatch
choice. All 12 configs landing within ±1% is **evidence of the Tuneable
dispatcher not regressing**, not evidence that tile choice doesn't matter.

## GSM8K correctness (all configs)

Every shortlisted config: **8/8 PASS** on the sanity gate
(`gsm8k_sanity/<config>.json`). No silent gibberish producers; any of the 12 is
a viable candidate for the final dispatch table.

## Handoff — future heuristic session

- **Input dataset:**
  - `microbench.csv` — per-(shape × M × config) min μs (840 rows)
  - `shortlist.json` — top-3 per (shape × M-bucket) → 12 unique configs
  - `e2e_kernels.csv` — per-(config × kernel_symbol) aggregate μs from live traces (949 rows)
  - `per_trace_kernels/<config>.csv` — per-config kernel breakdowns (12 files)
  - `e2e_results.json` — per-config status + elapsed (12/12 `ok`)
  - `gsm8k_sanity/<config>.{json,txt}` — correctness gate logs (12/12 8/8)
  - `decode_logs/<config>.txt` — per-config container logs
  - [`gemm_microbench_analysis.ipynb`](../../../../docs/research/gemm_sweep/gemm_microbench_analysis.ipynb),
    [`gemm_e2e_verdict.ipynb`](../../../../docs/research/gemm_sweep/gemm_e2e_verdict.ipynb) — rendered analysis
- **Task for that session:** replace the `NVLLM_FP4_GEMM_CONFIG_M256` env-var
  with a compile-time `(shape, M-bucket) → ShortlistCfg_<idx>` lookup table
  using the winners table above. The `Fp4GemmSm120Tuneable` template and
  `try_run_shortlist_config<OutType>` header machinery are already shipped
  (commit `c83583721`); the dispatch site just needs the static table. Ship as
  the "Qwen3.5-27B driver update" (first model-specific tuning pack). Then
  `gemm-sweep` skill (`~/.claude/skills/gemm-sweep/SKILL.md`) runs for the
  next model.

## How to reproduce

```bash
# Phase A (Stream-K × CUDA graphs verdict, ~2 hr incl rebuilds)
bash docs/research/gemm_sweep/capture_phase_a_trace.sh streamk
bash docs/research/gemm_sweep/capture_phase_a_trace.sh baseline

# Phase B.1 microbench (~3 min)
.venv/bin/python docs/research/gemm_sweep/run_sweep.py \
    --config-json "$HOME/.cache/huggingface/hub/models--ig1--Qwen3.5-27B-NVFP4/snapshots/*/config.json" \
    --output benchmarks/nvllm/traces/gemm_sweep_sm120/2026-04-21-qwen35-27b/microbench.csv

# Phase B.2.1 shortlist
.venv/bin/python -m jupyter nbconvert --to notebook --execute --inplace \
    docs/research/gemm_sweep/gemm_microbench_analysis.ipynb
.venv/bin/python docs/research/gemm_sweep/gen_shortlist_header.py

# Phase B.2.2 rebuild with runtime dispatcher (~50 min in tmux)
docker build -f docker/Dockerfile.gb10 -t nvllm:gb10 .

# Phase B.2.3 per-config E2E traces (~3 hr, 12 × ~15 min)
.venv/bin/python docs/research/gemm_sweep/run_e2e_traces.py

# Phase B.3.1 verdict extraction
OUT=benchmarks/nvllm/traces/gemm_sweep_sm120/2026-04-21-qwen35-27b
mkdir -p "$OUT/per_trace_kernels"
echo "config_id,kernel_symbol,n_calls,mean_us,p50_us,p95_us,total_ms" > "$OUT/e2e_kernels.csv"
for trace in "$OUT/e2e"/*.pt.trace.json.gz; do
  cfg=$(basename "$trace" .pt.trace.json.gz)
  .venv/bin/python docs/research/gemm_sweep/extract_e2e_kernels.py \
    --trace "$trace" --config "$cfg" --out "$OUT/per_trace_kernels/$cfg.csv"
  tail -n +2 "$OUT/per_trace_kernels/$cfg.csv" >> "$OUT/e2e_kernels.csv"
done
.venv/bin/python docs/research/gemm_sweep/gen_verdict_notebook.py
.venv/bin/python -m jupyter nbconvert --to notebook --execute --inplace \
    docs/research/gemm_sweep/gemm_e2e_verdict.ipynb
```

## Files

| File | Purpose |
|---|---|
| `microbench.csv` | B.1 full sweep dataset (840 rows) |
| `shortlist.json` | top-3 per (shape × M-bucket) → 12 unique configs |
| `e2e/*.pt.trace.json.gz` | B.2.3 raw profiler traces (local-only, gitignored; ~2.1 GB total) |
| `e2e_kernels.csv` | B.3.1 per-(config × kernel_symbol) aggregate μs (949 rows) |
| `per_trace_kernels/<config>.csv` | per-trace kernel breakdown (12 files) |
| `e2e_results.json` | per-config status + elapsed (12 entries, all `ok`) |
| `gsm8k_sanity/<config>.{json,txt}` | correctness gate logs (12 configs, 12/12 8/8) |
| `decode_logs/<config>.txt` | per-config container logs |
| `e2e_primary_total_ms.png` | B.3.1 verdict bar chart — primary FP4 GEMM total_ms per config |
| `e2e_primary_mean_us.png` | B.3.1 verdict bar chart — primary FP4 GEMM mean μs per config |
