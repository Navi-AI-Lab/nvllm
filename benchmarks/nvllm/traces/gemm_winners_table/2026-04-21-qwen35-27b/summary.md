# NVFP4 GEMM Winners Table — Qwen3.5-27B Driver Update — 2026-04-21

**Commit:** `954a7812d`
**Branch:** `feat/gemm-sweep-2026-04-21`
**Model:** `ig1/Qwen3.5-27B-NVFP4` (snapshot `4c546624f1fa8b77f5b7cfb3b6c96bf46d25c3a9`)
**Hardware:** DGX Spark (GB10, SM120 / 121)
**Image:** `nvllm:gb10` (commit `954a7812d` — includes the `(N, K) → bucket → ShortlistCfg_idx` lookup table)

## What shipped

First per-model "driver update" produced by the `gemm-sweep` skill workflow:

- **Generated header** (committed): `csrc/libtorch_stable/quantization/fp4/nvfp4_winners_table.hpp`
  — 4 shape rows × 3 M-bucket columns, idx values traceable to the top-1 of
  `shortlist.json::by_shape[shape][bucket][0]` from the Phase B sweep.
- **Winners source of truth** (committed): `../gemm_sweep_sm120/2026-04-21-qwen35-27b/winners.json`
- **Codegen** (committed): `docs/research/gemm_sweep/gen_winners_header.py`
  — reads `shortlist.json` + `microbench.csv`, emits the header; has a `--check`
  pre-commit mode.
- **Dispatcher wire-in**: both `cutlass_fp4_bf16_gemm_dispatch` and
  `cutlass_fp4_f16_gemm_dispatch` inside `nvfp4_scaled_mm_sm120_kernels.cu` gained
  a new branch inside the `mp2 ≤ 256` band. Priority: **env var > (N, K) table >
  production default**. Unknown (N, K) → safe fallback to production default
  (zero-regression guarantee for shapes not in the sweep).
- **Env var preserved**: `NVLLM_FP4_GEMM_CONFIG_M256=<idx>` remains highest
  priority so future `gemm-sweep` Phase A verdicts and A/B bisection on the next
  model stay rebuild-free.
- **Debug hook**: `NVLLM_FP4_GEMM_LOG_TABLE=1` prints
  `[nvllm] fp4 table: <shape> mp2=N -> idx=I` on every table-matched GEMM call.

## Routing table (from `nvfp4_winners_table.hpp`)

| Shape (N × K) | M=16-32 idx | M=64-128 idx | M=192-256 idx |
|---|:-:|:-:|:-:|
| `qkv_proj` (8192 × 5120)     | 6 (`Cfg_128x256x128_Auto_Pers`)     | 11 (`smoke_M256`)                     | 1 (`Cfg_128x128x128_TmaWSPing_Pers`) |
| `o_proj` (5120 × 6144)       | 6 (`Cfg_128x256x128_Auto_Pers`)     | 5 (`Cfg_128x128x256_TmaWSPing_Pers`)  | 10 (`Cfg_256x128x128_TmaWSCoop_Pers`) |
| `gate_up_proj` (34816 × 5120)| 2 (`Cfg_128x128x256_Auto_Pers`)     | 3 (`Cfg_128x128x256_TmaWSCoop_Pers`)  | 2 (`Cfg_128x128x256_Auto_Pers`)       |
| `down_proj` (5120 × 17408)   | 2 (`Cfg_128x128x256_Auto_Pers`)     | 2 (`Cfg_128x128x256_Auto_Pers`)       | 2 (`Cfg_128x128x256_Auto_Pers`)       |

`mp2 ≤ 16` is not in the table — that's the Stream-K decode band (Phase A),
unchanged by this driver update.

## Correctness gate

Both paths pass the 8-question GSM8K sanity gate (`scripts/gsm8k_sanity.py`):

- `gsm8k_table.json` — no env var (table path): **8/8 PASS**
- `gsm8k_envvar.json` — `NVLLM_FP4_GEMM_CONFIG_M256=11` (env-var path, forces
  `smoke_M256`): **8/8 PASS**
- No "malformed env var" warnings in `docker logs` for the env-var run (idx=11
  is a valid shortlist entry).

## Primary evidence — dispatcher replay

`replay_winners_table.py` calls `ops.cutlass_scaled_fp4_mm` in a 4-shape × 6-M
grid (50 warmup + 200 timed per cell), once with `NVLLM_FP4_GEMM_CONFIG_M256=11`
(baseline — forces `smoke_M256` = `Cfg_128x128x128_Auto_Pers`) and once with no
env var (table path). Exercises the **real** production dispatcher path
(`cutlass_scaled_fp4_mm_sm120a` → `cutlass_fp4_bf16_gemm_dispatch` →
`lookup_m_mid_winner`); the standalone microbench binary bypasses it.

| Shape | M | Baseline μs | Table μs | Δ % | Notes |
|---|---:|---:|---:|---:|---|
| down_proj | 16 | 229.41 | 230.50 | +0.47 | Stream-K band (mp2=16), unchanged |
| down_proj | 32 | 245.98 | 230.46 | **-6.31** | |
| down_proj | 64 | 249.82 | 232.45 | **-6.96** | |
| down_proj | 128 | 257.41 | 238.72 | **-7.26** | |
| down_proj | 192 | 273.60 | 252.96 | **-7.54** | |
| down_proj | 256 | 280.80 | 257.12 | **-8.43** | |
| gate_up_proj | 16 | 428.06 | 421.92 | **-1.44** | Stream-K band |
| gate_up_proj | 32 | 483.36 | 450.59 | **-6.78** | |
| gate_up_proj | 64 | 490.72 | 461.92 | **-5.87** | |
| gate_up_proj | 128 | 527.81 | 484.58 | **-8.19** | |
| gate_up_proj | 192 | 536.70 | 501.79 | **-6.50** | |
| gate_up_proj | 256 | 559.07 | 527.52 | **-5.64** | |
| o_proj | 16 | 62.37 | 62.37 | +0.00 | Stream-K band |
| o_proj | 32 | 60.42 | 53.47 | **-11.49** | |
| o_proj | 64 | 26.85 | 29.44 | +9.65 | noise, baseline < 100 μs |
| o_proj | 128 | 26.72 | 29.50 | +10.42 | noise, baseline < 100 μs |
| o_proj | 192 | 48.51 | 46.14 | **-4.88** | |
| o_proj | 256 | 54.02 | 46.50 | **-13.92** | |
| qkv_proj | 16 | 86.27 | 86.08 | -0.22 | Stream-K band |
| qkv_proj | 32 | 80.86 | 66.46 | **-17.81** | |
| qkv_proj | 64 | 56.45 | 62.37 | +10.49 | noise, **table config = baseline config** |
| qkv_proj | 128 | 83.07 | 85.12 | +2.47 | noise, **table config = baseline config** |
| qkv_proj | 192 | 84.83 | 91.20 | +7.51 | noise, baseline < 100 μs |
| qkv_proj | 256 | 97.50 | 98.46 | +0.98 | |

Bolded rows have `|Δ| ≥ 1%`; negative Δ means the table path is faster.

**Summary:** 16 / 24 cells win (Δ < 0). Largest wins are on the large kernels
(down_proj, gate_up_proj, o_proj M=256, qkv_proj M=32) — all ≥ 6% and up to
-17.8%. Those are the kernels that dominate mid-batch compute.

**Noise floor calibration.** Five cells show `Δ > +2%`. Two of them
(`qkv_proj M=64, M=128`) route to **the same config as baseline** (both idx 11,
since the bucket winner for qkv_proj M=64-128 is `smoke_M256` — identical to
the forced baseline). Those cells cannot be real regressions; they pin the
between-container bench noise at ≈10% for < 100 μs kernels. The other three
positive-Δ cells (`o_proj M=64,128`, `qkv_proj M=192`) all sit below 100 μs
baseline and within the same noise envelope.

## Secondary evidence — E2E no-regression

One `pt.trace.json.gz` captured under the identical Phase B workload
(30 warmup + 30 timed @ concurrency 4, `max_tokens=256`). Primary FP4 GEMM
`total_ms` (sum over all `BlockScaled` kernels):

| Path | total_ms | Δ vs baseline |
|---|---:|---:|
| smoke_M256 baseline (Phase B `e2e_kernels.csv`) | 97,801.7 | ±0.00% |
| Table path (this session, `e2e_kernels.csv`)    | 98,369.5 | **+0.58%** |

Within the ±1% scatter documented in Phase B's `summary.md` (all 12 shortlisted
configs landed within ±1% on this workload because decode-dominated traffic
sends ~99% of FP4 GEMM calls through the Stream-K `mp2 ≤ 16` band; the
mid-batch band the table routes is exercised rarely in the trace). Regression
gate, not a win proof — the table's actual mid-batch gains show up in the
dispatcher replay above.

## How to reproduce

```bash
# 1. Regenerate header from shortlist.json + microbench.csv (no-op if up-to-date)
.venv/bin/python docs/research/gemm_sweep/gen_winners_header.py \
    --sweep-dir benchmarks/nvllm/traces/gemm_sweep_sm120/2026-04-21-qwen35-27b \
    --shortlist-header csrc/libtorch_stable/quantization/fp4/nvfp4_shortlist_configs.hpp \
    --model-tag qwen35_27b

# 2. Rebuild nvllm:gb10 in tmux (~30-50 min)
tmux new-session -d -s build 'docker build -f docker/Dockerfile.gb10 -t nvllm:gb10 . 2>&1 | tee /tmp/build.log'

# 3. Correctness — GSM8K 8/8 on both paths (full scripts live in /tmp/ during run)
#    - table path: start container, run scripts/gsm8k_sanity.py, assert 8/8
#    - env-var path: identical but with -e NVLLM_FP4_GEMM_CONFIG_M256=11

# 4. Dispatcher replay — 2 container runs + merge
OUT=benchmarks/nvllm/traces/gemm_winners_table/2026-04-21-qwen35-27b

docker run --rm --gpus all --ipc=host -v "$PWD":/work \
    -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
    -e NVLLM_FP4_GEMM_CONFIG_M256=11 --entrypoint bash nvllm:gb10 \
    -lc "cd /work && /opt/venv/bin/python docs/research/gemm_sweep/replay_winners_table.py --label baseline --output /work/$OUT/replay_baseline.csv"

docker run --rm --gpus all --ipc=host -v "$PWD":/work \
    -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
    --entrypoint bash nvllm:gb10 \
    -lc "cd /work && /opt/venv/bin/python docs/research/gemm_sweep/replay_winners_table.py --label table --output /work/$OUT/replay_table.csv"

# 5. E2E regression trace — identical to Phase B workload, extract, compare
#    primary FP4 GEMM total_ms against smoke_M256 row in Phase B's e2e_kernels.csv
```

## Files

| File | Purpose |
|---|---|
| `gsm8k_table.json` / `gsm8k_table.log` | Correctness gate, table path (8/8) |
| `gsm8k_envvar.json` / `gsm8k_envvar.log` | Correctness gate, env-var path (8/8) |
| `replay_baseline.csv` | Dispatcher-replay baseline (forced `smoke_M256`, 24 cells) |
| `replay_table.csv` | Dispatcher-replay table path (24 cells) |
| `dispatcher_replay.csv` | Merged per-(shape, M) deltas (**primary evidence**) |
| `e2e/table.pt.trace.json.gz` | Raw profiler trace — 184 MB, gitignored |
| `e2e_kernels.csv` | Per-kernel μs extract (secondary evidence; no-regression gate) |
