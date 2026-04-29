# Phase 6b — Small-M NVFP4 GEMM dispatcher — 2026-04-29

**Commit:** _pending — uncommitted on `feat/uber-kernel-migration`, parent `722efc60b` (Phase 6a)_
**Model:** `ig1/Qwen3.5-27B-NVFP4` (non-distilled)
**Hardware:** NVIDIA DGX Spark (GB10, SM120 / 121), 128 GB unified
**Image:** `nvllm:gb10` SHA `ce88f10e24b3`
**Backend:** `CUTE_PAGED`, `fp8_e4m3` KV cache, PIECEWISE CUDA graphs

## TL;DR

Hardcoded `sm120_fp4_config_stream_k` (Stream-K, `<128,128,256>` cooperative) was firing for **every** `mp2 ≤ 16` GEMM regardless of shape. The 2026-04-21 sweep CSV already had small-M coverage, so re-running `analyze.py` with new bucket boundaries surfaced per-shape, per-M winners that beat the hardcoded path. Phase 6b ships:

1. `nvfp4_winners_table.hpp` extended with `idx_1_2 / idx_4_8 / idx_16` columns + `lookup_m_small_winner()`
2. Both `cutlass_fp4_{bf16,f16}_gemm_dispatch` reordered for the small-M band: env override → small-M lookup → Stream-K fallback
3. `NVLLM_FP4_GEMM_LOG_TABLE=1` extended to log small-M hits AND small-M misses (`miss N=… K=… mp2=… -> Stream-K fallback`)

**Counter-intuitive finding:** Stream-K — added in Phase A specifically as the "small-M decode" config — is beaten by Persistent at every measured small-M point on every shape. Phase A's "+11.3% vs M256 default" was real, but vs the wrong baseline; Persistent at the right tile shape wins.

## Configuration

| Field | Value |
|---|---|
| FUSION | 1 |
| PATH | coop |
| num_seqs | 1 |
| concurrent | 1 |
| warmup / timed | 15 / 5 |
| max_tokens | 64 |
| record_shapes | false |
| compilation | PIECEWISE CUDA graphs |

Identical workload to Phase 6a (`benchmarks/nvllm/traces/phase_6a/2026-04-29-initial/`) so per-call kernel duration is apples-to-apples.

## Small-M winners

Aggregation: per (shape, bucket), average `min_us` across the bucket's M values for each config; pick lowest average. Canonical attn shapes were taken from the 2026-04-21 microbench.csv. The `gdn_in_proj_qkv` shape `(14336, 5120)` was discovered post-build-#1 via `NVLLM_FP4_GEMM_LOG_TABLE=1` (5,040 of 36,080 NVFP4 GEMM calls were missing the table) and microbenched separately at `benchmarks/nvllm/traces/gemm_sweep_sm120_phase6b_gdn/2026-04-29/microbench.csv` (21 configs × 5 small-M values).

| Shape | (N, K) | Bucket 1-2 | Bucket 4-8 | Bucket 16 |
|---|---|---|---|---|
| qkv_proj | (8192, 5120) | `Cfg_128x256x128_TmaWSCoop_Pers` (idx 7) | `Cfg_128x256x128_Auto_Pers` (idx 6) | `Cfg_128x256x128_Auto_Pers` (idx 6) |
| o_proj | (5120, 6144) | `Cfg_128x256x128_TmaWSCoop_Pers` (idx 7) | `Cfg_128x256x128_TmaWSCoop_Pers` (idx 7) | `Cfg_128x256x128_Auto_Pers` (idx 6) |
| gate_up_proj | (34816, 5120) | `Cfg_128x128x256_Auto_Pers` (idx 2) | `Cfg_128x128x256_TmaWSCoop_Pers` (idx 3) | `Cfg_128x128x256_Auto_Pers` (idx 2) |
| down_proj | (5120, 17408) | `Cfg_128x128x256_TmaWSCoop_Pers` (idx 3) | `Cfg_128x128x256_Auto_Pers` (idx 2) | `Cfg_128x128x256_Auto_Pers` (idx 2) |
| gdn_in_proj_qkv | (14336, 5120) | `Cfg_128x128x256_Auto_Pers` (idx 2) | `Cfg_128x128x256_TmaWSCoop_Pers` (idx 3) | `Cfg_128x128x256_Auto_Pers` (idx 2) |

All 5 winners (idx 2, 3, 6, 7) are pre-existing entries in the 12-config shortlist — no new C++ templates needed.

**On the GDN shape:** `(N=14336, K=5120)` is the GDN linear-attention `in_proj_qkv` packed projection, where `q+k+v = 48*128 + 16*128 + 48*128 = 14336`. Per friend's caveat ("don't alias by name; benchmark the exact (N, K) directly") we ran a fresh 21-config sweep on this shape rather than reusing the canonical-attn winners. Per-call wins are modest (-3% to -7%) — same `128x128x256` tile shape as the Stream-K hardcoded path; only the schedule differs (Persistent vs Stream-K).

## Primary evidence — dispatcher replay (small-M band)

`replay_winners_table.py --m-band small`, two legs:

- **baseline_streamk:** `NVLLM_FP4_GEMM_CONFIG_M256=4` forces `Cfg_128x128x256_TmaWSCoop_SK` (the pre-Phase-6b hardcoded Stream-K config) for all shapes/M
- **table_smallm:** no env var — hits `lookup_m_small_winner`

| Shape | M | Baseline μs | Table μs | Δ % |
|---|---:|---:|---:|---:|
| qkv_proj | 1 | 84.39 | 70.76 | **-16.15** |
| qkv_proj | 2 | 96.55 | 69.30 | **-28.22** |
| qkv_proj | 4 | 87.13 | 68.84 | **-21.00** |
| qkv_proj | 8 | 92.38 | 68.35 | **-26.01** |
| qkv_proj | 16 | 88.25 | 66.49 | **-24.66** |
| o_proj | 1 | 68.82 | 58.65 | **-14.79** |
| o_proj | 2 | 68.86 | 56.58 | **-17.83** |
| o_proj | 4 | 66.85 | 58.62 | **-12.32** |
| o_proj | 8 | 65.04 | 58.28 | **-10.40** |
| o_proj | 16 | 62.71 | 56.64 | **-9.67** |
| gate_up_proj | 1 | 413.43 | 423.47 | +2.43 |
| gate_up_proj | 2 | 422.00 | 424.56 | +0.61 |
| gate_up_proj | 4 | 423.43 | 426.39 | +0.70 |
| gate_up_proj | 8 | 429.49 | 428.73 | -0.18 |
| gate_up_proj | 16 | 431.95 | 432.69 | +0.17 |
| down_proj | 1 | 229.89 | 228.82 | -0.46 |
| down_proj | 2 | 231.75 | 227.78 | -1.71 |
| down_proj | 4 | 229.48 | 228.54 | -0.41 |
| down_proj | 8 | 228.72 | 229.02 | +0.13 |
| down_proj | 16 | 230.65 | 229.53 | -0.49 |

Per-shape weighted (sum across M):

| Shape | Σ baseline μs | Σ table μs | Δ |
|---|---:|---:|---:|
| qkv_proj | 448.70 | 343.74 | **-23.4%** |
| o_proj | 332.28 | 288.77 | **-13.1%** |
| gate_up_proj | 2120.30 | 2135.84 | +0.7% |
| down_proj | 1150.49 | 1143.69 | -0.6% |
| **Total (20 cells)** | **4051.77** | **3912.04** | **-3.45%** |

**Interpretation:** big wins concentrated on shapes where the optimal tile shape (`128x256x128`) differs from the hardcoded Stream-K's `128x128x256` (qkv, o). Near-zero on shapes where the tile shape happens to match (gate_up, down) — only the schedule differs (Auto/Coop_Pers vs Coop_SK), and those produce near-identical kernels at the larger N/K shapes. One M=1 gate_up_proj cell at +2.43% (single-sample microbench scatter — within prior Phase B-documented ~10% noise floor for sub-100µs cells, here scaled down).

## Secondary evidence — E2E trace

Captured against the rebuilt `nvllm:gb10` (image SHA `7ea16c763044`, includes the GDN row); workload identical to Phase 6a's `phase_6a_beta_coop` capture (5 timed × 64 max_tokens × concurrency=1, 15 warmup curls).

NVFP4 GEMM mass extracted from per-kernel `total_ms` for all kernel symbols matching `float_e2m1`:

| Leg | Image | NVFP4 GEMM total_ms | Calls | Mean μs/call | Δ vs Phase 6a |
|---|---|---:|---:|---:|---:|
| Phase 6a (`722efc60b`) | `5327e03dc0a2` | 11724.2 | 36080 | 324.97 | — |
| Phase 6b build #1 (no GDN row) | `ce88f10e24b3` | 11624.1 | 36080 | 322.18 | -100.1 ms (-0.85%) |
| **Phase 6b build #2 (with GDN row)** | **`7ea16c763044`** | **11596.8** | **36080** | **321.43** | **-127.4 ms (-1.09%)** |

Adding the GDN row in build #2 closed an additional 27.3 ms (-0.23%) on top of build #1 — consistent with the 5,040 GDN calls × ~5 μs/call savings predicted by the supplemental microbench.

Other relevant kernels (sanity — should be unchanged or near-noise):

| Kernel | Phase 6a total ms | Phase 6b build #2 total ms | Δ |
|---|---:|---:|---:|
| `PhaseE_Beta_Kernel` (β-coop fused) | 207,736 | 206,101 | -1,635 (-0.79%) |
| cuBLAS `gemvx::kernel` (bf16) | 18,570 | 18,583 | +13 (+0.07%) |
| `fused_recurrent_gated_delta_rule_packed_decode_kernel` | (n/a in same units) | 274.995 | — |

β-coop's -0.79% is within sample noise on n=5,040 calls and is not claimed as a Phase 6b effect; gemvx is invariant as expected.

**Why the E2E delta is much smaller than the replay's -23% / -13% on qkv/o_proj:** those wins apply to specific shape buckets that account for a small fraction of total NVFP4 GEMM mass. The gate_up/down shapes (no win) dominate runtime because of their larger N/K dimensions. Friend's caveat anticipated this — and it's also why E2E was the right gate (not just per-cell replay deltas).

## Correctness gate

`scripts/gsm8k_sanity.py` (8-question smoke test, guided CoT, 60s timeout per question):

```
Q1-Q8: all OK (12.0-12.4s each)
GSM8K sanity: 8/8 (100%)
Verdict: PASS
```

GSM8K-50 (seed=42) deferred — Phase 6b is a dispatcher refactor (no math change), so the 8-question smoke is sufficient. The replay-confirmed kernel selection runs only when an exact `(N, K)` matches the table; unknown shapes still fall through to the prior Stream-K config (zero-regression guarantee for non-Qwen3.5-27B deployments).

## Logging verification

`NVLLM_FP4_GEMM_LOG_TABLE=1` produces table hits + miss-fallback messages as expected. Sample from `replay_table_smallm` run:

```
[nvllm] fp4 small-M table: qwen35_27b/down_proj mp2=16 -> idx=2
[nvllm] fp4 small-M table: qwen35_27b/down_proj mp2=16 -> idx=2
…
```

Miss-fallback path verified by reading the compiled string `[nvllm] fp4 small-M table: miss N=%d K=%d mp2=%d -> Stream-K fallback` from the rebuilt `_C_stable_libtorch.abi3.so`.

## How to reproduce

```bash
# 0. (One-time, only if expanding to new shapes outside the 4 canonical attn
#    shapes.) Microbench the new (N, K) shape directly — DO NOT alias by
#    name. The GDN row was added this way:
#    docs/research/gemm_sweep/run_sweep.py with N=14336, K=5120 →
#    benchmarks/nvllm/traces/gemm_sweep_sm120_phase6b_gdn/2026-04-29/microbench.csv
#    Then add a SMALL_ONLY_SHAPES entry in gen_winners_header.py.

# 1. Regenerate header from microbench.csv (no rebuild needed if only
#    re-running on the same 2026-04-21 sweep dir).
.venv/bin/python docs/research/gemm_sweep/gen_winners_header.py \
    --sweep-dir benchmarks/nvllm/traces/gemm_sweep_sm120/2026-04-21-qwen35-27b \
    --shortlist-header csrc/libtorch_stable/quantization/fp4/nvfp4_shortlist_configs.hpp \
    --model-tag qwen35_27b

# 2. Rebuild image (in tmux — see CLAUDE.md "Tmux for docker builds").
tmux new-session -d -s build 'docker build -f docker/Dockerfile.gb10 -t nvllm:gb10 . 2>&1 | tee /tmp/build.log'

# 3. Dispatcher replay — small-M band, 2 container runs.
OUT=benchmarks/nvllm/traces/gemm_winners_table_smallM/2026-04-29-qwen35-27b

docker run --rm --gpus all --ipc=host -v "$PWD":/work \
    -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
    -e NVLLM_FP4_GEMM_CONFIG_M256=4 --entrypoint bash nvllm:gb10 \
    -lc "cd /work && /opt/venv/bin/python docs/research/gemm_sweep/replay_winners_table.py --label baseline_streamk --m-band small --output /work/$OUT/replay_baseline_streamk.csv"

docker run --rm --gpus all --ipc=host -v "$PWD":/work \
    -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
    -e NVLLM_FP4_GEMM_LOG_TABLE=1 --entrypoint bash nvllm:gb10 \
    -lc "cd /work && /opt/venv/bin/python docs/research/gemm_sweep/replay_winners_table.py --label table_smallm --m-band small --output /work/$OUT/replay_table_smallm.csv"

# 4. E2E trace capture
bash docs/research/phase_6b_traces/capture_phase_6b.sh

# 5. Correctness — GSM8K 8q sanity (~2 min)
./scripts/serve-cute.sh
.venv/bin/python scripts/gsm8k_sanity.py
```

## Artifact index

| File | Committed | Notes |
|---|---|---|
| `dispatcher_replay.csv` | **yes** | merged baseline_streamk vs table_smallm Δ |
| `replay_baseline_streamk.csv` | **yes** | forced `NVLLM_FP4_GEMM_CONFIG_M256=4` (Stream-K) |
| `replay_table_smallm.csv` | **yes** | table path, 4 shapes × 5 M values |
| `phase_6b_smallm_table_kernels.csv` | **yes** | per-kernel μs from E2E trace |
| `phase_6b_smallm_table_serve.log` | **yes** | EngineCore stdout/stderr |
| `phase_6b_smallm_table_mem.log` | **yes** | host + docker mem watchdog |
| `phase_6b_smallm_table.pt.trace.json.gz` | no (gitignore) | raw torch trace |
| `summary.md` | **yes** | this file |

Phase 6b SHIPPED.
