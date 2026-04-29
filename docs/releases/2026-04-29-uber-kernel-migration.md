# Release — Uber-Kernel Migration

**Date:** 2026-04-29
**Branch:** `feat/uber-kernel-migration`
**Merge base (main):** [`76b88ba21`](https://github.com/Navi-AI-Lab/nvllm/commit/76b88ba2165d74d1665b60eaeeab933958f0fd18)
**Branch tip:** [`1f91013b8`](https://github.com/Navi-AI-Lab/nvllm/commit/1f91013b8432f01d5bc3cddfbd401a2d4d1cf0ea)
**Diffstat:** 50 files changed, +10,714 / −589
**Hardware target:** NVIDIA DGX Spark (GB10, SM120 / 121), 128 GB unified
**Model under test:** `ig1/Qwen3.5-27B-NVFP4`

---

## What this release contains

The β-coop "uber" kernel ([`PhaseE_Beta_Kernel`](https://github.com/Navi-AI-Lab/nvllm/blob/1f91013b8432f01d5bc3cddfbd401a2d4d1cf0ea/vllm/v1/attention/backends/cute_paged/phase_e_kernel.py)) — a single cooperative-launch CuTe DSL kernel that subsumes per-full-attention-layer decode work (Phase A attention + Phase B W_O + Phase C post-attention RMSNorm + Phase E MLP) — was already present on `main` at [`bc9037955`](https://github.com/Navi-AI-Lab/nvllm/commit/bc9037955) (Phase E ship, 2026-04-23). It compiled, captured under PIECEWISE CUDA graphs, and produced coherent output in the smoke harness, but the captured FX graph still ran the legacy split path because β-coop's outputs were structurally unobservable to graph capture (consume-gate DCE, [findings 2026-04-26](https://github.com/Navi-AI-Lab/nvllm/blob/1f91013b8432f01d5bc3cddfbd401a2d4d1cf0ea/docs/research/uber_kernel_migration/2026-04-26-consume-gate-dce-and-graph-capture.md)).

This branch brings β-coop into production as the actual decode path for full-attention layers under PIECEWISE CUDA graphs, then layers on three rounds of perf polish on top.

---

## Commits (oldest → newest)

| # | Hash | Subject |
|---|---|---|
| 1 | [`2b21f3450`](https://github.com/Navi-AI-Lab/nvllm/commit/2b21f3450) | chore(serve): bake flashinfer-autotune-off flag into serve-cute.sh |
| 2 | [`a65bcef31`](https://github.com/Navi-AI-Lab/nvllm/commit/a65bcef31) | fix(cute): C1 — β-coop and β-lite read residual_buf, not residual_output |
| 3 | [`54da780f3`](https://github.com/Navi-AI-Lab/nvllm/commit/54da780f3) | refactor(cute): C1.5 — delete Phase 4 + F.1 layer-LN bake plumbing |
| 4 | [`5a0311ca3`](https://github.com/Navi-AI-Lab/nvllm/commit/5a0311ca3) | fix(cute): C2 plumbing — residual/gate mirror op + β-coop predicate hard-gate |
| 5 | [`514b88c6f`](https://github.com/Navi-AI-Lab/nvllm/commit/514b88c6f) | wip(cute): B-fix attempt — consume-gate DCE + post-attn-LN dispatch ops *(reverted in #6, kept in history for the architectural-pass reference)* |
| 6 | [`3ffcf8740`](https://github.com/Navi-AI-Lab/nvllm/commit/3ffcf8740) | Revert "wip(cute): B-fix attempt" |
| 7 | [`90b06d5df`](https://github.com/Navi-AI-Lab/nvllm/commit/90b06d5df) | docs(uber-kernel): consume-gate DCE + graph-capture findings (2026-04-26) |
| 8 | [`788697bff`](https://github.com/Navi-AI-Lab/nvllm/commit/788697bff) | docs(uber-kernel): C2 diagnostic spec — β-coop vs legacy under PIECEWISE+graphs |
| 9 | [`7d429f1b7`](https://github.com/Navi-AI-Lab/nvllm/commit/7d429f1b7) | diag(c2): β-coop-vs-legacy diagnostic harness (env-gated, halt-on-divergence) |
| 10 | [`0185f84a0`](https://github.com/Navi-AI-Lab/nvllm/commit/0185f84a0) | feat(cute): β-coop under PIECEWISE+graphs — Phase 4 + KV-update DCE fix |
| 11 | [`e7c9c38e9`](https://github.com/Navi-AI-Lab/nvllm/commit/e7c9c38e9) | perf(cute): Phase 5 — restore paged-skip optimization with except-replay |
| 12 | [`722efc60b`](https://github.com/Navi-AI-Lab/nvllm/commit/722efc60b) | perf(cute): Phase 6a — β-coop hot-path Python diet (-4.0% per kernel call) |
| 13 | [`1f91013b8`](https://github.com/Navi-AI-Lab/nvllm/commit/1f91013b8) | perf(cutlass): Phase 6b — small-M NVFP4 GEMM dispatcher (-1.09% NVFP4 mass) |

---

## Code surfaces (line refs pinned to branch tip `1f91013b8`)

### Phase 4 — β-coop fires under PIECEWISE+graphs

| Surface | File | Lines |
|---|---|---|
| β-coop torch op + fake registration | [`vllm/v1/attention/backends/cute_paged/_beta_coop_op.py`](https://github.com/Navi-AI-Lab/nvllm/blob/1f91013b8432f01d5bc3cddfbd401a2d4d1cf0ea/vllm/v1/attention/backends/cute_paged/_beta_coop_op.py) | [L40](https://github.com/Navi-AI-Lab/nvllm/blob/1f91013b8432f01d5bc3cddfbd401a2d4d1cf0ea/vllm/v1/attention/backends/cute_paged/_beta_coop_op.py#L40), [L112](https://github.com/Navi-AI-Lab/nvllm/blob/1f91013b8432f01d5bc3cddfbd401a2d4d1cf0ea/vllm/v1/attention/backends/cute_paged/_beta_coop_op.py#L112), [L129](https://github.com/Navi-AI-Lab/nvllm/blob/1f91013b8432f01d5bc3cddfbd401a2d4d1cf0ea/vllm/v1/attention/backends/cute_paged/_beta_coop_op.py#L129) |
| Model-side dispatch (`Qwen3_5Attention`) | [`vllm/nvllm/models/qwen3_5.py`](https://github.com/Navi-AI-Lab/nvllm/blob/1f91013b8432f01d5bc3cddfbd401a2d4d1cf0ea/vllm/nvllm/models/qwen3_5.py) | [L295-L348](https://github.com/Navi-AI-Lab/nvllm/blob/1f91013b8432f01d5bc3cddfbd401a2d4d1cf0ea/vllm/nvllm/models/qwen3_5.py#L295-L348), [L582](https://github.com/Navi-AI-Lab/nvllm/blob/1f91013b8432f01d5bc3cddfbd401a2d4d1cf0ea/vllm/nvllm/models/qwen3_5.py#L582) |
| `_use_beta_coop` predicate + framework-output bind | [`vllm/v1/attention/backends/cute_paged/_backend.py`](https://github.com/Navi-AI-Lab/nvllm/blob/1f91013b8432f01d5bc3cddfbd401a2d4d1cf0ea/vllm/v1/attention/backends/cute_paged/_backend.py) | [L1246](https://github.com/Navi-AI-Lab/nvllm/blob/1f91013b8432f01d5bc3cddfbd401a2d4d1cf0ea/vllm/v1/attention/backends/cute_paged/_backend.py#L1246), [L1268](https://github.com/Navi-AI-Lab/nvllm/blob/1f91013b8432f01d5bc3cddfbd401a2d4d1cf0ea/vllm/v1/attention/backends/cute_paged/_backend.py#L1268) |

### Phase 5 — paged-skip narrowed to `_use_beta_coop` with except-handler replay

| Surface | File | Lines |
|---|---|---|
| `_skip_paged = _use_beta_coop` | [`vllm/v1/attention/backends/cute_paged/_backend.py`](https://github.com/Navi-AI-Lab/nvllm/blob/1f91013b8432f01d5bc3cddfbd401a2d4d1cf0ea/vllm/v1/attention/backends/cute_paged/_backend.py#L1267-L1268) | [L1267-L1268](https://github.com/Navi-AI-Lab/nvllm/blob/1f91013b8432f01d5bc3cddfbd401a2d4d1cf0ea/vllm/v1/attention/backends/cute_paged/_backend.py#L1267-L1268) |
| Skip-paged guard | [`vllm/v1/attention/backends/cute_paged/_backend.py`](https://github.com/Navi-AI-Lab/nvllm/blob/1f91013b8432f01d5bc3cddfbd401a2d4d1cf0ea/vllm/v1/attention/backends/cute_paged/_backend.py#L1326) | [L1326](https://github.com/Navi-AI-Lab/nvllm/blob/1f91013b8432f01d5bc3cddfbd401a2d4d1cf0ea/vllm/v1/attention/backends/cute_paged/_backend.py#L1326) |
| Except-replay branch | [`vllm/v1/attention/backends/cute_paged/_backend.py`](https://github.com/Navi-AI-Lab/nvllm/blob/1f91013b8432f01d5bc3cddfbd401a2d4d1cf0ea/vllm/v1/attention/backends/cute_paged/_backend.py#L1605-L1622) | [L1605-L1622](https://github.com/Navi-AI-Lab/nvllm/blob/1f91013b8432f01d5bc3cddfbd401a2d4d1cf0ea/vllm/v1/attention/backends/cute_paged/_backend.py#L1605-L1622) |

### Phase 6a — hot-path Python diet (module-level env caches)

| Surface | File | Lines |
|---|---|---|
| `_CUTE_DUMP_TENSORS`, `_VERIFY_FRAMEWORK_OUTPUTS`, `_PHASE_E_ENV` | [`vllm/v1/attention/backends/cute_paged/_backend.py`](https://github.com/Navi-AI-Lab/nvllm/blob/1f91013b8432f01d5bc3cddfbd401a2d4d1cf0ea/vllm/v1/attention/backends/cute_paged/_backend.py) | [L46](https://github.com/Navi-AI-Lab/nvllm/blob/1f91013b8432f01d5bc3cddfbd401a2d4d1cf0ea/vllm/v1/attention/backends/cute_paged/_backend.py#L46), [L52](https://github.com/Navi-AI-Lab/nvllm/blob/1f91013b8432f01d5bc3cddfbd401a2d4d1cf0ea/vllm/v1/attention/backends/cute_paged/_backend.py#L52), [L130](https://github.com/Navi-AI-Lab/nvllm/blob/1f91013b8432f01d5bc3cddfbd401a2d4d1cf0ea/vllm/v1/attention/backends/cute_paged/_backend.py#L130) |
| `_BETA_COOP_COUNT_FIRES` flag | [`vllm/v1/attention/backends/cute_paged/_beta_coop_op.py`](https://github.com/Navi-AI-Lab/nvllm/blob/1f91013b8432f01d5bc3cddfbd401a2d4d1cf0ea/vllm/v1/attention/backends/cute_paged/_beta_coop_op.py#L36) | [L36](https://github.com/Navi-AI-Lab/nvllm/blob/1f91013b8432f01d5bc3cddfbd401a2d4d1cf0ea/vllm/v1/attention/backends/cute_paged/_beta_coop_op.py#L36) |

### Phase 6b — small-M NVFP4 GEMM dispatcher

| Surface | File | Lines |
|---|---|---|
| Winners table + `lookup_m_small_winner` | [`csrc/libtorch_stable/quantization/fp4/nvfp4_winners_table.hpp`](https://github.com/Navi-AI-Lab/nvllm/blob/1f91013b8432f01d5bc3cddfbd401a2d4d1cf0ea/csrc/libtorch_stable/quantization/fp4/nvfp4_winners_table.hpp) | [L28](https://github.com/Navi-AI-Lab/nvllm/blob/1f91013b8432f01d5bc3cddfbd401a2d4d1cf0ea/csrc/libtorch_stable/quantization/fp4/nvfp4_winners_table.hpp#L28), [L33](https://github.com/Navi-AI-Lab/nvllm/blob/1f91013b8432f01d5bc3cddfbd401a2d4d1cf0ea/csrc/libtorch_stable/quantization/fp4/nvfp4_winners_table.hpp#L33) |
| BF16 dispatch (small-M reorder) | [`csrc/libtorch_stable/quantization/fp4/nvfp4_scaled_mm_sm120_kernels.cu`](https://github.com/Navi-AI-Lab/nvllm/blob/1f91013b8432f01d5bc3cddfbd401a2d4d1cf0ea/csrc/libtorch_stable/quantization/fp4/nvfp4_scaled_mm_sm120_kernels.cu#L340-L380) | [L340-L380](https://github.com/Navi-AI-Lab/nvllm/blob/1f91013b8432f01d5bc3cddfbd401a2d4d1cf0ea/csrc/libtorch_stable/quantization/fp4/nvfp4_scaled_mm_sm120_kernels.cu#L340-L380) |
| FP16 dispatch (small-M reorder) | [`csrc/libtorch_stable/quantization/fp4/nvfp4_scaled_mm_sm120_kernels.cu`](https://github.com/Navi-AI-Lab/nvllm/blob/1f91013b8432f01d5bc3cddfbd401a2d4d1cf0ea/csrc/libtorch_stable/quantization/fp4/nvfp4_scaled_mm_sm120_kernels.cu#L419-L460) | [L419-L460](https://github.com/Navi-AI-Lab/nvllm/blob/1f91013b8432f01d5bc3cddfbd401a2d4d1cf0ea/csrc/libtorch_stable/quantization/fp4/nvfp4_scaled_mm_sm120_kernels.cu#L419-L460) |
| Codegen (incl. `SMALL_ONLY_SHAPES` for the GDN row) | [`docs/research/gemm_sweep/gen_winners_header.py`](https://github.com/Navi-AI-Lab/nvllm/blob/1f91013b8432f01d5bc3cddfbd401a2d4d1cf0ea/docs/research/gemm_sweep/gen_winners_header.py) | full file |
| Replay harness (`--m-band` + new label modes) | [`docs/research/gemm_sweep/replay_winners_table.py`](https://github.com/Navi-AI-Lab/nvllm/blob/1f91013b8432f01d5bc3cddfbd401a2d4d1cf0ea/docs/research/gemm_sweep/replay_winners_table.py) | full file |

### C2 diagnostic harness

| Surface | File | Lines |
|---|---|---|
| Halt-on-divergence comparator | [`vllm/v1/attention/backends/cute_paged/_c2_diag.py`](https://github.com/Navi-AI-Lab/nvllm/blob/1f91013b8432f01d5bc3cddfbd401a2d4d1cf0ea/vllm/v1/attention/backends/cute_paged/_c2_diag.py) | full file (308 lines) |
| Test coverage | [`tests/v1/cute_paged/test_c2_diag.py`](https://github.com/Navi-AI-Lab/nvllm/blob/1f91013b8432f01d5bc3cddfbd401a2d4d1cf0ea/tests/v1/cute_paged/test_c2_diag.py) | full file (235 lines) |

---

## Evidence

All measurements taken under identical workloads (5 timed × 64 max_tokens × concurrency=1, 15 warmup curls, PIECEWISE CUDA graphs, FP8 KV cache, FUSION=1). Per-kernel μs values from torch profiler via `--profiler-config` + `/start_profile` / `/stop_profile`; nsys CUPTI cannot trace vLLM V1's spawned EngineCore.

### `PhaseE_Beta_Kernel` per-call (μs)

| Run | Commit | Calls | Mean μs | Δ vs Phase E baseline |
|---|---|---:|---:|---:|
| Phase E β-coop baseline (main) | [`bc9037955`](https://github.com/Navi-AI-Lab/nvllm/commit/bc9037955) | 5,040 | 42,933.771 | — |
| Phase 6a (this branch) | [`722efc60b`](https://github.com/Navi-AI-Lab/nvllm/commit/722efc60b) | 5,040 | 41,217.510 | −1,716.261 (−4.00%) |
| Phase 6b (this branch tip) | [`1f91013b8`](https://github.com/Navi-AI-Lab/nvllm/commit/1f91013b8) | 5,040 | 40,893.101 | **−2,040.670 (−4.75%)** |

Sources: [phase_e summary](https://github.com/Navi-AI-Lab/nvllm/blob/1f91013b8432f01d5bc3cddfbd401a2d4d1cf0ea/benchmarks/nvllm/traces/phase_e/2026-04-23-initial/summary.md), [phase_6a summary](https://github.com/Navi-AI-Lab/nvllm/blob/1f91013b8432f01d5bc3cddfbd401a2d4d1cf0ea/benchmarks/nvllm/traces/phase_6a/2026-04-29-initial/summary.md), [phase_6b summary](https://github.com/Navi-AI-Lab/nvllm/blob/1f91013b8432f01d5bc3cddfbd401a2d4d1cf0ea/benchmarks/nvllm/traces/gemm_winners_table_smallM/2026-04-29-qwen35-27b/summary.md).

### NVFP4 GEMM mass (Phase 6b small-M dispatcher)

| Run | Commit | Calls | Total ms | Mean μs/call | Δ vs Phase 6a |
|---|---|---:|---:|---:|---:|
| Phase 6a | [`722efc60b`](https://github.com/Navi-AI-Lab/nvllm/commit/722efc60b) | 36,080 | 11,724.2 | 324.97 | — |
| Phase 6b build #1 (no GDN row) | (intermediate) | 36,080 | 11,624.1 | 322.18 | −100.1 ms (−0.85%) |
| Phase 6b build #2 (GDN row added) | [`1f91013b8`](https://github.com/Navi-AI-Lab/nvllm/commit/1f91013b8) | 36,080 | 11,596.8 | 321.43 | **−127.4 ms (−1.09%)** |

### Phase 6b dispatcher replay (per-shape × M, small-M band)

20-cell replay total against forced-Stream-K baseline (`NVLLM_FP4_GEMM_CONFIG_M256=4` vs no env var):

| Shape | Σ baseline μs | Σ table μs | Δ |
|---|---:|---:|---:|
| `qkv_proj` (8192, 5120) | 448.70 | 343.74 | **−23.4%** |
| `o_proj` (5120, 6144) | 332.28 | 288.77 | **−13.1%** |
| `gate_up_proj` (34816, 5120) | 2,120.30 | 2,135.84 | +0.7% |
| `down_proj` (5120, 17408) | 1,150.49 | 1,143.69 | −0.6% |
| **Total (20 cells)** | **4,051.77** | **3,912.04** | **−3.45%** |

Source: [phase_6b summary § Primary evidence](https://github.com/Navi-AI-Lab/nvllm/blob/1f91013b8432f01d5bc3cddfbd401a2d4d1cf0ea/benchmarks/nvllm/traces/gemm_winners_table_smallM/2026-04-29-qwen35-27b/summary.md). Wins concentrate on shapes where the optimal tile differs (`128x256x128` vs Stream-K's `128x128x256`); near-zero where the tile shapes match (only the schedule differs).

### Phase 5 — paged-skip + except-replay

GSM8K sanity per-question latency dropped 16 s/Q → 12 s/Q (~25%) vs Phase 4 (`0185f84a0`). 8/8 PASS. The legacy paged-attention forward calls do not appear in the kernel-time table for `_use_beta_coop` paths. Source: [phase_5_paged_skip summary](https://github.com/Navi-AI-Lab/nvllm/blob/1f91013b8432f01d5bc3cddfbd401a2d4d1cf0ea/benchmarks/nvllm/traces/phase_5_paged_skip/2026-04-28-restored/summary.md).

### End-to-end wall (Phase 6a)

GSM8K-50, seed=42, max_tokens=512, thinking off, identical workload:

| Run | Commit | Correct | Wall (s) | Δ |
|---|---|---:|---:|---:|
| Phase 5 | [`e7c9c38e9`](https://github.com/Navi-AI-Lab/nvllm/commit/e7c9c38e9) | 30/50 | 7,030 | — |
| Phase 6a | [`722efc60b`](https://github.com/Navi-AI-Lab/nvllm/commit/722efc60b) | 31/50 | 6,838 | **−192 s (−2.7%)** |

---

## Correctness gates

| Gate | Result | Reference |
|---|---|---|
| GSM8K 8/8 sanity at Phase 4 ship | 8/8 PASS | [`0185f84a0`](https://github.com/Navi-AI-Lab/nvllm/commit/0185f84a0) |
| GSM8K 8/8 sanity at Phase 5 ship | 8/8 PASS, 12 s/Q | [phase_5 summary](https://github.com/Navi-AI-Lab/nvllm/blob/1f91013b8432f01d5bc3cddfbd401a2d4d1cf0ea/benchmarks/nvllm/traces/phase_5_paged_skip/2026-04-28-restored/summary.md) |
| GSM8K-50 (seed=42) at Phase 6a ship | 31/50 (62.0%); Phase 5 baseline 30/50 | [phase_6a summary](https://github.com/Navi-AI-Lab/nvllm/blob/1f91013b8432f01d5bc3cddfbd401a2d4d1cf0ea/benchmarks/nvllm/traces/phase_6a/2026-04-29-initial/summary.md) |
| GSM8K 8/8 sanity at Phase 6b ship | 8/8 PASS (dispatcher refactor, no math change) | [phase_6b summary § Correctness gate](https://github.com/Navi-AI-Lab/nvllm/blob/1f91013b8432f01d5bc3cddfbd401a2d4d1cf0ea/benchmarks/nvllm/traces/gemm_winners_table_smallM/2026-04-29-qwen35-27b/summary.md) |

Test files added on this branch:

- [`tests/v1/cute_paged/test_beta_coop_skeleton.py`](https://github.com/Navi-AI-Lab/nvllm/blob/1f91013b8432f01d5bc3cddfbd401a2d4d1cf0ea/tests/v1/cute_paged/test_beta_coop_skeleton.py) — 123 lines
- [`tests/v1/cute_paged/test_c2_diag.py`](https://github.com/Navi-AI-Lab/nvllm/blob/1f91013b8432f01d5bc3cddfbd401a2d4d1cf0ea/tests/v1/cute_paged/test_c2_diag.py) — 235 lines
- [`tests/v1/cute_paged/test_uber_kernel_buffer_contracts.py`](https://github.com/Navi-AI-Lab/nvllm/blob/1f91013b8432f01d5bc3cddfbd401a2d4d1cf0ea/tests/v1/cute_paged/test_uber_kernel_buffer_contracts.py) — 48 lines
- [`tests/v1/cute_paged/test_uber_kernel_multi_layer.py`](https://github.com/Navi-AI-Lab/nvllm/blob/1f91013b8432f01d5bc3cddfbd401a2d4d1cf0ea/tests/v1/cute_paged/test_uber_kernel_multi_layer.py) — 114 lines

---

## Artifact index (committed evidence)

| Path | Contents |
|---|---|
| `benchmarks/nvllm/traces/phase_e/2026-04-23-initial/` | Phase E baseline (kernel CSV, serve log, summary) |
| `benchmarks/nvllm/traces/phase_5_paged_skip/2026-04-28-restored/` | Phase 5 paged-skip restoration evidence |
| `benchmarks/nvllm/traces/phase_6a/2026-04-29-initial/` | Phase 6a Python-diet evidence + GSM8K-50 wall |
| `benchmarks/nvllm/traces/gemm_winners_table_smallM/2026-04-29-qwen35-27b/` | Phase 6b dispatcher replay + E2E + summary |
| `benchmarks/nvllm/traces/gemm_sweep_sm120_phase6b_gdn/2026-04-29/` | GDN `(14336, 5120)` supplemental microbench |

Per [AGENTS.md §4](https://github.com/Navi-AI-Lab/nvllm/blob/1f91013b8432f01d5bc3cddfbd401a2d4d1cf0ea/AGENTS.md): raw `*.pt.trace.json.gz` files are gitignored (size > 30 MB each, reproducible from the capture scripts in `docs/research/phase_*_traces/`); per-kernel CSVs, serve logs, memory watchdog logs, and `summary.md` are committed.

---

## Configuration verified at branch tip

| Field | Value |
|---|---|
| Backend | `CUTE_PAGED` |
| KV cache dtype | `fp8_e4m3` |
| Compilation | PIECEWISE CUDA graphs |
| Attention path | β-coop (`_use_beta_coop=True`) for full-attention layers when `64 * num_seqs ≤ _resident_cap` (96 on GB10) |
| Attention path (fallback) | β-lite for `num_seqs > 1` where the cooperative-launch resident cap blocks β-coop |
| Linear-attention layers | unaffected (FLA GDN, 48 of 64 layers) |
| Image | `nvllm:gb10` SHA `7ea16c763044` (Phase 6b build #2) |

---

## AI assistance disclosure

Branch authored with AI assistance (Claude Opus 4.7, 1M context). Each commit lists `Co-Authored-By` in the trailer. The submitting human reviewed every changed line and ran the listed correctness gates. No upstream `vllm-project/vllm` PRs are produced by this branch.
