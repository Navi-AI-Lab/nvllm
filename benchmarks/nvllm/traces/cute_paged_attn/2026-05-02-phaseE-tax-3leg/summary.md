# Phase-E-tax screening — 3-leg torch profiler

## Verdict

β-coop is not a tax **versus the phaseE-off production-ish fallback**, but more
β-coop is **not currently monotonically faster**. Lower8 is the **current best
tested point** (not "sweet spot" — we did not run a 3-layer leg). Adding β to
layers 11 and 15 replaces two DecodeKernel calls (~17.1 ms each) with two β
calls (~40.8 ms each) — a net regression of ~47 ms/token across those two
layers, not a net win. The phaseE-off configuration loses by an order of
magnitude per token; its 2/50 GSM8K result reflects 180 s read-timeouts on 48
prompts (the 2 that completed gave correct answers), not numerics breakage.

## Provenance

| Field             | Value                                                                 |
| :---------------- | :-------------------------------------------------------------------- |
| Date              | 2026-05-02                                                            |
| Commit            | `10aa78757`                                                           |
| Image ID          | `sha256:a3f3f609a8ec873b0c8f6ddeb71573514eb84bf41b814ac82303d998a6ac5b88` |
| Model             | `ig1/Qwen3.5-27B-NVFP4`                                               |
| Model revision    | `4c546624f1fa8b77f5b7cfb3b6c96bf46d25c3a9`                            |
| Hardware          | NVIDIA DGX Spark (GB10, SM120/SM121), 48 SMs, 128 GB unified LPDDR5x  |
| GSM8K seed        | `42` (n=50, `/v1/completions`, default label)                         |

All three legs share the same image, commit, model, and hardware. Per-leg
configuration differs only in the env vars in the next section.

## Leg configuration

| Leg          | `CUTE_PHASE_E_FUSION` | `CUTE_PHASE_E_LAYERS` | β-coop layers       | Profile warmup | Profile timed | GSM8K seed | GSM8K n |
| :----------- | :-------------------- | :-------------------- | :------------------ | -------------: | ------------: | ---------: | ------: |
| `lower8`     | `1`                   | `0..7`                | layers 3, 7 (2L)    |             15 |            10 |         42 |      50 |
| `phaseE-off` | `0`                   | `0..7` (irrelevant)   | none                |              4 |            10 |         42 |      50 |
| `all-beta`   | `1`                   | `0..15`               | layers 3, 7, 11, 15 |             20 |             4 |         42 |      50 |

All legs: `--max-num-seqs 1`, `--max-model-len 16384`,
`--max-num-batched-tokens 65536`, `--kv-cache-dtype fp8_e4m3`,
`--attention-backend CUTE_PAGED`, `--gpu-memory-utilization 0.65`,
`B12X_CUTE_COMPILE_DISK_CACHE=1` (disk cache shared across legs at
`/tmp/nvllm-cute-cache`), profile boots ran `cudagraph_mode=FULL_AND_PIECEWISE`
without a blessed-cache mount, GSM8K boots ran `cudagraph_mode=PIECEWISE`. The
Qwen3.5-27B checkpoint has 16 full-attention layers (out of 64 decoder layers
total); DecodeKernel and PhaseE_Beta_Kernel fire on those 16.

## Kernel duration table (per-leg, from `profile_kernels.csv`)

Top custom CuTe kernels and top GEMM/GEMV rows by `total_ms`. Numbers are
exactly as reported by `extract_e2e_kernels.py`; no rounding.

### Custom CuTe kernels

| Leg          | Kernel                  | `n_calls` | `mean_us` | `total_ms` |
| :----------- | :---------------------- | --------: | --------: | ---------: |
| `lower8`     | DecodeKernel            |     35700 | 17088.491 | 610059.134 |
| `lower8`     | PhaseE_Beta_Kernel      |      5100 | 40635.606 | 207241.590 |
| `phaseE-off` | DecodeKernel            |     40800 | 17040.088 | 695235.576 |
| `phaseE-off` | Phase_D_MLP_Kernel      |     40800 | 23931.397 | 976401.001 |
| `all-beta`   | DecodeKernel            |     12240 | 17106.305 | 209381.173 |
| `all-beta`   | PhaseE_Beta_Kernel      |      4080 | 40829.264 | 166583.397 |

Per-layer call counts within each leg are perfectly consistent:
`lower8` 35700/14 = 2550 = 5100/2; `phaseE-off` 40800/16 = 2550;
`all-beta` 12240/12 = 1020 = 4080/4. The cross-leg 2.5× ratio reflects
warmup+timed burst-count differences (lower8 25 vs all-beta 24 with shorter
timed phase), not a contamination signal — see "Kernel-inventory contamination
check" below.

### Top GEMM/GEMV by `total_ms` (per leg)

| Leg          | Kernel (truncated)                                            | `n_calls` | `mean_us` | `total_ms` |
| :----------- | :------------------------------------------------------------ | --------: | --------: | ---------: |
| `lower8`     | `internal::gemvx::kernel<… __nv_bfloat16, … float, …>`        |    369760 |   391.545 | 144777.688 |
| `lower8`     | `cutlass…SM120 BlockScaledSm120 NVFP4 GEMM (variant A)`       |    358280 |   315.065 | 112881.488 |
| `lower8`     | `nvjet_sm121_tst_mma_128x96x64_3_64x24x64_tmaAB_bz_TNNN`      |       480 |   752.498 |    361.199 |
| `lower8`     | `cutlass_80_wmma_tensorop_bf16_s161616gemm_bf16_32x32_128x1`  |       480 |   305.589 |    146.683 |
| `lower8`     | `cutlass…SM120 BlockScaledSm120 NVFP4 GEMM (variant B)`       |       160 |   210.622 |     33.699 |
| `phaseE-off` | `internal::gemvx::kernel<… __nv_bfloat16, … float, …>`        |    369760 |   393.817 | 145617.729 |
| `phaseE-off` | `cutlass…SM120 BlockScaledSm120 NVFP4 GEMM (variant A)`       |    286880 |   313.318 |  89884.602 |
| `phaseE-off` | `cutlass…SM120 BlockScaledSm120 NVFP4 GEMM (variant C, 256×128)` |  40960 |    58.046 |   2377.578 |
| `phaseE-off` | `nvjet_sm121_tst_mma_128x96x64_3_64x24x64_tmaAB_bz_TNNN`      |       480 |   764.239 |    366.835 |
| `phaseE-off` | `cutlass_80_wmma_tensorop_bf16_s161616gemm_bf16_32x32_128x1`  |       480 |   309.867 |    148.736 |
| `all-beta`   | `internal::gemvx::kernel<… __nv_bfloat16, … float, …>`        |    147904 |   393.920 |  58262.367 |
| `all-beta`   | `cutlass…SM120 BlockScaledSm120 NVFP4 GEMM (variant A)`       |    139232 |   316.065 |  44006.430 |
| `all-beta`   | `nvjet_sm121_tst_mma_128x96x64_3_64x24x64_tmaAB_bz_TNNN`      |       192 |   752.237 |    144.430 |
| `all-beta`   | `cutlass_80_wmma_tensorop_bf16_s161616gemm_bf16_32x32_128x1`  |       192 |   309.350 |     59.395 |
| `all-beta`   | `cutlass…SM120 BlockScaledSm120 NVFP4 GEMM (variant B)`       |        64 |   215.012 |     13.761 |

Mean μs on the cublas `gemvx` anchor differs by ≤0.6% across legs
(391.545 / 393.817 / 393.920) — stable.

## Per-token decode budget (16 full-attention layers)

Counting only the two dominant per-layer kernels (DecodeKernel and either
PhaseE_Beta_Kernel or Phase_D_MLP_Kernel). This **does not** cover GEMV/GEMM
rows, RMSNorm, RoPE, sampling, or per-token Python overhead — it is the
"per-layer two-kernel" budget, not the total per-token wall time.

| Leg          | Decode layers × `mean_us`                | β / Phase_D layers × `mean_us`                            | Per-token (ms) |
| :----------- | :--------------------------------------- | :-------------------------------------------------------- | -------------: |
| `lower8`     | 14 × 17088.491 us = 239.239 ms           | 2 × 40635.606 us = 81.271 ms (PhaseE_Beta_Kernel)         |        320.510 |
| `all-beta`   | 12 × 17106.305 us = 205.276 ms           | 4 × 40829.264 us = 163.317 ms (PhaseE_Beta_Kernel)        |        368.593 |
| `phaseE-off` | 16 × 17040.088 us = 272.641 ms           | 16 × 23931.397 us = 382.902 ms (Phase_D_MLP_Kernel)       |        655.544 |

Marginal cost of adding β to one more full-attn layer (using `all-beta` means):
`40829.264 us − 17106.305 us = 23722.959 us ≈ 23.7 ms/token/added-layer`.
β-coop fused-kernel mean μs is 40635.606 (lower8) and 40829.264 (all-beta),
i.e. ~40.6–40.8 ms — matches the analysis input.

## GSM8K results

| Leg          | β-coop layers     | `correct` | `accuracy`    | `total_seconds` | `gate_floor` | `gate_ship` |
| :----------- | :---------------- | --------: | :------------ | --------------: | :----------- | :---------- |
| `lower8`     | 3, 7 (2L)         |     47/50 | 47/50 (94.0%) |          3624.3 | PASS         | PASS        |
| `phaseE-off` | none (fusion=0)   |      2/50 |  2/50 (4.0%)  |          8922.4 | FAIL         | FAIL        |
| `all-beta`   | 3, 7, 11, 15 (4L) |     47/50 | 47/50 (94.0%) |          4012.8 | PASS         | PASS        |

**phaseE-off failure is read timeouts, not numerics.** 48/50 prompts hit the
client-side 180 s read timeout (`elapsed: 180.0`,
`raw_tail: "ERROR: HTTPConnectionPool(...): Read timed out. (read timeout=180)"`).
The two prompts that completed (i=11 in 172.5 s, i=17 in 106.9 s) returned
correct answers. The decoder pathway works; per-token wall time at ~656 ms ×
several-hundred-token answers blows past the client timeout. Future readers
should not misread "2/50" as a phaseE-off math regression.

## Kernel-inventory contamination check

Per the README's plan-A fallback rule ("If symbol inventory or call counts
differ unexpectedly between legs … method D is contaminated").

**(a) Custom-kernel inventory — consistent.** Each leg has exactly the
custom CuTe symbols its config implies:

- `lower8` and `all-beta`: present `PhaseE_Beta_Kernel`, absent
  `Phase_D_MLP_Kernel`.
- `phaseE-off`: present `Phase_D_MLP_Kernel`, absent `PhaseE_Beta_Kernel`.
- `DecodeKernel`, `fused_recurrent_gated_delta_rule_packed_decode_kernel`,
  `reshape_and_cache_flash_kernel`, `_compute_slot_mapping_kernel`, the
  Qwen3.5 sigmoid-output-gate path, and all NVFP4 cutlass GEMM variants are
  present in every leg.

**(b) No 5×+ call-count drift on stable kernels.** Per-layer-per-burst counts
are exact:
- `lower8`: 35700/14 = 2550 (Decode) = 5100/2 (β).
- `phaseE-off`: 40800/16 = 2550 (Decode) = 40800/16 (Phase_D).
- `all-beta`: 12240/12 = 1020 (Decode) = 4080/4 (β).

The 2.5× cross-leg ratio is the warmup+timed burst-count difference
(`all-beta` ran fewer timed bursts: 4 vs 10). It hits all stable kernels
uniformly (`fused_recurrent_gated_delta_rule`: 122400/122400/48960;
`reshape_and_cache_flash`: 40960/40960/16384; cublas `gemvx`:
369760/369760/147904 — same 2.5× factor), confirming uniform burst-count
scaling, not custom-kernel contamination.

**(c) Inductor/Triton pointwise differences are expected and non-driving.**
Six triton symbols appear only in `lower8`, eight only in `phaseE-off`, and
four are shared between `phaseE-off` and `all-beta` but absent from `lower8`
(`triton_poi_fused__to_copy_add_cat_*` family, `triton_poi_fused_6`,
`triton_red_fused_7`). These are inductor-generated pointwise/reduce kernels
whose codegen depends on the residual+RMSNorm dispatch path the leg's fusion
config takes. Per the README's analysis rule, inductor one-offs are reported
but do not drive the verdict. Total time on any single triton symbol is ≤67 ms
across the entire timed window — orders of magnitude below the
`PhaseE_Beta_Kernel`/`DecodeKernel`/`Phase_D_MLP_Kernel` rows.

**Verdict: contamination check PASSES.** Method D is valid; we do not need to
fall back to plan A (bless phaseE-off first, re-measure under blessed FULL).
Per-call kernel μs comparisons stand.

## Recommendation

Keep lower8 as production. Do not pursue phaseE-off. Do not revive Phase 4.
Put kernel effort into making β's Phase 3 cheaper. The Veitner NVFP4 GEMV /
K-parallel reduction direction is now even more clearly the next serious bet.
If that drops β from ~40 ms toward the low-30s, then 4L becomes interesting
again.

## Caveats

- **Cold FULL graph.** Profile boots ran `FULL_AND_PIECEWISE` with no
  blessed-cache mount. Z1 inductor non-determinism (memory:
  `project_full_graph_blocked.md`) means each cold capture can yield a
  different inductor pointwise inventory; CuTe and CUTLASS GEMM symbols are
  the stable comparison anchor and were used for the verdict.
- **β-coop disk cache shared across legs.** All three profile boots reused
  `/tmp/nvllm-cute-cache`, so cute.compile cost is amortized — but kernel
  μs comparisons are unaffected (cache only affects compile time, not
  kernel runtime).
- **PIECEWISE GSM8K is not FULL-mode quality.** It tells us "does the kernel
  pathway produce correct outputs under PIECEWISE?" It does not tell us
  "does FULL+phaseE-off produce sane outputs?" That question is not answered
  by this experiment and is not blocking, because per-call kernel μs already
  rules phaseE-off out on speed.
- **No 3-layer leg.** We did not run β on (3, 7, 11) or similar 3L
  configurations; the optimum within 2L < N < 4L is unknown. Hence "lower8 is
  the current best tested point" — not "sweet spot".
- **Not a production-ready FULL trace.** Per the README, this is a screening
  experiment. Cold cold-graph variance plus the absence of a blessed FULL
  cache for any leg means the per-token totals here are not the production
  number under blessed FULL. The relative comparison across legs holds.

## How to reproduce

```bash
# tmux required — total wall time ~210-240 min (GSM8K dominates)
tmux new -s bench-3leg
bash docs/research/2026-05-02-phaseE-tax-3leg/run_3leg.sh
# Skip a leg if needed:
SKIP_ALL_BETA=1 bash docs/research/2026-05-02-phaseE-tax-3leg/run_3leg.sh
SKIP_LOWER8=1 bash docs/research/2026-05-02-phaseE-tax-3leg/run_3leg.sh
SKIP_PHASEE_OFF=1 bash docs/research/2026-05-02-phaseE-tax-3leg/run_3leg.sh
# Force re-run of one leg:
rm benchmarks/nvllm/traces/cute_paged_attn/2026-05-02-phaseE-tax-3leg/<leg>/profile_DONE
rm benchmarks/nvllm/traces/cute_paged_attn/2026-05-02-phaseE-tax-3leg/<leg>/gsm8k_DONE
```

## Related work

- Experiment design and decision matrix:
  `docs/research/2026-05-02-phaseE-tax-3leg/README.md`.
- Prior FULL trace (lower8 only, with blessed cache and reset workspace
  hypothesis ruled out):
  `benchmarks/nvllm/traces/cute_paged_attn/2026-04-30-coop-wo-reset/`. The
  per-layer breakdown there is the analysis seed for this experiment.
- Blessed-cache infrastructure (used if a follow-up plan-A FULL bless of
  any leg is needed): `scripts/serve-cute-full.sh`,
  `scripts/bless-cute-full-cache.sh`, and the design doc at
  `docs/superpowers/specs/2026-05-01-cute-full-cache-production-workaround-design.md`.
