# Torch profile (probe 4 / MLP-only) — Phase D MLP kernel is 80% of GPU budget

**Date:** 2026-04-24
**Config:** P4 (PHASE_E=0 MLP=1 ATTN=0), --enforce-eager, autotune-OFF.
**Method:** vLLM `/start_profile` + `/stop_profile` bracketing a single 32-token
completion, after a 16-token warmup to drain JIT cliffs. Trace at
`profile/rank0.*.pt.trace.json.gz` (14 MB).

## Top CUDA kernels (worker rank0)

| Kernel | Total CUDA | % | Calls | Avg/call |
|---|---|---|---|---|
| **`vllm::cute_mlp_forward` / kernel_cutlass…cute_paged…** | **12.929 s** | **63.1%** | 512 | **26.07 ms** |
| `aten::fill_` / `vectorized_elementwise_kernel<4>` | 3.846 s | 18.8% | 7,882 | 488 µs |
| `aten::mm` / `gemvx` (cuBLAS) | 1.779 s | 8.7% | 4,496 | 396 µs |
| `_C::cutlass_scaled_fp4_mm` | 1.200 s | 5.9% | 4,128 | 290 µs |
| `vllm::unified_attention_with_output` (CUTE paged attn) | 670 ms | 0.6% (CUDA only 128ms) | 512 | 1.31 ms |
| `vllm::gdn_attention_core` (linear attn, GDN/Mamba) | 4.57 s **CPU** / 32 ms CUDA | 0.16% | 1,536 | 2.97 ms (mostly CPU coord) |
| `_C::scaled_fp4_quant` | 11.5 ms | 0.06% | 4,128 | 2.79 µs |
| Triton RMSNorm `_to_copy_add_mean_mul_pow_rsqrt_0` | 11.2 ms | 0.05% | 4,852 | 2.31 µs |

## Cross-check: Phase D MLP kernel cost matches per-layer instrumentation

Profile says: 26.07 ms per call × 16 fused layers × 32 tokens = 13.348 s.
Probe 4 instrumentation said: 25.5 ms / fused layer (sync_end average).

Both methods independently land on **~26 ms per Phase D MLP kernel call**.
This is the bug.

For reference: cuBLAS gate+up SGEMM + W_down SGEMM + activation for the same
shape (`5120 × 17408 × 5120` BF16, NVFP4 weights) total ~1 ms. The Phase D
kernel is **~26× slower than cuBLAS reference**.

## Where the time is NOT going

- ATTN-side work (CUTE paged attn) is healthy at 1.3 ms/call.
- Linear_attention path is fine.
- Per-step CPU/Python overhead is small (~700 µs/layer, ~1% of budget).
- NVFP4 dequant + GEMM (`cutlass_scaled_fp4_mm`) is 290 µs avg — fine.

## Memset overhead (secondary suspect)

`aten::fill_` shows 3.85 s / 7,882 calls = 488 µs avg. Most are likely the
β-coop per-launch zeroing flagged by the token-POV subagent and/or the
`wo_output.zero_()` at `_backend.py:1023`. With β-coop OFF in probe 4 these
shouldn't all be from β; some come from MLP fusion's own scratch. Worth
auditing if Phase D MLP gets fixed and we need the next-tier win — but
right now Phase D MLP itself dwarfs everything.

## Proposed fix path

1. **Identify the Phase D MLP kernel's tile/grid config** —
   `vllm/v1/attention/backends/cute_paged/_kernels/mlp_kernel.py` and
   `_tile_presets.py`.
2. The serve log already prints:
   `Phase D MLP tile preset: prefill-legacy → tile_s=256 tile_k=640 slice_ctas=8`
   This is the prefill preset firing on decode. There's likely a different
   preset for `nat=1` decode that wasn't selected (or doesn't exist).
3. **Single-token decode case**: tile_s=256 with nat=1 wastes 255/256 of the
   tile. The kernel is launching huge tiles that are 99.6% padding.
4. Add a `decode-fast` preset or shrink `tile_s` to 1, 2, 4 for nat-aware
   launches.

## Memory / project updates triggered

- `project_fused_path_perf_collapse` — already updated with the four-probe
  evidence; can be tightened further with this profile data.
- New memory candidate: `project_phase_d_mlp_kernel_decode_mistune` — the
  prefill-legacy tile preset wastes nat=1 decode work.

## Next debug step

- Look at `_tile_presets.py:127` (the line that printed `prefill-legacy →
  tile_s=256...`) and the `Phase_D_MLP_Kernel` kernel launch site to confirm
  the wasted-tile hypothesis.
- Try forcing a smaller `tile_s` via env override and re-run probe 4 — should
  drop the 26 ms/call dramatically if the hypothesis is right.

## Files

- `serve.log` — full vLLM startup + warmup
- `warmup.json` — 16-token warmup completion (drained JIT cliffs)
- `completion.json` — the 32-token completion that was profiled
- `profile/profiler_out_0.txt` — vLLM's auto-generated kernel summary (also
  the source of the table above)
- `profile/rank0.*.pt.trace.json.gz` — 14 MB worker trace, openable in
  chrome://tracing
- `profile/*.async_llm.*.pt.trace.json.gz` — frontend trace (1 KB)
- `analyze.py` — local script to re-aggregate top kernels from the trace
