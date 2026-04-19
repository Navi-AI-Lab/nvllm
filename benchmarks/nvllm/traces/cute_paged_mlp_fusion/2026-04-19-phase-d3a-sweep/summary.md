# Phase D3a sweep — MLP decode-tile retune results

**Spec:** `docs/superpowers/specs/2026-04-19-phase-d3a-mlp-decode-retune-design.md`
**Image:** `nvllm:gb10-phaseD3a`
**Model:** `natfii/Qwen3.5-27B-NVFP4-Opus-GB10`
**Config:** `max-model-len=65536, max-num-seqs=4, kv-cache-dtype=fp8_e4m3, cudagraph_mode=PIECEWISE, CUTE_ATTN_FUSION=1, CUTE_MLP_FUSION=1, --language-model-only, --gpu-memory-utilization 0.80`
**Workload:** 4 concurrent × 128 tok, temperature=0, ignore_eos=true.
**Code:** commit `a1f1bedad` on branch `feat/unreal-kernel-phase-d`.

## Per-preset results

Per-call MLP timing is `kernel_cutlass__kernel_vllmv1attentionbackendscute_p...` first row of `Phase_D_MLP_Kernel` / 2032 calls (identical call count across presets). Attn fused kernel is the second `kernel_cutlass_...` row adjacent to `vllm::unified_attention_with_output`. `s/Q (GSM8K)` is mean per-question wallclock from the GSM8K sanity harness (single-stream, 256 max tokens) — directly comparable to the D2e-baked 8.7 figure.

| preset | grid @ nat=4 | CTAs | ~waves @ 96r | slices/CTA | MLP μs/call | MLP Self CUDA (s) | attn fused Self CUDA (s) | s/Q (GSM8K) | GSM8K | verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| prefill-legacy | (8,8,4) | 256 | 2.7 | 9 | 69.61 ms | 141.45 | 34.44 | 12.18 | 7/8 | fresh session baseline |
| decode-balanced | (16,8,4) | 512 | 5.3 | 9 | 48.61 ms | 98.77 | 34.61 | 9.93 | 7/8 | +30.2% MLP vs baseline |
| decode-small | (32,8,4) | 1024 | 10.7 | 9 | 45.83 ms | 93.14 | 34.44 | 10.14 | 7/8 | +34.2% MLP vs baseline (best μs/call) |
| decode-narrow-grid | (8,4,4) | 128 | 1.3 | 9 | 56.48 ms | 114.76 | 34.30 | 18.15 | 7/8 | +18.9% MLP, **−49% wallclock** (anti-win) |

## Observations

### 1. 7/8 floor reproduces on `prefill-legacy`

All four presets — including `prefill-legacy`, which has bit-for-bit identical tile values (256, 640, 8) to D2e's shipped config — score 7/8 on GSM8K, failing Q2 ("Weng earns $12/hour, 50 min babysitting → expected 10"). **This contradicts the D2e summary's claim of 8/8 true-math correctness** and is a session-level finding that is **not tile-induced** — it shows up on the baseline.

D2e's explanation for Q2 was "model wrote `600/60 = 10`, extractor grabbed `600`" (arithmetic correct, extraction artifact). **D3a's Q2 raw outputs are mathematically broken** — the model is writing nonsense like `50/5. 5/12. 5`, not coherent reasoning. This is a real regression, not an extraction bug. Investigation is scoped to the follow-up phase.

### 2. Per-preset Q2 raw outputs diverge

At `temperature=0` (greedy decode), the same prompt should produce bit-identical output across runs if the fused MLP kernel is numerically invariant under tile config. It isn't:

| preset | Q2 `got` | Q2 raw |
|---|---|---|
| prefill-legacy | 50 | `50/5.  5/12.  5` |
| decode-balanced | 50 | `50/5.  12/12.` |
| decode-small | 1200 | `1200/60 = 200/10` |
| decode-narrow-grid | 50 | `50/5.  1200/10.` |

Each tile config produces slightly different token trajectories at greedy decode, indicating the kernel is **not bit-exact across tile configs**. This is expected for FP4 fusion (per-block scale + order-of-accumulation differences) but is now documented as the sensitivity the next layer of correctness testing needs to account for.

### 3. `decode-narrow-grid` paradox

`decode-narrow-grid` (tile_k=1280, slice_ctas=8 → grid 128 CTAs, ~1.3 waves) delivers the fewest waves in the sweep and does improve MLP μs/call by 18.9%, but its **end-to-end GSM8K wallclock is 49% worse** than baseline (18.15 s/Q vs 12.18 s/Q). Hypotheses:
- At 1.3 waves, the tiny grid under-subscribes 48 SMs, so the kernel completes each wave fast but the PIECEWISE graph's non-MLP kernels (attention, silu, cast-to-FP4, aten::mm post-process) bubble more between MLP calls.
- `rows_per_thread=10` (2× prefill-legacy's 5) doubles per-thread register pressure; occupancy drops below 2 CTA/SM, and latency hiding breaks down.

**Do not ship `decode-narrow-grid`** despite the kernel-level metric improvement. This is a useful anti-win for the D3b SMEM pipelining design doc: pipelining + `decode-small`-style parallelism likely composes; pipelining + narrow grid likely does not.

### 4. Attn kernel time is flat across presets (expected invariant)

`attn fused Self CUDA` stays at 34.30–34.61 s across all four presets (variance < 1%). This confirms the tile retune does not leak into the attention path. The D2e baseline was 32.01 s — the ~7.5% higher number in D3a is session/image variance, also affecting `prefill-legacy` equally.

## Verdict

**Winning preset (kernel-level):** `decode-small` — 34.2% faster MLP kernel, flat attn kernel, matches session baseline GSM8K score.

**GSM8K 8/8:** none. **GSM8K regressions vs fresh `prefill-legacy` baseline (7/8):** none — all four presets, including baseline, score 7/8 with the same Q2 failure class.

**Ship decision:**
- Default preset (`_DEFAULT_PRESET_NAME`): **unchanged (prefill-legacy).** Rationale: the 7/8 session floor is a real regression (Q2 raw outputs are mathematically incoherent, not extractor artifacts) and needs root-cause investigation before any deployment flip. Operators can still opt in to `decode-small` via `CUTE_MLP_TILE=decode-small` for a ~34% MLP speedup at their own correctness risk.
- `CUTE_MLP_FUSION` default: unchanged (opt-in). Per spec, default-flip is deferred to whichever follow-on change closes the gap to unfused (2.5 s/Q); tile retune alone delivers 1.52× at best (observed), within the audit's 1.1-1.5× prediction band, nowhere near the 3.5× needed.

## Follow-on motivation

1. **7/8 floor investigation (blocker for any default-flip).** D3a image produces different Q2 output than D2e image at identical tile values — suggests a D3a-image dep drift (torch nightly, cutlass, vllm transitive) rather than a D3a code regression. Action: rerun `prefill-legacy` against `nvllm:gb10-phaseD2e` to confirm 8/8 reproduces there. If yes, bisect deps.
2. **Byte-load vectorization (next perf lever).** `decode-small` at 45.83 ms/call is still **~20× above the 2.23 ms 4×-traffic roofline**. The tile retune delivered the predicted 1.5× band; the audit's 4× prediction for `_ld_global_u8 → _ld_global_b32` vectorization remains the highest-leverage single change toward closing the gap to unfused.
3. **SMEM pipelining (D3b scope).** Compose with whichever decode tile wins post-investigation.

## Reproduce

```bash
cd /home/natfii/docker/nvllm
scripts/phase_d3a_sweep.sh
# Per-preset: ~10-12 min. Full sweep: ~45 min.
# Requires nvllm:gb10-phaseD3a image.
# Individual preset: CUTE_MLP_TILE=<name> docker run ... (see script for full flags).
```

## Artifacts

Per-preset subdirectory `<preset>/` contains:
- `profiler_out_0.txt` — torch-profiler textual table (MLP + attn kernel rows)
- `rank0.*.pt.trace.json.gz` — full trace for timeline analysis
- `decode_log.txt` — vLLM server log
- `gsm8k_<preset>.{json,log,exit}` — GSM8K sanity results + raw per-question output + harness exit code sidecar
- `workload_{1..4}.json` — raw completion responses from the 4×128 tok profiler workload
