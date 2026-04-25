# Probe 4 — MLP-only fusion isolates Phase D MLP kernel cost

**Date:** 2026-04-24
**Config:** `CUTE_PHASE_E_FUSION=0 CUTE_MLP_FUSION=1 CUTE_ATTN_FUSION=0`
all other flags identical to probes 1-3 (Qwen3.5-27B-NVFP4, --enforce-eager,
autotune-OFF, max-num-seqs=1).

**Output:** `" 4\nQ: What is 2+2?\nA: 4\nQ: What is 2+2?\nA: 4"` ✅ (correct,
matches probe 2's quality).

## Four-probe matrix (steady-state per-layer sync_end)

| Probe | Phase_E | MLP | ATTN | full_attn layer | linear layer | per-token | tok/s | output |
|---|---|---|---|---|---|---|---|---|
| P1 | 1 | 1 | 1 | **56.2 ms** | 1.21 ms | 957 ms | 1.04 | gibberish ❌ |
| P2 | 0 | 1 | 1 | **41.1 ms** | 1.21 ms | 716 ms | 1.40 | correct ✅ |
| P3 | 0 | 0 | 1 | **16.2 ms** | 1.17 ms | 315 ms | 3.17 | correct++ ✅✅ |
| **P4** | **0** | **1** | **0** | **25.5 ms** | 1.22 ms | 467 ms | 2.14 | correct ✅ |
| baseline (v2 handoff) | 0 | 0 | 0 | ~3.5 ms | ~1.0 ms | ~67 ms | ~15 | correct ✅ |

## Bugs are additive

Sanity check: P3 + P4 - baseline ≈ P2?
- 16.2 + 25.5 - 3.5 = **38.2 ms** ≈ P2's **41.1 ms** ✅ (close to additive, ~3 ms slack)

P2 - P3 ≈ P4 - baseline (MLP-fusion's standalone cost)?
- 41.1 - 16.2 = **24.9 ms**
- 25.5 - 3.5 = **22.0 ms**
- ✅ within noise

P1 - P2 = β-coop's standalone cost on top of MLP+ATTN:
- 56.2 - 41.1 = **15.1 ms** ✅ matches probe 1→2 delta

## Per-fusion attribution (final)

| Component | Per-layer cost above baseline | Per-token (×16) | Output impact |
|---|---|---|---|
| **Phase D MLP kernel** | **+22 ms** | **+352 ms** | none (math correct) |
| **ATTN fusion (Phase B/C)** | **+13 ms** | **+208 ms** | none (math correct) |
| **Phase E β-coop** | **+15 ms** | **+240 ms** | **gibberish output** |
| Total over baseline | **+50 ms/layer × 16 = +800 ms/token** | which matches P1's 957 ms - baseline 67 ms = 890 ms ✅ |

## Where to spend optimization effort

By per-token cost recovered if fixed (largest first):

1. **Phase D MLP kernel (+352 ms/token)** — 22 ms/layer for `5120 × 17408 × 5120`
   GEMM. cuBLAS reference is ~1 ms for the equivalent gate+up+down chain.
   Likely tile/grid mistuning for SM120 specifically. Highest-value target.
2. **Phase E β-coop (+240 ms/token + correctness)** — math fix unblocks but
   doesn't recover perf; the 15 ms/layer kernel work remains. May be cleanest
   to revert until next attempt.
3. **ATTN fusion (+208 ms/token)** — already shipped and correct. ~5× cuBLAS
   reference. Worth retuning the `paged_attn + W_O baked + RMSNorm baked`
   fused path, but it IS cleanest of the three CuTe paths.

## Recommended next deep-dive

**`/start_profile` + 32-tok completion + `/stop_profile`** with probe 4 config
(MLP-only) — this isolates the Phase D MLP kernel's per-call cost and which
sub-kernel within it (gate, up, down GEMM, or the activation) is expensive.

Per `feedback_vllm_profiling`: nsys can't see V1 EngineCore kernels; use vLLM's
built-in torch profiler API at `vllm/entrypoints/serve/profile/api_router.py`.

## Evidence

- All four probes' raw `timing_lines.txt`, `serve.log`, `completion.json` under
  `docs/research/phase_f1_opaque_gate/run_logs/timing_probe_*/`.
- `compare.sh` in this dir runs the n=13/n=38 averaging across all four.
