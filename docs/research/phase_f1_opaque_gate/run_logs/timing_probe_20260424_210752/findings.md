# Timing probe — fused-path perf collapse localized to fused full_attention kernels

**Date:** 2026-04-24
**Config:** Qwen3.5-27B-NVFP4, fused (CUTE_*_FUSION=1), --enforce-eager,
autotune-OFF, max-num-seqs=1, single 32-token completion ("Q: What is 2+2?\nA:").
Output: " 2                              " (degenerate — 1 real token then padding,
matches v2 handoff fused-leg gibberish pattern).

## Steady-state per-layer cost (token 2+, after JIT compiles drained)

| Layer type | self_attn dispatch | mlp_op dispatch | sync_end (drain) | Total/layer |
|---|---|---|---|---|
| **fused full_attention** | ~670 µs | ~14 µs | **~55,000 µs** | **~57 ms** |
| linear_attention | ~280 µs | ~145 µs | ~1,200 µs | ~1.6 ms |

## Per-token total (steady state)

- 16 full_attention layers × 57 ms = **912 ms**
- 48 linear_attention layers × 1.6 ms = **77 ms**
- **Total: ~990 ms/token ≈ 1.0 tok/s** ✅ matches the v2 handoff's 0.8 tok/s

## Diagnosis

**The bug is in the GPU kernels of the fused full_attention path.**
- Python+dispatch overhead = ~700 µs (~1.2% of total) → NOT a Python issue
- GPU work for one fused full_attention layer = ~55 ms
- Unfused/cuBLAS reference for the same layer ≈ 3-4 ms
- **The fused full_attention kernel chain is ~15-18× slower than the cuBLAS reference**

Suspect: the β-coop kernel ε-epilogue (Phase E does next-layer's `input_layernorm`
gamma multiply + MLP). The MLP portion is `hidden=5120 → intermediate=17408 →
hidden=5120` — a large GEMM where bad tile/grid selection could plausibly cost
this much.

## First-token startup cost (one-shot per server)

| Event | Cost |
|---|---|
| Each first-encounter MLP-only path JIT compile (legacy MLP, layers 3,7,...,63) | ~960 ms × 16 = **15.4 s** |
| First β-coop kernel JIT compile (token 2, layer 3) | **~43 s** |

These are one-shot warmup costs — they don't affect steady-state tok/s but
explain why the 50-Q fused leg in the v2 handoff timed out questions early
(server spent first ~60s of token generation just JIT-compiling).

## What this rules in / out

✅ **In:** kernel-side perf bug (β-coop or its sub-kernels)
✅ **In:** GEMM tile config mismatch for the MLP shape
❌ **Out:** Python per-step overhead
❌ **Out:** torch.empty_like / torch.zeros allocator thrashing
❌ **Out:** opaque-op infrastructure
❌ **Out:** graph-capture vs eager interaction (eager and PIECEWISE both ~1 tok/s)

## Next probe candidates (ranked)

1. **Disable just Phase E β** (`CUTE_PHASE_E_FUSION=0`, keep MLP+ATTN fusion).
   If steady-state jumps to 5+ tok/s, the bug is the β-coop ε-epilogue chain.
   If still ~1 tok/s, the bug is in the Phase D MLP kernel itself.
   Cost: ~8 min (rebuild not needed, env-var change + restart).

2. **Add sync between self_attn and mlp_op checkpoints.** Splits the 55 ms
   sync_end into "GPU work attributable to self_attn (β-coop kernel)" vs
   "GPU work attributable to mlp_op (Phase D fallback or β-lite consume)".
   Cost: ~5 min (single-line instrumentation edit + restart).

3. **Run nsys profile of one decode step** — gives per-kernel µs and which
   sub-kernel is slow. Cost: ~30 min.

## Evidence

- `serve.log` — full vLLM startup
- `timing_lines.txt` — 120 [CUTE_TIMING] log lines (all decoder layer forwards)
- `completion.json` — the actual "2 + spaces" output the fused path produced
- `runner.log` — orchestration

## Reproduce

```bash
DIR="docs/research/phase_f1_opaque_gate/run_logs/timing_probe_20260424_210752"
$DIR/run.sh
```

Mount-overlay applies the instrumented `qwen3_5.py` from the host without
needing a rebuild.
