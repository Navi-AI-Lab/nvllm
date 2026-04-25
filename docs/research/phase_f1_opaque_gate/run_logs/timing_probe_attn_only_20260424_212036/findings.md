# Timing probes 1-3 — fused-path bottlenecks isolated by fusion-flag bisection

**Date:** 2026-04-24
**Method:** Three back-to-back single-32-token completions, fused+eager+autotune-OFF,
mount-overlay of `qwen3_5.py` instrumented with per-checkpoint `time.perf_counter()`
and end-of-layer `torch.cuda.synchronize()`. Each probe toggles one or two
`CUTE_*_FUSION` env vars while keeping the rest constant.

## The three probes

| Probe | PHASE_E | MLP | ATTN | Output | Steady full_attn layer |
|---|---|---|---|---|---|
| **1** | 1 | 1 | 1 | `" 2                              "` ❌ | **56.2 ms** |
| **2** | 0 | 1 | 1 | `" 4\nQ: What is 2+2?\nA: 4\n..."` ✅ | **41.1 ms** |
| **3** | 0 | 0 | 1 | `" 4\n\nQ: What is 3+3?\nA: 6\n\nQ: What is 4+4?\nA: 8"` ✅ | **16.2 ms** |

Linear_attention layers stable across all three probes at ~1.2 ms/layer (78 ms/token total).

## Per-bottleneck attribution

Subtract probes pairwise to isolate each fusion's per-layer cost:

| Component | Per-layer cost | Per-token cost (×16 fused layers) | Notes |
|---|---|---|---|
| Phase E β-coop kernel | **+15 ms** | **+240 ms** | **Also: produces gibberish.** Math bug + perf bug. |
| Phase D MLP fusion kernel | **+25 ms** | **+400 ms** | Pure perf bug. Output stays correct. |
| ATTN fusion (Phase B/C, W_O+RMSNorm) | ~16 ms | ~256 ms | Output correct. Slower than cuBLAS reference (~3 ms/layer ⇒ ~13 ms/layer of regression) but the published "Phase B/C SHIPPED" win was real per-kernel — the slowness here is the un-fused MLP path being faster than fused. |
| (Reference: cuBLAS unfused) | ~3-4 ms | ~50-65 ms | from the v2 handoff's 96.0% 50-Q at ~15 tok/s |

## Per-token math

Sum 16 full_attn × per-layer + 48 linear_attn × 1.2 ms:

| Config | per-token | tok/s |
|---|---|---|
| Probe 1 (all fusion) | ~957 ms | **1.05 tok/s** ❌ |
| Probe 2 (β-coop off) | ~714 ms | **1.40 tok/s** ✅ |
| Probe 3 (only ATTN) | ~314 ms | **3.18 tok/s** ✅ |
| **Unfused baseline (v2 handoff)** | ~67 ms | **~15 tok/s** ✅ |

**Probe 3 is 4-5× faster than probe 1 AND produces correct output**, but still ~5× slower than unfused. The remaining gap is the ATTN fusion itself running ~16 ms/layer where cuBLAS reference does ~3 ms.

## Three independent bugs in the fused path

1. **Phase E β-coop ε-epilogue** — `vllm/v1/attention/backends/cute_paged/_phase_e/`
   - **Math broken**: produces `" 2                              "` for "What is 2+2?"
   - **Perf**: +15 ms/layer (~+240 ms/token, 25% of probe-1 budget)
   - **Status**: ship-blocker. Disable until rewritten OR rollback to pre-Phase-E.
   - Confirms `project_phase_e_phantom_speedup` and `project_phase_e_beta_math_bug` —
     β was not just orphaned-output, it's actively destructive when consumed.

2. **Phase D MLP kernel** — `vllm/v1/attention/backends/cute_paged/_mlp_op.py` +
   `_kernels/mlp_kernel.py`
   - **Math correct** on probe 2.
   - **Perf**: +25 ms/layer (~+400 ms/token, 42% of probe-1 budget!) when cuBLAS
     gate+up GEMM + down GEMM run in <1 ms together.
   - This is the **biggest single perf opportunity**. A working Phase D kernel would
     unlock most of the gap.

3. **ATTN fusion (Phase B/C, W_O + RMSNorm baked into attention kernel)** —
   `vllm/v1/attention/backends/cute_paged/_backend.py` impl forward + paged_attention_forward
   - **Math correct.**
   - **Perf**: ~16 ms/layer baseline cost vs cuBLAS reference ~3 ms.
   - This is the kernel that `project_cute_paged_bench` reported as shipped.
     The shipped numbers were per-kernel μs (real, not phantom) but didn't include
     the cost of the W_O + RMSNorm baked work — apparently this fused path is
     ~5× slower than the unfused W_O cuBLAS GEMV + Triton RMSNorm chain.
     **Worth re-tracing this separately to confirm.**

## Outputs in detail (qualitative correctness)

- **Probe 1**: `" 2                              "` — produces "2" then 30 padding
  spaces. `2` is wrong (gold=4). Pattern matches the v2 handoff's fused-leg
  Q1 ERROR (empty pred after timeout).
- **Probe 2**: `" 4\nQ: What is 2+2?\nA: 4\nQ: What is 2+2?\nA: 4"` — answer is 4 (correct);
  loops the prompt because `temperature=0` + no stop tokens.
- **Probe 3**: `" 4\n\nQ: What is 3+3?\nA: 6\n\nQ: What is 4+4?\nA: 8"` — model is
  cleanly extrapolating sequence: 2+2=4, then synthesizes 3+3=6, then 4+4=8.
  Best output of the three. Suggests probe 3 has the cleanest numerical path.

## What this rules in / out

✅ **In:** All three CUTE_*_FUSION layers contribute to the regression in some way.
✅ **In:** Phase E β-coop is the biggest bug (math + perf).
✅ **In:** Phase D MLP kernel is the biggest perf-only bug.
❌ **Out:** Python overhead, opaque-op infrastructure (~1% of budget).
❌ **Out:** Graph-capture-vs-fusion interaction (eager and PIECEWISE both ~1 tok/s).

## Recommended next actions

1. **Production config update** — set the default serve config to all-fusion-OFF
   (`CUTE_*_FUSION=0`). The 96.0% 50-Q working baseline IS the unfused config.
2. **Update memory `project_phase_e_phantom_speedup`** — β-coop is not phantom;
   it actively breaks math. Memory wording needs strengthening.
3. **Update memory `project_fused_path_perf_collapse`** — root cause now known:
   Phase D MLP +25ms/layer + Phase E β-coop +15ms/layer + ATTN fusion +13ms/layer
   over cuBLAS reference, ~10× cumulative.
4. **Decide on Phase D MLP kernel future**:
   - Option A: nsys-profile probe 3 (ATTN-only) and find the actual GEMM kernel
     time vs theoretical, retune tile sizes.
   - Option B: revert MLP fusion until tuning is done; ship just ATTN fusion.
5. **Decide on Phase E β future**:
   - Option A: deep-dive the math bug (consume copy is wrong + ε-epilogue layout
     needs audit) — may unlock more wins after fix.
   - Option B: revert Phase E entirely; it's been a long net negative.

## Evidence

- `/home/natfii/docker/nvllm/docs/research/phase_f1_opaque_gate/run_logs/timing_probe_20260424_210752/` — probe 1 (all fusion)
- `/home/natfii/docker/nvllm/docs/research/phase_f1_opaque_gate/run_logs/timing_probe_phaseE_off_20260424_211547/` — probe 2 (β off)
- `/home/natfii/docker/nvllm/docs/research/phase_f1_opaque_gate/run_logs/timing_probe_attn_only_20260424_212036/` — probe 3 (ATTN only) ← this dir
- Each contains `serve.log`, `timing_lines.txt` (120 lines), `completion.json`, `runner.log`, and `run.sh`.

## Reproduce

Each probe's `run.sh` is self-contained. They mount-override
`vllm/nvllm/models/qwen3_5.py` from the host (which currently carries
the CUTE_DEBUG_TIMING instrumentation from this session) into the
running container — no rebuild needed.

```bash
$DIR/run.sh   # for any of the three dirs
```
