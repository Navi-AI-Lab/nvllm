# Phase D close-out on ig1/Qwen3.5-27B-NVFP4 (non-distilled)

**Date:** 2026-04-20 evening
**Commit:** 49b06fbb2 (image `nvllm:gb10`)
**Model:** `ig1/Qwen3.5-27B-NVFP4` (non-distilled, llm-compressor official Qwen3.5 VL recipe, 1024 calibration samples)
**Backend:** CUTE_PAGED, FP8 E4M3 KV, PIECEWISE CUDA graphs, `--max-num-seqs 4`, `--max-model-len 65536`
**Hardware:** DGX Spark GB10 (SM121)

## TL;DR

1. **Deterministic per-CTA slot + fixed-order gather fix (commit 60d65c9f0) works.**
   3 fused runs produced byte-identical GSM8K output.

2. **Kernel math is correct in all fusion configurations.**
   Apparent Q2 "regression" under both-fused was a parser artifact in
   `scripts/gsm8k_sanity.py` (takes the first number; model emits
   `600/60 = 10\n#### 10` which is arithmetically correct but starts
   with 600). Re-scoring raw outputs with a `####`-aware extractor
   gives 8/8 in every fusion mode.

3. **Real Phase D blocker: 6.5× decode slowdown with both fusions on**
   (17.0 s/q vs 2.6 s/q unfused, `max_tokens=16`). MLP fusion alone
   is 5.0× slower; attn fusion alone is 2.6× slower. Kernel math is
   fine — the regression is infrastructure overhead.

## Isolation matrix (single request, temperature=0, `max_tokens=16`)

| Config | GSM8K (original parser) | GSM8K (fixed parser) | Time/q | Q2 raw |
|---|---|---|---|---|
| Unfused (both OFF) | 8/8 | 8/8 | **2.6 s** | `'10. The answer is 10.'` |
| MLP only | 8/8 | 8/8 | 13.0 s | `'10.  10\nThe answer is 10.'` |
| Attn only | 8/8 | 8/8 | 6.8 s | `'10. The answer is 10.'` |
| Both fused | 7/8 | **8/8** | 17.0 s | `'600/60 = 10\n#### 10'` |

All three "both-fused" runs byte-identical → 2026-04-20 fix holds.

Performance roughly superposes (both ≈ attn_only + mlp_only − unfused),
suggesting the overhead is per-call opaque-op / graph-capture costs
paid independently per fused kernel.

## Kernel math verification (`CUTE_DEBUG_FUSION=1`)

With both fusions on and `CUTE_DEBUG_FUSION=1`, `kernel.py`'s
Phase B (`wo_output` vs Python `attn @ W_O.T`) and Phase C
(residual + RMSNorm vs Python ref) diff-against-reference checks
fired on every fusion-active attn layer × every decoded token across
two Q2 requests:

```
close=True:  496
close=False: 0
```

Per-layer `diff.max` below 1e-3 at every call. Attn kernel math is
numerically correct on ig1. MLP kernel math was validated indirectly
via MLP-only GSM8K 8/8.

Full decode log (one compile + two Q2 requests × 64 layers × 2
phases): `decode_log_debug_fusion.txt` saved to the trace dir.

## Raw evidence files (already in benchmarks/nvllm/traces/cute_fusion_determinism/2026-04-20-ig1-fused/)

- `run1.json` / `run2.json` / `run3.json` — 3 byte-identical fused
  GSM8K runs (original parser: 7/8 each, fixed parser: 8/8 each).
- `run_unfused.json` — both fusions off baseline, 8/8.
- `run_mlp_only.json` — MLP only, 8/8.
- `run_attn_only.json` — Attn only, 8/8.
- `decode_log_debug_fusion.txt` — `CUTE_DEBUG_FUSION=1` full decode
  log, 1765 lines, contains all 496 `close=True` lines.

## Parser fix sketch (not yet landed)

Original extractor at `scripts/gsm8k_sanity.py:150` takes
`re.findall(...)[0]` — the first number. Under both fusions, the
model writes verbose arithmetic `"600/60 = 10\n#### 10"`, so the
first number is the intermediate step (600), not the final answer (10).

Fix hierarchy (verified on all 6 saved runs → every config 8/8):

1. If `#### N` canonical GSM8K marker present → use it.
2. Else, strip everything after a next-question boundary
   (`\n\s*(Q:|Question:|####)`) and take the first number of what
   remains.

Left unlanded because changes to `gsm8k_sanity.py` affect historical
comparability; user approval required.

## Perf regression — candidates for next session

Additive superposition suggests each fused kernel costs its own
per-call overhead on top of its GPU work. Ranked candidates:

1. **Verify CUDA graph capture of the fused path.** Prior fix
   (memory `project_cute_not_capturing`, 2026-04-16) threaded
   `stream=...` through `.launch()` so CuTe kernels could capture
   under FULL graphs. But PIECEWISE splits at the torch-native
   `unified_attention` op, not at `vllm::cute_mlp_forward` — verify
   the opaque op is actually captured inside a graph piece, not
   running as a between-pieces eager launch.

2. **Measure Python-body replay overhead.** `cute_mlp_forward` is
   `direct_register_custom_op`-registered; the body (dict lookup +
   getattr + branch + zero_() + kernel launch) re-runs in Python on
   every graph replay. 64 layers × Python dispatch × per-token could
   add up. Time a single decode forward with `torch.profiler` to
   localize.

3. **Opaque-op absent from `splitting_ops`.** Config shows
   `splitting_ops` contains only upstream vLLM attention/kv-cache
   ops, not `vllm::cute_mlp_forward`. If this is forcing the whole
   MLP region to be outside captured pieces, that would explain the
   MLP-only slowdown. Add the op to splitting_ops and re-test.

## Phase D verdict

- **Phase D correctness** — SHIPPED and verified on non-distilled.
  Determinism fix works; kernel math is correct. Both the
  Opus-distilled knife-edge non-determinism (pre-2026-04-20) and
  the ig1 "Q2=600" apparent regression were diagnostic artifacts,
  not kernel bugs.

- **Phase D performance** — NOT shippable as default-on.
  6.5× slowdown is a real regression that must be explained before
  `CUTE_MLP_FUSION=1 CUTE_ATTN_FUSION=1` can become the serve
  default. Document as the immediate follow-up work; Phase E
  (residual fusion) stays blocked behind this.
