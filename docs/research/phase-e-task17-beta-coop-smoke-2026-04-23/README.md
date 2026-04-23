# Task 17 — β-coop smoke (cooperative launch, num_seqs=1)

End-to-end smoke for the unified β-coop kernel (`PhaseE_Beta_Kernel`,
phases 0→4 in a single cooperative launch) under PIECEWISE CUDA graphs
on `ig1/Qwen3.5-27B-NVFP4`.

## Setup

- Commit: `f2691e2fe` (β-lite smoke evidence; β-coop wiring landed in
  `d9fe26634`; β-coop kernel in `44b9a980e`).
- Image: `nvllm:gb10` SHA `0465e9d15ee0`.
- Model: `ig1/Qwen3.5-27B-NVFP4` (non-distilled).
- Backend: `CUTE_PAGED`, `fp8_e4m3` KV cache, PIECEWISE graphs.
- Env: `CUTE_PHASE_E_FUSION=1 CUTE_PHASE_E_PATH=coop MAX_NUM_SEQS=1`.
- max_model_len=65536.

## Repro

```bash
CUTE_PHASE_E_FUSION=1 CUTE_PHASE_E_PATH=coop MAX_NUM_SEQS=1 \
  bash scripts/serve-cute.sh

# Wait for "Application startup complete" in `docker logs -f nvllm`.
# NOTE: first request after readiness triggers per-layer β-coop JIT
# compiles (~16 × 23 s ≈ 6 min) that exceed the 120 s per-prompt
# timeout — cold runs will show 3 early timeouts. Warm cache is 8/8.

.venv/bin/python scripts/gsm8k_sanity.py \
  --api http://localhost:8000/v1 --model default \
  --save docs/research/phase-e-task17-beta-coop-smoke-2026-04-23/gsm8k.json \
  --timeout 120
# Immediately re-run to collect warm-cache verdict:
.venv/bin/python scripts/gsm8k_sanity.py \
  --api http://localhost:8000/v1 --model default \
  --save docs/research/phase-e-task17-beta-coop-smoke-2026-04-23/gsm8k_warm.json \
  --timeout 120
```

## Result

- **Cold run (`gsm8k.json`):** 5/8 OK, 3/8 ERROR (Q1-Q3 timeout at 120 s
  each) — Q4-Q8 correct. The 3 timeouts are β-coop JIT compile warmup
  (not math failures); server-side GSM8K math was correct on every
  warmed call.
- **Warm run (`gsm8k_warm.json`, immediately after cold run): 8/8
  PASS.** All 8 prompts ~22.9 s each, matching β-lite timing.

**Verdict: β-coop is correct under PIECEWISE CUDA graphs** at
num_seqs=1 on `ig1/Qwen3.5-27B-NVFP4`. Gate 6.2.2 PASSED for β-coop.

## Phase E attach log

`phase_e_attach.log` — 64 layer attach entries confirming
`resident_cap=96 num_seqs_coop_max=1`, 63 layers bound
`emit_next_layernorm=True`, last layer `emit_next_layernorm=False`.
Also includes the 16 per-layer `Compiling PhaseE_Beta_Kernel β-coop
full (first call)…` retries between 21:19:21 and 21:24:59 — the
source of the cold-run timeout behavior.

## Phase E.1 follow-up

The β-coop kernel recompiles 16× on first invocation (once per
distinct per-layer closure — likely keyed on `rmsnorm_gamma_next`
pointer or similar per-layer constant). Each compile is ~23 s, so
total first-call warmup is ~6 min. Options for Phase E.1:

1. Lift per-layer constants out of the kernel closure so one compile
   serves all 64 layers.
2. Warmup inside `post_warmup` hook (after readiness, before first
   request) so users never hit the cost.
3. Document as a known cold-start cost and move on.

Per `memory:feedback_no_shortcuts`, option 1 is the right fix; option
2 hides the symptom without addressing it.
