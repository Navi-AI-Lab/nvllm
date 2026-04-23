# Task 17 — β-lite smoke (post-Task-16 backend wiring)

Post-Task-16 re-verification that β-lite dispatch still produces 8/8
GSM8K after commit `d9fe26634` added the β-coop branch in
`_backend.py::forward`.

## Setup

- Commit: `480f67dad` (Tasks 18/19/20 tests landed; same kernel + backend
  as `d9fe26634`).
- Image: `nvllm:gb10` SHA `0465e9d15ee0` (rebuilt 2026-04-23).
- Model: `ig1/Qwen3.5-27B-NVFP4` (non-distilled, per
  `memory:feedback_correct_model`).
- Backend: `CUTE_PAGED`, `fp8_e4m3` KV cache, PIECEWISE CUDA graphs.
- Env: `CUTE_PHASE_E_FUSION=1`, `CUTE_PHASE_E_PATH=lite`,
  `CUTE_MLP_FUSION=1`, `CUTE_ATTN_FUSION=1`.
- max_model_len=65536, max_num_seqs=4.

## Repro

```bash
CUTE_PHASE_E_FUSION=1 CUTE_PHASE_E_PATH=lite \
  bash scripts/serve-cute.sh

# Wait for "Application startup complete" in `docker logs -f nvllm`.

.venv/bin/python scripts/gsm8k_sanity.py \
  --api http://localhost:8000/v1 --model default \
  --save /tmp/gsm8k_beta_lite_rerun.json --timeout 120
```

## Result

**8/8 PASS**, matches the 2026-04-22 baseline from commit
`e797a9217`. Q1 = 52.2s (first-call JIT warmup), Q2-Q8 ~19s each.
Total ≈ 185s.

## Verdict

Task 16 β-coop dispatch wiring did not regress the β-lite path. Phase
E attach logged for all 64 layers with `resident_cap=96
num_seqs_coop_max=1`; 63 layers bound `emit_next_layernorm=True`, 1
last layer `emit_next_layernorm=False` — correct.

## Deferred

β-coop smoke (`CUTE_PHASE_E_PATH=coop`) not run this session.
Cooperative-launch composability with PIECEWISE CUDA graph capture is
untested (per `memory:reference_cute_cooperative_launch`) — needs a
focused session with graph-capture diagnostics ready in case it
breaks. Follow-up in next session.

Task 21 (nsys traces + summary.md + MEMORY.md) also deferred — needs
β-coop smoke first so all three traces (baseline, coop, lite) can
land together.
