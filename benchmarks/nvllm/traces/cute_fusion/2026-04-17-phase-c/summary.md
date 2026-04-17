# Phase C ship-gate evidence

**Date:** 2026-04-17
**Commit:** `cbfadb6a9` (`cbfadb6a9104ea49da1c5a8cbd118cb157662978`)
**Branch:** `feat/own-the-stack-phase-c`
**Image:** `nvllm:gb10-ots-phaseC` (ID `6b6c71c9daa3`, 20GB)
**Model:** `natfii/Qwen3.5-27B-NVFP4-Opus-GB10`
**Backend:** CuTe paged, PIECEWISE CUDA graphs
**KV cache:** fp8_e4m3
**Context:** 65536 tokens, max_num_seqs=4
**Env:** `CUTE_DEBUG_FUSION=1`

## Ship gate

GSM8K sanity: **8/8 correct (100%) — PASS**

```
  Q1: OK  expected=72  got=72  (36.6s — first-request warmup)
  Q2: OK  expected=10  got=10  (6.1s)
  Q3: OK  expected=5   got=5   (6.4s)
  Q4: OK  expected=42  got=42  (6.3s)
  Q5: OK  expected=624 got=624 (6.1s)
  Q6: OK  expected=35  got=35  (6.4s)
  Q7: OK  expected=48  got=48  (6.2s)
  Q8: OK  expected=16  got=16  (6.2s)
```

Matches Phase B baseline (commit `4110dc77a`, image `nvllm:gb10-ots`) — same 8/8, same ~6 s warm latency per prompt after first-request JIT warmup.

## Tier-1 host-side evidence

`notebooks/nvllm/layers_smoke_tests.py` — **5/5 passed** on host CPU:
- `test_rms_block_level_equivalence` — `Qwen3_5RMSNorm(5120)` vs `GemmaRMSNorm(5120)`, `rtol=0 atol=0` on bf16 input.
- `test_rms_head_dim_equivalence` — same at `hidden_size=256` (q/k_norm shape).
- `test_rms_fused_residual_equivalence` — fused add+norm path.
- `test_customop_registry_no_collision` — `qwen3_5_rms_norm` and `gemma_rms_norm` coexist in `op_registry`.
- `test_mlp_class_surface` — `Qwen3_5MLP.__init__` has exactly `(hidden_size, intermediate_size, hidden_act, quant_config, prefix)` — no `expert_gate`, no `reduce_results`.

## Tier-2 lint evidence

- `pre-commit run --files ...` — all hooks pass (ruff, format, typos, SPDX, etc.).
  SPDX-header hook auto-added canonical vLLM-project copyright line to the 4 new files during first run; re-run clean.
- `pre-commit run mypy-3.10 --all-files --hook-stage manual` — zero errors in Phase C files.
  11 pre-existing errors remain in `vllm/v1/attention/backends/cute_paged/{kernel.py,_backend.py}` and `vllm/v1/core/kv_cache_utils.py` — identical to Phase B baseline, not introduced by this refactor.

## Tier-3 image-content verification

```bash
docker run --rm --gpus all --entrypoint python nvllm:gb10-ots-phaseC \
  -c "from vllm.nvllm.layers import Qwen3_5RMSNorm, Qwen3_5MLP; \
      from vllm.model_executor.custom_op import op_registry; \
      print('IMPORT_OK'); \
      print('qwen3_5_rms_norm in registry:', 'qwen3_5_rms_norm' in op_registry)"
# → IMPORT_OK
# → qwen3_5_rms_norm in registry: True
```

## Repro

```bash
git checkout feat/own-the-stack-phase-c
docker build -f docker/Dockerfile.gb10 -t nvllm:gb10-ots-phaseC .
NVLLM_IMAGE=nvllm:gb10-ots-phaseC CUTE_DEBUG_FUSION=1 ./scripts/serve-cute.sh
# wait for "Application startup complete."
.venv/bin/python scripts/gsm8k_sanity.py --api http://localhost:8000/v1 --model default
```

## Rollback

Phase B image `nvllm:gb10-ots` stays on disk as the rollback snapshot.
`nvllm:gb10-preshim-phaseC-20260417` tag also points at the same Phase B image ID (`240c48497d10`).

```bash
docker tag nvllm:gb10-ots nvllm:gb10
git checkout main
```

No data migration, no weight-format changes, no config-schema changes — rollback is instant.

## Kernel durations

No kernel changes in Phase C — this is a pure layer-ownership refactor.
Decode kernel durations match the Phase B baseline
(see [`../2026-04-17-own-the-stack/summary.md`](../2026-04-17-own-the-stack/summary.md)).
No new `.nsys-rep` required per AGENTS.md §4 — new traces are only mandatory for perf claims,
not for semantically-equivalent refactors.
