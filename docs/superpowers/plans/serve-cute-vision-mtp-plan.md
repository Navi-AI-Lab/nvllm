# CUTE Vision + MTP 64K Bring-Up Plan

## Summary

Bring up `sakamakismile/Huihui-Qwen3.6-27B-abliterated-NVFP4-MTP` on GB10 with vision enabled, MTP speculative decode, 64K context, and `CUTE_PAGED` attention.

The current `/tmp/serve-cute-vision-mtp.sh` script already matches the intended first-run defaults: `MAX_NUM_SEQS=1`, `MTP_TOKENS=1`, `SERVE_GPU_UTIL=0.70`, fusions off, and `CUTE_WO_SPLIT=1`.

Primary fix before execution: move the script to `scripts/serve-cute-vision-mtp.sh` or adjust its `source "$(dirname "$0")/common.sh"`, because `/tmp/common.sh` is absent.

## Cross-Match Findings

Matches the plan:

- `HF_MODEL` defaults to `sakamakismile/Huihui-Qwen3.6-27B-abliterated-NVFP4-MTP`.
- `--language-model-only` and `--limit-mm-per-prompt` are absent, so vision is enabled.
- `--quantization modelopt` and `--speculative-config "$SPEC_CONFIG"` are present.
- `KV_CACHE=fp8_e4m3`, `ATTN_BACKEND=CUTE_PAGED`, and `MAX_MODEL_LEN=65536` are set.
- `MAX_NUM_SEQS=1`, `MTP_TOKENS=1`, and `SERVE_GPU_UTIL=0.70` are the defaults.
- `CUTE_MLP_FUSION=0`, `CUTE_ATTN_FUSION=0`, `CUTE_PHASE_E_FUSION=0`, and `CUTE_WO_SPLIT=1` are the defaults.
- The validation order is captured in the script header.

Fix before run:

- Place the script at `scripts/serve-cute-vision-mtp.sh` so `source "$(dirname "$0")/common.sh"` resolves to `scripts/common.sh`.

Keep as tunable:

- Leave `--max-num-batched-tokens 65536` for the first attempt.
- If CUDA graph capture OOMs, rerun with that lowered to `16384`.

## Implementation Changes

- Move `/tmp/serve-cute-vision-mtp.sh` to `scripts/serve-cute-vision-mtp.sh` after review.
- Do not modify existing `scripts/serve.sh` or `scripts/serve-cute.sh` during initial bring-up.
- Optional script polish before moving:
  - Add `MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-65536}"`.
  - Replace `--max-num-batched-tokens 65536` with `--max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS"`.

## Bring-Up Runbook

Run from repo root after moving the script:

```bash
./scripts/serve-cute-vision-mtp.sh
```

Validation order:

1. Confirm server loads and resolves `Qwen3_5ForConditionalGeneration`.
2. Run short text-only completion.
3. Run text-only 8K prompt.
4. Run 64K admission check with `MAX_NUM_SEQS=1`.
5. Confirm MTP n=1 is active and acceptance is nonzero in logs or metrics.
6. Send one image request with MTP still enabled.
7. If image path fails, rerun with `--debug`; if still unclear, temporarily remove `--speculative-config` to isolate vision.
8. Only after all above pass, test `MTP_TOKENS=3`.
9. Only after `MTP_TOKENS=3` passes, test `MAX_NUM_SEQS=2`.

## Failure Handling

- If the script fails immediately with missing `common.sh`, it has not been moved to `scripts/`.
- If model load fails on quantization, retry with `--quantization modelopt_fp4` only as a diagnostic; preferred target remains `modelopt`.
- If CUDA graph capture OOMs, lower `--max-num-batched-tokens` to `16384`.
- If `CUTE_PAGED` plus vision fails before decode, rerun with `--debug` to separate graph capture from model/runtime errors.
- If MTP fails but vision works, keep the vision path and temporarily remove `--speculative-config` for isolation.
- Do not enable prefix caching.
- Do not enable CUTE fusions until MTP n=1 is stable.

## Test Plan

Smoke tests:

- Text-only short chat completion.
- Image chat completion.
- 8K prompt.
- 64K admission.

Spec decode tests:

- MTP n=1 acceptance is nonzero.
- MTP n=3 only after n=1 stability.
- Compare a few deterministic prompts against non-MTP output for gross regressions.

Performance evidence:

- Do not make performance claims until nsys traces are captured under `benchmarks/nvllm/traces/...` with exact commands and summaries per AGENTS.md.

## Assumptions

- Initial target is bring-up correctness, not throughput tuning.
- `MAX_NUM_SEQS=1`, `MTP_TOKENS=1`, fusions off, and `SERVE_GPU_UTIL=0.70` remain the first-run profile.
- `CUTE_WO_SPLIT=1` stays enabled because it is the current production-blessed K-parallel decode path.
- This plan artifact lives in `/tmp` while the script is in `/tmp`, then both move into repo together after review.
