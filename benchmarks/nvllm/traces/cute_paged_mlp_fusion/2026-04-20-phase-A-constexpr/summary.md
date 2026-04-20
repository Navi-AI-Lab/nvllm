# Phase D3a sweep — MLP decode-tile retune results

**Spec:** `docs/superpowers/specs/2026-04-19-phase-d3a-mlp-decode-retune-design.md`
**Image:** `nvllm:gb10-phaseD3a`
**Model:** `natfii/Qwen3.5-27B-NVFP4-Opus-GB10`
**Config:** `max-model-len=65536, max-num-seqs=4, kv-cache-dtype=fp8_e4m3, cudagraph_mode=PIECEWISE, CUTE_ATTN_FUSION=1, CUTE_MLP_FUSION=1, --language-model-only, --gpu-memory-utilization 0.80`
**Workload:** 4 concurrent × 128 tok, temperature=0, ignore_eos=true.

## Per-preset results

| preset | grid @ nat=4 | CTAs | ~waves @ 96r | slices/CTA | MLP μs/call | MLP Self CUDA (s) | attn fused Self CUDA (s) | s/Q | GSM8K | verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| prefill-legacy | (8,8,4) | 256 | 2.7 | 9 | 68.58 ms | 139.36 | 32.01 | 8.7 | 8/8 | baseline (from D2e) |
