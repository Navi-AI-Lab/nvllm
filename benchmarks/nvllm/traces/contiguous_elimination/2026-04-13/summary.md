# Benchmark: .contiguous() Copy Elimination — CuTe vs FlashInfer

**Commit:** `748c9695c`
**Date:** 2026-04-13
**Model:** natfii/Qwen3.5-27B-NVFP4-Opus-GB10
**Config:** max-num-seqs=2, max-model-len=4096, kv-cache-dtype=fp8_e4m3, enforce-eager=true, gpu-memory-utilization=0.75

## Methodology

Sequential serving benchmark: 10 requests, 256 max_tokens each, temperature=0.
Same prompt for both backends. No concurrency (requests fired serially).
Both backends warmed up before measurement.

## Results

| Backend | Total Tokens | Wall Time (s) | tok/s | Avg Latency (s) |
|---------|-------------|---------------|-------|-----------------|
| CuTe Paged (zero-copy) | 2560 | 266.8 | 9.6 | 26.7 |
| FlashInfer | 2560 | 238.8 | 10.7 | 23.9 |
| **Delta** | — | — | **-10.3%** | **+11.7%** |

## Context: Before vs After .contiguous() Fix

The previous nsys profile (commit `6e78e8ec3`) showed `.contiguous()` copies
consuming 5.77s / 87.7% of GPU time per profiling window. With copies eliminated
(commit `748c9695c`), CuTe is now within 10% of FlashInfer end-to-end.

The remaining gap is the attention kernel itself: CuTe DSL decode kernel runs at
~244μs vs FlashInfer's ~7.25μs. However, attention is <1% of total GPU time
(FP4 GEMM dominates at 81%), so the kernel speed difference only accounts for a
small fraction of the 10% e2e delta. The rest is likely JIT overhead, different
codepaths in the fallback prefill path, or measurement noise from thinking mode
token variation.

## How to Reproduce

```bash
# CuTe Paged backend
docker run -d --gpus all --privileged --name nvllm-test \
  -p 8000:8000 -v /data/models:/data/models \
  nvllm:gb10 serve \
    --model natfii/Qwen3.5-27B-NVFP4-Opus-GB10 \
    --attention-backend CUTE_PAGED \
    --kv-cache-dtype fp8_e4m3 --enforce-eager \
    --max-model-len 4096 --max-num-seqs 2 \
    --gpu-memory-utilization 0.75

# FlashInfer backend (same config, different --attention-backend)
# ... --attention-backend FLASHINFER ...

# Bench: 10 sequential requests, 256 tokens each
curl -s http://localhost:8000/v1/completions \
  -d '{"model":"natfii/Qwen3.5-27B-NVFP4-Opus-GB10",
       "prompt":"Explain the implications of quantum entanglement...",
       "max_tokens":256,"temperature":0}'
```
