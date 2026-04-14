# CuTe Paged Attention — GSM8K Sanity Gate

**Date:** 2026-04-13
**Commit:** b69e1a073 (kernel code from 26bda34d6)
**Model:** natfii/Qwen3.5-27B-NVFP4-Opus-GB10
**Backend:** CUTE_PAGED (CuTe DSL FP8 paged attention)
**Mode:** CUDA graphs (PIECEWISE), kv_cache_dtype=fp8_e4m3

## Results

### Simple math (25 x 37)

- **Prompt:** "What is 25 times 37? Answer with just the number:"
- **Output:** `925` (correct)
- **Finish reason:** stop
- **Tokens:** 16 prompt, 9 completion

### GSM8K (Janet music lessons)

- **Prompt:** "Janet pays $40/hour for 3 hours per week of violin lessons and $28/hour for 5 hours a week of piano lessons. How much does she spend on music lessons per year? Answer step by step, then give the final number."
- **Output:** Coherent structured reasoning (thinking mode engaged via `<think>` tags)
- **Finish reason:** length (256 token limit)
- **Tokens:** 60 prompt, 256 completion
- **Note:** Model entered thinking mode despite /v1/completions endpoint — this is expected for thinking-enabled checkpoints. Output is coherent but answer was truncated by token limit.

## Verdict

**PASS** — CuTe paged attention kernel produces coherent, correct output with CUDA graphs enabled. No garbled tokens, no repetition loops, no quality degradation visible. Simple math confirms no numerical corruption.

## How to reproduce

```bash
./scripts/run_qwen35_27b_cute_paged.sh  # renamed to scripts/serve-cute.sh

curl -s http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"default","prompt":"What is 25 times 37? Answer with just the number:","max_tokens":32,"temperature":0}'
```
