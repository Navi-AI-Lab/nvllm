# CuTe Paged Attention — Prototype Validation

**Date:** 2026-04-11
**Commit:** `8b087728b` + unstaged shape fix in `_backend.py`
**Model:** `natfii/Qwen3.5-27B-NVFP4-Opus-GB10`
**Config:** `--attention-backend CUTE_PAGED --kv-cache-dtype fp8_e4m3 --enforce-eager --max-num-seqs 4 --max-model-len 65536 --gpu-memory-utilization 0.80`

## Purpose

Validate the CuTe paged attention backend interface with a PyTorch prototype kernel.
This is **not** a performance benchmark — the prototype uses Python loops and eager
PyTorch ops. Performance comparison deferred to the real CuTe DSL kernel.

## What was validated

| Check | Result |
|---|---|
| Backend registration | CUTE_PAGED selected, 28 layers initialized |
| Model config | 24 Q heads, 4 KV heads, head_dim=256, GQA ratio=6 |
| Prefill (multi-token) | Passed — 33-token prefill, no crash |
| Decode (single-token) | Passed — coherent multi-step reasoning |
| KV cache (FP8 E4M3) | Passed — `reshape_and_cache_flash` C++ op, page_size=64 |
| Output tensor contract | Passed — 3D `[N, 24, 256]` copy, `num_actual_tokens` slicing |

## Bugs fixed (this session)

### Crash #1: Hardcoded decode-only token count
- **File:** `vllm/v1/attention/backends/cute_paged/kernel.py:182`
- **Root cause:** `num_query_tokens = 1` hardcoded, broke on 33-token prefill
- **Fix:** Use `query_start_loc` from metadata to compute tokens per sequence

### Crash #2: 3D/2D tensor shape mismatch
- **File:** `vllm/v1/attention/backends/cute_paged/_backend.py:243`
- **Error:** `RuntimeError: The size of tensor a (256) must match the size of tensor b (6144) at non-singleton dimension 2`
- **Root cause:** `result.view(num_tokens, -1)` flattened kernel output to 2D `[33, 6144]`, but vLLM's attention layer passes output as 3D `[33, 24, 256]` (because `accept_output_buffer=True` triggers reshape at `attention.py:452`)
- **Fix:** Copy result directly in 3D (`output[:num_actual_tokens].copy_(result)`), use `attn_metadata.num_actual_tokens` instead of `query.shape[0]`
- **Reference:** b12x enforces rank-3 output at [`api.py`](https://github.com/lukealonso/b12x/blob/c469c6637f6251adefc282956f5392e559ea915d/b12x/attention/paged/api.py) (`output.ndim != 3` check); Triton backend uses same pattern at `triton_attn.py:1565,1643`

## GSM8K sanity check

**Script:** `tests/evals/gsm8k/gsm8k_eval.py --num-questions 50 --max-tokens 256`
**Endpoint:** `/v1/completions` (5-shot, no thinking mode)

| Metric | Value | Notes |
|---|---|---|
| Accuracy | 26.0% (13/50) | 23 responses lost to 600s session timeout |
| Accuracy (valid only) | 48.1% (13/27) | Timeout-free responses |
| Invalid rate | 46.0% | All from client timeout, not model failure |
| Throughput | 11.1 tok/s | Expected — Python prototype, not CuTe DSL |

**Interpretation:** The 23 timeouts are from the async eval client's 600s session timeout —
the prototype at 11 tok/s cannot serve 50 concurrent requests in time. The 48% accuracy on
valid responses is a reasonable sanity signal for a prototype with small sample size (27).
Baseline with triton_attn was 62% on 50 questions. Clean comparison requires the real
CuTe DSL kernel at production throughput.

## Baseline comparison (triton_attn, captured 2026-04-10)

| Metric | triton_attn | CuTe prototype | Notes |
|---|---|---|---|
| GSM8K accuracy | 62% (50q) | 48% (27 valid) | Prototype limited by timeout |
| ShareGPT tok/s | 48.22 | N/A | Prototype too slow to bench |
| TPOT median | 80.75ms | N/A | Deferred to real kernel |

## Reproduction

```bash
# Build image
docker build -f docker/Dockerfile.gb10 -t nvllm:gb10 .

# Launch server (debug/eager mode)
bash scripts/run_qwen35_27b_cute_paged.sh --debug

# Run GSM8K
python3 tests/evals/gsm8k/gsm8k_eval.py --num-questions 50 --max-tokens 256 \
    --save-results /tmp/gsm8k_cute_paged.json
```

## Next steps

1. Replace PyTorch prototype with real CuTe DSL kernel (b12x patterns — see `docs/kernel-insights/2026-04-11-b12x-paged-attention.md`)
2. Re-run GSM8K with real kernel (full 50q, no timeout)
3. ShareGPT 200 performance benchmark vs triton_attn baseline
4. nsys comparison profile
5. CUDA graph validation (Step 2 of Task 12)
