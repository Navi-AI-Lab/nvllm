# CuTe Paged Attention vs FlashInfer — Kernel Profile Comparison

**Date:** 2026-04-13
**Commit:** `26bda34d6` (CuTe DSL decode kernel)
**Model:** natfii/Qwen3.5-27B-NVFP4-Opus-GB10 (GQA=6: 24Q/4KV, head_dim=256)
**Hardware:** NVIDIA DGX Spark (GB10, SM121)

## Configuration

| Setting | Value |
|---------|-------|
| kv_cache_dtype | fp8_e4m3 |
| max_model_len | 65536 |
| max_num_seqs | 4 |
| gpu_memory_utilization | 0.70 |
| cudagraph_mode | PIECEWISE |
| Workload | 10 requests, 128 max_tokens, 4 concurrent |
| Prompt | "Explain the implications of quantum entanglement for secure communication systems." |

## Profiling Method

vLLM's built-in torch profiler via `--profiler-config`, activated with `/start_profile` and `/stop_profile` HTTP endpoints. This captures kernel-level data from inside the EngineCore process, bypassing the nsys CUPTI limitation with vLLM V1's multiprocessing architecture (nsys cannot trace spawned child processes).

Config: `{"profiler":"torch","torch_profiler_dir":"/tmp/profiles","ignore_frontend":true,"delay_iterations":5,"max_iterations":10}`

## Attention Kernel Comparison

| Metric | CuTe Paged | FlashInfer | Ratio |
|--------|-----------|------------|-------|
| **Kernel** | `kernel_cutlass__cute_paged` | `BatchPrefillWithPagedKVCache` | — |
| **Calls** | 160 | 160 | 1.0x |
| **Total CUDA time** | 39.092ms | 1.160ms | 33.7x |
| **Avg per call** | 244.323us | 7.250us | 33.7x |
| **% of GPU time** | 0.59% | 0.15% | — |

## Full Kernel Breakdown

### CuTe Paged Attention Run

Total CUDA time: 6.580s

| # | Kernel | Calls | Total | Avg | % GPU | Category |
|---|--------|-------|-------|-----|-------|----------|
| 1 | `elementwise_kernel` (aten::copy_) | 640 | 5.773s | 9.021ms | 87.74% | **KV cache .contiguous() copy** |
| 2 | `cutlass::GemmUniversal` (FP4) | 3040 | 626.141ms | 205.967us | 9.52% | FP4 decode GEMM |
| 3 | `cutlass::Kernel2` (wmma bf16) | 10 | 99.203ms | 9.920ms | 1.51% | lm_head BF16 GEMM |
| 4 | `kernel_cutlass__cute_paged` | 160 | 39.092ms | 244.323us | 0.59% | **CuTe attention decode** |
| 5 | `fused_recurrent_gated_delta_rule` | 480 | 14.074ms | 29.320us | 0.21% | FLA linear attention (GDN) |
| 6 | `cvt_fp16_to_fp4` | 2400 | 5.108ms | 2.129us | 0.08% | FP4 weight quant |

### FlashInfer Baseline Run

Total CUDA time: 762.803ms

| # | Kernel | Calls | Total | Avg | % GPU | Category |
|---|--------|-------|-------|-----|-------|----------|
| 1 | `cutlass::GemmUniversal` (FP4) | 3040 | 621.005ms | 204.278us | 81.41% | FP4 decode GEMM |
| 2 | `cutlass::Kernel2` (wmma bf16) | 10 | 99.492ms | 9.949ms | 13.04% | lm_head BF16 GEMM |
| 3 | `fused_recurrent_gated_delta_rule` | 480 | 13.934ms | 29.029us | 1.83% | FLA linear attention (GDN) |
| 4 | `cvt_fp16_to_fp4` | 2400 | 5.065ms | 2.111us | 0.66% | FP4 weight quant |
| 5 | `BatchPrefillWithPagedKVCache` | 160 | 1.160ms | 7.250us | 0.15% | **FlashInfer attention** |

## Analysis

### The .contiguous() bottleneck

The CuTe run's total CUDA time is **8.6x higher** than FlashInfer (6.58s vs 763ms), but this is almost entirely due to a **5.77s `aten::copy_` overhead** (87.7% of GPU time). This comes from two `.contiguous()` calls in `_backend.py:229-230`:

```python
k_cache = kv_cache[:, 0].contiguous()  # copies entire K cache slice
v_cache = kv_cache[:, 1].contiguous()  # copies entire V cache slice
```

This is a workaround for CuTe DSL's inability to read uint8 tensors via tensor indexing (returns all zeros). The kernel uses raw `ld.global` via `data_ptr()` which requires contiguous memory. **Eliminating this copy is the highest-priority optimization** — it would reduce CuTe's total CUDA time from 6.58s to ~807ms, within 6% of FlashInfer.

### Kernel compute comparison (excluding copy overhead)

Removing the copy overhead, the effective kernel-only comparison:

| Component | CuTe | FlashInfer | Delta |
|-----------|------|------------|-------|
| FP4 GEMM | 626ms | 621ms | +0.8% (noise) |
| lm_head | 99ms | 99ms | same |
| **Attention** | **39ms** | **1.2ms** | **+3170%** |
| GDN linear | 14ms | 14ms | same |
| Other | 29ms | 28ms | same |
| **Total (excl. copy)** | **807ms** | **763ms** | **+5.8%** |

The CuTe attention kernel is 33.7x slower per-call than FlashInfer, but attention is only 0.15-0.59% of total GPU time. The FP4 GEMM at 81% dominates. Even with the slower attention, the kernel-only overhead is just 5.8%.

### Optimization priority

1. **Eliminate .contiguous() copies** (87.7% of overhead) — restructure KV cache layout or use strided addressing in the kernel
2. **Attention kernel optimization** (244us → target <50us) — tile size tuning, pipeline depth, SMEM staging
3. Both are dwarfed by FP4 GEMM (81%) — attention optimization has diminishing returns beyond ~50us

## Trace Files

| File | Description |
|------|-------------|
| [`cute_paged_torch_profile.txt`](cute_paged_torch_profile.txt) | CuTe paged attention kernel summary |
| [`flashinfer_torch_profile.txt`](flashinfer_torch_profile.txt) | FlashInfer baseline kernel summary |
| [`rank0.1776115858498148028.pt.trace.json.gz`](rank0.1776115858498148028.pt.trace.json.gz) | CuTe torch trace (view in chrome://tracing or perfetto.dev) |
| [`rank0.1776116491109655086.pt.trace.json.gz`](rank0.1776116491109655086.pt.trace.json.gz) | FlashInfer torch trace |

## How to Reproduce

```bash
# CuTe Paged Attention
docker run -d --name nvllm --gpus all --ipc=host --network host --privileged \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  -e VLLM_NVFP4_GEMM_BACKEND=cutlass \
  -e VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
  nvllm:gb10 serve \
  --model natfii/Qwen3.5-27B-NVFP4-Opus-GB10 --served-model-name default \
  --kv-cache-dtype fp8_e4m3 --attention-backend CUTE_PAGED \
  --max-model-len 65536 --max-num-seqs 4 --language-model-only \
  --enable-prefix-caching --mamba-cache-mode align --mamba-block-size 64 \
  --trust-remote-code --gpu-memory-utilization 0.70 \
  --compilation-config '{"cudagraph_mode":"PIECEWISE"}' \
  --profiler-config '{"profiler":"torch","torch_profiler_dir":"/tmp/profiles","ignore_frontend":true,"delay_iterations":5,"max_iterations":10}'

# Wait for model to load, then:
curl -s http://localhost:8000/v1/completions -H "Content-Type: application/json" \
  -d '{"model":"default","prompt":"Hello","max_tokens":16}' > /dev/null  # warmup
curl -X POST http://localhost:8000/start_profile
# Fire 10 requests...
curl -X POST http://localhost:8000/stop_profile
docker cp nvllm:/tmp/profiles/ ./traces/

# FlashInfer: same command with --attention-backend FLASHINFER
```
