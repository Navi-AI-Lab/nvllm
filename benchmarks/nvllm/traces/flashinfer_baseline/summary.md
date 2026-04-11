# Baseline Profile: FlashInfer/Triton Attention on SM121

Date: 2026-04-10
Commit: 012090b82
Model: natfii/Qwen3.5-27B-NVFP4
Config: --kv-cache-dtype fp8_e4m3 --enforce-eager --max-model-len 4096/8192 --attention-backend triton_attn

## Serving Performance (from workload runs)

| Metric | Value |
|---|---|
| Decode throughput | ~12 tok/s (batch=1) |
| Inter-token latency (p50) | ~80ms |
| Tokens per request | 256-512 |
| Time per 512-token request | ~42s (max-model-len=8192) |
| Time per 256-token request | ~21s (max-model-len=4096) |

## Kernel-Level Profiling Status

**nsys kernel-level data was NOT captured** due to vLLM v1's multiprocessing
architecture: the EngineCore (which runs all GPU work) is a forked child process,
and nsys CUPTI injection does not follow across Python multiprocessing fork
boundaries, even with `--privileged` and `--trace-fork-before-exec=true`.

CUDA API-level data was captured (2.5M cudaLaunchKernel calls) but without the
GPU kernel execution table, we cannot determine attention % of total GPU time.

### Profiling approaches attempted:
1. nsys system-wide from host: captured API calls, no kernel data
2. nsys with --trace-fork-before-exec=true: same result
3. nsys attached to EngineCore PID (-p): requires sudo
4. torch.profiler in parent process: only captures parent CPU, not child GPU
5. VLLM_TORCH_PROFILER_DIR + start_profile API: endpoint not available in build
6. vllm bench latency --profile: no output captured (entrypoint redirect issue)

### What would work (for future reference):
- Install nsys inside the container image (add to Dockerfile)
- Use nsys as the EngineCore entrypoint (requires vLLM source modification)
- Use CUDA Profiler API (cudaProfilerStart/Stop) instrumented in EngineCore
- Use the cuTile TileGym profiler (announced 2026-04-10, may handle this)

## Kernel Breakdown (from April 7 profile, TQ KV config)

Data from `benchmarks/nvllm/results/profile-20260407-053350.nsys-rep` (249 MB, full
kernel data captured with PIECEWISE cudagraphs + cuda-graph-trace=node).

| Rank | Time% | Category |
|------|-------|----------|
| 1 | 39.2% | FP4 decode GEMM (cutlass) |
| 2 | 18.2% | FP4 prefill GEMM (cutlass) |
| 3 | 8.0% | Memory fill (FillFunctor) |
| 4 | 6.1% | lm_head BF16 GEMM |
| 5 | 4.2% | TurboQuant KV quantize |
| 6 | **3.5%** | **Triton attention reduce (triton_red_fused_7)** |
| 7 | ~5.6% | RMSNorm+FP4 quant fusions |
| 8 | 1.3% | TurboQuant KV decode |
| 9 | **0.8%** | **FLA linear attention (GDN decode)** |

**Attention total: ~4.3%** (3.5% full attention + 0.8% linear attention).

With FP8 KV instead of TQ (no TQ overhead), attention's relative share rises to
~5-6% of total GPU time.

Note: The Triton attention kernel names are mangled (triton_red_fused_7). The QK/PV
matmul may be fused into different numbered Triton kernels. The 3.5% is a lower
bound for full attention.

## Go/No-Go Decision

**PROCEED** — based on:

1. **Control justification** — FlashInfer's SM120 path uses sm_mma=80 (SM80 fallback),
   with known xfail tests for batch attention due to oversized tiles. Custom kernel
   gives direct control over tile configs, MMA atoms, and pipeline depth.

2. **Known suboptimality** — FlashInfer's batch attention tests explicitly fail on
   SM120 because "tile size/number of stages is too large" for 101KB SMEM. The
   Triton attention backend is the current workaround but not SM121-optimized either.

3. **FP8 MMA opportunity** — SM120's native FP8 m16n8k32 MMA offers 2x throughput
   vs BF16 m16n8k16 for the QK pass. No existing backend exploits this.

4. **Stack unification** — Custom CuTe GEMM kernel already works on SM121. Adding
   custom attention completes the owned kernel stack (GEMM + attention).

Kernel-level profiling will be performed once the custom backend exists, providing
a direct A/B comparison with exact kernel duration measurements.

## How to reproduce

```bash
# Server launch (from repo root)
docker run -d --privileged --gpus all --ipc=host --network host \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  -e VLLM_NVFP4_GEMM_BACKEND=cutlass \
  nvllm:gb10 serve \
  --model /root/.cache/huggingface/hub/Qwen3.5-27B-NVFP4 \
  --kv-cache-dtype fp8_e4m3 --enforce-eager --max-model-len 8192 \
  --attention-backend triton_attn --max-num-seqs 2 \
  --mamba-cache-mode align --mamba-block-size 64 \
  --trust-remote-code --gpu-memory-utilization 0.55

# Workload (10 prompts, 512 max tokens each)
python3 /tmp/profile_workload.py
```
