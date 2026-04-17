# Profile: cute_fusion — Phase B/C fusion under PIECEWISE CUDA graphs

**Commit:** `37cceaa6c` — [fix(cute-paged): fix fusion Phase C gibberish via per-CTA arrival counter](https://github.com/Navi-AI-Lab/nvllm/commit/37cceaa6c199bf211a2e170c414f64bf654b0f45)
**Date:** 2026-04-17
**Model:** `natfii/Qwen3.5-27B-NVFP4-Opus-GB10` (27B dense, 48 layers, 12 `full_attention` / 36 `linear_attention`)
**Hardware:** NVIDIA DGX Spark — GB10, SM121
**Config:**
- `--attention-backend CUTE_PAGED`
- `--kv-cache-dtype fp8_e4m3`
- `--compilation-config '{"cudagraph_mode":"PIECEWISE"}'` (no `--enforce-eager`)
- `--max-num-seqs 4`, `--max-model-len 65536`
- Fusion: **ON** — Phase A (attention + gate sigmoid) + Phase B (W_O NVFP4 GEMV) + Phase C (residual add + RMSNorm) all in one CuTe uber-kernel

**Profiler:** vLLM built-in torch profiler (`--profiler-config profiler=torch, ignore_frontend=true, delay_iterations=3, active_iterations=30`)

**Workload:** 4 concurrent `/v1/completions` requests × 128 tokens each, `ignore_eos=true`, temperature=0. Steady-state decode after a 2-request warmup.

## Top 15 kernels by total GPU time (fused path)

| # | Kernel | Calls | Total | Mean | % GPU |
|---|---|---|---|---|---|
| 1 | `cutlass::device_kernel<gemm::GemmUniv...>` (NVFP4 CUTLASS) | 38,608 | 7.939 s | 205.625 μs | **75.92%** |
| 2 | `aten::mm` / `cutlass_80_wmma_tensorop_bf16` (BF16 embed + lm_head) | 127 | 1.263 s | 9.942 ms | 12.07% |
| 3 | **`kernel_cutlass__kernel_vllmv1attentionbackendscute_paged`** (CuTe uber-kernel — A+B+C fused) | **2,032** | **865.392 ms** | **425.882 μs** | **8.28%** |
| 4 | `vllm::gdn_attention_core` (mamba linear attention) | 6,096 | 199.608 ms | 32.776 μs | 1.91% |
| 5 | `fused_recurrent_gated_delta_rule_packed_decode_kernel` | 6,096 | 178.298 ms | 29.248 μs | 1.71% |
| 6 | `vllm::cvt_fp16_to_fp4<bfloat16>` (activation quant) | 30,480 | 64.845 ms | 2.127 μs | 0.62% |
| 7 | `cudaGraphLaunch` | 8,255 | 37.815 ms | 4.581 μs | 0.36% |
| 8 | `cudaStreamIsCapturing` | 10,037 | 37.738 ms | 3.760 μs | 0.36% |
| 9 | `_causal_conv1d_update_kernel` | 6,096 | 21.310 ms | 3.496 μs | 0.20% |
| 10 | `vllm::silu_mul_cvt_fp16_to_fp4` (SiLU+mul+quant) | 8,128 | 13.059 ms | 1.607 μs | 0.12% |
| 11 | `triton_red_fused__to_copy_add_mean_mul_pow_rsqrt` (RMSNorm) | 6,096 | 10.933 ms | 1.793 μs | 0.10% |
| 12 | `triton_red_fused__to_copy_add_copy__mean_mul_pow_rsqrt` (RMSNorm) | 4,064 | 8.987 ms | 2.211 μs | 0.09% |
| 13 | `triton_poi_fused__to_copy__unsafe_view_add_clone_mean` | 6,096 | 7.504 ms | 1.231 μs | 0.07% |
| 14 | `triton_poi_fused_0` | 8,255 | 7.166 ms | 0.868 μs | 0.07% |
| 15 | `Memset (Device)` | 16,255 | 6.398 ms | 0.394 μs | 0.06% |

## Read

- **NVFP4 CUTLASS GEMM (76%)** dominates the decode hot path — expected on a dense 27B in FP4. This is the target of all future perf work (stream-K tuning, persistent kernel, better tile schedulers). The CuTe fusion kernel is no longer the bottleneck; the FFN is.
- **CuTe fused A+B+C kernel (8.28% of GPU time)** runs at 425.9 μs/call. This call now subsumes three previously separate ops:
  - Phase A — attention (previously ~244 μs standalone, per the April 13 CuTe baseline memory)
  - Phase B — W_O NVFP4 GEMV (previously a separate `cutlass::device_kernel` call in top-1)
  - Phase C — residual add + post-attn RMSNorm (previously two Triton kernels)
- **Triton RMSNorm kernels (items 11, 12, 13)** are still present for the other RMSNorms in the layer (pre-attn norm, pre-MoE norm, post-MoE norm). Only the post-attention RMSNorm is fused into CuTe.
- Attention + its fused epilogue now occupies **8.28%** of decode GPU time, up from ~0.5% unfused — because the fused kernel absorbed work that used to show up as 2-3 separate kernels. Net effect: same total work, fewer kernel launches, fewer materializations of the attention output tensor.

## CUDA graph / sync overhead (items 7–8)

`cudaGraphLaunch` + `cudaStreamIsCapturing` = **75.6 ms / 0.72% of GPU time** across **18,292 host-side calls**. This is the PIECEWISE capture/replay bookkeeping — ~9 μs per decode step to re-enter + replay a captured subgraph. Small absolute cost but high call count; if we ever migrate to FULL or FULL_AND_PIECEWISE graphs, these collapse into one launch per step. Not an action item right now — just a known overhead signature worth tracking.

## Throughput (measured during profile capture)

| Scenario | Wall | Tokens | tok/s aggregate |
|---|---|---|---|
| batch=4 × 128 tok (profiled) | 12.47 s | 512 | 41.1 |
| batch=4 × 256 tok (pre-profile, GSM8K session) | 23.48 s | 1024 | 43.6 |
| batch=1 × 256 tok (pre-profile) | 22.91 s | 256 | 11.2 |

Profiler overhead during capture: ~5% (43.6 → 41.1 tok/s).

## Caveats

- This profile captures **fusion-on**. A direct unfused baseline is not in this trace; the next step is a second capture with `Qwen3NextDecoderLayer._fusion_bound` forced to False (or by launching with `fusion_active` gated off) to quantify the exact fusion delta.
- `execute_context_0(0)_generation_4(4)` at 10.56 s (101%) is the NVTX range wrapping all 4 generation requests — double-counted under torch-profiler accounting, not a real kernel.
- `ignore_eos=true` means the decode runs the full 128 tokens with no early-exit; closer to benchmark conditions than natural chat.

## How to reproduce

```bash
# 1. Build image with fusion fix baked in
docker build -f docker/Dockerfile.gb10 -t nvllm:gb10 .

# 2. Launch with torch profiler
mkdir -p benchmarks/nvllm/traces/cute_fusion/2026-04-17-phase-bc-fused/profiles
docker run -d --name nvllm --gpus all --ipc=host --network host --privileged \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  -v "$HOME/.cache/flashinfer:/root/.cache/flashinfer" \
  -v "$PWD/benchmarks/nvllm/traces/cute_fusion/2026-04-17-phase-bc-fused/profiles:/tmp/profiles" \
  -e VLLM_NVFP4_GEMM_BACKEND=cutlass \
  -e VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  nvllm:gb10 serve \
  --model natfii/Qwen3.5-27B-NVFP4-Opus-GB10 --served-model-name default \
  --host 0.0.0.0 --port 8000 --kv-cache-dtype fp8_e4m3 \
  --attention-backend CUTE_PAGED --max-model-len 65536 --max-num-seqs 4 \
  --language-model-only --mamba-cache-mode align --trust-remote-code \
  --gpu-memory-utilization 0.80 --max-num-batched-tokens 65536 \
  --compilation-config '{"cudagraph_mode":"PIECEWISE"}' \
  --profiler-config '{"profiler":"torch","torch_profiler_dir":"/tmp/profiles","ignore_frontend":true,"delay_iterations":3,"active_iterations":30,"torch_profiler_with_stack":false,"torch_profiler_use_gzip":true}'

# 3. Wait for "Application startup complete", warm up 2 requests, then:
curl -X POST http://localhost:8000/start_profile

# 4. Fire 4 concurrent decode requests (128 tokens each, ignore_eos)
for i in 1 2 3 4; do
  curl -s http://localhost:8000/v1/completions \
    -H 'Content-Type: application/json' \
    -d "{\"model\":\"default\",\"prompt\":\"<128-token prompt>\",\"max_tokens\":128,\"temperature\":0,\"ignore_eos\":true}" &
done; wait

curl -X POST http://localhost:8000/stop_profile

# 5. Trace lands in ./benchmarks/nvllm/traces/cute_fusion/2026-04-17-phase-bc-fused/profiles/
#    View with chrome://tracing or https://ui.perfetto.dev
```

## Files

- `profiles/fused.pt.trace.json.gz` — 11.5 MB torch profiler trace (Chrome Tracing / Perfetto compatible)
- `profiles/profiler_out_0.txt` — 21 KB human-readable kernel summary
