# nsys Profile: Qwen3.5-27B-NVFP4 on GB10

**Date:** 2026-04-07
**Model:** natfii/Qwen3.5-27B-NVFP4-Opus-GB10
**Config:** TQ KV cache, 64k context, max-num-seqs=4, PIECEWISE cudagraphs
**Load:** 40 requests × 256 max_tokens, 4 concurrent
**File:** profile-20260407-053350.nsys-rep

## Top Kernel Hotspots

| Rank | Time% | Total (s) | Calls | Avg (ms) | Kernel | Category |
|------|-------|-----------|-------|----------|--------|----------|
| 1 | 39.2% | 114.4 | 497,721 | 0.23 | cutlass::GemmUniversal (FP4) | **FP4 decode GEMM** |
| 2 | 18.2% | 53.1 | 608 | 87.4 | cutlass::GemmUniversal (FP4) | **FP4 prefill GEMM** |
| 3 | 8.0% | 23.2 | 97,234 | 0.24 | FillFunctor<int> | Memory fill/zeroing |
| 4 | 6.1% | 17.9 | 1,629 | 10.9 | cutlass::Kernel2 wmma bf16 | **BF16 lm_head GEMM** |
| 5 | 4.2% | 12.3 | 52,614 | 0.23 | _turboquant_quantize_store | TQ KV cache quantize |
| 6 | 3.5% | 10.3 | 27,111 | 0.38 | triton_red_fused_7 | Triton attention reduce |
| 7 | 2.4% | 7.1 | 26,601 | 0.27 | triton_red_fused_...fp4_quant | RMSNorm+FP4 quant fusion |
| 8 | 1.7% | 5.1 | 79,175 | 0.06 | triton_red_fused_...fp4_quant | RMSNorm+FP4 quant fusion (variant) |
| 9 | 1.3% | 3.8 | 25,971 | 0.15 | _turboquant_decode_kernel | TQ KV cache decode |
| 10 | 0.8% | 2.2 | 77,580 | 0.03 | fused_recurrent_gated_delta_rule | FLA linear attention decode |

## Analysis

### FP4 GEMM (57.4% total)
- Decode (#1, 39.2%): 230μs avg, 498K calls — bandwidth-bound at small M
- Prefill (#2, 18.2%): 87ms avg, 608 calls — compute-bound at large M
- This is the CUTLASS FP4 kernel from `nvfp4_scaled_mm_sm120_kernels.cu`

### lm_head BF16 GEMM (6.1%)
- 1,629 calls × 10.9ms = 17.9s total
- Uses wmma_tensorop_bf16 (BF16 GEMM, not FP4)
- FP4 lm_head would reduce this to ~3-4ms (est. ~60-65% reduction)
- Expected overall improvement: ~4-5% of total GPU time

### Memory Operations (8.0%)
- FillFunctor at 8% is suspiciously high — buffer zeroing/init
- Worth investigating if this is necessary or can be eliminated

### TurboQuant KV Cache (5.5%)
- Quantize: 4.2% (52K calls, 234μs avg)
- Decode: 1.3% (26K calls, 146μs avg)
- Postprocess: 0.2% (52K calls, 9μs avg)

### RMSNorm+FP4 Quant Fusions (~5.6%)
- Multiple Triton kernels handling RMSNorm→quant pipeline
- Already fused (not separate RMSNorm + separate quant)

### FLA Linear Attention (0.8%)
- fused_recurrent_gated_delta_rule: 77K calls, 28μs avg
- Very efficient — not a bottleneck

## Optimization Priority

1. **FP4 decode GEMM tile tuning** (39.2%) — SM121 M16 decode config
2. **lm_head FP4** (6.1%) — switch from BF16 to FP4
3. **Memory fill investigation** (8.0%) — why is buffer zeroing so expensive?
4. **TQ KV cache** (5.5%) — already optimized, low priority

## nsys Setup Notes

- Container requires `--privileged` for CUPTI injection (CUDA tracing)
- nsys must be volume-mounted from host: `/opt/nvidia/nsight-systems/2025.6.3`
- `--cuda-graph-trace=node` required to see inside PIECEWISE cudagraph launches
- Model takes ~290s to load; nsys needs duration > startup time
