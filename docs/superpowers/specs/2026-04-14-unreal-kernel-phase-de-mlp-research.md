# Unreal Kernel Phase D+E: MLP Fusion — Research Notes

**Date:** 2026-04-14
**Status:** Research / Pre-Spec
**Target:** SM120/SM121 (DGX Spark GB10)
**Model:** Qwen3.5-27B NVFP4

## What Phase D+E Would Fuse

After Phase C (RMSNorm), the MLP is:
```
gate = gate_proj(x)       →  [1, 5120] × [5120, 17408]  →  [1, 17408]
up   = up_proj(x)         →  [1, 5120] × [5120, 17408]  →  [1, 17408]
act  = silu(gate) * up    →  [1, 17408]  (element-wise)
out  = down_proj(act)     →  [1, 17408] × [17408, 5120]  →  [1, 5120]
residual += out           →  [1, 5120]   (residual add)
```

Qwen3.5-27B uses **SwiGLU** (SiLU-gated linear unit).

## The Core Challenge: intermediate_size = 17408

The intermediate vector between gate/up and down is 17408 FP32 values = **68 KB**.

This doesn't fit in:
- **Registers:** 128 threads × 256 regs × 4B = 128 KB total register file. But
  17408/128 = 136 values per thread = 136 registers JUST for the intermediate.
  That's 53% of the register file for one buffer. Not feasible alongside other
  state.
- **SMEM:** 101 KB available, 45 KB used by Phase A. 56 KB headroom < 68 KB
  intermediate. Doesn't fit.

This is fundamentally different from Phase B (W_O), where the output vector was
only 5120 elements (40 per thread, 40 registers).

## Tiling Strategies to Investigate

### Strategy A: Tiled Streaming (Gate+Up→SiLU→Down in Tiles)

Process the 17408 intermediate in tiles of T elements. For each tile:
1. Compute T elements of gate_proj (partial GEMV)
2. Compute T elements of up_proj (partial GEMV)
3. SiLU(gate_tile) * up_tile → T intermediate values
4. Multiply by corresponding T rows of down_proj, accumulate into output[5120]

The intermediate never fully materializes — each tile's T values live in registers
temporarily, get consumed by down_proj, then the registers are recycled.

**The key insight:** down_proj's output dimension is 5120. Each thread owns 40
output elements (same as Phase B). For each tile of T intermediate values, each
thread reads T × 40 down_proj weights and accumulates 40 dot products.

**Tile size T:** Must be small enough for registers (T values of gate + T of up +
40 accumulators + scratch). With T=16: 16+16+40+~10 = ~82 registers. Feasible.

**Inner loop structure:**
```
// 40 FP32 accumulators for down_proj output (same as Phase B pattern)
acc[0..39] = 0.0

for tile in range(0, 17408, T):           // 17408/16 = 1088 outer iterations
    // Compute T elements of gate and up
    for t in range(T):                     // constexpr unroll
        gate[t] = dot(x[0..5119], gate_weight[tile+t, 0..5119])
        up[t]   = dot(x[0..5119], up_weight[tile+t, 0..5119])
    
    // SiLU + elementwise multiply
    for t in range(T):
        intermediate[t] = silu(gate[t]) * up[t]
    
    // Accumulate into down_proj output
    for t in range(T):
        for o in range(40):               // constexpr unroll
            acc[o] += down_weight[my_row+o, tile+t] * intermediate[t]
```

**Problem:** The inner GEMV for gate/up is `[1, 5120] × [5120, T]`. This means
reading 5120 input elements × T times. The input `x` is only 5120 × 2B = 10 KB —
easily SMEM-cached. But each tile reads T rows of gate_weight and up_weight, each
5120 × 0.5B = 2.5 KB (NVFP4). Total weight reads per tile: 2 × T × 2.5 KB.

Total gate+up weight reads: 2 × 5120 × 17408 × 0.5B = 89 MB. Same as the
unfused path — weight bandwidth doesn't change.

**But down_proj is now streamed:** Instead of reading all 17408 × 5120 × 0.5B =
44.5 MB of down_proj in one GEMV, each tile reads T × 40 × 0.5B per thread.
Total down_proj reads: 17408 × 5120 × 0.5B = 44.5 MB. Same total bandwidth.

**Benefit of tiling:** The 68 KB intermediate vector never touches global memory.
It's generated in registers (T values at a time) and immediately consumed. The
unfused path writes 17408 × 4B = 68 KB to global, then reads it back.

**Cost of tiling:** The input vector `x[5120]` is read once per tile iteration
(1088 times for T=16). With SMEM caching, this is fine — but it's 1088 SMEM
broadcast reads vs 1 global read in the unfused path.

### Strategy B: Phase B Pattern (atomicAdd Output, No Down Tiling)

Keep gate+up+silu as the "Phase D" GEMV (separate from down_proj), write the
17408 intermediate to global memory, then Phase E reads it back for down_proj.

This is simpler but loses the key fusion benefit (eliminating the 68 KB
intermediate write/read). It still saves kernel launch overhead.

### Strategy C: Warp-Cooperative Tiling

Instead of each thread independently processing 40 output rows, warps cooperate:
- 4 warps handle different output row ranges
- Within each warp, 32 lanes tile across the intermediate dimension
- MMA-style data sharing via shuffle for dot products

This could use the Phase A MMA infrastructure, but M=1 decode GEMVs don't
benefit from MMA tiles (too few rows).

### Strategy D: Two-Kernel Split

Phase C produces RMSNorm output → return from kernel.
Separate "Phase D+E kernel" does gate+up+silu+down as a standalone fused MLP.

This is less ambitious but still valuable — the fused MLP kernel eliminates the
intermediate write/read without needing to fit inside the attention kernel.

## Weight Bandwidth (The Dominant Cost)

For decode (batch_size=1), the MLP is entirely **memory-bandwidth bound**:

| Operation | Weight Size (NVFP4) | FLOPs |
|-----------|--------------------|-------|
| gate_proj | 5120 × 17408 × 0.5B = 44.5 MB | 178M |
| up_proj | 5120 × 17408 × 0.5B = 44.5 MB | 178M |
| down_proj | 17408 × 5120 × 0.5B = 44.5 MB | 178M |
| **Total** | **133.5 MB** | **534M** |

GB10 memory bandwidth: ~273 GB/s.
Minimum time to stream weights: 133.5 MB / 273 GB/s ≈ **489 μs**.
Compute time at ~200 GFLOPS (scalar FP32): 534M / 200G ≈ 2.7 μs.

**Arithmetic intensity: 534M / 133.5M = 4 FLOPs/byte.** Deep in the memory-bound
regime. Fusion won't change the weight bandwidth — the savings are:

1. Eliminating 68 KB intermediate write + read = 136 KB saved
2. Eliminating 2-3 kernel launches
3. Better L2 utilization (weights from consecutive operations may share L2 lines)

The 136 KB savings is ~0.1% of the 133.5 MB weight traffic. **Kernel launch
elimination is the main benefit for decode.**

## SiLU Implementation

SiLU(x) = x × σ(x) = x × (1 / (1 + exp(-x)))

In terms of proven CuTe DSL primitives:
```python
# exp(-x) = exp2(-x * log2(e)) = exp2(-x * 1.4426950)
neg_x_log2e = x * Float32(-1.4426950)
exp_neg_x = exp2_approx_ftz_f32(neg_x_log2e)

# sigmoid = 1 / (1 + exp(-x))
# Need rcp (reciprocal) — new PTX helper: rcp.approx.ftz.f32
sigmoid = _rcp_approx_f32(Float32(1.0) + exp_neg_x)

# SiLU = x * sigmoid(x)
silu = x * sigmoid
```

New helper needed: `_rcp_approx_f32` — one PTX instruction via `@dsl_user_op`.

## Research Questions

### Must-answer before spec

1. **Does TensorRT-LLM fuse gate+up+down into one kernel for decode?**
   If so, how do they tile the intermediate? Check their FusedMoE or similar.
   Look at: `tensorrt_llm/kernels/fused_gated_gemm/` if it exists.

2. **Can Strategy A (tiled streaming) reuse the input `x` efficiently?**
   The RMSNorm output `x[5120]` is read 1088 times (once per tile). Must be in
   SMEM. Phase A's K/V SMEM (32 KB) is free after Phase A finishes — more than
   enough for x (10 KB in BF16).

3. **What tile size T gives best register balance?**
   T=8: 8+8+40+10=66 regs, 2176 outer iterations, more loop overhead
   T=16: 16+16+40+10=82 regs, 1088 outer iterations
   T=32: 32+32+40+10=114 regs, 544 outer iterations, still fits in 256

4. **Can gate and up tiles be computed simultaneously?**
   Both read the same input `x`. If tiled with T=16, each tile needs
   2 × 16 × 5120 × 0.5B = 80 KB of gate+up weights per tile. That's almost all
   the SMEM. Probably need to stream from global with L2 caching.

5. **Is the tiled approach actually faster than separate GEMMs?**
   CUTLASS's tuned GEMV kernels may have better L2 behavior for weight streaming
   than our hand-written CuTe DSL loop. The 68 KB intermediate elimination might
   not compensate. **Needs nsys comparison.**

### Nice-to-answer

6. **Can we overlap gate and up computation?** They're independent — could
   interleave tile iterations to hide latency.

7. **Is there a benefit to fusing across layers?** Phase C of layer N+1's
   input_layernorm could immediately follow Phase E's residual add, keeping the
   5120-element vector in L2.

8. **b12x reference:** Does lukealonso/b12x@c469c66 have any MLP fusion or
   SwiGLU kernels?

## Resources to Read

- [ ] TensorRT-LLM fused MLP kernels (if publicly available)
- [ ] CUTLASS 3.x persistent GEMM / StreamK for skinny M=1 GEMVs
- [ ] b12x@c469c66 — search for gate/up/down/MLP/SwiGLU patterns
- [ ] FlashInfer — any fused decode MLP kernels?
- [ ] "Reducing Activation Recomputation in Large Transformer Models" (Korthikanti et al.)
- [ ] vLLM's existing FusedMoE kernel — how does it handle gate/up interleaving?

## Preliminary Recommendation

**Strategy A (tiled streaming)** is the most promising for full fusion — it
eliminates the intermediate materialization while keeping the weight bandwidth
identical. The key enabling insight is that Phase A's SMEM is free after
attention completes, giving us 32+ KB to cache the RMSNorm output `x`.

**Strategy D (two-kernel split)** is the pragmatic fallback if Strategy A proves
too complex or doesn't outperform separate GEMMs.

**Phase C should ship independently** regardless of the D+E decision — it's
clearly feasible, well-scoped, and valuable even if D+E never happens.
