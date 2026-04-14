# Unreal Kernel Phase 2: W_O GEMV Fusion into Attention Decode Epilogue

**Date:** 2026-04-13
**Status:** Approved
**Target:** SM120/SM121 (DGX Spark GB10)
**Model:** Qwen3.5-27B NVFP4 (natfii/Qwen3.5-27B-NVFP4-Opus-GB10)

## Overview

Fuse the W_O output projection (o_proj) into the CuTe DSL paged attention decode
kernel as a second phase within the same kernel launch. Eliminates one HBM round-trip
per decode layer — the attention output stays in SMEM and is consumed on-chip by the
W_O GEMV, never written to global memory.

Inspired by id Tech 6/7 rendering pass fusion: compute the G-buffer (attention) in
one pass, resolve lighting (W_O projection) in a second pass within the same tile,
data stays on-chip between passes.

This is a **fun/creative project** — the goal is learning and stack ownership, not
primarily performance optimization.

## Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Scope | Phase 2 only (W_O fusion) | Ship and learn before Phase 3 |
| Weight format | NVFP4 | Production path, no training wheels |
| Decode/prefill | Decode only | Prefill GEMM is a different problem |
| Cross-head reduction | atomicAdd in global memory | 4 CTAs per seq, low contention |
| Accumulation | Register-file, single flush | Carmack way — maximize on-chip residency |
| Output buffer | Caller-zeroed, separate residual add | Clean boundary with existing layernorm |
| Architecture | Two-phase epilogue (Approach 2) | Clean passes, on-chip handoff — id Tech pattern |

## Kernel Architecture: Two-Phase Fused Decode

### Phase A — Attention (existing, unchanged)

The current _md loop runs exactly as-is: QK dot products, online softmax, PV
accumulation across 16-column blocks. Cross-warp reduction merges partial softmax
states. Final BF16 attention output lands in SMEM.

```
SMEM after Phase A:
  attn_output: [cta_q=16, head_dim=256] × BF16 = 8,192 bytes
```

### Phase B — W_O GEMV (new epilogue)

All 128 threads (4 warps) cooperatively compute the W_O projection.

- **Input:** Attention output in SMEM (from Phase A)
- **Weights:** NVFP4 W_O matrix streamed from global memory
- **Output:** Partial hidden state (FP32, length 3584) in registers → atomicAdd to global buffer

### Handoff

A single `__syncthreads()` barrier between Phase A and Phase B. Phase A's cross-warp
reduction already ends with a sync, so the attention output is guaranteed coherent in
SMEM before Phase B begins.

### Grid

Unchanged: `(ceil(group_size/cta_q), num_kv_heads, num_seqs)`
For Qwen 27B: `(1, 4, num_seqs)`. Each CTA handles one KV head group (6 Q heads).

## W_O GEMV Tiling and Compute

### Per-CTA math

Each CTA owns `group_size=6` attention heads. W_O decomposes per KV-head-group:

```
W_O slice: [hidden_dim=3584, group_size × head_dim=1536]
Input:     [1536, 1]  (6 head outputs concatenated)
Output:    [3584, 1]  (partial contribution to hidden state)
```

### Thread-level tiling

128 threads, each owns a contiguous slice of the output vector:
`3584 / 128 = 28 output elements per thread`.

Each thread accumulates 28 FP32 values across the full inner dimension (1536).

### Inner loop

For decode, `cta_q=16` but only 1 token is active per sequence. The attention
output is effectively a single row: `[1, head_dim=256]` per head, giving a
concatenated input vector of `[1, 1536]` across the 6 heads in the group.

Process the 1536 input elements in tiles of 16 (NVFP4 dequant granularity).
96 inner iterations. Per iteration, each thread:

1. Reads attention input value from SMEM (broadcast — same value for all 128 threads)
2. Loads 28 NVFP4 weight values from global memory (14 bytes)
3. Dequants FP4 → BF16 → FP32
4. FMA: `acc[i] += weight[i] * attn_input` for i in 0..27

### Register budget

| Component | Registers/thread |
|-----------|-----------------|
| W_O FP32 accumulators | 28 |
| Reused from Phase A scratch | ~10 |
| Dequant + FMA scratch | ~8-10 |
| **Total** | **~38** |

SM120 allows 256 registers/thread. 38 = 15% utilization. No occupancy concern.

### Memory traffic

- W_O weights per CTA: `3584 × 1536 × 0.5B = 2.75 MB` (NVFP4)
- Scale factors: `3584 × 1536 / 16 × 1B = 344 KB` (FP8 E4M3)
- Attention output from SMEM: 8,192 bytes (already on-chip)

## NVFP4 Weight Loading and Dequant

### Loading strategy

Same raw-pointer pattern as FP8 KV cache. CuTe DSL can't read uint8 natively
(known bug), so we use `_ld_global_b32` via `data_ptr` to load 4 bytes at a time
(8 FP4 weights per load).

### Dequant pipeline (per 8-weight pack)

```
_ld_global_b32 → Uint32 (8 × FP4 weights)
  → shift + mask to extract FP4 nibble pairs
  → FP4 → FP8 E4M3 (left-shift exponent, zero-pad mantissa)
  → load FP8 scale factor for this group of 16
  → cvt.rn.f16x2.e4m3x2 → extract → cvt.f32.f16 (existing chain)
  → weight_fp32 = dequanted_fp32 × scale_fp32
```

### Scale factor access

One scale per 16 weights. Per thread per inner iteration: 4 b32 weight loads,
2 scale loads. Small enough for direct global loads with L2 caching — no SMEM
staging needed.

### Risk: NVFP4 packed layout

The exact memory layout (row-major vs column-major, interleaving pattern) of
NVFP4 weights as stored by vLLM's `Fp4Quant` needs investigation during
implementation. If the CUTLASS GEMM expects a specific interleaved format, the
dequant indexing logic gets more complex. This is the main implementation unknown.

## Output Path and Integration

### Kernel output

Each CTA atomicAdds its FP32 partial W_O result to a caller-zeroed global buffer
of shape `[num_tokens, hidden_dim]`. After all 4 CTAs finish, this buffer contains
the complete o_proj output.

### Caller responsibilities

1. Zero the output buffer before kernel launch (`torch.zeros`)
2. Pass W_O weight pointer + scale pointer as additional kernel parameters
3. Feed output buffer to `post_attention_layernorm(hidden_states, residual)` — unchanged

### Integration point: Qwen2Attention.forward()

**Today:**
```python
attn_output = self.attn(q, k, v)          # attention only
output, _ = self.o_proj(attn_output)       # separate GEMM
return output
```

**After fusion:**
```python
output = self.attn(q, k, v, wo_fused=True) # attention + W_O
return output                               # o_proj skipped
```

### Backend changes (_backend.py)

The CuTe backend's `forward()` gains optional `wo_weight`, `wo_scales`, and
`wo_output` parameters. When provided, it launches the fused kernel. When not
provided (prefill, or fallback), it launches the attention-only kernel and returns
`attn_output` as today. Clean opt-in, zero disruption to the non-fused path.

### Fallback guarantee

If the fused kernel fails compilation, raises an error, or is disabled by flag,
the existing attention-only kernel + separate o_proj GEMM path is always available.

## SMEM Budget

```
Phase A (attention):
  Q:       16 × 256 × 2B (BF16)  =  8,192 B
  K:       64 × 256 × 1B (FP8)   = 16,384 B
  V:       64 × 256 × 1B (FP8)   = 16,384 B
  sync_o:  4 × 16 × 16 × 4B      =  4,096 B
  sync_md: 4 × 16 × 8B            =    512 B

Phase B (W_O GEMV) — reuses SMEM after Phase A:
  attn_output: 16 × 256 × 2B     =  8,192 B  (written by Phase A, read by Phase B)
  K/V SMEM:                         freed after Phase A

Peak SMEM: 45,568 B (Phase A) — unchanged from current kernel
Phase B uses strictly less SMEM than Phase A (only attn_output needed)
Total available: 101,376 B — 55,808 B headroom
```

## Testing and Validation

### Correctness — three levels

1. **Standalone kernel test:** Extend `test_cute_kernel_standalone.py`. Random Q, K, V,
   W_O weights. Fused vs unfused (attention → matmul). Tolerance: `max_diff < 0.05`
   (FP4 dequant introduces more noise than FP8 KV).

2. **GSM8K sanity gate:** Serve Qwen3.5-27B with fused W_O, run GSM8K. Must match
   unfused baseline within 5% accuracy.

3. **A/B output comparison:** Same prompt through fused and unfused, compare logits.
   Manual smoke check during development.

### Performance profiling

Not a primary goal, but captured after correctness:
- nsys trace (`--privileged`), fused vs unfused decode step
- Kernel launches per layer: 12 → 11
- Attention kernel duration delta from W_O phase

### Not tested

- Prefill (unchanged, not fused)
- CUDA graphs (deferred to separate project)
- Multi-GPU / TP > 1 (single GB10, TP=1)

### Success criteria

- Standalone test: max_diff < 0.05
- GSM8K: within 5% of unfused baseline
- No regression in unfused path when fusion disabled
- Kernel compiles and runs on SM120/121

## Key Files

| File | Role |
|------|------|
| `vllm/v1/attention/backends/cute_paged/kernel.py` | Kernel — add Phase B epilogue |
| `vllm/v1/attention/backends/cute_paged/_backend.py` | Backend — wire W_O params |
| `vllm/model_executor/models/qwen2.py` | Model — opt into fused path |
| `vllm/v1/attention/backends/cute_paged/warmup.py` | Warmup — fused variant |
| `tests/test_cute_kernel_standalone.py` | Standalone correctness test |

## Game Engine Parallels

| Game Engine Concept | Kernel Equivalent |
|--------------------|-------------------|
| G-buffer pass | Phase A: attention computes output to SMEM |
| Lighting resolve | Phase B: W_O GEMV consumes SMEM, produces hidden state |
| Tile-local LDS | SMEM — data stays on-chip between passes |
| Additive light accumulation | atomicAdd from 4 CTAs to output buffer |
| Uber-shader register budget | 38/256 regs — plenty of headroom for future phases |
| Forward+ tile boundaries | CTA grid — each tile handles one KV head group |
