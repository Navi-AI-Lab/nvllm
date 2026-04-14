# Unreal Kernel Phase C: Residual Add + RMSNorm Fusion

**Date:** 2026-04-14
**Status:** Implemented — GSM8K gate PASSED (8/8, 2026-04-14)
**Target:** SM120/SM121 (DGX Spark GB10)
**Model:** Qwen3.5-27B NVFP4 (natfii/Qwen3.5-27B-NVFP4-Opus-GB10)
**Depends on:** Phase B (W_O GEMV fusion) — commit 80564378c

## Overview

Fuse the `post_attention_layernorm` (residual add + RMSNorm) into the CuTe DSL
attention kernel as Phase C, immediately following the Phase B W_O GEMV epilogue.
Eliminates one kernel launch and one global memory round-trip per decode layer.

Today's data flow (with Phase B):
```
Phase A (attention) → Phase B (W_O GEMV, atomicAdd to global FP32)
    → [kernel returns] → fused_add_rms_norm kernel → MLP
```

After Phase C:
```
Phase A → Phase B (atomicAdd to global FP32)
    → [cross-CTA sync] → Phase C (residual add + RMSNorm)
    → [kernel returns] → MLP
```

Game engine parallel: **deferred shading resolve** — after the tile-local lighting
accumulation (Phase B atomicAdd from 4 CTAs), one tile does the final tonemap
(RMSNorm) before writing to the framebuffer.

## The Cross-CTA Sync Problem

Phase B has **`num_q_tiles × num_kv_heads`** CTAs per sequence, each atomicAdd'ing
its partial W_O contribution to a shared FP32 output buffer. Phase C (RMSNorm)
requires the **complete sum** of all CTAs before it can proceed.

For Qwen3.5-27B: `num_q_tiles = ceil(group_size/cta_q) = ceil(6/16) = 1`,
`num_kv_heads = 4`, so **4 CTAs per sequence**. But the kernel is generic — a model
with `group_size=32, cta_q=16` would have `num_q_tiles=2`, giving **8 CTAs per
sequence**. The arrival threshold must NOT be hardcoded.

This is the same problem as CUTLASS split-K GEMM epilogues.

### Solution: Atomic Arrival Counter + Last-CTA-Runs-Epilogue

```
// total_ctas_per_seq is passed as a kernel parameter:
//   total_ctas_per_seq = grid_dim_x * grid_dim_y  (num_q_tiles × num_kv_heads)
// For Qwen3.5-27B: 1 × 4 = 4

Phase B (all CTAs):
    1. atomicAdd W_O partial sums to global buffer       (existing)
    2. __threadfence()                                    (NEW — ensures writes visible)
    3. old = atomicAdd(&arrival_count[seq_idx], 1)        (NEW — integer atomic)
    4. if old == total_ctas_per_seq - 1:                  (I'm the last CTA)
           → run Phase C (residual add + RMSNorm)
       else:
           → return (other CTAs retire early)
```

Only 1 CTA (the last to arrive) executes Phase C. This CTA has 128 threads — more
than enough for a 5120-element reduction.

### Arrival Counter Reset

The arrival counter buffer **must be zeroed before every kernel launch**, not just
once at allocation. vLLM reuses buffers across decode steps — a stale counter (value
4 from the previous step) would cause no CTA to match the arrival threshold.

```python
# In __call__, before kernel launch:
arrival_count.zero_()  # [num_seqs] int32, negligible cost
```

### New PTX Helpers Required

| Helper | PTX | Purpose |
|--------|-----|---------|
| `_threadfence()` | `membar.gl;` | Global memory fence — ensures Phase B atomicAdd writes are visible to Phase C reads |
| `_atomic_add_u32(addr, val)` | `atom.global.add.u32 $0, [$1], $2;` | Integer atomic add, returns old value (for arrival detection) |
| `_rsqrt_approx_f32(x)` | `rsqrt.approx.ftz.f32 $0, $1;` | Hardware reciprocal square root |
| `_cvt_f32_to_bf16_store(addr, val)` | `cvt.rn.bf16x2.f32 + st.global.b16` | FP32 → BF16 conversion + global store (for output writes via raw pointers) |

All are single-instruction (or short-sequence) `@dsl_user_op` wrappers — trivial,
matching the existing Phase B helpers.

**Memory fence correctness note:** `membar.gl` ensures this thread's prior stores
are visible to all other threads in the GPU. Combined with the `atom.global.add.u32`
arrival counter (which acts as an acquire-release synchronization point — the return
value `old == N-1` is only observable after all prior CTAs' atomicAdds have
committed), this provides sufficient ordering. If correctness issues arise on SM120,
escalate to `fence.sc.gpu` (system-scope fence).

## Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Cross-CTA sync | Atomic arrival counter | Proven pattern (CUTLASS split-K), minimal overhead |
| Phase C executor | Single CTA (last arrival) | 128 threads suffice for 5120 elements; simpler than grid-wide barrier |
| Reduction | Warp shuffle + SMEM cross-warp | Same pattern as Phase A softmax reduction — proven in this kernel |
| rsqrt | PTX hardware approx | Single instruction, sufficient precision for RMSNorm |
| Gamma weights | Stream from global (BF16) | 10KB, L2-cached — same gamma every layer, no SMEM staging needed |
| Output format | BF16 to global | Matches input format expected by MLP / next-layer QKV proj |
| Residual output | FP32→BF16 to global | Must persist for next layer's residual connection |
| Arrival counter buffer | Per-launch zeroed, [num_seqs] int32 | One counter per sequence, `zero_()` before **every** kernel launch |
| Arrival threshold | `grid_dim_x * grid_dim_y - 1` (computed, not hardcoded) | Generic across models — Qwen3.5-27B=3, but group_size>16 models would differ |
| Register strategy | Serialized 5×8 (default) | 8 accumulators per group, SMEM staging between reduction passes — matches Phase B pattern |

## RMSNorm Math

```
Input:  wo_output[5120] (FP32, from Phase B atomicAdd)
        residual[5120]  (BF16, from previous layer)
        gamma[5120]     (BF16, learned weight)
        eps = 1e-6

Step 1: new_residual = residual + wo_output          (FP32 arithmetic)
Step 2: variance = sum(new_residual^2) / 5120         (reduction)
Step 3: inv_rms = rsqrt(variance + eps)               (PTX hardware)
Step 4: hidden_states = new_residual * inv_rms * gamma (element-wise)

Output: hidden_states[5120] (→BF16, to global — MLP input)
        new_residual[5120]  (→BF16, to global — next layer's residual)
```

## Thread Tiling

128 threads, each owns 40 contiguous elements of hidden_dim=5120:
`my_start = tid * 40`, processes elements `[my_start, my_start+40)`.

Same tiling as Phase B (W_O GEMV).

### Per-Thread Work (Serialized 5×8 — Default Strategy)

Each thread owns 40 elements, but processes them in **5 groups of 8** — matching
the Phase B `range_constexpr(5)` pattern. This keeps only 8 `new_residual` FP32
values live at a time instead of 40.

**Pass 1 — Residual add + sum-of-squares accumulation (no SMEM staging):**

```
ss = 0.0                              // per-thread sum-of-squares

for _grp in range_constexpr(5):       // constexpr unroll, 5 groups
    base = my_start + _grp * 8

    // 8 scalar accumulators (same pattern as Phase B a0..a7)
    nr0 = bf16_to_f32(residual[base+0]) + wo_output[base+0]
    nr1 = bf16_to_f32(residual[base+1]) + wo_output[base+1]
    ...
    nr7 = bf16_to_f32(residual[base+7]) + wo_output[base+7]

    ss += nr0*nr0 + nr1*nr1 + ... + nr7*nr7

    // NOTE: Do NOT stage to SMEM here. Writing 5 groups to the same 8
    // SMEM slots per thread would overwrite groups 0-3 with group 4.
    // Instead, Pass 3 re-reads from global (L2-hot).
```

**Pass 2 — Reduction (sum-of-squares → inv_rms):**

Intra-warp: 5 butterfly shuffle steps reduce 32 lanes → 1 partial sum.

```
ss = ss + shfl_xor_sync(ss, 1)
ss = ss + shfl_xor_sync(ss, 2)
ss = ss + shfl_xor_sync(ss, 4)
ss = ss + shfl_xor_sync(ss, 8)
ss = ss + shfl_xor_sync(ss, 16)
```

Cross-warp: 4 warp partial sums → SMEM → warp 0 reads all 4, sums.
Same pattern as Phase A's cross-warp softmax reduction (kernel.py:1253-1364).
Uses Phase A's `sync_md` buffer (4 FP32 slots) — only 16 bytes.

```
lane 0 of each warp: st.shared.f32(reduce_scratch[warp_id], ss)
sync_threads()
warp 0, lane 0:
    total = ld(smem[0]) + ld(smem[1]) + ld(smem[2]) + ld(smem[3])
    variance = total / Float32(hidden_dim)   // NOT hardcoded 5120 — use parameter
    inv_rms = rsqrt_approx(variance + rmsnorm_eps)
    st.shared.f32(smem[0], inv_rms)          // broadcast slot — safe to overwrite,
                                              // partial sums already consumed above
sync_threads()
all threads: inv_rms = ld.shared.f32(smem[0])
```

**Pass 3 — Re-read from global, scale by inv_rms × gamma, write outputs:**

Re-reads `residual` and `wo_output` from global memory. Both are L2-hot: `wo_output`
was just written by Phase B (same kernel), and `residual` was just read in Pass 1.
This avoids SMEM staging entirely — no SMEM needed beyond the 16-byte reduction
scratch.

```
for _grp in range_constexpr(5):
    base = my_start + _grp * 8

    for _oi in range_constexpr(8):
        idx = base + _oi
        // Re-read and recompute new_residual (L2-hot, ~free)
        new_res = bf16_to_f32(residual[seq_idx * hidden_dim + idx])
                  + wo_output[seq_idx * hidden_dim + idx]
        gamma_f32 = bf16_to_f32(gamma[idx])
        hidden = new_res * inv_rms * gamma_f32

        write hidden (→BF16) to hidden_states_output[seq_idx * hidden_dim + idx]
        write new_res (→BF16) to residual_output[seq_idx * hidden_dim + idx]
```

**Why re-read instead of SMEM staging:** Staging 40 values per thread to SMEM
would require either (a) 20 KB SMEM (`128 threads × 40 values × 4B`) with unique
offsets per group, or (b) a sync_threads between every group to reuse 4 KB of SMEM.
Both add complexity. Re-reading 20 KB of L2-hot data is simpler and effectively free
— the L2 bandwidth for 40 KB of reads is negligible compared to Phase B's 3.93 MB
weight stream.

### Register Budget

| Component | Registers/thread |
|-----------|-----------------|
| Phase A (attention) | ~20 |
| Phase B (W_O GEMV) | ~50 |
| Phase C new_residual (8 per group, recycled) | 8 |
| Phase C sum-of-squares accumulator | 1 |
| Phase C inv_rms broadcast | 1 |
| Phase C scratch (loads, BF16 conversions) | ~4 |
| **Phase C marginal cost** | **~14** |
| **Cumulative A+B+C** | **~64 of 256** |

Phases A, B, and C are sequential — their registers are recycled, not stacked.
The peak at any point is max(A, B, C) ≈ 50 (Phase B), not the sum.
Phase C's 14 registers are well under Phase B's 50, so **no new peak**.

### SMEM Budget

```
Phase A (attention):                    45,568 B
Phase C reduction scratch:                  16 B  (reuses sync_md — 4 × FP32)
Phase C new_residual staging:                0 B  (re-reads from global, no SMEM)
                                       ─────────
Peak:                                   45,584 B  (of 101,376 B available)
Headroom:                               55,792 B  (unchanged from Phase B)
```

Phase C adds only 16 bytes of SMEM for the cross-warp reduction scratch, reusing
Phase A's `sync_md` buffer. No SMEM staging for `new_residual` — Pass 3 re-reads
from L2-hot global memory instead.

## New Kernel Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `residual_ptr` | Int64 | Pointer to residual tensor [num_seqs, hidden_dim] BF16 |
| `gamma_ptr` | Int64 | Pointer to RMSNorm gamma weights [hidden_dim] BF16 |
| `rmsnorm_output_ptr` | Int64 | Pointer to normalized output [num_seqs, hidden_dim] BF16 |
| `residual_output_ptr` | Int64 | Pointer to updated residual [num_seqs, hidden_dim] BF16 |
| `arrival_count_ptr` | Int64 | Pointer to arrival counter [num_seqs] int32, **zeroed before every launch** |
| `rmsnorm_eps` | Float32 | Epsilon (1e-6 for Qwen3.5-27B, from `config.rms_norm_eps`) |
| `hidden_dim` | Int32 | Hidden dimension (5120 for Qwen3.5-27B — **do not hardcode**, read from model config) |
| `total_ctas_per_seq` | Int32 | `grid_dim_x * grid_dim_y` — arrival threshold is `total_ctas_per_seq - 1` |
| `rmsnorm_fused` | Int32 | 0=disabled, 1=enabled (same pattern as `wo_fused`) |

When `rmsnorm_fused == 0`, Phase C is skipped — kernel returns after Phase B.
Clean opt-in, zero disruption to existing path.

**Implementation note on hidden_dim:** The variance divisor and per-thread element
count (`hidden_dim / 128`) must be computed from the `hidden_dim` parameter, not
hardcoded as 5120 or 40. The Phase 2 spec had a hardcoded `3584` that cost a
rebuild — don't repeat this. For Qwen3.5-27B: `hidden_dim=5120`, `n_per_thread=40`.
Models with `variance_size_override` (layernorm.py:128) would need the divisor
adjusted — not applicable to Qwen3.5 but worth a guard.

## Integration Point: Qwen2DecoderLayer.forward()

### Today (with Phase B only — qwen2.py:264-302):
```python
# In Qwen2Attention.forward(), Phase B fused path:
attn_output = self.attn(q, k, v)          # Phase A+B fused
output = wo_output.to(hidden_states.dtype) # FP32 → BF16 cast
return output

# In Qwen2DecoderLayer.forward():
hidden_states = self.self_attn(positions, hidden_states)
hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
hidden_states = self.mlp(hidden_states)
```

### After Phase C:
```python
# In Qwen2Attention.forward(), Phase B+C fused path:
attn_output = self.attn(q, k, v)          # Phase A+B+C fused
# CRITICAL: Do NOT call wo_output.to(hidden_states.dtype) when Phase C is active.
# Phase C reads wo_output as FP32 directly. The BF16 cast would corrupt the input.
if rmsnorm_fused:
    return rmsnorm_output, residual_output  # already BF16, already RMSNorm'd
else:
    output = wo_output.to(hidden_states.dtype)
    return output

# In Qwen2DecoderLayer.forward():
hidden_states, residual = self.self_attn(positions, hidden_states, residual)
# post_attention_layernorm call is SKIPPED — Phase C already did it
hidden_states = self.mlp(hidden_states)
```

### Side-Channel Extension

Same pattern as Phase B — set attributes on `self.attn` before the call:
```python
self.attn._rmsnorm_gamma = self.post_attention_layernorm.weight
self.attn._rmsnorm_residual = residual
self.attn._rmsnorm_output = hidden_states_buf    # pre-allocated [num_seqs, 5120] BF16
self.attn._residual_output = residual_buf         # pre-allocated [num_seqs, 5120] BF16
self.attn._arrival_count = arrival_count_buf      # pre-allocated [num_seqs] int32
```

### Buffer Semantics

Phase C writes to **separate output buffers** (`rmsnorm_output`, `residual_output`),
NOT in-place to the input tensors. This differs from vLLM's `fused_add_rms_norm`
which operates in-place. The separate buffers avoid aliasing hazards — Phase C reads
`residual` while writing `residual_output`, and these must not be the same tensor.

The caller is responsible for:
1. Pre-allocating `rmsnorm_output` and `residual_output` (persistent across steps)
2. Zeroing `arrival_count` before **every** kernel launch (`arrival_count.zero_()`)
3. Zeroing `wo_output` before every kernel launch (existing Phase B requirement)
4. Passing `residual_output` as the next layer's `residual` input

## Memory Traffic Analysis

### Current (Phase B only):
```
Phase B writes:   5120 × 4B = 20 KB (atomicAdd FP32 to global)
Kernel returns.
fused_add_rms_norm reads:  5120 × 2B = 10 KB (residual BF16)
                         + 5120 × 4B = 20 KB (W_O output FP32→BF16)
                         + 5120 × 2B = 10 KB (gamma BF16)
fused_add_rms_norm writes: 5120 × 2B = 10 KB (hidden_states BF16)
                         + 5120 × 2B = 10 KB (residual BF16)
Total reads:  40 KB    Total writes: 40 KB
```

### After Phase C fusion:
```
Phase B writes:   20 KB (atomicAdd FP32 — unchanged, goes to L2)
Phase C reads:    20 KB (W_O output — L2 hit, just written by Phase B)
                + 10 KB (residual BF16 from global)
                + 10 KB (gamma BF16 — L2-cached, same every layer)
Phase C writes:   10 KB (hidden_states BF16)
                + 10 KB (residual BF16)
Total reads:  40 KB    Total writes: 40 KB
```

The **byte counts are identical** — the savings come from:
1. **L2 locality**: Phase B's atomicAdd output is guaranteed L2-hot when Phase C reads it (same kernel, immediate succession). The separate kernel launch would have L2 eviction risk.
2. **Eliminated kernel launch**: One fewer CUDA kernel dispatch per layer (×64 layers = 64 fewer launches per decode step).
3. **Scheduler efficiency**: 3 CTAs retire after Phase B, freeing SM resources for other work.

## Testing and Validation

### Correctness — three levels (same structure as Phase B)

1. **Standalone kernel test:** Extend `test_cute_kernel_standalone.py`.
   - Random Q, K, V, W_O weights, residual, gamma
   - Fused (A+B+C) vs unfused (A+B → torch rms_norm)
   - Tolerance: `max_diff < 0.01` for both hidden_states and residual
     (BF16 quantization alone introduces ~0.004 relative error for values >1.0;
     0.001 absolute would be too tight for large residual magnitudes)

2. **GSM8K sanity gate:** Serve with fused Phase C, compare to Phase B baseline.
   Must match within 5% accuracy.

3. **A/B output comparison:** Same prompt, fused vs unfused, compare logits.

### Specific things to validate

- **Arrival counter correctness:** All `total_ctas_per_seq` CTAs must arrive.
  No CTA runs Phase C prematurely. Test with `num_seqs > 1` to verify per-sequence
  isolation. Test with `num_seqs=1` to verify the simple case.
- **Arrival counter reset:** Run 2+ decode steps and verify Phase C fires on every
  step (catches stale counter bug — WARNING #5 in audit).
- **threadfence correctness:** Phase C reads must see Phase B's atomicAdd results.
  Without `membar.gl`, Phase C could read stale zeros.
- **Reduction accuracy:** Compare `inv_rms` to `torch.rsqrt(x.pow(2).mean() + eps)`.
- **dtype guard:** Verify `wo_output.to(dtype)` is NOT called when Phase C is active
  (catches BLOCKER #2 from audit — FP32 buffer would be truncated to BF16 before
  Phase C reads it).
- **Buffer non-aliasing:** Verify `residual_ptr != residual_output_ptr` — Phase C
  reads residual while writing residual_output, so they must not alias.

### Success criteria

- Standalone: max_diff < 0.01 (hidden_states), < 0.001 (residual)
- GSM8K: within 5% of Phase B baseline
- No regression in Phase B path when Phase C disabled
- Arrival counter never incorrect (fuzz with num_seqs=1..32)

## Risk Assessment

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| `membar.gl` insufficient | Low | Escalate to `fence.sc.gpu`; see memory fence note above |
| Arrival counter race | Low | Atomic returns old value — deterministic |
| Stale arrival counter | Medium | `zero_()` before every launch — easy to forget, add to checklist |
| rsqrt precision | Very low | Hardware approx matches PyTorch within FP32 ULP |
| `wo_output.to(dtype)` called with Phase C | High if not guarded | Explicit `if rmsnorm_fused` guard in Qwen2Attention.forward() |
| Hardcoded dimensions | Medium | Pass `hidden_dim`, `total_ctas_per_seq` as params — learned from Phase 2 |

## Key Files

| File | Role |
|------|------|
| `vllm/v1/attention/backends/cute_paged/kernel.py` | Kernel — add Phase C after Phase B |
| `vllm/v1/attention/backends/cute_paged/_backend.py` | Backend — wire RMSNorm params |
| `vllm/model_executor/models/qwen2.py` | Model — skip post_attention_layernorm when fused |
| `tests/test_cute_kernel_standalone.py` | Standalone correctness test |

## Implementation Guide

### Step-by-step build order

Each step should compile+test independently before moving to the next.

**Step 1: New PTX helpers (kernel.py, ~line 800)**

Add these `@dsl_user_op` helpers next to the existing `_atomic_add_f32`:

```python
@dsl_user_op
def _threadfence(*, loc=None, ip=None):
    """Global memory fence — membar.gl"""
    _llvm_dialect.inline_asm(
        T.i32(), [],                    # dummy return, no inputs
        "membar.gl; mov.u32 $0, 0;",    # fence + dummy mov for return
        "=r",
        has_side_effects=True, loc=loc, ip=ip)

@dsl_user_op
def _atomic_add_u32(addr: Int64, val: Int32, *, loc=None, ip=None) -> Int32:
    """Integer atomicAdd, returns old value."""
    addr_ir = Int64(addr).ir_value(loc=loc, ip=ip)
    val_ir = Int32(val).ir_value(loc=loc, ip=ip)
    result_ir = _llvm_dialect.inline_asm(
        T.i32(), [addr_ir, val_ir],
        "atom.global.add.u32 $0, [$1], $2;", "=r,l,r",
        has_side_effects=True, loc=loc, ip=ip)
    return Int32(result_ir)

@dsl_user_op
def _rsqrt_approx_f32(x: Float32, *, loc=None, ip=None) -> Float32:
    """Hardware reciprocal square root — rsqrt.approx.ftz.f32"""
    x_ir = Float32(x).ir_value(loc=loc, ip=ip)
    result_ir = _llvm_dialect.inline_asm(
        T.f32(), [x_ir],
        "rsqrt.approx.ftz.f32 $0, $1;", "=f,f",
        has_side_effects=True, loc=loc, ip=ip)
    return Float32(result_ir)
```

**Test:** Bind-mount kernel.py, verify existing Phase A+B still compiles.
The new helpers are defined but not called yet.

**Step 2: Wire Phase C parameters through `__call__` (kernel.py, ~line 1513)**

Add parameters to `__call__` following the Phase B pattern:

```python
# Phase C: RMSNorm fusion params (optional)
rmsnorm_gamma = kwargs.get("rmsnorm_gamma", None)      # [hidden_dim] BF16
rmsnorm_residual = kwargs.get("rmsnorm_residual", None) # [num_seqs, hidden_dim] BF16
rmsnorm_output = kwargs.get("rmsnorm_output", None)     # [num_seqs, hidden_dim] BF16
residual_output = kwargs.get("residual_output", None)   # [num_seqs, hidden_dim] BF16
arrival_count = kwargs.get("arrival_count", None)        # [num_seqs] int32

if rmsnorm_gamma is not None:
    # Arrival counter self-resets in kernel (last CTA atomicAdds -N).
    # Zero-init once at allocation; no per-launch zero_() needed.
    rmsnorm_gamma_ptr = Int64(rmsnorm_gamma.data_ptr())
    rmsnorm_residual_ptr = Int64(rmsnorm_residual.data_ptr())
    rmsnorm_output_ptr = Int64(rmsnorm_output.data_ptr())
    residual_output_ptr = Int64(residual_output.data_ptr())
    arrival_count_ptr = Int64(arrival_count.data_ptr())
    rmsnorm_eps_val = 1e-6  # from config, don't hardcode in kernel
    hidden_dim_val = Int32(rmsnorm_gamma.shape[0])  # derive from tensor shape
    total_ctas = Int32(grid[0] * grid[1])
    rmsnorm_fused_flag = Int32(1)
else:
    # zeros — kernel guards on rmsnorm_fused_flag
    rmsnorm_gamma_ptr = Int64(0)
    # ... (same zero pattern as Phase B wo_* params)
    rmsnorm_fused_flag = Int32(0)
```

**Key:** Derive `hidden_dim` from `rmsnorm_gamma.shape[0]`, don't hardcode.
Derive `total_ctas` from `grid[0] * grid[1]`, don't hardcode.

**Test:** Bind-mount, verify Phase A+B still works with the extra (zeroed) params.

**Step 3: Phase C kernel code (kernel.py, after Phase B's atomicAdd block)**

Insert after the `# atomicAdd accumulators to global FP32 output buffer` block
(~line 1511):

```python
# ═══════════════════════════════════════════════════
# Phase C: Residual Add + RMSNorm (last CTA only)
# ═══════════════════════════════════════════════════
if rmsnorm_fused != Int32(0):
    # Fence: ensure all Phase B atomicAdd writes are globally visible
    _threadfence()

    # Arrival counter: last CTA runs Phase C, others return
    old_count = _atomic_add_u32(
        arrival_count_ptr + Int64(seq_idx * Int32(4)),
        Int32(1))

    if old_count == total_ctas_per_seq - Int32(1):
        # I am the last CTA — run Phase C

        hd = hidden_dim  # NOT hardcoded 5120
        n_per_thr = hd // Int32(128)  # 40 for Qwen3.5-27B
        res_base = rmsnorm_residual_ptr + Int64(seq_idx * hd * Int32(2))  # BF16
        wo_base  = wo_output_ptr + Int64(seq_idx * hd * Int32(4))         # FP32

        ss = Float32(0.0)  # per-thread sum-of-squares

        # Pass 1: residual add + sum-of-squares (NO SMEM staging)
        for _grp in cutlass.range_constexpr(5):
            # ... 8 scalar accumulators, same Phase B pattern ...
            # Load residual (BF16 → FP32), add wo_output (FP32)
            # Accumulate sum-of-squares into ss
            # Values are NOT stored — will be re-read in Pass 3

        # Pass 2: reduction (warp shuffle + cross-warp SMEM)
        # ... see pseudocode above ...
        # Result: all threads have inv_rms

        # Pass 3: re-read from global (L2-hot), scale + write
        for _grp in cutlass.range_constexpr(5):
            # ... re-read residual + wo_output (L2 hit) ...
            # ... new_res * inv_rms * gamma → write BF16 ...

        # Self-reset arrival counter (avoids caller zero_() per layer)
        if tid == Int32(0):
            _atomic_add_u32(
                arrival_count_ptr + Int64(seq_idx * Int32(4)),
                Int32(-total_ctas_per_seq))  # reset to 0
```

**Critical implementation details:**
- The `if old_count == total_ctas_per_seq - Int32(1)` is a **dynamic if** — CuTe
  DSL supports this, but no variables can be defined inside it that are used outside.
  Since Phase C is the last thing before kernel return, this is fine.
- Use `n_per_thr = hd // Int32(128)` not hardcoded 40.
- The `for _grp in cutlass.range_constexpr(5)` assumes `n_per_thr=40` and
  group_size=8. For generality, this should be `range_constexpr(n_per_thr // 8)`,
  but constexpr requires a Python int, not a runtime value. **For Qwen3.5-27B,
  hardcoding 5 groups of 8 is acceptable.** Document this assumption.
- **No SMEM staging:** Pass 3 re-reads `residual` and `wo_output` from global
  memory rather than staging `new_residual` to SMEM. Both are L2-hot (Phase B just
  wrote `wo_output`, Pass 1 just read `residual`). This avoids a bug where 5 groups
  writing to the same 8 SMEM slots per thread would overwrite groups 0-3.
- **Self-resetting counter:** The last CTA resets `arrival_count[seq_idx]` to 0
  after Phase C, eliminating the need for `arrival_count.zero_()` in the caller.
  Still zero-init the buffer once at allocation as a safety net.

**Test:** Standalone kernel test with random inputs. Fused vs
`torch.nn.functional.rms_norm`. Target: max_diff < 0.01.

**Step 4: Backend wiring (_backend.py)**

Wire the RMSNorm side-channel params from `Qwen2Attention` to the kernel `__call__`.
Follow the exact Phase B pattern — check for `_rmsnorm_gamma` attribute, pass or
don't pass.

**Step 5: Model integration (qwen2.py)**

In `Qwen2Attention.forward()`:
- When Phase C is active: return `(rmsnorm_output, residual_output)` directly.
  Do **NOT** call `wo_output.to(hidden_states.dtype)`.
- When Phase C is inactive: existing Phase B path unchanged.

In `Qwen2DecoderLayer.forward()`:
- When Phase C is active: skip `self.post_attention_layernorm()` call.
- `self_attn()` now returns `(hidden_states, residual)` instead of just `hidden_states`.

**Step 6: Docker build + GSM8K gate**

Build with `--no-cache`, serve, run GSM8K sanity check. Must match Phase B baseline.

### Checklist before implementation

- [ ] Read `config.json` for `hidden_size`, `rms_norm_eps` — do NOT rely on memory
- [ ] Verify `post_attention_layernorm.weight.shape` matches `hidden_size`
- [ ] Verify `arrival_count` is zero-init'd at allocation AND kernel self-resets after Phase C
- [ ] Verify `wo_output.to(dtype)` is guarded by `if not rmsnorm_fused`
- [ ] Verify `residual_ptr != residual_output_ptr` (no aliasing)
- [ ] Verify Pass 3 re-reads from global (NOT SMEM) — SMEM staging has overwrite bug
- [ ] Verify `Qwen2Attention.forward()` return type changes are handled in `DecoderLayer`
- [ ] Test with `num_seqs=1` AND `num_seqs>1`
- [ ] Test 2+ consecutive decode steps (catches stale arrival counter)
- [ ] Test that Phase B path still works when Phase C is disabled (`rmsnorm_fused=0`)

## Game Engine Parallels

| Game Engine Concept | Kernel Equivalent |
|--------------------|-------------------|
| G-buffer resolve (Phase A) | Attention output |
| Additive light accumulation (Phase B) | W_O atomicAdd from 4 CTAs |
| Tonemapping / color grading (Phase C) | RMSNorm — normalize the accumulated result |
| "Last tile" post-process trigger | Atomic arrival counter — last CTA runs epilogue |
| HDR → LDR conversion | FP32 → BF16 output conversion |
| Tile retirement | 3 CTAs retire early, freeing shader cores |
