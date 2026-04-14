# Unreal Kernel Phase 2: W_O GEMV Fusion — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fuse the W_O output projection (o_proj) into the CuTe DSL paged attention decode kernel as a Phase B epilogue, eliminating one HBM round-trip per decode layer.

**Architecture:** Two-phase kernel — Phase A (attention, unchanged) writes BF16 output to SMEM, Phase B (new W_O GEMV epilogue) consumes it on-chip. 128 threads each own 28 output elements, accumulate in registers across 96 inner iterations, then atomicAdd to a caller-zeroed global buffer. NVFP4 weights loaded via raw pointers with swizzled scale factor decoding.

**Tech Stack:** CuTe DSL (Python), inline PTX via `@dsl_user_op` + `llvm.inline_asm`, NVFP4 E2M1 weights, FP8 E4M3 swizzled block scales, SM120/SM121.

**Spec:** `docs/superpowers/specs/2026-04-13-unreal-kernel-wo-fusion-design.md`

---

## NVFP4 Weight Format Reference

This section documents the NVFP4 weight layout for `o_proj` (RowParallelLinear). Every task that touches weight loading or dequant should refer here.

**Qwen3.5-27B o_proj dimensions (TP=1):**
- Logical weight: `[N=3584, K=6144]` (output_size × input_size, where input_size = 24 Q heads × 256 head_dim)
- `layer.weight`: `[3584, 3072]` uint8 — row-major, 2 FP4 values packed per byte
- `layer.weight_scale`: `[3584, 384]` float8_e4m3fn — **swizzled** block scales (group_size=16, so K/16 = 384)
- `layer.alpha`: scalar float32 — `input_global_scale * weight_global_scale`
- `layer.weight_global_scale`: scalar float32

**FP4 E2M1 nibble packing:**
- Low nibble (bits [3:0]) = even-indexed element (element 2i)
- High nibble (bits [7:4]) = odd-indexed element (element 2i+1)
- Each nibble: `[sign(1) | exp(2) | mant(1)]`
- Unsigned values: `{0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0}`

**Swizzled scale factor offset formula** (for logical position `(m, k_group)`):
```
numKTiles = ceil(K_sf / 4)        # K_sf = K/16 = 384, numKTiles = 96
mTileIdx  = m / 128
outerMIdx = m % 32
innerMIdx = (m / 32) % 4
kTileIdx  = k_group / 4
innerKIdx = k_group % 4
SFOffset  = (mTileIdx * numKTiles + kTileIdx) * 512
          + outerMIdx * 16 + innerMIdx * 4 + innerKIdx
```

**Per-CTA slice (one KV head group of 6 Q heads):**
- Attention output: `[1, 1536]` (6 heads × 256 head_dim, concatenated)
- W_O column range: `[kv_head_idx * 1536 .. (kv_head_idx+1) * 1536 - 1]`
- Packed bytes per row: `1536 / 2 = 768` (columns `[kv_head_idx * 768 .. (kv_head_idx+1) * 768 - 1]`)
- Scale k_group range: `[kv_head_idx * 96 .. (kv_head_idx+1) * 96 - 1]`

**Dequant formula (per element):**
```
dequant(nibble, scale_fp8, global_scale) =
    e2m1_to_f32(nibble) * fp8_e4m3_to_f32(scale_fp8) * global_scale
```

Where `global_scale = layer.weight_global_scale` (NOT alpha — alpha includes the input scale which doesn't apply to weight-only dequant).

**Important:** The CUTLASS GEMM applies `alpha = input_global_scale * weight_global_scale` as an epilogue scalar to the full matmul result. For our fused GEMV, we dequant weights with `weight_global_scale` only, since the attention output is already in BF16 (not quantized). The final atomicAdd output should match `o_proj(attn_output)` — which is `attn_output @ W_O^T * 1.0` (no scale on input side since attention output is BF16, and the CUTLASS kernel internally handles the scale split).

**Correction:** Actually, the CUTLASS kernel quantizes the input activation with `input_global_scale_inv`, then multiplies the output by `alpha`. So the net effect is `output = (x * inv_scale) @ W_dequant^T * alpha = x @ W_dequant^T * (alpha * inv_scale) = x @ W_dequant^T * weight_global_scale`. Our kernel gets BF16 attention output (not quantized), so we multiply dequanted weights by `weight_global_scale`: `output = attn_bf16 @ (W_fp4_dequant * weight_scale * weight_global_scale)^T`. This is equivalent.

---

## File Structure

| File | Action | Responsibility |
|------|--------|---------------|
| `vllm/v1/attention/backends/cute_paged/kernel.py` | Modify | Add FP4 dequant PTX helpers, Phase B GEMV epilogue to DecodeKernel, wire new params in `__call__` |
| `vllm/v1/attention/backends/cute_paged/_backend.py` | Modify | Accept W_O params in `forward()`, pass to kernel |
| `vllm/model_executor/models/qwen2.py` | Modify | Extract o_proj weights, pass to attention, skip o_proj when fused |
| `vllm/v1/attention/backends/cute_paged/warmup.py` | No change | W_O params default to None/Int64(0); existing warmup compiles the same kernel since `wo_fused` is a runtime branch |
| `tests/nvllm/attention/test_cute_kernel_standalone.py` | Modify | Add fused W_O GEMV test case |

---

### Task 1: Add FP4 E2M1 Dequant PTX Helpers

**Files:**
- Modify: `vllm/v1/attention/backends/cute_paged/kernel.py:70-170` (PTX utilities section)

These helpers go in the `if _CUTE_AVAILABLE:` block alongside existing PTX utilities.

- [ ] **Step 1: Add `_fp4_nibble_to_f32` — convert a 4-bit E2M1 nibble (as Int32) to Float32**

Add after the existing `fp8x4_e4m3_to_bfloat2x2` helper:

```python
@cute.jit
def _fp4_nibble_to_f32(nibble: Int32) -> Float32:
    """Convert a single FP4 E2M1 nibble (4 bits in an Int32) to Float32.

    E2M1 format: [sign(1) | exp(2) | mant(1)]
    Unsigned values: {0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0}

    Conversion: build an IEEE 754 float32 from the E2M1 fields.
    - For exp > 0 (normal): f32_exp = e2m1_exp + 126 (bias adjust: 127 - 1)
                             f32_mant = e2m1_mant << 22
    - For exp == 0, mant == 1 (subnormal 0.5): hardcode 0x3F000000
    - For exp == 0, mant == 0 (zero): hardcode 0x00000000
    - Sign: OR into bit 31
    """
    sign = (nibble >> Int32(3)) & Int32(1)
    exp2 = (nibble >> Int32(1)) & Int32(3)
    mant1 = nibble & Int32(1)

    # Normal case: exp > 0
    # f32 bits = sign<<31 | (exp2 + 126)<<23 | mant1<<22
    f32_normal = (sign << Int32(31)) | ((exp2 + Int32(126)) << Int32(23)) | (mant1 << Int32(22))

    # Subnormal case: exp == 0, mant == 1 → value = 0.5
    # 0x3F000000 = 0.5f, with sign
    f32_subnormal = (sign << Int32(31)) | Int32(0x3F000000)

    # Zero case: exp == 0, mant == 0
    f32_zero = sign << Int32(31)  # +0 or -0

    # Select: if exp > 0 → normal, elif mant > 0 → subnormal, else → zero
    is_normal = exp2 > Int32(0)
    is_subnormal = (exp2 == Int32(0)) & (mant1 > Int32(0))

    # Branchless select using multiply-mask
    # CuTe DSL doesn't have ternary, use arithmetic
    result_bits = f32_normal * is_normal + f32_subnormal * is_subnormal + f32_zero * (Int32(1) - is_normal - is_subnormal)

    return _bitcast_i32_to_f32(result_bits)
```

- [ ] **Step 2: Add `_bitcast_i32_to_f32` PTX helper**

```python
@dsl_user_op
def _bitcast_i32_to_f32(bits, *, loc=None, ip=None) -> Float32:
    """Reinterpret Int32 bits as Float32 (mov.b32)."""
    bits_ir = bits.ir_value(loc=loc, ip=ip)
    result_ir = _llvm_dialect.inline_asm(
        T.f32(),
        [T.i32()],
        "mov.b32 $0, $1;",
        "=f,r",
        asm_dialect=0,
        operands_=[bits_ir],
        loc=loc,
        ip=ip,
    )
    return Float32(result_ir)
```

- [ ] **Step 3: Add `_fp4_byte_to_f32x2` — unpack uint8 byte into two Float32 values**

```python
@cute.jit
def _fp4_byte_to_f32x2(byte_val: Int32) -> tuple:
    """Unpack one uint8 byte (two packed FP4 E2M1 values) into (Float32, Float32).

    Low nibble  = even-indexed element (element 2i)
    High nibble = odd-indexed element (element 2i+1)
    """
    lo = byte_val & Int32(0x0F)
    hi = (byte_val >> Int32(4)) & Int32(0x0F)
    return _fp4_nibble_to_f32(lo), _fp4_nibble_to_f32(hi)
```

- [ ] **Step 4: Add `_ld_swizzled_scale` — compute swizzled offset and load FP8 E4M3 scale as Float32**

```python
@cute.jit
def _ld_swizzled_scale(
    sf_ptr: Int64,
    m: Int32,
    k_group: Int32,
    num_k_tiles: Int32,
) -> Float32:
    """Load one FP8 E4M3 block scale from the swizzled layout, return as Float32.

    Swizzle layout: [numMTiles, numKTiles, 32, 4, 4]
    where numMTiles = ceil(N/128), numKTiles = ceil(K_sf/4).
    """
    m_tile = m >> Int32(7)           # m / 128
    outer_m = m & Int32(31)          # m % 32
    inner_m = (m >> Int32(5)) & Int32(3)  # (m / 32) % 4
    k_tile = k_group >> Int32(2)     # k_group / 4
    inner_k = k_group & Int32(3)     # k_group % 4

    sf_offset = (m_tile * num_k_tiles + k_tile) * Int32(512) \
        + outer_m * Int32(16) + inner_m * Int32(4) + inner_k

    # Load 1 byte (FP8 E4M3) via _ld_global_b32 at aligned address,
    # then extract the target byte
    aligned_addr = sf_ptr + Int64(sf_offset & Int32(0xFFFFFFFC))
    raw_word = _ld_global_b32(aligned_addr)
    byte_pos = sf_offset & Int32(3)
    scale_byte = _extract_byte_from_b32(raw_word, byte_pos)

    # Convert FP8 E4M3 byte to Float32 via the existing chain:
    # Pack two identical bytes, convert pair, take first
    packed = _pack_lo16(scale_byte, scale_byte)
    bf16_lo, _bf16_hi = fp8x4_e4m3_to_bfloat2x2(packed)
    # bf16_lo contains two BF16 values packed as Uint32 — extract first
    # Actually we need a simpler path: single FP8 to F32
    # Use: cvt.f32.f16 on the low half of bf16_lo
    return _cvt_bf16x2_lo_to_f32(bf16_lo)
```

- [ ] **Step 5: Add `_cvt_bf16x2_lo_to_f32` — extract low BF16 from packed pair as Float32**

```python
@dsl_user_op
def _cvt_bf16x2_lo_to_f32(bf16x2, *, loc=None, ip=None) -> Float32:
    """Extract the low BF16 from a packed BF16x2 Uint32 and convert to Float32."""
    val_ir = bf16x2.ir_value(loc=loc, ip=ip)
    result_ir = _llvm_dialect.inline_asm(
        T.f32(),
        [T.i32()],
        "{ .reg .b16 %lo;\n"
        "  mov.b16 %lo, {$1:lo};\n"  # extract low 16 bits
        "  cvt.f32.bf16 $0, %lo; }",
        "=f,r",
        asm_dialect=0,
        operands_=[val_ir],
        loc=loc,
        ip=ip,
    )
    return Float32(result_ir)
```

- [ ] **Step 6: Add `_atomic_add_f32` — global memory atomicAdd**

```python
@dsl_user_op
def _atomic_add_f32(addr, val, *, loc=None, ip=None) -> Uint32:
    """atomicAdd a Float32 value to global memory. Returns old value (discarded)."""
    addr_ir = addr.ir_value(loc=loc, ip=ip)
    val_ir = val.ir_value(loc=loc, ip=ip)
    result_ir = _llvm_dialect.inline_asm(
        T.i32(),
        [T.i64(), T.f32()],
        "atom.global.add.f32 $0, [$1], $2;",
        "=r,l,f",
        asm_dialect=0,
        operands_=[addr_ir, val_ir],
        loc=loc,
        ip=ip,
    )
    return Uint32(result_ir)
```

- [ ] **Step 7: Commit**

```bash
git add vllm/v1/attention/backends/cute_paged/kernel.py
git commit -m "feat(kernel): add FP4 E2M1 dequant + atomicAdd PTX helpers for W_O fusion"
```

---

### Task 2: Add Phase B W_O GEMV Epilogue to DecodeKernel._kernel()

**Files:**
- Modify: `vllm/v1/attention/backends/cute_paged/kernel.py:700-770` (DecodeKernel class `__init__`)
- Modify: `vllm/v1/attention/backends/cute_paged/kernel.py:1234` (after `# end _md loop` in `_kernel`)

- [ ] **Step 1: Add W_O config constants to DecodeKernel.__init__()**

Add after the existing SMEM size calculations (around line 748):

```python
# --- Phase B: W_O GEMV config ---
# hidden_dim and group_size are runtime values, but we need
# compile-time constants for the per-thread output count.
# For Qwen3.5-27B: hidden_dim=3584, 128 threads → 28 outputs/thread.
# This is parameterized at compile time via wo_hidden_dim.
self.wo_hidden_dim = 3584      # Qwen3.5-27B hidden_size
self.wo_outputs_per_thread = self.wo_hidden_dim // self.num_threads  # 28
self.wo_k_tile = 16            # FP4 dequant group size
```

- [ ] **Step 2: Add `_kernel` parameter for W_O fusion**

Modify the `_kernel` method signature. The current signature (around line 775) is:

```python
@cute.kernel
def _kernel(self, query, k_ptr, v_ptr, page_table, seq_lens,
             output, scale, k_scale, v_scale,
             num_q_heads, num_kv_heads, kv_page_stride,
             grid_x: Int32, grid_y: Int32, grid_z: Int32):
```

Add new parameters:

```python
@cute.kernel
def _kernel(self, query, k_ptr, v_ptr, page_table, seq_lens,
             output, scale, k_scale, v_scale,
             num_q_heads, num_kv_heads, kv_page_stride,
             # Phase B: W_O GEMV params (all Int64(0) when fusion disabled)
             wo_weight_ptr, wo_scale_ptr, wo_output_ptr,
             wo_global_scale, wo_num_k_tiles,
             wo_weight_row_stride, wo_k_offset,
             wo_fused,
             grid_x: Int32, grid_y: Int32, grid_z: Int32):
```

- [ ] **Step 3: Add Phase B GEMV code after the _md loop**

Insert after `# end _md loop` (line 1234) and before `def __call__`:

```python
            # end _md loop

            # === Phase B: Fused W_O GEMV (when wo_fused != 0) ===
            if wo_fused != Int32(0):
                cute.arch.sync_threads()  # ensure Phase A output is coherent

                # Phase A wrote attention output to SMEM in scattered locations
                # via sync_o + cross-warp reduction → global output.
                # Phase B reads from the GLOBAL output (not SMEM) because the
                # cross-warp reduction already wrote the final BF16 result there.
                #
                # Thread tiling: each thread owns wo_outputs_per_thread (28)
                # consecutive rows of the output dimension.
                # tid 0 → rows [0..27], tid 1 → rows [28..55], etc.

                hd_wo = Int32(self.wo_hidden_dim)
                n_per_thr = Int32(self.wo_outputs_per_thread)
                k_tile = Int32(self.wo_k_tile)
                n_iters = Int32(self.head_dim * group_size // self.wo_k_tile)

                # Base output row for this thread
                my_row_base = tid * n_per_thr

                # Attention output base: the output buffer from Phase A
                # Shape: [num_seqs, num_q_heads, head_dim], flattened
                # This CTA's attention output starts at:
                attn_base_idx = seq_idx * num_q_heads * hd + q_head_start * hd

                # W_O weight base for this CTA's column slice
                # wo_k_offset = kv_head_idx * group_size * head_dim / 2 (byte offset per row)
                # wo_weight_row_stride = K/2 bytes per row

                # 28 FP32 accumulators per thread — register-resident
                # CuTe DSL doesn't support arrays, so we use the _md serialization
                # trick: process output rows in tiles of N_PER_ITER at a time.
                # With 28 outputs and 8 scalar regs, we do ceil(28/8)=4 output tiles.
                # Each output tile: 7 rows (last tile: 7 rows).
                # Actually simpler: just do 28 iterations of 1 output each.
                # But that's slow. Let's use 4 tiles of 7 outputs.
                #
                # Revised approach: serialize over output rows in groups.
                # Process 7 output rows at a time (4 groups), 96 inner K iters each.
                # 7 FP32 accumulators = 7 registers. Very manageable.

                for _out_group in cutlass.range_constexpr(4):
                    out_group_base = my_row_base + Int32(_out_group * 7)

                    # 7 FP32 accumulators
                    acc0 = Float32(0.0)
                    acc1 = Float32(0.0)
                    acc2 = Float32(0.0)
                    acc3 = Float32(0.0)
                    acc4 = Float32(0.0)
                    acc5 = Float32(0.0)
                    acc6 = Float32(0.0)

                    # Inner loop over K dimension (96 iterations for 1536 elements)
                    for _ki in cutlass.range_constexpr(
                        self.head_dim * 6 // self.wo_k_tile  # 256*6/16 = 96
                    ):
                        ki = Int32(_ki)

                        # Load attention input value from global memory
                        # (Phase A already wrote final BF16 to output buffer)
                        # attn_input[ki*16 .. ki*16+15] — but we broadcast
                        # one value at a time to all threads. Actually each
                        # inner iteration covers 16 K elements, and the GEMV
                        # inner product needs all 16. We sub-iterate.
                        for _ksub in cutlass.range_constexpr(self.wo_k_tile):
                            k_idx = ki * k_tile + Int32(_ksub)
                            # Load attention output element (BF16 → F32)
                            attn_elem_idx = attn_base_idx + k_idx
                            attn_val = Float32(output[attn_elem_idx])

                            # Load 7 FP4 weight values for this thread's output rows
                            # Weight layout: [N, K/2] row-major uint8
                            # Element (n, k): byte at n * row_stride + k/2
                            # Low nibble if k even, high nibble if k odd
                            abs_k = wo_k_offset * Int32(2) + k_idx  # absolute K index
                            k_byte_off = abs_k >> Int32(1)  # k/2
                            k_is_odd = abs_k & Int32(1)

                            # Load weight for each of 7 output rows
                            # Row n = out_group_base + i
                            for _oi in cutlass.range_constexpr(7):
                                out_row = out_group_base + Int32(_oi)
                                if out_row < hd_wo:
                                    w_byte_addr = wo_weight_ptr + Int64(
                                        out_row * wo_weight_row_stride + k_byte_off)
                                    # Load aligned dword containing our byte
                                    aligned = w_byte_addr & Int64(0xFFFFFFFFFFFFFFFC)
                                    raw_word = _ld_global_b32(aligned)
                                    byte_pos_w = Int32(w_byte_addr & Int64(3))
                                    the_byte = _extract_byte_from_b32(raw_word, byte_pos_w)
                                    # Extract correct nibble
                                    nibble = the_byte & Int32(0x0F) if k_is_odd == Int32(0) \
                                        else (the_byte >> Int32(4)) & Int32(0x0F)
                                    w_f32 = _fp4_nibble_to_f32(nibble)

                                    # Load scale factor for this (out_row, k_group)
                                    k_group = (wo_k_offset * Int32(2) + ki * k_tile + Int32(_ksub)) >> Int32(4)
                                    # Only load scale once per group of 16
                                    # (k_group changes every 16 k elements)
                                    sf_val = _ld_swizzled_scale(
                                        wo_scale_ptr, out_row, k_group, wo_num_k_tiles)

                                    # Dequant and FMA
                                    w_dequant = w_f32 * sf_val * wo_global_scale

                                    # Accumulate
                                    if _oi == 0:
                                        acc0 = acc0 + w_dequant * attn_val
                                    elif _oi == 1:
                                        acc1 = acc1 + w_dequant * attn_val
                                    elif _oi == 2:
                                        acc2 = acc2 + w_dequant * attn_val
                                    elif _oi == 3:
                                        acc3 = acc3 + w_dequant * attn_val
                                    elif _oi == 4:
                                        acc4 = acc4 + w_dequant * attn_val
                                    elif _oi == 5:
                                        acc5 = acc5 + w_dequant * attn_val
                                    elif _oi == 6:
                                        acc6 = acc6 + w_dequant * attn_val

                    # atomicAdd accumulators to global output buffer
                    # wo_output_ptr: base pointer to [num_tokens, hidden_dim] FP32 buffer
                    wo_out_base = wo_output_ptr + Int64(seq_idx * hd_wo * Int32(4))

                    for _oi in cutlass.range_constexpr(7):
                        out_row = out_group_base + Int32(_oi)
                        if out_row < hd_wo:
                            acc_val = Float32(0.0)
                            if _oi == 0:
                                acc_val = acc0
                            elif _oi == 1:
                                acc_val = acc1
                            elif _oi == 2:
                                acc_val = acc2
                            elif _oi == 3:
                                acc_val = acc3
                            elif _oi == 4:
                                acc_val = acc4
                            elif _oi == 5:
                                acc_val = acc5
                            elif _oi == 6:
                                acc_val = acc6
                            _atomic_add_f32(
                                wo_out_base + Int64(out_row * Int32(4)),
                                acc_val)
```

**Important CuTe DSL constraints applied:**
- No arrays — all accumulators are explicit scalar variables
- `range_constexpr` for compile-time-known loop bounds
- `if/elif` chains instead of indexing for accumulator selection
- Raw pointer arithmetic via Int64 for all global memory access
- `_extract_byte_from_b32` for sub-dword loads (existing helper)

**Note:** The inner loop structure above iterates per-element (16 sub-iterations per K tile × 96 K tiles = 1536 total). This is straightforward but generates a lot of unrolled code. If compile time is excessive, the 16-element sub-loop can be restructured to load 2 bytes (4 FP4 values) at once with `_ld_global_b32`. This is an optimization for a later pass — correctness first.

- [ ] **Step 3a: Handle the `_jit_launch` wrapper**

Modify `_jit_launch` to pass the new parameters:

```python
@cute.jit
def _jit_launch(self, query, k_ptr, v_ptr, page_table, seq_lens,
                output, scale, k_scale, v_scale,
                num_q_heads, num_kv_heads, kv_page_stride,
                wo_weight_ptr, wo_scale_ptr, wo_output_ptr,
                wo_global_scale, wo_num_k_tiles,
                wo_weight_row_stride, wo_k_offset,
                wo_fused,
                grid_x: Int32, grid_y: Int32, grid_z: Int32):
    """JIT host wrapper: compiles kernel launch into MLIR."""
    self._kernel(
        query, k_ptr, v_ptr, page_table, seq_lens,
        output, scale, k_scale, v_scale,
        num_q_heads, num_kv_heads, kv_page_stride,
        wo_weight_ptr, wo_scale_ptr, wo_output_ptr,
        wo_global_scale, wo_num_k_tiles,
        wo_weight_row_stride, wo_k_offset,
        wo_fused,
    ).launch(
        grid=[grid_x, grid_y, grid_z],
        block=[self.num_threads, 1, 1],
        smem=self.smem_bytes,
    )
```

- [ ] **Step 4: Commit**

```bash
git add vllm/v1/attention/backends/cute_paged/kernel.py
git commit -m "feat(kernel): Phase B W_O GEMV epilogue in DecodeKernel — NVFP4 dequant, register accum, atomicAdd"
```

---

### Task 3: Wire W_O Params into DecodeKernel.__call__() and Compilation

**Files:**
- Modify: `vllm/v1/attention/backends/cute_paged/kernel.py:1236-1301` (DecodeKernel `__call__`)

- [ ] **Step 1: Update `__call__` to accept and forward W_O parameters**

Replace the existing `__call__` method:

```python
def __call__(self, **kwargs):
    """Python-level wrapper: compute grid/block and launch."""
    query = kwargs["query"]
    kv_cache = kwargs["kv_cache"]
    page_table = kwargs["page_table"]
    seq_lens = kwargs["seq_lens"]
    scale = kwargs["scale"]
    k_scale = kwargs["k_scale"]
    v_scale = kwargs["v_scale"]

    # Phase B: W_O fusion params (optional)
    wo_weight = kwargs.get("wo_weight", None)     # [N, K/2] uint8
    wo_scales = kwargs.get("wo_scales", None)      # [N, K_sf] fp8_e4m3fn (swizzled)
    wo_global_scale = kwargs.get("wo_global_scale", None)  # scalar f32
    wo_output = kwargs.get("wo_output", None)      # [num_seqs, N] f32 (pre-zeroed)

    wo_fused = wo_weight is not None

    num_q_heads = query.shape[1]
    num_kv_heads = kv_cache.shape[3]
    group_size = num_q_heads // num_kv_heads
    num_seqs = len(seq_lens)

    # Unified base pointer + K/V byte offsets
    kv_base = Int64(kv_cache.data_ptr())
    kv_slot_stride = Int64(
        kv_cache.stride(1) * kv_cache.element_size()
    )
    k_ptr = kv_base
    v_ptr = kv_base + kv_slot_stride
    kv_page_stride = Int32(
        kv_cache.stride(0) * kv_cache.element_size()
    )

    q_flat = query.contiguous().view(-1)

    num_q_tiles = max(
        (group_size + self.cta_q - 1) // self.cta_q, 1,
    )
    grid = (num_q_tiles, num_kv_heads, num_seqs)

    output = torch.empty_like(query)
    out_flat = output.view(-1)

    # W_O pointer args (Int64(0) / Int32(0) / 0.0 when disabled)
    if wo_fused:
        wo_weight_ptr = Int64(wo_weight.data_ptr())
        wo_scale_ptr = Int64(wo_scales.data_ptr())
        wo_output_ptr = Int64(wo_output.data_ptr())
        wo_gs = float(wo_global_scale.item())
        # K = group_size * head_dim (input dim for this model's o_proj)
        wo_K = num_q_heads * self.head_dim
        wo_nkt = Int32((wo_K // 16 + 3) // 4)  # numKTiles for swizzle
        wo_row_stride = Int32(wo_weight.shape[1])  # K/2 bytes per row
        # wo_k_offset: column byte offset per row for this CTA's KV head group
        # Passed as 0 — kernel computes per-CTA offset from kv_head_idx
        wo_k_off = Int32(0)  # kernel uses kv_head_idx * group_size * head_dim / 2
        wo_fused_flag = Int32(1)
    else:
        wo_weight_ptr = Int64(0)
        wo_scale_ptr = Int64(0)
        wo_output_ptr = Int64(0)
        wo_gs = 0.0
        wo_nkt = Int32(0)
        wo_row_stride = Int32(0)
        wo_k_off = Int32(0)
        wo_fused_flag = Int32(0)

    all_args = (
        q_flat, k_ptr, v_ptr, page_table, seq_lens,
        out_flat,
        float(scale), float(k_scale), float(v_scale),
        Int32(num_q_heads), Int32(num_kv_heads),
        kv_page_stride,
        wo_weight_ptr, wo_scale_ptr, wo_output_ptr,
        wo_gs, wo_nkt,
        wo_row_stride, wo_k_off,
        wo_fused_flag,
        Int32(grid[0]), Int32(grid[1]), Int32(grid[2]),
    )

    if self._compiled is None:
        logger.info("Compiling CuTe decode kernel (first call)...")
        self._compiled = cute.compile(self._jit_launch, *all_args)

    self._compiled(*all_args)
    return output
```

- [ ] **Step 2: Update `paged_attention_forward` to forward W_O kwargs**

Modify the public entry point (around line 1437) to accept and pass W_O params:

```python
def paged_attention_forward(
    query: torch.Tensor,
    kv_cache: torch.Tensor,
    page_table: torch.Tensor,
    seq_lens: torch.Tensor,
    scale: float,
    k_scale: float = 1.0,
    v_scale: float = 1.0,
    page_size: int = 64,
    query_start_loc: torch.Tensor | None = None,
    # Phase B: W_O fusion (optional)
    wo_weight: torch.Tensor | None = None,
    wo_scales: torch.Tensor | None = None,
    wo_global_scale: torch.Tensor | None = None,
    wo_output: torch.Tensor | None = None,
) -> torch.Tensor:
    # ... existing fallback/prefill checks unchanged ...

    config = DECODE_CONFIG
    kernel = _get_compiled_kernel(config)

    cute_out = kernel(
        query=query,
        kv_cache=kv_cache,
        page_table=page_table,
        seq_lens=seq_lens,
        scale=scale,
        k_scale=k_scale,
        v_scale=v_scale,
        page_size=page_size,
        query_start_loc=query_start_loc,
        wo_weight=wo_weight,
        wo_scales=wo_scales,
        wo_global_scale=wo_global_scale,
        wo_output=wo_output,
    )

    return cute_out
```

- [ ] **Step 3: Fix kernel's Phase B to compute per-CTA wo_k_offset from kv_head_idx**

In the Phase B code (Task 2 Step 3), replace the use of `wo_k_offset` with runtime computation:

```python
# Per-CTA column offset: this CTA handles kv_head_idx's group
# Column range in the K dimension: [kv_head_idx * group_size * head_dim ...]
# Byte offset per row: kv_head_idx * group_size * head_dim / 2
wo_col_byte_offset = kv_head_idx * group_size * hd // Int32(2)
```

Use `wo_col_byte_offset` instead of `wo_k_offset` throughout Phase B.

- [ ] **Step 4: Commit**

```bash
git add vllm/v1/attention/backends/cute_paged/kernel.py
git commit -m "feat(kernel): wire W_O params into DecodeKernel.__call__ and paged_attention_forward"
```

---

### Task 4: Backend Integration (_backend.py)

**Files:**
- Modify: `vllm/v1/attention/backends/cute_paged/_backend.py:198-245`

- [ ] **Step 1: Add W_O params to CutePagedAttentionImpl.forward()**

```python
def forward(
    self,
    layer: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    kv_cache: torch.Tensor,
    attn_metadata: CutePagedMetadata,
    output: torch.Tensor | None = None,
    output_scale: torch.Tensor | None = None,
    output_block_scale: torch.Tensor | None = None,
    # Phase B: W_O fusion (optional, set by model layer)
    wo_weight: torch.Tensor | None = None,
    wo_scales: torch.Tensor | None = None,
    wo_global_scale: torch.Tensor | None = None,
    wo_output: torch.Tensor | None = None,
) -> torch.Tensor:
    assert output is not None, "Output tensor must be provided"

    if attn_metadata is None:
        return output.fill_(0)

    k_scale = getattr(layer, "_k_scale_float", 1.0)
    v_scale = getattr(layer, "_v_scale_float", 1.0)

    from vllm.v1.attention.backends.cute_paged.kernel import (
        paged_attention_forward,
    )

    num_actual_tokens = attn_metadata.num_actual_tokens

    result = paged_attention_forward(
        query=query[:num_actual_tokens],
        kv_cache=kv_cache,
        page_table=attn_metadata.block_table,
        seq_lens=attn_metadata.seq_lens,
        scale=self.scale,
        k_scale=k_scale,
        v_scale=v_scale,
        page_size=64,
        query_start_loc=attn_metadata.query_start_loc,
        wo_weight=wo_weight,
        wo_scales=wo_scales,
        wo_global_scale=wo_global_scale,
        wo_output=wo_output,
    )

    output[:num_actual_tokens].copy_(result)
    return output
```

- [ ] **Step 2: Commit**

```bash
git add vllm/v1/attention/backends/cute_paged/_backend.py
git commit -m "feat(backend): pass W_O fusion params through CutePagedAttentionImpl.forward"
```

---

### Task 5: Model Integration (qwen2.py)

**Files:**
- Modify: `vllm/model_executor/models/qwen2.py:209-236` (Qwen2Attention.forward)

This is the most delicate integration point. The attention layer needs to:
1. Detect that the CuTe paged backend is active
2. Extract o_proj weight tensors
3. Allocate and zero a W_O output buffer
4. Pass weights to the attention call
5. Skip the separate o_proj GEMM

- [ ] **Step 1: Add a `_wo_fusion_enabled` flag to Qwen2Attention.__init__()**

Add after existing init (around line 207):

```python
# W_O fusion: detect CuTe paged backend at runtime
self._wo_fusion_ready = False
```

- [ ] **Step 2: Add a `_prepare_wo_fusion` method**

```python
def _prepare_wo_fusion(self) -> bool:
    """Check if W_O fusion is available and cache weight pointers.

    Called once on first forward pass. Returns True if fusion is active.
    """
    # Check if the attention backend is CuTe paged
    attn_impl = getattr(self.attn, 'impl', None)
    if attn_impl is None:
        return False

    from vllm.v1.attention.backends.cute_paged._backend import (
        CutePagedAttentionImpl,
    )
    if not isinstance(attn_impl, CutePagedAttentionImpl):
        return False

    # Check if o_proj has NVFP4 weights
    if not hasattr(self.o_proj, 'weight') or self.o_proj.weight.dtype != torch.uint8:
        return False

    # Cache weight tensors for the fused path
    self._wo_weight = self.o_proj.weight          # [N, K/2] uint8
    self._wo_scales = self.o_proj.weight_scale     # [N, K_sf] fp8_e4m3fn swizzled
    self._wo_global_scale = getattr(
        self.o_proj, 'weight_global_scale', None
    )
    if self._wo_global_scale is None:
        return False

    return True
```

- [ ] **Step 3: Modify Qwen2Attention.forward() to use fused path**

```python
def forward(
    self,
    positions: torch.Tensor,
    hidden_states: torch.Tensor,
) -> torch.Tensor:
    qkv, _ = self.qkv_proj(hidden_states)
    q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

    if self.qk_norm:
        total_tokens = q.shape[0]
        q = q.view(total_tokens, self.num_heads, self.head_dim)
        k = k.view(total_tokens, self.num_kv_heads, self.head_dim)
        q = self.q_norm(q)
        k = self.k_norm(k)
        q = q.view(total_tokens, self.q_size)
        k = k.view(total_tokens, self.kv_size)

    q, k = self.rotary_emb(positions, q, k)

    # Check W_O fusion readiness (once)
    if not self._wo_fusion_ready:
        self._wo_fusion_ready = self._prepare_wo_fusion()

    if self._wo_fusion_ready:
        # Fused path: attention + W_O in one kernel launch
        num_tokens = q.shape[0]
        wo_output = torch.zeros(
            num_tokens, self._wo_weight.shape[0],
            dtype=torch.float32, device=q.device,
        )
        # The attn call will pass W_O params through to the backend
        # We need to inject them via a side-channel since the Attention
        # layer's forward() signature is fixed by vLLM's framework.
        # Store on the layer for the backend to pick up.
        self.attn._wo_weight = self._wo_weight
        self.attn._wo_scales = self._wo_scales
        self.attn._wo_global_scale = self._wo_global_scale
        self.attn._wo_output = wo_output

        attn_output = self.attn(q, k, v)

        # Clean up side-channel
        self.attn._wo_weight = None
        self.attn._wo_scales = None
        self.attn._wo_global_scale = None
        self.attn._wo_output = None

        # wo_output now has the fused result (FP32), convert to model dtype
        output = wo_output.to(hidden_states.dtype)
        return output
    else:
        # Unfused path (unchanged)
        attn_output = self.attn(q, k, v)
        output, _ = self.o_proj(attn_output)
        return output
```

**Important note:** The side-channel approach (`self.attn._wo_*`) is a pragmatic workaround because vLLM's `Attention` layer has a fixed `forward()` signature that goes through `AttentionImpl.forward()`. The backend picks these up from the `layer` parameter. This is admittedly not clean — a proper integration would modify vLLM's Attention abstraction, but that's too invasive for a fun project. Document this as tech debt.

- [ ] **Step 4: Update the backend to read from the side-channel**

Modify `CutePagedAttentionImpl.forward()` in `_backend.py` to check for the side-channel:

```python
# In forward(), before calling paged_attention_forward:
wo_weight = getattr(layer, '_wo_weight', None)
wo_scales = getattr(layer, '_wo_scales', None)
wo_global_scale = getattr(layer, '_wo_global_scale', None)
wo_output = getattr(layer, '_wo_output', None)

result = paged_attention_forward(
    ...,
    wo_weight=wo_weight,
    wo_scales=wo_scales,
    wo_global_scale=wo_global_scale,
    wo_output=wo_output,
)
```

Wait — this revises Task 4. The `layer` parameter in `forward()` is the `Attention` module, which holds the side-channel attributes. Update the Task 4 code to use `getattr(layer, ...)` instead of explicit parameters.

- [ ] **Step 5: Commit**

```bash
git add vllm/model_executor/models/qwen2.py vllm/v1/attention/backends/cute_paged/_backend.py
git commit -m "feat(model): Qwen2Attention W_O fusion opt-in with side-channel weight passing"
```

---

### Task 6: Standalone Test — Fused W_O GEMV

**Files:**
- Modify: `tests/nvllm/attention/test_cute_kernel_standalone.py`

- [ ] **Step 1: Add `test_wo_fusion` function**

Add after the existing `test_cute_kernel` function:

```python
def test_wo_fusion():
    """Test fused attention + W_O GEMV vs unfused (attention then matmul)."""
    from vllm.v1.attention.backends.cute_paged.kernel import (
        paged_attention_forward, _CUTE_AVAILABLE,
    )

    if not _CUTE_AVAILABLE:
        print("CUTLASS not available, skipping")
        return

    # Config matching Qwen3.5-27B
    num_q_heads = 24
    num_kv_heads = 4
    head_dim = 256
    hidden_dim = 3584
    page_size = 64
    scale = 1.0 / (head_dim ** 0.5)
    group_size = num_q_heads // num_kv_heads  # 6

    torch.manual_seed(42)
    device = "cuda"

    # 1 sequence, 6 tokens
    num_seqs = 1
    seq_lens = torch.tensor([6], dtype=torch.int32, device=device)
    query = torch.randn(1, num_q_heads, head_dim, dtype=torch.bfloat16, device=device)

    # KV cache
    num_pages = 2
    kv_cache = torch.zeros(num_pages, 2, page_size, num_kv_heads, head_dim,
                           dtype=torch.uint8, device=device)
    k_float = torch.randn(num_pages, page_size, num_kv_heads, head_dim, device=device).clamp(-10, 10)
    kv_cache[:, 0] = k_float.to(torch.float8_e4m3fn).view(torch.uint8)
    v_float = torch.randn(num_pages, page_size, num_kv_heads, head_dim, device=device).clamp(-10, 10)
    kv_cache[:, 1] = v_float.to(torch.float8_e4m3fn).view(torch.uint8)

    page_table = torch.zeros(1, 2, dtype=torch.int32, device=device)

    # --- Create fake NVFP4 W_O weights ---
    # For testing, create small random weights in FP4 format
    K = num_q_heads * head_dim  # 6144
    N = hidden_dim              # 3584

    # Generate random FP4 values (nibbles 0-15)
    torch.manual_seed(123)
    wo_nibbles = torch.randint(0, 16, (N, K), dtype=torch.uint8, device=device)

    # Pack into uint8 (2 nibbles per byte)
    wo_weight = torch.zeros(N, K // 2, dtype=torch.uint8, device=device)
    for k in range(0, K, 2):
        lo = wo_nibbles[:, k] & 0x0F
        hi = wo_nibbles[:, k + 1] & 0x0F
        wo_weight[:, k // 2] = lo | (hi << 4)

    # Create unswizzled scales (all 1.0 in FP8 E4M3)
    # 0x38 = 1.0 in FP8 E4M3
    K_sf = K // 16  # 384
    wo_scales_linear = torch.full((N, K_sf), 0x38, dtype=torch.uint8, device=device)
    wo_scales_linear = wo_scales_linear.view(torch.float8_e4m3fn)

    # Swizzle the scales
    from vllm.model_executor.layers.quantization.utils.nvfp4_utils import swizzle_blockscale
    wo_scales = swizzle_blockscale(wo_scales_linear)

    wo_global_scale = torch.tensor(1.0, dtype=torch.float32, device=device)

    # --- Reference: unfused attention → matmul ---
    # Step 1: attention
    attn_out = paged_attention_forward(
        query=query, kv_cache=kv_cache, page_table=page_table,
        seq_lens=seq_lens, scale=scale, k_scale=1.0, v_scale=1.0,
        page_size=page_size,
    )  # [1, 24, 256]

    # Step 2: dequant W_O and matmul
    from vllm.model_executor.layers.quantization.utils.nvfp4_emulation_utils import (
        break_fp4_bytes, dequantize_to_dtype,
    )
    wo_dequant = break_fp4_bytes(wo_weight.cpu(), torch.float32).cuda()
    # wo_dequant: [N, K] float32
    # Reference output: attn_flat @ W_O^T
    attn_flat = attn_out.view(1, -1).float()  # [1, 6144]
    ref_output = attn_flat @ wo_dequant.T  # [1, 3584]

    # --- Fused: attention + W_O in one kernel ---
    wo_output = torch.zeros(num_seqs, N, dtype=torch.float32, device=device)

    fused_attn_out = paged_attention_forward(
        query=query, kv_cache=kv_cache, page_table=page_table,
        seq_lens=seq_lens, scale=scale, k_scale=1.0, v_scale=1.0,
        page_size=page_size,
        wo_weight=wo_weight,
        wo_scales=wo_scales,
        wo_global_scale=wo_global_scale,
        wo_output=wo_output,
    )

    # --- Compare ---
    fused_f = wo_output.float()
    ref_f = ref_output.float()
    diff = (fused_f - ref_f).abs()

    print(f"\nW_O Fusion Test:")
    print(f"  fused[0,:8]  = {fused_f[0,:8].tolist()}")
    print(f"  ref[0,:8]    = {ref_f[0,:8].tolist()}")
    print(f"  max diff     = {diff.max().item():.6f}")
    print(f"  mean diff    = {diff.mean().item():.6f}")

    max_diff = diff.max().item()
    if max_diff < 0.05:
        print(f"\nPASS: max diff = {max_diff:.4f}")
    else:
        print(f"\nFAIL: max diff = {max_diff:.4f}")


if __name__ == "__main__":
    test_cute_kernel()
    print("\n" + "="*60 + "\n")
    test_wo_fusion()
```

- [ ] **Step 2: Commit**

```bash
git add tests/nvllm/attention/test_cute_kernel_standalone.py
git commit -m "test: standalone test for fused attention + W_O GEMV vs reference"
```

---

### Task 7: Docker Build and Standalone Test

**Files:** None (build + test execution)

- [ ] **Step 1: Build Docker image**

```bash
# In tmux session 'build':
tmux new-session -d -s build 'docker build -f docker/Dockerfile.gb10 -t nvllm:gb10 --no-cache . 2>&1 | tee /tmp/build.log'
```

Wait for build to complete. Check `/tmp/build.log` for errors.

- [ ] **Step 2: Run standalone test**

```bash
docker run --rm --gpus all --privileged \
  -v /home/natfii/docker/nvllm/tests:/app/nvllm/tests \
  nvllm:gb10 \
  python /app/nvllm/tests/nvllm/attention/test_cute_kernel_standalone.py
```

Expected: Both `test_cute_kernel` and `test_wo_fusion` PASS.

- [ ] **Step 3: Debug if needed**

If `test_wo_fusion` fails:
1. Check for NaN in output (sign of bad FP4 dequant)
2. Compare per-head outputs to isolate which KV head group is wrong
3. Verify scale factor swizzle offsets by printing a few values
4. Try with all-ones weights to test the pure GEMV accumulation path

---

### Task 8: GSM8K Sanity Gate

**Files:** None (serving + eval execution)

- [ ] **Step 1: Serve Qwen3.5-27B with fused W_O**

```bash
docker run -d --name nvllm-wo-test --gpus all --privileged \
  -p 8000:8000 \
  -v /home/natfii/.cache/huggingface:/root/.cache/huggingface \
  nvllm:gb10 \
  vllm serve natfii/Qwen3.5-27B-NVFP4-Opus-GB10 \
    --host 0.0.0.0 --port 8000 \
    --enforce-eager \
    --max-model-len 4096 \
    --max-num-seqs 4 \
    --gpu-memory-utilization 0.85 \
    --kv-cache-dtype fp8_e4m3
```

Wait for model to load, verify with:
```bash
curl -s http://localhost:8000/v1/models | python3 -m json.tool
```

- [ ] **Step 2: Quick smoke test**

```bash
curl -s http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"natfii/Qwen3.5-27B-NVFP4-Opus-GB10","prompt":"The capital of France is","max_tokens":32}' | python3 -m json.tool
```

Verify output is coherent (not gibberish). If gibberish → fusion bug, revert to unfused and debug.

- [ ] **Step 3: Run GSM8K eval**

```bash
# Using /v1/completions endpoint (not chat — avoids thinking mode)
# Run GSM8K subset for fast validation
python benchmarks/nvllm/run_gsm8k.py \
  --base-url http://localhost:8000/v1 \
  --model natfii/Qwen3.5-27B-NVFP4-Opus-GB10 \
  --num-samples 100 \
  --max-tokens 256
```

Expected: Accuracy within 5% of unfused baseline.

- [ ] **Step 4: Clean up**

```bash
docker stop nvllm-wo-test && docker rm nvllm-wo-test
```

- [ ] **Step 5: Commit all verified changes**

```bash
git add -A
git commit -m "feat(kernel): Unreal Kernel Phase 2 — fused W_O GEMV in attention decode epilogue

Two-phase kernel: Phase A (attention) outputs to global BF16, Phase B (W_O GEMV)
reads attention output and NVFP4 weights, accumulates in registers, atomicAdds
to caller-zeroed FP32 buffer. Eliminates o_proj GEMM kernel launch per decode layer.

NVFP4 E2M1 dequant with swizzled FP8 E4M3 block scale decode in PTX.
128 threads × 28 output elements × 96 K iterations. Standalone test passes.
GSM8K accuracy within tolerance of unfused baseline.

Spec: docs/superpowers/specs/2026-04-13-unreal-kernel-wo-fusion-design.md"
```

---

## Known Risks and Mitigations

1. **NVFP4 swizzled scale offset formula** — The swizzle pattern was reverse-engineered from `nvfp4_utils.py` and `nvfp4_utils.cuh`. If the formula is wrong, the standalone test will show systematic errors in the output. Mitigation: the test uses known scale values (all 1.0) to isolate weight dequant from scale issues.

2. **CuTe DSL compile time** — The Phase B code has deeply nested `range_constexpr` loops (4 output groups × 96 K iters × 16 K sub-iters × 7 output rows). This may generate enormous unrolled code. Mitigation: if compile time exceeds 10 minutes, reduce `range_constexpr` nesting by restructuring the inner loop to process multiple K elements per iteration via dword loads.

3. **CuTe DSL if/elif accumulator selection** — The `if _oi == 0: acc0 = ...` pattern is ugly but necessary because CuTe DSL has no array support. If the compiler rejects this pattern, fall back to SMEM-based accumulation (write all 7 values to SMEM, reduce in a second pass).

4. **atomicAdd FP32 contention** — 4 CTAs per sequence × 3584 output elements × 4 bytes = ~14 KB write footprint. With 4 CTAs, each address sees exactly 4 atomic operations. This should be negligible on SM120.

5. **Side-channel weight passing** — The `self.attn._wo_*` pattern in qwen2.py is a pragmatic hack. It works for single-threaded decode but could race under concurrent forward passes if vLLM batches layers differently. For this single-GPU fun project, it's fine. Document as tech debt for Phase 3.

6. **Phase B reads attention output from global memory, not SMEM** — The spec says "attention output stays in SMEM," but the cross-warp reduction in Phase A writes the final BF16 result to global memory (the `output` buffer). Phase B reads it back. This is one extra global read (8 KB per CTA) but avoids restructuring Phase A's output path. The primary HBM savings come from eliminating the separate o_proj GEMM's input read of the attention output — that read is replaced by Phase B's read from the same buffer that Phase A just wrote (likely still in L2).
