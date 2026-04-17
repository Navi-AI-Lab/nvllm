# Kernel Insights: b12x MLP Fusion (FC1 -> SiLU*Up -> FP4 Quantize -> FC2)

## Source

- Repository: https://github.com/lukealonso/b12x
- Pinned commit: `c469c6637f6251adefc282956f5392e559ea915d`
- License: Apache-2.0 (declared in [`pyproject.toml`](https://github.com/lukealonso/b12x/blob/c469c6637f6251adefc282956f5392e559ea915d/pyproject.toml))
- Date reviewed: 2026-04-17
- Purpose: reference for nvllm Phase D fused MLP decode kernel
  (gate+up GEMM -> SwiGLU -> FP4 re-quantize -> down GEMM) on SM120/SM121.

## Why This Source

SM120 (DGX Spark / GB10) plus NVFP4 is a niche combination. b12x is the only
public-and-working reference that actually ships a CuTe-DSL MLP fusion for our
exact hardware target. Everything else in this space either targets Blackwell
datacenter (tcgen05 / SM100) or requires the TRTLLM closed-source stack.
b12x's relevance:

- Already validates NVFP4 block-scale math on SM120 CuTe DSL.
- Provides the UE4M3 (a.k.a. FP8 E4M3) blockscale encoding pipeline at
  `SF_VEC_SIZE = 16` granularity that matches vLLM's NVFP4 checkpoint layout.
- Supplies the exact PTX intrinsic wrappers (cvt.rn.satfinite.e2m1x2.f32,
  cvt.rn.satfinite.e4m3x2.f32) we need for the Phase D write primitive.
- Ships a fused-MoE kernel (`b12x/moe/fused/static.py`) that already fuses
  GEMM1 -> SiLU*Up -> FP4 re-quant in the epilogue, which is algorithmically
  equivalent to our dense MLP fusion (minus the expert-routing dispatch).

The MoE orchestrator files (`static.py`, `dynamic.py`, `micro.py`) are the
*structural* reference for per-slice streaming; the primitives in
`b12x/cute/fp4.py` are the *arithmetic* reference we port into vLLM.

---

## 1. NVFP4 Block-Scale Encoding (UE4M3 Blockscale)

**File:** [`b12x/cute/fp4.py`](https://github.com/lukealonso/b12x/blob/c469c6637f6251adefc282956f5392e559ea915d/b12x/cute/fp4.py)

### 1a. Constants and block layout

[L40-L44](https://github.com/lukealonso/b12x/blob/c469c6637f6251adefc282956f5392e559ea915d/b12x/cute/fp4.py#L40-L44)

```python
FLOAT4_E2M1_MAX = 6.0  # Maximum value representable in FP4 E2M1
FLOAT8_E4M3_MAX = 448.0  # Maximum value representable in FP8 E4M3
SF_VEC_SIZE = 16  # Elements per scale factor block
COPY_BITS = 128  # 128-bit vectorized loads
_FP4_MAG_LUT = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
```

`SF_VEC_SIZE = 16` matches vLLM's NVFP4 checkpoint layout exactly — 16 FP4
elements per scale, one E4M3 byte per block. `_FP4_MAG_LUT` is the canonical
E2M1 magnitude table we use in the Phase 1 Python reference.

### 1b. E4M3 blockscale derivation (fast path)

[L2029-L2046](https://github.com/lukealonso/b12x/blob/c469c6637f6251adefc282956f5392e559ea915d/b12x/cute/fp4.py#L2029-L2046) — `quantize_block_fp4_fast`

```python
@cute.jit
def quantize_block_fp4_fast(
    values: cute.Tensor, max_abs: Float32, global_scale_val: Float32,
) -> Tuple[Uint64, cutlass.Uint8]:
    """Fast approximate FP4 block quantization using reciprocal/vector path."""
    scale_u32 = Uint32(0)
    scale_byte = cutlass.Uint8(0)
    packed64 = Uint64(0)
    if global_scale_val != Float32(0.0):
        fp4_max_rcp = rcp_approx_ftz(Float32(FLOAT4_E2M1_MAX))
        gs_recip = rcp_approx_ftz(global_scale_val)
        scale_float = gs_recip * (max_abs * fp4_max_rcp)
        scale_float = fmin_f32(scale_float, Float32(FLOAT8_E4M3_MAX))
        scale_u32 = cvt_f32_to_e4m3(scale_float)
        scale_byte = cutlass.Uint8(scale_u32 & Uint32(0xFF))
        inv_quantized_scale = fp8_e4m3_to_f32_and_rcp(scale_u32)
        if inv_quantized_scale != Float32(0.0):
            packed64 = quantize_and_pack_16_fast(values, inv_quantized_scale * gs_recip)
    return packed64, scale_byte
```

The math: `scale = max_abs / (FLOAT4_E2M1_MAX * global_scale)`, clamped to
E4M3 max (448.0), then rounded to E4M3. This is the canonical NVFP4
per-block-scale derivation. The "fast" variant replaces divisions with
`rcp.approx.ftz.f32` PTX; the "exact" path at
[L2007-L2025](https://github.com/lukealonso/b12x/blob/c469c6637f6251adefc282956f5392e559ea915d/b12x/cute/fp4.py#L2007-L2025)
divides directly.

### 1c. F32 -> E4M3 PTX intrinsic

[L1461-L1482](https://github.com/lukealonso/b12x/blob/c469c6637f6251adefc282956f5392e559ea915d/b12x/cute/fp4.py#L1461-L1482) — `cvt_f32_to_e4m3`

Wraps `cvt.rn.satfinite.e4m3x2.f32` so one f32 goes in and one E4M3 byte
comes back in the low 8 bits of a u32. This is the primitive we reuse
verbatim in the Phase D write-primitive task.

### 1d. E4M3 -> F32 and E4M3 -> 1/F32 for round-trip

- [L1486-L1521](https://github.com/lukealonso/b12x/blob/c469c6637f6251adefc282956f5392e559ea915d/b12x/cute/fp4.py#L1486-L1521) — `fp8_e4m3_to_f32`
- [L1525-L1557](https://github.com/lukealonso/b12x/blob/c469c6637f6251adefc282956f5392e559ea915d/b12x/cute/fp4.py#L1525-L1557) — `fp8_e4m3_to_f32_and_rcp`

Used to recover the exact quantized scale value (so the caller can redo the
multiply against the original f32 before packing nibbles; this is the
"double-round" correctness trick that prevents scale/value drift).

---

## 2. FP4 E2M1 Packing

### 2a. Per-value classification (exact path)

[L1960-L1992](https://github.com/lukealonso/b12x/blob/c469c6637f6251adefc282956f5392e559ea915d/b12x/cute/fp4.py#L1960-L1992) — `quantize_and_pack_16`

```python
@cute.jit
def quantize_and_pack_16(y_f32: cute.Tensor, value_scale: Float32) -> Uint64:
    """Quantize 16 float32 values to FP4 and pack into uint64."""
    t025 = value_scale * Float32(0.25)
    t075 = value_scale * Float32(0.75)
    t125 = value_scale * Float32(1.25)
    t175 = value_scale * Float32(1.75)
    t250 = value_scale * Float32(2.5)
    t350 = value_scale * Float32(3.5)
    t500 = value_scale * Float32(5.0)
    packed = Uint64(0)
    for i in cutlass.range_constexpr(16):
        q = y_f32[i]
        mag = fabs_f32(q)
        nibble = Uint8(0)
        if mag > t025 and mag < t075:
            nibble = Uint8(1)
        elif mag >= t075 and mag <= t125:
            nibble = Uint8(2)
        # ...
        elif mag > t500:
            nibble = Uint8(7)
        if nibble != Uint8(0) and q < Float32(0.0):
            nibble = nibble | Uint8(0x8)
        packed = packed | (Uint64(nibble) << Uint64(i * 4))
    return packed
```

16 independent nibbles packed into one uint64 via left-shifts. The sign bit
is the MSB of each nibble (`| 0x8` for negative, except for `±0`).

### 2b. Per-value packing (fast path) via `cvt.e2m1x2.f32`

[L1995-L2004](https://github.com/lukealonso/b12x/blob/c469c6637f6251adefc282956f5392e559ea915d/b12x/cute/fp4.py#L1995-L2004) — `quantize_and_pack_16_fast`

```python
@cute.jit
def quantize_and_pack_16_fast(y_f32: cute.Tensor, inv_scale: Float32) -> Uint64:
    """Fast approximate FP4 quantize/pack for 16 float32 values."""
    q = cute.make_rmem_tensor((16,), Float32)
    for i in cutlass.range_constexpr(16):
        q[i] = y_f32[i] * inv_scale

    packed_lo = cvt_e2m1x8_f32(q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7])
    packed_hi = cvt_e2m1x8_f32(q[8], q[9], q[10], q[11], q[12], q[13], q[14], q[15])
    return (Uint64(packed_hi) << Uint64(32)) | Uint64(packed_lo)
```

Uses the Blackwell `cvt.rn.satfinite.e2m1x2.f32` hardware instruction
(2 floats -> 1 byte of 2 x E2M1) — eight invocations cover 16 f32 values
into one uint64. This is the primitive we port into the Phase 2 CuTe DSL
FP4 write primitive.

### 2c. cvt.e2m1x8.f32 wrapper

[L1653-L1696](https://github.com/lukealonso/b12x/blob/c469c6637f6251adefc282956f5392e559ea915d/b12x/cute/fp4.py#L1653-L1696) — `cvt_e2m1x8_f32`

```python
@dsl_user_op
def cvt_e2m1x8_f32(v0..v7) -> Uint32:
    """Convert eight float32 values to eight E2M1 (4-bit) values packed into uint32."""
    # PTX:
    # cvt.rn.satfinite.e2m1x2.f32 byte0, v1, v0;
    # cvt.rn.satfinite.e2m1x2.f32 byte1, v3, v2;
    # cvt.rn.satfinite.e2m1x2.f32 byte2, v5, v4;
    # cvt.rn.satfinite.e2m1x2.f32 byte3, v7, v6;
    # mov.b32 $0, {byte0, byte1, byte2, byte3};
```

Note the argument order: the PTX emits `$2, $1` (i.e. v1 then v0) per
instruction, so lane 0 ends up in the *low* nibble of byte 0. This matches
vLLM's on-disk NVFP4 nibble packing. We preserve this convention in Phase 2.

---

## 3. Per-Slice Streaming Pattern (FC1 -> SiLU*Up -> FP4 -> FC2)

**File:** [`b12x/moe/fused/static.py`](https://github.com/lukealonso/b12x/blob/c469c6637f6251adefc282956f5392e559ea915d/b12x/moe/fused/static.py)

### 3a. Fused epilogue in the GEMM1 kernel

[L780-L840](https://github.com/lukealonso/b12x/blob/c469c6637f6251adefc282956f5392e559ea915d/b12x/moe/fused/static.py#L780-L840)

b12x distributes the per-block FP4 quantize across every thread in the CTA
rather than a single "leader" thread. The inner loop is:

```python
# per-token, per-sf-block loop body:
values = cute.make_rmem_tensor((16,), Float32)
block_max = Float32(0.0)
for elem_idx in cutlass.range_constexpr(16):
    value = Float32(a_input[token_idx, block_start + elem_idx])
    values[elem_idx] = value
    block_max = fmax_f32(block_max, fabs_f32(value))
packed64, scale_byte = quantize_block_fp4_fast(values, block_max, gs_value)
st_global_u64(get_ptr_as_int64(packed_a_storage, output_offset), packed64)
# ... scale-byte swizzled store ...
```

Each SF-block (16 elements) is independent, so we get trivial inter-thread
parallelism. The pattern we borrow is **load-16 -> compute-max -> quantize ->
store-u64-packed -> store-scale-byte**, with no intermediate gmem round-trip
on the f32 activation.

### 3b. SiLU*Up fused with FP4 quantize

[L2058-L2088](https://github.com/lukealonso/b12x/blob/c469c6637f6251adefc282956f5392e559ea915d/b12x/cute/fp4.py#L2058-L2088) — `silu_mul_16`, `silu_mul_quantize_block_fp4`

```python
@cute.jit
def silu_mul_16(gate: cute.Tensor, up: cute.Tensor) -> cute.Tensor:
    out = cute.make_rmem_tensor((16,), Float32)
    for i in cutlass.range_constexpr(16):
        g = gate[i]
        sigmoid_g = cute.arch.rcp_approx(
            Float32(1.0) + cute.math.exp(-g, fastmath=False)
        )
        out[i] = g * sigmoid_g * up[i]
    return out

@cute.jit
def silu_mul_quantize_block_fp4(
    gate: cute.Tensor, up: cute.Tensor, global_scale_val: Float32,
) -> Tuple[Uint64, cutlass.Uint8]:
    activated = silu_mul_16(gate, up)
    block_max = max_abs_16(activated)
    return quantize_block_fp4(activated, block_max, global_scale_val)
```

This is the exact helper we replicate for the Phase D fusion: SiLU(gate) *
up, then block-max, then FP4 quantize — all register-resident. The 16-lane
vector width matches `SF_VEC_SIZE`, so one call emits one scale byte plus
one packed u64.

### 3c. Block-scale swizzle (for gmem layout compatibility)

- [L51-L56](https://github.com/lukealonso/b12x/blob/c469c6637f6251adefc282956f5392e559ea915d/b12x/cute/fp4.py#L51-L56) — `make_swizzle_indices`
- [L59-L77](https://github.com/lukealonso/b12x/blob/c469c6637f6251adefc282956f5392e559ea915d/b12x/cute/fp4.py#L59-L77) — `swizzle_block_scale`
- [L80-L86](https://github.com/lukealonso/b12x/blob/c469c6637f6251adefc282956f5392e559ea915d/b12x/cute/fp4.py#L80-L86) — `as_grouped_scale_view`

Scales are stored in a `128 x 4` swizzled layout (rows grouped by 128 /
cols by 4) to match NVIDIA's `nvfp4_block_scales_view` expected by the
downstream GEMM. We borrow this layout for Phase 5 (Python-side wiring)
so the FC2 input blockscale matches what vLLM's NVFP4 GEMM expects.

---

## 4. Supporting PTX Intrinsics We Reuse

All reusable for the Phase D write primitive. Every one has a pinned line
range below.

| Intrinsic | Purpose | Permalink |
|---|---|---|
| `cvt_f32_to_e4m3` | F32 -> UE4M3 blockscale byte | [L1461-L1482](https://github.com/lukealonso/b12x/blob/c469c6637f6251adefc282956f5392e559ea915d/b12x/cute/fp4.py#L1461-L1482) |
| `cvt_e2m1x8_f32` | 8 F32 -> 8 E2M1 packed into u32 | [L1653-L1696](https://github.com/lukealonso/b12x/blob/c469c6637f6251adefc282956f5392e559ea915d/b12x/cute/fp4.py#L1653-L1696) |
| `rcp_approx_ftz` | Fast reciprocal for scale division | [L700-L713](https://github.com/lukealonso/b12x/blob/c469c6637f6251adefc282956f5392e559ea915d/b12x/cute/fp4.py#L700-L713) |
| `fmax_f32` / `fmin_f32` / `fabs_f32` | PTX min/max/abs | [L716-L761](https://github.com/lukealonso/b12x/blob/c469c6637f6251adefc282956f5392e559ea915d/b12x/cute/fp4.py#L716-L761) |
| `st_global_u64` | 64-bit aligned FP4 block store | [L282-L296](https://github.com/lukealonso/b12x/blob/c469c6637f6251adefc282956f5392e559ea915d/b12x/cute/fp4.py#L282-L296) |
| `st_global_u8` | 8-bit scale byte store | [L328-L342](https://github.com/lukealonso/b12x/blob/c469c6637f6251adefc282956f5392e559ea915d/b12x/cute/fp4.py#L328-L342) |
| `atomic_add_global_i32` | Split-K arrival counter | [L602-L618](https://github.com/lukealonso/b12x/blob/c469c6637f6251adefc282956f5392e559ea915d/b12x/cute/fp4.py#L602-L618) |
| `scatter_add_bf16x2` | BF16 epilogue atomic (alt FC2 path) | [L643-L663](https://github.com/lukealonso/b12x/blob/c469c6637f6251adefc282956f5392e559ea915d/b12x/cute/fp4.py#L643-L663) |

---

## 5. Python Reference: Grouped NVFP4 Quantize

[L125-L155](https://github.com/lukealonso/b12x/blob/c469c6637f6251adefc282956f5392e559ea915d/b12x/cute/fp4.py#L125-L155) — `quantize_grouped_nvfp4_torch`

Pure-PyTorch reference that matches the CuTe kernel's tie-breaking exactly.
We port this (simplified for dense / no-grouping) as the Phase 1 Python
reference against which the Phase 3 CuTe kernel must match bit-for-bit.

[L158-L167](https://github.com/lukealonso/b12x/blob/c469c6637f6251adefc282956f5392e559ea915d/b12x/cute/fp4.py#L158-L167) — `silu_mul_quantize_grouped_nvfp4_torch`

```python
def silu_mul_quantize_grouped_nvfp4_torch(
    input_tensor: torch.Tensor,
    row_counts: torch.Tensor,
    global_scale: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    cols = input_tensor.shape[-1] // 2
    left = input_tensor[..., :cols].float()
    right = input_tensor[..., cols:].float()
    activated = (F.silu(left) * right).to(input_tensor.dtype).to(torch.float32)
    return quantize_grouped_nvfp4_torch(activated, row_counts, global_scale)
```

Input is shape `[..., 2*cols]` with gate/up interleaved in halves — this
matches vLLM's `merge_column_parallel_linear` layout for qkv/gate_up fused
proj. We port this structure for the Phase 1 reference.

---

## How the Adaptation Differs from b12x

- **b12x:** MoE routing with expert dispatch, grouped tokens, up to 4 experts
  per token. One-CTA-owns-all-K accumulation (it's the static-graph variant).
- **nvllm Phase D:** dense MLP (no experts, no routing). Split-K with global
  `atomicAdd` into a partial-sum buffer and a small reduce kernel, because
  SM120 has limited smem per SM and our decode batch is tiny (1-8 tokens).
- **b12x:** mixes `tcgen05` MMAs (SM100 Blackwell datacenter) with the CuTe
  DSL path on older arches. We strip the tcgen05 path entirely.
- **b12x:** supports MXFP4 (UE8M0) alongside NVFP4 (UE4M3). We only need
  NVFP4 for vLLM, so we drop the UE8M0 intrinsics (at
  [L1565-L1645](https://github.com/lukealonso/b12x/blob/c469c6637f6251adefc282956f5392e559ea915d/b12x/cute/fp4.py#L1565-L1645))
  from the port.
- **b12x:** exact-path FP4 quantize uses a cascade of f32 comparisons
  (L1960-L1992). We use only the fast path (`cvt.e2m1x8.f32`) because our
  target hardware supports it natively and we trust the round-trip.
- **b12x:** scalar FP32 accumulators match our approach (we do not use
  `tcgen05` MMA either).
- **b12x:** per-CTA arrival mbarrier for cluster reduce. We use a
  `st.global.s32 + atom.global.add.s32` pair for the Phase D split-K
  reduce (simpler, no cluster required).

---

## License Compatibility

- b12x: Apache-2.0 (`pyproject.toml` @
  [pinned commit](https://github.com/lukealonso/b12x/blob/c469c6637f6251adefc282956f5392e559ea915d/pyproject.toml)).
- nvllm (vLLM fork): Apache-2.0.
- Compatible. No copyleft concerns. Must preserve the FlashInfer team
  copyright header from `b12x/cute/fp4.py` when porting verbatim chunks.

---

## Verification Checklist (per AGENTS.md §3)

- [x] Pinned commit (`c469c6637f6251adefc282956f5392e559ea915d`) — not a branch.
- [x] Per-piece permalinks with line numbers.
- [x] License compatibility confirmed.
- [x] "How adapted" section present.
- [x] README acknowledgment updated (Phase D section).
- [x] Archive copies committed to `/tmp/mlp-fusion-research/` for
      deep-dive during implementation (this is a scratch dir, not a
      repo path; the canonical reference is this doc + the pinned
      permalinks).

---

## Phase-to-Task Map

| Phase | Uses borrowed piece | Source section |
|---|---|---|
| 1 (Python ref) | `quantize_grouped_nvfp4_torch`, `silu_mul_quantize_grouped_nvfp4_torch` | §5 |
| 2 (FP4 write primitive) | `cvt_e2m1x8_f32`, `cvt_f32_to_e4m3`, `quantize_and_pack_16_fast`, `quantize_block_fp4_fast` | §1b, §1c, §2b, §2c |
| 3 (CuTe kernel) | `silu_mul_16`, `silu_mul_quantize_block_fp4`, per-slice streaming loop | §3 |
| 4 (Backend integration) | Blockscale swizzle layout for FC2 | §3c |
| 5 (Python-side wiring) | `swizzle_block_scale`, `as_grouped_scale_view` | §3c |
