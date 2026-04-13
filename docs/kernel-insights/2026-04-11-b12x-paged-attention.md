# Kernel Insights: b12x Paged Attention

## Source
- Repository: https://github.com/lukealonso/b12x
- Pinned commit: `c469c6637f6251adefc282956f5392e559ea915d`
- License: Apache-2.0 (declared in `pyproject.toml`)
- Date reviewed: 2026-04-11

---

## 1. Kernel Traits and Tile Configs

**File:** [`b12x/attention/paged/traits.py`](https://github.com/lukealonso/b12x/blob/c469c6637f6251adefc282956f5392e559ea915d/b12x/attention/paged/traits.py)

### 1a. Trait Dataclass

[L40-L65](https://github.com/lukealonso/b12x/blob/c469c6637f6251adefc282956f5392e559ea915d/b12x/attention/paged/traits.py#L40-L65)

```python
@dataclass(frozen=True)
class PagedForwardTraits:
    cta_tile_q: int
    cta_tile_kv: int
    num_mma_q: int
    num_mma_kv: int
    num_mma_d_qk: int
    num_mma_d_vo: int
    num_warps_q: int
    num_warps_kv: int
    num_threads: int
    head_dim_qk: int
    head_dim_vo: int
    upcast_stride_q: int
    upcast_stride_k: int
    upcast_stride_v: int
    upcast_stride_o: int
    q_dtype: torch.dtype
    kv_dtype: torch.dtype
    o_dtype: torch.dtype
    q_smem_bytes: int
    shared_storage_bytes: int
    max_smem_per_sm: int
    num_ctas_per_sm: int
    max_smem_per_threadblock: int

    @property
    def uses_fp8_kv(self) -> bool:
        return self.kv_dtype == _FP8_KV_DTYPE
```

### 1b. Warp Count Helpers

[L26-L36](https://github.com/lukealonso/b12x/blob/c469c6637f6251adefc282956f5392e559ea915d/b12x/attention/paged/traits.py#L26-L36)

```python
def paged_get_num_warps_q(cta_tile_q: int) -> int:
    return 4 if cta_tile_q > 16 else 1

def paged_get_num_warps_kv(cta_tile_q: int) -> int:
    return 4 // paged_get_num_warps_q(cta_tile_q)

def paged_get_num_mma_q(cta_tile_q: int) -> int:
    return 2 if cta_tile_q > 64 else 1
```

### 1c. FP8 KV Special-Case Tile Selection (cta_tile_q=48)

[L104-L126](https://github.com/lukealonso/b12x/blob/c469c6637f6251adefc282956f5392e559ea915d/b12x/attention/paged/traits.py#L104-L126)

```python
    if kv_dtype == _FP8_KV_DTYPE and cta_tile_q == 48:
        device_props = torch.cuda.get_device_properties(torch.cuda.current_device() if device is None else device)
        max_smem_per_sm = int(device_props.shared_memory_per_multiprocessor)
        return PagedForwardTraits(
            cta_tile_q=48,
            cta_tile_kv=32,
            num_mma_q=1,
            num_mma_kv=2,
            num_mma_d_qk=head_dim_qk // 16,
            num_mma_d_vo=head_dim_vo // 16,
            num_warps_q=3,
            num_warps_kv=1,
            num_threads=96,
            head_dim_qk=head_dim_qk,
            head_dim_vo=head_dim_vo,
            upcast_stride_q=head_dim_qk // 8,
            upcast_stride_k=head_dim_qk // 16,
            upcast_stride_v=head_dim_vo // 16,
            upcast_stride_o=head_dim_vo // (16 // _dtype_num_bytes(o_dtype)),
            q_dtype=q_dtype,
            kv_dtype=kv_dtype,
            o_dtype=o_dtype,
            q_smem_bytes=48 * head_dim_qk * _dtype_num_bytes(q_dtype),
            shared_storage_bytes=49152,
            ...
```

### 1d. Generic Trait Selection (num_mma_kv from SMEM budget)

[L128-L165](https://github.com/lukealonso/b12x/blob/c469c6637f6251adefc282956f5392e559ea915d/b12x/attention/paged/traits.py#L128-L165)

```python
    num_mma_d_qk = head_dim_qk // 16
    num_mma_d_vo = head_dim_vo // 16
    num_warps_q = paged_get_num_warps_q(cta_tile_q)
    num_warps_kv = paged_get_num_warps_kv(cta_tile_q)
    num_mma_q = paged_get_num_mma_q(cta_tile_q)

    device_props = torch.cuda.get_device_properties(torch.cuda.current_device() if device is None else device)
    max_smem_per_sm = int(device_props.shared_memory_per_multiprocessor)

    q_bytes = _dtype_num_bytes(q_dtype)
    kv_bytes = _dtype_num_bytes(kv_dtype)
    o_bytes = _dtype_num_bytes(o_dtype)
    q_smem_bytes = cta_tile_q * head_dim_qk * q_bytes
    kv_bytes_per_mma = (head_dim_qk + head_dim_vo) * 16 * num_warps_kv * kv_bytes
    num_ctas_per_sm = 2 if max_smem_per_sm >= 2 * (q_smem_bytes + kv_bytes_per_mma) else 1
    max_smem_per_threadblock = max_smem_per_sm // num_ctas_per_sm
    max_num_mma_kv_reg = 8 // num_mma_q
    max_num_mma_kv_smem = max((max_smem_per_threadblock - q_smem_bytes) // kv_bytes_per_mma, 0)
    num_mma_kv = min(max_num_mma_kv_smem, max_num_mma_kv_reg)
```

### 1e. SMEM Budget Computation

[L167-L180](https://github.com/lukealonso/b12x/blob/c469c6637f6251adefc282956f5392e559ea915d/b12x/attention/paged/traits.py#L167-L180)

```python
    k_smem_bytes = cta_tile_kv * head_dim_qk * kv_bytes
    v_smem_bytes = cta_tile_kv * head_dim_vo * kv_bytes
    qkv_storage_bytes = q_smem_bytes + k_smem_bytes + v_smem_bytes
    cta_sync_o_bytes = 4 if num_warps_kv == 1 else num_warps_kv * cta_tile_q * head_dim_vo * 4
    cta_sync_md_bytes = 8 if num_warps_kv == 1 else num_warps_kv * cta_tile_q * 8
    cta_sync_storage_bytes = cta_sync_o_bytes + cta_sync_md_bytes
    smem_o_bytes = cta_tile_q * head_dim_vo * o_bytes
    shared_storage_bytes = _align_up(max(qkv_storage_bytes, cta_sync_storage_bytes, smem_o_bytes), 16)
```

### 1f. Validity Check (FlashInfer rules)

[L68-L80](https://github.com/lukealonso/b12x/blob/c469c6637f6251adefc282956f5392e559ea915d/b12x/attention/paged/traits.py#L68-L80)

```python
def _paged_is_invalid(
    *,
    num_mma_q: int,
    num_mma_kv: int,
    num_mma_d_vo: int,
    num_warps_q: int,
    kv_dtype: torch.dtype,
) -> bool:
    kv_is_fp8 = kv_dtype == _FP8_KV_DTYPE
    if num_mma_d_vo < 4:
        return True
    if num_mma_d_vo == 4 and num_mma_kv % 2 == 1:
        return True
    if num_mma_q * (8 * num_mma_d_vo + 8 * num_mma_kv) >= 256:
        return True
    if kv_is_fp8 and (num_mma_kv * 2) % num_warps_q != 0:
        return True
    return False
```

---

## 2. Decode Forward Kernel (forward_paged.py)

**File:** [`b12x/attention/paged/forward_paged.py`](https://github.com/lukealonso/b12x/blob/c469c6637f6251adefc282956f5392e559ea915d/b12x/attention/paged/forward_paged.py)

### 2a. Class Structure

[L2063-L2118](https://github.com/lukealonso/b12x/blob/c469c6637f6251adefc282956f5392e559ea915d/b12x/attention/paged/forward_paged.py#L2063-L2118)

```python
class PagedForwardKernel:
    DECODE_MXFP8_RUNTIME_CHUNK_THRESHOLD = 11 * 64

    def __init__(
        self,
        dtype_q: Type[cutlass.Numeric],
        dtype_kv: Type[cutlass.Numeric],
        dtype_kv_storage: Type[cutlass.Numeric],
        dtype_o: Type[cutlass.Numeric],
        *,
        traits: PagedForwardTraits,
        split_kv: bool,
        single_request_decode_graph: bool = False,
        single_qtile_decode_graph: bool = False,
        regularized_decode_graph: bool = False,
        mxfp8_turbo: bool = False,
        enable_mxfp8_pv: bool = False,
        decode_only: bool = False,
        decode_mxfp8_runtime_chunk_guard: bool = False,
    ):
        ...
        self.kv_is_fp8 = dtype_kv == cutlass.Float8E4M3FN
        self.vec_size = traits.head_dim_vo // 32
        self.total_warps = traits.num_warps_q * traits.num_warps_kv
        self.stage_tile_rows = traits.cta_tile_kv
        ...
        self.num_stages = (
            1
            if traits.num_warps_kv > 1 or self.kv_is_fp8
            else (2 if q_stage_bytes + 2 * kv_stage_bytes <= traits.max_smem_per_threadblock else 1)
        )
```

BF16 TMA-only decode constraint:

[L2108-L2128](https://github.com/lukealonso/b12x/blob/c469c6637f6251adefc282956f5392e559ea915d/b12x/attention/paged/forward_paged.py#L2108-L2128)

```python
        base_use_paged_kv_tma_decode = (
            dtype_q == cutlass.BFloat16
            and dtype_o == cutlass.BFloat16
            and traits.head_dim_qk == 256
            and traits.head_dim_vo == 256
            and self.num_stages == 1
            and traits.num_warps_kv > 1
            and traits.num_warps_q == 1
            and self.stage_tile_rows == 64
            and traits.cta_tile_q == 16
            and traits.num_mma_q == 1
            and traits.num_mma_kv == 1
        )
        self.use_paged_kv_tma_exact_plane_bf16_layout = base_use_paged_kv_tma_decode
        self.use_paged_kv_tma = self.use_paged_kv_tma_exact_plane_bf16_layout
        if not self.use_paged_kv_tma:
            raise NotImplementedError(
                "PagedForwardKernel now only supports exact-plane paged K/V TMA decode; "
                "extend and legacy non-TMA ingress use dedicated specialized kernels."
            )
```

### 2b. MMA Instruction Selection

**BF16 QK MMA:** [`_literal_qk_mma_into_sfrag_plane_bf16`](https://github.com/lukealonso/b12x/blob/c469c6637f6251adefc282956f5392e559ea915d/b12x/attention/paged/forward_paged.py#L951-L1018)

```python
@cute.jit
def _literal_qk_mma_into_sfrag_plane_bf16(
    s_frag: cute.Tensor,
    q_base_addr: Int32,
    k_plane0_base_addr: Int32,
    k_plane1_base_addr: Int32,
    k_plane2_base_addr: Int32,
    k_plane3_base_addr: Int32,
    lane,
    warp_q_idx,
    warp_kv_idx,
    row_base,
    num_mma_q,
    num_mma_kv,
    num_mma_d_qk,
    upcast_stride_q,
    upcast_stride_plane,
):
    for mma_d in cutlass.range_constexpr(num_mma_d_qk):
        plane_idx = mma_d // 4
        mma_d_local = mma_d - plane_idx * 4
        a_regs = cute.make_rmem_tensor(
            cute.make_layout((num_mma_q, 4), stride=(4, 1)),
            Uint32,
        )
        for mma_q in cutlass.range_constexpr(num_mma_q):
            q_row = warp_q_idx * num_mma_q * 16 + mma_q * 16 + lane % 16
            q_col = mma_d * 2 + lane // 16
            q_offset = _permuted_offset_128b(q_row, q_col, upcast_stride_q)
            a0, a1, a2, a3 = ldmatrix_m8n8x4_b16(_smem_addr_from_b128_offset(q_base_addr, q_offset))
            ...

        for mma_kv in cutlass.range_constexpr(num_mma_kv):
            k_row = row_base + warp_kv_idx * num_mma_kv * 16 + mma_kv * 16 + 8 * (lane // 16) + lane % 8
            k_col = mma_d_local * 2 + (lane % 16) // 8
            k_offset = _permuted_offset_128b(k_row, k_col, upcast_stride_plane)
            b0, b1, b2, b3 = ldmatrix_m8n8x4_b16(_smem_addr_from_b128_offset(k_plane_base_addr, k_offset))

            for mma_q in cutlass.range_constexpr(num_mma_q):
                d0, d1, d2, d3, d4, d5, d6, d7 = bf16_mma_m16n16k16_f32(
                    s_frag[mma_q, mma_kv, 0], ... s_frag[mma_q, mma_kv, 7],
                    a_regs[mma_q, 0], ... a_regs[mma_q, 3],
                    b0, b1, b2, b3,
                )
```

**FP8 QK MMA:** [`_literal_qk_mma_into_sfrag_plane_fp8_raw`](https://github.com/lukealonso/b12x/blob/c469c6637f6251adefc282956f5392e559ea915d/b12x/attention/paged/forward_paged.py#L1021-L1130)

```python
@cute.jit
def _literal_qk_mma_into_sfrag_plane_fp8_raw(
    s_frag, q_base_addr, k_plane0_base_addr, k_plane1_base_addr,
    lane, warp_q_idx, warp_kv_idx, row_base,
    num_mma_q, num_mma_kv, num_mma_d_qk, upcast_stride_q, upcast_stride_plane,
):
    upcast_stride_full = upcast_stride_plane * Int32(2)
    ...
    for mma_d in cutlass.range_constexpr(num_mma_d_qk):
        ...
        for mma_kv in cutlass.range_constexpr(num_mma_kv):
            k_addr = _smem_addr_from_split_planes_128b(
                k_plane0_base_addr, k_plane1_base_addr,
                k_offset_cur, upcast_stride_full,
            )
            if const_expr(mma_d % 2 == 0):
                b_f8_0, b_f8_1 = ldmatrix_m8n8x4_left_half_b16(k_addr)
            else:
                b_f8_0, b_f8_1 = ldmatrix_m8n8x4_right_half_b16(k_addr)
            b_f8_0 = frag_layout_swizzle_16b_to_8b(b_f8_0)
            b_f8_1 = frag_layout_swizzle_16b_to_8b(b_f8_1)
            b0, b1 = fp8x4_e4m3_to_bfloat2x2(b_f8_0)
            b2, b3 = fp8x4_e4m3_to_bfloat2x2(b_f8_1)
            ...
            for mma_q in cutlass.range_constexpr(num_mma_q):
                d0, d1, d2, d3, d4, d5, d6, d7 = bf16_mma_m16n16k16_f32(
                    s_frag[...], a_regs[...], b0, b1, b2, b3,
                )
```

**MXFP8 QK MMA (turbo mode):** [`_literal_qk_mma_into_sfrag_mxfp8_raw`](https://github.com/lukealonso/b12x/blob/c469c6637f6251adefc282956f5392e559ea915d/b12x/attention/paged/forward_paged.py#L1133-L1240)

```python
@cute.jit
def _literal_qk_mma_into_sfrag_mxfp8_raw(
    s_frag, q_base_addr, k_base_addr,
    lane, warp_q_idx, warp_kv_idx, row_base,
    num_mma_q, num_mma_kv, num_mma_d_qk, upcast_stride_q, upcast_stride_k,
):
    unit_scale = Uint32(0x7F7F7F7F)
    shift16 = Uint32(16)
    ...
    for mma_pair in cutlass.range_constexpr(num_mma_d_qk // 2):
        ...
        for mma_q in cutlass.range_constexpr(num_mma_q):
            q_regs[mma_q, 0] = cvt_bf16x2x2_to_e4m3x4(a_regs_k0[mma_q, 0], a0)
            ...
        for mma_kv in cutlass.range_constexpr(num_mma_kv):
            b0_k0, b0_k1, b1_k0, b1_k1 = ldmatrix_m8n8x4_b16(...)
            b0_k0 = frag_layout_swizzle_16b_to_8b(b0_k0)
            ...
```

### 2c. FP8 Descale Strategy

Per-head descale tensors loaded at kernel level, applied as FP32 scalars:

[L3164-L3178](https://github.com/lukealonso/b12x/blob/c469c6637f6251adefc282956f5392e559ea915d/b12x/attention/paged/forward_paged.py#L3164-L3178)

```python
        k_scale = (
            mKDescale[request_idx]
            if const_expr(mKDescale is not None and len(mKDescale.shape) == 1)
            else (
                mKDescale[request_idx, kv_head_idx]
                if const_expr(mKDescale is not None)
                else Float32(1.0)
            )
        )
        v_scale = (
            mVDescale[request_idx]
            if const_expr(mVDescale is not None and len(mVDescale.shape) == 1)
            else (
                mVDescale[request_idx, kv_head_idx]
                if const_expr(mVDescale is not None)
                else Float32(1.0)
            )
        )
```

FP8 decode raw kernel applies the same pattern:

[L4881-L4888](https://github.com/lukealonso/b12x/blob/c469c6637f6251adefc282956f5392e559ea915d/b12x/attention/paged/forward_paged.py#L4881-L4888)

```python
                mKDescale[request_idx]
                if const_expr(mKDescale is not None and len(mKDescale.shape) == 1)
                else (mKDescale[request_idx, kv_head_idx] if const_expr(mKDescale is not None) else Float32(1.0))
            )
            v_scale = (
                mVDescale[request_idx]
                if const_expr(mVDescale is not None and len(mVDescale.shape) == 1)
                else (mVDescale[request_idx, kv_head_idx] if const_expr(mVDescale is not None) else Float32(1.0))
```

### 2d. GQA Handling

Grid launch maps `kv_head_idx` to `block_y`, GQA group expansion computed in-kernel:

[L2562-L2567](https://github.com/lukealonso/b12x/blob/c469c6637f6251adefc282956f5392e559ea915d/b12x/attention/paged/forward_paged.py#L2562-L2567)

```python
        grid = (
            (
                mO.shape[0] // mPageTable.shape[0],
                mKCache.shape[2],
                mPageTable.shape[0],
            )
            if self.regularized_decode_graph
            else (mBlockValidMask.shape[0], mKCache.shape[2], 1)
        )
```

Block decomposition: `block=[32, num_warps_q, num_warps_kv]`

[L2586](https://github.com/lukealonso/b12x/blob/c469c6637f6251adefc282956f5392e559ea915d/b12x/attention/paged/forward_paged.py#L2586)

```python
            block=[32, self.traits.num_warps_q, self.traits.num_warps_kv],
```

GQA group size computed in kernel body:

[L2618](https://github.com/lukealonso/b12x/blob/c469c6637f6251adefc282956f5392e559ea915d/b12x/attention/paged/forward_paged.py#L2618)

```python
        group_size = mQ.shape[1] // mKCache.shape[2]
```

Q row packing into the GQA group for FP8 decode kernel:

[L4524-L4537](https://github.com/lukealonso/b12x/blob/c469c6637f6251adefc282956f5392e559ea915d/b12x/attention/paged/forward_paged.py#L4524-L4537)

```python
class PagedFp8DecodeRawForwardKernel:
    def __init__(self):
        self.cta_tile_q = 16
        self.stage_tile_rows = 64
        self.num_mma_q = 1
        self.num_mma_kv = 1
        self.num_mma_d_qk = 16
        self.num_mma_d_vo = 16
        self.num_warps_q = 1
        self.num_warps_kv = 4
        self.num_threads = 128
        self.head_dim_qk = 256
        self.head_dim_vo = 256
        self.group_q_rows = 16
```

### 2e. SMEM Partitioning

Shared storage uses a flat byte payload with separate TMA mbarriers:

[L2196-L2210](https://github.com/lukealonso/b12x/blob/c469c6637f6251adefc282956f5392e559ea915d/b12x/attention/paged/forward_paged.py#L2196-L2210)

```python
    def _get_shared_storage_cls(self):
        class SharedStorage:
            pass

        mbar_struct = cute.struct.MemRange[cutlass.Int64, 2 * self.num_stages]
        SharedStorage.__annotations__ = {
            "mbar_ptr_K": mbar_struct,
            "mbar_ptr_V": mbar_struct,
            "payload": cute.struct.Align[
                cute.struct.MemRange[
                    cutlass.Uint8,
                    int(self.traits.shared_storage_bytes),
                ],
                1024,
            ],
        }
```

FP8 decode raw kernel SMEM budget:

[L4548-L4555](https://github.com/lukealonso/b12x/blob/c469c6637f6251adefc282956f5392e559ea915d/b12x/attention/paged/forward_paged.py#L4548-L4555)

```python
        self.q_bytes = self.cta_tile_q * self.head_dim_qk * 2
        self.k_bytes = self.stage_tile_rows * self.head_dim_qk
        self.v_bytes = self.stage_tile_rows * self.head_dim_vo
        qkv_storage_bytes = self.q_bytes + self.k_bytes + self.v_bytes
        cta_sync_o_bytes = self.num_warps_kv * self.cta_tile_q * self.head_dim_vo * 4
        cta_sync_md_bytes = self.num_warps_kv * self.cta_tile_q * 8
        smem_o_bytes = self.cta_tile_q * self.head_dim_vo * 2
        self.shared_storage_bytes = max(qkv_storage_bytes, cta_sync_o_bytes + cta_sync_md_bytes, smem_o_bytes)
```

TMA plane layout with swizzle:

[L2213-L2232](https://github.com/lukealonso/b12x/blob/c469c6637f6251adefc282956f5392e559ea915d/b12x/attention/paged/forward_paged.py#L2213-L2232)

```python
    def _get_paged_kv_tma_plane_layout(self):
        plane_swizzle = os.environ.get("B12X_PAGED_KV_TMA_PLANE_SWIZZLE", "")
        if plane_swizzle == "none":
            return cute.make_layout(
                (self.stage_tile_rows, self.kv_tma_plane_head_dim),
                stride=(self.kv_tma_plane_head_dim, 1),
            )
        if plane_swizzle:
            mbase, bbits, sshift = [int(part) for part in plane_swizzle.split(",")]
            swizzle = make_swizzle(mbase, bbits, sshift)
        else:
            swizzle = make_swizzle(3, 4, 3)
        return cute.make_composed_layout(
            swizzle, 0,
            cute.make_layout(
                (self.stage_tile_rows, self.kv_tma_plane_head_dim),
                stride=(self.kv_tma_plane_head_dim, 1),
            ),
        )
```

KV plane dimensions differ by dtype:

[L2168-L2177](https://github.com/lukealonso/b12x/blob/c469c6637f6251adefc282956f5392e559ea915d/b12x/attention/paged/forward_paged.py#L2168-L2177)

```python
        self.kv_tma_plane_head_dim = 128 if self.kv_is_fp8 else 64
        self.kv_tma_plane_mem_dtype = cutlass.Uint8 if self.kv_is_fp8 else self.dtype_kv_storage
        ...
        self.kv_tma_plane_count = (
            (2 if self.kv_is_fp8 else 4)
            if self.use_paged_kv_tma_exact_plane_bf16_layout
            else 1
        )
```

---

## 3. Extend/Prefill Kernel (forward_extend_generic.py)

**File:** [`b12x/attention/paged/forward_extend_generic.py`](https://github.com/lukealonso/b12x/blob/c469c6637f6251adefc282956f5392e559ea915d/b12x/attention/paged/forward_extend_generic.py)

### 3a. Class Structure

BF16 extend kernel, same TMA-plane pattern:

[L2157-L2213](https://github.com/lukealonso/b12x/blob/c469c6637f6251adefc282956f5392e559ea915d/b12x/attention/paged/forward_extend_generic.py#L2157-L2213)

```python
class PagedForwardKernel:
    def __init__(
        self,
        dtype_q: Type[cutlass.Numeric],
        dtype_kv: Type[cutlass.Numeric],
        dtype_kv_storage: Type[cutlass.Numeric],
        dtype_o: Type[cutlass.Numeric],
        *,
        traits: PagedForwardTraits,
        mxfp8_turbo: bool = False,
        enable_mxfp8_pv: bool = False,
        enable_paged_kv_tma: bool = False,
    ):
        ...
        self.split_kv = False
        ...
        self.num_stages = (
            1
            if traits.num_warps_kv > 1 or self.kv_is_fp8
            else (2 if q_stage_bytes + 2 * kv_stage_bytes <= traits.max_smem_per_threadblock else 1)
        )
        base_use_paged_kv_tma_extend = (
            enable_paged_kv_tma
            and os.environ.get("B12X_PAGED_KV_TMA", "1") != "0"
            and dtype_q == cutlass.BFloat16
            and dtype_o == cutlass.BFloat16
            and traits.head_dim_qk == 256
            and traits.head_dim_vo == 256
            and self.num_stages == 1
            and traits.num_warps_kv > 1
            and traits.num_warps_q == 1
            and self.stage_tile_rows == 64
            and traits.cta_tile_q == 16
            and traits.num_mma_q == 1
            and traits.num_mma_kv == 1
        )
```

FP8 extend raw kernel with dedicated tile sizes:

[L5198-L5235](https://github.com/lukealonso/b12x/blob/c469c6637f6251adefc282956f5392e559ea915d/b12x/attention/paged/forward_extend_generic.py#L5198-L5235)

```python
class PagedFp8ExtendRawForwardKernel:
    def __init__(self, *, split_kv: bool):
        self.split_kv = split_kv
        self.cta_tile_q = 64
        self.stage_tile_rows = 64
        self.compute_tile_rows = 32
        self.num_mma_q = 1
        self.num_mma_kv = 2
        self.num_mma_d_qk = 16
        self.num_mma_d_vo = 16
        self.num_warps_q = 4
        self.num_warps_kv = 1
        self.num_threads = 128
        self.head_dim_qk = 256
        self.head_dim_vo = 256
        self.page_size = 64
        self.q_dtype = cutlass.BFloat16
        self.o_dtype = cutlass.BFloat16
        self.kv_storage_dtype = cutlass.Uint8
        ...
        self.q_bytes = self.cta_tile_q * self.head_dim_qk * 2
        self.k_bytes = self.stage_tile_rows * self.head_dim_qk
        self.v_bytes = self.stage_tile_rows * self.head_dim_vo
        self.shared_storage_bytes = self.q_bytes + self.k_bytes + self.v_bytes
```

### 3b. Softmax State Update with P-Scaling (extend, FP8)

[L810-L850](https://github.com/lukealonso/b12x/blob/c469c6637f6251adefc282956f5392e559ea915d/b12x/attention/paged/forward_extend_generic.py#L810-L850)

```python
@cute.jit
def _donor_update_mdo_states_fp32_pack_p(
    ...
    sm_scale_log2: Float32,
    num_mma_d_vo,
    ...
):
    ...
        scale_term = (
            ...
            else _exp2_approx_ftz_f32(m_prev * sm_scale_log2 - m_new * sm_scale_log2)
        )
        d_frag[0, row_slot] = Float32(d_frag[0, row_slot] * scale_term)
        for mma_d in cutlass.range_constexpr(num_mma_d_vo):
            o_frag[0, mma_d, row_slot * 2 + 0] *= scale_term
            o_frag[0, mma_d, row_slot * 2 + 1] *= scale_term
            o_frag[0, mma_d, row_slot * 2 + 4] *= scale_term
            o_frag[0, mma_d, row_slot * 2 + 5] *= scale_term

        m_scaled = Float32(m_new * sm_scale_log2)
        ...
                else _exp2_approx_ftz_f32(acc_S_mn[row_slot, c] * sm_scale_log2 - m_scaled)
```

---

## 4. Planning and Workspace

### 4a. Workspace Allocation

**File:** [`b12x/attention/paged/workspace.py`](https://github.com/lukealonso/b12x/blob/c469c6637f6251adefc282956f5392e559ea915d/b12x/attention/paged/workspace.py)

Workspace dataclass:

[L56-L102](https://github.com/lukealonso/b12x/blob/c469c6637f6251adefc282956f5392e559ea915d/b12x/attention/paged/workspace.py#L56-L102)

```python
@dataclass(kw_only=True)
class PagedAttentionWorkspace:
    mode: Literal["decode", "extend", "verify"]
    device: torch.device
    dtype: torch.dtype
    kv_dtype: torch.dtype
    num_q_heads: int
    num_kv_heads: int
    head_dim_qk: int
    head_dim_vo: int
    attn_mode: Literal["default", "turbo"] | None = None
    page_size: int = 64
    use_cuda_graph: bool = False
    fixed_capacity: bool = False
    request_indices: torch.Tensor | None = None
    qo_tile_indices: torch.Tensor | None = None
    kv_tile_indices: torch.Tensor | None = None
    merge_indptr: torch.Tensor | None = None
    o_indptr: torch.Tensor | None = None
    kv_chunk_size_ptr: torch.Tensor | None = None
    total_num_rows_ptr: torch.Tensor | None = None
    block_valid_mask: torch.Tensor | None = None
    page_table: torch.Tensor | None = None
    cache_seqlens: torch.Tensor | None = None
    cu_seqlens_q: torch.Tensor | None = None
    lse: torch.Tensor | None = None
    tmp_output: torch.Tensor | None = None
    tmp_lse: torch.Tensor | None = None
    ...
```

Runtime buffer allocation:

[L432-L472](https://github.com/lukealonso/b12x/blob/c469c6637f6251adefc282956f5392e559ea915d/b12x/attention/paged/workspace.py#L432-L472)

```python
    def _allocate_runtime_buffers(
        self, *,
        work_items_capacity: int,
        block_valid_capacity: int,
        total_q_capacity: int,
        batch_capacity: int,
        page_table_width_capacity: int,
        partial_rows_capacity: int,
    ) -> None:
        self.request_indices = torch.empty(work_items_capacity, dtype=torch.int32, device=self.device)
        self.qo_tile_indices = torch.empty(work_items_capacity, dtype=torch.int32, device=self.device)
        self.kv_tile_indices = torch.empty(work_items_capacity, dtype=torch.int32, device=self.device)
        self.block_valid_mask = torch.empty(block_valid_capacity, dtype=torch.int32, device=self.device)
        self.page_table = torch.empty(
            (batch_capacity, page_table_width_capacity), dtype=torch.int32, device=self.device
        )
        self.cache_seqlens = torch.empty(batch_capacity, dtype=torch.int32, device=self.device)
        self.cu_seqlens_q = torch.empty(batch_capacity + 1, dtype=torch.int32, device=self.device)
        self.merge_indptr = torch.empty(total_q_capacity + 1, dtype=torch.int32, device=self.device)
        self.o_indptr = torch.empty(batch_capacity + 1, dtype=torch.int32, device=self.device)
        self.kv_chunk_size_ptr = torch.empty(1, dtype=torch.int32, device=self.device)
        self.total_num_rows_ptr = torch.empty(1, dtype=torch.int32, device=self.device)
        self.lse = torch.empty(
            _paged_lse_storage_shape(total_q_capacity, self.num_q_heads),
            dtype=torch.float32, device=self.device,
        )
        if partial_rows_capacity > 0:
            self.tmp_output = torch.empty(
                (partial_rows_capacity, self.num_q_heads, self.head_dim_vo),
                dtype=self.dtype, device=self.device,
            )
            self.tmp_lse = torch.empty(
                (partial_rows_capacity, self.num_q_heads),
                dtype=torch.float32, device=self.device,
            )
```

### 4b. Split-KV Decision Logic

**File:** [`b12x/attention/paged/planner.py`](https://github.com/lukealonso/b12x/blob/c469c6637f6251adefc282956f5392e559ea915d/b12x/attention/paged/planner.py)

CTA tile Q selection (FlashInfer-faithful):

[L140-L165](https://github.com/lukealonso/b12x/blob/c469c6637f6251adefc282956f5392e559ea915d/b12x/attention/paged/planner.py#L140-L165)

```python
def _fa2_determine_cta_tile_q(avg_packed_qo_len: int, head_dim: int) -> int:
    # Faithful to FlashInfer's FA2DetermineCtaTileQ.
    if avg_packed_qo_len > 64 and head_dim < 256:
        return 128
    if avg_packed_qo_len > 16:
        return 64
    return 16

def _paged_determine_cta_tile_q(
    *,
    mode: _PagedMode,
    kv_dtype: torch.dtype,
    packed_qo_len: int,
    head_dim: int,
    max_effective_kv_pages: int,
) -> int:
    if mode == "verify":
        del kv_dtype, head_dim, max_effective_kv_pages
        return 16
    if mode == "extend" and packed_qo_len <= 32:
        del kv_dtype, head_dim, max_effective_kv_pages
        return 16
    cta_tile_q = _fa2_determine_cta_tile_q(packed_qo_len, head_dim)
    ...
```

Graph split-KV max batch size:

[L75-L85](https://github.com/lukealonso/b12x/blob/c469c6637f6251adefc282956f5392e559ea915d/b12x/attention/paged/planner.py#L75-L85)

```python
def _graph_max_batch_size_if_split(
    *,
    device: torch.device,
    num_kv_heads: int,
    graph_ctas_per_sm: int,
) -> int:
    blocks_per_sm = int(graph_ctas_per_sm)
    ...
    num_sms = int(torch.cuda.get_device_properties(device).multi_processor_count)
    return max((num_sms * blocks_per_sm) // num_kv_heads, 1)
```

FP8 extend chunk table (pages-to-chunk-pages mapping):

[L23-L46](https://github.com/lukealonso/b12x/blob/c469c6637f6251adefc282956f5392e559ea915d/b12x/attention/paged/planner.py#L23-L46)

```python
_PAGED_EXTEND_FP8_CHUNK_TABLE_PAGES = (
    (1, 1),
    (2, 1),
    (4, 1),
    (8, 1),
    (16, 1),
    (32, 2),
    (64, 3),
    (128, 6),
    (256, 6),
    (512, 24),
    (1024, 24),
    (2048, 24),
)
_PAGED_EXTEND_BF16_CHUNK_TABLE_PAGES = (
    (1, 1),
    (2, 1),
    (4, 1),
    (8, 1),
    (16, 1),
    (32, 2),
    (64, 3),
    (128, 6),
    (256, 6),
    (512, 32),
    (1024, 32),
    (2048, 32),
)
```

### 4c. Metadata Tensors

PagedPlan and PagedPlanKey dataclasses:

[L252-L312](https://github.com/lukealonso/b12x/blob/c469c6637f6251adefc282956f5392e559ea915d/b12x/attention/paged/planner.py#L252-L312)

```python
@dataclass(frozen=True)
class PagedPlanKey:
    total_q: int
    num_q_heads: int
    head_dim_qk: int
    head_dim_vo: int
    k_cache_shape: tuple[int, ...]
    v_cache_shape: tuple[int, ...]
    page_table_shape: tuple[int, ...]
    dtype: torch.dtype
    kv_dtype: torch.dtype
    mode: _PagedMode
    cta_tile_q: int
    kv_chunk_size: int
    split_kv: bool
    fixed_split_size: int
    disable_split_kv: bool
    enable_cuda_graph: bool
    graph_chunk_policy: bool
    graph_ctas_per_sm: int
    max_batch_size_if_split: int
    padded_batch_size: int
    new_batch_size: int
    num_qo_tiles: int
    total_num_partial_rows: int
    page_size: int
    num_kv_heads: int
    gqa_group_size: int
    device_index: int


@dataclass(frozen=True, kw_only=True)
class PagedPlan:
    key: PagedPlanKey
    request_indices: tuple[int, ...]
    qo_tile_indices: tuple[int, ...]
    kv_tile_indices: tuple[int, ...]
    merge_indptr: tuple[int, ...]
    o_indptr: tuple[int, ...]
    block_valid_mask: tuple[bool, ...]
```

Worklist generation loop:

[L462-L491](https://github.com/lukealonso/b12x/blob/c469c6637f6251adefc282956f5392e559ea915d/b12x/attention/paged/planner.py#L462-L491)

```python
    for request_idx, (packed_qo_len, qo_len, kv_len) in enumerate(
        zip(packed_qo_len_arr, q_lengths, effective_kv_len_arr)
    ):
        num_tiles_q = _ceil_div(packed_qo_len, cta_tile_q)
        num_chunks_kv = 1 if disable_split_kv and not force_split_kv else _ceil_div(max(kv_len, 1), kv_chunk_size_pages)
        request_num_chunks_kv.append(int(num_chunks_kv))
        if not disable_split_kv or force_split_kv:
            split_kv = split_kv or num_chunks_kv > 1
        for q_tile_idx in range(num_tiles_q):
            for kv_tile_idx in range(num_chunks_kv):
                new_batch_size += 1
                request_indices.append(request_idx)
                qo_tile_indices.append(q_tile_idx)
                kv_tile_indices.append(kv_tile_idx)
        for _ in range(qo_len):
            merge_indptr.append(merge_indptr[-1] + num_chunks_kv)
        o_indptr.append(o_indptr[-1] + qo_len * num_chunks_kv)
```

---

## 5. Merge Kernel

**File:** [`b12x/attention/paged/merge.py`](https://github.com/lukealonso/b12x/blob/c469c6637f6251adefc282956f5392e559ea915d/b12x/attention/paged/merge.py)

### 5a. Persistent Merge Class

[L220-L275](https://github.com/lukealonso/b12x/blob/c469c6637f6251adefc282956f5392e559ea915d/b12x/attention/paged/merge.py#L220-L275)

```python
class PagedPersistentMergeKernel:
    """Faithful row/head-persistent merge for the current paged backend path."""

    def __init__(
        self,
        dtype: Type[cutlass.Numeric],
        dtype_partial: Type[cutlass.Numeric],
        *,
        head_dim: int,
        vec_size: int = 8,
        bdx: int = 32,
        bdy: int = 4,
        num_smem_stages: int = 4,
        persistent_ctas: int | None = None,
        direct_grid: bool = False,
        regular_decode_graph: bool = False,
    ):
```

### 5b. State Merge Arithmetic (base-2 LSE)

[L109-L143](https://github.com/lukealonso/b12x/blob/c469c6637f6251adefc282956f5392e559ea915d/b12x/attention/paged/merge.py#L109-L143)

```python
@cute.jit
def _state_merge(
    state_o, state_m, state_d, other_o, other_m, other_d,
) -> tuple[Float32, Float32]:
    prev_m = state_m
    prev_d = state_d
    if prev_m == -Float32.inf:
        if other_m == -Float32.inf:
            state_m = Float32(prev_m)
            state_d = Float32(prev_d)
        else:
            for vec_idx in cutlass.range_constexpr(cute.size(state_o.shape)):
                state_o[vec_idx] = other_o[vec_idx]
            state_m = Float32(other_m)
            state_d = Float32(other_d)
    elif other_m == -Float32.inf:
        state_m = Float32(prev_m)
        state_d = Float32(prev_d)
    else:
        state_m = attention_utils.fmax(prev_m, other_m)
        prev_scale = _exp2_approx_ftz_f32(prev_m - state_m)
        other_scale = _exp2_approx_ftz_f32(other_m - state_m)
        state_d = Float32(prev_d * prev_scale + other_d * other_scale)
        for vec_idx in cutlass.range_constexpr(cute.size(state_o.shape)):
            state_o[vec_idx] = state_o[vec_idx] * prev_scale + other_o[vec_idx] * other_scale
    return Float32(state_m), state_d
```

### 5c. cp.async Staged Partial Load

[L170-L218](https://github.com/lukealonso/b12x/blob/c469c6637f6251adefc282956f5392e559ea915d/b12x/attention/paged/merge.py#L170-L218)

```python
@cute.jit
def _merge_async_slot(
    *, iter_idx, start_idx, num_heads, head_idx, num_index_sets, num_stage_iters,
    base_k, head_dim, vec_size, bdx, bdy, num_smem_stages,
    s_stage_partial, s_stage_lse, mV_partial, mLSE_partial,
    state_o, state_m, state_d, slot,
) -> tuple[Float32, Float32]:
    ...
    cur_iter = iter_idx + Int32(slot)
    if cur_iter < num_stage_iters:
        if cur_iter % bdx == 0:
            lse_linear_idx = cur_iter * bdy + ty * bdx + tx
            s_stage_lse[ty * bdx + tx] = (
                mLSE_partial[start_idx + lse_linear_idx, head_idx]
                if lse_linear_idx < num_index_sets
                else Float32(0.0)
            )
            cute.arch.sync_threads()

        cute.arch.cp_async_wait_group(num_smem_stages - 1)
        cute.arch.sync_threads()

        if cur_iter * bdy + ty < num_index_sets:
            other_o = cute.make_rmem_tensor(...)
            for vec_idx in cutlass.range_constexpr(vec_size):
                other_o[vec_idx] = s_stage_partial[
                    cur_iter % num_smem_stages, ty, base_k + vec_idx
                ].to(Float32)
            state_m, state_d = _state_merge_normalized_lse_base2(
                state_o, state_m, state_d, other_o,
                s_stage_lse[(cur_iter % bdx) * bdy + ty],
            )
        ...
        next_linear_idx = (cur_iter + num_smem_stages) * bdy + ty
        smem_addr = shared_ptr_to_u32(...)
        if next_linear_idx < num_index_sets:
            ...
            _cp_async_load_128b(smem_addr, gmem_addr)
        cute.arch.cp_async_commit_group()
```

### 5d. Persistent CTA Grid Sizing

[L86-L107](https://github.com/lukealonso/b12x/blob/c469c6637f6251adefc282956f5392e559ea915d/b12x/attention/paged/merge.py#L86-L107)

```python
def default_paged_persistent_ctas(
    *, total_rows: int, num_heads: int, device: torch.device | int | None = None,
) -> int:
    if device is None:
        device = torch.cuda.current_device()
    num_sms = int(torch.cuda.get_device_properties(device).multi_processor_count)
    total_work = max(int(total_rows) * int(num_heads), 1)
    blocks_per_sm = min(3, _ceil_div(total_work, num_sms))
    persistent_ctas = int(num_sms * max(blocks_per_sm, 1))
    if int(total_rows) == 8:
        return int(min(persistent_ctas, total_work))
    return persistent_ctas
```

### 5e. Grid Dep Control

Merge kernel uses `griddepcontrol` for persistent-CTA wave scheduling:

[L357](https://github.com/lukealonso/b12x/blob/c469c6637f6251adefc282956f5392e559ea915d/b12x/attention/paged/merge.py#L357) and [L442](https://github.com/lukealonso/b12x/blob/c469c6637f6251adefc282956f5392e559ea915d/b12x/attention/paged/merge.py#L442)

```python
        if const_expr(not self.direct_grid):
            cute.arch.griddepcontrol_wait()
        ...
        cute.arch.griddepcontrol_launch_dependents()
```

---

## 6. CUDA Graph Replay

**File:** [`b12x/attention/paged/graph_replay.py`](https://github.com/lukealonso/b12x/blob/c469c6637f6251adefc282956f5392e559ea915d/b12x/attention/paged/graph_replay.py)

### 6a. Page Table Build (Triton)

[L16-L36](https://github.com/lukealonso/b12x/blob/c469c6637f6251adefc282956f5392e559ea915d/b12x/attention/paged/graph_replay.py#L16-L36)

```python
@triton.jit
def build_decode_graph_page_table_triton(
    req_to_token_ptr, req_pool_indices_ptr, page_table_ptr, active_max_pages_ptr,
    req_to_token_row_stride, page_table_row_stride,
    PAGE_SIZE: tl.constexpr, BLOCK_PAGES: tl.constexpr,
):
    req_idx = tl.program_id(axis=0)
    page_block_idx = tl.program_id(axis=1)

    req_pool_idx = tl.load(req_pool_indices_ptr + req_idx).to(tl.int64)
    active_max_pages = tl.load(active_max_pages_ptr).to(tl.int32)
    page_offsets = page_block_idx * BLOCK_PAGES + tl.arange(0, BLOCK_PAGES)
    page_mask = page_offsets < active_max_pages
    flat_token_offsets = req_pool_idx * req_to_token_row_stride + page_offsets.to(tl.int64) * PAGE_SIZE
    token_indices = tl.load(req_to_token_ptr + flat_token_offsets, mask=page_mask, other=0)
    tl.store(
        page_table_ptr + req_idx * page_table_row_stride + page_offsets,
        (token_indices // PAGE_SIZE).to(tl.int32),
        mask=page_mask,
    )
```

### 6b. Chunk Metadata Update (Triton)

[L39-L62](https://github.com/lukealonso/b12x/blob/c469c6637f6251adefc282956f5392e559ea915d/b12x/attention/paged/graph_replay.py#L39-L62)

```python
@triton.jit
def update_decode_graph_metadata_triton(
    cache_seqlens_ptr, merge_indptr_ptr, block_valid_mask_ptr, chunk_pages_ptr,
    max_chunks_per_req, PAGE_SIZE: tl.constexpr, BLOCK_CHUNKS: tl.constexpr,
):
    req_idx = tl.program_id(axis=0)
    chunk_block_idx = tl.program_id(axis=1)

    cache_len = tl.load(cache_seqlens_ptr + req_idx).to(tl.int32)
    chunk_pages = tl.load(chunk_pages_ptr).to(tl.int32)
    num_pages = tl.maximum((cache_len + (PAGE_SIZE - 1)) // PAGE_SIZE, 1)
    num_chunks = (num_pages + chunk_pages - 1) // chunk_pages

    tl.store(merge_indptr_ptr + req_idx + 1, num_chunks)

    chunk_offsets = chunk_block_idx * BLOCK_CHUNKS + tl.arange(0, BLOCK_CHUNKS)
    chunk_mask = chunk_offsets < max_chunks_per_req
    is_active = chunk_offsets < num_chunks
    tl.store(
        block_valid_mask_ptr + req_idx * max_chunks_per_req + chunk_offsets,
        is_active.to(tl.int32),
        mask=chunk_mask,
    )
```

### 6c. Chunk Pages LUT

[L97-L122](https://github.com/lukealonso/b12x/blob/c469c6637f6251adefc282956f5392e559ea915d/b12x/attention/paged/graph_replay.py#L97-L122)

```python
def make_decode_chunk_pages_lut_tensor(
    decode_chunk_pages_lut: Sequence[int], *, device: torch.device,
) -> torch.Tensor:
    ...
    return torch.tensor(
        (int(decode_chunk_pages_lut[0]), *(int(chunk_pages) for chunk_pages in decode_chunk_pages_lut)),
        dtype=torch.int32, device=device,
    )


def summarize_decode_chunk_pages_lut(
    decode_chunk_pages_lut: Sequence[int],
) -> tuple[int, int]:
    ...
    worst_page_count = 1
    max_chunks_per_req = 1
    for page_count, chunk_pages in enumerate(decode_chunk_pages_lut, start=1):
        num_chunks = (page_count + int(chunk_pages) - 1) // int(chunk_pages)
        if num_chunks > max_chunks_per_req:
            max_chunks_per_req = num_chunks
            worst_page_count = page_count
    return int(worst_page_count), int(max_chunks_per_req)
```

---

## 7. Adaptation Notes for nvllm

### 7a. Trait Selection

| b12x source | nvllm target | Changes needed |
|---|---|---|
| [`traits.py` L40-L65](https://github.com/lukealonso/b12x/blob/c469c6637f6251adefc282956f5392e559ea915d/b12x/attention/paged/traits.py#L40-L65) | `cute_paged/traits.py` | Query SM120 SMEM budget (228KB/SM); our model: head_dim=256, GQA=6 (24Q/4KV), FP8 KV. Adjust `gqa_group_size` checks in planner for group_size=6. |
| [`traits.py` L128-L165](https://github.com/lukealonso/b12x/blob/c469c6637f6251adefc282956f5392e559ea915d/b12x/attention/paged/traits.py#L128-L165) | `cute_paged/traits.py` | `num_ctas_per_sm` calculation uses runtime SMEM query -- SM120 has 228KB vs SM100's 232KB. Verify 2-CTA occupancy still fits. |

### 7b. Decode Kernel

| b12x source | nvllm target | Changes needed |
|---|---|---|
| [`forward_paged.py` L2063-L2128](https://github.com/lukealonso/b12x/blob/c469c6637f6251adefc282956f5392e559ea915d/b12x/attention/paged/forward_paged.py#L2063-L2128) | `cute_paged/forward_decode.py` | Remove `single_request_decode_graph` / `single_qtile_decode_graph` paths (vLLM always uses regularized graph replay). Drop MXFP8 turbo for initial port. |
| [`forward_paged.py` L951-L1130](https://github.com/lukealonso/b12x/blob/c469c6637f6251adefc282956f5392e559ea915d/b12x/attention/paged/forward_paged.py#L951-L1130) | `cute_paged/mma.py` | BF16 `bf16_mma_m16n16k16_f32` and FP8 `fp8x4_e4m3_to_bfloat2x2` + `frag_layout_swizzle_16b_to_8b` are the core compute primitives. Same ISA on SM120. |
| [`forward_paged.py` L3164-L3178](https://github.com/lukealonso/b12x/blob/c469c6637f6251adefc282956f5392e559ea915d/b12x/attention/paged/forward_paged.py#L3164-L3178) | `cute_paged/forward_decode.py` | Descale: vLLM passes per-head `(num_kv_heads,)` descale tensors. b12x supports both `(batch,)` and `(batch, kv_heads)`. Use the `(batch, kv_heads)` path. |
| [`forward_paged.py` L2562-L2586](https://github.com/lukealonso/b12x/blob/c469c6637f6251adefc282956f5392e559ea915d/b12x/attention/paged/forward_paged.py#L2562-L2586) | `cute_paged/forward_decode.py` | Grid: `(work_items, num_kv_heads, 1)` or regularized `(max_chunks, num_kv_heads, batch)`. vLLM uses regularized decode graph replay. |

### 7c. Planner

| b12x source | nvllm target | Changes needed |
|---|---|---|
| [`planner.py` L140-L165](https://github.com/lukealonso/b12x/blob/c469c6637f6251adefc282956f5392e559ea915d/b12x/attention/paged/planner.py#L140-L165) | `cute_paged/planner.py` | Directly reusable. Our GQA ratio=6 means `packed_qo_len = q_len * 6` -- for decode this is 6, well under the 16-row tile, so `cta_tile_q=16` always. |
| [`planner.py` L23-L46](https://github.com/lukealonso/b12x/blob/c469c6637f6251adefc282956f5392e559ea915d/b12x/attention/paged/planner.py#L23-L46) | `cute_paged/planner.py` | Chunk tables are tuned for `gqa_group_size=8`. Needs re-tuning for `gqa_group_size=6` via `sweep_decode_graph_policy.py`. |
| [`planner.py` L252-L312](https://github.com/lukealonso/b12x/blob/c469c6637f6251adefc282956f5392e559ea915d/b12x/attention/paged/planner.py#L252-L312) | `cute_paged/planner.py` | `PagedPlan` / `PagedPlanKey` dataclasses can be used verbatim. |

### 7d. Merge Kernel

| b12x source | nvllm target | Changes needed |
|---|---|---|
| [`merge.py` L220-L275](https://github.com/lukealonso/b12x/blob/c469c6637f6251adefc282956f5392e559ea915d/b12x/attention/paged/merge.py#L220-L275) | `cute_paged/merge.py` | b12x uses `head_dim=256`, `vec_size=8`; our model also uses `head_dim=256` so this is directly reusable. `bdx=32` unchanged. `griddepcontrol` is SM90+ and works on SM120. |
| [`merge.py` L109-L143](https://github.com/lukealonso/b12x/blob/c469c6637f6251adefc282956f5392e559ea915d/b12x/attention/paged/merge.py#L109-L143) | `cute_paged/merge.py` | Base-2 LSE state merge arithmetic is architecture-independent. |

### 7e. Workspace / Graph Replay

| b12x source | nvllm target | Changes needed |
|---|---|---|
| [`workspace.py` L56-L102](https://github.com/lukealonso/b12x/blob/c469c6637f6251adefc282956f5392e559ea915d/b12x/attention/paged/workspace.py#L56-L102) | `cute_paged/workspace.py` | Adapt `for_contract` factory to accept vLLM's `ModelConfig` / `CacheConfig` instead of raw tensor shapes. Map `page_size=64` (hardcoded in b12x, matches our vLLM config). |
| [`graph_replay.py` L16-L62](https://github.com/lukealonso/b12x/blob/c469c6637f6251adefc282956f5392e559ea915d/b12x/attention/paged/graph_replay.py#L16-L62) | `cute_paged/graph_replay.py` | Triton graph replay kernels are arch-independent. Need to bridge vLLM's `req_to_token` layout to b12x's `(req_pool_idx, stride)` convention. |

### 7f. Key Differences: Qwen3.5-27B vs b12x Tuning Target

- **GQA ratio**: b12x tunes for 8 (Qwen3.5-397B). We need 6 (24Q/4KV for Qwen3.5-27B). Chunk tables and `packed_qo_len` calculations change.
- **SM count**: b12x targets SM120 B200 (132 SMs). DGX Spark GB10 has fewer SMs. Persistent CTA grid sizing in merge kernel needs SM count query.
- **FP8 validity rule**: `_paged_is_invalid` has `kv_is_fp8 and (num_mma_kv * 2) % num_warps_q != 0`. With our config (num_mma_kv=1, num_warps_q=1), this is `2 % 1 == 0` which is fine.

---

## Verification Checklist

- [x] All permalink URLs pinned to commit `c469c6637f6251adefc282956f5392e559ea915d`
- [x] License compatibility confirmed (Apache-2.0 to Apache-2.0)
- [x] Per-piece links provided for all quoted code
- [x] README acknowledgment added
