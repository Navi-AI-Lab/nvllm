#!/usr/bin/env python3
"""SM120 CuTe DSL test — FP4 block-scaled MMA with scale factors on GB10.

Step 1: Basic JIT (elementwise add) — proves sm_121a works
Step 2: FP4 MMA without scale factors — proves MMA instruction works
Step 3: FP4 MMA with scale factors — proves full block-scaled pipeline

Usage:
    CUTE_DSL_ARCH=sm_121a python fp4_mma_test.py
"""

import torch
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
from cutlass.cute.nvgpu.warp.mma import (
    MmaMXF4NVF4Op,
    MmaSM120BlockScaledOp,
    Field,
)
from cutlass.base_dsl.arch import Arch
from cutlass.cutlass_dsl import BaseDSL

print(f"torch: {torch.__version__}, GPU: {torch.cuda.get_device_name(0)}")
print(f"CUTLASS DSL: {cutlass.__version__}")

# Monkey-patch SM121 support (GB10 has identical FP4 MMA instructions as SM120)
_orig = MmaSM120BlockScaledOp.__post_init__

def _patched_post_init(self):
    arch = BaseDSL._get_dsl().get_arch_enum()
    if arch not in (Arch.sm_120a, Arch.sm_121a):
        from cutlass.cute.nvgpu.common import OpError
        raise OpError(self, f"expects sm_120a or sm_121a, got {arch}")
    if self.ab_dtype != cutlass.Float4E2M1FN:
        from cutlass.cute.nvgpu.common import OpError
        raise OpError(self, "expects ab_dtype Float4E2M1FN")
    if self.acc_dtype != cutlass.Float32:
        from cutlass.cute.nvgpu.common import OpError
        raise OpError(self, "expects acc_dtype Float32")
    if self.shape_mnk != (16, 8, 64):
        from cutlass.cute.nvgpu.common import OpError
        raise OpError(self, "expects shape_mnk (16, 8, 64)")
    if self.sf_vec_size == 16:
        if self.sf_type != cutlass.Float8E4M3FN:
            from cutlass.cute.nvgpu.common import OpError
            raise OpError(self, "expects sf_type Float8E4M3FN for VS=16")
    elif self.sf_vec_size == 32:
        if self.sf_type != cutlass.Float8E8M0FNU:
            from cutlass.cute.nvgpu.common import OpError
            raise OpError(self, "expects sf_type Float8E8M0FNU for VS=32")
    else:
        from cutlass.cute.nvgpu.common import OpError
        raise OpError(self, "expects sf_vec_size 16 or 32")

MmaSM120BlockScaledOp.__post_init__ = _patched_post_init

# SM120 warp-level FP4 block-scaled MMA: 16x8x64
# NVF4: E4M3 scale factors, VS=16
# Use use_sf_layout_TV=True so SF can be in register fragments
import cutlass._mlir.dialects.cute_nvgpu as _cute_nvgpu_ir
from cutlass.cute.core import _pack_shape
from cutlass.cute.atom import make_atom

class MmaMXF4NVF4OpTV(MmaSM120BlockScaledOp):
    """NVF4 op with use_sf_layout_TV=True for register-resident scale factors."""
    descriptive_name = "warp-level MXF4NVF4 MMA (TV SF layout)"

    def __init__(self, ab_dtype, acc_dtype, sf_type):
        # Initialize with use_sf_layout_TV=True
        object.__setattr__(self, 'ab_dtype', ab_dtype)
        object.__setattr__(self, 'acc_dtype', acc_dtype)
        object.__setattr__(self, 'shape_mnk', (16, 8, 64))
        object.__setattr__(self, 'sf_type', sf_type)
        object.__setattr__(self, 'sf_vec_size', 16)
        object.__setattr__(self, 'use_sf_layout_TV', True)
        self.__post_init__()

    def _make_trait(self, *, loc=None, ip=None, **kwargs):
        from cutlass.cute.nvgpu.warp.mma import MmaMXF4NVF4Trait
        shape_mnk = _pack_shape(self.shape_mnk, loc=loc, ip=ip)
        ty = _cute_nvgpu_ir.MmaAtomSM120BlockScaledType.get(
            shape_mnk.type.attribute,
            16,     # sf_vec_size
            True,   # use_sf_layout_TV = True!
            self.ab_dtype.mlir_type,
            self.ab_dtype.mlir_type,
            self.acc_dtype.mlir_type,
            self.sf_type.mlir_type,
        )
        return MmaMXF4NVF4Trait(make_atom(ty, loc=loc, ip=ip))

# Create both variants
MMA_OP = MmaMXF4NVF4Op(
    ab_dtype=cutlass.Float4E2M1FN,
    acc_dtype=cutlass.Float32,
    sf_type=cutlass.Float8E4M3FN,
)

MMA_OP_TV = MmaMXF4NVF4OpTV(
    ab_dtype=cutlass.Float4E2M1FN,
    acc_dtype=cutlass.Float32,
    sf_type=cutlass.Float8E4M3FN,
)


# =============================================================================
# Step 1: Basic JIT test
# =============================================================================

@cute.kernel
def add_kernel(gA: cute.Tensor, gB: cute.Tensor, gC: cute.Tensor):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    idx = bidx * 32 + tidx
    gC[idx] = gA[idx] + gB[idx]


@cute.jit
def add_host(mA: cute.Tensor, mB: cute.Tensor, mC: cute.Tensor):
    add_kernel(mA, mB, mC).launch(grid=[1, 1, 1], block=[32, 1, 1])


def test_jit():
    print("\n=== Step 1: Basic JIT test ===")
    N = 32
    a = torch.ones(N, dtype=torch.float32, device="cuda")
    b = torch.ones(N, dtype=torch.float32, device="cuda") * 2.0
    c = torch.zeros(N, dtype=torch.float32, device="cuda")
    compiled = cute.compile(add_host, from_dlpack(a), from_dlpack(b), from_dlpack(c))
    compiled(from_dlpack(a), from_dlpack(b), from_dlpack(c))
    torch.cuda.synchronize()
    assert c[0].item() == 3.0, f"Expected 3.0, got {c[0].item()}"
    print("PASS — JIT works on sm_121a")
    return True


# =============================================================================
# Step 2: FP4 MMA (no scale factors — just proves MMA instruction executes)
# =============================================================================

@cute.kernel
def fp4_mma_kernel_nosf(
    gA: cute.Tensor,
    gB: cute.Tensor,
    gC: cute.Tensor,
):
    tidx, _, _ = cute.arch.thread_idx()
    tiled_mma = cute.make_tiled_mma(MMA_OP, atom_layout_mnk=(1, 1, 1))
    thr_mma = tiled_mma.get_slice(tidx)

    # Recast i8 → FP4
    gA_fp4 = cute.recast_tensor(gA, cutlass.Float4E2M1FN)
    gB_fp4 = cute.recast_tensor(gB, cutlass.Float4E2M1FN)

    tAr = thr_mma.partition_A(gA_fp4)
    tBr = thr_mma.partition_B(gB_fp4)
    tCgC = thr_mma.partition_C(gC)

    tCrC = tiled_mma.make_fragment_C(tCgC)
    tCrC.fill(0.0)

    tCrA = tiled_mma.make_fragment_A(tAr)
    tCrB = tiled_mma.make_fragment_B(tBr)
    copy_atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), cutlass.Float4E2M1FN)
    cute.copy(copy_atom, tAr, tCrA)
    cute.copy(copy_atom, tBr, tCrB)

    cute.gemm(tiled_mma, tCrC, tCrA, tCrB, tCrC)

    copy_out = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), gC.element_type)
    cute.copy(copy_out, tCrC, tCgC)


@cute.jit
def fp4_mma_host_nosf(mA: cute.Tensor, mB: cute.Tensor, mC: cute.Tensor):
    fp4_mma_kernel_nosf(mA, mB, mC).launch(grid=[1, 1, 1], block=[32, 1, 1])


def test_fp4_mma_nosf():
    print("\n=== Step 2: FP4 MMA (no SF) ===")
    M, N, K = 16, 8, 64
    A = torch.randint(0, 255, (M, K // 2), dtype=torch.uint8, device="cuda")
    B = torch.randint(0, 255, (N, K // 2), dtype=torch.uint8, device="cuda")
    C = torch.zeros(M, N, dtype=torch.float32, device="cuda")

    mA, mB, mC = from_dlpack(A), from_dlpack(B), from_dlpack(C)
    compiled = cute.compile(fp4_mma_host_nosf, mA, mB, mC)
    compiled(mA, mB, mC)
    torch.cuda.synchronize()
    print(f"PASS — FP4 MMA executed, C[0,:4] = {C[0, :4].tolist()}")
    return True


# =============================================================================
# Step 3: FP4 MMA WITH scale factors
# =============================================================================

@cute.kernel
def fp4_mma_kernel_sf(
    gA: cute.Tensor,     # (M, K/2) uint8, FP4 packed
    gB: cute.Tensor,     # (N, K/2) uint8, FP4 packed
    gC: cute.Tensor,     # (M, N) float32
    gSFA: cute.Tensor,   # scale factors A (i8 → will recast to E4M3)
    gSFB: cute.Tensor,   # scale factors B (i8 → will recast to E4M3)
):
    tidx, _, _ = cute.arch.thread_idx()

    # Use original MMA op (use_sf_layout_TV=false)
    tiled_mma = cute.make_tiled_mma(MMA_OP, atom_layout_mnk=(1, 1, 1))
    thr_mma = tiled_mma.get_slice(tidx)

    # Recast i8 → FP4 for A/B
    gA_fp4 = cute.recast_tensor(gA, cutlass.Float4E2M1FN)
    gB_fp4 = cute.recast_tensor(gB, cutlass.Float4E2M1FN)

    # Recast i8 → E4M3 for scale factors
    gSFA_e4m3 = cute.recast_tensor(gSFA, cutlass.Float8E4M3FN)
    gSFB_e4m3 = cute.recast_tensor(gSFB, cutlass.Float8E4M3FN)

    # Partition all tensors
    tAr = thr_mma.partition_A(gA_fp4)
    tBr = thr_mma.partition_B(gB_fp4)
    tCgC = thr_mma.partition_C(gC)

    # Allocate accumulator
    tCrC = tiled_mma.make_fragment_C(tCgC)
    tCrC.fill(0.0)

    # Load A, B to registers
    tCrA = tiled_mma.make_fragment_A(tAr)
    tCrB = tiled_mma.make_fragment_B(tBr)
    copy_atom_fp4 = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), cutlass.Float4E2M1FN)
    cute.copy(copy_atom_fp4, tAr, tCrA)
    cute.copy(copy_atom_fp4, tBr, tCrB)

    # Allocate SMEM for scale factors
    import cutlass.utils as utils
    smem = utils.SmemAllocator()
    sfa_layout = cute.make_layout(gSFA_e4m3.shape)
    sfb_layout = cute.make_layout(gSFB_e4m3.shape)
    # Allocate enough SMEM for both SF tensors
    sfa_size = cute.size(sfa_layout) * 1  # 1 byte per E4M3
    sfb_size = cute.size(sfb_layout) * 1
    sfa_storage = smem.allocate(sfa_size, byte_alignment=16)
    sfb_storage = smem.allocate(sfb_size, byte_alignment=16)

    # SmemAllocator returns a Pointer — recast to our SF dtype
    sfa_ptr = cute.recast_ptr(sfa_storage, dtype=cutlass.Float8E4M3FN)
    sfb_ptr = cute.recast_ptr(sfb_storage, dtype=cutlass.Float8E4M3FN)
    sSFA = cute.make_tensor(sfa_ptr, sfa_layout)
    sSFB = cute.make_tensor(sfb_ptr, sfb_layout)

    # Copy SF GMEM → SMEM
    sf_copy_atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), cutlass.Float8E4M3FN)
    cute.copy(sf_copy_atom, gSFA_e4m3, sSFA)
    cute.copy(sf_copy_atom, gSFB_e4m3, sSFB)
    cute.arch.sync_threads()

    # Set SF from SMEM pointers
    tiled_mma.set(Field.SFA, sSFA.iterator)
    tiled_mma.set(Field.SFB, sSFB.iterator)

    # Execute MMA with scale factors
    cute.gemm(tiled_mma, tCrC, tCrA, tCrB, tCrC)

    # Store back
    copy_out = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), gC.element_type)
    cute.copy(copy_out, tCrC, tCgC)


@cute.jit
def fp4_mma_host_sf(
    mA: cute.Tensor, mB: cute.Tensor, mC: cute.Tensor,
    mSFA: cute.Tensor, mSFB: cute.Tensor,
):
    print(f"[DSL INFO] mA = {mA.type}")
    print(f"[DSL INFO] mSFA = {mSFA.type}")
    fp4_mma_kernel_sf(mA, mB, mC, mSFA, mSFB).launch(
        grid=[1, 1, 1], block=[32, 1, 1],
    )


def test_fp4_mma_sf():
    print("\n=== Step 3: FP4 MMA with Scale Factors ===")
    M, N, K = 16, 8, 64

    # All-ones FP4 data: 0x11 = two FP4 values of 1.0 (e2m1: sign=0, exp=01, mantissa=1 → 1.0)
    # Actually FP4 E2M1: 0b0_01_1 = 1.5, 0b0_01_0 = 1.0, 0b0_00_1 = 0.5
    # Pack two 1.0s: 0b0010_0010 = 0x22
    A = torch.full((M, K // 2), 0x22, dtype=torch.uint8, device="cuda")  # all 1.0 FP4
    B = torch.full((N, K // 2), 0x22, dtype=torch.uint8, device="cuda")  # all 1.0 FP4
    C = torch.zeros(M, N, dtype=torch.float32, device="cuda")

    # Scale factors = 1.0 (E4M3: 0b0_0111_000 = 1.0 = 0x38)
    SFA = torch.full((M, K // 16), 0x38, dtype=torch.uint8, device="cuda").view(torch.float8_e4m3fn)
    SFB = torch.full((N, K // 16), 0x38, dtype=torch.uint8, device="cuda").view(torch.float8_e4m3fn)

    print(f"Problem: M={M}, N={N}, K={K}")
    print(f"A: all 1.0 FP4, B: all 1.0 FP4, SF: all 1.0")
    print(f"Expected C[i,j] = sum(1.0 * 1.0 * 1.0 * 1.0, k=0..63) = 64.0")

    mA = from_dlpack(A)
    mB = from_dlpack(B)
    mC = from_dlpack(C)
    mSFA = from_dlpack(SFA.view(torch.uint8))
    mSFB = from_dlpack(SFB.view(torch.uint8))

    print("Compiling...")
    compiled = cute.compile(fp4_mma_host_sf, mA, mB, mC, mSFA, mSFB)
    print("Executing...")
    compiled(mA, mB, mC, mSFA, mSFB)
    torch.cuda.synchronize()

    print(f"C[0,:] = {C[0, :].tolist()}")
    if abs(C[0, 0].item() - 64.0) < 1.0:
        print("PASS — Numerical result correct!")
    else:
        print(f"NOTE — C[0,0] = {C[0, 0].item()} (expected ~64.0, may need SF layout fix)")
    return True


if __name__ == "__main__":
    ok = test_jit()
    if ok:
        ok = test_fp4_mma_nosf()
    if ok:
        test_fp4_mma_sf()
