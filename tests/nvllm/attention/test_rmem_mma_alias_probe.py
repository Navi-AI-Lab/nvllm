#!/usr/bin/env python3
"""Probe: does MMA into rmem_tensor[0..7] corrupt rmem_tensor[16..23]?

Tests whether calling bf16_mma_m16n16k16_f32 with accumulators from
rmem_tensor indices 0-7 causes values to appear at indices 16-23
(the stride-32 output pattern observed in the decode kernel).

Matches the kernel's exact pattern:
  o_lo = rmem_tensor(64, Float32), fill(0)
  for _md in constexpr(16):
    _ob = _md * 8
    (o_lo[_ob+0], ...) = MMA(o_lo[_ob+0], ..., A, B)
  dump all 64 elements
"""
import torch
import logging

logging.basicConfig(level=logging.WARNING)

try:
    import cutlass
    from cutlass import cute
    from cutlass._mlir import ir as _mlir_ir
    from cutlass._mlir.dialects import llvm as _llvm_dialect
    from cutlass.cute.typing import BFloat16, Float32, Int32, Int64, Uint32
    from cutlass.cutlass_dsl import T, dsl_user_op
    _CUTE_AVAILABLE = True
except ImportError:
    print("CUTLASS not available")
    exit(1)


# --- PTX helpers (same as kernel) ---

@dsl_user_op
def _mma_m16n8k16_f32(
    d0: Float32, d1: Float32, d2: Float32, d3: Float32,
    a0: Uint32, a1: Uint32, a2: Uint32, a3: Uint32,
    b0: Uint32, b1: Uint32,
    *, loc=None, ip=None,
):
    operands_ir = [
        Uint32(a0).ir_value(loc=loc, ip=ip),
        Uint32(a1).ir_value(loc=loc, ip=ip),
        Uint32(a2).ir_value(loc=loc, ip=ip),
        Uint32(a3).ir_value(loc=loc, ip=ip),
        Uint32(b0).ir_value(loc=loc, ip=ip),
        Uint32(b1).ir_value(loc=loc, ip=ip),
        Float32(d0).ir_value(loc=loc, ip=ip),
        Float32(d1).ir_value(loc=loc, ip=ip),
        Float32(d2).ir_value(loc=loc, ip=ip),
        Float32(d3).ir_value(loc=loc, ip=ip),
    ]
    result_struct = _llvm_dialect.inline_asm(
        _mlir_ir.Type.parse("!llvm.struct<(f32, f32, f32, f32)>"),
        operands_ir,
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{$0, $1, $2, $3}, {$4, $5, $6, $7}, {$8, $9}, "
        "{$10, $11, $12, $13};",
        "=f,=f,=f,=f,r,r,r,r,r,r,f,f,f,f",
        has_side_effects=True, asm_dialect=0, loc=loc, ip=ip,
    )
    r0 = Float32(_llvm_dialect.extractvalue(
        T.f32(), result_struct, [0], loc=loc, ip=ip))
    r1 = Float32(_llvm_dialect.extractvalue(
        T.f32(), result_struct, [1], loc=loc, ip=ip))
    r2 = Float32(_llvm_dialect.extractvalue(
        T.f32(), result_struct, [2], loc=loc, ip=ip))
    r3 = Float32(_llvm_dialect.extractvalue(
        T.f32(), result_struct, [3], loc=loc, ip=ip))
    return r0, r1, r2, r3


@dsl_user_op
def _cvt_2f32_to_bf16x2(lo: Float32, hi: Float32, *,
                         loc=None, ip=None) -> Uint32:
    lo_ir = Float32(lo).ir_value(loc=loc, ip=ip)
    hi_ir = Float32(hi).ir_value(loc=loc, ip=ip)
    result_ir = _llvm_dialect.inline_asm(
        T.i32(), [lo_ir, hi_ir],
        "cvt.rn.bf16x2.f32 $0, $2, $1;", "=r,f,f",
        has_side_effects=True, asm_dialect=0, loc=loc, ip=ip,
    )
    return Uint32(result_ir)


@cute.jit
def bf16_mma_m16n16k16_f32(
    d0, d1, d2, d3, d4, d5, d6, d7,
    a0, a1, a2, a3, b0, b1, b2, b3,
):
    d0, d1, d2, d3 = _mma_m16n8k16_f32(
        d0, d1, d2, d3, a0, a1, a2, a3, b0, b1)
    d4, d5, d6, d7 = _mma_m16n8k16_f32(
        d4, d5, d6, d7, a0, a1, a2, a3, b2, b3)
    return d0, d1, d2, d3, d4, d5, d6, d7


# --- Probe kernel ---

class MmaRmemProbe:
    """Calls MMA into rmem_tensor[_md*8 .. _md*8+7] for 8 _md iterations.
    Only _md=0 gets nonzero B. Checks if _md=2 (indices 16-23) is nonzero."""

    def __init__(self):
        self.num_threads = 32  # 1 warp
        self._compiled = None

    @cute.jit
    def _jit_launch(self, output):
        self._kernel(output).launch(
            grid=[1, 1, 1], block=[self.num_threads, 1, 1],
            smem=0,
        )

    @cute.kernel
    def _kernel(self, output):
        lane = cute.arch.lane_idx()
        group = lane // Int32(4)
        sub = lane % Int32(4)

        # A operand: all 1.0 (simulate uniform P)
        one = _cvt_2f32_to_bf16x2(Float32(1.0), Float32(1.0))
        a0 = one
        a1 = one
        a2 = one
        a3 = one

        # B operand: 1.0 for _md=0, 0.0 for all others
        b_one = _cvt_2f32_to_bf16x2(Float32(1.0), Float32(1.0))
        b_zero = _cvt_2f32_to_bf16x2(Float32(0.0), Float32(0.0))

        # Create rmem_tensor(64) — matching the kernel's o_lo
        o = cute.make_rmem_tensor(
            cute.make_layout((64,)), Float32)
        o.fill(Float32(0.0))

        # MMA loop: 8 iterations (_md = 0..7)
        for _md in cutlass.range_constexpr(8):
            _ob = _md * 8
            if _md == 0:
                # Nonzero B for _md=0 only
                (o[_ob + 0], o[_ob + 1],
                 o[_ob + 2], o[_ob + 3],
                 o[_ob + 4], o[_ob + 5],
                 o[_ob + 6], o[_ob + 7],
                 ) = bf16_mma_m16n16k16_f32(
                    o[_ob + 0], o[_ob + 1],
                    o[_ob + 2], o[_ob + 3],
                    o[_ob + 4], o[_ob + 5],
                    o[_ob + 6], o[_ob + 7],
                    a0, a1, a2, a3,
                    b_one, b_one, b_one, b_one)
            else:
                # Zero B for all other _md
                (o[_ob + 0], o[_ob + 1],
                 o[_ob + 2], o[_ob + 3],
                 o[_ob + 4], o[_ob + 5],
                 o[_ob + 6], o[_ob + 7],
                 ) = bf16_mma_m16n16k16_f32(
                    o[_ob + 0], o[_ob + 1],
                    o[_ob + 2], o[_ob + 3],
                    o[_ob + 4], o[_ob + 5],
                    o[_ob + 6], o[_ob + 7],
                    a0, a1, a2, a3,
                    b_zero, b_zero, b_zero, b_zero)

        # Dump all 64 elements to output (lane 0 only)
        if lane == Int32(0):
            for _i in cutlass.range_constexpr(64):
                output[Int32(_i)] = o[_i]

    def __call__(self):
        output = torch.zeros(64, dtype=torch.float32, device="cuda")
        if self._compiled is None:
            self._compiled = cute.compile(
                self._jit_launch, output)
        self._compiled(output)
        return output


def main():
    print("MMA + rmem_tensor(64) aliasing probe")
    print("=" * 60)
    print("MMA into o[_md*8 .. _md*8+7] for _md=0..7")
    print("Only _md=0 has nonzero B. All others have B=0.")
    print("Expected: only o[0..7] nonzero, o[8..63] = 0")
    print("=" * 60)
    print()

    probe = MmaRmemProbe()
    out = probe()
    vals = out.cpu().tolist()

    # Print nonzero elements
    print("Nonzero elements:")
    for i, v in enumerate(vals):
        if abs(v) > 1e-6:
            md = i // 8
            off = i % 8
            tag = " (EXPECTED)" if md == 0 else " *** ALIAS ***"
            print(f"  o[{i:2d}] = {v:8.2f}  (_md={md}, off={off}){tag}")

    # Count nonzero per _md
    print("\nPer-_md summary:")
    for md in range(8):
        start = md * 8
        nz = sum(1 for v in vals[start:start+8] if abs(v) > 1e-6)
        if nz > 0:
            print(f"  _md={md}: {nz}/8 nonzero "
                  f"{'(EXPECTED)' if md == 0 else '*** ALIAS ***'}")
        else:
            print(f"  _md={md}: all zero (correct)")

    # Check for stride-16 aliasing
    stride16 = any(abs(vals[i] - vals[i - 16]) < 1e-6 and abs(vals[i]) > 1e-6
                    for i in range(16, 64))
    print(f"\nStride-16 aliasing: {'YES' if stride16 else 'NO'}")


if __name__ == "__main__":
    main()
