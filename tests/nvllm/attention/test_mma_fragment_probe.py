#!/usr/bin/env python3
"""Probe the ACTUAL m16n8k16 B fragment layout on SM121.

Creates a minimal kernel that does a single MMA with:
  A = identity-like (each row has 1.0 at specific K position)
  B = known pattern (each column has a unique marker)
Then inspects D to determine which B column maps to which output N position.

This reveals if SM121 has a non-standard MMA fragment layout.
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


# --- Copy essential PTX helpers from kernel.py ---

@dsl_user_op
def shared_ptr_to_i64(ptr, *, loc=None, ip=None) -> Int64:
    ptr_ir = ptr.toint(loc=loc, ip=ip).ir_value(loc=loc, ip=ip)
    result_ir = _llvm_dialect.inline_asm(
        T.i64(), [ptr_ir],
        "cvta.to.shared.u64 $0, $1;", "=l,l",
        has_side_effects=True, asm_dialect=0, loc=loc, ip=ip,
    )
    return Int64(result_ir)

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
def _cvt_2f32_to_bf16x2(lo: Float32, hi: Float32, *, loc=None, ip=None) -> Uint32:
    lo_ir = Float32(lo).ir_value(loc=loc, ip=ip)
    hi_ir = Float32(hi).ir_value(loc=loc, ip=ip)
    result_ir = _llvm_dialect.inline_asm(
        T.i32(), [lo_ir, hi_ir],
        "cvt.rn.bf16x2.f32 $0, $2, $1;", "=r,f,f",
        has_side_effects=True, asm_dialect=0, loc=loc, ip=ip,
    )
    return Uint32(result_ir)

@dsl_user_op
def _st_shared_f32(addr: Int64, val: Float32, *, loc=None, ip=None) -> None:
    addr_ir = Int64(addr).ir_value(loc=loc, ip=ip)
    val_ir = Float32(val).ir_value(loc=loc, ip=ip)
    _llvm_dialect.inline_asm(
        None, [addr_ir, val_ir],
        "st.shared.f32 [$0], $1;", "l,f",
        has_side_effects=True, asm_dialect=0, loc=loc, ip=ip,
    )


class MMAProbe:
    """Minimal kernel to probe MMA B fragment layout."""

    def __init__(self):
        self.num_threads = 32  # 1 warp
        # SMEM for 16x8 FP32 output = 512 bytes
        self.smem_bytes = 16 * 8 * 4
        self._compiled = None

    @cute.jit
    def _jit_launch(self, output, b_col_to_test: Int32):
        self._kernel(output, b_col_to_test).launch(
            grid=[1, 1, 1], block=[self.num_threads, 1, 1],
            smem=self.smem_bytes,
        )

    @cute.kernel
    def _kernel(self, output, b_col_to_test: Int32):
        """Single m16n8k16 MMA: A=all-ones, B=one-hot at b_col_to_test.

        A[m, k] = 1.0 for all m, k
        B[k, n] = 1.0 if n == b_col_to_test, else 0.0

        Expected D[m, n] = 16.0 if n == b_col_to_test, else 0.0
        (sum of 16 ones times 1.0)

        Output: D flattened to 16*8 = 128 FP32 values via SMEM.
        """
        lane = cute.arch.lane_idx()
        group = lane // Int32(4)
        sub = lane % Int32(4)

        # A operand: all 1.0 in BF16
        # a0 = bf16x2(1.0, 1.0) for all threads
        one_pair = _cvt_2f32_to_bf16x2(Float32(1.0), Float32(1.0))
        a0 = one_pair
        a1 = one_pair
        a2 = one_pair
        a3 = one_pair

        # B operand: 1.0 only for the column == b_col_to_test
        # B fragment: thread group=g provides B[:,g]
        # b0 = {B[sub*2, group], B[sub*2+1, group]}
        # B[k, n] = 1.0 if n == b_col_to_test, else 0.0
        # So: if group == b_col_to_test, b0 = {1.0, 1.0}, else {0.0, 0.0}

        zero_pair = _cvt_2f32_to_bf16x2(Float32(0.0), Float32(0.0))
        # CuTe DSL requires variables defined before control flow
        b0 = zero_pair
        b1 = zero_pair
        if group == b_col_to_test:
            b0 = one_pair
            b1 = one_pair

        # MMA
        d0, d1, d2, d3 = _mma_m16n8k16_f32(
            Float32(0.0), Float32(0.0), Float32(0.0), Float32(0.0),
            a0, a1, a2, a3, b0, b1,
        )

        # Write D to SMEM then to output
        smem = cute.arch.get_dyn_smem(cutlass.Uint8, alignment=128)
        out_smem = shared_ptr_to_i64(smem)

        # d0 = D[group, sub*2]
        # d1 = D[group, sub*2+1]
        # d2 = D[group+8, sub*2]
        # d3 = D[group+8, sub*2+1]

        # Write to SMEM: out[row, col] at row*8 + col
        off_r0 = (group * Int32(8) + sub * Int32(2)) * Int32(4)
        _st_shared_f32(out_smem + Int64(off_r0), d0)
        _st_shared_f32(out_smem + Int64(off_r0 + Int32(4)), d1)

        off_r1 = ((group + Int32(8)) * Int32(8) + sub * Int32(2)) * Int32(4)
        _st_shared_f32(out_smem + Int64(off_r1), d2)
        _st_shared_f32(out_smem + Int64(off_r1 + Int32(4)), d3)

        cute.arch.sync_threads()

        # Lane 0 writes all 128 values to output
        if lane == Int32(0):
            for _i in cutlass.range_constexpr(128):
                idx = Int32(_i)
                # Read FP32 from SMEM
                val_addr = out_smem + Int64(idx * Int32(4))
                val_ir = Float32(val_addr).ir_value()
                # Actually, use ld.shared.f32
                # Just write row*8 + col pattern for now
                pass

        # Actually, just have ALL threads write their D values to output
        # Each thread writes 4 values at known positions
        # output layout: [16, 8] flattened
        out_idx_00 = group * Int32(8) + sub * Int32(2)
        output[out_idx_00] = Float32(d0)
        output[out_idx_00 + Int32(1)] = Float32(d1)

        out_idx_10 = (group + Int32(8)) * Int32(8) + sub * Int32(2)
        output[out_idx_10] = Float32(d2)
        output[out_idx_10 + Int32(1)] = Float32(d3)

    def __call__(self, b_col):
        device = "cuda"
        output = torch.zeros(128, dtype=torch.float32, device=device)

        if self._compiled is None:
            self._compiled = cute.compile(
                self._jit_launch, output, Int32(b_col),
            )

        self._compiled(output, Int32(b_col))
        return output.view(16, 8)


def main():
    print("MMA m16n8k16 B Fragment Layout Probe (SM121)")
    print("=" * 60)
    print("A = all 1.0, B = one-hot at specific column")
    print("Expected: D[:, col] = 16.0, rest = 0.0")
    print()

    probe = MMAProbe()

    for b_col in range(8):
        D = probe(b_col)
        # Find which output columns are nonzero
        col_sums = D.abs().sum(dim=0)
        nz_cols = (col_sums > 0.5).nonzero(as_tuple=False).flatten().tolist()
        # Also check values
        max_val = D.max().item()
        print(f"  B col {b_col} → nonzero output cols: {nz_cols}, "
              f"max value: {max_val:.1f}")

    print("\nDetailed output for B col 0:")
    D = probe(0)
    print(f"  Row 0: {D[0].tolist()}")
    print(f"  Row 1: {D[1].tolist()}")


if __name__ == "__main__":
    main()
