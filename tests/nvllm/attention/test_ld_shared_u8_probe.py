#!/usr/bin/env python3
"""Probe: compare _ld_shared_u8 (byte load) vs _ld_shared_b32+extract (word load).

Writes 256 known bytes to SMEM (byte[i] = i & 0xFF), then reads each byte
back via two methods:
  1. _ld_shared_u8  -- the inline PTX byte load used in the production kernel
  2. _ld_shared_b32 + extract_byte -- load aligned word, shift+mask

If these disagree, the byte load has an aliasing/addressing bug (e.g. the
suspected stride-32 pattern where byte X returns the same value as byte X-32).

Volume-mount into container and run:
  python /app/nvllm/tests/nvllm/attention/test_ld_shared_u8_probe.py
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


# ---------------------------------------------------------------------------
# PTX helpers (self-contained copies from the production kernel)
# ---------------------------------------------------------------------------

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
def _st_shared_b32(addr: Int64, val: Uint32, *, loc=None, ip=None) -> None:
    addr_ir = Int64(addr).ir_value(loc=loc, ip=ip)
    val_ir = Uint32(val).ir_value(loc=loc, ip=ip)
    _llvm_dialect.inline_asm(
        None, [addr_ir, val_ir],
        "st.shared.b32 [$0], $1;", "l,r",
        has_side_effects=True, asm_dialect=0, loc=loc, ip=ip,
    )


@dsl_user_op
def _ld_shared_u8(addr: Int64, *, loc=None, ip=None) -> Uint32:
    """Load 1 byte from shared memory, zero-extended to Uint32.

    This is the EXACT same PTX as the production kernel. If this returns
    wrong values, the bug is in the byte addressing.
    """
    addr_ir = Int64(addr).ir_value(loc=loc, ip=ip)
    result_ir = _llvm_dialect.inline_asm(
        T.i32(), [addr_ir],
        "{\n"
        "  .reg .b32 tmp;\n"
        "  ld.shared.b8 tmp, [$1];\n"
        "  and.b32 $0, tmp, 0xFF;\n"
        "}",
        "=r,l",
        has_side_effects=True, asm_dialect=0, loc=loc, ip=ip,
    )
    return Uint32(result_ir)


@dsl_user_op
def _ld_shared_b32(addr: Int64, *, loc=None, ip=None) -> Uint32:
    """Load a 4-byte word from shared memory."""
    addr_ir = Int64(addr).ir_value(loc=loc, ip=ip)
    result_ir = _llvm_dialect.inline_asm(
        T.i32(), [addr_ir],
        "ld.shared.b32 $0, [$1];", "=r,l",
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


@dsl_user_op
def _extract_byte_from_b32(
    word: Uint32, byte_pos: Int32, *, loc=None, ip=None,
) -> Uint32:
    """Extract byte at byte_pos (0-3) from a 32-bit word using PTX."""
    word_ir = Uint32(word).ir_value(loc=loc, ip=ip)
    pos_ir = Int32(byte_pos).ir_value(loc=loc, ip=ip)
    result_ir = _llvm_dialect.inline_asm(
        T.i32(), [word_ir, pos_ir],
        "{\n"
        "  .reg .b32 shift, tmp;\n"
        "  shl.b32 shift, $2, 3;\n"   # shift = byte_pos * 8
        "  shr.b32 tmp, $1, shift;\n"
        "  and.b32 $0, tmp, 0xFF;\n"
        "}",
        "=r,r,r",
        has_side_effects=True, asm_dialect=0, loc=loc, ip=ip,
    )
    return Uint32(result_ir)


@dsl_user_op
def _u32_to_f32(val: Uint32, *, loc=None, ip=None) -> Float32:
    """Convert Uint32 to Float32 via PTX cvt."""
    val_ir = Uint32(val).ir_value(loc=loc, ip=ip)
    result_ir = _llvm_dialect.inline_asm(
        T.f32(), [val_ir],
        "cvt.rn.f32.u32 $0, $1;", "=f,r",
        has_side_effects=True, asm_dialect=0, loc=loc, ip=ip,
    )
    return Float32(result_ir)


@dsl_user_op
def _ld_shared_f32(addr: Int64, *, loc=None, ip=None) -> Float32:
    """Load FP32 from shared memory."""
    addr_ir = Int64(addr).ir_value(loc=loc, ip=ip)
    result_ir = _llvm_dialect.inline_asm(
        T.f32(), [addr_ir],
        "ld.shared.f32 $0, [$1];", "=f,l",
        has_side_effects=True, asm_dialect=0, loc=loc, ip=ip,
    )
    return Float32(result_ir)


# ---------------------------------------------------------------------------
# Probe kernel
# ---------------------------------------------------------------------------

class LdSharedProbe:
    """Writes 256 known bytes to SMEM, reads back via u8 and b32+extract."""

    def __init__(self):
        self.num_threads = 32   # 1 warp
        # 256 bytes pattern + 256*2 FP32 outputs staged in SMEM = ~2304 bytes
        self.smem_bytes = 4096  # generous
        self._compiled = None

    @cute.jit
    def _jit_launch(self, output_u8, output_b32):
        self._kernel(output_u8, output_b32).launch(
            grid=[1, 1, 1], block=[self.num_threads, 1, 1],
            smem=self.smem_bytes,
        )

    @cute.kernel
    def _kernel(self, output_u8, output_b32):
        lane = cute.arch.lane_idx()
        smem = cute.arch.get_dyn_smem(cutlass.Uint8, alignment=128)
        base = shared_ptr_to_i64(smem)

        # ----- Phase 1: Thread 0 writes 256 bytes (64 words) to SMEM -----
        # Pattern: byte[i] = i & 0xFF
        # Word at position w: bytes [w*4, w*4+1, w*4+2, w*4+3]
        # Packed little-endian: b0 | (b1<<8) | (b2<<16) | (b3<<24)
        if lane == Int32(0):
            for _w in cutlass.range_constexpr(64):
                b0 = Int32(_w * 4)
                b1 = Int32(_w * 4 + 1)
                b2 = Int32(_w * 4 + 2)
                b3 = Int32(_w * 4 + 3)
                word = Uint32(b0 | (b1 << 8) | (b2 << 16) | (b3 << 24))
                _st_shared_b32(base + Int64(Int32(_w) * Int32(4)), word)

        cute.arch.sync_threads()

        # ----- Phase 2: Read bytes back via both methods -----
        # 32 threads * 8 bytes each = 256 bytes total
        byte_base = lane * Int32(8)

        for _j in cutlass.range_constexpr(8):
            byte_idx = byte_base + Int32(_j)

            # Method 1: _ld_shared_u8 (the suspected buggy path)
            val_u8 = _ld_shared_u8(base + Int64(byte_idx))

            # Method 2: _ld_shared_b32 + extract (the control)
            # Use bit ops for int division — CuTe DSL '/' is float
            aligned_off = byte_idx & Int32(0xFFFFFFFC)  # byte_idx & ~3
            word_off = byte_idx & Int32(3)
            word = _ld_shared_b32(base + Int64(aligned_off))
            val_b32 = _extract_byte_from_b32(word, word_off)

            # Convert both to FP32 and write to output tensors
            output_u8[byte_idx] = _u32_to_f32(val_u8)
            output_b32[byte_idx] = _u32_to_f32(val_b32)

    def __call__(self):
        device = "cuda"
        output_u8 = torch.zeros(256, dtype=torch.float32, device=device)
        output_b32 = torch.zeros(256, dtype=torch.float32, device=device)

        if self._compiled is None:
            self._compiled = cute.compile(
                self._jit_launch, output_u8, output_b32,
            )

        self._compiled(output_u8, output_b32)
        return output_u8, output_b32


def main():
    print("ld.shared.b8 vs ld.shared.b32+extract Probe (SM121)")
    print("=" * 60)
    print("Pattern: byte[i] = i (0..255)")
    print("Method 1: _ld_shared_u8  (inline PTX ld.shared.b8)")
    print("Method 2: _ld_shared_b32 + shift/mask (word load)")
    print("=" * 60)
    print()

    probe = LdSharedProbe()
    out_u8, out_b32 = probe()

    # Convert to integer for comparison
    vals_u8 = out_u8.cpu().int()
    vals_b32 = out_b32.cpu().int()
    expected = torch.arange(256, dtype=torch.int32)

    # Check b32 method against expected (sanity check the control)
    b32_errors = (vals_b32 != expected).nonzero(as_tuple=False).flatten().tolist()
    if b32_errors:
        print(f"WARNING: b32+extract control itself has errors at bytes: "
              f"{b32_errors[:20]}...")
        for i in b32_errors[:10]:
            print(f"  byte {i}: expected={i}, got={vals_b32[i].item()}")
    else:
        print("CONTROL OK: b32+extract matches expected for all 256 bytes")

    # Check u8 method against expected
    u8_errors = (vals_u8 != expected).nonzero(as_tuple=False).flatten().tolist()
    if u8_errors:
        print(f"\nld.shared.b8 ERRORS at {len(u8_errors)} bytes:")
        for i in u8_errors[:32]:
            print(f"  byte {i:3d}: expected={i:3d}, "
                  f"u8_got={vals_u8[i].item():3d}, "
                  f"b32_got={vals_b32[i].item():3d}")
        if len(u8_errors) > 32:
            print(f"  ... and {len(u8_errors) - 32} more")
    else:
        print("ld.shared.b8 OK: matches expected for all 256 bytes")

    # Direct comparison: u8 vs b32
    mismatches = (vals_u8 != vals_b32).nonzero(as_tuple=False).flatten().tolist()
    print(f"\nDirect comparison (u8 vs b32): {len(mismatches)} mismatches")
    if mismatches:
        print("\nMismatch details:")
        for i in mismatches[:32]:
            u8_val = vals_u8[i].item()
            b32_val = vals_b32[i].item()
            delta = i - u8_val if u8_val != i else 0
            print(f"  byte {i:3d}: u8={u8_val:3d}  b32={b32_val:3d}  "
                  f"(u8 returned value from byte {u8_val}, delta={delta})")
        if len(mismatches) > 32:
            print(f"  ... and {len(mismatches) - 32} more")

        # Analyze aliasing pattern
        deltas = []
        for i in mismatches:
            u8_val = vals_u8[i].item()
            if u8_val < 256:
                deltas.append(i - u8_val)
        if deltas:
            from collections import Counter
            delta_counts = Counter(deltas)
            print(f"\nAliasing pattern analysis (byte_idx - u8_returned_value):")
            for d, count in delta_counts.most_common(10):
                print(f"  delta={d:4d}: {count} occurrences"
                      f"  (u8 reads byte at offset-{d} instead)")

        # Check for stride-32 specifically
        stride32 = [i for i in mismatches
                     if vals_u8[i].item() == (i - 32) % 256]
        stride64 = [i for i in mismatches
                     if vals_u8[i].item() == (i - 64) % 256]
        if stride32:
            print(f"\nStride-32 aliasing confirmed at {len(stride32)} bytes: "
                  f"{stride32[:16]}...")
        if stride64:
            print(f"Stride-64 aliasing confirmed at {len(stride64)} bytes: "
                  f"{stride64[:16]}...")
    else:
        print("PASS: Both methods agree on all 256 bytes")

    # Summary
    print("\n" + "=" * 60)
    if not mismatches and not u8_errors:
        print("RESULT: PASS -- ld.shared.b8 works correctly")
    elif mismatches:
        print(f"RESULT: FAIL -- {len(mismatches)}/256 bytes differ between "
              "ld.shared.b8 and ld.shared.b32+extract")
    else:
        print(f"RESULT: FAIL -- ld.shared.b8 returned wrong values at "
              f"{len(u8_errors)} positions")
    print("=" * 60)


if __name__ == "__main__":
    main()
