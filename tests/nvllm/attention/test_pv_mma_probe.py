#!/usr/bin/env python3
"""Minimal PV MMA probe: tests whether the BF16 m16n16k16 MMA
produces correct output for a known P × V product.

Isolates the MMA fragment mapping from all other kernel logic
(no softmax, no page loop, no cross-warp reduction).

Setup:
  P = identity-like: P[row, col] = 1.0 for 2 specific (row,col) pairs
  V = identity-like: V[tok, dim] = 1.0 for specific positions
  Expected: O[row, dim] = P[row, tok] * V[tok, dim]

Volume-mount into container and run:
  python /app/nvllm/tests/nvllm/attention/test_pv_mma_probe.py
"""
import torch
import logging

logging.basicConfig(level=logging.WARNING)


def test_pv_mma_probe():
    try:
        import cutlass
        from cutlass import cute
        from cutlass._mlir import ir as _mlir_ir
        from cutlass._mlir.dialects import llvm as _llvm_dialect
        from cutlass.cute.typing import (
            BFloat16, Float32, Int32, Int64, Uint32,
        )
        from cutlass.cutlass_dsl import T, dsl_user_op
    except ImportError:
        print("CUTLASS not available, skipping")
        return

    # Import PTX helpers from kernel.py
    from vllm.v1.attention.backends.cute_paged.kernel import (
        _mma_m16n8k16_f32,
        bf16_mma_m16n16k16_f32,
        _cvt_2f32_to_bf16x2,
        fp8x4_e4m3_to_bfloat2x2,
        _pack_4bytes,
        _ld_shared_b32,
        _st_shared_b32,
        _st_shared_f32,
        _extract_byte_from_b32,
        _ld_shared_f32,
        shared_ptr_to_i64,
    )

    # --- Probe kernel: P × V with known data, dump all fragments ---
    class PVProbe:
        def __init__(self):
            self.num_threads = 128
            # SMEM: V buffer (16 tokens × 16 dims = 256 bytes FP8)
            #      + output buffer (16 rows × 16 cols × 4 bytes FP32)
            self.v_bytes = 16 * 16  # 256 bytes
            self.out_bytes = 16 * 16 * 4  # 1024 bytes
            self.smem_bytes = self.v_bytes + self.out_bytes
            self._compiled = None

        @cute.jit
        def _jit_launch(self, v_data, output,
                        grid_x: Int32):
            self._kernel(v_data, output).launch(
                grid=[grid_x, Int32(1), Int32(1)],
                block=[self.num_threads, 1, 1],
                smem=self.smem_bytes,
            )

        @cute.kernel
        def _kernel(self, v_data, output):
            """Probe: load V from global, set P=known, do PV MMA,
            dump all fragment outputs to global output."""
            lane = cute.arch.lane_idx()
            warp = cute.arch.warp_idx()
            tid = warp * Int32(32) + lane
            group = lane >> Int32(2)   # 0..7
            sub = lane & Int32(3)      # 0..3

            smem = cute.arch.get_dyn_smem(cutlass.Uint8, alignment=128)
            v_smem = shared_ptr_to_i64(smem)
            out_smem = shared_ptr_to_i64(
                smem + Int32(self.v_bytes))

            # Only warp 0 does the probe
            if warp == Int32(0):
                # Load V data into SMEM: 16 tokens × 16 dims = 256 bytes
                # Each thread loads 256/32 = 8 bytes = 2 uint32
                for _i in cutlass.range_constexpr(2):
                    flat_w = lane * Int32(2) + Int32(_i)
                    v_raw = v_data[flat_w]
                    _st_shared_b32(v_smem + Int64(
                        flat_w * Int32(4)), v_raw)

            cute.arch.sync_threads()

            if warp == Int32(0):
                # === Set P to NON-UNIFORM values to detect swaps ===
                # a0 = {A[group, sub*2], A[group, sub*2+1]}     → P for row group
                # a1 = {A[group, sub*2+8], A[group, sub*2+9]}   → P for row group, k+8
                # a2 = {A[group+8, sub*2], A[group+8, sub*2+1]} → P for row group+8
                # a3 = {A[group+8, sub*2+8], A[group+8, sub*2+9]} → P for row group+8, k+8
                #
                # Use: a0={1,1}, a1={0,0}, a2={2,2}, a3={0,0}
                # Row group gets P=1 for k<8, P=0 for k>=8
                # Row group+8 gets P=2 for k<8, P=0 for k>=8
                # If a1/a2 swap: row group gets P=2 at k+8, group+8 gets P=0 at k<8
                pa0 = _cvt_2f32_to_bf16x2(Float32(1.0), Float32(1.0))
                pa1 = _cvt_2f32_to_bf16x2(Float32(0.0), Float32(0.0))
                pa2 = _cvt_2f32_to_bf16x2(Float32(2.0), Float32(2.0))
                pa3 = _cvt_2f32_to_bf16x2(Float32(0.0), Float32(0.0))

                # === Load V fragments from SMEM (same as decode kernel) ===
                hd = Int32(16)
                v_tok0 = sub * Int32(2)  # no warp offset
                v_hd0 = group            # _md=0, so v_k_start=0

                # First m16n8: V cols [group]
                v_off_0a = v_tok0 * hd + v_hd0
                v_off_0b = (v_tok0 + Int32(1)) * hd + v_hd0
                v_off_8a = (v_tok0 + Int32(8)) * hd + v_hd0
                v_off_8b = (v_tok0 + Int32(9)) * hd + v_hd0

                vw0 = _ld_shared_b32(
                    v_smem + Int64(v_off_0a & Int32(0xFFFFFFFC)))
                vw1 = _ld_shared_b32(
                    v_smem + Int64(v_off_0b & Int32(0xFFFFFFFC)))
                vw8 = _ld_shared_b32(
                    v_smem + Int64(v_off_8a & Int32(0xFFFFFFFC)))
                vw9 = _ld_shared_b32(
                    v_smem + Int64(v_off_8b & Int32(0xFFFFFFFC)))

                v_byte_pos = v_hd0 & Int32(3)
                vb0_0 = _extract_byte_from_b32(vw0, v_byte_pos)
                vb0_1 = _extract_byte_from_b32(vw1, v_byte_pos)
                vb0_8 = _extract_byte_from_b32(vw8, v_byte_pos)
                vb0_9 = _extract_byte_from_b32(vw9, v_byte_pos)
                v_packed_0 = _pack_4bytes(
                    vb0_0, vb0_1, vb0_8, vb0_9)
                vb0, vb1 = fp8x4_e4m3_to_bfloat2x2(v_packed_0)

                # Second m16n8: V cols [group+8]
                v_hd1 = group + Int32(8)
                v_off_0c = v_tok0 * hd + v_hd1
                v_off_0d = (v_tok0 + Int32(1)) * hd + v_hd1
                v_off_8c = (v_tok0 + Int32(8)) * hd + v_hd1
                v_off_8d = (v_tok0 + Int32(9)) * hd + v_hd1

                vw0b = _ld_shared_b32(
                    v_smem + Int64(v_off_0c & Int32(0xFFFFFFFC)))
                vw1b = _ld_shared_b32(
                    v_smem + Int64(v_off_0d & Int32(0xFFFFFFFC)))
                vw8b = _ld_shared_b32(
                    v_smem + Int64(v_off_8c & Int32(0xFFFFFFFC)))
                vw9b = _ld_shared_b32(
                    v_smem + Int64(v_off_8d & Int32(0xFFFFFFFC)))

                v_byte_pos1 = v_hd1 & Int32(3)
                vb1_0 = _extract_byte_from_b32(vw0b, v_byte_pos1)
                vb1_1 = _extract_byte_from_b32(vw1b, v_byte_pos1)
                vb1_8 = _extract_byte_from_b32(vw8b, v_byte_pos1)
                vb1_9 = _extract_byte_from_b32(vw9b, v_byte_pos1)
                v_packed_1 = _pack_4bytes(
                    vb1_0, vb1_1, vb1_8, vb1_9)
                vb2, vb3 = fp8x4_e4m3_to_bfloat2x2(v_packed_1)

                # === PV MMA ===
                (t0, t1, t2, t3,
                 t4, t5, t6, t7) = bf16_mma_m16n16k16_f32(
                    Float32(0.0), Float32(0.0),
                    Float32(0.0), Float32(0.0),
                    Float32(0.0), Float32(0.0),
                    Float32(0.0), Float32(0.0),
                    pa0, pa1, pa2, pa3,
                    vb0, vb1, vb2, vb3)

                # === Dump t0..t7 to output SMEM ===
                # Layout: out_smem[row][col] where
                #   row = group (or group+8), col = sub*2 (or +1, +8, +9)
                W16 = Int32(16)
                r0 = group * W16 * Int32(4)
                r1 = (group + Int32(8)) * W16 * Int32(4)
                c0 = sub * Int32(2)
                c8 = sub * Int32(2) + Int32(8)

                _st_shared_f32(out_smem + Int64(
                    r0 + c0 * Int32(4)), t0)
                _st_shared_f32(out_smem + Int64(
                    r0 + (c0 + Int32(1)) * Int32(4)), t1)
                _st_shared_f32(out_smem + Int64(
                    r1 + c0 * Int32(4)), t2)
                _st_shared_f32(out_smem + Int64(
                    r1 + (c0 + Int32(1)) * Int32(4)), t3)
                _st_shared_f32(out_smem + Int64(
                    r0 + c8 * Int32(4)), t4)
                _st_shared_f32(out_smem + Int64(
                    r0 + (c8 + Int32(1)) * Int32(4)), t5)
                _st_shared_f32(out_smem + Int64(
                    r1 + c8 * Int32(4)), t6)
                _st_shared_f32(out_smem + Int64(
                    r1 + (c8 + Int32(1)) * Int32(4)), t7)

            cute.arch.sync_threads()

            # All threads copy out_smem to global output
            # 16×16 = 256 FP32 values, 128 threads → 2 per thread
            for _i in cutlass.range_constexpr(2):
                flat = tid * Int32(2) + Int32(_i)
                val = _ld_shared_f32(out_smem + Int64(
                    flat * Int32(4)))
                output[flat] = val

        def __call__(self, v_data):
            output = torch.zeros(256, dtype=torch.float32,
                                 device=v_data.device)
            if self._compiled is None:
                print("Compiling PV MMA probe...")
                self._compiled = cute.compile(
                    self._jit_launch,
                    v_data, output, Int32(1),
                )
            self._compiled(v_data, output, Int32(1))
            return output.reshape(16, 16)

    probe = PVProbe()

    # --- Test 1: V = identity, P non-uniform (detect a1/a2 swap) ---
    # P: row group=1.0 for k<8, 0 for k>=8; row group+8=2.0 for k<8, 0 for k>=8
    # V[tok, dim] = 1.0 iff tok == dim
    # Expected: O[group, dim] = 1.0 * V[dim, dim] = 1.0 for dim 0..7
    #           O[group+8, dim] = 2.0 * V[dim, dim] = 2.0 for dim 0..7
    #           All other dims = 0 (P for k>=8 is 0)
    v_ident = torch.zeros(16, 16, dtype=torch.uint8, device="cuda")
    for t in range(16):
        v_ident[t, t] = 0x38  # E4M3 1.0
    v_flat = v_ident.contiguous().view(-1)
    v_u32 = v_flat.view(torch.int32)

    result = probe(v_u32)
    print("=== Test 1: V=identity, P non-uniform (a0={1,1} a1={0,0} a2={2,2} a3={0,0}) ===")
    print("Expected row 0: 1.0 at cols 0-7, 0 at 8-15")
    print("Expected row 8: 2.0 at cols 0-7, 0 at 8-15")
    print(f"Result[0, :] = {result[0].tolist()}")
    print(f"Result[1, :] = {result[1].tolist()}")
    print(f"Result[8, :] = {result[8].tolist()}")
    print(f"Result[9, :] = {result[9].tolist()}")
    # Check specific positions for a1/a2 swap
    print(f"\nSwap detection:")
    print(f"  Result[0, 0] = {result[0, 0].item():.1f}  (expect 1.0)")
    print(f"  Result[8, 0] = {result[8, 0].item():.1f}  (expect 2.0)")
    print(f"  Result[0, 8] = {result[0, 8].item():.1f}  (expect 0.0)")
    print(f"  Result[8, 8] = {result[8, 8].item():.1f}  (expect 0.0)")
    # If a1/a2 swap: Result[0, 8] would be 2.0 and Result[8, 0] would be 0.0
    swap_detected = (abs(result[8, 0].item()) < 0.5 and
                     abs(result[0, 8].item() - 2.0) < 0.5)
    print(f"  a1/a2 SWAP: {'YES' if swap_detected else 'NO'}")

    # --- Test 2: V = all ones ---
    # Row group: O = sum_k 1.0 * 1.0 (k<8) + 0 (k>=8) = 8.0
    # Row group+8: O = sum_k 2.0 * 1.0 (k<8) + 0 = 16.0
    v_ones = torch.full((16, 16), 0x38, dtype=torch.uint8, device="cuda")
    v_ones_u32 = v_ones.contiguous().view(-1).view(torch.int32)
    result2 = probe(v_ones_u32)
    print("\n=== Test 2: V=all 1.0, P non-uniform ===")
    print("Expected row 0: 8.0 everywhere, row 8: 16.0 everywhere")
    print(f"Result[0, :] = {result2[0].tolist()}")
    print(f"Result[8, :] = {result2[8].tolist()}")
    print(f"Result[0, 0] = {result2[0, 0].item():.1f} (expect 8.0)")
    print(f"Result[8, 0] = {result2[8, 0].item():.1f} (expect 16.0)")
    print(f"Result[0, 8] = {result2[0, 8].item():.1f} (expect 8.0)")
    print(f"Result[8, 8] = {result2[8, 8].item():.1f} (expect 16.0)")


if __name__ == "__main__":
    test_pv_mma_probe()
