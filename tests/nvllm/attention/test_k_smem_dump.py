#!/usr/bin/env python3
"""K SMEM dump: verify K data is loaded correctly into SMEM,
and trace which bytes the B operand loading actually reads.

Uses head_dim=256, same Q/K data as test_qk_hd256_probe.py.
Dumps K SMEM values at specific diagnostic offsets.
"""
import torch
import logging

logging.basicConfig(level=logging.WARNING)

TOKEN_MAP = {
    0:  (0x30, 0.5),    1:  (0x28, 0.25),   2:  (0x24, 0.125),
    3:  (0x40, 2.0),    4:  (0x38, 1.0),     5:  (0x2C, 0.375),
    6:  (0x26, 0.1875), 7:  (0x48, 4.0),     8:  (0x3C, 1.5),
    9:  (0x34, 0.75),   10: (0x44, 3.0),     11: (0x4C, 6.0),
    12: (0x2A, 0.3125), 13: (0x2E, 0.4375),  14: (0x3A, 1.25),
    15: (0x4A, 5.0),
}

BYTE_TO_TOK = {}
for tok, (byte_val, _) in TOKEN_MAP.items():
    BYTE_TO_TOK[byte_val] = tok


def test_k_smem():
    try:
        import cutlass
        from cutlass import cute
        from cutlass.cute.typing import Float32, Int32, Int64, Uint32
    except ImportError:
        print("CUTLASS not available")
        return

    from vllm.v1.attention.backends.cute_paged.kernel import (
        _ld_shared_b16, _ld_shared_b32, _st_shared_b32, _ld_shared_f32,
        _st_shared_f32, shared_ptr_to_i64,
    )

    class KSmemProbe:
        def __init__(self):
            self.num_threads = 128
            self.hd = 256
            self.q_bytes = 16 * 256 * 2
            self.k_bytes = 16 * 256
            self.smem_bytes = self.q_bytes + self.k_bytes
            self._compiled = None

        @cute.jit
        def _jit_launch(self, k_data, output, gx: Int32):
            self._kernel(k_data, output).launch(
                grid=[gx, Int32(1), Int32(1)],
                block=[self.num_threads, 1, 1],
                smem=self.smem_bytes,
            )

        @cute.kernel
        def _kernel(self, k_data, output):
            lane = cute.arch.lane_idx()
            warp = cute.arch.warp_idx()
            tid = warp * Int32(32) + lane
            group = lane >> Int32(2)
            sub = lane & Int32(3)

            smem = cute.arch.get_dyn_smem(cutlass.Uint8, alignment=128)
            k_smem = shared_ptr_to_i64(
                smem + Int32(self.q_bytes))
            hd = Int32(self.hd)

            # Load K into SMEM (same as hd256 probe)
            for _i in cutlass.range_constexpr(8):
                flat = tid * Int32(8) + Int32(_i)
                k_raw = k_data[flat]
                _st_shared_b32(k_smem + Int64(
                    flat * Int32(4)), k_raw)

            cute.arch.sync_threads()

            # Dump K SMEM: read first 16 bytes for each token row
            # and B operand addresses for diagnostic threads
            if warp == Int32(0):
                # Part A: Dump k_smem[tok*256] for tok=0..15
                # Just read 4 bytes at each token's start
                for _tok in cutlass.range_constexpr(16):
                    off = Int32(_tok) * hd
                    val = _ld_shared_b32(k_smem + Int64(off))
                    # Store as float for easy reading
                    output[lane * Int32(64) + Int32(_tok)] = Float32(val)

                # Part B: Dump B operand addresses for sub=0..3
                # Read what _ld_shared_b16 gets at the same offsets
                # the QK MMA would use
                for _s in cutlass.range_constexpr(4):
                    # Simulate: kv_row=group, k_start=0, sub=_s
                    k_off = group * hd + Int32(_s) * Int32(2)
                    raw_b16 = _ld_shared_b16(k_smem + Int64(k_off))
                    output[lane * Int32(64) + Int32(16) + Int32(_s)] = Float32(raw_b16)

                    # Also k_off + 8
                    raw_b16_hi = _ld_shared_b16(
                        k_smem + Int64(k_off + Int32(8)))
                    output[lane * Int32(64) + Int32(20) + Int32(_s)] = Float32(raw_b16_hi)

        def __call__(self, k_data):
            output = torch.zeros(32 * 64, dtype=torch.float32,
                                 device="cuda")
            if self._compiled is None:
                print("Compiling K SMEM probe...")
                self._compiled = cute.compile(
                    self._jit_launch, k_data, output, Int32(1),
                )
            self._compiled(k_data, output, Int32(1))
            return output.reshape(32, 64)

    probe = KSmemProbe()

    k_bytes = torch.zeros(16, 256, dtype=torch.uint8, device="cuda")
    for tok, (fp8_byte, _) in TOKEN_MAP.items():
        k_bytes[tok, :] = fp8_byte
    k_u32 = k_bytes.contiguous().view(-1).view(torch.int32)

    result = probe(k_u32)

    print("=" * 70)
    print("K SMEM DUMP — verify data and B operand addresses")
    print("=" * 70)

    # Part A: Check token data at SMEM starts
    print("\n--- Part A: First 4 bytes of each token in SMEM ---")
    # All threads in warp 0 read the same addresses, so just check thread 0
    for tok in range(16):
        raw_u32 = int(result[0, tok].item())
        # Extract 4 bytes
        b0 = raw_u32 & 0xFF
        b1 = (raw_u32 >> 8) & 0xFF
        b2 = (raw_u32 >> 16) & 0xFF
        b3 = (raw_u32 >> 24) & 0xFF
        expected = TOKEN_MAP[tok][0]
        match = "OK" if b0 == expected else f"WRONG (exp 0x{expected:02X})"
        tok_found = BYTE_TO_TOK.get(b0, "??")
        print(f"  k_smem[tok={tok:2d} * 256] = "
              f"[0x{b0:02X}, 0x{b1:02X}, 0x{b2:02X}, 0x{b3:02X}] "
              f"→ tok {tok_found} {match}")

    # Part B: Check what B operand loads for each (group, simulated_sub)
    print("\n--- Part B: B operand _ld_shared_b16 values ---")
    print("  For each thread group, what does sub=0..3 read?")
    for g in [0, 2, 6]:  # diagnostic groups
        lane_id = g * 4  # sub=0 thread for this group
        print(f"\n  Group={g} (kv_row={g}):")
        for s in range(4):
            raw_lo = int(result[lane_id, 16 + s].item())
            raw_hi = int(result[lane_id, 20 + s].item())
            # _ld_shared_b16 returns Uint32 with low 16 bits
            b_lo_0 = raw_lo & 0xFF
            b_lo_1 = (raw_lo >> 8) & 0xFF
            b_hi_0 = raw_hi & 0xFF
            b_hi_1 = (raw_hi >> 8) & 0xFF
            tok_lo0 = BYTE_TO_TOK.get(b_lo_0, "??")
            tok_lo1 = BYTE_TO_TOK.get(b_lo_1, "??")
            off = g * 256 + s * 2
            print(f"    sub={s}: offset={off:5d} → "
                  f"lo=[0x{b_lo_0:02X}(t{tok_lo0}), 0x{b_lo_1:02X}(t{tok_lo1})] "
                  f"hi=[0x{b_hi_0:02X}, 0x{b_hi_1:02X}]")


if __name__ == "__main__":
    test_k_smem()
