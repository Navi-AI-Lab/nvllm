#!/usr/bin/env python3
"""QK MMA iteration sweep: test with head_dim=16,32,48,64,256.
Find where multi-iteration accumulation breaks the fragment mapping.
"""
import torch
import sys
import logging

logging.basicConfig(level=logging.WARNING)
sys.path.insert(0, "/app/nvllm")

try:
    import cutlass
    from cutlass import cute
    from cutlass.cute.typing import Float32, Int32, Int64, Uint32
except ImportError:
    print("CUTLASS not available")
    sys.exit(0)

from vllm.v1.attention.backends.cute_paged.kernel import (
    bf16_mma_m16n16k16_f32, _ld_shared_b16, _ld_shared_b32,
    _st_shared_b32, _pack_lo16, fp8x4_e4m3_to_bfloat2x2,
    shared_ptr_to_i64,
)

TOKEN_FP8 = {
    0: 0x30, 1: 0x28, 2: 0x24, 3: 0x40, 4: 0x38, 5: 0x2C,
    6: 0x26, 7: 0x48, 8: 0x3C, 9: 0x34, 10: 0x44, 11: 0x4C,
    12: 0x2A, 13: 0x2E, 14: 0x3A, 15: 0x4A,
}
TOKEN_VAL = {
    0: 0.5, 1: 0.25, 2: 0.125, 3: 2.0, 4: 1.0, 5: 0.375,
    6: 0.1875, 7: 4.0, 8: 1.5, 9: 0.75, 10: 3.0, 11: 6.0,
    12: 0.3125, 13: 0.4375, 14: 1.25, 15: 5.0,
}


def make_probe(hd):
    """Build a QK probe for a specific head_dim."""
    num_iters = hd // 16

    class Probe:
        def __init__(self):
            self.num_threads = 128
            self.hd = hd
            self.q_bytes = 16 * hd * 2
            self.k_bytes = 16 * hd
            self.smem_bytes = self.q_bytes + self.k_bytes
            self._compiled = None

        @cute.jit
        def _jit_launch(self, q_data, k_data, output, gx: Int32):
            self._kernel(q_data, k_data, output).launch(
                grid=[gx, Int32(1), Int32(1)],
                block=[self.num_threads, 1, 1],
                smem=self.smem_bytes,
            )

        @cute.kernel
        def _kernel(self, q_data, k_data, output):
            lane = cute.arch.lane_idx()
            warp = cute.arch.warp_idx()
            tid = warp * Int32(32) + lane
            group = lane >> Int32(2)
            sub = lane & Int32(3)

            smem = cute.arch.get_dyn_smem(cutlass.Uint8, alignment=128)
            q_smem = shared_ptr_to_i64(smem)
            k_smem = shared_ptr_to_i64(smem + Int32(self.q_bytes))
            hd_val = Int32(self.hd)

            # Load Q and K into SMEM
            total_q_words = self.q_bytes // 4
            total_k_words = self.k_bytes // 4
            # Use all 128 threads, each handles ceil(total/128) words
            q_per_thr = (total_q_words + 127) // 128
            k_per_thr = (total_k_words + 127) // 128

            for _i in cutlass.range_constexpr(
                (16 * hd * 2 // 4 + 127) // 128
            ):
                flat = tid * Int32(q_per_thr) + Int32(_i)
                if flat < Int32(total_q_words):
                    _st_shared_b32(q_smem + Int64(flat * Int32(4)),
                                   q_data[flat])

            for _i in cutlass.range_constexpr(
                (16 * hd // 4 + 127) // 128
            ):
                flat = tid * Int32(k_per_thr) + Int32(_i)
                if flat < Int32(total_k_words):
                    _st_shared_b32(k_smem + Int64(flat * Int32(4)),
                                   k_data[flat])

            cute.arch.sync_threads()

            if warp == Int32(0):
                s0 = Float32(0.0)
                s1 = Float32(0.0)
                s2 = Float32(0.0)
                s3 = Float32(0.0)
                s4 = Float32(0.0)
                s5 = Float32(0.0)
                s6 = Float32(0.0)
                s7 = Float32(0.0)

                for _kd in cutlass.range_constexpr(num_iters):
                    k_start = Int32(_kd * 16)
                    q_byte_a0 = (group * hd_val + k_start
                                 + sub * Int32(2)) * Int32(2)
                    a0 = _ld_shared_b32(q_smem + Int64(q_byte_a0))
                    a1 = _ld_shared_b32(
                        q_smem + Int64(q_byte_a0 + Int32(16)))
                    q_byte_a2 = ((group + Int32(8)) * hd_val
                                 + k_start
                                 + sub * Int32(2)) * Int32(2)
                    a2 = _ld_shared_b32(q_smem + Int64(q_byte_a2))
                    a3 = _ld_shared_b32(
                        q_smem + Int64(q_byte_a2 + Int32(16)))

                    n_t = group
                    k_off_0a = (n_t * hd_val + k_start
                                + sub * Int32(2))
                    k_raw_0a = _ld_shared_b16(
                        k_smem + Int64(k_off_0a))
                    k_raw_0b = _ld_shared_b16(
                        k_smem + Int64(k_off_0a + Int32(8)))
                    k_packed_0 = _pack_lo16(k_raw_0a, k_raw_0b)
                    b0, b1 = fp8x4_e4m3_to_bfloat2x2(k_packed_0)

                    kv_row_1 = n_t + Int32(8)
                    k_off_1a = (kv_row_1 * hd_val + k_start
                                + sub * Int32(2))
                    k_raw_1a = _ld_shared_b16(
                        k_smem + Int64(k_off_1a))
                    k_raw_1b = _ld_shared_b16(
                        k_smem + Int64(k_off_1a + Int32(8)))
                    k_packed_1 = _pack_lo16(k_raw_1a, k_raw_1b)
                    b2, b3 = fp8x4_e4m3_to_bfloat2x2(k_packed_1)

                    (s0, s1, s2, s3,
                     s4, s5, s6, s7) = bf16_mma_m16n16k16_f32(
                        s0, s1, s2, s3, s4, s5, s6, s7,
                        a0, a1, a2, a3, b0, b1, b2, b3)

                base = lane * Int32(8)
                output[base] = s0
                output[base + Int32(1)] = s1
                output[base + Int32(4)] = s4
                output[base + Int32(5)] = s5

        def __call__(self, q_data, k_data):
            output = torch.zeros(256, dtype=torch.float32, device="cuda")
            if self._compiled is None:
                print(f"  Compiling hd={self.hd}...")
                self._compiled = cute.compile(
                    self._jit_launch, q_data, k_data, output, Int32(1))
            self._compiled(q_data, k_data, output, Int32(1))
            return output.reshape(32, 8)

    return Probe()


def main():
    print("=" * 70)
    print("QK MMA iteration sweep: where does multi-iter break?")
    print("=" * 70)

    for test_hd in [16, 32, 48, 64, 128, 256]:
        probe = make_probe(test_hd)
        q = torch.ones(16, test_hd, dtype=torch.bfloat16, device="cuda")
        q_u32 = q.contiguous().view(torch.int16).view(torch.int32)
        k = torch.zeros(16, test_hd, dtype=torch.uint8, device="cuda")
        for tok, byte_val in TOKEN_FP8.items():
            k[tok, :] = byte_val
        k_u32 = k.contiguous().view(-1).view(torch.int32)

        r = probe(q_u32, k_u32)

        # Check s0 for sub=0..3 (group=0)
        issues = []
        for s in range(4):
            lane = s  # group=0
            val = r[lane, 0].item()
            exp_tok = s * 2
            exp_val = test_hd * TOKEN_VAL[exp_tok]
            ok = abs(val - exp_val) / max(exp_val, 0.01) < 0.05
            if not ok:
                # Find which token it matches
                best_tok = min(range(16),
                               key=lambda t: abs(val - test_hd * TOKEN_VAL[t]))
                issues.append(f"sub={s}:got_tok{best_tok}(exp{exp_tok})")

        status = "OK" if not issues else "WRONG: " + ", ".join(issues)
        print(f"  hd={test_hd:3d} ({test_hd//16:2d} iters): {status}")

        if issues:
            # Print all s0 values for sub=0..3
            for s in range(4):
                val = r[s, 0].item()
                exp = test_hd * TOKEN_VAL[s * 2]
                print(f"    sub={s}: s0={val:8.1f} (exp {exp:7.1f})")


if __name__ == "__main__":
    main()
