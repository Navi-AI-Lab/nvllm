#!/usr/bin/env python3
"""QK MMA probe with head_dim=256 and unique K values per token.

Isolates whether multi-iteration QK MMA (16 K-dim iterations) produces
correct scores by using K values that uniquely identify each token.

Setup:
  Q[row, :] = 1.0 (BF16) for all rows
  K[tok, :] = FP8_VALUE[tok] for all dims (unique per token)

  Expected: S[row, tok] = 256 * Q_val * K_val[tok] = 256 * K_val[tok]

  Token values (FP8 E4M3):
    tok 0:  0.5   (0x30) → S = 128.0
    tok 1:  0.25  (0x28) → S = 64.0
    tok 2:  0.125 (0x24) → S = 32.0
    tok 3:  2.0   (0x40) → S = 512.0
    tok 4:  1.0   (0x38) → S = 256.0
    tok 5:  0.375 (0x2C) → S = 96.0
    tok 6:  0.1875(0x26) → S = 48.0
    tok 7:  4.0   (0x48) → S = 1024.0
    tok 8:  1.5   (0x3C) → S = 384.0
    tok 9:  0.75  (0x34) → S = 192.0
    tok 10: 3.0   (0x44) → S = 768.0
    tok 11: 6.0   (0x4C) → S = 1536.0
    tok 12: 0.3125(0x2A) → S = 80.0
    tok 13: 0.4375(0x2E) → S = 112.0
    tok 14: 1.25  (0x3A) → S = 320.0
    tok 15: 5.0   (0x4A) → S = 1280.0

  Each S value is unique → we can identify which token each
  fragment position actually computes.

Volume-mount into container and run:
  python /app/nvllm/tests/nvllm/attention/test_qk_hd256_probe.py
"""
import torch
import logging

logging.basicConfig(level=logging.WARNING)


# Token → FP8 E4M3 byte value → expected dot product
TOKEN_MAP = {
    0:  (0x30, 0.5,    128.0),
    1:  (0x28, 0.25,   64.0),
    2:  (0x24, 0.125,  32.0),
    3:  (0x40, 2.0,    512.0),
    4:  (0x38, 1.0,    256.0),
    5:  (0x2C, 0.375,  96.0),
    6:  (0x26, 0.1875, 48.0),
    7:  (0x48, 4.0,    1024.0),
    8:  (0x3C, 1.5,    384.0),
    9:  (0x34, 0.75,   192.0),
    10: (0x44, 3.0,    768.0),
    11: (0x4C, 6.0,    1536.0),
    12: (0x2A, 0.3125, 80.0),
    13: (0x2E, 0.4375, 112.0),
    14: (0x3A, 1.25,   320.0),
    15: (0x4A, 5.0,    1280.0),
}


def score_to_token(score, tol=0.15):
    """Find which token a score value corresponds to."""
    best_tok, best_err = -1, 999999
    for tok, (_, _, expected) in TOKEN_MAP.items():
        err = abs(score - expected) / max(expected, 1.0)
        if err < best_err:
            best_err = err
            best_tok = tok
    if best_err < tol:
        return best_tok
    return None


def test_qk_hd256():
    try:
        import cutlass
        from cutlass import cute
        from cutlass.cute.typing import (
            BFloat16, Float32, Int32, Int64, Uint32,
        )
    except ImportError:
        print("CUTLASS not available, skipping")
        return

    from vllm.v1.attention.backends.cute_paged.kernel import (
        bf16_mma_m16n16k16_f32,
        _ld_shared_b16,
        _ld_shared_b32,
        _st_shared_b32,
        _st_shared_f32,
        _ld_shared_f32,
        _pack_lo16,
        fp8x4_e4m3_to_bfloat2x2,
        _st_shared_b16_from_u32,
        _cvt_2f32_to_bf16x2,
        shared_ptr_to_i64,
    )

    class QKProbeHD256:
        """QK MMA with head_dim=256, unique K per token."""
        def __init__(self):
            self.num_threads = 128
            self.hd = 256
            self.num_mma_d = 16  # 256/16
            self.q_bytes = 16 * 256 * 2   # BF16
            self.k_bytes = 16 * 256       # FP8
            self.smem_bytes = self.q_bytes + self.k_bytes
            self._compiled = None

        @cute.jit
        def _jit_launch(self, q_data, k_data, output, grid_x: Int32):
            self._kernel(q_data, k_data, output).launch(
                grid=[grid_x, Int32(1), Int32(1)],
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
            hd = Int32(self.hd)

            # Load Q: 16*256*2=8192 bytes = 2048 uint32
            # 128 threads × 16 words each
            for _i in cutlass.range_constexpr(16):
                flat = tid * Int32(16) + Int32(_i)
                q_raw = q_data[flat]
                _st_shared_b32(q_smem + Int64(flat * Int32(4)), q_raw)

            # Load K: 16*256=4096 bytes = 1024 uint32
            # 128 threads × 8 words each
            for _i in cutlass.range_constexpr(8):
                flat = tid * Int32(8) + Int32(_i)
                k_raw = k_data[flat]
                _st_shared_b32(k_smem + Int64(flat * Int32(4)), k_raw)

            cute.arch.sync_threads()

            if warp == Int32(0):
                # QK MMA: 16 K-dim iterations (head_dim=256)
                s0 = Float32(0.0)
                s1 = Float32(0.0)
                s2 = Float32(0.0)
                s3 = Float32(0.0)
                s4 = Float32(0.0)
                s5 = Float32(0.0)
                s6 = Float32(0.0)
                s7 = Float32(0.0)

                for _kd in cutlass.range_constexpr(16):
                    k_start = Int32(_kd * 16)

                    # Q fragments (A operand)
                    q_byte_a0 = (group * hd + k_start
                                 + sub * Int32(2)) * Int32(2)
                    a0 = _ld_shared_b32(
                        q_smem + Int64(q_byte_a0))
                    a1 = _ld_shared_b32(
                        q_smem + Int64(q_byte_a0 + Int32(16)))
                    q_byte_a2 = ((group + Int32(8)) * hd
                                 + k_start
                                 + sub * Int32(2)) * Int32(2)
                    a2 = _ld_shared_b32(
                        q_smem + Int64(q_byte_a2))
                    a3 = _ld_shared_b32(
                        q_smem + Int64(q_byte_a2 + Int32(16)))

                    # K fragments (B operand)
                    n_t = group
                    kv_row_0 = n_t
                    k_off_0a = (kv_row_0 * hd + k_start
                                + sub * Int32(2))
                    k_raw_0a = _ld_shared_b16(
                        k_smem + Int64(k_off_0a))
                    k_raw_0b = _ld_shared_b16(
                        k_smem + Int64(k_off_0a + Int32(8)))
                    k_packed_0 = _pack_lo16(k_raw_0a, k_raw_0b)
                    b0, b1 = fp8x4_e4m3_to_bfloat2x2(k_packed_0)

                    kv_row_1 = n_t + Int32(8)
                    k_off_1a = (kv_row_1 * hd + k_start
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
                        a0, a1, a2, a3,
                        b0, b1, b2, b3)

                # Dump raw scores per thread
                base = lane * Int32(8)
                output[base + Int32(0)] = s0
                output[base + Int32(1)] = s1
                output[base + Int32(2)] = s2
                output[base + Int32(3)] = s3
                output[base + Int32(4)] = s4
                output[base + Int32(5)] = s5
                output[base + Int32(6)] = s6
                output[base + Int32(7)] = s7

        def __call__(self, q_data, k_data):
            output = torch.zeros(256, dtype=torch.float32, device="cuda")
            if self._compiled is None:
                print("Compiling QK hd256 probe...")
                self._compiled = cute.compile(
                    self._jit_launch,
                    q_data, k_data, output, Int32(1),
                )
            self._compiled(q_data, k_data, output, Int32(1))
            return output.reshape(32, 8)

    probe = QKProbeHD256()

    # Q = all 1.0 BF16
    q_bf16 = torch.ones(16, 256, dtype=torch.bfloat16, device="cuda")
    q_u32 = q_bf16.contiguous().view(torch.int16).view(torch.int32)

    # K: unique FP8 value per token
    k_bytes = torch.zeros(16, 256, dtype=torch.uint8, device="cuda")
    for tok, (fp8_byte, _, _) in TOKEN_MAP.items():
        k_bytes[tok, :] = fp8_byte
    k_u32 = k_bytes.contiguous().view(-1).view(torch.int32)

    result = probe(q_u32, k_u32)

    print("=" * 70)
    print("QK MMA PROBE — head_dim=256, unique K per token")
    print("=" * 70)
    print()

    # Analyze key threads
    print("--- Fragment mapping verification ---")
    print(f"{'Lane':>4} {'g':>2} {'s':>2} | "
          f"{'s0':>8} {'tok':>4} | {'s1':>8} {'tok':>4} | "
          f"{'s4':>8} {'tok':>4} | {'s5':>8} {'tok':>4}")
    print("-" * 80)

    mapping_correct = True
    for lane_id in range(32):
        g = lane_id >> 2
        s = lane_id & 3
        vals = result[lane_id].tolist()

        # Expected tokens:
        # s0 → tok sub*2,  s1 → tok sub*2+1
        # s4 → tok sub*2+8, s5 → tok sub*2+9
        exp_toks = [s*2, s*2+1, s*2+8, s*2+9]
        act_toks = [score_to_token(vals[i]) for i in [0, 1, 4, 5]]

        markers = []
        for exp, act in zip(exp_toks, act_toks):
            if act == exp:
                markers.append("  ")
            else:
                markers.append("!!")
                mapping_correct = False

        print(f"{lane_id:4d} {g:2d} {s:2d} | "
              f"{vals[0]:8.1f} {str(act_toks[0]):>4}{markers[0]} | "
              f"{vals[1]:8.1f} {str(act_toks[1]):>4}{markers[1]} | "
              f"{vals[4]:8.1f} {str(act_toks[2]):>4}{markers[2]} | "
              f"{vals[5]:8.1f} {str(act_toks[3]):>4}{markers[3]}")

    print()
    if mapping_correct:
        print("RESULT: All fragment positions map to CORRECT tokens")
        print("  QK MMA is correct even with hd=256")
    else:
        print("RESULT: Fragment positions map to WRONG tokens!")
        print("  Bug is in multi-iteration QK MMA accumulation or K SMEM loading")

        # Build actual mapping
        print()
        print("--- Actual token mapping (s0 column) ---")
        for lane_id in range(min(8, 32)):
            g = lane_id >> 2
            s = lane_id & 3
            val = result[lane_id, 0].item()
            tok = score_to_token(val)
            exp = s * 2
            print(f"  Thread (g={g}, s={s}): s0={val:.1f} → tok {tok} "
                  f"(expected tok {exp})")


if __name__ == "__main__":
    test_qk_hd256()
