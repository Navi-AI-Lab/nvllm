#!/usr/bin/env python3
"""QK MMA probe: deterministic Q and K, dump raw D-fragment values per-thread.

The key question: for thread (group, sub), does s0 = S[group, sub*2]
(group=Q-row, sub*2=KV-token) or is it transposed?

Setup:
  Q[row, :] = 1.0 for all rows (BF16)  — uniform Q
  K[tok=0, :] = 1.0 (FP8 E4M3)          — token 0 unique
  K[tok=1..15, :] = 0.5 (FP8 E4M3)      — rest half-scale

  S[row, tok=0] = 16.0  (sum of 16 × 1.0×1.0)
  S[row, tok>0] = 8.0   (sum of 16 × 1.0×0.5)

  For thread (group=1, sub=0):
    Correct mapping  (s0 = S[group=1, sub*2=0]): s0 = S[1, 0] = 16.0
    Transposed       (s0 = S[sub*2=0, group=1]): s0 = S[0, 1] =  8.0

Volume-mount into container and run:
  python /app/nvllm/tests/nvllm/attention/test_qk_mma_probe.py
"""
import torch
import logging

logging.basicConfig(level=logging.WARNING)


def test_qk_mma_probe():
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

    from vllm.v1.attention.backends.cute_paged.kernel import (
        _mma_m16n8k16_f32,
        bf16_mma_m16n16k16_f32,
        _ld_shared_b16,
        _ld_shared_b32,
        _st_shared_b32,
        _st_shared_f32,
        _ld_shared_f32,
        _pack_lo16,
        fp8x4_e4m3_to_bfloat2x2,
        _st_shared_b16_from_u32,
        shared_ptr_to_i64,
    )

    class QKProbe:
        """Standalone QK MMA probe: deterministic Q and K, dump raw fragments.

        Uses head_dim=16 (single K-dim MMA iteration) to isolate the
        fragment layout question. The layout is the same regardless of
        the number of K-dim iterations.

        SMEM layout:
          Q: 16 rows × 16 cols × 2 bytes BF16 = 512 bytes
          K: 16 tokens × 16 cols × 1 byte FP8  = 256 bytes
          (total 768 bytes)

        Output: per-thread fragment dump, 32 threads × 8 values = 256 FP32
        """
        def __init__(self):
            self.num_threads = 128  # 4 warps (only warp 0 active)
            self.hd = 16
            self.q_bytes = 16 * 16 * 2   # 512
            self.k_bytes = 16 * 16        # 256
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
            """Load Q (BF16) and K (FP8) into SMEM, run QK MMA,
            dump raw s0..s7 per warp-0 thread to global output."""
            lane = cute.arch.lane_idx()
            warp = cute.arch.warp_idx()
            tid = warp * Int32(32) + lane
            group = lane >> Int32(2)   # 0..7
            sub = lane & Int32(3)      # 0..3

            smem = cute.arch.get_dyn_smem(cutlass.Uint8, alignment=128)
            q_smem = shared_ptr_to_i64(smem)
            k_smem = shared_ptr_to_i64(smem + Int32(self.q_bytes))

            hd = Int32(self.hd)  # 16

            # === Load Q into SMEM: 512 bytes = 128 uint32 words ===
            # 128 threads → 1 word each
            if tid < Int32(128):
                q_raw = q_data[tid]
                _st_shared_b32(q_smem + Int64(tid * Int32(4)), q_raw)

            # === Load K into SMEM: 256 bytes = 64 uint32 words ===
            # First 64 threads handle this
            if tid < Int32(64):
                k_raw = k_data[tid]
                _st_shared_b32(k_smem + Int64(tid * Int32(4)), k_raw)

            cute.arch.sync_threads()

            if warp == Int32(0):
                # === QK MMA: exact same code as production kernel ===
                s0 = Float32(0.0)
                s1 = Float32(0.0)
                s2 = Float32(0.0)
                s3 = Float32(0.0)
                s4 = Float32(0.0)
                s5 = Float32(0.0)
                s6 = Float32(0.0)
                s7 = Float32(0.0)

                # Single K-dim iteration (head_dim=16)
                k_start = Int32(0)

                # Q fragments (A operand) — same as production kernel
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

                # K fragments (B operand) — same as production kernel
                # First 8 KV tokens via n_t = group
                n_t = group
                kv_row_0 = n_t  # warp_kv_start = 0
                k_off_0a = (kv_row_0 * hd + k_start
                            + sub * Int32(2))
                k_raw_0a = _ld_shared_b16(
                    k_smem + Int64(k_off_0a))
                k_raw_0b = _ld_shared_b16(
                    k_smem + Int64(k_off_0a + Int32(8)))
                k_packed_0 = _pack_lo16(k_raw_0a, k_raw_0b)
                b0, b1 = fp8x4_e4m3_to_bfloat2x2(k_packed_0)

                # Next 8 KV tokens via n_t + 8
                kv_row_1 = n_t + Int32(8)
                k_off_1a = (kv_row_1 * hd + k_start
                            + sub * Int32(2))
                k_raw_1a = _ld_shared_b16(
                    k_smem + Int64(k_off_1a))
                k_raw_1b = _ld_shared_b16(
                    k_smem + Int64(k_off_1a + Int32(8)))
                k_packed_1 = _pack_lo16(k_raw_1a, k_raw_1b)
                b2, b3 = fp8x4_e4m3_to_bfloat2x2(k_packed_1)

                # QK MMA
                (s0, s1, s2, s3,
                 s4, s5, s6, s7) = bf16_mma_m16n16k16_f32(
                    s0, s1, s2, s3, s4, s5, s6, s7,
                    a0, a1, a2, a3,
                    b0, b1, b2, b3)

                # === Dump raw fragments to global output ===
                # output[lane * 8 + 0..7] = s0..s7
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
            output = torch.zeros(256, dtype=torch.float32,
                                 device="cuda")
            if self._compiled is None:
                print("Compiling QK MMA probe...")
                self._compiled = cute.compile(
                    self._jit_launch,
                    q_data, k_data, output, Int32(1),
                )
            self._compiled(q_data, k_data, output, Int32(1))
            return output.reshape(32, 8)

    probe = QKProbe()

    # === Prepare Q: all 1.0 BF16 (16 rows × 16 cols) ===
    q_bf16 = torch.ones(16, 16, dtype=torch.bfloat16, device="cuda")
    q_u32 = q_bf16.contiguous().view(torch.int16).view(torch.int32)

    # === Prepare K: token 0 = 1.0, tokens 1-15 = 0.5 (FP8 E4M3) ===
    k_bytes = torch.full((16, 16), 0x30, dtype=torch.uint8, device="cuda")
    k_bytes[0, :] = 0x38  # E4M3 1.0
    k_u32 = k_bytes.contiguous().view(-1).view(torch.int32)

    # === Run probe ===
    result = probe(q_u32, k_u32)

    # === Analysis ===
    print("=" * 70)
    print("QK MMA FRAGMENT PROBE")
    print("=" * 70)
    print()
    print("Setup: Q = 1.0 everywhere, K[tok=0] = 1.0, K[tok=1..15] = 0.5")
    print("Expected: S[row, tok=0] = 16.0, S[row, tok>0] = 8.0")
    print()

    # Key diagnostic threads
    print("--- Key diagnostic values ---")
    diag_threads = [
        (0, 0, "baseline: both mappings give S[0,0]=16.0"),
        (1, 0, "CRITICAL: correct→S[1,0]=16.0, transposed→S[0,1]=8.0"),
        (0, 1, "CRITICAL: correct→S[0,2]=8.0, transposed→S[2,0]=16.0"),
        (2, 0, "correct→S[2,0]=16.0, transposed→S[0,2]=8.0"),
        (0, 2, "correct→S[0,4]=8.0, transposed→S[4,0]=16.0"),
    ]
    for g, s, desc in diag_threads:
        lane = g * 4 + s
        val = result[lane, 0].item()
        print(f"  Thread (group={g}, sub={s}): s0 = {val:.1f}  [{desc}]")

    print()

    # Determine mapping
    s0_g1_s0 = result[1 * 4 + 0, 0].item()  # group=1, sub=0
    s0_g0_s1 = result[0 * 4 + 1, 0].item()  # group=0, sub=1

    if abs(s0_g1_s0 - 16.0) < 1.0 and abs(s0_g0_s1 - 8.0) < 1.0:
        print("RESULT: D-fragment mapping is CORRECT")
        print("  s0 = S[group, sub*2]  (group → Q-row, sub*2 → KV-token)")
        print("  Bug is NOT in the fragment mapping — look elsewhere")
    elif abs(s0_g1_s0 - 8.0) < 1.0 and abs(s0_g0_s1 - 16.0) < 1.0:
        print("RESULT: D-fragment mapping is TRANSPOSED")
        print("  s0 = S[sub*2, group]  (sub*2 → Q-row, group → KV-token)")
        print("  FIX: swap group↔sub*2 in softmax row tracking + causal mask")
    else:
        print(f"RESULT: UNEXPECTED VALUES — neither mapping matches")
        print(f"  (group=1,sub=0).s0 = {s0_g1_s0:.4f}  (expected 16.0 or 8.0)")
        print(f"  (group=0,sub=1).s0 = {s0_g0_s1:.4f}  (expected 8.0 or 16.0)")

    # Dump full fragment table for warp 0
    print()
    print("--- Full per-thread fragment dump (warp 0, 32 threads) ---")
    print(f"{'Lane':>4} {'group':>5} {'sub':>3} | "
          f"{'s0':>7} {'s1':>7} {'s2':>7} {'s3':>7} | "
          f"{'s4':>7} {'s5':>7} {'s6':>7} {'s7':>7}")
    print("-" * 90)
    for lane_id in range(32):
        g = lane_id >> 2
        s = lane_id & 3
        vals = result[lane_id].tolist()
        print(f"{lane_id:4d} {g:5d} {s:3d} | "
              f"{vals[0]:7.1f} {vals[1]:7.1f} {vals[2]:7.1f} {vals[3]:7.1f} | "
              f"{vals[4]:7.1f} {vals[5]:7.1f} {vals[6]:7.1f} {vals[7]:7.1f}")

    # Reconstruct S matrix under both mappings
    print()
    print("--- Reconstructed S matrix (assumed correct mapping: s0=S[group,sub*2]) ---")
    s_correct = torch.zeros(16, 16)
    for lane_id in range(32):
        g = lane_id >> 2
        s = lane_id & 3
        vals = result[lane_id]
        s_correct[g, s * 2] = vals[0]
        s_correct[g, s * 2 + 1] = vals[1]
        s_correct[g + 8, s * 2] = vals[2]
        s_correct[g + 8, s * 2 + 1] = vals[3]
        s_correct[g, s * 2 + 8] = vals[4]
        s_correct[g, s * 2 + 9] = vals[5]
        s_correct[g + 8, s * 2 + 8] = vals[6]
        s_correct[g + 8, s * 2 + 9] = vals[7]

    print("  Rows = Q-rows, Cols = KV-tokens")
    print("  Expect: col 0 = 16.0, cols 1-15 = 8.0 (same for all rows)")
    for r in range(16):
        row_str = " ".join(f"{s_correct[r, c].item():5.1f}" for c in range(16))
        marker = " ← Q-row 0" if r == 0 else ""
        print(f"  Row {r:2d}: {row_str}{marker}")

    print()
    print("--- Reconstructed S matrix (transposed mapping: s0=S[sub*2,group]) ---")
    s_transposed = torch.zeros(16, 16)
    for lane_id in range(32):
        g = lane_id >> 2
        s = lane_id & 3
        vals = result[lane_id]
        # Swap: group→KV-token, sub*2→Q-row
        s_transposed[s * 2, g] = vals[0]
        s_transposed[s * 2 + 1, g] = vals[1]
        s_transposed[s * 2, g + 8] = vals[2]
        s_transposed[s * 2 + 1, g + 8] = vals[3]
        s_transposed[s * 2 + 8, g] = vals[4]
        s_transposed[s * 2 + 9, g] = vals[5]
        s_transposed[s * 2 + 8, g + 8] = vals[6]
        s_transposed[s * 2 + 9, g + 8] = vals[7]

    print("  Rows = Q-rows, Cols = KV-tokens")
    print("  Expect: col 0 = 16.0, cols 1-15 = 8.0 (same for all rows)")
    for r in range(16):
        row_str = " ".join(f"{s_transposed[r, c].item():5.1f}" for c in range(16))
        marker = " ← Q-row 0" if r == 0 else ""
        print(f"  Row {r:2d}: {row_str}{marker}")


if __name__ == "__main__":
    test_qk_mma_probe()
