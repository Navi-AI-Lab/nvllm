#!/usr/bin/env python3
"""Diagnostic: isolate PV MMA output vs sync_o write vs reduction.

Standalone kernel that:
1. Loads one-hot V into SMEM (cooperative, same layout as production)
2. Uses HARDCODED P values (no QK, no softmax) -- P_u[tok] = tok + 1.0
3. Does PV MMA for _md_idx=0 (head_dim positions 0-15)
4. Writes o0..o7 to sync_o buffer (production sync_o write pattern)
5. Dumps BOTH per-thread PV output AND sync_o buffer to global memory

Expected math (one-hot V, P_u[tok] = tok+1):
  D[m, n] = sum_k(P[m,k] * V[k,n])
  V is one-hot: V[k, n] = 1.0 iff k == n (for k < 6)
  So D[m, n] = P[m, n] = P_u[n] = n + 1.0  for n < 6, else 0

Since all rows m share the same P values in this setup:
  D[*, 0] = 1.0, D[*, 1] = 2.0, ..., D[*, 5] = 6.0, D[*, 6..15] = 0.0

If per-thread dump is correct but sync_o is wrong: sync_o write is broken.
If per-thread dump is wrong: MMA itself or P packing is broken.

Volume-mount into container and run:
  docker cp tests/nvllm/attention/test_sync_o_dump.py nvllm:/app/nvllm/tests/nvllm/attention/
  docker exec nvllm python /app/nvllm/tests/nvllm/attention/test_sync_o_dump.py
"""
import sys
sys.path.insert(0, "/app/nvllm")

import torch
import logging

logging.basicConfig(level=logging.WARNING)

try:
    import cutlass
    from cutlass import cute
    from cutlass.cute.typing import Float32, Int32, Int64, Uint32
except ImportError:
    print("CUTLASS not available, skipping")
    sys.exit(0)

from vllm.v1.attention.backends.cute_paged.kernel import (
    shared_ptr_to_i64,
    _ld_global_b32,
    _ld_shared_b32,
    _st_shared_b32,
    _st_shared_f32,
    _ld_shared_f32,
    _extract_byte_from_b32,
    _pack_4bytes,
    fp8x4_e4m3_to_bfloat2x2,
    _cvt_2f32_to_bf16x2,
    bf16_mma_m16n16k16_f32,
)


class SyncOIsolationProbe:
    """Standalone kernel: hardcoded P + one-hot V -> PV MMA -> sync_o write.

    Only warp 0 runs (tokens 0-15), _md_idx = 0.
    Dumps:
      - pv_dump: per-thread o0..o7 (32 lanes x 8 = 256 floats)
      - sync_o_dump: sync_o buffer contents (16 rows x 16 cols = 256 floats)
    """

    def __init__(self):
        self.num_threads = 128       # 4 warps x 32 lanes
        self.head_dim = 256
        self.cta_kv = 64             # page_size
        self.cta_q = 16

        # SMEM layout: V buffer + sync_o buffer
        # V: 64 tokens x 256 dims x 1 byte (FP8) = 16384 bytes
        self.v_bytes = self.cta_kv * self.head_dim
        # sync_o: 4 warps x 16 rows x 16 cols x 4 bytes = 4096 bytes
        self.sync_o_bytes = 4 * self.cta_q * 16 * 4
        self.smem_bytes = self.v_bytes + self.sync_o_bytes
        self._compiled = None

    @cute.jit
    def _jit_launch(self, v_ptr: Int64, pv_dump, sync_o_dump,
                    gx: Int32):
        self._kernel(v_ptr, pv_dump, sync_o_dump).launch(
            grid=[gx, Int32(1), Int32(1)],
            block=[self.num_threads, 1, 1],
            smem=self.smem_bytes,
        )

    @cute.kernel
    def _kernel(self, v_ptr: Int64, pv_dump, sync_o_dump):
        """Standalone PV MMA + sync_o write diagnostic.

        All 128 threads cooperate on V load.
        Only warp 0 does PV MMA + sync_o write (tokens 0-15).
        All threads dump results.
        """
        lane = cute.arch.lane_idx()
        warp = cute.arch.warp_idx()
        tid = warp * Int32(32) + lane
        group = lane >> Int32(2)    # 0..7
        sub = lane & Int32(3)       # 0..3

        hd = Int32(self.head_dim)   # 256

        # === SMEM pointers ===
        smem = cute.arch.get_dyn_smem(cutlass.Uint8, alignment=128)
        v_smem = shared_ptr_to_i64(smem)
        sync_o = shared_ptr_to_i64(smem + Int32(self.v_bytes))

        # === Phase 1: Cooperative V load (all 128 threads) ===
        # V cache is row-major: [64 tokens, 256 dims], FP8 (1 byte each)
        # Total = 16384 bytes. 128 threads, 4 bytes/iter = 128 iterations.
        elems_per_thr_4 = Int32(
            self.cta_kv * self.head_dim // 4 // self.num_threads)
        for _i in cutlass.range_constexpr(
            self.cta_kv * self.head_dim // 4 // self.num_threads
        ):
            flat = tid * elems_per_thr_4 + Int32(_i)
            row = flat >> Int32(6)         # flat // 64 (256/4 = 64)
            col4 = flat & Int32(63)        # flat % 64
            # Global byte offset: row * 256 + col4 * 4
            v_byte_off = row * hd + col4 * Int32(4)
            v_raw = _ld_global_b32(v_ptr + Int64(v_byte_off))
            # SMEM offset: same layout (row-major, no transpose)
            _st_shared_b32(v_smem + Int64(v_byte_off), v_raw)

        # Zero the sync_o buffer (all threads, 4096 bytes = 1024 floats)
        # 128 threads x 8 iters = 1024 floats
        for _i in cutlass.range_constexpr(8):
            so_flat = tid * Int32(8) + Int32(_i)
            _st_shared_f32(sync_o + Int64(so_flat * Int32(4)),
                           Float32(0.0))

        cute.arch.sync_threads()

        # === Phase 2: PV MMA (warp 0 only) ===
        if warp == Int32(0):
            # Hardcode P values: P_unnorm[tok] = tok + 1.0
            # MMA A-operand fragment layout (m16n16k16, BF16):
            #   Thread (group, sub) holds:
            #     a0 = {P[group,   sub*2],   P[group,   sub*2+1]}
            #     a1 = {P[group,   sub*2+8],  P[group,   sub*2+9]}
            #     a2 = {P[group+8, sub*2],   P[group+8, sub*2+1]}
            #     a3 = {P[group+8, sub*2+8],  P[group+8, sub*2+9]}
            #
            # For PV: rows are Q tokens (all same here), cols = KV tokens
            # P_u[tok] = tok + 1.0 for tok 0..5, else 0
            #
            # Thread mapping to KV tokens:
            #   warp 0 handles tokens 0-15
            #   sub*2 gives base token: 0,2,4,6
            #   sub*2+1: 1,3,5,7
            #   sub*2+8: 8,10,12,14
            #   sub*2+9: 9,11,13,15
            #
            # P values (before BF16 pack):
            #   sub=0: p_a0_lo = P[0]=1.0, p_a0_hi = P[1]=2.0
            #          p_a1_lo = P[8]=0.0, p_a1_hi = P[9]=0.0
            #   sub=1: p_a0_lo = P[2]=3.0, p_a0_hi = P[3]=4.0
            #          p_a1_lo = P[10]=0.0, p_a1_hi = P[11]=0.0
            #   sub=2: p_a0_lo = P[4]=5.0, p_a0_hi = P[5]=6.0
            #          p_a1_lo = P[12]=0.0, p_a1_hi = P[13]=0.0
            #   sub=3: p_a0_lo = P[6]=0.0, p_a0_hi = P[7]=0.0
            #          p_a1_lo = P[14]=0.0, p_a1_hi = P[15]=0.0
            #
            # Same for all groups (all Q rows identical).
            # a2, a3 = same as a0, a1 (rows group+8 get same P).

            # v_scale = 1.0 (baked into P values for simplicity)
            v_scale_f32 = Float32(1.0)

            # Compute P values from sub index
            # tok_a0_lo = sub*2, tok_a0_hi = sub*2+1
            tok_a0_lo = sub * Int32(2)
            tok_a0_hi = sub * Int32(2) + Int32(1)
            tok_a1_lo = sub * Int32(2) + Int32(8)
            tok_a1_hi = sub * Int32(2) + Int32(9)

            # P_u[tok] = tok + 1.0 if tok < 6, else 0.0
            p_a0_lo = Float32(0.0)
            p_a0_hi = Float32(0.0)
            p_a1_lo = Float32(0.0)
            p_a1_hi = Float32(0.0)

            # sub=0: tok 0,1 -> 1.0, 2.0
            # sub=1: tok 2,3 -> 3.0, 4.0
            # sub=2: tok 4,5 -> 5.0, 6.0
            # sub=3: tok 6,7 -> 0.0, 0.0
            # All a1 values: tok >= 8 -> 0.0
            if sub == Int32(0):
                p_a0_lo = Float32(1.0)
                p_a0_hi = Float32(2.0)
            if sub == Int32(1):
                p_a0_lo = Float32(3.0)
                p_a0_hi = Float32(4.0)
            if sub == Int32(2):
                p_a0_lo = Float32(5.0)
                p_a0_hi = Float32(6.0)
            # sub==3 and all a1: already 0.0

            # Pack into BF16x2 MMA operands
            # Production kernel packs: pa0 = cvt(p0, p1), pa1 = cvt(p4, p5)
            # where p0=s0-lm0 etc. The fragment mapping is:
            #   pa0 = {P[row, sub*2], P[row, sub*2+1]}     k=0..7
            #   pa1 = {P[row, sub*2+8], P[row, sub*2+9]}   k=8..15
            #   pa2 = {P[row+8, sub*2], P[row+8, sub*2+1]} k=0..7
            #   pa3 = {P[row+8, sub*2+8], P[row+8, sub*2+9]} k=8..15
            pa0 = _cvt_2f32_to_bf16x2(
                p_a0_lo * v_scale_f32, p_a0_hi * v_scale_f32)
            pa1 = _cvt_2f32_to_bf16x2(
                p_a1_lo * v_scale_f32, p_a1_hi * v_scale_f32)
            # rows 8-15 get same P (all Q rows identical)
            pa2 = _cvt_2f32_to_bf16x2(
                p_a0_lo * v_scale_f32, p_a0_hi * v_scale_f32)
            pa3 = _cvt_2f32_to_bf16x2(
                p_a1_lo * v_scale_f32, p_a1_hi * v_scale_f32)

            # === V fragment loading (exact copy of production kernel) ===
            # _md_idx = 0, so v_k_start = 0
            v_k_start = Int32(0)
            # warp 0: warp_kv_start = 0
            v_tok0 = sub * Int32(2)  # tokens 0,2,4,6 for sub 0,1,2,3

            # First m16n8: V cols [v_k_start+group] = [group]
            v_hd0 = v_k_start + group
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

            # Second m16n8: V cols [v_k_start+group+8] = [group+8]
            v_hd1 = v_k_start + group + Int32(8)
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

            # === Dump per-thread PV output to global ===
            # pv_dump[lane * 8 + i] for warp 0 only
            pv_base = lane * Int32(8)
            pv_dump[pv_base + Int32(0)] = t0
            pv_dump[pv_base + Int32(1)] = t1
            pv_dump[pv_base + Int32(2)] = t2
            pv_dump[pv_base + Int32(3)] = t3
            pv_dump[pv_base + Int32(4)] = t4
            pv_dump[pv_base + Int32(5)] = t5
            pv_dump[pv_base + Int32(6)] = t6
            pv_dump[pv_base + Int32(7)] = t7

            # === Sync_o write (exact copy of production kernel) ===
            # sync_o_small layout: [warp][row][col16], FP32
            # warp=0, so so_warp_off = 0
            W16 = Int32(16)
            so_warp_off = Int32(0)  # warp 0
            so_r0 = so_warp_off + group * W16 * Int32(4)
            so_r1 = so_warp_off + (group + Int32(8)) * W16 * Int32(4)
            lc0 = sub * Int32(2)
            lc8 = sub * Int32(2) + Int32(8)

            _st_shared_f32(sync_o + Int64(
                so_r0 + lc0 * Int32(4)), t0)
            _st_shared_f32(sync_o + Int64(
                so_r0 + (lc0 + Int32(1)) * Int32(4)), t1)
            _st_shared_f32(sync_o + Int64(
                so_r1 + lc0 * Int32(4)), t2)
            _st_shared_f32(sync_o + Int64(
                so_r1 + (lc0 + Int32(1)) * Int32(4)), t3)
            _st_shared_f32(sync_o + Int64(
                so_r0 + lc8 * Int32(4)), t4)
            _st_shared_f32(sync_o + Int64(
                so_r0 + (lc8 + Int32(1)) * Int32(4)), t5)
            _st_shared_f32(sync_o + Int64(
                so_r1 + lc8 * Int32(4)), t6)
            _st_shared_f32(sync_o + Int64(
                so_r1 + (lc8 + Int32(1)) * Int32(4)), t7)

        cute.arch.sync_threads()

        # === Phase 3: Dump sync_o buffer to global ===
        # 16 rows x 16 cols = 256 FP32 values, only warp 0's slice
        # 128 threads x 2 = 256
        for _i in cutlass.range_constexpr(2):
            so_flat = tid * Int32(2) + Int32(_i)
            val = _ld_shared_f32(sync_o + Int64(
                so_flat * Int32(4)))
            sync_o_dump[so_flat] = val

    def __call__(self, v_cache_flat):
        """Run the probe.

        Args:
            v_cache_flat: FP8 V cache, shape [64*256], as uint8 contiguous
        Returns:
            pv_dump: [32, 8] per-thread PV MMA output (warp 0 only)
            sync_o_dump: [16, 16] sync_o buffer contents (warp 0 slice)
        """
        pv_dump = torch.zeros(256, dtype=torch.float32, device="cuda")
        sync_o_dump = torch.zeros(256, dtype=torch.float32, device="cuda")
        v_ptr = Int64(v_cache_flat.data_ptr())

        if self._compiled is None:
            print("Compiling sync_o isolation probe...")
            self._compiled = cute.compile(
                self._jit_launch,
                v_ptr, pv_dump, sync_o_dump, Int32(1),
            )
        self._compiled(v_ptr, pv_dump, sync_o_dump, Int32(1))
        return pv_dump.reshape(32, 8), sync_o_dump.reshape(16, 16)


def build_onehot_v():
    """Build one-hot V cache: V[tok=t, dim=t] = 1.0 for t=0..5.

    Shape: [64, 256] as uint8 (FP8 E4M3). All other entries = 0.
    FP8 E4M3 1.0 = 0x38.
    """
    v = torch.zeros(64, 256, dtype=torch.uint8, device="cuda")
    for t in range(6):
        v[t, t] = 0x38  # E4M3 1.0
    return v.contiguous().view(-1)


def compute_expected():
    """Compute expected D[16, 16] for one-hot V, P_u[tok] = tok+1.

    D[m, n] = sum_k(P[m,k] * V[k,n])
    V[k, n] = 1.0 iff k == n and k < 6
    P[m, k] = P_u[k] = k + 1.0 for k < 6, else 0

    So D[m, n] = P_u[n] = n + 1.0 for n < 6, else 0.0
    Same for ALL rows m (P is row-independent in this test).
    """
    D = torch.zeros(16, 16, dtype=torch.float32)
    for n in range(6):
        D[:, n] = float(n + 1)
    return D


def reconstruct_16x16_from_pv(pv):
    """Reconstruct 16x16 matrix from per-thread PV dump.

    pv: [32, 8] -- 32 lanes, 8 values each (t0..t7).
    MMA fragment layout:
      t0: row=group,   col=sub*2       (first m16n8)
      t1: row=group,   col=sub*2+1
      t2: row=group+8, col=sub*2
      t3: row=group+8, col=sub*2+1
      t4: row=group,   col=sub*2+8     (second m16n8)
      t5: row=group,   col=sub*2+9
      t6: row=group+8, col=sub*2+8
      t7: row=group+8, col=sub*2+9
    """
    out = torch.zeros(16, 16, dtype=torch.float32)
    for lane in range(32):
        g = lane // 4
        s = lane % 4
        vals = pv[lane]
        out[g,     s * 2]     = vals[0]
        out[g,     s * 2 + 1] = vals[1]
        out[g + 8, s * 2]     = vals[2]
        out[g + 8, s * 2 + 1] = vals[3]
        out[g,     s * 2 + 8] = vals[4]
        out[g,     s * 2 + 9] = vals[5]
        out[g + 8, s * 2 + 8] = vals[6]
        out[g + 8, s * 2 + 9] = vals[7]
    return out


def main():
    print("=" * 70)
    print("Diagnostic: PV MMA output vs sync_o write isolation")
    print("  V = one-hot (tok t -> dim t, t=0..5)")
    print("  P = hardcoded: P_u[tok] = tok + 1.0 for tok < 6, else 0")
    print("  _md_idx = 0 (dims 0-15), warp 0 only (tokens 0-15)")
    print("  v_scale = 1.0")
    print("=" * 70)

    probe = SyncOIsolationProbe()
    v_flat = build_onehot_v()
    expected = compute_expected()

    pv, sync_o = probe(v_flat)

    # --- Analysis 1: Per-thread PV dump ---
    print("\n" + "=" * 70)
    print("PART A: Per-thread PV MMA output (o0..o7)")
    print("=" * 70)

    pv_matrix = reconstruct_16x16_from_pv(pv)

    print("\nReconstructed 16x16 from PV dump:")
    for r in range(16):
        vals = [f"{pv_matrix[r, c].item():7.2f}" for c in range(16)]
        print(f"  row {r:2d}: [{', '.join(vals)}]")

    print("\nExpected 16x16:")
    for r in range(min(2, 16)):
        vals = [f"{expected[r, c].item():7.2f}" for c in range(16)]
        print(f"  row {r:2d}: [{', '.join(vals)}]")
    print("  ... (all rows identical)")

    # Check PV correctness
    pv_match = torch.allclose(pv_matrix, expected, atol=0.15)
    pv_max_err = (pv_matrix - expected).abs().max().item()
    print(f"\nPV MMA match: {'PASS' if pv_match else 'FAIL'}")
    print(f"  Max error: {pv_max_err:.4f}")

    if not pv_match:
        print("\n  Per-thread detail (group, sub -> values):")
        for lane in range(32):
            g = lane // 4
            s = lane % 4
            vals = pv[lane].tolist()
            nz = [i for i, v in enumerate(vals) if abs(v) > 0.01]
            if nz:
                print(f"    lane {lane:2d} (g={g}, s={s}): "
                      f"t0={vals[0]:6.2f} t1={vals[1]:6.2f} "
                      f"t2={vals[2]:6.2f} t3={vals[3]:6.2f} "
                      f"t4={vals[4]:6.2f} t5={vals[5]:6.2f} "
                      f"t6={vals[6]:6.2f} t7={vals[7]:6.2f}")

        # Diagnose the diagonal pattern
        print("\n  Diagonal check: does output vary by group?")
        col0_by_row = pv_matrix[:, 0].tolist()
        print(f"    D[row, col=0]: {[f'{v:.2f}' for v in col0_by_row]}")
        if len(set(f"{v:.2f}" for v in col0_by_row
                   if abs(v) > 0.01)) > 1:
            print("    --> Column 0 varies by row: MMA output is "
                  "row-dependent (WRONG)")
        else:
            print("    --> Column 0 constant across rows (correct)")

    # --- Analysis 2: Sync_o buffer dump ---
    print("\n" + "=" * 70)
    print("PART B: sync_o buffer contents (warp 0 slice)")
    print("=" * 70)

    print("\nsync_o[16][16]:")
    for r in range(16):
        vals = [f"{sync_o[r, c].item():7.2f}" for c in range(16)]
        print(f"  row {r:2d}: [{', '.join(vals)}]")

    # Check sync_o correctness
    so_match = torch.allclose(sync_o, expected, atol=0.15)
    so_max_err = (sync_o - expected).abs().max().item()
    print(f"\nsync_o match: {'PASS' if so_match else 'FAIL'}")
    print(f"  Max error: {so_max_err:.4f}")

    # --- Analysis 3: Compare PV dump vs sync_o ---
    print("\n" + "=" * 70)
    print("PART C: PV dump vs sync_o comparison")
    print("=" * 70)

    pv_vs_so = torch.allclose(pv_matrix, sync_o, atol=1e-6)
    pv_so_err = (pv_matrix - sync_o).abs().max().item()
    print(f"  PV dump == sync_o: {'MATCH' if pv_vs_so else 'MISMATCH'}")
    print(f"  Max difference: {pv_so_err:.6f}")

    if not pv_vs_so:
        print("\n  Positions where PV and sync_o differ:")
        diff = (pv_matrix - sync_o).abs()
        for r in range(16):
            for c in range(16):
                if diff[r, c] > 1e-6:
                    print(f"    [{r:2d},{c:2d}]: PV={pv_matrix[r,c].item():.4f}"
                          f"  sync_o={sync_o[r,c].item():.4f}")

    # --- Summary ---
    print("\n" + "=" * 70)
    print("DIAGNOSIS SUMMARY")
    print("=" * 70)

    if pv_match and so_match:
        print("  Both PV MMA and sync_o are CORRECT.")
        print("  Bug must be in cross-warp reduction or output write.")
    elif pv_match and not so_match:
        print("  PV MMA is CORRECT but sync_o is WRONG.")
        print("  --> Bug is in the sync_o WRITE pattern.")
    elif not pv_match and pv_vs_so:
        print("  PV MMA is WRONG, but sync_o matches PV dump.")
        print("  --> Bug is in PV MMA itself (or V loading or P packing).")
    else:
        print("  PV MMA is WRONG AND sync_o differs from PV dump.")
        print("  --> Multiple bugs: both MMA and sync_o write are broken.")

    # Final pass/fail
    all_pass = pv_match and so_match and pv_vs_so
    print(f"\nOverall: {'PASS' if all_pass else 'FAIL'}")
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
