#!/usr/bin/env python3
"""Session 11: Dump o0..o7 accumulators AFTER while page loop.

Replicates exact decode kernel flow (while loop with QK+softmax+PV)
for one _md block, then dumps o0..o7 to global memory instead of
writing to sync_o. Compares with the direct-dump probe (no while loop)
to check if the while loop corrupts accumulated values.

Volume-mount into container and run:
  python /app/nvllm/tests/nvllm/attention/test_accumulator_dump.py
"""
import sys
sys.path.insert(0, "/app/nvllm")

import torch
import logging

logging.basicConfig(level=logging.WARNING)

try:
    import cutlass
    from cutlass import cute
    from cutlass.cute.typing import BFloat16, Float32, Int32, Int64, Uint32
    from cutlass.cutlass_dsl import T, dsl_user_op
except ImportError:
    print("CUTLASS not available")
    sys.exit(0)

from vllm.v1.attention.backends.cute_paged.kernel import (
    _mma_m16n8k16_f32,
    bf16_mma_m16n16k16_f32,
    _cvt_2f32_to_bf16x2,
    fp8x4_e4m3_to_bfloat2x2,
    _pack_4bytes,
    _pack_lo16,
    _ld_shared_b32,
    _ld_shared_b16,
    _st_shared_b32,
    _st_shared_f32,
    _st_shared_b16_from_u32,
    _ld_shared_f32,
    _extract_byte_from_b32,
    _ld_global_b32,
    shared_ptr_to_i64,
    exp2_approx_ftz_f32,
    shfl_xor_sync,
    _fmax,
)


class AccumulatorDumpProbe:
    """Exact decode kernel flow with while loop, dumps o0..o7 after."""

    def __init__(self):
        self.cta_q = 16
        self.cta_kv = 64
        self.head_dim = 256
        self.num_mma_d = self.head_dim // 16
        self.num_threads = 128
        self.num_warps_kv = 4

        self.q_bytes = self.cta_q * self.head_dim * 2
        self.k_bytes = self.cta_kv * self.head_dim
        self.v_bytes = self.cta_kv * self.head_dim
        self.smem_bytes = self.q_bytes + self.k_bytes + self.v_bytes
        self._compiled = None

    @cute.jit
    def _jit_launch(self, query, k_ptr: Int64, v_ptr: Int64,
                    page_table, seq_lens, accum_dump,
                    scale, k_scale, v_scale,
                    num_q_heads, num_kv_heads,
                    gx: Int32, gy: Int32, gz: Int32):
        self._kernel(
            query, k_ptr, v_ptr, page_table, seq_lens,
            accum_dump, scale, k_scale, v_scale,
            num_q_heads, num_kv_heads,
        ).launch(
            grid=[gx, gy, gz],
            block=[self.num_threads, 1, 1],
            smem=self.smem_bytes,
        )

    @cute.kernel
    def _kernel(self, query, k_ptr: Int64, v_ptr: Int64,
                page_table, seq_lens, accum_dump,
                scale, k_scale, v_scale,
                num_q_heads, num_kv_heads):
        """Exact copy of decode kernel for _md=0, dumps o0..o7 after page loop."""
        bx, by, bz = cute.arch.block_idx()
        lane = cute.arch.lane_idx()
        warp = cute.arch.warp_idx()
        tid = warp * Int32(32) + lane
        group = lane >> Int32(2)
        sub = lane & Int32(3)

        kv_head_idx = by
        seq_idx = bz
        group_size = num_q_heads // num_kv_heads
        q_head_start = kv_head_idx * group_size + bx * Int32(self.cta_q)

        seq_len = seq_lens[seq_idx]
        num_pages = (seq_len + Int32(self.cta_kv - 1)) \
            // Int32(self.cta_kv)

        LOG2E = Float32(1.4426950408889634)
        sm_scale_log2 = Float32(scale) * Float32(k_scale) * LOG2E
        v_scale_f32 = Float32(v_scale)

        smem = cute.arch.get_dyn_smem(cutlass.Uint8, alignment=128)
        q_smem = shared_ptr_to_i64(smem)
        k_smem = shared_ptr_to_i64(smem + Int32(self.q_bytes))
        v_smem = shared_ptr_to_i64(
            smem + Int32(self.q_bytes + self.k_bytes))

        hd = Int32(self.head_dim)
        warp_kv_start = warp * Int32(16)

        kv_tok_stride = num_kv_heads * hd
        kv_page_stride = Int32(self.cta_kv) * kv_tok_stride

        # Load Q (exact copy from kernel)
        q_stride_tok = num_q_heads * hd
        elems_per_thr_q = Int32(self.cta_q * self.head_dim
                                // self.num_threads)
        for _i in cutlass.range_constexpr(
            self.cta_q * self.head_dim // self.num_threads
        ):
            flat = tid * elems_per_thr_q + Int32(_i)
            row = flat // hd
            col = flat % hd
            gmem_idx = (seq_idx * q_stride_tok
                        + (q_head_start + row) * hd + col)
            smem_byte = (row * hd + col) * Int32(2)
            val = query[gmem_idx]
            val_u32 = _cvt_2f32_to_bf16x2(
                Float32(val), Float32(0.0))
            _st_shared_b16_from_u32(
                q_smem + Int64(smem_byte), val_u32)

        cute.arch.sync_threads()

        # _md=0 only
        _md_idx = Int32(0)
        o0 = Float32(0.0)
        o1 = Float32(0.0)
        o2 = Float32(0.0)
        o3 = Float32(0.0)
        o4 = Float32(0.0)
        o5 = Float32(0.0)
        o6 = Float32(0.0)
        o7 = Float32(0.0)
        m_r0 = Float32(-1e30)
        m_r1 = Float32(-1e30)
        d_r0 = Float32(0.0)
        d_r1 = Float32(0.0)

        # WHILE LOOP (exact copy from kernel)
        page_idx = Int32(0)
        while page_idx < num_pages:
            phys_page = page_table[seq_idx, page_idx]

            # Load K
            elems_per_thr_kv4 = Int32(
                self.cta_kv * self.head_dim // 4 // self.num_threads)
            for _i in cutlass.range_constexpr(
                self.cta_kv * self.head_dim // 4 // self.num_threads
            ):
                flat = tid * elems_per_thr_kv4 + Int32(_i)
                row = flat >> Int32(6)
                col4 = flat & Int32(63)
                k_byte_off = (phys_page * kv_page_stride
                              + row * kv_tok_stride
                              + kv_head_idx * hd
                              + col4 * Int32(4))
                k_raw = _ld_global_b32(k_ptr + Int64(k_byte_off))
                smem_byte = row * hd + col4 * Int32(4)
                _st_shared_b32(k_smem + Int64(smem_byte), k_raw)

            # Load V
            for _i in cutlass.range_constexpr(
                self.cta_kv * self.head_dim // 4 // self.num_threads
            ):
                flat = tid * elems_per_thr_kv4 + Int32(_i)
                row = flat >> Int32(6)
                col4 = flat & Int32(63)
                v_byte_off = (phys_page * kv_page_stride
                              + row * kv_tok_stride
                              + kv_head_idx * hd
                              + col4 * Int32(4))
                v_raw = _ld_global_b32(v_ptr + Int64(v_byte_off))
                v_smem_byte = row * hd + col4 * Int32(4)
                _st_shared_b32(v_smem + Int64(v_smem_byte), v_raw)

            cute.arch.sync_threads()

            # QK MMA (exact copy)
            s0 = Float32(0.0)
            s1 = Float32(0.0)
            s2 = Float32(0.0)
            s3 = Float32(0.0)
            s4 = Float32(0.0)
            s5 = Float32(0.0)
            s6 = Float32(0.0)
            s7 = Float32(0.0)

            for _kd in cutlass.range_constexpr(self.num_mma_d):
                k_start = Int32(_kd * 16)
                q_byte_a0 = (group * hd + k_start
                             + sub * Int32(2)) * Int32(2)
                a0 = _ld_shared_b32(q_smem + Int64(q_byte_a0))
                a1 = _ld_shared_b32(
                    q_smem + Int64(q_byte_a0 + Int32(16)))
                q_byte_a2 = ((group + Int32(8)) * hd + k_start
                             + sub * Int32(2)) * Int32(2)
                a2 = _ld_shared_b32(q_smem + Int64(q_byte_a2))
                a3 = _ld_shared_b32(
                    q_smem + Int64(q_byte_a2 + Int32(16)))

                n_t = group
                kv_row_0 = warp_kv_start + n_t
                k_off_0a = (kv_row_0 * hd + k_start
                            + sub * Int32(2))
                k_raw_0a = _ld_shared_b16(k_smem + Int64(k_off_0a))
                k_raw_0b = _ld_shared_b16(
                    k_smem + Int64(k_off_0a + Int32(8)))
                k_packed_0 = _pack_lo16(k_raw_0a, k_raw_0b)
                b0, b1 = fp8x4_e4m3_to_bfloat2x2(k_packed_0)

                kv_row_1 = warp_kv_start + n_t + Int32(8)
                k_off_1a = (kv_row_1 * hd + k_start
                            + sub * Int32(2))
                k_raw_1a = _ld_shared_b16(k_smem + Int64(k_off_1a))
                k_raw_1b = _ld_shared_b16(
                    k_smem + Int64(k_off_1a + Int32(8)))
                k_packed_1 = _pack_lo16(k_raw_1a, k_raw_1b)
                b2, b3 = fp8x4_e4m3_to_bfloat2x2(k_packed_1)

                (s0, s1, s2, s3,
                 s4, s5, s6, s7) = bf16_mma_m16n16k16_f32(
                    s0, s1, s2, s3, s4, s5, s6, s7,
                    a0, a1, a2, a3, b0, b1, b2, b3)

            # Softmax (exact copy)
            s0 = s0 * sm_scale_log2
            s1 = s1 * sm_scale_log2
            s2 = s2 * sm_scale_log2
            s3 = s3 * sm_scale_log2
            s4 = s4 * sm_scale_log2
            s5 = s5 * sm_scale_log2
            s6 = s6 * sm_scale_log2
            s7 = s7 * sm_scale_log2

            tok_base = page_idx * Int32(self.cta_kv) + warp_kv_start
            NEG = Float32(-1e20)
            tok0 = tok_base + sub * Int32(2)
            tok1 = tok0 + Int32(1)
            tok8 = tok0 + Int32(8)
            tok9 = tok8 + Int32(1)
            if tok0 >= seq_len:
                s0 = NEG
                s2 = NEG
            if tok1 >= seq_len:
                s1 = NEG
                s3 = NEG
            if tok8 >= seq_len:
                s4 = NEG
                s6 = NEG
            if tok9 >= seq_len:
                s5 = NEG
                s7 = NEG

            lm0 = _fmax(_fmax(s0, s1), _fmax(s4, s5))
            lm1 = _fmax(_fmax(s2, s3), _fmax(s6, s7))
            lm0 = _fmax(lm0, shfl_xor_sync(lm0, Int32(1)))
            lm0 = _fmax(lm0, shfl_xor_sync(lm0, Int32(2)))
            lm1 = _fmax(lm1, shfl_xor_sync(lm1, Int32(1)))
            lm1 = _fmax(lm1, shfl_xor_sync(lm1, Int32(2)))

            m0_new = _fmax(m_r0, lm0)
            m1_new = _fmax(m_r1, lm1)
            sc0 = exp2_approx_ftz_f32(m_r0 - m0_new)
            sc1 = exp2_approx_ftz_f32(m_r1 - m1_new)
            d_r0 = d_r0 * sc0
            d_r1 = d_r1 * sc1
            o0 = o0 * sc0
            o1 = o1 * sc0
            o2 = o2 * sc1
            o3 = o3 * sc1
            o4 = o4 * sc0
            o5 = o5 * sc0
            o6 = o6 * sc1
            o7 = o7 * sc1
            m_r0 = m0_new
            m_r1 = m1_new

            p0 = exp2_approx_ftz_f32(s0 - m_r0)
            p1 = exp2_approx_ftz_f32(s1 - m_r0)
            p2 = exp2_approx_ftz_f32(s2 - m_r1)
            p3 = exp2_approx_ftz_f32(s3 - m_r1)
            p4 = exp2_approx_ftz_f32(s4 - m_r0)
            p5 = exp2_approx_ftz_f32(s5 - m_r0)
            p6 = exp2_approx_ftz_f32(s6 - m_r1)
            p7 = exp2_approx_ftz_f32(s7 - m_r1)

            ls0 = (p0 + p1) + (p4 + p5)
            ls1 = (p2 + p3) + (p6 + p7)
            ls0 = ls0 + shfl_xor_sync(ls0, Int32(1))
            ls0 = ls0 + shfl_xor_sync(ls0, Int32(2))
            ls1 = ls1 + shfl_xor_sync(ls1, Int32(1))
            ls1 = ls1 + shfl_xor_sync(ls1, Int32(2))
            d_r0 = d_r0 + ls0
            d_r1 = d_r1 + ls1

            # PV MMA for _md=0 only
            pa0 = _cvt_2f32_to_bf16x2(
                p0 * v_scale_f32, p1 * v_scale_f32)
            pa1 = _cvt_2f32_to_bf16x2(
                p4 * v_scale_f32, p5 * v_scale_f32)
            pa2 = _cvt_2f32_to_bf16x2(
                p2 * v_scale_f32, p3 * v_scale_f32)
            pa3 = _cvt_2f32_to_bf16x2(
                p6 * v_scale_f32, p7 * v_scale_f32)

            v_k_start = _md_idx * Int32(16)
            v_tok0 = warp_kv_start + sub * Int32(2)

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

            (t0, t1, t2, t3,
             t4, t5, t6, t7) = bf16_mma_m16n16k16_f32(
                Float32(0.0), Float32(0.0),
                Float32(0.0), Float32(0.0),
                Float32(0.0), Float32(0.0),
                Float32(0.0), Float32(0.0),
                pa0, pa1, pa2, pa3,
                vb0, vb1, vb2, vb3)
            o0 = o0 + t0
            o1 = o1 + t1
            o2 = o2 + t2
            o3 = o3 + t3
            o4 = o4 + t4
            o5 = o5 + t5
            o6 = o6 + t6
            o7 = o7 + t7

            cute.arch.sync_threads()
            page_idx = page_idx + Int32(1)
        # END while loop

        # === DUMP o0..o7 AFTER while loop (the key diagnostic) ===
        dump_base = (warp * Int32(32) + lane) * Int32(8)
        accum_dump[dump_base + Int32(0)] = o0
        accum_dump[dump_base + Int32(1)] = o1
        accum_dump[dump_base + Int32(2)] = o2
        accum_dump[dump_base + Int32(3)] = o3
        accum_dump[dump_base + Int32(4)] = o4
        accum_dump[dump_base + Int32(5)] = o5
        accum_dump[dump_base + Int32(6)] = o6
        accum_dump[dump_base + Int32(7)] = o7

    def __call__(self, query, k_cache, v_cache, page_table, seq_lens,
                 scale, k_scale, v_scale, num_q_heads, num_kv_heads):
        accum_dump = torch.zeros(1024, dtype=torch.float32, device="cuda")
        k_ptr = Int64(k_cache.data_ptr())
        v_ptr = Int64(v_cache.data_ptr())
        group_size = num_q_heads // num_kv_heads
        num_q_tiles = max((group_size + self.cta_q - 1) // self.cta_q, 1)
        grid = (num_q_tiles, num_kv_heads, len(seq_lens))

        if self._compiled is None:
            print("Compiling accumulator dump probe...")
            self._compiled = cute.compile(
                self._jit_launch,
                query, k_ptr, v_ptr, page_table, seq_lens,
                accum_dump, float(scale), float(k_scale), float(v_scale),
                Int32(num_q_heads), Int32(num_kv_heads),
                Int32(grid[0]), Int32(grid[1]), Int32(grid[2]),
            )
        self._compiled(
            query, k_ptr, v_ptr, page_table, seq_lens,
            accum_dump, float(scale), float(k_scale), float(v_scale),
            Int32(num_q_heads), Int32(num_kv_heads),
            Int32(grid[0]), Int32(grid[1]), Int32(grid[2]),
        )
        return accum_dump.reshape(4, 32, 8)


def main():
    print("=" * 60)
    print("Session 11: Accumulator dump after while loop")
    print("=" * 60)

    probe = AccumulatorDumpProbe()

    torch.manual_seed(42)
    dev = "cuda"
    num_q, num_kv, hd = 24, 4, 256
    scale = 1.0 / (hd ** 0.5)

    seq_lens = torch.tensor([64], dtype=torch.int32, device=dev)
    query = torch.randn(1, num_q, hd, dtype=torch.bfloat16, device=dev)

    kv_shape = (1, 64, num_kv, hd)
    k_float = torch.randn(*kv_shape, device=dev).clamp(-5, 5)
    k_cache = k_float.to(torch.float8_e4m3fn).view(torch.uint8)

    v_float = torch.zeros(*kv_shape, device=dev)
    for t in range(64):
        for h in range(num_kv):
            for d in range(hd):
                v_float[0, t, h, d] = ((t * 7 + d * 3 + h * 13) % 19 - 9) * 0.5
    v_cache = v_float.clamp(-448, 448).to(torch.float8_e4m3fn).view(torch.uint8)

    page_table = torch.zeros(1, 2, dtype=torch.int32, device=dev)

    # Run probe
    accum = probe(query, k_cache.contiguous(), v_cache.contiguous(),
                  page_table, seq_lens, scale, 1.0, 1.0, num_q, num_kv)

    print("\n=== Accumulator values after while loop (should match PV probe) ===")
    for w in range(4):
        wpv = accum[w]  # [32, 8]
        # Reconstruct 16×16
        out_16x16 = torch.zeros(16, 16, dtype=torch.float32)
        for ln in range(32):
            g = ln // 4
            s = ln % 4
            vals = wpv[ln]
            out_16x16[g, s*2] = vals[0]
            out_16x16[g, s*2+1] = vals[1]
            out_16x16[g+8, s*2] = vals[2]
            out_16x16[g+8, s*2+1] = vals[3]
            out_16x16[g, s*2+8] = vals[4]
            out_16x16[g, s*2+9] = vals[5]
            out_16x16[g+8, s*2+8] = vals[6]
            out_16x16[g+8, s*2+9] = vals[7]

        nz = (out_16x16.abs() > 1e-8).sum().item()
        nz_cols = sorted(set(
            c for r in range(16) for c in range(16)
            if abs(out_16x16[r, c].item()) > 1e-8
        ))
        print(f"\n  Warp {w}: {nz}/256 nonzero, cols={nz_cols}")
        print(f"    Row 0: {[f'{x:.3f}' for x in out_16x16[0].tolist()]}")

    # Also run the direct probe (no while loop) for comparison
    from test_synco_dump import SyncODumpProbe
    direct = SyncODumpProbe()
    # Need to create matching test data for the direct probe
    # Direct probe uses different data format (single KV head)
    print("\n=== Direct PV probe comparison (no while loop) ===")
    print("(See test_synco_dump.py results above for the direct probe)")
    print("\nIf accum has fewer nonzero than direct probe → while loop corrupts values")
    print("If accum matches direct probe → sync_o store is the issue")


if __name__ == "__main__":
    main()
