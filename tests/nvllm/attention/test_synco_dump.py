#!/usr/bin/env python3
"""Session 11 probe: Dump sync_o values from 4-warp PV MMA.

Tests just the PV MMA + sync_o store from the decode kernel with
4 warps. No cross-warp reduction — raw per-warp output dumped to
global memory for inspection.

This isolates: are the per-warp PV MMA results correct, or does
the sync_o store pattern already show the 16/256 corruption?

Volume-mount into container and run:
  python /app/nvllm/tests/nvllm/attention/test_synco_dump.py
"""
import sys
sys.path.insert(0, "/app/nvllm")

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


class SyncODumpProbe:
    """4-warp probe: runs full QK→softmax→PV for ONE _md block,
    dumps the 8 PV MMA output values (o0..o7) per thread per warp
    directly to global memory. No sync_o store, no reduction.
    """

    def __init__(self):
        self.cta_q = 16
        self.cta_kv = 64  # Full page
        self.head_dim = 256
        self.num_mma_d = self.head_dim // 16
        self.num_threads = 128  # 4 warps
        self.num_warps_kv = 4

        self.q_bytes = self.cta_q * self.head_dim * 2  # BF16
        self.k_bytes = self.cta_kv * self.head_dim  # FP8
        self.v_bytes = self.cta_kv * self.head_dim  # FP8
        self.smem_bytes = self.q_bytes + self.k_bytes + self.v_bytes
        self._compiled = None

    @cute.jit
    def _jit_launch(self, query, k_ptr: Int64, v_ptr: Int64,
                    seq_len: Int32, pv_dump, qk_dump,
                    scale, k_scale,
                    num_q_heads, num_kv_heads,
                    md_block: Int32,
                    gx: Int32):
        self._kernel(
            query, k_ptr, v_ptr, seq_len,
            pv_dump, qk_dump,
            scale, k_scale, num_q_heads, num_kv_heads,
            md_block,
        ).launch(
            grid=[gx, Int32(1), Int32(1)],
            block=[self.num_threads, 1, 1],
            smem=self.smem_bytes,
        )

    @cute.kernel
    def _kernel(self, query, k_ptr: Int64, v_ptr: Int64,
                seq_len: Int32,
                pv_dump, qk_dump,
                scale, k_scale,
                num_q_heads, num_kv_heads,
                md_block: Int32):
        """4-warp probe for ONE _md block.

        Dumps per-thread PV MMA results to pv_dump[warp*32*8 + lane*8 + i].
        Also dumps QK scores to qk_dump[warp*32*8 + lane*8 + i].
        """
        lane = cute.arch.lane_idx()
        warp = cute.arch.warp_idx()
        tid = warp * Int32(32) + lane
        group = lane >> Int32(2)
        sub = lane & Int32(3)

        smem = cute.arch.get_dyn_smem(cutlass.Uint8, alignment=128)
        q_smem = shared_ptr_to_i64(smem)
        k_smem = shared_ptr_to_i64(smem + Int32(self.q_bytes))
        v_smem = shared_ptr_to_i64(
            smem + Int32(self.q_bytes + self.k_bytes))

        hd = Int32(self.head_dim)
        warp_kv_start = warp * Int32(16)

        kv_tok_stride = num_kv_heads * hd
        kv_page_stride = Int32(self.cta_kv) * kv_tok_stride

        LOG2E = Float32(1.4426950408889634)
        sm_scale_log2 = Float32(scale) * Float32(k_scale) * LOG2E

        # === Load Q into SMEM ===
        elems_per_thr_q = Int32(
            self.cta_q * self.head_dim // self.num_threads)
        for _i in cutlass.range_constexpr(
            self.cta_q * self.head_dim // self.num_threads
        ):
            flat = tid * elems_per_thr_q + Int32(_i)
            row = flat // hd
            col = flat % hd
            gmem_idx = row * hd + col
            smem_byte = (row * hd + col) * Int32(2)
            val = query[gmem_idx]
            val_u32 = _cvt_2f32_to_bf16x2(
                Float32(val), Float32(0.0))
            _st_shared_b16_from_u32(
                q_smem + Int64(smem_byte), val_u32)

        # === Load K (full 64 tokens) ===
        elems_per_thr_kv4 = Int32(
            self.cta_kv * self.head_dim // 4 // self.num_threads)
        for _i in cutlass.range_constexpr(
            self.cta_kv * self.head_dim // 4 // self.num_threads
        ):
            flat = tid * elems_per_thr_kv4 + Int32(_i)
            row = flat >> Int32(6)
            col4 = flat & Int32(63)
            k_byte_off = row * kv_tok_stride + col4 * Int32(4)
            k_raw = _ld_global_b32(k_ptr + Int64(k_byte_off))
            smem_byte = row * hd + col4 * Int32(4)
            _st_shared_b32(k_smem + Int64(smem_byte), k_raw)

        # === Load V (full 64 tokens) ===
        for _i in cutlass.range_constexpr(
            self.cta_kv * self.head_dim // 4 // self.num_threads
        ):
            flat = tid * elems_per_thr_kv4 + Int32(_i)
            row = flat >> Int32(6)
            col4 = flat & Int32(63)
            v_byte_off = row * kv_tok_stride + col4 * Int32(4)
            v_raw = _ld_global_b32(v_ptr + Int64(v_byte_off))
            v_smem_byte = row * hd + col4 * Int32(4)
            _st_shared_b32(v_smem + Int64(v_smem_byte), v_raw)

        cute.arch.sync_threads()

        # === QK MMA (all 16 K-dim iterations) ===
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
            k_raw_0a = _ld_shared_b16(
                k_smem + Int64(k_off_0a))
            k_raw_0b = _ld_shared_b16(
                k_smem + Int64(k_off_0a + Int32(8)))
            k_packed_0 = _pack_lo16(k_raw_0a, k_raw_0b)
            b0, b1 = fp8x4_e4m3_to_bfloat2x2(k_packed_0)

            kv_row_1 = warp_kv_start + n_t + Int32(8)
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

        # Scale + mask
        s0 = s0 * sm_scale_log2
        s1 = s1 * sm_scale_log2
        s2 = s2 * sm_scale_log2
        s3 = s3 * sm_scale_log2
        s4 = s4 * sm_scale_log2
        s5 = s5 * sm_scale_log2
        s6 = s6 * sm_scale_log2
        s7 = s7 * sm_scale_log2

        tok_base = warp_kv_start
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

        # Dump QK scores before softmax
        qk_base = (warp * Int32(32) + lane) * Int32(8)
        qk_dump[qk_base + Int32(0)] = s0
        qk_dump[qk_base + Int32(1)] = s1
        qk_dump[qk_base + Int32(2)] = s2
        qk_dump[qk_base + Int32(3)] = s3
        qk_dump[qk_base + Int32(4)] = s4
        qk_dump[qk_base + Int32(5)] = s5
        qk_dump[qk_base + Int32(6)] = s6
        qk_dump[qk_base + Int32(7)] = s7

        # Softmax
        lm0 = _fmax(_fmax(s0, s1), _fmax(s4, s5))
        lm1 = _fmax(_fmax(s2, s3), _fmax(s6, s7))
        lm0 = _fmax(lm0, shfl_xor_sync(lm0, Int32(1)))
        lm0 = _fmax(lm0, shfl_xor_sync(lm0, Int32(2)))
        lm1 = _fmax(lm1, shfl_xor_sync(lm1, Int32(1)))
        lm1 = _fmax(lm1, shfl_xor_sync(lm1, Int32(2)))

        p0 = exp2_approx_ftz_f32(s0 - lm0)
        p1 = exp2_approx_ftz_f32(s1 - lm0)
        p2 = exp2_approx_ftz_f32(s2 - lm1)
        p3 = exp2_approx_ftz_f32(s3 - lm1)
        p4 = exp2_approx_ftz_f32(s4 - lm0)
        p5 = exp2_approx_ftz_f32(s5 - lm0)
        p6 = exp2_approx_ftz_f32(s6 - lm1)
        p7 = exp2_approx_ftz_f32(s7 - lm1)

        v_scale_f32 = Float32(1.0)

        # PV MMA for specified _md block
        pa0 = _cvt_2f32_to_bf16x2(
            p0 * v_scale_f32, p1 * v_scale_f32)
        pa1 = _cvt_2f32_to_bf16x2(
            p4 * v_scale_f32, p5 * v_scale_f32)
        pa2 = _cvt_2f32_to_bf16x2(
            p2 * v_scale_f32, p3 * v_scale_f32)
        pa3 = _cvt_2f32_to_bf16x2(
            p6 * v_scale_f32, p7 * v_scale_f32)

        v_k_start = md_block * Int32(16)
        v_tok0 = warp_kv_start + sub * Int32(2)

        # First m16n8: V cols [v_k_start+group]
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

        # Second m16n8: V cols [v_k_start+group+8]
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

        # PV MMA
        (t0, t1, t2, t3,
         t4, t5, t6, t7) = bf16_mma_m16n16k16_f32(
            Float32(0.0), Float32(0.0),
            Float32(0.0), Float32(0.0),
            Float32(0.0), Float32(0.0),
            Float32(0.0), Float32(0.0),
            pa0, pa1, pa2, pa3,
            vb0, vb1, vb2, vb3)

        # Dump raw PV MMA output per thread
        pv_base = (warp * Int32(32) + lane) * Int32(8)
        pv_dump[pv_base + Int32(0)] = t0
        pv_dump[pv_base + Int32(1)] = t1
        pv_dump[pv_base + Int32(2)] = t2
        pv_dump[pv_base + Int32(3)] = t3
        pv_dump[pv_base + Int32(4)] = t4
        pv_dump[pv_base + Int32(5)] = t5
        pv_dump[pv_base + Int32(6)] = t6
        pv_dump[pv_base + Int32(7)] = t7

    def __call__(self, query, k_cache, v_cache, seq_len,
                 scale, k_scale, num_q_heads, num_kv_heads,
                 md_block=0):
        # 128 threads × 8 values = 1024
        pv_dump = torch.zeros(1024, dtype=torch.float32, device="cuda")
        qk_dump = torch.zeros(1024, dtype=torch.float32, device="cuda")
        k_ptr = Int64(k_cache.data_ptr())
        v_ptr = Int64(v_cache.data_ptr())

        if self._compiled is None:
            print("Compiling sync_o dump probe...")
            self._compiled = cute.compile(
                self._jit_launch,
                query, k_ptr, v_ptr, Int32(seq_len),
                pv_dump, qk_dump,
                float(scale), float(k_scale),
                Int32(num_q_heads), Int32(num_kv_heads),
                Int32(md_block), Int32(1),
            )
        self._compiled(
            query, k_ptr, v_ptr, Int32(seq_len),
            pv_dump, qk_dump,
            float(scale), float(k_scale),
            Int32(num_q_heads), Int32(num_kv_heads),
            Int32(md_block), Int32(1),
        )
        # Reshape: [4 warps, 32 lanes, 8 values]
        return pv_dump.reshape(4, 32, 8), qk_dump.reshape(4, 32, 8)


def main():
    print("=" * 60)
    print("Session 11: Sync_o dump probe (4 warps, single _md)")
    print("=" * 60)

    probe = SyncODumpProbe()

    torch.manual_seed(42)
    dev = "cuda"
    num_q, num_kv, hd = 16, 1, 256
    scale = 1.0 / (hd ** 0.5)
    seq_len = 64  # All warps valid

    query = torch.randn(num_q, hd, dtype=torch.bfloat16, device=dev)

    # K cache: 64 tokens × 1 KV head × 256 dim (FP8)
    k_float = torch.randn(64, num_kv, hd, device=dev).clamp(-5, 5)
    k_cache = k_float.to(torch.float8_e4m3fn).view(torch.uint8)

    # V cache: linearly varying
    v_float = torch.zeros(64, num_kv, hd, device=dev)
    for t in range(64):
        for d in range(hd):
            v_float[t, 0, d] = ((t * 7 + d * 3) % 19 - 9) * 0.5
    v_cache = v_float.clamp(-448, 448).to(torch.float8_e4m3fn).view(
        torch.uint8)

    # Test _md=0 and _md=1
    for md_block in [0, 1]:
        pv, qk = probe(query, k_cache.contiguous(), v_cache.contiguous(),
                        seq_len, scale, 1.0, num_q, num_kv, md_block)

        print(f"\n{'='*60}")
        print(f"_md={md_block}: V dims [{md_block*16}..{md_block*16+15}]")
        print(f"{'='*60}")

        # Per-warp analysis
        for w in range(4):
            wpv = pv[w]  # [32, 8]
            nz_total = (wpv.abs() > 1e-8).sum().item()

            # Reconstruct 16×16 output from MMA fragment layout
            out_16x16 = torch.zeros(16, 16, dtype=torch.float32)
            for lane in range(32):
                g = lane // 4
                s = lane % 4
                vals = wpv[lane]
                # t0..t3 from first m16n8, t4..t7 from second
                out_16x16[g, s*2] = vals[0]
                out_16x16[g, s*2+1] = vals[1]
                out_16x16[g+8, s*2] = vals[2]
                out_16x16[g+8, s*2+1] = vals[3]
                out_16x16[g, s*2+8] = vals[4]
                out_16x16[g, s*2+9] = vals[5]
                out_16x16[g+8, s*2+8] = vals[6]
                out_16x16[g+8, s*2+9] = vals[7]

            nz_16x16 = (out_16x16.abs() > 1e-8)
            nz_count = nz_16x16.sum().item()
            nz_cols = set()
            for r in range(16):
                for c in range(16):
                    if nz_16x16[r, c]:
                        nz_cols.add(c)

            print(f"\n  Warp {w} (tokens {w*16}-{w*16+15}):")
            print(f"    Raw nonzero: {nz_total}/256")
            print(f"    16x16 nonzero: {nz_count}/256")
            print(f"    Nonzero cols: {sorted(nz_cols)}")
            print(f"    Row 0: {[f'{x:.3f}' for x in out_16x16[0].tolist()]}")
            print(f"    Row 1: {[f'{x:.3f}' for x in out_16x16[1].tolist()]}")

            # Check QK scores
            wqk = qk[w]
            qk_row0_scores = []
            for s in range(4):
                lane_idx = 0 * 4 + s  # group=0
                vals = wqk[lane_idx]
                qk_row0_scores.extend([
                    vals[0].item(), vals[1].item(),  # s0,s1 (tok sub*2, sub*2+1)
                    vals[4].item(), vals[5].item(),  # s4,s5 (tok sub*2+8, sub*2+9)
                ])
            print(f"    QK[row0, tok0..15]: "
                  f"{[f'{x:.2f}' for x in qk_row0_scores]}")

        # Also check V SMEM content by looking at what each warp sees
        print(f"\n  V check: first 4 bytes for dim {md_block*16}:")
        for t in range(4):
            v_byte = v_cache[t, 0, md_block*16].item()
            v_ref = v_float[t, 0, md_block*16].item()
            print(f"    V[tok={t}, dim={md_block*16}]: "
                  f"byte=0x{v_byte:02x}, ref={v_ref:.2f}")


if __name__ == "__main__":
    main()
