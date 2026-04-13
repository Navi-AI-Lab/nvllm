#!/usr/bin/env python3
"""Full-flow diagnostic: dumps intermediate values at each stage
(QK scores, softmax state, P values, PV MMA output, sync_o, reduced output)
for a single _md=0 iteration with seq_len=6.

Identifies exactly where the stride-32 corruption occurs.

Volume-mount into container and run:
  python /app/nvllm/tests/nvllm/attention/test_full_flow_dump.py
"""
import torch
import logging

logging.basicConfig(level=logging.WARNING)


def test_full_flow():
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

    class FullFlowProbe:
        """Single-warp, single-page, _md=0 only. No cross-warp reduction.
        Dumps: QK scores, softmax P, PV MMA output."""

        def __init__(self):
            self.cta_q = 16
            self.cta_kv = 16   # Only 16 tokens (not 64)
            self.head_dim = 256
            self.num_mma_d = self.head_dim // 16  # 16
            self.num_threads = 32  # Single warp

            self.q_bytes = self.cta_q * self.head_dim * 2  # 8192
            self.k_bytes = self.cta_kv * self.head_dim      # 4096
            self.v_bytes = self.cta_kv * self.head_dim      # 4096
            self.smem_bytes = self.q_bytes + self.k_bytes + self.v_bytes
            self._compiled = None

        @cute.jit
        def _jit_launch(self, query, k_ptr: Int64, v_ptr: Int64,
                        seq_len: Int32, output, dump,
                        scale, k_scale,
                        num_q_heads, num_kv_heads,
                        gx: Int32):
            self._kernel(
                query, k_ptr, v_ptr, seq_len,
                output, dump, scale, k_scale,
                num_q_heads, num_kv_heads,
            ).launch(
                grid=[gx, Int32(1), Int32(1)],
                block=[self.num_threads, 1, 1],
                smem=self.smem_bytes,
            )

        @cute.kernel
        def _kernel(self, query, k_ptr: Int64, v_ptr: Int64,
                    seq_len: Int32, output, dump,
                    scale, k_scale,
                    num_q_heads, num_kv_heads):
            """Single warp full flow: QK → softmax → PV for _md=0.
            Dumps intermediate values to `dump` tensor."""
            lane = cute.arch.lane_idx()
            group = lane >> Int32(2)
            sub = lane & Int32(3)

            smem = cute.arch.get_dyn_smem(cutlass.Uint8, alignment=128)
            q_smem = shared_ptr_to_i64(smem)
            k_smem = shared_ptr_to_i64(
                smem + Int32(self.q_bytes))
            v_smem = shared_ptr_to_i64(
                smem + Int32(self.q_bytes + self.k_bytes))

            hd = Int32(self.head_dim)
            kv_head_idx = Int32(0)

            LOG2E = Float32(1.4426950408889634)
            sm_scale_log2 = Float32(scale) * Float32(k_scale) * LOG2E

            # === Load Q into SMEM ===
            q_stride_tok = num_q_heads * hd
            for _i in cutlass.range_constexpr(
                self.cta_q * self.head_dim // self.num_threads
            ):
                flat = lane * Int32(
                    self.cta_q * self.head_dim
                    // self.num_threads) + Int32(_i)
                row = flat // hd
                col = flat % hd
                gmem_idx = row * hd + col
                smem_byte = (row * hd + col) * Int32(2)
                val = query[gmem_idx]
                val_u32 = _cvt_2f32_to_bf16x2(
                    Float32(val), Float32(0.0))
                _st_shared_b16_from_u32(
                    q_smem + Int64(smem_byte), val_u32)

            # === Load K into SMEM (16 tokens only) ===
            kv_tok_stride = num_kv_heads * hd
            for _i in cutlass.range_constexpr(
                self.cta_kv * self.head_dim // self.num_threads
            ):
                flat = lane * Int32(
                    self.cta_kv * self.head_dim
                    // self.num_threads) + Int32(_i)
                row = flat // hd
                col = flat % hd
                k_byte_off = row * kv_tok_stride + col
                k_raw_byte = _ld_global_b32(
                    k_ptr + Int64(k_byte_off & Int32(0xFFFFFFFC)))
                # Store full word to SMEM
                smem_off = row * hd + (col & Int32(0xFFFFFFFC))
                if (col & Int32(3)) == Int32(0):
                    _st_shared_b32(
                        k_smem + Int64(smem_off), k_raw_byte)

            # === Load V into SMEM (16 tokens only) ===
            for _i in cutlass.range_constexpr(
                self.cta_kv * self.head_dim // self.num_threads
            ):
                flat = lane * Int32(
                    self.cta_kv * self.head_dim
                    // self.num_threads) + Int32(_i)
                row = flat // hd
                col = flat % hd
                v_byte_off = row * kv_tok_stride + col
                v_raw_byte = _ld_global_b32(
                    v_ptr + Int64(v_byte_off & Int32(0xFFFFFFFC)))
                smem_off = row * hd + (col & Int32(0xFFFFFFFC))
                if (col & Int32(3)) == Int32(0):
                    _st_shared_b32(
                        v_smem + Int64(smem_off), v_raw_byte)

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

                kv_row_0 = group
                k_off_0a = kv_row_0 * hd + k_start + sub * Int32(2)
                k_raw_0a = _ld_shared_b16(
                    k_smem + Int64(k_off_0a))
                k_raw_0b = _ld_shared_b16(
                    k_smem + Int64(k_off_0a + Int32(8)))
                k_packed_0 = _pack_lo16(k_raw_0a, k_raw_0b)
                b0, b1 = fp8x4_e4m3_to_bfloat2x2(k_packed_0)

                kv_row_1 = group + Int32(8)
                k_off_1a = kv_row_1 * hd + k_start + sub * Int32(2)
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

            # Scale
            s0 = s0 * sm_scale_log2
            s1 = s1 * sm_scale_log2
            s2 = s2 * sm_scale_log2
            s3 = s3 * sm_scale_log2
            s4 = s4 * sm_scale_log2
            s5 = s5 * sm_scale_log2
            s6 = s6 * sm_scale_log2
            s7 = s7 * sm_scale_log2

            # Dump raw QK scores for thread 0 (group=0, sub=0)
            # dump[0..7] = s0..s7 for this thread
            dump_base = lane * Int32(32)
            dump[dump_base + Int32(0)] = s0
            dump[dump_base + Int32(1)] = s1
            dump[dump_base + Int32(2)] = s2
            dump[dump_base + Int32(3)] = s3
            dump[dump_base + Int32(4)] = s4
            dump[dump_base + Int32(5)] = s5
            dump[dump_base + Int32(6)] = s6
            dump[dump_base + Int32(7)] = s7

            # Mask
            NEG = Float32(-1e20)
            tok0 = sub * Int32(2)
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

            d0 = (p0 + p1) + (p4 + p5)
            d0 = d0 + shfl_xor_sync(d0, Int32(1))
            d0 = d0 + shfl_xor_sync(d0, Int32(2))
            d1_s = (p2 + p3) + (p6 + p7)
            d1_s = d1_s + shfl_xor_sync(d1_s, Int32(1))
            d1_s = d1_s + shfl_xor_sync(d1_s, Int32(2))

            # Dump P values
            dump[dump_base + Int32(8)] = p0
            dump[dump_base + Int32(9)] = p1
            dump[dump_base + Int32(10)] = p2
            dump[dump_base + Int32(11)] = p3
            dump[dump_base + Int32(12)] = p4
            dump[dump_base + Int32(13)] = p5
            dump[dump_base + Int32(14)] = p6
            dump[dump_base + Int32(15)] = p7
            dump[dump_base + Int32(16)] = lm0
            dump[dump_base + Int32(17)] = d0
            dump[dump_base + Int32(18)] = lm1
            dump[dump_base + Int32(19)] = d1_s

            # PV MMA for _md=0 only
            pa0 = _cvt_2f32_to_bf16x2(p0, p1)
            pa1 = _cvt_2f32_to_bf16x2(p4, p5)
            pa2 = _cvt_2f32_to_bf16x2(p2, p3)
            pa3 = _cvt_2f32_to_bf16x2(p6, p7)

            v_tok0 = sub * Int32(2)
            v_hd0 = group

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

            (t0, t1, t2, t3,
             t4, t5, t6, t7) = bf16_mma_m16n16k16_f32(
                Float32(0.0), Float32(0.0),
                Float32(0.0), Float32(0.0),
                Float32(0.0), Float32(0.0),
                Float32(0.0), Float32(0.0),
                pa0, pa1, pa2, pa3,
                vb0, vb1, vb2, vb3)

            # Dump PV output
            dump[dump_base + Int32(20)] = t0
            dump[dump_base + Int32(21)] = t1
            dump[dump_base + Int32(22)] = t2
            dump[dump_base + Int32(23)] = t3
            dump[dump_base + Int32(24)] = t4
            dump[dump_base + Int32(25)] = t5
            dump[dump_base + Int32(26)] = t6
            dump[dump_base + Int32(27)] = t7

            # Normalize and write output (no cross-warp reduction)
            # output[row, col] = t / d
            W16 = Int32(16)
            c0 = sub * Int32(2)
            c8 = sub * Int32(2) + Int32(8)
            output[group * W16 + c0] = BFloat16(t0 / d0)
            output[group * W16 + c0 + Int32(1)] = BFloat16(
                t1 / d0)
            output[(group + Int32(8)) * W16 + c0] = BFloat16(
                t2 / d1_s)
            output[(group + Int32(8)) * W16 + c0
                   + Int32(1)] = BFloat16(t3 / d1_s)
            output[group * W16 + c8] = BFloat16(t4 / d0)
            output[group * W16 + c8 + Int32(1)] = BFloat16(
                t5 / d0)
            output[(group + Int32(8)) * W16 + c8] = BFloat16(
                t6 / d1_s)
            output[(group + Int32(8)) * W16 + c8
                   + Int32(1)] = BFloat16(t7 / d1_s)

        def __call__(self, query, k_cache, v_cache, seq_len, scale,
                     k_scale, num_q_heads, num_kv_heads):
            output = torch.zeros(16 * 16, dtype=torch.bfloat16,
                                 device="cuda")
            dump = torch.zeros(32 * 32, dtype=torch.float32,
                               device="cuda")
            k_ptr = Int64(k_cache.data_ptr())
            v_ptr = Int64(v_cache.data_ptr())

            if self._compiled is None:
                print("Compiling full-flow probe...")
                self._compiled = cute.compile(
                    self._jit_launch,
                    query, k_ptr, v_ptr, Int32(seq_len),
                    output, dump, float(scale), float(k_scale),
                    Int32(num_q_heads), Int32(num_kv_heads),
                    Int32(1),
                )
            self._compiled(
                query, k_ptr, v_ptr, Int32(seq_len),
                output, dump, float(scale), float(k_scale),
                Int32(num_q_heads), Int32(num_kv_heads),
                Int32(1),
            )
            return output.reshape(16, 16), dump.reshape(32, 32)

    probe = FullFlowProbe()

    torch.manual_seed(42)
    dev = "cuda"
    num_q, num_kv, hd = 16, 1, 256  # Simplified: 16 Q heads, 1 KV head
    scale = 1.0 / (hd ** 0.5)

    # Q: 16 heads × 256 dim
    query = torch.randn(num_q, hd, dtype=torch.bfloat16, device=dev)

    # K: 16 tokens × 1 KV head × 256 dim (FP8)
    k_float = torch.randn(16, num_kv, hd, device=dev).clamp(-5, 5)
    k_cache = k_float.to(torch.float8_e4m3fn).view(torch.uint8)

    # V: identity one-hot (16 tokens, first 16 dims)
    v_cache = torch.zeros(16, num_kv, hd, dtype=torch.uint8, device=dev)
    for t in range(16):
        v_cache[t, 0, t] = 0x38

    out, dump = probe(query, k_cache.contiguous(), v_cache.contiguous(),
                      6, scale, 1.0, num_q, num_kv)

    print("=== Full-Flow Probe: 16Q×16KV, seq_len=6, identity V ===")
    print(f"scale={scale:.6f}, scale*log2e={scale*1.4427:.6f}")

    # Thread 0 (group=0, sub=0): handles QK output row=0, KV tokens 0,1,8,9
    t0_dump = dump[0].tolist()
    print(f"\nThread 0 (group=0, sub=0):")
    print(f"  QK scores (s0..s7): {[round(x,4) for x in t0_dump[:8]]}")
    print(f"  P values (p0..p7):  {[round(x,4) for x in t0_dump[8:16]]}")
    print(f"  lm0={t0_dump[16]:.4f} d0={t0_dump[17]:.4f}")
    print(f"  lm1={t0_dump[18]:.4f} d1={t0_dump[19]:.4f}")
    print(f"  PV output (t0..t7): {[round(x,4) for x in t0_dump[20:28]]}")

    # Thread 4 (group=1, sub=0): provides V column 1 data
    t4_dump = dump[4].tolist()
    print(f"\nThread 4 (group=1, sub=0):")
    print(f"  P values (p0..p7):  {[round(x,4) for x in t4_dump[8:16]]}")
    print(f"  PV output (t0..t7): {[round(x,4) for x in t4_dump[20:28]]}")

    # Output matrix (16 rows × 16 cols)
    out_f = out.float()
    print(f"\nOutput[0, :] (Q row 0, dims 0-15):")
    print(f"  {[round(x, 4) for x in out_f[0].tolist()]}")
    print(f"Output[1, :] (Q row 1, dims 0-15):")
    print(f"  {[round(x, 4) for x in out_f[1].tolist()]}")

    nz = (out_f.abs() > 1e-6).nonzero(as_tuple=False).tolist()
    print(f"\nNonzero positions (row, col): {nz[:20]}")
    print(f"Nonzero count: {len(nz)} / 256")

    # Reference: compute in PyTorch
    q_f = query.float()  # [16, 256]
    k_f = k_cache[:, 0, :].view(torch.float8_e4m3fn).to(
        torch.bfloat16).float()  # [16, 256]
    v_f = v_cache[:, 0, :].view(torch.float8_e4m3fn).to(
        torch.bfloat16).float()  # [16, 256]

    scores = (q_f @ k_f.T) * scale  # [16, 16]
    # Mask tokens >= 6
    scores[:, 6:] = -1e20
    probs = torch.softmax(scores, dim=-1)  # [16, 16]
    ref = probs @ v_f  # [16, 256]
    ref_block = ref[:, :16]  # First 16 dims only

    print(f"\nRef[0, :16]: {[round(x, 4) for x in ref_block[0].tolist()]}")
    print(f"Ref[1, :16]: {[round(x, 4) for x in ref_block[1].tolist()]}")

    diff = (out_f - ref_block.to(torch.bfloat16).float()).abs()
    print(f"\nMax diff: {diff.max().item():.6f}")
    print(f"Mean diff: {diff.mean().item():.6f}")


if __name__ == "__main__":
    test_full_flow()
