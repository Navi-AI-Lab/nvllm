#!/usr/bin/env python3
"""Compare QK scores: proven-correct probe vs PyTorch reference.
Uses the EXACT same data as test_cute_kernel_standalone.py.
If scores match: the K loading/MMA is fine, bug is in softmax/PV/reduction.
"""
import torch
import math
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
    _st_shared_b32, _st_shared_f32, _pack_lo16,
    fp8x4_e4m3_to_bfloat2x2, _st_shared_b16_from_u32,
    _cvt_2f32_to_bf16x2, shared_ptr_to_i64,
)


class ProductionKLoadProbe:
    """QK probe using the PRODUCTION kernel's K loading pattern:
    k_ptr + phys_page * kv_page_stride + row * kv_tok_stride + kv_head_idx * hd + col4 * 4

    16 Q-rows, 16 KV tokens (warp 0), head_dim=256.
    """
    def __init__(self):
        self.num_threads = 128
        self.cta_q = 16
        self.cta_kv = 64  # full page
        self.head_dim = 256
        self.num_mma_d = 16
        self.q_bytes = 16 * 256 * 2
        self.k_bytes = 64 * 256
        self.smem_bytes = self.q_bytes + self.k_bytes
        self._compiled = None

    @cute.jit
    def _jit_launch(self, query, k_ptr: Int64, page_table, output,
                    num_q_heads, num_kv_heads,
                    gx: Int32):
        self._kernel(query, k_ptr, page_table, output,
                     num_q_heads, num_kv_heads).launch(
            grid=[gx, Int32(1), Int32(1)],
            block=[self.num_threads, 1, 1],
            smem=self.smem_bytes,
        )

    @cute.kernel
    def _kernel(self, query, k_ptr: Int64, page_table, output,
                num_q_heads, num_kv_heads):
        lane = cute.arch.lane_idx()
        warp = cute.arch.warp_idx()
        tid = warp * Int32(32) + lane
        group = lane >> Int32(2)
        sub = lane & Int32(3)

        smem = cute.arch.get_dyn_smem(cutlass.Uint8, alignment=128)
        q_smem = shared_ptr_to_i64(smem)
        k_smem = shared_ptr_to_i64(smem + Int32(self.q_bytes))
        hd = Int32(self.head_dim)

        kv_head_idx = Int32(0)  # KV head 0
        seq_idx = Int32(0)
        warp_kv_start = warp * Int32(16)

        # === Load Q: exact same as production kernel ===
        q_stride_tok = num_q_heads * hd
        elems_per_thr_q = Int32(
            self.cta_q * self.head_dim // self.num_threads)
        for _i in cutlass.range_constexpr(
            self.cta_q * self.head_dim // self.num_threads
        ):
            flat = tid * elems_per_thr_q + Int32(_i)
            row = flat // hd
            col = flat % hd
            gmem_idx = (seq_idx * q_stride_tok + row * hd + col)
            smem_byte = (row * hd + col) * Int32(2)
            val = query[gmem_idx]
            val_u32 = _cvt_2f32_to_bf16x2(
                Float32(val), Float32(0.0))
            _st_shared_b16_from_u32(
                q_smem + Int64(smem_byte), val_u32)

        # === Load K: exact same as production kernel ===
        kv_tok_stride = num_kv_heads * hd
        kv_page_stride = Int32(self.cta_kv) * kv_tok_stride
        phys_page = page_table[seq_idx, Int32(0)]

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
            from vllm.v1.attention.backends.cute_paged.kernel import (
                _ld_global_b32,
            )
            k_raw = _ld_global_b32(k_ptr + Int64(k_byte_off))
            smem_byte = row * hd + col4 * Int32(4)
            _st_shared_b32(k_smem + Int64(smem_byte), k_raw)

        cute.arch.sync_threads()

        # === QK MMA: warp 0 only, same as production ===
        if warp == Int32(0):
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

            # Dump raw QK scores
            base = lane * Int32(8)
            output[base] = s0
            output[base + Int32(1)] = s1
            output[base + Int32(2)] = s2
            output[base + Int32(3)] = s3
            output[base + Int32(4)] = s4
            output[base + Int32(5)] = s5
            output[base + Int32(6)] = s6
            output[base + Int32(7)] = s7

    def __call__(self, query, k_cache, page_table, num_q_heads, num_kv_heads):
        output = torch.zeros(256, dtype=torch.float32, device="cuda")
        k_ptr = Int64(k_cache.data_ptr())
        # FLATTEN query to 1D to avoid CuTe DSL multi-dim indexing issues
        q_flat = query.contiguous().view(-1)
        if self._compiled is None:
            print("Compiling production K-load QK probe...")
            self._compiled = cute.compile(
                self._jit_launch,
                q_flat, k_ptr, page_table, output,
                Int32(num_q_heads), Int32(num_kv_heads),
                Int32(1),
            )
        self._compiled(
            q_flat, k_ptr, page_table, output,
            Int32(num_q_heads), Int32(num_kv_heads),
            Int32(1),
        )
        return output.reshape(32, 8)


def main():
    # Reproduce exact standalone test data
    torch.manual_seed(42)
    num_q_heads, num_kv_heads, head_dim = 24, 4, 256
    page_size = 64
    scale = 1.0 / (head_dim ** 0.5)
    sm_scale_log2 = scale * math.log2(math.e)

    kv_shape = (2, page_size, num_kv_heads, head_dim)
    torch.manual_seed(42)
    query = torch.randn(1, num_q_heads, head_dim, dtype=torch.bfloat16,
                        device="cuda")
    k_float = torch.randn(*kv_shape, device="cuda").clamp(-10, 10)
    k_cache = k_float.to(torch.float8_e4m3fn).view(torch.uint8)

    page_table = torch.zeros(1, 2, dtype=torch.int32, device="cuda")

    probe = ProductionKLoadProbe()
    result = probe(query, k_cache.contiguous(), page_table,
                   num_q_heads, num_kv_heads)

    # Reference QK scores
    q_f = query[0].float()
    k_f = k_cache[0, :, 0, :].view(torch.float8_e4m3fn).to(
        torch.bfloat16).float()

    print("=" * 70)
    print("Production K-load QK check (head 0, kv_head 0)")
    print("=" * 70)

    # Thread (g=0, s=0) has s0 = raw_QK[head=0, tok=0], s1 = raw_QK[head=0, tok=1]
    # Scaled by sm_scale_log2 to match the production softmax
    for head_row in range(min(3, num_q_heads)):
        # Find the lane for this head row
        # head_row maps to SMEM row `head_row`
        # group = head_row for rows 0-7, group = head_row-8 for rows 8-15
        if head_row < 8:
            g = head_row
        else:
            g = head_row - 8

        ref_scores = (q_f[head_row] @ k_f[:6].T) * scale
        ref_scores_log2 = ref_scores * math.log2(math.e)

        print(f"\n  Head {head_row} (row={head_row}):")
        print(f"    Ref scores (log2): "
              f"{[round(x.item(), 4) for x in ref_scores_log2]}")

        # Get probe scores for tokens 0-5
        # tok 0: thread (g, sub=0) → s0
        # tok 1: thread (g, sub=0) → s1
        # tok 2: thread (g, sub=1) → s0
        # tok 3: thread (g, sub=1) → s1
        # tok 4: thread (g, sub=2) → s0
        # tok 5: thread (g, sub=2) → s1
        probe_scores = []
        for tok in range(6):
            s = tok // 2  # sub
            si = tok % 2  # s0 or s1
            lane = g * 4 + s
            if head_row < 8:
                val = result[lane, si].item()
            else:
                val = result[lane, si + 2].item()  # s2, s3 for rows 8-15
            probe_scores.append(val)

        # Scale probe scores (raw MMA output, no sm_scale_log2 yet)
        # Actually the probe does NOT multiply by sm_scale_log2
        probe_log2 = [x * sm_scale_log2 for x in probe_scores]

        print(f"    Probe raw scores:  "
              f"{[round(x, 4) for x in probe_scores]}")
        print(f"    Probe scaled log2: "
              f"{[round(x, 4) for x in probe_log2]}")
        print(f"    Diff (log2):       "
              f"{[round(probe_log2[i] - ref_scores_log2[i].item(), 4) for i in range(6)]}")


if __name__ == "__main__":
    main()
