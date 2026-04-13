#!/usr/bin/env python3
"""V-loading isolation probe for CuTe paged attention kernel.

Tests the complete V data path in isolation from QK/softmax:

  Phase 1 — SMEM dump: cooperative GMEM→SMEM load, then read all SMEM
  bytes back and compare with the original V cache tensor.  If this fails,
  the cooperative load itself is wrong.

  Phase 2 — Fragment extraction: from correct SMEM, extract bytes the
  exact same way the PV MMA does (aligned b32 load → byte extract →
  pack_4bytes → fp8x4_e4m3_to_bfloat2x2), then write the resulting
  BF16→FP32 values to output.  Compared with PyTorch's fp8→bf16→fp32.

Uses position-encoded V values: V[tok, dim] = f(tok, dim) so any
mis-addressed load produces a visible, diagnosable mismatch.

Volume-mount into container and run:
  docker cp tests/nvllm/attention/test_v_load_probe.py \\
      nvllm:/app/nvllm/tests/nvllm/attention/
  docker exec nvllm python \\
      /app/nvllm/tests/nvllm/attention/test_v_load_probe.py
"""
import torch
import sys
import logging

logging.basicConfig(level=logging.WARNING)

# ---------------------------------------------------------------------------
# Imports — fail fast if CUTLASS isn't available
# ---------------------------------------------------------------------------
try:
    import cutlass                                       # noqa: F401
    from cutlass import cute
    from cutlass.cute.typing import (
        BFloat16, Float32, Int32, Int64, Uint32,
    )
    from cutlass.cutlass_dsl import T, dsl_user_op
except ImportError:
    print("CUTLASS not available — run inside the nvllm container")
    sys.exit(1)

# Re-use the *exact* PTX helpers from the production kernel.
from vllm.v1.attention.backends.cute_paged.kernel import (
    shared_ptr_to_i64,
    _ld_global_b32,
    _ld_shared_b32,
    _st_shared_b32,
    _extract_byte_from_b32,
    _pack_4bytes,
    fp8x4_e4m3_to_bfloat2x2,
)

from cutlass._mlir import ir as _mlir_ir
from cutlass._mlir.dialects import llvm as _llvm_dialect


# ---------------------------------------------------------------------------
# Helper: unpack BF16x2 (Uint32) → individual FP32
# ---------------------------------------------------------------------------
@dsl_user_op
def _bf16x2_to_f32_lo(val: Uint32, *, loc=None, ip=None) -> Float32:
    """Extract low BF16 from BF16x2 pack and convert to FP32."""
    val_ir = Uint32(val).ir_value(loc=loc, ip=ip)
    result_ir = _llvm_dialect.inline_asm(
        T.f32(), [val_ir],
        "{\n"
        "  .reg .b16 lo, hi;\n"
        "  mov.b32 {lo, hi}, $1;\n"
        "  cvt.f32.bf16 $0, lo;\n"
        "}",
        "=f,r", has_side_effects=True, asm_dialect=0,
        loc=loc, ip=ip,
    )
    return Float32(result_ir)


@dsl_user_op
def _bf16x2_to_f32_hi(val: Uint32, *, loc=None, ip=None) -> Float32:
    """Extract high BF16 from BF16x2 pack and convert to FP32."""
    val_ir = Uint32(val).ir_value(loc=loc, ip=ip)
    result_ir = _llvm_dialect.inline_asm(
        T.f32(), [val_ir],
        "{\n"
        "  .reg .b16 lo, hi;\n"
        "  mov.b32 {lo, hi}, $1;\n"
        "  cvt.f32.bf16 $0, hi;\n"
        "}",
        "=f,r", has_side_effects=True, asm_dialect=0,
        loc=loc, ip=ip,
    )
    return Float32(result_ir)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
NUM_THREADS = 128
PAGE_SIZE = 64
HEAD_DIM = 256
V_SMEM_BYTES = PAGE_SIZE * HEAD_DIM   # 16 384
LOADS_PER_THREAD = V_SMEM_BYTES // 4 // NUM_THREADS   # 32


# ---------------------------------------------------------------------------
# Phase 1 — SMEM byte dump
# ---------------------------------------------------------------------------
class _Phase1:
    """Load V page to SMEM, dump raw bytes to output."""

    def __init__(self):
        self.num_threads = NUM_THREADS
        self.smem_bytes = V_SMEM_BYTES
        self._compiled = None

    @cute.jit
    def _jit_launch(self, v_ptr: Int64, num_kv_heads: Int32,
                    kv_head_idx: Int32, output):
        self._kernel(v_ptr, num_kv_heads, kv_head_idx,
                     output).launch(
            grid=[Int32(1), Int32(1), Int32(1)],
            block=[self.num_threads, 1, 1],
            smem=self.smem_bytes,
        )

    @cute.kernel
    def _kernel(self, v_ptr: Int64, num_kv_heads: Int32,
                kv_head_idx: Int32, output):
        warp = cute.arch.warp_idx()
        lane = cute.arch.lane_idx()
        tid = warp * Int32(32) + lane

        smem = cute.arch.get_dyn_smem(cutlass.Uint8, alignment=128)
        v_smem = shared_ptr_to_i64(smem)

        hd = Int32(HEAD_DIM)
        kv_tok_stride = num_kv_heads * hd

        # --- Cooperative V GMEM → SMEM (mirrors production kernel) ---
        elems = Int32(LOADS_PER_THREAD)
        for _i in cutlass.range_constexpr(LOADS_PER_THREAD):
            flat = tid * elems + Int32(_i)
            row = flat >> Int32(6)           # token index (0..63)
            col4 = flat & Int32(63)          # 4-byte column (0..63)
            v_byte_off = (row * kv_tok_stride
                          + kv_head_idx * hd
                          + col4 * Int32(4))
            v_raw = _ld_global_b32(v_ptr + Int64(v_byte_off))
            smem_byte = row * hd + col4 * Int32(4)
            _st_shared_b32(v_smem + Int64(smem_byte), v_raw)

        cute.arch.sync_threads()

        # --- Read SMEM back and dump to global output ---
        # output is BFloat16 [V_SMEM_BYTES] — write each byte as a
        # BFloat16 float value (0..255) for host comparison.
        bytes_per_thr = Int32(V_SMEM_BYTES // NUM_THREADS)   # 128
        base = tid * bytes_per_thr
        for _i in cutlass.range_constexpr(
            V_SMEM_BYTES // NUM_THREADS // 4
        ):
            word_off = base + Int32(_i) * Int32(4)
            word = _ld_shared_b32(v_smem + Int64(word_off))
            b0 = _extract_byte_from_b32(word, Int32(0))
            b1 = _extract_byte_from_b32(word, Int32(1))
            b2 = _extract_byte_from_b32(word, Int32(2))
            b3 = _extract_byte_from_b32(word, Int32(3))
            idx = word_off
            output[idx] = BFloat16(Float32(Int32(b0)))
            output[idx + Int32(1)] = BFloat16(Float32(Int32(b1)))
            output[idx + Int32(2)] = BFloat16(Float32(Int32(b2)))
            output[idx + Int32(3)] = BFloat16(Float32(Int32(b3)))

        cute.arch.sync_threads()

    def run(self, v_cache, kv_head_idx=0):
        """Run phase 1 and return (got_float, expected_float)."""
        num_kv_heads = v_cache.shape[2]
        v_ptr = Int64(v_cache.data_ptr())

        output = torch.zeros(V_SMEM_BYTES, dtype=torch.bfloat16,
                             device=v_cache.device)

        if self._compiled is None:
            self._compiled = cute.compile(
                self._jit_launch,
                v_ptr, Int32(num_kv_heads),
                Int32(kv_head_idx), output,
            )

        self._compiled(
            v_ptr, Int32(num_kv_heads),
            Int32(kv_head_idx), output,
        )
        torch.cuda.synchronize()

        expected = v_cache[0, :, kv_head_idx, :].contiguous().flatten()
        expected_f = expected.float()   # uint8 → float (0..255)
        got_f = output.float()
        return got_f, expected_f


# ---------------------------------------------------------------------------
# Phase 2 — Per-fragment byte extraction + FP8→BF16 conversion
# ---------------------------------------------------------------------------
class _Phase2:
    """Load V to SMEM, extract per B-fragment, convert FP8→BF16, dump FP32.

    For _md_idx=0 (head_dim positions 0-15), each of 128 threads extracts
    the exact same V bytes the PV MMA would use, converts via
    fp8x4_e4m3_to_bfloat2x2, unpacks BF16→FP32, and writes to output.

    Output layout: [128 threads × 8 values]
      [tid, 0] = V[tok0,   hd0]  as BF16→FP32
      [tid, 1] = V[tok0+1, hd0]
      [tid, 2] = V[tok0+8, hd0]
      [tid, 3] = V[tok0+9, hd0]
      [tid, 4] = V[tok0,   hd1]  (hd1 = hd0 + 8)
      [tid, 5] = V[tok0+1, hd1]
      [tid, 6] = V[tok0+8, hd1]
      [tid, 7] = V[tok0+9, hd1]

    Where tok0 = warp*16 + sub*2, hd0 = group, hd1 = group + 8.
    """

    def __init__(self):
        self.num_threads = NUM_THREADS
        self.smem_bytes = V_SMEM_BYTES
        self._compiled = None

    @cute.jit
    def _jit_launch(self, v_ptr: Int64, num_kv_heads: Int32,
                    kv_head_idx: Int32, output):
        self._kernel(v_ptr, num_kv_heads, kv_head_idx,
                     output).launch(
            grid=[Int32(1), Int32(1), Int32(1)],
            block=[self.num_threads, 1, 1],
            smem=self.smem_bytes,
        )

    @cute.kernel
    def _kernel(self, v_ptr: Int64, num_kv_heads: Int32,
                kv_head_idx: Int32, output):
        warp = cute.arch.warp_idx()
        lane = cute.arch.lane_idx()
        tid = warp * Int32(32) + lane
        group = lane >> Int32(2)       # 0-7
        sub = lane & Int32(3)          # 0-3

        smem = cute.arch.get_dyn_smem(cutlass.Uint8, alignment=128)
        v_smem = shared_ptr_to_i64(smem)

        hd = Int32(HEAD_DIM)
        kv_tok_stride = num_kv_heads * hd

        # --- Cooperative V GMEM → SMEM (same as Phase 1 / production) ---
        elems = Int32(LOADS_PER_THREAD)
        for _i in cutlass.range_constexpr(LOADS_PER_THREAD):
            flat = tid * elems + Int32(_i)
            row = flat >> Int32(6)
            col4 = flat & Int32(63)
            v_byte_off = (row * kv_tok_stride
                          + kv_head_idx * hd
                          + col4 * Int32(4))
            v_raw = _ld_global_b32(v_ptr + Int64(v_byte_off))
            smem_byte = row * hd + col4 * Int32(4)
            _st_shared_b32(v_smem + Int64(smem_byte), v_raw)

        cute.arch.sync_threads()

        # --- Per-fragment V extraction (_md_idx = 0) ---
        warp_kv_start = warp * Int32(16)
        v_tok0 = warp_kv_start + sub * Int32(2)

        # ---- First m16n8k16: hd0 = group (0-7) ----
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

        v_packed_0 = _pack_4bytes(vb0_0, vb0_1, vb0_8, vb0_9)
        vb0, vb1 = fp8x4_e4m3_to_bfloat2x2(v_packed_0)

        # Unpack BF16x2 → individual FP32 for output
        f0_0 = _bf16x2_to_f32_lo(vb0)    # V[tok0, hd0]
        f0_1 = _bf16x2_to_f32_hi(vb0)    # V[tok0+1, hd0]
        f0_8 = _bf16x2_to_f32_lo(vb1)    # V[tok0+8, hd0]
        f0_9 = _bf16x2_to_f32_hi(vb1)    # V[tok0+9, hd0]

        # ---- Second m16n8k16: hd1 = group + 8 ----
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

        v_packed_1 = _pack_4bytes(vb1_0, vb1_1, vb1_8, vb1_9)
        vb2, vb3 = fp8x4_e4m3_to_bfloat2x2(v_packed_1)

        f1_0 = _bf16x2_to_f32_lo(vb2)
        f1_1 = _bf16x2_to_f32_hi(vb2)
        f1_8 = _bf16x2_to_f32_lo(vb3)
        f1_9 = _bf16x2_to_f32_hi(vb3)

        # --- Write 8 FP32 values to output[tid*8 .. tid*8+7] ---
        out_base = tid * Int32(8)
        output[out_base] = f0_0
        output[out_base + Int32(1)] = f0_1
        output[out_base + Int32(2)] = f0_8
        output[out_base + Int32(3)] = f0_9
        output[out_base + Int32(4)] = f1_0
        output[out_base + Int32(5)] = f1_1
        output[out_base + Int32(6)] = f1_8
        output[out_base + Int32(7)] = f1_9

        cute.arch.sync_threads()

    def run(self, v_cache, kv_head_idx=0):
        """Run phase 2. Returns (got_f32[128,8], expected_f32[128,8])."""
        num_kv_heads = v_cache.shape[2]
        v_ptr = Int64(v_cache.data_ptr())

        output = torch.zeros(NUM_THREADS * 8, dtype=torch.float32,
                             device=v_cache.device)

        if self._compiled is None:
            self._compiled = cute.compile(
                self._jit_launch,
                v_ptr, Int32(num_kv_heads),
                Int32(kv_head_idx), output,
            )

        self._compiled(
            v_ptr, Int32(num_kv_heads),
            Int32(kv_head_idx), output,
        )
        torch.cuda.synchronize()

        got = output.view(NUM_THREADS, 8).cpu()

        # Build expected: for each thread, compute which V bytes it reads
        v_page = v_cache[0, :, kv_head_idx, :].contiguous()  # [64, 256]
        expected = torch.zeros(NUM_THREADS, 8, dtype=torch.float32)

        for tid in range(NUM_THREADS):
            warp = tid // 32
            lane = tid % 32
            grp = lane // 4
            sub = lane % 4

            tok0 = warp * 16 + sub * 2
            hd0 = grp           # _md_idx = 0
            hd1 = grp + 8

            toks = [tok0, tok0 + 1, tok0 + 8, tok0 + 9]

            for i, (hd_pos, base) in enumerate([(hd0, 0), (hd1, 4)]):
                for j, t in enumerate(toks):
                    byte_val = v_page[t, hd_pos].item()
                    # Same chain: FP8 E4M3 → FP16 → FP32 → BF16 → FP32
                    fp8_t = torch.tensor([byte_val], dtype=torch.uint8)
                    bf16_val = (fp8_t.view(torch.float8_e4m3fn)
                                .to(torch.bfloat16)
                                .to(torch.float32).item())
                    expected[tid, base + j] = bf16_val

        return got, expected


# ---------------------------------------------------------------------------
# V cache factory
# ---------------------------------------------------------------------------
def make_v_cache(num_kv_heads=4, device="cuda"):
    """Create a V cache with position-encoded FP8 values.

    V[tok, head, dim] = ((tok*17 + dim*7 + head*3) % 125) + 1
    Range 1-125: all valid E4M3 (avoids 0x00=zero, 0x7F/0xFF=NaN).
    """
    v = torch.zeros(1, PAGE_SIZE, num_kv_heads, HEAD_DIM,
                     dtype=torch.uint8, device=device)
    for tok in range(PAGE_SIZE):
        for dim in range(HEAD_DIM):
            for head in range(num_kv_heads):
                v[0, tok, head, dim] = ((tok * 17 + dim * 7
                                          + head * 3) % 125) + 1
    return v


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    device = "cuda"
    num_kv_heads = 4
    kv_head_idx = 0

    print("=" * 60)
    print("V-Loading Isolation Probe")
    print("=" * 60)

    v_cache = make_v_cache(num_kv_heads=num_kv_heads, device=device)
    print(f"V cache: shape={list(v_cache.shape)} dtype={v_cache.dtype}")
    print(f"V[0,0,0,:8] = {v_cache[0,0,0,:8].tolist()}")
    print(f"V[0,1,0,:8] = {v_cache[0,1,0,:8].tolist()}")

    # ---- Phase 1: SMEM byte dump ----
    print("\n--- Phase 1: SMEM byte dump ---")
    p1 = _Phase1()
    print("Compiling Phase 1 kernel...")
    got_f, exp_f = p1.run(v_cache, kv_head_idx=kv_head_idx)

    diff = (got_f.cpu() - exp_f.cpu()).abs()
    n_wrong = (diff > 0.5).sum().item()
    print(f"Total bytes: {V_SMEM_BYTES}")
    print(f"Mismatched bytes: {n_wrong}")
    if n_wrong > 0:
        wrong_idx = (diff > 0.5).nonzero(as_tuple=False).flatten()
        print(f"First 20 wrong indices: {wrong_idx[:20].tolist()}")
        for idx in wrong_idx[:10].tolist():
            tok = idx // HEAD_DIM
            dim = idx % HEAD_DIM
            print(f"  SMEM[{idx}] tok={tok} dim={dim}: "
                  f"got={got_f[idx].item():.0f} "
                  f"exp={exp_f[idx].item():.0f}")
        print("PHASE 1 FAIL — SMEM load is broken")
        return
    else:
        print("PHASE 1 PASS — all SMEM bytes match")

    # ---- Phase 2: Fragment extraction ----
    print("\n--- Phase 2: Per-fragment byte extraction + FP8→BF16 ---")
    p2 = _Phase2()
    print("Compiling Phase 2 kernel...")
    got, exp = p2.run(v_cache, kv_head_idx=kv_head_idx)

    diff2 = (got - exp).abs()
    max_diff = diff2.max().item()
    n_wrong2 = (diff2 > 1e-3).sum().item()
    print(f"Total values: {NUM_THREADS * 8}")
    print(f"Max diff: {max_diff:.6f}")
    print(f"Values with diff > 0.001: {n_wrong2}")

    if n_wrong2 > 0:
        wrong_mask = diff2 > 1e-3
        wrong_tids = wrong_mask.any(dim=1).nonzero(
            as_tuple=False).flatten()
        print(f"\nWrong threads ({len(wrong_tids)} total):")
        slot_names = [
            "V[tok0,   hd0]", "V[tok0+1, hd0]",
            "V[tok0+8, hd0]", "V[tok0+9, hd0]",
            "V[tok0,   hd1]", "V[tok0+1, hd1]",
            "V[tok0+8, hd1]", "V[tok0+9, hd1]",
        ]
        for tid_t in wrong_tids[:20].tolist():
            warp = tid_t // 32
            lane = tid_t % 32
            grp = lane // 4
            sub = lane % 4
            tok0 = warp * 16 + sub * 2
            hd0 = grp
            hd1 = grp + 8
            print(f"  tid={tid_t:3d} warp={warp} grp={grp} sub={sub} "
                  f"tok0={tok0} hd0={hd0} hd1={hd1}")
            for s in range(8):
                if diff2[tid_t, s] > 1e-3:
                    print(f"    {slot_names[s]}: "
                          f"got={got[tid_t, s].item():.6f} "
                          f"exp={exp[tid_t, s].item():.6f} "
                          f"diff={diff2[tid_t, s].item():.6f}")
        print("\nPHASE 2 FAIL — byte extraction or conversion is wrong")
    else:
        print("PHASE 2 PASS — all BF16 values match reference")

    # Summary
    print("\n" + "=" * 60)
    if n_wrong == 0 and n_wrong2 == 0:
        print("ALL PASS — V loading path is correct end-to-end")
        print("If the full kernel still shows ~1-6 error, the issue is")
        print("in how P×V interact in the MMA, or BF16 precision loss")
        print("when quantizing softmax weights before the PV MMA.")
    elif n_wrong == 0 and n_wrong2 > 0:
        print("SMEM correct but extraction wrong →")
        print("  Bug in aligned-load + byte-extract + pack + convert")
    else:
        print("SMEM load itself is wrong →")
        print("  Bug in cooperative GMEM→SMEM copy")
    print("=" * 60)


if __name__ == "__main__":
    main()
