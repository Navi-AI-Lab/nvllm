# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright contributors to the nvllm fork
"""Phase D FP4 write-side quantization primitive and pure-PyTorch reference.

Reference: b12x @ c469c66 cute/fp4 module; TRT-LLM fp4_quantize_kernel.
See `docs/superpowers/specs/2026-04-17-unreal-kernel-phase-d-mlp-fusion-design.md`
section "New CuTe DSL Primitive" for math.
"""

from __future__ import annotations
import torch

# FP4 E2M1 representable magnitudes (positive side only); sign bit adds ± variant.
# Values: {0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0}
FP4_VALUES = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
FP4_MAX = 6.0
FP4_BLOCK_SIZE = 16


def encode_ue4m3(scale: torch.Tensor) -> torch.Tensor:
    """Encode FP32 scale to UE4M3 (unsigned E4M3): 4-bit exponent, 3-bit mantissa.

    Returns uint8 tensor where each element is the UE4M3 encoding of the input.
    Matches the hardware ``cvt.rn.satfinite.e4m3x2.f32`` round-to-nearest
    behavior, including carry propagation from mantissa overflow. The old
    implementation clamped the mantissa to [0, 7] before carry detection,
    which caused boundary values (where rounding bumps the exponent) to
    encode differently than the hardware and mismatched the CuTe kernel's
    FP4 quantize stage by 1 exponent.
    """
    # UE4M3: exp_bias = 7, max finite = 2^8 * (1 + 7/8) = 480
    scale = scale.clamp(min=2 ** -9, max=448.0)
    log2_scale = torch.log2(scale)
    exp = torch.floor(log2_scale).clamp(-6, 8).to(torch.int8)
    mantissa_linear = scale / (2.0 ** exp.to(torch.float32)) - 1.0  # in [0, 1]
    # Round-to-nearest FIRST (may produce 8), then detect carry, then clamp.
    mantissa_rounded = torch.round(mantissa_linear * 8).to(torch.int32)
    carry = mantissa_rounded == 8
    mantissa_bits = torch.where(
        carry, torch.zeros_like(mantissa_rounded), mantissa_rounded
    ).clamp(0, 7).to(torch.uint8)
    exp = torch.where(carry, (exp.to(torch.int32) + 1).to(torch.int8), exp)
    biased_exp = (exp + 7).to(torch.int32).clamp(0, 15).to(torch.uint8)
    return (biased_exp << 3) | mantissa_bits


def decode_ue4m3(code: torch.Tensor) -> torch.Tensor:
    """Decode UE4M3 uint8 back to FP32 scale."""
    biased_exp = (code >> 3).to(torch.int32)
    mantissa = (code & 0x7).to(torch.float32)
    exp = biased_exp - 7
    return (1.0 + mantissa / 8.0) * (2.0 ** exp.to(torch.float32))


def quantize_fp4_block_reference(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a 1-D BF16/FP32 tensor to FP4 + UE4M3 blockscales.

    Returns:
        fp4_packed: uint8 tensor of shape [ceil(N/2)], two FP4 nibbles per byte.
        blockscale_ue4m3: uint8 tensor of shape [ceil(N/FP4_BLOCK_SIZE)].
    """
    x = x.to(torch.float32)
    n = x.numel()
    assert n % FP4_BLOCK_SIZE == 0, f"N must be multiple of {FP4_BLOCK_SIZE}"
    num_blocks = n // FP4_BLOCK_SIZE

    x_blocked = x.view(num_blocks, FP4_BLOCK_SIZE)
    max_abs = x_blocked.abs().max(dim=1).values  # [num_blocks]
    scale_fp32 = (max_abs / FP4_MAX).clamp(min=1e-12)
    blockscale_ue4m3 = encode_ue4m3(scale_fp32)

    # Requantize x with the encoded-then-decoded scale so FP4 values match kernel
    scale_roundtrip = decode_ue4m3(blockscale_ue4m3)
    x_scaled = x_blocked / scale_roundtrip.unsqueeze(1)  # [num_blocks, 16] in FP4 range
    # Round each element to nearest FP4 value, preserving sign
    sign = torch.sign(x_scaled)
    mag = x_scaled.abs()
    fp4_vals = FP4_VALUES.to(x.device)
    # Find nearest index for each element
    distances = (mag.unsqueeze(-1) - fp4_vals.view(1, 1, -1)).abs()
    nearest_idx = distances.argmin(dim=-1).to(torch.uint8)  # [num_blocks, 16]
    # Sign bit in FP4: MSB
    fp4_nibbles = nearest_idx | ((sign < 0).to(torch.uint8) << 3)
    fp4_nibbles = fp4_nibbles.view(-1)  # [N]
    # Pack 2 nibbles per byte (low nibble first, high nibble second)
    fp4_packed = (fp4_nibbles[0::2] | (fp4_nibbles[1::2] << 4))
    return fp4_packed, blockscale_ue4m3


def dequantize_fp4_block_reference(
    fp4_packed: torch.Tensor,
    blockscale_ue4m3: torch.Tensor,
) -> torch.Tensor:
    """Inverse of `quantize_fp4_block_reference` — returns FP32."""
    num_blocks = blockscale_ue4m3.numel()
    # Unpack nibbles
    low_nibbles = fp4_packed & 0xF
    high_nibbles = (fp4_packed >> 4) & 0xF
    nibbles = torch.stack([low_nibbles, high_nibbles], dim=1).view(-1)
    # Decode sign + magnitude
    sign = torch.where(nibbles & 0x8 != 0, torch.tensor(-1.0), torch.tensor(1.0))
    idx = (nibbles & 0x7).to(torch.int64)
    fp4_vals = FP4_VALUES.to(fp4_packed.device)
    mag = fp4_vals[idx]
    values = sign * mag  # [N]
    values = values.view(num_blocks, FP4_BLOCK_SIZE)
    # Apply blockscale
    scale = decode_ue4m3(blockscale_ue4m3).unsqueeze(1)
    return (values * scale).view(-1)


if __name__ == "__main__":
    # Round-trip test: quantize(x) -> dequantize -> should be close to x within 1 ULP
    # for values in [-6, +6] at block granularity.
    torch.manual_seed(0)
    x = torch.randn(256).clamp(-6, 6)
    fp4, scale = quantize_fp4_block_reference(x)
    x_rt = dequantize_fp4_block_reference(fp4, scale)
    err = (x - x_rt).abs().max().item()
    # FP4 worst-case noise per block: scale * 1.0 where scale = max_abs/6.
    # With clamp(-6, 6), max_abs ≤ 6 → worst-case absolute err ≤ 1.0.
    # 0.6 is a safe empirical bound for torch.randn(256) (observed ~0.37).
    assert err < 0.6, f"Round-trip error too large: {err}"
    print(f"FP4 round-trip PASSED: max_err={err:.4f}")

    # Block-size boundary test
    assert fp4.numel() == 256 // 2
    assert scale.numel() == 256 // FP4_BLOCK_SIZE
    print("FP4 shape PASSED")


# =============================================================================
# CuTe DSL kernel-side primitives (inline PTX)
# =============================================================================
# These helpers are callable ONLY from within a CuTe kernel compiled by the DSL.
# They wrap SM120 native PTX conversions and are the write-side counterpart to
# `_fp4_nibble_to_f32` / `_ld_shared_u8` in kernel.py.
#
# References (pinned to b12x @ c469c66):
#   cvt_f32_to_e4m3   — github.com/lukealonso/b12x/blob/c469c66/cute/fp4.py#L1462
#   cvt_e2m1x8_f32    — github.com/lukealonso/b12x/blob/c469c66/cute/fp4.py#L1654
# =============================================================================

try:
    import cutlass  # noqa: F401
    from cutlass._mlir.dialects import llvm as _llvm_dialect
    from cutlass.cute.typing import Float32, Int32, Int64, Uint32
    from cutlass.cutlass_dsl import T, dsl_user_op
    _HAS_CUTE_DSL = True
except ImportError:
    _HAS_CUTE_DSL = False


if _HAS_CUTE_DSL:

    @dsl_user_op
    def _encode_ue4m3_f32_to_u8(scale: Float32, *, loc=None, ip=None) -> Int32:
        """Encode FP32 scale (positive) to UE4M3 packed into low byte of a u32.

        Uses SM120 native ``cvt.rn.satfinite.e4m3x2.f32`` which packs two
        FP32→FP8 (E4M3) conversions into a b16 register. We pass the scale as
        the low operand and zero as the high operand, then extract the low
        byte. UE4M3 == E4M3 when the input is non-negative (sign bit == 0).

        Returned as Int32 so callers can OR/shift/mask in register space.

        Reference: b12x cvt_f32_to_e4m3 (fp4.py:1462).
        """
        scale_ir = Float32(scale).ir_value(loc=loc, ip=ip)
        result_ir = _llvm_dialect.inline_asm(
            T.i32(),
            [scale_ir],
            # Single-line PTX: pack (zero, scale) -> b16 -> extract low byte.
            # cvt emits {high_byte, low_byte} in the b16, with operand order
            # (high=$2, low=$1). We want $1 = scale in the LOW byte.
            "{ .reg .b16 pair; .reg .f32 zero; mov.f32 zero, 0f00000000;"
            " cvt.rn.satfinite.e4m3x2.f32 pair, zero, $1;"
            " cvt.u32.u16 $0, pair; and.b32 $0, $0, 0xFF; }",
            "=r,f",
            has_side_effects=False,
            asm_dialect=0,
            loc=loc, ip=ip,
        )
        return Int32(result_ir)

    @dsl_user_op
    def _f32_to_fp4_nibble(
        value: Float32, scale_rcp: Float32, *,
        loc=None, ip=None,
    ) -> Int32:
        """Convert one FP32 value (pre-multiplied by reciprocal scale) to a
        single FP4 E2M1 nibble (sign bit + 3 magnitude bits) in the low 4 bits
        of an Int32.

        Implementation: multiply ``value * scale_rcp`` (host-side would do
        this, but we fuse here for register economy), then use the native
        ``cvt.rn.satfinite.e2m1x2.f32`` instruction which packs two f32→fp4
        conversions into one byte. We pair our value with a zero sentinel
        and keep only the low nibble.

        Operand order: ``cvt.rn.satfinite.e2m1x2.f32 byte, hi_src, lo_src;``
        produces byte with ``lo_src`` nibble in bits [3:0] and ``hi_src`` in
        bits [7:4]. We put the real value in the LOW position (hi=zero).

        Reference: b12x cvt_e2m1x8_f32 (fp4.py:1654).
        """
        value_ir = Float32(value).ir_value(loc=loc, ip=ip)
        rcp_ir = Float32(scale_rcp).ir_value(loc=loc, ip=ip)
        result_ir = _llvm_dialect.inline_asm(
            T.i32(),
            [value_ir, rcp_ir],
            # Pre-scale, then native cvt, then mask low nibble.
            "{ .reg .f32 scaled, zero; .reg .b8 byte;"
            " mul.f32 scaled, $1, $2;"
            " mov.f32 zero, 0f00000000;"
            " cvt.rn.satfinite.e2m1x2.f32 byte, zero, scaled;"
            " cvt.u32.u8 $0, byte; and.b32 $0, $0, 0xF; }",
            "=r,f,f",
            has_side_effects=False,
            asm_dialect=0,
            loc=loc, ip=ip,
        )
        return Int32(result_ir)

    @dsl_user_op
    def _st_shared_u8(smem_addr: Int64, value: Int32, *,
                      loc=None, ip=None) -> None:
        """Store the low 8 bits of ``value`` to shared memory at ``smem_addr``.

        PTX ``st.shared.b8`` accepts a b32 register and truncates to the
        low byte, matching the per-byte store pattern in kernel.py:581-599
        (b0..b3 there are `.reg .b32` temporaries).
        """
        addr_ir = Int64(smem_addr).ir_value(loc=loc, ip=ip)
        val_ir = Int32(value).ir_value(loc=loc, ip=ip)
        _llvm_dialect.inline_asm(
            None,
            [addr_ir, val_ir],
            "st.shared.b8 [$0], $1;",
            "l,r",
            has_side_effects=True,
            asm_dialect=0,
            loc=loc, ip=ip,
        )
