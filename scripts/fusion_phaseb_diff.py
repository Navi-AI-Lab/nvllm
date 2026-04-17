"""Phase B W_O GEMV — math isolation harness.

Compares three views of the W_O dequant+GEMV computation:
  (1) helpers.nvfp4_dequant from raw (unswizzled) scales
  (2) our emulator using the kernel's swizzled-load indexing
  (3) full matmul reference: attn @ W_dq.T vs per-CTA K-slice accumulation

If (1) == (2) and matmul == per-CTA sum, the kernel's Phase B formulas are correct,
and the remaining bug is in runtime (CuTe compile-time specialization or sync).

Run: .venv/bin/python scripts/fusion_phaseb_diff.py
"""

from __future__ import annotations

import sys

import torch

sys.path.insert(0, "/home/natfii/docker/nvllm")
sys.path.insert(0, "/home/natfii/.claude/skills/kernel-math-debug")

from helpers import (  # type: ignore  # noqa: E402
    compare,
    nvfp4_dequant,
)
from safetensors import safe_open  # noqa: E402

from vllm.model_executor.layers.quantization.utils.nvfp4_utils import (  # noqa: E402
    swizzle_blockscale,
)

MODEL = (
    "/home/natfii/.cache/huggingface/hub/"
    "models--natfii--Qwen3.5-27B-NVFP4-Opus-GB10/snapshots/"
    "1496fc6e90a170fe575051d292284ecaa7053b6b/model.safetensors"
)
LAYER = 3

NUM_Q_HEADS = 24
NUM_KV_HEADS = 4
HEAD_DIM = 256
GROUP_SIZE = NUM_Q_HEADS // NUM_KV_HEADS  # 6
HIDDEN_DIM = 5120
K_DIM = NUM_Q_HEADS * HEAD_DIM  # 6144
NUM_K_GROUPS = K_DIM // 16  # 384
NUM_K_TILES = (NUM_K_GROUPS + 3) // 4  # 96 — matches kernel wo_nkt


def kernel_swizzled_scale_offset(m: int, k_group: int, num_k_tiles: int) -> int:
    """Mirror the kernel's _ld_swizzled_scale offset math."""
    m_tile = m >> 7
    outer_m = m & 31
    inner_m = (m >> 5) & 3
    k_tile = k_group >> 2
    inner_k = k_group & 3
    return (m_tile * num_k_tiles + k_tile) * 512 + outer_m * 16 + inner_m * 4 + inner_k


def main() -> None:
    print("== Loading layer", LAYER, "o_proj from checkpoint ==")
    with safe_open(MODEL, framework="pt", device="cuda") as sf:
        W_packed = sf.get_tensor(f"model.layers.{LAYER}.self_attn.o_proj.weight_packed")
        S_raw = sf.get_tensor(f"model.layers.{LAYER}.self_attn.o_proj.weight_scale")
        GS_stored = sf.get_tensor(
            f"model.layers.{LAYER}.self_attn.o_proj.weight_global_scale"
        )

    print(f"  W_packed: {list(W_packed.shape)} {W_packed.dtype}")
    print(f"  S_raw: {list(S_raw.shape)} {S_raw.dtype}")
    print(f"  GS_stored (divisor): {GS_stored.item():.6f}")

    # Mirror process_weights_after_loading
    S_swizzled = swizzle_blockscale(S_raw)
    GS = (1.0 / GS_stored.max().to(torch.float32)).to(torch.float32)
    print(f"  S_swizzled: {list(S_swizzled.shape)} {S_swizzled.dtype}")
    print(f"  GS (true weight_global_scale): {GS.item():.6f}\n")

    # Reference dequant from raw scales
    print("== Reference dequant (helpers, raw scales) ==")
    W_dq_ref = nvfp4_dequant(W_packed, S_raw, GS).to("cuda")
    print(f"  W_dq_ref: {list(W_dq_ref.shape)} absmax={W_dq_ref.abs().max().item():.4f}\n")

    # --- 1. Scale load sanity: swizzled index vs raw index ---
    print("== Scale-load sanity: swizzled offset vs raw[n, kg] ==")
    S_swizzled_flat = S_swizzled.reshape(-1).to("cuda")
    S_raw_cuda = S_raw.to("cuda")

    mismatches = 0
    torch.manual_seed(0)
    sample_pairs = [(0, 0), (0, 5), (100, 50), (5119, 383), (2000, 200),
                    (128, 0), (128, 4), (127, 3), (63, 383)]
    for n, kg in sample_pairs:
        off = kernel_swizzled_scale_offset(n, kg, NUM_K_TILES)
        sw = S_swizzled_flat[off].float().item()
        raw = S_raw_cuda[n, kg].float().item()
        match = abs(sw - raw) < 1e-6
        if not match:
            mismatches += 1
        print(f"  n={n:4d} kg={kg:3d}  swizzled={sw:.6f}  raw={raw:.6f}  match={match}")
    if mismatches:
        print(f"\n  !! {mismatches} scale mismatches — swizzle/offset math disagree.\n")
    else:
        print("  ✓ all sample (n, kg) pairs match across swizzle and raw\n")

    # --- 2. Full-matrix scale reconstruction via swizzle path ---
    print("== Full scale reconstruction via kernel swizzle path ==")
    # Vectorized equivalent of kernel_swizzled_scale_offset over all (n, kg).
    n_idx = torch.arange(HIDDEN_DIM, device="cuda")
    k_idx = torch.arange(NUM_K_GROUPS, device="cuda")
    N, K = torch.meshgrid(n_idx, k_idx, indexing="ij")
    m_tile = N >> 7
    outer_m = N & 31
    inner_m = (N >> 5) & 3
    k_tile = K >> 2
    inner_k = K & 3
    sf_offset = (m_tile * NUM_K_TILES + k_tile) * 512 + outer_m * 16 + inner_m * 4 + inner_k
    sf_from_swizzle = S_swizzled_flat[sf_offset.view(-1)].view(HIDDEN_DIM, NUM_K_GROUPS).float()
    compare(sf_from_swizzle, S_raw_cuda.float(), name="scale matrix (swizzle-path vs raw)")
    print()

    # --- 3. Dequant via kernel indexing (swizzle path) ---
    print("== W dequant via kernel indexing (swizzle path) ==")
    W_packed_cuda = W_packed.to("cuda")
    low_nib = (W_packed_cuda & 0x0F).to(torch.int64)
    high_nib = ((W_packed_cuda >> 4) & 0x0F).to(torch.int64)
    W_nibbles = torch.empty(HIDDEN_DIM, K_DIM, dtype=torch.int64, device="cuda")
    W_nibbles[:, 0::2] = low_nib
    W_nibbles[:, 1::2] = high_nib

    fp4_lut = torch.tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
         -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
        dtype=torch.float32,
        device="cuda",
    )
    W_fp = fp4_lut[W_nibbles]
    sf_expanded = sf_from_swizzle.repeat_interleave(16, dim=1)
    W_dq_emu = W_fp * sf_expanded * GS.item()
    compare(W_dq_emu, W_dq_ref, name="W_dq kernel-swizzle vs helpers raw")
    print()

    # --- 4. Full GEMV accumulation: monolithic matmul vs per-CTA K-slice sum ---
    print("== Full GEMV: matmul vs per-CTA K-slice accumulation ==")
    torch.manual_seed(0)
    NUM_SEQS = 2
    attn_output = (
        torch.randn(NUM_SEQS, NUM_Q_HEADS, HEAD_DIM, dtype=torch.bfloat16, device="cuda")
        * 0.1
    )
    attn_flat = attn_output.view(NUM_SEQS, K_DIM).float()

    ref_wo_full = attn_flat @ W_dq_emu.T

    emu_wo = torch.zeros(NUM_SEQS, HIDDEN_DIM, dtype=torch.float32, device="cuda")
    for kv_head_idx in range(NUM_KV_HEADS):
        k_start = kv_head_idx * GROUP_SIZE * HEAD_DIM
        k_end = k_start + GROUP_SIZE * HEAD_DIM
        partial = attn_flat[:, k_start:k_end] @ W_dq_emu[:, k_start:k_end].T
        emu_wo += partial
    compare(emu_wo, ref_wo_full, name="GEMV per-CTA sum vs single matmul")
    print()

    # --- 5. Final: emulator vs helpers reference end-to-end ---
    print("== End-to-end: per-CTA emulator vs helpers reference ==")
    helpers_ref = attn_flat @ W_dq_ref.T
    compare(emu_wo, helpers_ref, name="per-CTA emulator vs helpers reference")
    print()

    print("== Summary ==")
    print("If all three checks are 'close=True', Phase B formulas are correct.")
    print("The remaining bug is runtime — CuTe DSL specializing on Int32 fusion flags,")
    print("or Phase A→B memory visibility, or actual launch path.")


if __name__ == "__main__":
    main()
