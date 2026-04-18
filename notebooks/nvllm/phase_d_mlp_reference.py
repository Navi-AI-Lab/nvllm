# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright contributors to the nvllm fork
"""Pure-PyTorch fused MLP reference for Phase D kernel validation.

Implements `down(silu(gate(x)) * up(x))` with an explicit per-slice
structure that mirrors the CuTe kernel's iteration order. Used by
Tier-1 smoke tests to check numeric equivalence before Docker.
"""

from __future__ import annotations
import torch
from torch import nn

from vllm.v1.attention.backends.cute_paged._fp4_writer import (
    FP4_BLOCK_SIZE,
    dequantize_fp4_block_reference,
    quantize_fp4_block_reference,
)


def fused_mlp_reference(
    x: torch.Tensor,           # [nat, hidden_size] BF16
    gate_w: torch.Tensor,      # [intermediate, hidden] FP32 (dequantized)
    up_w: torch.Tensor,        # [intermediate, hidden] FP32 (dequantized)
    down_w: torch.Tensor,      # [hidden, intermediate] FP32 (dequantized)
    tile_s: int = 256,
    quantize_intermediate: bool = True,
    bf16_intermediate: bool = False,
) -> torch.Tensor:
    """Reference fused MLP: output = down(silu(gate(x)) * up(x)).

    When `quantize_intermediate=True`, the intermediate is round-tripped
    through FP4 blockscale to match the kernel's actual behavior.

    When `bf16_intermediate=True`, the intermediate is first round-tripped
    through BF16 before FP4 quant — the kernel stores silu(gate)*up as BF16
    in SMEM before FP4-quantizing, so the end-to-end kernel test sets this
    True to match the kernel's numeric path. Default False preserves the
    original behavior for prior callers.
    """
    nat, hidden = x.shape
    interm = gate_w.shape[0]
    x32 = x.to(torch.float32)
    output = torch.zeros(nat, hidden, dtype=torch.float32, device=x.device)

    assert interm % tile_s == 0, f"interm={interm} must be multiple of tile_s={tile_s}"
    num_slices = interm // tile_s

    for s in range(num_slices):
        s_start = s * tile_s
        s_end = s_start + tile_s
        gate_slice = x32 @ gate_w[s_start:s_end, :].t()       # [nat, tile_s]
        up_slice = x32 @ up_w[s_start:s_end, :].t()           # [nat, tile_s]
        silu_gate = gate_slice * torch.sigmoid(gate_slice)
        interm_slice = silu_gate * up_slice                   # [nat, tile_s]

        if bf16_intermediate:
            # Kernel stores silu(gate)*up as BF16 in SMEM before quant.
            interm_slice = interm_slice.to(torch.bfloat16).to(torch.float32)

        if quantize_intermediate:
            # Per-row FP4 round-trip, matches kernel's per-slice quantize
            for t in range(nat):
                fp4, scale = quantize_fp4_block_reference(interm_slice[t])
                interm_slice[t] = dequantize_fp4_block_reference(fp4, scale)

        # FC2 contribution from this slice
        output += interm_slice @ down_w[:, s_start:s_end].t()

    return output.to(x.dtype)
