# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright contributors to the nvllm fork
"""Fork-owned RMS normalization for Qwen3.5 (and Qwen3.6, same architecture).

Full copy of `GemmaRMSNorm` from `vllm/model_executor/layers/layernorm.py:360`
with only two changes:
  1. Class renamed `GemmaRMSNorm` -> `Qwen3_5RMSNorm`.
  2. CustomOp registration key `"gemma_rms_norm"` -> `"qwen3_5_rms_norm"`
     (avoids collision with upstream GemmaRMSNorm's existing registration).

Math is identical: x -> (1 + w) * x / sqrt(E[x^2] + eps) cast back to orig
dtype at the very end (Gemma variant). Used at five call-sites in
`vllm/nvllm/models/qwen3_5.py`: q_norm and k_norm (head-dim hidden_size),
input_layernorm, post_attention_layernorm, and model-level norm.
"""

import torch
from torch import nn

from vllm.model_executor.custom_op import CustomOp


@CustomOp.register("qwen3_5_rms_norm")
class Qwen3_5RMSNorm(CustomOp):
    """RMS normalization for Qwen3.5.

    Identical semantics to `GemmaRMSNorm`:
        1. x * (1 + w) instead of x * w.
        2. (x * w).to(orig_dtype) instead of x.to(orig_dtype) * w.
    """

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    @staticmethod
    def _forward_static_no_residual(
        weight: torch.Tensor,
        variance_epsilon: float,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """PyTorch-native implementation equivalent to forward() without residual."""
        orig_dtype = x.dtype
        x = x.float()
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(variance + variance_epsilon)
        x = x * (1.0 + weight.float())
        x = x.to(orig_dtype)
        return x

    @staticmethod
    def _forward_static_with_residual(
        weight: torch.Tensor,
        variance_epsilon: float,
        x: torch.Tensor,
        residual: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """PyTorch-native implementation equivalent to forward() with residual."""
        orig_dtype = x.dtype
        x = (
            x.float() + residual.float()
            if orig_dtype == torch.float16
            else x + residual
        )
        residual = x

        x = x.float()
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(variance + variance_epsilon)
        # Llama does x.to(float16) * w whilst Gemma is (x * w).to(float16)
        # See https://github.com/huggingface/transformers/pull/29402
        x = x * (1.0 + weight.float())
        x = x.to(orig_dtype)
        return x, residual

    def forward_native(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """PyTorch-native implementation equivalent to forward()."""
        if residual is None:
            return self._forward_static_no_residual(
                self.weight.data, self.variance_epsilon, x
            )
        else:
            return self._forward_static_with_residual(
                self.weight.data, self.variance_epsilon, x, residual
            )

    def forward_cuda(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if torch.compiler.is_compiling():
            return self.forward_native(x, residual)

        if not getattr(self, "_is_compiled", False):
            self._forward_static_no_residual = torch.compile(  # type: ignore
                self._forward_static_no_residual
            )
            self._forward_static_with_residual = torch.compile(  # type: ignore
                self._forward_static_with_residual
            )
            self._is_compiled = True
        return self.forward_native(x, residual)
