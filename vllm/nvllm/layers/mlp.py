# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright contributors to the nvllm fork
"""Fork-owned dense MLP for Qwen3.5 (and Qwen3.6, same architecture).

Full copy of dense-path `Qwen2MoeMLP` from
`vllm/model_executor/models/qwen2_moe.py:74` with two surface trims:
  - `expert_gate: torch.nn.Linear | None = None` kwarg dropped
    (MoE-expert-gate fast-path never exercised by Qwen3.5-27B dense).
  - `reduce_results: bool = True` kwarg dropped; hardcoded to True in
    the RowParallelLinear(...) call (dense always wants all-reduce).

Composes upstream parallel-linear and activation primitives; does NOT
own those (per Phase C spec §Non-goals). Used at one call-site:
`Qwen3_5DecoderLayer.mlp` when `config.model_type == "qwen3_5_text"`.
"""

import torch
from torch import nn

# Side-effect import: registers torch.ops.vllm.cute_mlp_forward. Importing
# here (rather than lazily inside forward) ensures the op exists at
# torch.compile trace time even for the attached-fusion branch.
import vllm.v1.attention.backends.cute_paged._mlp_op  # noqa: F401
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.linear import (
    MergedColumnParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.quantization import QuantizationConfig


class Qwen3_5MLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.gate_up_proj",
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            reduce_results=True,
            prefix=f"{prefix}.down_proj",
        )
        if hidden_act != "silu":
            raise ValueError(
                f"Unsupported activation: {hidden_act}. Only silu is supported for now."
            )
        self.act_fn = SiluAndMul()

    def forward(self, x):
        # Phase D2 custom-op dispatch. torch.compile treats
        # `torch.ops.vllm.cute_mlp_forward` as opaque — the op body owns
        # both the per-step fuse/fallback gate AND the kernel launch, so
        # the compiled graph sees only one opaque call (no dual-firing,
        # no unfused GEMMs leaking into Inductor). `_cute_layer_name` is
        # set by `CutePagedImpl.attach_mlp_fusion` iff fusion is wired up
        # (env var on, dense MLP, NVFP4 weights, shape match). When unset
        # (env var off, MTP layer, shape mismatch, non-NVFP4), fall
        # through to the original unfused body — identical to upstream.
        layer_name = getattr(self, "_cute_layer_name", None)
        if layer_name is None:
            gate_up, _ = self.gate_up_proj(x)
            out = self.act_fn(gate_up)
            out, _ = self.down_proj(out)
            return out

        output = torch.empty_like(x)
        torch.ops.vllm.cute_mlp_forward(x, output, layer_name)
        return output

        # Phase D (pre-D1) gate — left for reference. Replaced by the
        # custom-op path above because torch.compile was dead-branching
        # the `if impl._mlp_fusion_active:` check at trace time.
        # impl = getattr(self, "_cute_impl", None)
        # if impl is not None and getattr(impl, "_mlp_fusion_active", False):
        #     nat = impl._mlp_fusion_nat
        #     return impl.mlp_output[:nat]
        # gate_up, _ = self.gate_up_proj(x)
        # out = self.act_fn(gate_up)
        # out, _ = self.down_proj(out)
        # return out
