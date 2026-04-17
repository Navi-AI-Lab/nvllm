# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright contributors to the nvllm fork
"""nvllm layer primitives — fork-owned RMSNorm, MLP, etc.

Classes here are full copies of upstream class bodies (not subclasses).
Upstream renames must not silently break fusion wiring. Only layer primitives
the uber-kernel will fuse against live here — embedding, MoE-block, parallel-
linear, and activation primitives remain upstream (see Phase C spec §Non-goals).
"""

from vllm.nvllm.layers.layernorm import Qwen3_5RMSNorm
from vllm.nvllm.layers.mlp import Qwen3_5MLP

__all__ = ["Qwen3_5RMSNorm", "Qwen3_5MLP"]
