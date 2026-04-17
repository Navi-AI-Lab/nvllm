# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright contributors to the nvllm fork
"""Shim: Qwen3.5 model surface moved to `vllm.nvllm.models.qwen3_5`.

This shim re-exports every public symbol so the upstream registry
(`vllm/model_executor/models/registry.py:1283-1284` hardcodes the
`vllm.model_executor.models.` prefix), `vllm/model_executor/models/colqwen3_5.py`,
and `vllm/model_executor/models/qwen3_5_mtp.py` continue to resolve
their existing import paths without edits.

See `vllm/nvllm/README.md` for the ownership boundary.
"""

from vllm.nvllm.models.qwen3_5 import *  # noqa: F401, F403
