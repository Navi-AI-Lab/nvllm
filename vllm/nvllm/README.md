# nvllm — owned-stack subpackage

Fork-owned code for the CuTe paged attention + fusion stack. Code here does
NOT subclass upstream model / layer classes: renames in upstream vLLM must
not silently break fusion wiring.

## Phase B (this subpackage, shipped 2026-04-17)

- `vllm/nvllm/models/qwen3_5.py` — self-contained Qwen3.5 model with
  `Qwen3_5Attention` inlined from the current fusion-patched
  `Qwen3NextAttention`. `vllm/model_executor/models/qwen3_5.py` is a 1-line
  re-export shim so the upstream registry keeps working unchanged.

## Phase C (next, gated before uber-kernel Phase D+E)

- `vllm/nvllm/layers/` for RMSNorm, MLP, embedding. Required before fusion
  grows to cover MLP / embedding / head.

## Registry

Registry loader at `vllm/model_executor/models/registry.py:1283-1284` hardcodes
a `vllm.model_executor.models.<mod_relname>` prefix. Rather than modify the
loader, we ship a shim at the old module path that re-exports everything here.
