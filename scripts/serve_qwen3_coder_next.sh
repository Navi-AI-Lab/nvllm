#!/bin/bash
# nvllm serve -- Qwen3-Coder-Next (NVFP4) inside container
#
# Hybrid DeltaNet architecture with tool calling for coding agents.
# Clients should send chat_template_kwargs.enable_thinking = false to disable thinking.
#
# Usage (from host):
#   docker run --gpus all --ipc=host --network host \
#     -v ~/.cache/huggingface:/root/.cache/huggingface \
#     -v ~/.cache/flashinfer:/root/.cache/flashinfer \
#     --entrypoint bash ghcr.io/navi-ai-lab/nvllm:latest \
#     scripts/serve_qwen3_coder_next.sh
set -euo pipefail

export VLLM_NVFP4_GEMM_BACKEND=cutlass
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

exec vllm serve \
  --model GadflyII/Qwen3-Coder-Next-NVFP4 \
  --served-model-name default \
  --host 0.0.0.0 --port 8000 \
  --kv-cache-dtype turboquant35 \
  --attention-backend TRITON_ATTN \
  --max-model-len 131072 \
  --max-num-seqs 4 \
  --enable-prefix-caching \
  --trust-remote-code \
  --gpu-memory-utilization 0.85 \
  --max-num-batched-tokens 16384 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --compilation-config '{"cudagraph_mode":"PIECEWISE"}'
