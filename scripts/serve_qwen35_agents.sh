#!/bin/bash
# nvllm serve -- Qwen3.5-122B-A10B (NVFP4) agent config inside container
#
# Long-context agent server with tool calling.
# Clients should send chat_template_kwargs.enable_thinking = false to disable thinking.
#
# Usage (from host):
#   docker run --gpus all --ipc=host --network host \
#     -v ~/.cache/huggingface:/root/.cache/huggingface \
#     -v ~/.cache/flashinfer:/root/.cache/flashinfer \
#     --entrypoint bash ghcr.io/navi-ai-lab/nvllm:latest \
#     scripts/serve_qwen35_agents.sh
set -euo pipefail

export VLLM_NVFP4_GEMM_BACKEND=cutlass
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

exec vllm serve \
  --model Sehyo/Qwen3.5-122B-A10B-NVFP4 \
  --served-model-name default \
  --host 0.0.0.0 --port 8000 \
  --kv-cache-dtype turboquant35 \
  --attention-backend TRITON_ATTN \
  --max-model-len 65536 \
  --max-num-seqs 4 \
  --language-model-only \
  --enable-prefix-caching \
  --trust-remote-code \
  --gpu-memory-utilization 0.85 \
  --max-num-batched-tokens 16384 \
  --speculative-config '{"method": "mtp", "num_speculative_tokens": 1}' \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --compilation-config '{"cudagraph_mode":"PIECEWISE"}'
