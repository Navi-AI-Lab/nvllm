#!/bin/bash
# nvllm serve -- Qwen3.5-122B-A10B (NVFP4) inside container
#
# MoE model with MTP speculative decoding for batched throughput.
#
# Usage (from host):
#   docker run --gpus all --ipc=host --network host \
#     -v ~/.cache/huggingface:/root/.cache/huggingface \
#     -v ~/.cache/flashinfer:/root/.cache/flashinfer \
#     --entrypoint bash ghcr.io/navi-ai-lab/nvllm:latest \
#     scripts/serve_qwen35.sh
set -euo pipefail

export VLLM_NVFP4_GEMM_BACKEND=cutlass
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

exec vllm serve \
  --model Sehyo/Qwen3.5-122B-A10B-NVFP4 \
  --served-model-name default \
  --host 0.0.0.0 --port 8000 \
  --kv-cache-dtype turboquant35 \
  --attention-backend TRITON_ATTN \
  --max-model-len 32768 \
  --max-num-seqs 4 \
  --language-model-only \
  --enable-prefix-caching \
  --trust-remote-code \
  --gpu-memory-utilization 0.85 \
  --max-num-batched-tokens 16384 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --speculative-config '{"method": "mtp", "num_speculative_tokens": 1}' \
  --compilation-config '{"cudagraph_mode":"PIECEWISE"}'
