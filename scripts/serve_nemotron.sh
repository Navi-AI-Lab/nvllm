#!/bin/bash
# nvllm serve -- Nemotron-3-Super-120B-A12B (NVFP4) inside container
#
# MoE model with FlashInfer attention and tool calling.
# Chat template disables thinking tags for cleaner output.
#
# Usage (from host):
#   docker run --gpus all --ipc=host --network host \
#     -v ~/.cache/huggingface:/root/.cache/huggingface \
#     -v ~/.cache/flashinfer:/root/.cache/flashinfer \
#     -v ~/.cache/vllm_compile:/root/.cache/vllm/torch_compile_cache \
#     --entrypoint bash ghcr.io/navi-ai-lab/nvllm:latest \
#     scripts/serve_nemotron.sh
set -euo pipefail

export VLLM_NVFP4_GEMM_BACKEND=cutlass
export VLLM_USE_FLASHINFER_MOE_FP4=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

exec vllm serve \
  --model nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4 \
  --served-model-name default \
  --host 0.0.0.0 --port 8000 \
  --kv-cache-dtype turboquant35 \
  --attention-backend TRITON_ATTN \
  --max-model-len 131072 \
  --max-num-seqs 4 \
  --quantization modelopt_mixed \
  --language-model-only \
  --enable-prefix-caching \
  --trust-remote-code \
  --gpu-memory-utilization 0.85 \
  --max-num-batched-tokens 16384 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --chat-template /templates/nemotron_no_think.jinja \
  --compilation-config '{"cudagraph_mode":"PIECEWISE"}'
