#!/bin/bash
# nvllm serve -- Qwen3.5-27B-NVFP4-Opus inside container
#
# Dense Qwen3.5 with hybrid attention (linear + full). ~18 GB NVFP4.
# No prefix caching — corrupts SSM state in hybrid attention models.
#
# Usage (from host):
#   docker run --gpus all --ipc=host --network host \
#     -v ~/.cache/huggingface:/root/.cache/huggingface \
#     -v ~/.cache/flashinfer:/root/.cache/flashinfer \
#     --entrypoint bash ghcr.io/navi-ai-lab/nvllm:latest \
#     scripts/serve_qwen35_27b.sh
set -euo pipefail

export VLLM_NVFP4_GEMM_BACKEND=cutlass
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

exec vllm serve \
  --model natfii/Qwen3.5-27B-NVFP4-Opus-GB10 \
  --served-model-name default \
  --host 0.0.0.0 --port 8000 \
  --kv-cache-dtype auto \
  --attention-backend triton_attn \
  --max-model-len 65536 \
  --max-num-seqs 4 \
  --language-model-only \
  --mamba-cache-mode align \
  --mamba-block-size 64 \
  --trust-remote-code \
  --gpu-memory-utilization 0.80 \
  --max-num-batched-tokens 65536 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --compilation-config '{"cudagraph_mode":"PIECEWISE"}'
