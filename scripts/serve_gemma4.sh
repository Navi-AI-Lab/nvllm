#!/bin/bash
# nvllm serve -- Gemma 4 31B IT (NVFP4) inside container
#
# Dense vision-language model with ngram speculative decoding.
# Model must be quantized locally — bind mount the checkpoint as /model.
#
# Usage (from host):
#   docker run --gpus all --ipc=host --network host \
#     -v /path/to/gemma-4-31B-it-NVFP4:/model \
#     -v ~/.cache/huggingface:/root/.cache/huggingface \
#     -v ~/.cache/flashinfer:/root/.cache/flashinfer \
#     -v ~/.cache/vllm_compile:/root/.cache/vllm/torch_compile_cache \
#     --entrypoint bash ghcr.io/navi-ai-lab/nvllm:latest \
#     scripts/serve_gemma4.sh
set -euo pipefail

export VLLM_NVFP4_GEMM_BACKEND=cutlass
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Gemma4 requires a newer numba than what ships in the image
pip install --no-deps 'numba>=0.65' 2>&1 | tail -1

exec vllm serve \
  --model /model \
  --served-model-name default \
  --host 0.0.0.0 --port 8000 \
  --kv-cache-dtype turboquant35 \
  --attention-backend TRITON_ATTN \
  --max-model-len 32768 \
  --max-num-seqs 4 \
  --quantization modelopt_fp4 \
  --language-model-only \
  --enable-prefix-caching \
  --trust-remote-code \
  --gpu-memory-utilization 0.85 \
  --max-num-batched-tokens 16384 \
  --enable-auto-tool-choice \
  --tool-call-parser gemma4 \
  --speculative-config '{"method": "ngram", "num_speculative_tokens": 3, "prompt_lookup_max": 3}' \
  --compilation-config '{"cudagraph_mode":"PIECEWISE"}'
