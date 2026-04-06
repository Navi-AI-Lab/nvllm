#!/bin/bash
# nvllm -- Run Gemma 4 31B IT (NVFP4) on DGX Spark (GB10)
#
# Dense vision-language model with ngram speculative decoding.
# Model must be quantized locally first — no auto-pull available.
#
# Usage:
#   ./scripts/run_gemma4.sh          # Standard launch
#   ./scripts/run_gemma4.sh --debug  # Eager mode, no CUDA graphs

set -euo pipefail

source "$(dirname "$0")/common.sh"

MODEL="${GEMMA4_MODEL_PATH:-$HOME/.cache/huggingface/hub/gemma-4-31B-it-NVFP4}"
CONTAINER="nvllm"
SERVED_NAME="default"
PORT=8000

# Parse flags
DEBUG=0
for arg in "$@"; do
  case "$arg" in
    --debug) DEBUG=1 ;;
    *) echo "Unknown argument: $arg" >&2; exit 1 ;;
  esac
done

# Pre-flight checks
nvllm_check_image
nvllm_cleanup_container "$CONTAINER"
nvllm_check_port "$PORT"

# Gemma4 must be quantized locally — no HF pull
if [ ! -d "$MODEL" ]; then
  echo "ERROR: NVFP4 checkpoint not found at: $MODEL" >&2
  echo "" >&2
  echo "Quantize it first:" >&2
  echo "  ./scripts/quantize_gemma4.sh" >&2
  echo "" >&2
  echo "Or point to an existing checkpoint:" >&2
  echo "  GEMMA4_MODEL_PATH=/path/to/gemma-4-31B-it-NVFP4 $0" >&2
  exit 1
fi

# Serving config — TurboQuant KV cache for max context
KV_CACHE="turboquant35"
ATTN_BACKEND="TRITON_ATTN"

# JSON args passed via env vars to avoid quoting issues inside bash -c
SPEC_CONFIG='{"method": "ngram", "num_speculative_tokens": 3, "prompt_lookup_max": 3}'
if [ "$DEBUG" -eq 1 ]; then
  COMPILE_ARG="--enforce-eager"
else
  COMPILE_ARG="--compilation-config {\"cudagraph_mode\":\"PIECEWISE\"}"
fi

echo "=== Launching Gemma 4 31B IT (NVFP4) ==="
echo "  Model:       $MODEL"
echo "  KV cache:    $KV_CACHE"
echo "  Context:     32768 tokens"
echo "  Spec decode: ngram (3 tokens, prompt lookup max 3)"
echo "  Max seqs:    4"
echo "  Port:        $PORT"
if [ "$DEBUG" -eq 1 ]; then echo "  Mode:        Debug (eager, no CUDA graphs)"; fi
echo ""

# Gemma4 requires a numba workaround: pip install inside the container
# before starting vLLM. We use --entrypoint bash for this.
# The model directory is mounted as /model inside the container.
# JSON args are passed via environment variables to avoid quoting issues
# inside the bash -c string.
docker run -d \
  --name "$CONTAINER" \
  --gpus all \
  --ipc=host \
  --network host \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  -v "$HOME/.cache/flashinfer:/root/.cache/flashinfer" \
  -v "$HOME/.cache/vllm_compile:/root/.cache/vllm/torch_compile_cache" \
  -v "$MODEL:/model" \
  -e VLLM_NVFP4_GEMM_BACKEND=cutlass \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e NVLLM_SPEC_CONFIG="$SPEC_CONFIG" \
  -e NVLLM_COMPILE_ARG="$COMPILE_ARG" \
  --entrypoint bash \
  "$NVLLM_IMAGE" \
  -c "pip install --no-deps 'numba>=0.65' 2>&1 | tail -1 && \
  exec python3 -m vllm.entrypoints.cli.main serve \
  --model /model \
  --served-model-name $SERVED_NAME \
  --host 0.0.0.0 --port $PORT \
  --kv-cache-dtype $KV_CACHE \
  --attention-backend $ATTN_BACKEND \
  --max-model-len 32768 \
  --max-num-seqs 4 \
  --quantization modelopt_fp4 \
  --language-model-only \
  --enable-prefix-caching \
  --trust-remote-code \
  --gpu-memory-utilization 0.85 \
  --max-num-batched-tokens 16384 \
  --speculative-config \"\$NVLLM_SPEC_CONFIG\" \
  \$NVLLM_COMPILE_ARG"

echo "Container started: $CONTAINER"
echo "  API:  http://localhost:${PORT}/v1"
echo "  Logs: docker logs -f $CONTAINER"
