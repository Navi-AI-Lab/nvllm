#!/bin/bash
# nvllm -- Run Gemma 4 31B IT (NVFP4) on DGX Spark (GB10)
#
# Dense vision-language model with ngram speculative decoding.
# Supports both local checkpoints and HF model IDs (e.g., RedHatAI/gemma-4-31B-it-NVFP4).
#
# Usage:
#   ./scripts/run_gemma4.sh          # Standard launch
#   ./scripts/run_gemma4.sh --debug  # Eager mode, no CUDA graphs

set -euo pipefail

source "$(dirname "$0")/common.sh"

MODEL="${GEMMA4_MODEL_PATH:-RedHatAI/gemma-4-31B-it-NVFP4}"
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

# If MODEL is a local path, check it exists. If it looks like an HF ID, let vLLM pull it.
if [[ "$MODEL" != */* ]] && [ ! -d "$MODEL" ]; then
  echo "ERROR: Local model path not found: $MODEL" >&2
  echo "  Set GEMMA4_MODEL_PATH to a local path or HF model ID" >&2
  exit 1
elif [[ "$MODEL" == */* ]] && [ -d "$MODEL" ]; then
  # Local path with a slash — mount it (assume modelopt quant)
  MOUNT_ARGS="-v $MODEL:/model"
  QUANT_ARG="--quantization modelopt_fp4"
  MODEL="/model"
elif [[ "$MODEL" == */* ]]; then
  # HF model ID — vLLM pulls directly, auto-detect quant format
  MOUNT_ARGS=""
  QUANT_ARG=""
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
  $MOUNT_ARGS \
  -e VLLM_NVFP4_GEMM_BACKEND=cutlass \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e NVLLM_SPEC_CONFIG="$SPEC_CONFIG" \
  -e NVLLM_MODEL="$MODEL" \
  -e NVLLM_QUANT_ARG="$QUANT_ARG" \
  -e NVLLM_COMPILE_ARG="$COMPILE_ARG" \
  --entrypoint bash \
  "$NVLLM_IMAGE" \
  -c "pip install 'numba>=0.65' 'llvmlite>=0.47' 2>&1 | tail -1 && \
  exec python3 -m vllm.entrypoints.cli.main serve \
  --model \$NVLLM_MODEL \
  --served-model-name $SERVED_NAME \
  --host 0.0.0.0 --port $PORT \
  --kv-cache-dtype $KV_CACHE \
  --attention-backend $ATTN_BACKEND \
  --max-model-len 32768 \
  --max-num-seqs 4 \
  \$NVLLM_QUANT_ARG \
  --language-model-only \
  --enable-prefix-caching \
  --trust-remote-code \
  --gpu-memory-utilization 0.85 \
  --max-num-batched-tokens 16384 \
  --enable-auto-tool-choice \
  --tool-call-parser gemma4 \
  --speculative-config \"\$NVLLM_SPEC_CONFIG\" \
  \$NVLLM_COMPILE_ARG"

echo "Container started: $CONTAINER"
echo "  API:  http://localhost:${PORT}/v1"
echo "  Logs: docker logs -f $CONTAINER"
