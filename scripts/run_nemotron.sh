#!/bin/bash
# nvllm -- Run Nemotron-3-Super-120B on DGX Spark (GB10)
#
# MoE model with NVFP4 weights, FlashInfer attention, tool calling.
# Automatically downloads the model on first run.
#
# Usage:
#   ./scripts/run_nemotron.sh          # Standard launch
#   ./scripts/run_nemotron.sh --tq     # TurboQuant KV cache (saves memory)
#   ./scripts/run_nemotron.sh --debug  # Eager mode, no CUDA graphs

set -euo pipefail

source "$(dirname "$0")/common.sh"

MODEL_ID="nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4"
CONTAINER="nvllm"
SERVED_NAME="agents"
PORT=8000

# Parse flags
TQ=0
DEBUG=0
for arg in "$@"; do
  case "$arg" in
    --tq)    TQ=1 ;;
    --debug) DEBUG=1 ;;
    *) echo "Unknown argument: $arg" >&2; exit 1 ;;
  esac
done

# Pre-flight checks
nvllm_check_image
nvllm_ensure_model "$MODEL_ID"
nvllm_cleanup_container "$CONTAINER"
nvllm_check_port "$PORT"

# Mode-specific flags
if [ "$TQ" -eq 1 ]; then
  KV_CACHE="turboquant35"
  ATTN_BACKEND="TRITON_ATTN"
  MAX_MODEL_LEN=131072
  MAX_NUM_SEQS=2
else
  KV_CACHE="fp8"
  ATTN_BACKEND="flashinfer"
  MAX_MODEL_LEN=16384
  MAX_NUM_SEQS=16
fi

# Build extra args as array to preserve JSON quoting
EXTRA_ARGS=()
if [ "$DEBUG" -eq 1 ]; then
  EXTRA_ARGS+=(--enforce-eager)
else
  EXTRA_ARGS+=(--compilation-config '{"cudagraph_mode":"PIECEWISE"}')
fi

echo "=== Launching Nemotron-3-Super-120B ==="
echo "  Model:    $MODEL_ID"
echo "  KV cache: $KV_CACHE"
echo "  Context:  $MAX_MODEL_LEN tokens"
echo "  Port:     $PORT"
if [ "$TQ" -eq 1 ];   then echo "  Mode:     TurboQuant KV cache"; fi
if [ "$DEBUG" -eq 1 ]; then echo "  Mode:     Debug (eager, no CUDA graphs)"; fi
echo ""

docker run -d \
  --name "$CONTAINER" \
  --gpus all \
  --ipc=host \
  --network host \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  -v "$HOME/.cache/flashinfer:/root/.cache/flashinfer" \
  -v "$HOME/.cache/vllm_compile:/root/.cache/vllm/torch_compile_cache" \
  -e VLLM_NVFP4_GEMM_BACKEND=cutlass \
  -e VLLM_USE_FLASHINFER_MOE_FP4=0 \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  "$NVLLM_IMAGE" \
  serve \
  --model "$MODEL_ID" \
  --served-model-name "$SERVED_NAME" \
  --host 0.0.0.0 --port "$PORT" \
  --kv-cache-dtype "$KV_CACHE" \
  --attention-backend "$ATTN_BACKEND" \
  --max-model-len "$MAX_MODEL_LEN" \
  --max-num-seqs "$MAX_NUM_SEQS" \
  --quantization modelopt_mixed \
  --language-model-only \
  --enable-prefix-caching \
  --trust-remote-code \
  --gpu-memory-utilization 0.85 \
  --max-num-batched-tokens 16384 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --chat-template /templates/nemotron_no_think.jinja \
  "${EXTRA_ARGS[@]}"

echo "Container started: $CONTAINER"
echo "  API:  http://localhost:${PORT}/v1"
echo "  Logs: docker logs -f $CONTAINER"
