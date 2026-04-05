#!/bin/bash
# nvllm -- Run Qwen3-Coder-Next (NVFP4) on DGX Spark (GB10)
#
# Hybrid DeltaNet architecture with tool calling for coding agents.
# Automatically downloads the model on first run.
#
# Usage:
#   ./scripts/run_qwen3_coder_next.sh          # Standard launch
#   ./scripts/run_qwen3_coder_next.sh --tq     # TurboQuant KV cache (saves memory)
#   ./scripts/run_qwen3_coder_next.sh --debug  # Eager mode, no CUDA graphs

set -euo pipefail

source "$(dirname "$0")/common.sh"

MODEL_ID="GadflyII/Qwen3-Coder-Next-NVFP4"
CONTAINER="nvllm-agents"
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
else
  KV_CACHE="fp8"
  ATTN_BACKEND="triton_attn"
fi
MAX_MODEL_LEN=131072
MAX_NUM_SEQS=8

if [ "$DEBUG" -eq 1 ]; then
  COMPILE_FLAGS="--enforce-eager"
else
  COMPILE_FLAGS="--compilation-config '{\"cudagraph_mode\":\"PIECEWISE\"}'"
fi

echo "=== Launching Qwen3-Coder-Next (NVFP4) ==="
echo "  Model:    $MODEL_ID"
echo "  Arch:     Hybrid DeltaNet + Attention + MoE (80B/3B active)"
echo "  KV cache: $KV_CACHE"
echo "  Context:  $MAX_MODEL_LEN tokens (128K)"
echo "  Max seqs: $MAX_NUM_SEQS"
echo "  Tool call: qwen3_coder (Hermes-compatible)"
echo "  Port:     $PORT"
if [ "$TQ" -eq 1 ];   then echo "  Mode:     TurboQuant KV cache"; fi
if [ "$DEBUG" -eq 1 ]; then echo "  Mode:     Debug (eager, no CUDA graphs)"; fi
echo ""

# shellcheck disable=SC2086
docker run -d \
  --name "$CONTAINER" \
  --gpus all \
  --ipc=host \
  --network host \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  -v "$HOME/.cache/flashinfer:/root/.cache/flashinfer" \
  -e VLLM_NVFP4_GEMM_BACKEND=cutlass \
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
  --enable-prefix-caching \
  --trust-remote-code \
  --gpu-memory-utilization 0.92 \
  --max-num-batched-tokens 16384 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --override-generation-config '{"chat_template_kwargs": {"enable_thinking": false}}' \
  $COMPILE_FLAGS

echo "Container started: $CONTAINER"
echo "  API:  http://localhost:${PORT}/v1"
echo "  Logs: docker logs -f $CONTAINER"
