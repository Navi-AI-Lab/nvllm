#!/bin/bash
# nvllm -- Run Qwen3.5-122B (agents) on DGX Spark (GB10)
#
# Long-context agent server with tool calling and thinking disabled.
# Automatically downloads the model on first run.
#
# Usage:
#   ./scripts/run_qwen35_agents.sh          # Standard launch
#   ./scripts/run_qwen35_agents.sh --tq     # TurboQuant KV cache (saves memory)
#   ./scripts/run_qwen35_agents.sh --debug  # Eager mode, no CUDA graphs

set -euo pipefail

source "$(dirname "$0")/common.sh"

MODEL_ID="Sehyo/Qwen3.5-122B-A10B-NVFP4"
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
MAX_MODEL_LEN=65536
MAX_NUM_SEQS=4

# Build extra args as array to preserve JSON quoting
EXTRA_ARGS=()
if [ "$DEBUG" -eq 1 ]; then
  EXTRA_ARGS+=(--enforce-eager)
else
  EXTRA_ARGS+=(--compilation-config '{"cudagraph_mode":"PIECEWISE"}')
fi

echo "=== Launching Qwen3.5-122B Agent Server ==="
echo "  Model:       $MODEL_ID"
echo "  KV cache:    $KV_CACHE"
echo "  Context:     $MAX_MODEL_LEN tokens"
echo "  Spec decode: MTP (native, 1 token)"
echo "  Max seqs:    $MAX_NUM_SEQS concurrent agents"
echo "  Tool call:   qwen3_coder (Hermes-compatible)"
echo "  Port:        $PORT"
if [ "$TQ" -eq 1 ];   then echo "  Mode:        TurboQuant KV cache"; fi
if [ "$DEBUG" -eq 1 ]; then echo "  Mode:        Debug (eager, no CUDA graphs)"; fi
echo ""

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
  --language-model-only \
  --enable-prefix-caching \
  --trust-remote-code \
  --gpu-memory-utilization 0.90 \
  --max-num-batched-tokens 16384 \
  --speculative-config '{"method": "mtp", "num_speculative_tokens": 1}' \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --override-generation-config '{"chat_template_kwargs": {"enable_thinking": false}}' \
  "${EXTRA_ARGS[@]}"

echo "Container started: $CONTAINER"
echo "  API:  http://localhost:${PORT}/v1"
echo "  Logs: docker logs -f $CONTAINER"
