#!/bin/bash
# nvllm -- Run Qwen3.5-27B (NVFP4) on DGX Spark (GB10)
#
# Dense Qwen3.5 model with hybrid attention (linear + full) and MTP.
# ~14 GB quantized — leaves plenty of room for context and batches.
# Automatically downloads the model on first run.
#
# Usage:
#   ./scripts/run_qwen35_27b.sh          # Standard launch (fp8 KV)
#   ./scripts/run_qwen35_27b.sh --tq     # TurboQuant KV cache (more context)
#   ./scripts/run_qwen35_27b.sh --debug  # Eager mode, no CUDA graphs

set -euo pipefail

source "$(dirname "$0")/common.sh"

MODEL="${MODEL:-$HOME/.cache/huggingface/hub/Qwen3.5-27B-NVFP4}"
# Container sees the HF cache at /root/.cache/huggingface via volume mount
CONTAINER_MODEL="/root/.cache/huggingface/hub/Qwen3.5-27B-NVFP4"
CONTAINER="nvllm"
SERVED_NAME="default"
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

# Verify quantized checkpoint exists
if [ ! -d "$MODEL" ]; then
  echo "ERROR: Quantized checkpoint not found at $MODEL" >&2
  echo "" >&2
  echo "Run quantization first:" >&2
  echo "  ./training/quantize_qwen35_27b.sh" >&2
  exit 1
fi

# Pre-flight checks
nvllm_check_image
nvllm_cleanup_container "$CONTAINER"
nvllm_check_port "$PORT"

# Mode-specific flags
if [ "$TQ" -eq 1 ]; then
  KV_CACHE="turboquant35"
  ATTN_BACKEND="TRITON_ATTN"
else
  KV_CACHE="auto"
  ATTN_BACKEND="triton_attn"
fi
MAX_MODEL_LEN=16384
MAX_NUM_SEQS=4

# Build extra args as array to preserve JSON quoting
EXTRA_ARGS=()
if [ "$DEBUG" -eq 1 ]; then
  EXTRA_ARGS+=(--enforce-eager)
else
  EXTRA_ARGS+=(--compilation-config '{"cudagraph_mode":"PIECEWISE"}')
fi

echo "=== Launching Qwen3.5-27B (NVFP4) ==="
echo "  Model:       $MODEL"
echo "  KV cache:    $KV_CACHE"
echo "  Context:     $MAX_MODEL_LEN tokens"
echo "  Max seqs:    $MAX_NUM_SEQS"
echo "  Spec decode: disabled (MTP pending upstream support)"
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
  -e VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  "$NVLLM_IMAGE" \
  serve \
  --model "$CONTAINER_MODEL" \
  --served-model-name "$SERVED_NAME" \
  --host 0.0.0.0 --port "$PORT" \
  --kv-cache-dtype "$KV_CACHE" \
  --attention-backend "$ATTN_BACKEND" \
  --max-model-len "$MAX_MODEL_LEN" \
  --max-num-seqs "$MAX_NUM_SEQS" \
  --language-model-only \
  --enable-prefix-caching \
  --mamba-cache-mode align \
  --mamba-block-size 64 \
  --trust-remote-code \
  --gpu-memory-utilization 0.45 \
  --max-num-batched-tokens 16384 \
  "${EXTRA_ARGS[@]}"

# MTP spec decode not yet supported for Qwen3_5ForCausalLM in this vLLM build.
# Re-enable when upstream adds support:
#  --speculative-config '{"method": "mtp", "num_speculative_tokens": 1}'

echo "Container started: $CONTAINER"
echo "  API:  http://localhost:${PORT}/v1"
echo "  Logs: docker logs -f $CONTAINER"
