#!/bin/bash
# nvllm -- Run natfii/Qwen3.5-27B-NVFP4-Opus-GB10 on DGX Spark (GB10)
#
# Dense Qwen3.5 model with hybrid attention (linear + full) and MTP.
# ~18 GB NVFP4 quantized — fits in GB10's 128 GB with 64k context.
# Automatically downloads the model from Hugging Face on first run.
#
# Usage:
#   ./scripts/run_qwen35_27b_nvfp4.sh          # Standard launch
#   ./scripts/run_qwen35_27b_nvfp4.sh --debug  # Eager mode, no CUDA graphs

set -euo pipefail

source "$(dirname "$0")/common.sh"

HF_MODEL="natfii/Qwen3.5-27B-NVFP4-Opus-GB10"
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

# Serving config — TurboQuant KV cache for max context
KV_CACHE="turboquant35"
ATTN_BACKEND="TRITON_ATTN"
MAX_MODEL_LEN=65536
MAX_NUM_SEQS=4

# Build extra args as array to preserve JSON quoting
EXTRA_ARGS=()
if [ "$DEBUG" -eq 1 ]; then
  EXTRA_ARGS+=(--enforce-eager)
else
  EXTRA_ARGS+=(--compilation-config '{"cudagraph_mode":"PIECEWISE"}')
fi

echo "=== Launching Qwen3.5-27B-NVFP4-Opus-GB10 ==="
echo "  Model:       $HF_MODEL"
echo "  KV cache:    $KV_CACHE"
echo "  Context:     $MAX_MODEL_LEN tokens"
echo "  Max seqs:    $MAX_NUM_SEQS"
echo "  Spec decode: disabled (MTP pending upstream support)"
echo "  Port:        $PORT"
if [ "$DEBUG" -eq 1 ]; then echo "  Mode:        Debug (eager, no CUDA graphs)"; fi
echo ""

# NOTE: --enable-prefix-caching removed — corrupts SSM state in hybrid attention models.
# Re-evaluate when upstream vLLM explicitly supports prefix caching + FLA/mamba.
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
  --model "$HF_MODEL" \
  --served-model-name "$SERVED_NAME" \
  --host 0.0.0.0 --port "$PORT" \
  --kv-cache-dtype "$KV_CACHE" \
  --attention-backend "$ATTN_BACKEND" \
  --max-model-len "$MAX_MODEL_LEN" \
  --max-num-seqs "$MAX_NUM_SEQS" \
  --language-model-only \
  --mamba-cache-mode align \
  --mamba-block-size 64 \
  --trust-remote-code \
  --gpu-memory-utilization 0.80 \
  --max-num-batched-tokens 65536 \
  --enable-auto-tool-choice \
  --tool-call-parser hermes \
  "${EXTRA_ARGS[@]}"

# MTP spec decode not yet supported for Qwen3_5ForCausalLM in this vLLM build.
# Re-enable when upstream adds support:
#  --speculative-config '{"method": "mtp", "num_speculative_tokens": 1}'

echo "Container started: $CONTAINER"
echo "  API:  http://localhost:${PORT}/v1"
echo "  Logs: docker logs -f $CONTAINER"
