#!/bin/bash
# nvllm -- Serve natfii/Qwen3.5-27B-NVFP4-Opus-GB10 on DGX Spark (GB10)
#
# Dense Qwen3.5 with hybrid attention (linear + full).
# ~18 GB NVFP4 quantized — fits GB10's 128 GB with 64k context.
# Uses triton_attn backend (production default).
#
# Usage:
#   ./scripts/serve.sh          # Standard launch (FP8 KV)
#   ./scripts/serve.sh --tq     # TurboQuant KV cache (more context)
#   ./scripts/serve.sh --debug  # Eager mode, no CUDA graphs

set -euo pipefail

source "$(dirname "$0")/common.sh"

HF_MODEL="natfii/Qwen3.5-27B-NVFP4-Opus-GB10"
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

# Pre-flight checks
nvllm_check_image
nvllm_cleanup_container "$CONTAINER"
nvllm_check_port "$PORT"

# Serving config
if [ "$TQ" -eq 1 ]; then
  KV_CACHE="turboquant35"
  ATTN_BACKEND="TRITON_ATTN"
else
  KV_CACHE="auto"
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

echo "=== Launching Qwen3.5-27B-NVFP4-Opus-GB10 ==="
echo "  Model:       $HF_MODEL"
echo "  Attention:   $ATTN_BACKEND"
echo "  KV cache:    $KV_CACHE"
echo "  Context:     $MAX_MODEL_LEN tokens"
echo "  Max seqs:    $MAX_NUM_SEQS"
echo "  Spec decode: disabled (MTP pending upstream support)"
echo "  Port:        $PORT"
if [ "$TQ" -eq 1 ];   then echo "  Mode:        TurboQuant KV cache"; fi
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
  --tool-call-parser qwen3_coder \
  "${EXTRA_ARGS[@]}"

# MTP spec decode not yet supported for Qwen3_5ForCausalLM in this vLLM build.
# Re-enable when upstream adds support:
#  --speculative-config '{"method": "mtp", "num_speculative_tokens": 1}'

# NOTE: --override-generation-config chat_template_kwargs does NOT work on this
# vLLM build. To disable thinking, clients must send:
#   "chat_template_kwargs": {"enable_thinking": false}
# in each request body.

echo "Container started: $CONTAINER"
echo "  API:  http://localhost:${PORT}/v1"
echo "  Logs: docker logs -f $CONTAINER"
