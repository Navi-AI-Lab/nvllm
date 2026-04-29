#!/bin/bash
# nvllm -- Serve Qwen3.5-27B-NVFP4 with CuTe Paged Attention
# in FULL_AND_PIECEWISE CUDA graph mode.
#
# SPIKE PROFILE (n=1 only). Configured per spec:
#   docs/superpowers/specs/2026-04-29-full-and-piecewise-cute-spike-design.md
#
# Forces:
#   - MAX_NUM_SEQS=1 (single-seq batch — uniform UNIFORM_SINGLE_TOKEN_DECODE)
#   - cudagraph_capture_sizes=[1] (only the n=1 FULL graph is captured)
#   - CUTE_PHASE_E_FUSION=1 (β-coop ON; spike target)
#   - CUTE_PHASE_E_FALLBACK_RAISE=1 (fail-fast on β-coop except)
#   - CUTE_FULL_GRAPH_PROBE=1 (log first-N FULL dispatches; C1 gate proof)
#
# Default checkpoint: ig1/Qwen3.5-27B-NVFP4. Override via HF_MODEL env var.
#
# Usage:
#   ./scripts/serve-cute-full.sh          # Standard launch (FULL_AND_PIECEWISE, n=1)
#   ./scripts/serve-cute-full.sh --debug  # Eager mode, no CUDA graphs

set -euo pipefail

source "$(dirname "$0")/common.sh"

HF_MODEL="${HF_MODEL:-ig1/Qwen3.5-27B-NVFP4}"
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
nvllm_check_free_mem "${NVLLM_MIN_FREE_GB:-90}"

# CuTe backend requires fp8_e4m3 KV cache
KV_CACHE="fp8_e4m3"
ATTN_BACKEND="CUTE_PAGED"
# FULL_AND_PIECEWISE capture needs extra workspace — halved from prod 65536.
MAX_MODEL_LEN=16384
MAX_NUM_SEQS=1

# Build extra args
EXTRA_ARGS=()
if [ "$DEBUG" -eq 1 ]; then
  EXTRA_ARGS+=(--enforce-eager)
else
  EXTRA_ARGS+=(--compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE","cudagraph_capture_sizes":[1]}')
fi

echo "=== Launching Qwen3.5-27B-NVFP4 ($HF_MODEL) — CuTe + FULL_AND_PIECEWISE ==="
echo "  Model:       $HF_MODEL"
echo "  Attention:   $ATTN_BACKEND"
echo "  KV cache:    $KV_CACHE"
echo "  Context:     $MAX_MODEL_LEN tokens"
echo "  Max seqs:    $MAX_NUM_SEQS"
echo "  Port:        $PORT"
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
  -e CUTE_MLP_FUSION="${CUTE_MLP_FUSION:-1}" \
  -e CUTE_ATTN_FUSION="${CUTE_ATTN_FUSION:-1}" \
  -e CUTE_BETA_MIN_FREE_GB="${CUTE_BETA_MIN_FREE_GB:-8}" \
  -e CUTE_PHASE_E_FUSION=1 \
  -e CUTE_PHASE_E_FALLBACK_RAISE=1 \
  -e CUTE_FULL_GRAPH_PROBE=1 \
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
  --limit-mm-per-prompt '{"image": 0, "video": 0}' \
  --mamba-cache-mode align \
  --trust-remote-code \
  --gpu-memory-utilization "${SERVE_GPU_UTIL:-0.65}" \
  --max-num-batched-tokens 65536 \
  "${EXTRA_ARGS[@]}"

echo "Container started: $CONTAINER"
echo "  API:  http://localhost:${PORT}/v1"
echo "  Logs: docker logs -f $CONTAINER"
