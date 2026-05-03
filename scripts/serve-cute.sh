#!/bin/bash
# nvllm -- Serve Qwen3.5-27B-NVFP4 with CuTe Paged Attention
#
# Custom CuTe DSL paged attention backend for SM120/SM121 (GB10).
# Requires FP8 E4M3 KV cache (the only kv_cache_dtype CuTe backend supports).
# This is the kernel development script — use scripts/serve.sh for production.
#
# Default checkpoint: ig1/Qwen3.5-27B-NVFP4 (non-distilled, official llm-compressor
# VL recipe). Override via HF_MODEL env var
# (e.g. HF_MODEL=natfii/Qwen3.5-27B-NVFP4-Opus-GB10 for the distilled stress-test).
#
# Usage:
#   ./scripts/serve-cute.sh          # Standard launch
#   ./scripts/serve-cute.sh --debug  # Eager mode, no CUDA graphs

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
MAX_MODEL_LEN=65536
MAX_NUM_SEQS="${MAX_NUM_SEQS:-4}"

# Build extra args
EXTRA_ARGS=()
if [ "$DEBUG" -eq 1 ]; then
  EXTRA_ARGS+=(--enforce-eager)
else
  EXTRA_ARGS+=(--compilation-config '{"cudagraph_mode":"PIECEWISE"}')
fi

echo "=== Launching Qwen3.5-27B-NVFP4 ($HF_MODEL) — CuTe Paged Attention ==="
echo "  Model:       $HF_MODEL"
echo "  Attention:   $ATTN_BACKEND"
echo "  KV cache:    $KV_CACHE"
echo "  Context:     $MAX_MODEL_LEN tokens"
echo "  Max seqs:    $MAX_NUM_SEQS"
echo "  Port:        $PORT"
if [ "$DEBUG" -eq 1 ]; then echo "  Mode:        Debug (eager, no CUDA graphs)"; fi
echo ""

# NOTE: --enable-prefix-caching removed — corrupts SSM state in hybrid attention models.
# Re-evaluate when upstream vLLM explicitly supports prefix caching + FLA/mamba.

# C2 diag env file: vLLM's EngineCore subprocess strips most env vars from its
# parent. Write a sentinel file the model code can read at module import time.
# (See docs/research/uber_kernel_migration/2026-04-26-c2-diagnostic-plan.md.)
mkdir -p /tmp/c2_diag
{
  echo "CUTE_C2_DIAG=${CUTE_C2_DIAG:-}"
  echo "CUTE_C2_DIAG_INJECT_NOISE=${CUTE_C2_DIAG_INJECT_NOISE:-}"
  echo "CUTE_C2_DIAG_DUMP_DIR=${CUTE_C2_DIAG_DUMP_DIR:-}"
  echo "CUTE_C2_DIAG_TOL_ATOL=${CUTE_C2_DIAG_TOL_ATOL:-}"
  echo "CUTE_C2_DIAG_TOL_RTOL=${CUTE_C2_DIAG_TOL_RTOL:-}"
} > /tmp/c2_diag/ENV

# Optional bind-mount of the cute_paged subdir for Python-only iteration
# without a docker rebuild. Pure-Python directory (no .so), so safe to
# overlay onto the in-image editable install at /app/nvllm/...
# Enable with NVLLM_BIND_MOUNT_CUTE_PAGED=1.
BIND_MOUNT_CUTE=()
if [ "${NVLLM_BIND_MOUNT_CUTE_PAGED:-0}" = "1" ]; then
  HOST_CUTE_DIR="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)/vllm/v1/attention/backends/cute_paged"
  BIND_MOUNT_CUTE=(-v "$HOST_CUTE_DIR:/app/nvllm/vllm/v1/attention/backends/cute_paged")
  echo "  Bind mount:  $HOST_CUTE_DIR -> /app/nvllm/vllm/v1/attention/backends/cute_paged"
fi

docker run -d \
  --name "$CONTAINER" \
  --gpus all \
  --ipc=host \
  --network host \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  -v "$HOME/.cache/flashinfer:/root/.cache/flashinfer" \
  -v "/tmp/nvllm-dumps:/tmp/nvllm-dumps" \
  -v "/tmp/c2_diag:/tmp/c2_diag" \
  "${BIND_MOUNT_CUTE[@]}" \
  -e VLLM_NVFP4_GEMM_BACKEND=cutlass \
  -e VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e CUTE_DEBUG_FUSION="${CUTE_DEBUG_FUSION:-0}" \
  -e CUTE_DUMP_TENSORS="${CUTE_DUMP_TENSORS:-0}" \
  -e CUTE_MLP_FUSION="${CUTE_MLP_FUSION:-1}" \
  -e CUTE_ATTN_FUSION="${CUTE_ATTN_FUSION:-1}" \
  -e CUTE_DEBUG_MLP_FUSION="${CUTE_DEBUG_MLP_FUSION:-0}" \
  -e CUTE_C2_DIAG="${CUTE_C2_DIAG:-}" \
  -e CUTE_C2_DIAG_INJECT_NOISE="${CUTE_C2_DIAG_INJECT_NOISE:-}" \
  -e CUTE_C2_DIAG_DUMP_DIR="${CUTE_C2_DIAG_DUMP_DIR:-}" \
  -e CUTE_C2_DIAG_TOL_ATOL="${CUTE_C2_DIAG_TOL_ATOL:-}" \
  -e CUTE_C2_DIAG_TOL_RTOL="${CUTE_C2_DIAG_TOL_RTOL:-}" \
  -e CUTE_BETA_MIN_FREE_GB="${CUTE_BETA_MIN_FREE_GB:-8}" \
  -e CUTE_PHASE_E_FUSION="${CUTE_PHASE_E_FUSION:-0}" \
  -e CUTE_PHASE_E_PATH="${CUTE_PHASE_E_PATH:-auto}" \
  -e CUTE_PHASE_E_LAYERS="${CUTE_PHASE_E_LAYERS:-}" \
  -e CUTE_PHASE_E_FALLBACK_RAISE="${CUTE_PHASE_E_FALLBACK_RAISE:-0}" \
  -e CUTE_BETA_REGION_TIMING="${CUTE_BETA_REGION_TIMING:-0}" \
  -e VLLM_TORCH_PROFILER_DIR="${VLLM_TORCH_PROFILER_DIR:-}" \
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
  --gpu-memory-utilization "${SERVE_GPU_UTIL:-0.70}" \
  --max-num-batched-tokens 65536 \
  --kernel-config '{"enable_flashinfer_autotune":false}' \
  "${EXTRA_ARGS[@]}"

echo "Container started: $CONTAINER"
echo "  API:  http://localhost:${PORT}/v1"
echo "  Logs: docker logs -f $CONTAINER"
