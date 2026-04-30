#!/bin/bash
# C2 layer-bisect under FULL_AND_PIECEWISE β-coop (2026-04-30).
#
# Mirrors serve-cute-full.sh exactly, but adds CUTE_PHASE_E_LAYERS=<csv>
# to restrict β-coop to specific layers. All other full-attn layers
# fall through to the legacy split-attention path.
#
# Usage:
#   ./c2_full_layer_bisect.sh 3                  # only layer 3
#   ./c2_full_layer_bisect.sh "3,7,11,15"        # bisect lower half
#   ./c2_full_layer_bisect.sh "47,51,55,59,63"   # bisect upper half
#
# Decision matrix:
#   Single-layer PASS → β-coop is single-layer-safe; cumulative effect from N>1
#   Single-layer FAIL → β-coop is single-layer-unsafe; tensor-localize that layer

set -euo pipefail

if [ "$#" -ne 1 ]; then
  echo "Usage: $0 <CUTE_PHASE_E_LAYERS-csv>" >&2
  echo "  e.g.: $0 3                  # only layer 3" >&2
  echo "  e.g.: $0 \"3,7,11,15\"        # bisect lower half" >&2
  exit 1
fi
LAYERS="$1"

REPO_ROOT="$(git rev-parse --show-toplevel)"
source "$REPO_ROOT/scripts/common.sh"

HF_MODEL="${HF_MODEL:-ig1/Qwen3.5-27B-NVFP4}"
CONTAINER="nvllm"
SERVED_NAME="default"
PORT=8000

nvllm_check_image
nvllm_cleanup_container "$CONTAINER"

CUTE_COMPILE_HOST_CACHE_DIR="${CUTE_COMPILE_HOST_CACHE_DIR:-/tmp/nvllm-cute-cache}"
mkdir -p "$CUTE_COMPILE_HOST_CACHE_DIR"

KV_CACHE="fp8_e4m3"
ATTN_BACKEND="CUTE_PAGED"
MAX_MODEL_LEN=16384
MAX_NUM_SEQS=1

echo "=== FULL_AND_PIECEWISE β-coop layer-bisect ==="
echo "  Model:       $HF_MODEL"
echo "  cg_mode:     FULL_AND_PIECEWISE"
echo "  β-coop:      ON, restricted to layers: $LAYERS"
echo "  Max seqs:    $MAX_NUM_SEQS"
echo ""

docker run -d \
  --name "$CONTAINER" \
  --gpus all \
  --ipc=host \
  --network host \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  -v "$HOME/.cache/flashinfer:/root/.cache/flashinfer" \
  -v "$CUTE_COMPILE_HOST_CACHE_DIR:/opt/vllm/kernel_cache" \
  -e B12X_CUTE_COMPILE_DISK_CACHE=1 \
  -e B12X_CUTE_COMPILE_CACHE_DIR=/opt/vllm/kernel_cache \
  -e VLLM_NVFP4_GEMM_BACKEND=cutlass \
  -e VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e CUTE_MLP_FUSION="${CUTE_MLP_FUSION:-1}" \
  -e CUTE_ATTN_FUSION="${CUTE_ATTN_FUSION:-1}" \
  -e CUTE_BETA_MIN_FREE_GB="${CUTE_BETA_MIN_FREE_GB:-8}" \
  -e CUTE_PHASE_E_FUSION=1 \
  -e CUTE_PHASE_E_LAYERS="$LAYERS" \
  -e CUTE_PHASE_E_FALLBACK_RAISE=1 \
  -e CUTE_FULL_GRAPH_PROBE=1 \
  -e CUTE_WO_RESET_LOG="${CUTE_WO_RESET_LOG:-0}" \
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
  --kernel-config '{"enable_flashinfer_autotune":false}' \
  --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE","cudagraph_capture_sizes":[1]}'

echo "Container started: $CONTAINER (FULL_AND_PIECEWISE, layers=$LAYERS)"

bash "$REPO_ROOT/docs/research/2026-04-29-full-graph-spike/_sync_host_edits.sh"

echo "[wrapper] container ready for model load. Wait for /v1/models then run c2_replay_coherence.py"
