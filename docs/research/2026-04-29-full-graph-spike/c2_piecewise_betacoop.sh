#!/bin/bash
# C2 multi-token under PIECEWISE β-coop — localization A/B (2026-04-30).
#
# Mirrors scripts/serve-cute-full.sh exactly, except cudagraph_mode is set
# to PIECEWISE instead of FULL_AND_PIECEWISE. β-coop ON, MAX_NUM_SEQS=1,
# same workspace allocations. The only diff is FULL graph capture vs
# PIECEWISE.
#
# Decision matrix:
#   PIECEWISE C2 PASS → FULL graph boundary/state issue (mutates_args,
#                       capture-time aliasing, etc.)
#   PIECEWISE C2 FAIL → kernel/runtime nondet under β-coop, independent
#                       of FULL replay
#
# Usage: ./c2_piecewise_betacoop.sh

set -euo pipefail

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

echo "=== PIECEWISE β-coop A/B — same args as serve-cute-full.sh except cudagraph_mode ==="
echo "  Model:       $HF_MODEL"
echo "  cg_mode:     PIECEWISE (NOT FULL_AND_PIECEWISE)"
echo "  β-coop:      ON (CUTE_PHASE_E_FUSION=1)"
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
  --kernel-config '{"enable_flashinfer_autotune":false}' \
  --compilation-config '{"cudagraph_mode":"PIECEWISE"}'

echo "Container started: $CONTAINER (PIECEWISE)"

# Sync v7-reverted host edits BEFORE model load
echo "[wrapper] running _sync_host_edits.sh..."
bash "$REPO_ROOT/docs/research/2026-04-29-full-graph-spike/_sync_host_edits.sh"

echo "[wrapper] container ready for model load. Wait for /v1/models then run c2_replay_coherence.py"
