#!/bin/bash
# nvllm -- Serve Qwen3.5-27B-NVFP4 with CuTe Paged Attention
# in FULL_AND_PIECEWISE CUDA graph mode (production via blessed cache).
#
# Production semantics (Phase 2 cache workaround):
#   - All CUTE_* probes default OFF (probes-on => different config_hash =>
#     no manifest match => refuse).
#   - CUTE_PHASE_E_LAYERS defaults 0,1,2,3,4,5,6,7 (lower-8, Z1-validated).
#   - Pre-`docker run` blessed-cache verify-and-mount (RO).
#   - Refuse-on-no-match / refuse-on-drift / refuse-on-unsafe-dev manifest.
#
# Spec: docs/superpowers/specs/2026-05-01-cute-full-cache-production-workaround-design.md
#
# Default checkpoint: ig1/Qwen3.5-27B-NVFP4. Override via HF_MODEL env var.
#
# Usage:
#   ./scripts/serve-cute-full.sh             # Production (verify + RO mount)
#   ./scripts/serve-cute-full.sh --debug     # Eager mode, NO blessed-cache verify

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

# β-coop FULL kernel cache — see docs/superpowers/specs/2026-04-29-cute-full-compile-cache-design.md §4
CUTE_COMPILE_HOST_CACHE_DIR="${CUTE_COMPILE_HOST_CACHE_DIR:-/tmp/nvllm-cute-cache}"
mkdir -p "$CUTE_COMPILE_HOST_CACHE_DIR"
echo "  Cute cache:  $CUTE_COMPILE_HOST_CACHE_DIR -> /opt/vllm/kernel_cache"

# Build extra args
EXTRA_ARGS=()
if [ "$DEBUG" -eq 1 ]; then
  EXTRA_ARGS+=(--enforce-eager)
else
  EXTRA_ARGS+=(--compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE","cudagraph_capture_sizes":[1]}')
fi

# Production launch state — Phase 2 defaults.
CUTE_FULL_GRAPH_PROBE_VAL="${CUTE_FULL_GRAPH_PROBE:-0}"
CUTE_WO_RESET_LOG_VAL="${CUTE_WO_RESET_LOG:-0}"
CUTE_DISPATCH_AUDIT_VAL="${CUTE_DISPATCH_AUDIT:-0}"
CUTE_PHASE_E_LAYERS_VAL="${CUTE_PHASE_E_LAYERS:-0,1,2,3,4,5,6,7}"
KV_CACHE_DTYPE="fp8_e4m3"
ATTN_BACKEND_VAL="CUTE_PAGED"
MAX_NUM_SEQS_VAL=1
MAX_MODEL_LEN_VAL=16384
MAX_NUM_BATCHED_TOKENS_VAL=65536
CUDAGRAPH_MODE="FULL_AND_PIECEWISE"
CUDAGRAPH_CAPTURE_SIZES="[1]"
CUTE_PHASE_E_FUSION_VAL=1
CUTE_PHASE_E_FALLBACK_RAISE_VAL=1
CUTE_MLP_FUSION_VAL="${CUTE_MLP_FUSION:-1}"
CUTE_ATTN_FUSION_VAL="${CUTE_ATTN_FUSION:-1}"

# Pre-`docker run` blessed-cache verify-and-mount (skip in --debug eager).
BLESSED_MOUNT_ARGS=()
if [ "$DEBUG" -ne 1 ]; then
  IMAGE_ID=$(docker image inspect "$NVLLM_IMAGE" --format '{{.Id}}')
  HF_REVISION=$(nvllm_resolve_hf_revision "$HF_MODEL") || {
    echo "ERROR: cannot resolve HF revision for $HF_MODEL" >&2; exit 1; }
  CONFIG_HASH=$(nvllm_compute_blessed_config_hash \
    "$IMAGE_ID" "$HF_MODEL" "$HF_REVISION" \
    "$KV_CACHE_DTYPE" "$ATTN_BACKEND_VAL" "$CUDAGRAPH_MODE" \
    "$CUDAGRAPH_CAPTURE_SIZES" \
    "$MAX_NUM_SEQS_VAL" "$MAX_MODEL_LEN_VAL" "$MAX_NUM_BATCHED_TOKENS_VAL" \
    "$CUTE_PHASE_E_FUSION_VAL" "$CUTE_PHASE_E_LAYERS_VAL" \
    "$CUTE_PHASE_E_FALLBACK_RAISE_VAL" \
    "$CUTE_FULL_GRAPH_PROBE_VAL" "$CUTE_WO_RESET_LOG_VAL" \
    "$CUTE_DISPATCH_AUDIT_VAL" \
    "$CUTE_MLP_FUSION_VAL" "$CUTE_ATTN_FUSION_VAL")
  echo "[blessed-cache] Derived config_hash: $CONFIG_HASH"

  rc=0
  MANIFEST_PATH=$(nvllm_resolve_blessed_manifest "$CONFIG_HASH") || rc=$?
  case "$rc" in
    0) ;;
    2)  # corruption: duplicate config_hash
        echo "[blessed-cache] CORRUPTION: duplicate manifests for config_hash $CONFIG_HASH" >&2
        echo "[blessed-cache] Refusing to start. Resolve manually before continuing." >&2
        exit 1 ;;
    *)  nvllm_refuse_no_manifest "$CONFIG_HASH"
        exit 1 ;;
  esac
  echo "[blessed-cache] manifest: $MANIFEST_PATH"

  # Refuse if manifest carries unsafe_dev_trials = true.
  if [ "$(jq -r '.validation.unsafe_dev_trials // false' "$MANIFEST_PATH")" = "true" ]; then
    nvllm_refuse_unsafe_dev_manifest "$MANIFEST_PATH"
    exit 1
  fi

  if ! nvllm_verify_blessed_cache "$MANIFEST_PATH"; then
    nvllm_refuse_cache_drift "$CONFIG_HASH" "$MANIFEST_PATH"
    exit 1
  fi
  echo "[blessed-cache] verify PASS — mounting :ro"

  BLESSED_HOST_PATH=$(jq -r '.mount.host_path' "$MANIFEST_PATH")
  BLESSED_HOST_PATH="${BLESSED_HOST_PATH/#\~/$HOME}"
  BLESSED_MOUNT_ARGS=("-v" "${BLESSED_HOST_PATH}:/root/.cache/vllm:ro")
fi

echo "=== Launching Qwen3.5-27B-NVFP4 ($HF_MODEL) — CuTe + FULL_AND_PIECEWISE ==="
echo "  Model:       $HF_MODEL"
echo "  Attention:   $ATTN_BACKEND_VAL"
echo "  KV cache:    $KV_CACHE_DTYPE"
echo "  Context:     $MAX_MODEL_LEN_VAL tokens"
echo "  Max seqs:    $MAX_NUM_SEQS_VAL"
echo "  Layer set:   $CUTE_PHASE_E_LAYERS_VAL"
echo "  Probes:      probe=$CUTE_FULL_GRAPH_PROBE_VAL wo_reset=$CUTE_WO_RESET_LOG_VAL audit=$CUTE_DISPATCH_AUDIT_VAL"
echo "  Port:        $PORT"
if [ "$DEBUG" -eq 1 ]; then echo "  Mode:        Debug (eager, no CUDA graphs, no blessed-cache verify)"; fi
echo ""

docker run -d \
  --name "$CONTAINER" \
  --gpus all \
  --ipc=host \
  --network host \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  -v "$HOME/.cache/flashinfer:/root/.cache/flashinfer" \
  -v "$CUTE_COMPILE_HOST_CACHE_DIR:/opt/vllm/kernel_cache" \
  "${BLESSED_MOUNT_ARGS[@]}" \
  -e B12X_CUTE_COMPILE_DISK_CACHE=1 \
  -e B12X_CUTE_COMPILE_CACHE_DIR=/opt/vllm/kernel_cache \
  -e VLLM_NVFP4_GEMM_BACKEND=cutlass \
  -e VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e CUTE_MLP_FUSION="$CUTE_MLP_FUSION_VAL" \
  -e CUTE_ATTN_FUSION="$CUTE_ATTN_FUSION_VAL" \
  -e CUTE_BETA_MIN_FREE_GB="${CUTE_BETA_MIN_FREE_GB:-8}" \
  -e CUTE_PHASE_E_FUSION="$CUTE_PHASE_E_FUSION_VAL" \
  -e CUTE_PHASE_E_LAYERS="$CUTE_PHASE_E_LAYERS_VAL" \
  -e CUTE_PHASE_E_FALLBACK_RAISE="$CUTE_PHASE_E_FALLBACK_RAISE_VAL" \
  -e CUTE_FULL_GRAPH_PROBE="$CUTE_FULL_GRAPH_PROBE_VAL" \
  -e CUTE_WO_RESET_LOG="$CUTE_WO_RESET_LOG_VAL" \
  -e CUTE_DISPATCH_AUDIT="$CUTE_DISPATCH_AUDIT_VAL" \
  "$NVLLM_IMAGE" \
  serve \
  --model "$HF_MODEL" \
  --served-model-name "$SERVED_NAME" \
  --host 0.0.0.0 --port "$PORT" \
  --kv-cache-dtype "$KV_CACHE_DTYPE" \
  --attention-backend "$ATTN_BACKEND_VAL" \
  --max-model-len "$MAX_MODEL_LEN_VAL" \
  --max-num-seqs "$MAX_NUM_SEQS_VAL" \
  --language-model-only \
  --limit-mm-per-prompt '{"image": 0, "video": 0}' \
  --mamba-cache-mode align \
  --trust-remote-code \
  --gpu-memory-utilization "${SERVE_GPU_UTIL:-0.65}" \
  --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS_VAL" \
  --kernel-config '{"enable_flashinfer_autotune":false}' \
  "${EXTRA_ARGS[@]}"

echo "Container started: $CONTAINER"
echo "  API:  http://localhost:${PORT}/v1"
echo "  Logs: docker logs -f $CONTAINER"
