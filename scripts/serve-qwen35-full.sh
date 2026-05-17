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
#   ./scripts/serve-qwen35-full.sh                        # Production (verify + RO mount)
#   ./scripts/serve-qwen35-full.sh --debug                # Eager mode, NO blessed-cache verify
#   ./scripts/serve-qwen35-full.sh --no-blessed-verify    # FULL+PIECEWISE, skip bless verify+mount
#                                                       # (cold capture this session). Used by
#                                                       # off-blessed-config diagnostics, e.g.
#                                                       # CUTE_PHASE_E_FUSION=0, CUTE_PHASE_E_LAYERS=0..15.
#                                                       # NOT for production serve — Z1 inductor
#                                                       # non-determinism risk applies.
#
# Env-var overrides honored (defaults preserve production behavior):
#   CUTE_PHASE_E_FUSION       (default 1)               # set 0 for phaseE-off diagnostic
#   CUTE_PHASE_E_LAYERS       (default 0,1,2,3,4,5,6,7) # set 0..15 for all-beta diagnostic

set -euo pipefail

source "$(dirname "$0")/common.sh"

HF_MODEL="${HF_MODEL:-ig1/Qwen3.5-27B-NVFP4}"
CONTAINER="nvllm"
SERVED_NAME="default"
PORT=8000

# Parse flags
DEBUG=0
NO_BLESSED_VERIFY=0
for arg in "$@"; do
  case "$arg" in
    --debug) DEBUG=1 ;;
    --no-blessed-verify) NO_BLESSED_VERIFY=1 ;;
    *) echo "Unknown argument: $arg" >&2; exit 1 ;;
  esac
done

if [ "$DEBUG" -eq 1 ] && [ "$NO_BLESSED_VERIFY" -eq 1 ]; then
  echo "ERROR: --debug and --no-blessed-verify are mutually exclusive." >&2
  echo "       --debug already skips blessed-cache verify (and uses --enforce-eager)." >&2
  exit 1
fi

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
CUTE_PHASE_E_FUSION_VAL="${CUTE_PHASE_E_FUSION:-1}"
CUTE_PHASE_E_FALLBACK_RAISE_VAL=1
CUTE_MLP_FUSION_VAL="${CUTE_MLP_FUSION:-1}"
CUTE_ATTN_FUSION_VAL="${CUTE_ATTN_FUSION:-1}"

# Pre-`docker run` blessed-cache verify-and-mount.
#   --debug                → skip block entirely (eager mode, no graphs).
#   --no-blessed-verify    → compute config_hash for metadata, but skip
#                            manifest lookup, drift verify, and RO mount.
#                            Kernel capture happens cold this session;
#                            Z1 inductor non-determinism risk applies.
#   default (production)   → full verify + mount; refuse on drift/no-match.
#                            Also mounts manifest into container and enables
#                            the import-time Python gate
#                            (NVLLM_BLESSED_CACHE_GATE=1 +
#                            NVLLM_BLESSED_CACHE_STRICT=1) so the in-engine
#                            verify catches mount-path / RO-mode / cold-
#                            compile errors that the host-side preflight
#                            can't see.
BLESSED_MOUNT_ARGS=()
BLESSED_GATE_ENV=()
BLESS_MOUNTED="false"
MANIFEST_ENFORCED="false"
CONFIG_HASH=""
BLESSED_MANIFEST_CONTAINER_PATH="/opt/nvllm/blessed_manifest.json"
# When a v2 manifest pins the CuTe .o cache, the blessed dir's
# cute_kernel_cache subdir is mounted at /opt/vllm/kernel_cache and we
# MUST NOT also bind-mount the host /tmp/nvllm-cute-cache (would shadow
# the blessed mount, defeating the verify).
BLESSED_OVERRIDES_CUTE_MOUNT="false"
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

  if [ "$NO_BLESSED_VERIFY" -eq 1 ]; then
    echo "[blessed-cache] --no-blessed-verify: SKIPPING manifest enforcement and RO mount."
    echo "[blessed-cache] FULL graphs will capture cold for this session."
    echo "[blessed-cache] bless_mounted=false manifest_enforced=false"
    MANIFEST_ENFORCED="false"
    BLESS_MOUNTED="false"
  else
    MANIFEST_ENFORCED="true"
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

    # Build mount args from manifest. v1: single .mount object. v2:
    # .mounts[] array (each with host_path + container_path + mode).
    MANIFEST_SCHEMA=$(jq -r '.schema_version // 1' "$MANIFEST_PATH")
    BLESSED_MOUNT_ARGS=()
    if [ "$MANIFEST_SCHEMA" = "1" ]; then
      BLESSED_HOST_PATH=$(jq -r '.mount.host_path' "$MANIFEST_PATH")
      BLESSED_HOST_PATH="${BLESSED_HOST_PATH/#\~/$HOME}"
      BLESSED_MOUNT_ARGS+=("-v" "${BLESSED_HOST_PATH}:/root/.cache/vllm:ro")
    else
      # v2: iterate mounts[]. CuTe-cache mount shadows the host /tmp mount.
      while IFS=$'\t' read -r m_id m_host m_container m_mode; do
        m_host="${m_host/#\~/$HOME}"
        BLESSED_MOUNT_ARGS+=("-v" "${m_host}:${m_container}:${m_mode}")
        if [ "$m_id" = "cute_kernel_cache" ]; then
          BLESSED_OVERRIDES_CUTE_MOUNT="true"
        fi
      done < <(jq -r '.mounts[] | "\(.id)\t\(.host_path)\t\(.container_path)\t\(.mode)"' "$MANIFEST_PATH")
    fi
    # Mount manifest read-only into the container for the Python gate to
    # verify from inside, then enable strict mode (cold compile = hard fail).
    BLESSED_MOUNT_ARGS+=(
      "-v" "${MANIFEST_PATH}:${BLESSED_MANIFEST_CONTAINER_PATH}:ro"
    )
    BLESSED_GATE_ENV=(
      "-e" "NVLLM_BLESSED_CACHE_GATE=1"
      "-e" "NVLLM_BLESSED_CACHE_MANIFEST=${BLESSED_MANIFEST_CONTAINER_PATH}"
      "-e" "NVLLM_BLESSED_CACHE_CONFIG_HASH=${CONFIG_HASH}"
      "-e" "NVLLM_BLESSED_CACHE_STRICT=1"
    )
    BLESS_MOUNTED="true"
  fi
fi

echo "=== Launching Qwen3.5-27B-NVFP4 ($HF_MODEL) — CuTe + FULL_AND_PIECEWISE ==="
echo "  Model:       $HF_MODEL"
echo "  Attention:   $ATTN_BACKEND_VAL"
echo "  KV cache:    $KV_CACHE_DTYPE"
echo "  Context:     $MAX_MODEL_LEN_VAL tokens"
echo "  Max seqs:    $MAX_NUM_SEQS_VAL"
echo "  Layer set:   $CUTE_PHASE_E_LAYERS_VAL"
echo "  Phase E:     fusion=$CUTE_PHASE_E_FUSION_VAL"
echo "  Probes:      probe=$CUTE_FULL_GRAPH_PROBE_VAL wo_reset=$CUTE_WO_RESET_LOG_VAL audit=$CUTE_DISPATCH_AUDIT_VAL"
echo "  Bless:       mounted=$BLESS_MOUNTED enforced=$MANIFEST_ENFORCED config_hash=${CONFIG_HASH:-n/a}"
echo "  Port:        $PORT"
if [ "$DEBUG" -eq 1 ]; then echo "  Mode:        Debug (eager, no CUDA graphs, no blessed-cache verify)"; fi
if [ "$NO_BLESSED_VERIFY" -eq 1 ]; then echo "  Mode:        FULL+PIECEWISE cold (--no-blessed-verify diagnostic)"; fi
echo ""

# When the manifest (v2) provides its own cute_kernel_cache mount, skip
# the host /tmp/nvllm-cute-cache bind-mount — otherwise it would shadow
# the blessed mount and defeat the in-engine verify.
HOST_CUTE_MOUNT_ARGS=()
if [ "$BLESSED_OVERRIDES_CUTE_MOUNT" != "true" ]; then
  HOST_CUTE_MOUNT_ARGS=("-v" "$CUTE_COMPILE_HOST_CACHE_DIR:/opt/vllm/kernel_cache")
fi

docker run -d \
  --name "$CONTAINER" \
  --gpus all \
  --ipc=host \
  --network host \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  -v "$HOME/.cache/flashinfer:/root/.cache/flashinfer" \
  "${HOST_CUTE_MOUNT_ARGS[@]}" \
  "${BLESSED_MOUNT_ARGS[@]}" \
  "${BLESSED_GATE_ENV[@]}" \
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
