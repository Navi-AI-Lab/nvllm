#!/bin/bash
# nvllm -- Serve Qwen3.6-27B-NVFP4 (text-only + MTP + CUTE_PAGED + 64K)
#
# Default checkpoint: unsloth/Qwen3.6-27B-NVFP4 (stock Qwen3.6 weights, NVFP4
# quantized by Unsloth with UltraChat calibration). Text-only (vision disabled),
# MTP speculative decode, CUTE_PAGED attention, FP8 E4M3 KV cache, 65536
# max-model-len.
#
# Override HF_MODEL to test other Qwen3.6 NVFP4 variants.
#
# Knobs vs scripts/serve-qwen35.sh:
#   - default HF_MODEL is a Qwen3.6 NVFP4 checkpoint
#   - --speculative-config '{"method":"qwen3_5_mtp","num_speculative_tokens":N}' added
#   - QUANTIZATION='' by default (auto-detect from config.json). Set
#     QUANTIZATION=modelopt for checkpoints that ship ModelOpt-format metadata.
#   - MAX_NUM_SEQS defaults to 1 during bring-up
#   - CUTE_MLP_FUSION / CUTE_ATTN_FUSION / CUTE_PHASE_E_FUSION default to 0
#     during Qwen3.6 bring-up (re-enable after MTP-aware kernel re-tuning).
#     CUTE_WO_SPLIT defaults to 1 (the non-K-parallel path); set
#     CUTE_WO_SPLIT=8 to opt into the evidenced K-parallel W_O GEMV
#     (GSM8K-50 47/50, ≈ -13% wall — see README Qwen3.6 bring-up note).
#
# To enable vision later: remove the --language-model-only and
# --limit-mm-per-prompt flags.
#
# Bring-up validation order (stop at first failure):
#   1. text-only short completion
#   2. text-only ~8K prompt, then 64K admission check
#   3. MTP n=1 acceptance counters nonzero in logs
#   4. MTP n=3 only after n=1 is stable
#   5. MAX_NUM_SEQS=2 only after n=3 stable
#
# Usage:
#   ./serve-qwen36.sh                # bring-up defaults (n_seqs=1, MTP n=1, fusions off)
#   MAX_NUM_SEQS=2 ./serve-qwen36.sh # after correctness passes
#   MTP_TOKENS=3 ./serve-qwen36.sh   # only after n=1 stable
#   ./serve-qwen36.sh --debug        # eager mode, no CUDA graphs

set -euo pipefail

source "$(dirname "$0")/common.sh"

HF_MODEL="${HF_MODEL:-natfii/Qwen3.6-27B-VLM-NVFP4-MTP}"
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
MAX_MODEL_LEN="${MAX_MODEL_LEN:-65536}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-1}"
MTP_TOKENS="${MTP_TOKENS:-1}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-65536}"

# Build extra args
EXTRA_ARGS=()
if [ "$DEBUG" -eq 1 ]; then
  EXTRA_ARGS+=(--enforce-eager)
else
  EXTRA_ARGS+=(--compilation-config '{"cudagraph_mode":"PIECEWISE"}')
fi

SPEC_ARGS=()
if [ "$MTP_TOKENS" -gt 0 ]; then
  SPEC_ARGS+=(--speculative-config "{\"method\":\"qwen3_5_mtp\",\"num_speculative_tokens\":${MTP_TOKENS}}")
fi

# Quant flag — empty by default, vLLM auto-detects format from config.json
# (compressed-tensors, modelopt, etc). Set QUANTIZATION=modelopt explicitly for
# checkpoints that ship ModelOpt-format metadata vLLM can't auto-detect.
QUANTIZATION="${QUANTIZATION-}"  # single dash: empty means "no flag"; only unset triggers default
QUANT_ARGS=()
if [ -n "$QUANTIZATION" ]; then
  QUANT_ARGS+=(--quantization "$QUANTIZATION")
fi

echo "=== Launching Qwen3.6-27B-NVFP4 ($HF_MODEL) — text-only + MTP + CUTE_PAGED ==="
echo "  Model:       $HF_MODEL"
echo "  Quant:       ${QUANTIZATION:-auto-detect} (NVFP4)"
echo "  Attention:   $ATTN_BACKEND"
echo "  KV cache:    $KV_CACHE"
echo "  Context:     $MAX_MODEL_LEN tokens"
echo "  Max seqs:    $MAX_NUM_SEQS"
echo "  MTP tokens:  $MTP_TOKENS"
echo "  Vision:      disabled (--language-model-only)"
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
  echo "CUTE_WO_SPLIT=${CUTE_WO_SPLIT:-1}"
} > /tmp/c2_diag/ENV

# Optional bind-mount of the cute_paged subdir for Python-only iteration
# without a docker rebuild.
BIND_MOUNT_CUTE=()
if [ "${NVLLM_BIND_MOUNT_CUTE_PAGED:-0}" = "1" ]; then
  HOST_CUTE_DIR="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)/vllm/v1/attention/backends/cute_paged"
  BIND_MOUNT_CUTE=(-v "$HOST_CUTE_DIR:/app/nvllm/vllm/v1/attention/backends/cute_paged")
  echo "  Bind mount:  $HOST_CUTE_DIR -> /app/nvllm/vllm/v1/attention/backends/cute_paged"
fi

# Tokenizer-config wrapper patch for Qwen3.6 NVFP4 checkpoints.
# Every Qwen3.6 NVFP4 release seen so far declares tokenizer_class=
# "TokenizersBackend" (transformers 5.x wrapper); our image is pinned to
# transformers 4.57.6 and rejects it. Overlay a patched config that names the
# concrete Qwen2Tokenizer class — underlying tokenizer.json is unchanged. See
# models/qwen36-tokenizer-patch/README.md.
#
# Auto-applies when the snapshot's tokenizer_config has the bad wrapper.
# Disable with NVLLM_TOKENIZER_PATCH=0 (e.g. once the image moves to
# transformers 5.x, or to debug the raw upstream failure).
TOKENIZER_PATCH_MOUNT=()
if [ "${NVLLM_TOKENIZER_PATCH:-auto}" != "0" ]; then
  REPO_ROOT="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
  PATCH_FILE="$REPO_ROOT/models/qwen36-tokenizer-patch/tokenizer_config.json"
  # Derive HF cache snapshot dir from HF_MODEL (e.g. unsloth/Qwen3.6-27B-NVFP4 ->
  # models--unsloth--Qwen3.6-27B-NVFP4). Local paths (starting with /) are skipped.
  if [ -f "$PATCH_FILE" ] && [[ "$HF_MODEL" != /* ]]; then
    REPO_CACHE_NAME="models--${HF_MODEL//\//--}"
    HOST_SNAPSHOT_DIR="$HOME/.cache/huggingface/hub/$REPO_CACHE_NAME/snapshots"
    if [ -d "$HOST_SNAPSHOT_DIR" ]; then
      SNAPSHOT_ID="$(ls "$HOST_SNAPSHOT_DIR" | head -1)"
      SNAPSHOT_TOK="$HOST_SNAPSHOT_DIR/$SNAPSHOT_ID/tokenizer_config.json"
      if [ -n "$SNAPSHOT_ID" ] && [ -f "$SNAPSHOT_TOK" ]; then
        # Only patch if the snapshot actually carries the bad wrapper (or if user forced patch=1)
        NEEDS_PATCH=0
        if [ "${NVLLM_TOKENIZER_PATCH:-auto}" = "1" ]; then
          NEEDS_PATCH=1
        elif grep -q '"tokenizer_class": "TokenizersBackend"' "$SNAPSHOT_TOK" 2>/dev/null; then
          NEEDS_PATCH=1
        fi
        if [ "$NEEDS_PATCH" = "1" ]; then
          TOKENIZER_DST="/root/.cache/huggingface/hub/$REPO_CACHE_NAME/snapshots/$SNAPSHOT_ID/tokenizer_config.json"
          TOKENIZER_PATCH_MOUNT=(-v "$PATCH_FILE:$TOKENIZER_DST:ro")
          echo "  Tokenizer:   patched (TokenizersBackend -> Qwen2Tokenizer) via bind-mount"
        fi
      fi
    fi
  fi
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
  "${TOKENIZER_PATCH_MOUNT[@]}" \
  -e VLLM_NVFP4_GEMM_BACKEND=cutlass \
  -e VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e CUTE_DEBUG_FUSION="${CUTE_DEBUG_FUSION:-0}" \
  -e CUTE_DUMP_TENSORS="${CUTE_DUMP_TENSORS:-0}" \
  -e CUTE_MLP_FUSION="${CUTE_MLP_FUSION:-0}" \
  -e CUTE_ATTN_FUSION="${CUTE_ATTN_FUSION:-0}" \
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
  -e CUTE_WO_SPLIT="${CUTE_WO_SPLIT:-1}" \
  -e VLLM_TORCH_PROFILER_DIR="${VLLM_TORCH_PROFILER_DIR:-}" \
  "$NVLLM_IMAGE" \
  serve \
  --model "$HF_MODEL" \
  --served-model-name "$SERVED_NAME" \
  --host 0.0.0.0 --port "$PORT" \
  "${QUANT_ARGS[@]}" \
  "${SPEC_ARGS[@]}" \
  --kv-cache-dtype "$KV_CACHE" \
  --attention-backend "$ATTN_BACKEND" \
  --max-model-len "$MAX_MODEL_LEN" \
  --max-num-seqs "$MAX_NUM_SEQS" \
  --language-model-only \
  --limit-mm-per-prompt '{"image": 0, "video": 0}' \
  --mamba-cache-mode align \
  --trust-remote-code \
  --gpu-memory-utilization "${SERVE_GPU_UTIL:-0.70}" \
  --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS" \
  --kernel-config '{"enable_flashinfer_autotune":false}' \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_xml \
  "${EXTRA_ARGS[@]}"

echo "Container started: $CONTAINER"
echo "  API:  http://localhost:${PORT}/v1"
echo "  Logs: docker logs -f $CONTAINER"
