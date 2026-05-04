#!/bin/bash
# nvllm -- Serve Huihui-Qwen3.6-27B-abliterated-NVFP4-MTP (vision + MTP + 64K)
#
# Bring-up script for sakamakismile/Huihui-Qwen3.6-27B-abliterated-NVFP4-MTP on GB10.
# Multimodal (vision enabled), MTP speculative decode, CUTE_PAGED attention,
# FP8 E4M3 KV cache, 65536 max-model-len.
#
# Differences from scripts/serve-cute.sh:
#   - default HF_MODEL points at the Huihui MTP NVFP4 checkpoint
#   - --language-model-only removed (vision enabled)
#   - --limit-mm-per-prompt removed (image/video allowed)
#   - --quantization modelopt added (ModelOpt NVFP4 loader path)
#   - --speculative-config '{"method":"qwen3_5_mtp","num_speculative_tokens":N}' added
#   - MAX_NUM_SEQS defaults to 1 during bring-up
#   - CUTE_MLP_FUSION / CUTE_ATTN_FUSION / CUTE_PHASE_E_FUSION default to 0
#     (CUTE_WO_SPLIT stays at 1 — the production-blessed K-parallel decode path)
#
# Bring-up validation order (stop at first failure):
#   1. text-only short completion (no image)
#   2. text-only ~8K prompt, then 64K admission check
#   3. MTP n=1 text-only completion; check logs for draft acceptance
#   4. image request without MTP (only if vision needs isolation)
#   5. image request with MTP n=1
#   6. MTP n=3 only after n=1 is stable
#
# Usage:
#   ./serve-cute-vision-mtp.sh                # bring-up defaults (n_seqs=1, MTP n=1, fusions off)
#   MAX_NUM_SEQS=2 ./serve-cute-vision-mtp.sh # after correctness passes
#   MTP_TOKENS=3 ./serve-cute-vision-mtp.sh   # only after n=1 stable
#   ./serve-cute-vision-mtp.sh --debug        # eager mode, no CUDA graphs

set -euo pipefail

source "$(dirname "$0")/common.sh"

HF_MODEL="${HF_MODEL:-sakamakismile/Huihui-Qwen3.6-27B-abliterated-NVFP4-MTP}"
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

# Build extra args
EXTRA_ARGS=()
if [ "$DEBUG" -eq 1 ]; then
  EXTRA_ARGS+=(--enforce-eager)
else
  EXTRA_ARGS+=(--compilation-config '{"cudagraph_mode":"PIECEWISE"}')
fi
if [ "${NVLLM_TORCH_PROFILER:-0}" = "1" ]; then
  PROFILER_DIR="${VLLM_TORCH_PROFILER_DIR:-/root/.cache/vllm/profiler}"
  PROFILER_CONFIG="{\"profiler\":\"torch\",\"torch_profiler_dir\":\"${PROFILER_DIR}\",\"ignore_frontend\":true,\"delay_iterations\":0,\"active_iterations\":200,\"torch_profiler_with_stack\":false,\"torch_profiler_use_gzip\":true,\"torch_profiler_record_shapes\":false}"
  EXTRA_ARGS+=(--profiler-config "$PROFILER_CONFIG")
fi

SPEC_CONFIG="{\"method\":\"qwen3_5_mtp\",\"num_speculative_tokens\":${MTP_TOKENS}}"

echo "=== Launching Huihui-Qwen3.6-27B-NVFP4-MTP ($HF_MODEL) — vision + MTP + CUTE_PAGED ==="
echo "  Model:       $HF_MODEL"
echo "  Quant:       modelopt (NVFP4)"
echo "  Attention:   $ATTN_BACKEND"
echo "  KV cache:    $KV_CACHE"
echo "  Context:     $MAX_MODEL_LEN tokens"
echo "  Max seqs:    $MAX_NUM_SEQS"
echo "  MTP tokens:  $MTP_TOKENS"
echo "  Vision:      enabled"
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
  --quantization modelopt \
  --speculative-config "$SPEC_CONFIG" \
  --kv-cache-dtype "$KV_CACHE" \
  --attention-backend "$ATTN_BACKEND" \
  --max-model-len "$MAX_MODEL_LEN" \
  --max-num-seqs "$MAX_NUM_SEQS" \
  --mamba-cache-mode align \
  --trust-remote-code \
  --gpu-memory-utilization "${SERVE_GPU_UTIL:-0.70}" \
  --max-num-batched-tokens 65536 \
  --kernel-config '{"enable_flashinfer_autotune":false}' \
  "${EXTRA_ARGS[@]}"

echo "Container started: $CONTAINER"
echo "  API:  http://localhost:${PORT}/v1"
echo "  Logs: docker logs -f $CONTAINER"
