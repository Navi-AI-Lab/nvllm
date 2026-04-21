#!/bin/bash
# Copyright 2026 Navi Ai Labs
# SPDX-License-Identifier: Apache-2.0
#
# nvllm -- Profile CuTe Paged Attention kernel with nsys
#
# Wraps vLLM entrypoint with nsys (CUPTI must inject at process start).
# Uses a calibrated --delay so nsys captures only steady-state inference,
# not model loading. The script health-checks, warms up, and fires
# requests during the delay period.
#
# nsys must be installed on the HOST at /opt/nvidia/nsight-systems/2025.6.3/
# (it gets volume-mounted into the container).
#
# Usage:
#   ./scripts/profile_cute_paged.sh                     # CuTe Paged Attention
#   BACKEND=FLASHINFER ./scripts/profile_cute_paged.sh  # FlashInfer baseline
#   DURATION=90 ./scripts/profile_cute_paged.sh         # custom capture duration
#   NSYS_DELAY=400 ./scripts/profile_cute_paged.sh      # custom delay

set -euo pipefail

source "$(dirname "$0")/common.sh"

PORT="${PORT:-8000}"
BASE_URL="http://localhost:${PORT}"
DURATION="${DURATION:-60}"
PROMPTS="${PROMPTS:-50}"
BACKEND="${BACKEND:-CUTE_PAGED}"
# Delay must exceed model load (~295s under nsys) + warmup (~120s).
# Default 480s gives 65s buffer after typical warmup completes.
NSYS_DELAY="${NSYS_DELAY:-480}"
TIMESTAMP=$(date +%Y%m%d-%H%M%S)

# Determine kv-cache-dtype based on backend
case "$BACKEND" in
  CUTE_PAGED)
    KV_CACHE_DTYPE="fp8_e4m3"
    LABEL="cute_paged"
    ;;
  FLASHINFER)
    KV_CACHE_DTYPE="fp8_e4m3"
    LABEL="flashinfer"
    ;;
  *)
    echo "ERROR: Unknown backend '$BACKEND'. Use CUTE_PAGED or FLASHINFER." >&2
    exit 1
    ;;
esac

RESULT_DIR="benchmarks/nvllm/traces/cute_paged_attn/$(date +%Y-%m-%d)-nsys"
OUTPUT_NAME="profile-${LABEL}-${TIMESTAMP}"

# nsys host installation (volume-mounted into container)
NSYS_HOST_DIR="/opt/nvidia/nsight-systems/2025.6.3"
NSYS_CONTAINER_DIR="/opt/nsight-systems"
NSYS_BIN="${NSYS_CONTAINER_DIR}/target-linux-sbsa-armv8/nsys"

HF_MODEL="${HF_MODEL:-ig1/Qwen3.5-27B-NVFP4}"
CONTAINER="nvllm"
SERVED_NAME="default"

mkdir -p "$RESULT_DIR"

# Verify nsys on host
if [ ! -d "$NSYS_HOST_DIR" ]; then
  echo "ERROR: nsys not found at $NSYS_HOST_DIR" >&2
  echo "Install NVIDIA Nsight Systems 2025.6.3" >&2
  exit 1
fi

echo "=== nsys Profiling: ${BACKEND} ==="
echo "  Model:       $HF_MODEL"
echo "  Backend:     $BACKEND"
echo "  KV cache:    $KV_CACHE_DTYPE"
echo "  nsys delay:  ${NSYS_DELAY}s (capture starts after model load + warmup)"
echo "  nsys dur:    ${DURATION}s"
echo "  Prompts:     $PROMPTS inference requests"
echo "  Output:      $RESULT_DIR/$OUTPUT_NAME"
echo ""

# ── Stop existing container ──
echo "Stopping existing container..."
nvllm_cleanup_container "$CONTAINER"
sleep 2

# ── Start container with nsys wrapping vLLM entrypoint ──
# CUPTI injection requires nsys to launch the target process.
echo "Starting container with nsys entrypoint..."
docker run -d \
  --name "$CONTAINER" \
  --gpus all \
  --ipc=host \
  --network host \
  --privileged \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  -v "$HOME/.cache/flashinfer:/root/.cache/flashinfer" \
  -v "${NSYS_HOST_DIR}:${NSYS_CONTAINER_DIR}:ro" \
  -e VLLM_NVFP4_GEMM_BACKEND=cutlass \
  -e VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  --entrypoint "$NSYS_BIN" \
  "$NVLLM_IMAGE" \
  profile \
  --trace=cuda,nvtx,cublas \
  --cuda-trace-scope=system-wide \
  --cuda-graph-trace=node \
  --sample=none \
  --cpuctxsw=none \
  --delay="$NSYS_DELAY" \
  --duration="$DURATION" \
  --output="/tmp/$OUTPUT_NAME" \
  --force-overwrite=true \
  --stats=true \
  python3 -m vllm.entrypoints.cli.main serve \
  --model "$HF_MODEL" \
  --served-model-name "$SERVED_NAME" \
  --host 0.0.0.0 --port "$PORT" \
  --kv-cache-dtype "$KV_CACHE_DTYPE" \
  --attention-backend "$BACKEND" \
  --max-model-len 65536 \
  --max-num-seqs 4 \
  --language-model-only \
  --enable-prefix-caching \
  --mamba-cache-mode align \
  --mamba-block-size 64 \
  --trust-remote-code \
  --gpu-memory-utilization 0.80 \
  --max-num-batched-tokens 65536 \
  --compilation-config '{"cudagraph_mode":"PIECEWISE"}'

echo "Container started. nsys will capture at t=${NSYS_DELAY}s for ${DURATION}s."
echo "Waiting for model to load..."

# ── Wait for server ready ──
MAX_WAIT=600
ELAPSED=0
while ! curl -sf "${BASE_URL}/v1/models" > /dev/null 2>&1; do
  sleep 5
  ELAPSED=$((ELAPSED + 5))
  if [ "$ELAPSED" -ge "$MAX_WAIT" ]; then
    echo "ERROR: Server not ready after ${MAX_WAIT}s" >&2
    docker logs "$CONTAINER" --tail 30 >&2
    exit 1
  fi
  printf "  Waiting... (%ds)\r" "$ELAPSED"
done
echo "Server ready after ${ELAPSED}s"
echo ""

# ── Warmup (4 requests to trigger JIT + CUDA graph capture) ──
echo "Warmup (4 sequential requests — triggers CUDA graph capture)..."
for i in 1 2 3 4; do
  curl -s "$BASE_URL/v1/completions" \
    -H "Content-Type: application/json" \
    -d "{\"model\":\"$SERVED_NAME\",\"prompt\":\"Hello, world! Tell me about quantum computing.\",\"max_tokens\":64}" > /dev/null 2>&1
done
echo "Warmup done."
echo ""

# ── Fire steady-state inference load ──
# These requests run during nsys delay AND capture window.
# With delay=480 and model load ~295s + warmup ~120s, requests start
# at ~415s. nsys capture window is 480-540s, overlapping with decode.
echo "Firing $PROMPTS inference requests (4 concurrent)..."
echo "  nsys capture window: t=${NSYS_DELAY}s to t=$((NSYS_DELAY + DURATION))s"
for i in $(seq 1 "$PROMPTS"); do
  curl -s "$BASE_URL/v1/completions" \
    -H "Content-Type: application/json" \
    -d "{\"model\":\"$SERVED_NAME\",\"prompt\":\"Explain the implications of quantum entanglement for secure communication systems, covering BB84 protocol, quantum key distribution, and post-quantum cryptography approaches.\",\"max_tokens\":128}" > /dev/null 2>&1 &

  # Keep 4 concurrent
  if (( i % 4 == 0 )); then
    wait
  fi
done
wait
echo "All requests completed."
echo ""

# ── Wait for nsys capture to finish + report generation ──
echo "Waiting for nsys report..."
MAX_NSYS_WAIT=$((NSYS_DELAY + DURATION + 60))
NSYS_ELAPSED=0
while ! docker exec "$CONTAINER" test -f "/tmp/${OUTPUT_NAME}.nsys-rep" 2>/dev/null; do
  sleep 10
  NSYS_ELAPSED=$((NSYS_ELAPSED + 10))
  # Check if container is still running
  if ! docker ps --filter name="$CONTAINER" --format '{{.Status}}' | grep -q Up; then
    echo "Container exited. Checking for report..."
    break
  fi
  if [ "$NSYS_ELAPSED" -ge "$MAX_NSYS_WAIT" ]; then
    echo "WARNING: nsys report not found after ${MAX_NSYS_WAIT}s" >&2
    break
  fi
  printf "  Waiting for nsys report... (%ds)\r" "$NSYS_ELAPSED"
done
echo ""

# ── Copy results out ──
echo "Copying results..."
docker cp "$CONTAINER:/tmp/${OUTPUT_NAME}.nsys-rep" "${RESULT_DIR}/${OUTPUT_NAME}.nsys-rep" 2>/dev/null || true
docker cp "$CONTAINER:/tmp/${OUTPUT_NAME}.sqlite" "${RESULT_DIR}/${OUTPUT_NAME}.sqlite" 2>/dev/null || true

# ── Extract stats ──
if [ -f "${RESULT_DIR}/${OUTPUT_NAME}.nsys-rep" ]; then
  echo ""
  echo "=== Top CUDA Kernels by Total GPU Time ==="
  nsys stats --force-export=true --report cuda_gpu_kern_sum \
    "${RESULT_DIR}/${OUTPUT_NAME}.nsys-rep" 2>/dev/null | head -40 || \
    echo "(View with: nsys-ui ${RESULT_DIR}/${OUTPUT_NAME}.nsys-rep)"
  echo ""
  echo "=== CUDA Memory Operations ==="
  nsys stats --force-export=true --report cuda_gpu_mem_size_sum \
    "${RESULT_DIR}/${OUTPUT_NAME}.nsys-rep" 2>/dev/null | head -20 || true
  echo ""
  echo "Profile saved to: ${RESULT_DIR}/${OUTPUT_NAME}.nsys-rep"
else
  echo "WARNING: nsys report not found. Check container logs:" >&2
  docker logs "$CONTAINER" --tail 30 >&2
fi

echo ""
echo "=== Profiling Complete: ${BACKEND} ==="
echo "  Report: ${RESULT_DIR}/${OUTPUT_NAME}.nsys-rep"
echo "  View:   nsys-ui ${RESULT_DIR}/${OUTPUT_NAME}.nsys-rep"
echo ""
echo "To profile the other backend for comparison:"
if [ "$BACKEND" = "CUTE_PAGED" ]; then
  echo "  BACKEND=FLASHINFER ./scripts/profile_cute_paged.sh"
else
  echo "  ./scripts/profile_cute_paged.sh"
fi
