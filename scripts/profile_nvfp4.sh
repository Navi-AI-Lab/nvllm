#!/bin/bash
# nvllm -- Profile NVFP4 model serving with nsys
#
# Captures a CUDA kernel trace during a benchmark run.
# Restarts the container with nsys wrapping vLLM, fires load, then collects.
#
# nsys must be installed on the HOST at /opt/nvidia/nsight-systems/2025.6.3/
# (it gets volume-mounted into the container).
#
# Usage:
#   ./scripts/profile_nvfp4.sh                  # profile the nvllm model
#   DURATION=60 ./scripts/profile_nvfp4.sh      # custom capture duration
#   PROMPTS=100 ./scripts/profile_nvfp4.sh      # more inference load

set -euo pipefail

source "$(dirname "$0")/common.sh"

PORT="${PORT:-8000}"
BASE_URL="http://localhost:${PORT}"
DURATION="${DURATION:-45}"
PROMPTS="${PROMPTS:-50}"
RESULT_DIR="benchmarks/nvllm/results"
TIMESTAMP=$(date +%Y%m%d-%H%M%S)
OUTPUT_NAME="profile-${TIMESTAMP}"

# nsys host installation
NSYS_HOST_DIR="/opt/nvidia/nsight-systems/2025.6.3"
NSYS_CONTAINER_DIR="/opt/nsight-systems"
NSYS_BIN="${NSYS_CONTAINER_DIR}/target-linux-sbsa-armv8/nsys"

mkdir -p "$RESULT_DIR"

# Verify nsys on host
if [ ! -d "$NSYS_HOST_DIR" ]; then
  echo "ERROR: nsys not found at $NSYS_HOST_DIR" >&2
  echo "Install NVIDIA Nsight Systems 2025.6.3" >&2
  exit 1
fi

# ── Configuration from the current run script ──
HF_MODEL="natfii/Qwen3.5-27B-NVFP4-Opus-GB10"
CONTAINER="nvllm"
SERVED_NAME="default"

echo "=== nsys Profiling ==="
echo "  Model:       $HF_MODEL"
echo "  Duration:    ${DURATION}s capture"
echo "  Prompts:     $PROMPTS inference requests"
echo "  Output:      $RESULT_DIR/$OUTPUT_NAME"
echo ""

# ── Stop existing container ──
echo "Stopping existing container..."
nvllm_cleanup_container "$CONTAINER"
sleep 2

# ── Start container with nsys wrapping vLLM ──
echo "Starting container with nsys profiling..."
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
  --cuda-trace-scope=process-tree \
  --cuda-graph-trace=node \
  --sample=none \
  --cpuctxsw=none \
  --delay=120 \
  --duration="$DURATION" \
  --output="/tmp/$OUTPUT_NAME" \
  --force-overwrite=true \
  --stats=true \
  python3 -m vllm.entrypoints.cli.main serve \
  --model "$HF_MODEL" \
  --served-model-name "$SERVED_NAME" \
  --host 0.0.0.0 --port "$PORT" \
  --kv-cache-dtype turboquant35 \
  --attention-backend TRITON_ATTN \
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

echo "Container started under nsys (120s delay before capture starts)."
echo "Waiting for model to load..."

# ── Wait for server ready ──
MAX_WAIT=300
ELAPSED=0
while ! curl -sf "${BASE_URL}/v1/models" > /dev/null 2>&1; do
  sleep 5
  ELAPSED=$((ELAPSED + 5))
  if [ "$ELAPSED" -ge "$MAX_WAIT" ]; then
    echo "ERROR: Server not ready after ${MAX_WAIT}s" >&2
    echo "Logs:" >&2
    docker logs "$CONTAINER" --tail 30 >&2
    exit 1
  fi
  printf "  Waiting... (%ds)\r" "$ELAPSED"
done
echo "Server ready after ${ELAPSED}s"
echo ""

# ── Fire inference load ──
echo "Firing $PROMPTS inference requests (4 concurrent)..."
for i in $(seq 1 "$PROMPTS"); do
  # Send request
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

# ── Wait for nsys to finish (duration-based) ──
echo "Waiting for nsys capture to complete..."
while docker exec "$CONTAINER" test -f /tmp/${OUTPUT_NAME}.nsys-rep 2>/dev/null; do
  break
done
# Give it extra time for the report generation
sleep 10

# ── Copy results out ──
echo "Copying results..."
docker cp "$CONTAINER:/tmp/${OUTPUT_NAME}.nsys-rep" "${RESULT_DIR}/${OUTPUT_NAME}.nsys-rep" 2>/dev/null || true
docker cp "$CONTAINER:/tmp/${OUTPUT_NAME}.sqlite" "${RESULT_DIR}/${OUTPUT_NAME}.sqlite" 2>/dev/null || true

# ── Extract stats ──
if [ -f "${RESULT_DIR}/${OUTPUT_NAME}.nsys-rep" ]; then
  echo ""
  echo "=== Top CUDA Kernels by Total GPU Time ==="
  nsys stats --report cuda_gpu_kern_sum "${RESULT_DIR}/${OUTPUT_NAME}.nsys-rep" 2>/dev/null | head -30 || \
    echo "(Install nsys-ui to view: nsys-ui ${RESULT_DIR}/${OUTPUT_NAME}.nsys-rep)"
  echo ""
  echo "=== CUDA Memory Operations ==="
  nsys stats --report cuda_gpu_mem_size_sum "${RESULT_DIR}/${OUTPUT_NAME}.nsys-rep" 2>/dev/null | head -20 || true
  echo ""
  echo "Profile saved to: ${RESULT_DIR}/${OUTPUT_NAME}.nsys-rep"
else
  echo "WARNING: nsys report not found. Check container logs:" >&2
  docker logs "$CONTAINER" --tail 30 >&2
fi

echo ""
echo "=== Profiling Complete ==="
echo "  Report: ${RESULT_DIR}/${OUTPUT_NAME}.nsys-rep"
echo "  View:   nsys-ui ${RESULT_DIR}/${OUTPUT_NAME}.nsys-rep"
echo ""
echo "NOTE: Container is still running under nsys. To restart normally:"
echo "  docker rm -f $CONTAINER && ./scripts/run_qwen35_27b_nvfp4.sh"
