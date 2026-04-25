#!/bin/bash
# Timing probe — mount-override qwen3_5.py with CUTE_DEBUG_TIMING
# instrumentation. Run fused+eager. Send single short completion.
# Goal: identify which per-step checkpoint owns the 0.8 tok/s budget.
set -euo pipefail
DIR="$(dirname "$0")"
DIR="$(cd "$DIR" && pwd)"
SERVE_LOG="$DIR/serve.log"
COMPLETION_LOG="$DIR/completion.json"
TIMING_GREP="$DIR/timing_lines.txt"
HOST_FILE="/home/natfii/docker/nvllm/vllm/nvllm/models/qwen3_5.py"
CONT_FILE="/app/nvllm/vllm/nvllm/models/qwen3_5.py"

CONTAINER="nvllm-timing-phaseE-off"
HF_MODEL="ig1/Qwen3.5-27B-NVFP4"
PORT=8000
NVLLM_IMAGE="nvllm:gb10"

docker rm -f "$CONTAINER" >/dev/null 2>&1 || true

echo "=== Launching fused+eager+timing probe (PHASE_E=0, MLP+ATTN=1) ==="
echo "  Host file:      $HOST_FILE"
echo "  Container path: $CONT_FILE (mount-override, ro)"
echo "  Evidence:       $DIR"
echo ""

CONTAINER_ID=$(docker run -d \
  --name "$CONTAINER" \
  --gpus all --ipc=host --network host \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  -v "$HOME/.cache/flashinfer:/root/.cache/flashinfer" \
  -v "$HOST_FILE:$CONT_FILE:ro" \
  -e VLLM_NVFP4_GEMM_BACKEND=cutlass \
  -e VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e CUTE_DEBUG_FUSION=0 \
  -e CUTE_MLP_FUSION=1 \
  -e CUTE_ATTN_FUSION=1 \
  -e CUTE_DEBUG_MLP_FUSION=0 \
  -e CUTE_BETA_MIN_FREE_GB=8 \
  -e CUTE_PHASE_E_FUSION=0 \
  -e CUTE_PHASE_E_PATH=auto \
  -e CUTE_DEBUG_TIMING=1 \
  -e CUTE_DEBUG_TIMING_BUDGET=120 \
  "$NVLLM_IMAGE" \
  serve \
    --model "$HF_MODEL" \
    --served-model-name default \
    --host 0.0.0.0 --port "$PORT" \
    --kv-cache-dtype fp8_e4m3 \
    --attention-backend CUTE_PAGED \
    --max-model-len 65536 \
    --max-num-seqs 1 \
    --max-num-batched-tokens 65536 \
    --language-model-only \
    --limit-mm-per-prompt '{"image": 0, "video": 0}' \
    --mamba-cache-mode align \
    --trust-remote-code \
    --gpu-memory-utilization 0.70 \
    --kernel-config '{"enable_flashinfer_autotune": false}' \
    --enforce-eager)

echo "Container started: $CONTAINER_ID"

docker logs -f "$CONTAINER" > "$SERVE_LOG" 2>&1 &
LOG_PID=$!

cleanup() {
  echo ""
  echo "=== Cleanup ==="
  kill "$LOG_PID" 2>/dev/null || true
  docker stop "$CONTAINER" >/dev/null 2>&1 || true
  docker rm "$CONTAINER" >/dev/null 2>&1 || true
}
trap cleanup EXIT

echo "Waiting for /v1/models 200..."
START=$(date +%s)
DEADLINE=$((START + 600))
while true; do
  if curl -s -o /dev/null -w "%{http_code}" "http://localhost:${PORT}/v1/models" 2>/dev/null | grep -q "^200$"; then
    READY=$(date +%s)
    echo "READY in $((READY - START))s"
    break
  fi
  if [ "$(date +%s)" -ge "$DEADLINE" ]; then
    echo "TIMEOUT waiting for ready"
    tail -n 30 "$SERVE_LOG"
    exit 1
  fi
  sleep 5
done

# Confirm our mount overlay actually loaded by grepping the log for our marker
echo ""
echo "=== Sanity: confirm CUTE_DEBUG_TIMING import survived ==="
docker exec "$CONTAINER" head -30 "$CONT_FILE" | grep -E "_CUTE_DEBUG_TIMING|_cute_tlog" | head -3 || echo "WARNING: timing markers not visible in container file"

echo ""
echo "=== Running single 32-token completion ==="
curl -sS http://localhost:${PORT}/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "default",
    "prompt": "Q: What is 2+2?\nA:",
    "max_tokens": 32,
    "temperature": 0.0
  }' | tee "$COMPLETION_LOG"

echo ""
echo ""
echo "=== Extracting timing lines from serve log ==="
grep "\[CUTE_TIMING\]" "$SERVE_LOG" > "$TIMING_GREP" || true
echo "Found $(wc -l < "$TIMING_GREP") timing log lines."

echo ""
echo "=== Done ==="
ls -la "$DIR"
