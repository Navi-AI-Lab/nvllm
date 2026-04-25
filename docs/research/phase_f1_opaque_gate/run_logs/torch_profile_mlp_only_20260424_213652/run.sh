#!/bin/bash
# Torch profile probe — capture vLLM torch profiler trace under
# probe-4 config (PHASE_E=0 MLP=1 ATTN=0). Goal: identify which
# sub-kernel inside the Phase D MLP fusion contributes most to its
# +22 ms/layer cost over baseline.
#
# Per memory:feedback_vllm_profiling, nsys can't see V1 EngineCore;
# use vLLM's built-in /start_profile + /stop_profile API.
set -euo pipefail
DIR="$(dirname "$0")"
DIR="$(cd "$DIR" && pwd)"
SERVE_LOG="$DIR/serve.log"
COMPLETION_LOG="$DIR/completion.json"
HOST_FILE="/home/natfii/docker/nvllm/vllm/nvllm/models/qwen3_5.py"
CONT_FILE="/app/nvllm/vllm/nvllm/models/qwen3_5.py"
HOST_PROFILE_DIR="$DIR/profile"
CONT_PROFILE_DIR="/tmp/vllm_profile"
mkdir -p "$HOST_PROFILE_DIR"

CONTAINER="nvllm-torch-profile"
HF_MODEL="ig1/Qwen3.5-27B-NVFP4"
PORT=8000
NVLLM_IMAGE="nvllm:gb10"

docker rm -f "$CONTAINER" >/dev/null 2>&1 || true

echo "=== Launching torch-profile probe (P4 config: PHASE_E=0 MLP=1 ATTN=0) ==="
echo "  Evidence: $DIR"
echo ""

CONTAINER_ID=$(docker run -d \
  --name "$CONTAINER" \
  --gpus all --ipc=host --network host \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  -v "$HOME/.cache/flashinfer:/root/.cache/flashinfer" \
  -v "$HOST_FILE:$CONT_FILE:ro" \
  -v "$HOST_PROFILE_DIR:$CONT_PROFILE_DIR" \
  -e VLLM_NVFP4_GEMM_BACKEND=cutlass \
  -e VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e CUTE_DEBUG_FUSION=0 \
  -e CUTE_MLP_FUSION=1 \
  -e CUTE_ATTN_FUSION=0 \
  -e CUTE_DEBUG_MLP_FUSION=0 \
  -e CUTE_BETA_MIN_FREE_GB=8 \
  -e CUTE_PHASE_E_FUSION=0 \
  -e CUTE_PHASE_E_PATH=auto \
  -e CUTE_DEBUG_TIMING=0 \
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
    --profiler-config "{\"profiler\": \"torch\", \"torch_profiler_dir\": \"$CONT_PROFILE_DIR\", \"torch_profiler_with_stack\": false, \"torch_profiler_use_gzip\": true}" \
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

# Warmup: do a small completion BEFORE profiling, so the JIT compile cliff
# (15s legacy MLP + 43s β-coop) drains and we profile steady-state only.
echo ""
echo "=== Warmup completion (16 tokens) — drains JIT compile cliffs ==="
curl -sS http://localhost:${PORT}/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"default","prompt":"Hello","max_tokens":16,"temperature":0.0}' \
  > "$DIR/warmup.json"
echo "Warmup done."

echo ""
echo "=== /start_profile ==="
curl -sS -X POST "http://localhost:${PORT}/start_profile"
echo ""

echo "=== Send 32-tok completion (this is what gets profiled) ==="
curl -sS http://localhost:${PORT}/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "default",
    "prompt": "Q: What is 2+2?\nA:",
    "max_tokens": 32,
    "temperature": 0.0
  }' | tee "$COMPLETION_LOG"
echo ""

echo "=== /stop_profile ==="
curl -sS -X POST "http://localhost:${PORT}/stop_profile"
echo ""

# Profiler may be flushing — wait briefly for trace files to land
echo ""
echo "=== Waiting up to 60s for trace files to appear ==="
WAIT_DEADLINE=$(( $(date +%s) + 60 ))
while [ "$(date +%s)" -lt "$WAIT_DEADLINE" ]; do
  if find "$HOST_PROFILE_DIR" -name "*.gz" -o -name "*.json" 2>/dev/null | head -1 | grep -q .; then
    break
  fi
  sleep 2
done

echo ""
echo "=== Trace files ==="
find "$HOST_PROFILE_DIR" -type f -ls 2>/dev/null

echo ""
echo "=== Done ==="
ls -la "$DIR"
