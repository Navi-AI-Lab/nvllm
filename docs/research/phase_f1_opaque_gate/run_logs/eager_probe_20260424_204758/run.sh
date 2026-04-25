#!/bin/bash
# Eager probe — does the fused-path 0.8 tok/s perf collapse persist
# under --enforce-eager? Discriminates kernel-bug from graph-capture-bug.
#
# Mirror flags from prior fused leg (serve_fused_20260424_191700.log):
#   - ig1/Qwen3.5-27B-NVFP4
#   - max_model_len=65536, max_num_seqs=1, max_num_batched_tokens=65536
#   - gpu_memory_utilization=0.7
#   - attention-backend CUTE_PAGED, kv-cache-dtype fp8_e4m3
#   - CUTE_*_FUSION=1 (all three)
#   - kernel_config: enable_flashinfer_autotune=False
# Differences:
#   - --enforce-eager (no --compilation-config)
set -euo pipefail
DIR="$(dirname "$0")"
DIR="$(cd "$DIR" && pwd)"
SERVE_LOG="$DIR/serve.log"
EVAL_JSON="$DIR/eval.json"
SUMMARY="$DIR/summary.txt"

CONTAINER="nvllm-eager-probe"
HF_MODEL="ig1/Qwen3.5-27B-NVFP4"
PORT=8000
NVLLM_IMAGE="nvllm:gb10"

# Cleanup any leftover
docker rm -f "$CONTAINER" >/dev/null 2>&1 || true

echo "=== Launching fused+eager probe ==="
echo "  Evidence: $DIR"
echo "  Container: $CONTAINER"
echo "  Image: $NVLLM_IMAGE"
echo ""

CONTAINER_ID=$(docker run -d \
  --name "$CONTAINER" \
  --gpus all --ipc=host --network host \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  -v "$HOME/.cache/flashinfer:/root/.cache/flashinfer" \
  -e VLLM_NVFP4_GEMM_BACKEND=cutlass \
  -e VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e CUTE_DEBUG_FUSION=0 \
  -e CUTE_MLP_FUSION=1 \
  -e CUTE_ATTN_FUSION=1 \
  -e CUTE_DEBUG_MLP_FUSION=0 \
  -e CUTE_BETA_MIN_FREE_GB=8 \
  -e CUTE_PHASE_E_FUSION=1 \
  -e CUTE_PHASE_E_PATH=auto \
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

# Tail log to file in background
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

# Wait for ready (poll /v1/models, 600s timeout — eager mode skips graph capture but still loads weights)
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
  # Watch for fatal startup errors
  if grep -qE "AssertionError|Traceback|Error launching|EngineCore.*died" "$SERVE_LOG" 2>/dev/null; then
    if ! grep -q "Available KV cache memory" "$SERVE_LOG" 2>/dev/null; then
      :  # only worry once we're past init
    fi
  fi
  sleep 5
done

echo ""
echo "=== Running N=8 GSM8K eval (max_tokens=512, timeout=300) ==="
cd /home/natfii/docker/nvllm
.venv/bin/python scripts/gsm8k_eval_50.py \
  --api "http://localhost:${PORT}/v1" \
  --model default \
  --n 8 --seed 42 \
  --max-tokens 512 --timeout 300 \
  --label fused_eager \
  --save "$EVAL_JSON" 2>&1 | tee "$SUMMARY"

echo ""
echo "=== Done ==="
echo "Evidence in: $DIR"
ls -la "$DIR"
