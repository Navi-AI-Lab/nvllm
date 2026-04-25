#!/bin/bash
# Tile-preset env sweep — probe-4 config (PHASE_E=0 MLP=1 ATTN=0), --enforce-eager.
# Runs the same 32-token completion twice: once with CUTE_MLP_TILE=decode-small,
# once with decode-balanced. No rebuild — just env override on the shipped image.
#
# Hypothesis from torch profile: vllm::cute_mlp_forward 26 ms/call is per-element
# sync_threads()-dominated. `slice_ctas` is the real parallelism lever:
#   prefill-legacy  : slice_ctas=8   (grid 64 CTAs)   — baseline 26 ms/call
#   decode-balanced : slice_ctas=16  (grid 128 CTAs)
#   decode-small    : slice_ctas=32  (grid 256 CTAs)  — 4x CTAs, expected win
set -euo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"
NVLLM_IMAGE="nvllm:gb10"
HF_MODEL="ig1/Qwen3.5-27B-NVFP4"
HOST_FILE="/home/natfii/docker/nvllm/vllm/nvllm/models/qwen3_5.py"
CONT_FILE="/app/nvllm/vllm/nvllm/models/qwen3_5.py"
PORT=8000

run_preset() {
  local PRESET="$1"
  local OUT="$DIR/${PRESET//-/_}"
  local SERVE_LOG="$OUT/serve.log"
  local COMPLETION_LOG="$OUT/completion.json"
  local TIMING_GREP="$OUT/timing_lines.txt"
  local CONTAINER="nvllm-tile-sweep-${PRESET//_/-}"

  mkdir -p "$OUT"
  docker rm -f "$CONTAINER" >/dev/null 2>&1 || true

  echo ""
  echo "==================================================================="
  echo "=== CUTE_MLP_TILE=$PRESET — launching fused+eager probe ==="
  echo "==================================================================="

  docker run -d \
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
    -e CUTE_ATTN_FUSION=0 \
    -e CUTE_DEBUG_MLP_FUSION=0 \
    -e CUTE_BETA_MIN_FREE_GB=8 \
    -e CUTE_PHASE_E_FUSION=0 \
    -e CUTE_PHASE_E_PATH=auto \
    -e CUTE_DEBUG_TIMING=1 \
    -e CUTE_DEBUG_TIMING_BUDGET=120 \
    -e CUTE_MLP_TILE="$PRESET" \
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
      --enforce-eager >/dev/null

  docker logs -f "$CONTAINER" > "$SERVE_LOG" 2>&1 &
  local LOG_PID=$!

  cleanup() {
    kill "$LOG_PID" 2>/dev/null || true
    docker stop "$CONTAINER" >/dev/null 2>&1 || true
    docker rm "$CONTAINER" >/dev/null 2>&1 || true
  }

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
      cleanup
      return 1
    fi
    sleep 5
  done

  echo ""
  echo "=== Sanity: confirm CUTE_MLP_TILE=$PRESET was selected ==="
  grep "Phase D MLP tile preset" "$SERVE_LOG" | head -3 || echo "WARN: no tile-preset line in serve log"

  echo ""
  echo "=== 32-token completion ==="
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
  echo "=== Extracting timing lines ==="
  grep "\[CUTE_TIMING\]" "$SERVE_LOG" > "$TIMING_GREP" || true
  echo "Found $(wc -l < "$TIMING_GREP") timing log lines -> $TIMING_GREP"

  cleanup
  echo "=== $PRESET done ==="
}

run_preset "decode-small"
run_preset "decode-balanced"

echo ""
echo "=== SWEEP COMPLETE ==="
ls -la "$DIR"
