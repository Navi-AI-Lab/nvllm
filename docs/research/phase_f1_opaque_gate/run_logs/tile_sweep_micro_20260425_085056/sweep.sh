#!/bin/bash
# Tile-preset MICRO sweep — narrower than decode-balanced (winner of
# the 4-preset pass at 2.80 tok/s). All three candidates hold total
# CTAs at 2176 (same as balanced) by halving slice_ctas alongside tile_s.
#
# Hypothesis: smaller tile_s → less smem + fewer regs per CTA → better
# occupancy at same CTA count. Tested in PIECEWISE end-to-end via
# `CUTE_MLP_TILE` env override on the SHIPPED image; we bind-mount only
# `_tile_presets.py` to inject the new registry entries (no rebuild).
#
# Tested presets (all ÷ intermediate_size=17408 cleanly):
#   decode-mini:  ( 64, 640,  8)  →  num_slices=272, total_CTAs=2176
#   decode-32:    ( 32, 640,  4)  →  num_slices=544, total_CTAs=2176
#   decode-micro: ( 16, 640,  2)  →  num_slices=1088, total_CTAs=2176
#
# Per-preset wall: ~5 min compile + ~1.5 min completion = ~6.5 min.
# Total sweep: ~20 min for 3 presets.
set -euo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"
NVLLM_IMAGE="nvllm:gb10"
HF_MODEL="ig1/Qwen3.5-27B-NVFP4"
HOST_PRESETS="/home/natfii/docker/nvllm/vllm/v1/attention/backends/cute_paged/_tile_presets.py"
CONT_PRESETS="/app/nvllm/vllm/v1/attention/backends/cute_paged/_tile_presets.py"
PORT=8000

run_preset() {
  local PRESET="$1"
  local OUT="$DIR/${PRESET//-/_}"
  local SERVE_LOG="$OUT/serve.log"
  local COMPLETION_LOG="$OUT/completion.json"
  local TIMING_LOG="$OUT/curl_timing.txt"
  local CONTAINER="nvllm-tile-micro-${PRESET//_/-}"

  mkdir -p "$OUT"
  docker rm -f "$CONTAINER" >/dev/null 2>&1 || true

  echo ""
  echo "==================================================================="
  echo "=== CUTE_MLP_TILE=$PRESET — PIECEWISE 256-token e2e timing ==="
  echo "==================================================================="

  docker run -d \
    --name "$CONTAINER" \
    --gpus all --ipc=host --network host \
    -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
    -v "$HOME/.cache/flashinfer:/root/.cache/flashinfer" \
    -v "$HOST_PRESETS:$CONT_PRESETS:ro" \
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
      --kernel-config '{"enable_flashinfer_autotune": false}' >/dev/null

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
  grep "Phase D MLP tile preset" "$SERVE_LOG" | head -1 || echo "WARN: no tile-preset line in serve log"

  echo ""
  echo "=== Warmup completion (32 tokens, discard timing) ==="
  curl -sS http://localhost:${PORT}/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
      "model": "default",
      "prompt": "Q: What is 2+2?\nA:",
      "max_tokens": 32,
      "temperature": 0.0
    }' >/dev/null

  echo ""
  echo "=== Measured completion (256 tokens, sustained tok/s) ==="
  TIME_OUT=$(curl -sS \
    -w 'PERF: total=%{time_total}s connect=%{time_connect}s starttransfer=%{time_starttransfer}s\n' \
    -o "$COMPLETION_LOG" \
    http://localhost:${PORT}/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
      "model": "default",
      "prompt": "Write a short paragraph about clouds, weather, and rain. Then count to ten. Then list three colors.",
      "max_tokens": 256,
      "temperature": 0.0
    }' 2>&1 | tail -1)
  echo "$TIME_OUT" > "$TIMING_LOG"

  echo ""
  echo "=== Completion preview ==="
  python3 -c "import json; j=json.load(open('$COMPLETION_LOG')); print(j['choices'][0]['text'][:200])" 2>/dev/null || cat "$COMPLETION_LOG" | head -c 400
  echo ""
  echo "=== Timing ==="
  cat "$TIMING_LOG"

  cleanup
  echo "=== $PRESET done ==="
}

run_preset "decode-mini"
run_preset "decode-32"
run_preset "decode-micro"

echo ""
echo "=== MICRO SWEEP COMPLETE ==="
echo ""
echo "Per-preset results:"
for P in decode_mini decode_32 decode_micro; do
  T=$(grep -oE 'total=[0-9.]+s' "$DIR/$P/curl_timing.txt" 2>/dev/null | head -1 || echo "n/a")
  TXT=$(python3 -c "import json; j=json.load(open('$DIR/$P/completion.json')); print(j['choices'][0]['text'][:80].replace(chr(10),' '))" 2>/dev/null || echo "n/a")
  echo "  $P  $T  preview=$TXT"
done
echo ""
echo "Reference (from prior 4-preset sweep at 075336):"
echo "  prefill_legacy   total=134.74s  → 1.90 tok/s"
echo "  decode_balanced  total= 91.53s  → 2.80 tok/s ← prior winner"
echo "  decode_small     total= 96.94s  → 2.64 tok/s"
echo "  decode_narrow    total=145.78s  → 1.76 tok/s"
