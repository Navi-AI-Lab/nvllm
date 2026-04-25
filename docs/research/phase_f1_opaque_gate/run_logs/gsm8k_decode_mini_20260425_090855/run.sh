#!/bin/bash
# GSM8K-50 gate for decode-mini preset (8.45 tok/s, +345% over prefill-legacy).
# Tile registry edits live in `vllm/v1/attention/backends/cute_paged/_tile_presets.py`
# (bind-mounted; no rebuild needed). Same MLP-only fusion config as the sweeps:
# CUTE_MLP_FUSION=1, others 0.
#
# Pass criterion: ≥ 90% (≥45/50) on seed=42 — the agreed gate per
# project_fusion_debug_plan + feedback_post_quant_sanity.
#
# Total wall: ~5 min compile + ~50 min eval = ~55 min worst case.
set -euo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"
NVLLM_IMAGE="nvllm:gb10"
HF_MODEL="ig1/Qwen3.5-27B-NVFP4"
HOST_PRESETS="/home/natfii/docker/nvllm/vllm/v1/attention/backends/cute_paged/_tile_presets.py"
CONT_PRESETS="/app/nvllm/vllm/v1/attention/backends/cute_paged/_tile_presets.py"
PORT=8000
CONTAINER="nvllm-gsm8k-decode-mini"
SERVE_LOG="$DIR/serve.log"
RESULT_JSON="$DIR/eval_result.json"
RUN_LOG="$DIR/run.log"

exec > >(tee -a "$RUN_LOG") 2>&1

docker rm -f "$CONTAINER" >/dev/null 2>&1 || true

echo "==================================================================="
echo "=== GSM8K-50 gate — CUTE_MLP_TILE=decode-mini (64, 640, 8) ==="
echo "==================================================================="
echo "  Image:   $NVLLM_IMAGE"
echo "  Model:   $HF_MODEL"
echo "  Port:    $PORT"
echo "  Bind:    $HOST_PRESETS -> $CONT_PRESETS (ro)"
echo "  Result:  $RESULT_JSON"
echo ""

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
  -e CUTE_MLP_TILE=decode-mini \
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
LOG_PID=$!

cleanup() {
  echo ""
  echo "=== Cleanup ==="
  kill "$LOG_PID" 2>/dev/null || true
  docker stop "$CONTAINER" >/dev/null 2>&1 || true
  docker rm "$CONTAINER" >/dev/null 2>&1 || true
}
trap cleanup EXIT

echo ""
echo "Waiting for /v1/models 200 (up to 10 min)..."
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

echo ""
echo "=== Sanity: confirm CUTE_MLP_TILE=decode-mini was selected ==="
grep "Phase D MLP tile preset" "$SERVE_LOG" | head -1 || echo "WARN: no tile-preset line in serve log"

echo ""
echo "=== Running gsm8k_eval_50.py seed=42 ==="
EVAL_START=$(date +%s)
cd /home/natfii/docker/nvllm
.venv/bin/python scripts/gsm8k_eval_50.py \
  --api "http://localhost:${PORT}/v1" \
  --model default \
  --n 50 \
  --seed 42 \
  --max-tokens 512 \
  --label "decode_mini_gate" \
  --save "$RESULT_JSON"
EVAL_END=$(date +%s)
echo ""
echo "=== Eval wall: $((EVAL_END - EVAL_START))s ==="
echo ""
echo "=== Result summary ==="
python3 -c "
import json
j = json.load(open('$RESULT_JSON'))
print(f'  Label:    {j.get(\"label\", \"(unknown)\")}')
print(f'  N:        {j.get(\"n\", \"?\")}')
print(f'  Correct:  {j.get(\"correct\", \"?\")}')
print(f'  Accuracy: {j.get(\"accuracy\", 0)*100:.1f}%')
print(f'  Wall:     {j.get(\"wall_seconds\", \"?\")}s')
" 2>/dev/null || cat "$RESULT_JSON" | head -c 800

echo ""
echo "=== Done ==="
