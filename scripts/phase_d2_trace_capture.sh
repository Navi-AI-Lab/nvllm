#!/bin/bash
# Phase D2 trace capture — identical workload to phase_d_trace_capture.sh
# but targets the phase-d2 image + output dir (op-body-move design).
#
# Usage:
#   ./scripts/phase_d2_trace_capture.sh <baseline|changed>
#
# Requires nvllm:gb10-phaseD2 image (CUTE_MLP_FUSION gate + op-body launch).
set -euo pipefail

MODE="${1:?must pass 'baseline' or 'changed'}"
case "$MODE" in
  baseline) CUTE_MLP_FUSION=0 ;;
  changed)  CUTE_MLP_FUSION=1 ;;
  *) echo "MODE must be baseline|changed" >&2; exit 1 ;;
esac

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT_DIR="$REPO_ROOT/benchmarks/nvllm/traces/cute_paged_mlp_fusion/2026-04-18-phase-d2-op-body-move/$MODE"
mkdir -p "$OUT_DIR"

CONTAINER="nvllm-phased2-$MODE"
IMAGE="${NVLLM_IMAGE:-nvllm:gb10-phaseD2}"
PORT=8000

docker rm -f "$CONTAINER" 2>/dev/null || true
docker rm -f nvllm 2>/dev/null || true

echo "=== Phase D2 trace capture: $MODE (CUTE_MLP_FUSION=$CUTE_MLP_FUSION) ==="
echo "  Image:    $IMAGE"
echo "  Output:   $OUT_DIR"
echo "  Container $CONTAINER on port $PORT"

PROFILER_CONFIG='{"profiler":"torch","torch_profiler_dir":"/tmp/profiles","ignore_frontend":true,"delay_iterations":3,"active_iterations":30,"torch_profiler_with_stack":false,"torch_profiler_use_gzip":true}'

docker run -d \
  --name "$CONTAINER" \
  --gpus all \
  --ipc=host \
  --network host \
  --privileged \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  -v "$HOME/.cache/flashinfer:/root/.cache/flashinfer" \
  -v "$OUT_DIR:/tmp/profiles" \
  -e VLLM_NVFP4_GEMM_BACKEND=cutlass \
  -e VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e CUTE_ATTN_FUSION=0 \
  -e CUTE_DEBUG_FUSION=1 \
  -e CUTE_MLP_FUSION="$CUTE_MLP_FUSION" \
  -e CUTE_DEBUG_MLP_FUSION=1 \
  "$IMAGE" \
  serve \
  --model natfii/Qwen3.5-27B-NVFP4-Opus-GB10 \
  --served-model-name default \
  --host 0.0.0.0 --port "$PORT" \
  --kv-cache-dtype fp8_e4m3 \
  --attention-backend CUTE_PAGED \
  --max-model-len 65536 \
  --max-num-seqs 4 \
  --language-model-only \
  --mamba-cache-mode align \
  --trust-remote-code \
  --gpu-memory-utilization 0.80 \
  --max-num-batched-tokens 65536 \
  --compilation-config '{"cudagraph_mode":"PIECEWISE"}' \
  --profiler-config "$PROFILER_CONFIG"

echo "Container up. Waiting for readiness..."

for i in $(seq 1 240); do
  if curl -sf http://localhost:$PORT/v1/models >/dev/null 2>&1; then
    echo "  Server ready at iter=$i (~$((i * 5)) s)."
    break
  fi
  sleep 5
done

if ! curl -sf http://localhost:$PORT/v1/models >/dev/null 2>&1; then
  echo "ERROR: server never became ready — dumping last 100 log lines." >&2
  docker logs --tail 100 "$CONTAINER" >&2
  docker rm -f "$CONTAINER" || true
  exit 1
fi

echo "Warmup..."
for i in 1 2; do
  curl -s http://localhost:$PORT/v1/completions \
    -H 'Content-Type: application/json' \
    -d '{"model":"default","prompt":"Hello","max_tokens":32,"temperature":0,"ignore_eos":true}' \
    >/dev/null
done
echo "  Warmup done."

echo "Starting profiler..."
curl -sf -X POST http://localhost:$PORT/start_profile
echo ""

echo "Running profile workload (4×128 tok)..."
PROMPT="The quick brown fox jumps over the lazy dog. The sun rose over the mountains casting long shadows across the valley floor. Birds sang in the trees as"
for i in 1 2 3 4; do
  curl -s http://localhost:$PORT/v1/completions \
    -H 'Content-Type: application/json' \
    -d "{\"model\":\"default\",\"prompt\":\"$PROMPT\",\"max_tokens\":128,\"temperature\":0,\"ignore_eos\":true}" \
    > "$OUT_DIR/workload_$i.json" &
done
wait
echo "  Workload done."

echo "Stopping profiler..."
curl -sf -X POST http://localhost:$PORT/stop_profile
echo ""
sleep 3

echo "GSM8K sanity check..."
"$REPO_ROOT/.venv/bin/python" "$REPO_ROOT/scripts/gsm8k_sanity.py" \
  --api "http://localhost:$PORT/v1" --model default \
  --label "phase_d2_$MODE" \
  --save "$OUT_DIR/gsm8k_$MODE.json" \
  > "$OUT_DIR/gsm8k_$MODE.log" 2>&1 || true

echo "Collecting logs..."
docker logs "$CONTAINER" > "$OUT_DIR/decode_log.txt" 2>&1

echo "Stopping container..."
docker stop "$CONTAINER" >/dev/null
docker rm "$CONTAINER" >/dev/null

echo ""
echo "=== Trace artifacts in $OUT_DIR ==="
ls -la "$OUT_DIR"
