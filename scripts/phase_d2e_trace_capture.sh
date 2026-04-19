#!/bin/bash
# Phase D2e trace capture — identical to phase_d2_trace_capture.sh but
# targets the D2e image (stable-output-buf fix) and a fresh output dir.
# Keeps CUTE_ATTN_FUSION=0 as the D2d diagnostic isolates the MLP path.
set -euo pipefail

MODE="${1:?must pass 'baseline' or 'changed'}"
case "$MODE" in
  baseline) CUTE_MLP_FUSION=0 ;;
  changed)  CUTE_MLP_FUSION=1 ;;
  *) echo "MODE must be baseline|changed" >&2; exit 1 ;;
esac

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# Env override: CUTE_ATTN_FUSION controls whether attn fusion runs.
# Default 1 (full stack). Set to 0 for the D2d isolation diagnostic.
ATTN_FUSION="${CUTE_ATTN_FUSION:-1}"
SUBDIR="2026-04-19-phase-d2e-global-scale-fix"
if [ "$ATTN_FUSION" = "0" ]; then
  SUBDIR="2026-04-19-phase-d2e-stable-output-buf"
fi
OUT_DIR="$REPO_ROOT/benchmarks/nvllm/traces/cute_paged_mlp_fusion/$SUBDIR/$MODE"
mkdir -p "$OUT_DIR"

CONTAINER="nvllm-phased2e-$MODE"
IMAGE="${NVLLM_IMAGE:-nvllm:gb10-phaseD2e}"
PORT=8000

docker rm -f "$CONTAINER" 2>/dev/null || true
docker rm -f nvllm 2>/dev/null || true

echo "=== Phase D2e trace capture: $MODE (CUTE_MLP_FUSION=$CUTE_MLP_FUSION) ==="
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
  -e CUTE_ATTN_FUSION="$ATTN_FUSION" \
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
  docker logs --tail 100 "$CONTAINER" >&2 || true
  docker rm -f "$CONTAINER" 2>/dev/null || true
  exit 1
fi

# Reuse D2 helpers for profiler start/stop/workload/gsm8k/log collection.
# phase_d2_trace_capture.sh inlined them; for D2e we shell into the same
# helpers via docker exec + the same curl endpoints.
echo "Warmup..."
for w in 1 2; do
  curl -sf http://localhost:$PORT/v1/completions \
    -H "Content-Type: application/json" \
    -d '{"model":"default","prompt":"Warmup.","max_tokens":16,"temperature":0}' \
    >/dev/null || true
done
echo "  Warmup done."

echo "Starting profiler..."
curl -sf -X POST http://localhost:$PORT/start_profile >/dev/null

echo
echo "Running profile workload (4×128 tok)..."
for i in 1 2 3 4; do
  curl -sf http://localhost:$PORT/v1/completions \
    -H "Content-Type: application/json" \
    -d '{"model":"default","prompt":"Count 1 to 20: 1,","max_tokens":128,"temperature":0,"ignore_eos":true}' \
    -o "$OUT_DIR/workload_$i.json" &
done
wait
echo "  Workload done."

echo "Stopping profiler..."
curl -sf -X POST http://localhost:$PORT/stop_profile >/dev/null
sleep 3

echo
echo "GSM8K sanity check..."
"$REPO_ROOT/.venv/bin/python" "$REPO_ROOT/scripts/gsm8k_sanity.py" \
  --api "http://localhost:$PORT/v1" --model default \
  --label "phase_d2e_$MODE" \
  --save "$OUT_DIR/gsm8k_$MODE.json" \
  2>&1 | tee "$OUT_DIR/gsm8k_$MODE.log"

echo "Collecting logs..."
docker logs "$CONTAINER" > "$OUT_DIR/decode_log.txt" 2>&1 || true

echo "Stopping container..."
docker rm -f "$CONTAINER" >/dev/null 2>&1 || true

echo
echo "=== Trace artifacts in $OUT_DIR ==="
ls -la "$OUT_DIR"
