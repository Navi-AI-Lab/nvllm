#!/bin/bash
# Phase D3a 7/8 GSM8K floor investigation.
# Runs the GSM8K sanity harness against nvllm:gb10-phaseD2e image
# (D2e-era code, pre-D3a tile registry) to A/B against D3a's prefill-legacy
# result. If D2e image scores 8/8, the D3a 7/8 floor is a D3a-image
# regression. If D2e image also scores ≤7/8, D2e's summary "true 8/8" claim
# was optimistic and the 7/8 is the real session-baseline.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
IMAGE="nvllm:gb10-phaseD2e"
CONTAINER="nvllm-phased2e-q2-check"
PORT=8000
OUT_DIR="$REPO_ROOT/docs/research/phase_d3a_q2_investigation_out"
mkdir -p "$OUT_DIR"

# Port preflight (match D3a sweep's protection)
if curl -sf "http://localhost:$PORT/v1/models" >/dev/null 2>&1; then
  echo "ERROR: port $PORT already serving a /v1/models endpoint — free it first." >&2
  exit 1
fi

docker rm -f "$CONTAINER" 2>/dev/null || true
docker rm -f nvllm 2>/dev/null || true

echo "=== Q2 investigation: GSM8K against $IMAGE ==="

docker run -d \
  --name "$CONTAINER" \
  --gpus all \
  --ipc=host \
  --network host \
  --privileged \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  -v "$HOME/.cache/flashinfer:/root/.cache/flashinfer" \
  -e VLLM_NVFP4_GEMM_BACKEND=cutlass \
  -e VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e CUTE_ATTN_FUSION=1 \
  -e CUTE_MLP_FUSION=1 \
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
  --compilation-config '{"cudagraph_mode":"PIECEWISE"}'

echo "Container up. Waiting for readiness..."
for i in $(seq 1 240); do
  if curl -sf http://localhost:$PORT/v1/models >/dev/null 2>&1; then
    echo "Server ready at iter=$i (~$((i * 5)) s)."
    break
  fi
  sleep 5
done

if ! curl -sf http://localhost:$PORT/v1/models >/dev/null 2>&1; then
  echo "ERROR: server never ready — dumping last 100 log lines." >&2
  docker logs --tail 100 "$CONTAINER" >&2 || true
  docker rm -f "$CONTAINER" 2>/dev/null || true
  exit 1
fi

echo "Warmup (matches sweep pattern)..."
for w in 1 2; do
  curl -sf http://localhost:$PORT/v1/completions \
    -H "Content-Type: application/json" \
    -d '{"model":"default","prompt":"Warmup.","max_tokens":16,"temperature":0}' \
    >/dev/null || true
done

echo "Running 4x128 workload (matches sweep pattern, no profiler)..."
for i in 1 2 3 4; do
  curl -sf http://localhost:$PORT/v1/completions \
    -H "Content-Type: application/json" \
    -d '{"model":"default","prompt":"Count 1 to 20: 1,","max_tokens":128,"temperature":0,"ignore_eos":true}' \
    -o "$OUT_DIR/workload_$i.json" &
done
wait

echo "GSM8K sanity check..."
set +o pipefail
"$REPO_ROOT/.venv/bin/python" "$REPO_ROOT/scripts/gsm8k_sanity.py" \
  --api "http://localhost:$PORT/v1" --model default \
  --label "phase_d2e_q2_check" \
  --save "$OUT_DIR/gsm8k_d2e_image.json" \
  2>&1 | tee "$OUT_DIR/gsm8k_d2e_image.log"
gsm8k_exit=${PIPESTATUS[0]}
set -o pipefail
echo "$gsm8k_exit" > "$OUT_DIR/gsm8k_d2e_image.exit"

echo "Collecting logs..."
docker logs "$CONTAINER" > "$OUT_DIR/decode_log.txt" 2>&1 || true

echo "Tearing down..."
docker rm -f "$CONTAINER" >/dev/null 2>&1 || true

echo ""
echo "=== Investigation artifacts in $OUT_DIR ==="
ls -la "$OUT_DIR"
