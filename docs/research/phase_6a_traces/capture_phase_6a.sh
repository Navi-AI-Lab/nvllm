#!/bin/bash
# Phase 6a β-coop hot-path Python diet — torch-profiler trace capture.
#
# Mirrors docs/research/phase_e_traces/capture_beta_only.sh β-coop leg
# settings so the result is comparable to phase_e/2026-04-23-initial. nsys
# can't capture vLLM V1's spawned EngineCore (see
# benchmarks/nvllm/traces/cute_paged_attn/2026-04-13-nsys/summary.md), so
# we use vLLM's torch profiler API via --profiler-config.
#
# Bind-mounts host Phase 6a sources at /app/nvllm/... so the in-container
# editable install picks them up without a rebuild (Phase 6a edits are
# Python-only).
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
OUT_DIR="$REPO_ROOT/benchmarks/nvllm/traces/phase_6a/2026-04-29-initial"
IMAGE="${NVLLM_IMAGE:-nvllm:gb10}"
HF_MODEL="${HF_MODEL:-ig1/Qwen3.5-27B-NVFP4}"
CONTAINER="nvllm"
PORT=8000
LABEL="phase_6a_beta_coop"
TRACE_TARGET="$OUT_DIR/${LABEL}.pt.trace.json.gz"
MEM_LOG="$OUT_DIR/${LABEL}_mem.log"

mkdir -p "$OUT_DIR"

PROFILER_CONFIG='{"profiler":"torch","torch_profiler_dir":"/tmp/profiles","ignore_frontend":true,"delay_iterations":0,"active_iterations":200,"torch_profiler_with_stack":false,"torch_profiler_use_gzip":true,"torch_profiler_record_shapes":false}'

docker rm -f "$CONTAINER" 2>/dev/null || true
sleep 2

: > "$MEM_LOG"
( while :; do
    echo "[$(date +%H:%M:%S)]" >> "$MEM_LOG"
    free -h >> "$MEM_LOG" 2>&1
    docker stats --no-stream --format 'docker: {{.Name}} mem={{.MemUsage}} cpu={{.CPUPerc}}' nvllm 2>/dev/null >> "$MEM_LOG"
    echo '---' >> "$MEM_LOG"
    sleep 30
  done ) &
WATCHDOG_PID=$!
trap "kill $WATCHDOG_PID 2>/dev/null" EXIT

docker run -d \
  --name "$CONTAINER" \
  --gpus all \
  --ipc=host \
  --network host \
  --privileged \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  -v "$HOME/.cache/flashinfer:/root/.cache/flashinfer" \
  -v "$OUT_DIR:/tmp/profiles" \
  -v "$REPO_ROOT/vllm/v1/attention/backends/cute_paged/_backend.py:/app/nvllm/vllm/v1/attention/backends/cute_paged/_backend.py:ro" \
  -v "$REPO_ROOT/vllm/v1/attention/backends/cute_paged/_beta_coop_op.py:/app/nvllm/vllm/v1/attention/backends/cute_paged/_beta_coop_op.py:ro" \
  -v "/tmp/phase5_pycache_blank:/app/nvllm/vllm/v1/attention/backends/cute_paged/__pycache__:ro" \
  -e VLLM_NVFP4_GEMM_BACKEND=cutlass \
  -e VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e CUTE_DEBUG_FUSION=0 \
  -e CUTE_MLP_FUSION=1 \
  -e CUTE_ATTN_FUSION=1 \
  -e CUTE_DEBUG_MLP_FUSION=0 \
  -e CUTE_BETA_MIN_FREE_GB=8 \
  -e CUTE_PHASE_E_FUSION=1 \
  -e CUTE_PHASE_E_PATH=coop \
  "$IMAGE" \
  serve \
  --model "$HF_MODEL" \
  --served-model-name default \
  --host 0.0.0.0 --port "$PORT" \
  --kv-cache-dtype fp8_e4m3 \
  --attention-backend CUTE_PAGED \
  --max-model-len 65536 \
  --max-num-seqs 1 \
  --language-model-only \
  --limit-mm-per-prompt '{"image": 0, "video": 0}' \
  --mamba-cache-mode align \
  --trust-remote-code \
  --gpu-memory-utilization 0.70 \
  --max-num-batched-tokens 65536 \
  --compilation-config '{"cudagraph_mode":"PIECEWISE"}' \
  --profiler-config "$PROFILER_CONFIG"

echo "Container up. Waiting for readiness (up to 20 min)..."
READY=0
for i in $(seq 1 240); do
  if ! docker ps --filter name="$CONTAINER" --format '{{.Names}}' | grep -q "^${CONTAINER}$"; then
    echo "ERROR: container died while loading"
    docker logs --tail 200 "$CONTAINER" > "$OUT_DIR/${LABEL}_fail.log" 2>&1 || true
    exit 1
  fi
  if curl -sf "http://localhost:$PORT/v1/models" -o /dev/null 2>&1; then
    echo "  Server ready at iter=$i (~$((i * 5)) s)."
    READY=1
    break
  fi
  sleep 5
done
if [ "$READY" -ne 1 ]; then
  echo "ERROR: server never became ready"
  docker logs --tail 200 "$CONTAINER" > "$OUT_DIR/${LABEL}_fail.log" 2>&1 || true
  exit 1
fi

echo "  Warmup: 15 requests (β-coop JIT)..."
for i in $(seq 1 15); do
  curl -s --max-time 600 "http://localhost:$PORT/v1/completions" \
    -H "Content-Type: application/json" \
    -d '{"model":"default","prompt":"Q: Janet has 3 apples and buys 5 more.\nA:","max_tokens":32,"temperature":0}' > /dev/null || \
    echo "    warmup $i timeout/err (ok during JIT)"
done
echo "  Warmup done."

cd "$REPO_ROOT"
echo "  Capturing with timed workload (5 requests, concurrent=1, max_tokens=64)..."
.venv/bin/python docs/research/gemm_sweep/trace_workload.py \
  --base-url "http://localhost:$PORT/v1" \
  --model default \
  --warmup 2 --timed 5 --concurrent 1 \
  --max-tokens 64 \
  --timeout 600 \
  --profile-start "http://localhost:$PORT/start_profile" \
  --profile-stop  "http://localhost:$PORT/stop_profile"

echo "  CUPTI flush: sleeping 120s..."
for i in 1 2 3 4; do
  sleep 30
  echo "    [flush +${i}0s]"
  ls -la "$OUT_DIR"/rank*.pt.trace.json.gz 2>/dev/null | awk '{print "      current size:", $5}' || true
done

docker logs "$CONTAINER" > "$OUT_DIR/${LABEL}_serve.log" 2>&1
docker stop "$CONTAINER" >/dev/null 2>&1 || true
docker rm "$CONTAINER" >/dev/null 2>&1 || true

LATEST_TRACE=$(ls -t "$OUT_DIR"/rank*.pt.trace.json.gz 2>/dev/null | head -n1 || true)
if [ -z "$LATEST_TRACE" ]; then
  echo "ERROR: no fresh trace in $OUT_DIR — profiler likely never flushed"
  ls -la "$OUT_DIR"
  exit 1
fi
mv "$LATEST_TRACE" "$TRACE_TARGET"
TRACE_BYTES=$(stat -c%s "$TRACE_TARGET")
echo "  Saved: ${LABEL}.pt.trace.json.gz ($TRACE_BYTES bytes)"

kill $WATCHDOG_PID 2>/dev/null || true
trap - EXIT

echo ""
echo "Extracting per-kernel CSV..."
.venv/bin/python docs/research/gemm_sweep/extract_e2e_kernels.py \
  --trace "$TRACE_TARGET" --config "$LABEL" \
  --out "$OUT_DIR/${LABEL}_kernels.csv" || \
  echo "  WARN: kernel extraction failed (trace may still be usable)"

echo ""
echo "=== Final state: ==="
ls -la "$OUT_DIR"
