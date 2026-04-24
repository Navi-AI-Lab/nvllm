#!/bin/bash
# Phase E.1 follow-up #4 — matched-concurrency baseline for β-lite comparison.
#
# Existing β-lite leg (benchmarks/nvllm/traces/phase_e/2026-04-23-initial/
# beta_lite.pt.trace.json.gz) ran at concurrent=8, max_tokens=64.
# Original baseline ran at concurrent=4, max_tokens=256 — comparison was
# NOT apples-to-apples per Caveat #1 in that leg's summary.md.
#
# This script captures a baseline (CUTE_PHASE_E_FUSION=0) leg at
# concurrent=8, max_tokens=64, num_seqs=8 so β-lite's per-kernel μs can
# be compared against it directly.
#
# Output: benchmarks/nvllm/traces/phase_e_1/2026-04-24-baseline-matched/
#   - baseline_matched.pt.trace.json.gz (gitignored; extracted CSV committed)
#   - baseline_matched_kernels.csv (via extract_e2e_kernels.py)
#   - baseline_matched_serve.log
#   - baseline_matched_mem.log
#   - summary.md (hand-written: kernel-duration table β-lite vs matched baseline)
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
OUT_DIR="$REPO_ROOT/benchmarks/nvllm/traces/phase_e_1/2026-04-24-baseline-matched"
IMAGE="${NVLLM_IMAGE:-nvllm:gb10}"
HF_MODEL="${HF_MODEL:-ig1/Qwen3.5-27B-NVFP4}"
CONTAINER="nvllm"
PORT=8000

mkdir -p "$OUT_DIR"

PROFILER_CONFIG='{"profiler":"torch","torch_profiler_dir":"/tmp/profiles","ignore_frontend":true,"delay_iterations":0,"active_iterations":200,"torch_profiler_with_stack":false,"torch_profiler_use_gzip":true,"torch_profiler_record_shapes":false}'

run_leg() {
  local LABEL="$1" FUSION="$2" E_PATH="$3" NUM_SEQS="$4" WARMUP_N="$5" TIMED_N="$6" CONCURRENT="$7"
  local TRACE_TARGET="$OUT_DIR/${LABEL}.pt.trace.json.gz"
  local MEM_LOG="$OUT_DIR/${LABEL}_mem.log"

  echo ""
  echo "=============================================================="
  echo "=== Leg: $LABEL  (FUSION=$FUSION PATH=$E_PATH num_seqs=$NUM_SEQS concurrent=$CONCURRENT)"
  echo "=============================================================="

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
  local WATCHDOG_PID=$!
  trap "kill $WATCHDOG_PID 2>/dev/null" RETURN

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
    -e CUTE_DEBUG_FUSION=0 \
    -e CUTE_MLP_FUSION=1 \
    -e CUTE_ATTN_FUSION=1 \
    -e CUTE_DEBUG_MLP_FUSION=0 \
    -e CUTE_BETA_MIN_FREE_GB=8 \
    -e CUTE_PHASE_E_FUSION="$FUSION" \
    -e CUTE_PHASE_E_PATH="$E_PATH" \
    "$IMAGE" \
    serve \
    --model "$HF_MODEL" \
    --served-model-name default \
    --host 0.0.0.0 --port "$PORT" \
    --kv-cache-dtype fp8_e4m3 \
    --attention-backend CUTE_PAGED \
    --max-model-len 65536 \
    --max-num-seqs "$NUM_SEQS" \
    --language-model-only \
    --limit-mm-per-prompt '{"image": 0, "video": 0}' \
    --mamba-cache-mode align \
    --trust-remote-code \
    --gpu-memory-utilization 0.70 \
    --max-num-batched-tokens 65536 \
    --compilation-config '{"cudagraph_mode":"PIECEWISE"}' \
    --profiler-config "$PROFILER_CONFIG"

  echo "Container up. Waiting for readiness (up to 20 min)..."
  local READY=0
  for i in $(seq 1 240); do
    if ! docker ps --filter name="$CONTAINER" --format '{{.Names}}' | grep -q "^${CONTAINER}$"; then
      echo "ERROR: container died while loading"
      docker logs --tail 200 "$CONTAINER" > "$OUT_DIR/${LABEL}_fail.log" 2>&1 || true
      return 1
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
    return 1
  fi

  echo "  Warmup: $WARMUP_N requests (tolerant timeout)..."
  for i in $(seq 1 "$WARMUP_N"); do
    curl -s --max-time 600 "http://localhost:$PORT/v1/completions" \
      -H "Content-Type: application/json" \
      -d '{"model":"default","prompt":"Q: Janet has 3 apples and buys 5 more.\nA:","max_tokens":32,"temperature":0}' > /dev/null || \
      echo "    warmup $i timeout/err (ok during JIT)"
  done
  echo "  Warmup done."

  cd "$REPO_ROOT"
  echo "  Capturing with timed workload ($TIMED_N requests, concurrent=$CONCURRENT, max_tokens=64)..."
  .venv/bin/python docs/research/gemm_sweep/trace_workload.py \
    --base-url "http://localhost:$PORT/v1" \
    --model default \
    --warmup 2 --timed "$TIMED_N" --concurrent "$CONCURRENT" \
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

  local LATEST_TRACE
  LATEST_TRACE=$(ls -t "$OUT_DIR"/rank*.pt.trace.json.gz 2>/dev/null | head -n1 || true)
  if [ -z "$LATEST_TRACE" ]; then
    echo "ERROR: no fresh trace in $OUT_DIR — profiler likely never flushed"
    ls -la "$OUT_DIR"
    return 1
  fi
  mv "$LATEST_TRACE" "$TRACE_TARGET"
  local TRACE_BYTES
  TRACE_BYTES=$(stat -c%s "$TRACE_TARGET")
  echo "  Saved: ${LABEL}.pt.trace.json.gz ($TRACE_BYTES bytes)"

  # Extract per-kernel CSV for committed evidence.
  echo "  Extracting per-kernel CSV..."
  .venv/bin/python docs/research/gemm_sweep/extract_e2e_kernels.py \
    --trace "$TRACE_TARGET" --config "$LABEL" \
    --out "$OUT_DIR/${LABEL}_kernels.csv" || \
    echo "  WARN: CSV extraction failed (check trace integrity)"

  kill $WATCHDOG_PID 2>/dev/null || true
  trap - RETURN
  return 0
}

# Matched baseline — FUSION=0 ("Phase E disabled", pre-E decode path),
# same workload as the β-lite leg: num_seqs=8, concurrent=8, timed=5,
# max_tokens=64.
run_leg "baseline_matched" "0" "auto" "8" "4" "5" "8" || \
  echo "baseline_matched leg FAILED"

echo ""
echo "=============================================================="
echo "=== Final state ==="
echo "=============================================================="
ls -la "$OUT_DIR"
