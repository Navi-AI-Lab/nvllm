#!/bin/bash
# Phase E evidence capture — three legs (baseline, β-coop, β-lite) sequentially.
#
# Uses vLLM's built-in torch profiler via --profiler-config + /start_profile
# + /stop_profile (matches current shipped practice; nsys can't capture V1
# EngineCore kernels — see benchmarks/nvllm/traces/cute_paged_attn/2026-04-13-nsys/summary.md).
#
# Output: benchmarks/nvllm/traces/phase_e/2026-04-23-initial/
#   baseline.pt.trace.json.gz, beta_coop.pt.trace.json.gz, beta_lite.pt.trace.json.gz
#   baseline_serve.log, beta_coop_serve.log, beta_lite_serve.log
#
# Each trace is ~170 MB and gitignored; the summary.md and per-leg kernels CSVs
# (extracted via docs/research/gemm_sweep/extract_e2e_kernels.py) are committed.
#
# Usage:  bash docs/research/phase_e_traces/capture_all.sh
#         NVLLM_IMAGE=nvllm:gb10 bash ...   # override image
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
OUT_DIR="$REPO_ROOT/benchmarks/nvllm/traces/phase_e/2026-04-23-initial"
IMAGE="${NVLLM_IMAGE:-nvllm:gb10}"
HF_MODEL="${HF_MODEL:-ig1/Qwen3.5-27B-NVFP4}"
CONTAINER="nvllm"
PORT=8000

mkdir -p "$OUT_DIR"

# Profiler config — torch + CUPTI, per-iteration trace with shape + module
# metadata for per-layer attribution. active_iterations=600 auto-finalizes
# the trace file even if /stop_profile races CUPTI serialization.
PROFILER_CONFIG='{"profiler":"torch","torch_profiler_dir":"/tmp/profiles","ignore_frontend":true,"delay_iterations":0,"active_iterations":600,"torch_profiler_with_stack":false,"torch_profiler_use_gzip":true,"torch_profiler_record_shapes":true}'

run_leg() {
  local LABEL="$1" FUSION="$2" E_PATH="$3" NUM_SEQS="$4" WARMUP_N="$5" TIMED_N="$6" CONCURRENT="$7"
  local TRACE_TARGET="$OUT_DIR/${LABEL}.pt.trace.json.gz"

  echo ""
  echo "=============================================================="
  echo "=== Leg: $LABEL  (FUSION=$FUSION PATH=$E_PATH num_seqs=$NUM_SEQS)"
  echo "=============================================================="

  docker rm -f "$CONTAINER" 2>/dev/null || true
  sleep 2

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
      echo "ERROR: container died while loading" >&2
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
    echo "ERROR: server never became ready" >&2
    docker logs --tail 200 "$CONTAINER" > "$OUT_DIR/${LABEL}_fail.log" 2>&1 || true
    return 1
  fi

  # β-coop warmup covers the 16 per-layer JIT recompiles (~6 min).
  # Other legs: short warmup just primes CuTe decode + MLP kernels.
  echo "  Warmup: $WARMUP_N requests (tolerant timeout)..."
  local WARM_TIMEOUT=600
  for i in $(seq 1 "$WARMUP_N"); do
    curl -s --max-time "$WARM_TIMEOUT" "http://localhost:$PORT/v1/completions" \
      -H "Content-Type: application/json" \
      -d '{"model":"default","prompt":"Q: Janet has 3 apples and buys 5 more.\nA:","max_tokens":64,"temperature":0}' > /dev/null || \
      echo "    warmup $i timeout/err (ok during JIT)"
  done
  echo "  Warmup done."

  # Timed profiled burst — deterministic workload via trace_workload.py
  cd "$REPO_ROOT"
  echo "  Capturing with timed workload ($TIMED_N requests, concurrent=$CONCURRENT)..."
  .venv/bin/python docs/research/gemm_sweep/trace_workload.py \
    --base-url "http://localhost:$PORT/v1" \
    --model default \
    --warmup 5 --timed "$TIMED_N" --concurrent "$CONCURRENT" \
    --max-tokens 256 \
    --timeout 600 \
    --profile-start "http://localhost:$PORT/start_profile" \
    --profile-stop  "http://localhost:$PORT/stop_profile"

  # CUPTI flush — torch+CUPTI serialization needs 30-90s after active_iterations
  # window closes. 120s is dumb-but-reliable.
  echo "  CUPTI flush: sleeping 120s..."
  for i in 1 2 3 4; do
    sleep 30
    echo "    [flush +${i}0s]"
    ls -la "$OUT_DIR"/rank*.pt.trace.json.gz 2>/dev/null | awk '{print "      current size:", $5}' || true
  done

  # Collect logs
  docker logs "$CONTAINER" > "$OUT_DIR/${LABEL}_serve.log" 2>&1

  # Teardown
  docker stop "$CONTAINER" >/dev/null
  docker rm "$CONTAINER" >/dev/null

  # Rename the profiler trace to our canonical name. Profiler writes
  # <hostname>_<pid>.<timestamp>.pt.trace.json.gz into /tmp/profiles
  # which is bind-mounted to $OUT_DIR.
  local LATEST_TRACE
  LATEST_TRACE=$(ls -t "$OUT_DIR"/rank*.pt.trace.json.gz 2>/dev/null | head -n1 || true)
  if [ -z "$LATEST_TRACE" ]; then
    echo "ERROR: no fresh trace in $OUT_DIR — profiler likely never flushed" >&2
    ls -la "$OUT_DIR" >&2
    return 1
  fi
  mv "$LATEST_TRACE" "$TRACE_TARGET"
  local TRACE_BYTES
  TRACE_BYTES=$(stat -c%s "$TRACE_TARGET")
  echo "  Saved: ${LABEL}.pt.trace.json.gz ($TRACE_BYTES bytes)"
}

# Leg 1 — baseline: Phase E disabled. num_seqs=4, concurrent=4.
run_leg "baseline" "0" "auto" "4" "4" "30" "4"

# Leg 2 — β-coop: unified cooperative launch at num_seqs=1, concurrent=1.
# Warmup=15 to cover 16 per-layer JIT compiles (~6 min first-call cost).
# Timed=10 because at concurrent=1 each request is ~23s serial; 10 × 23s = 4 min.
run_leg "beta_coop" "1" "coop" "1" "15" "10" "1"

# Leg 3 — β-lite: two-kernel path at num_seqs=8, concurrent=8. High load
# to give β-lite realistic multi-seq fan-out stats.
run_leg "beta_lite" "1" "lite" "8" "4" "30" "8"

echo ""
echo "=============================================================="
echo "=== All 3 legs complete ==="
echo "=============================================================="
ls -la "$OUT_DIR"
