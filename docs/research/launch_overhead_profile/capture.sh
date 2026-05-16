#!/bin/bash
# Launch-overhead profile capture — 2-leg torch profiler + B' sentinel dump.
#
# Purpose: convert the B' single-call diagnostic (~23 ms wall_minus_regions per
# β-coop call, see project_b_prime_shipped + PR #17) into an AGENTS.md §4
# perf claim with steady-state stats, AND bound the instrumentation tax via a
# CUTE_BETA_REGION_TIMING=0 control leg.
#
# Two legs, both lower8 (CUTE_PHASE_E_LAYERS=3,7) + CUTE_WO_SPLIT=8, decode bs=1:
#   timing_on  — CUTE_BETA_REGION_TIMING=1, sentinel dump of region buf + walls
#   timing_off — CUTE_BETA_REGION_TIMING=0, torch profiler only (control)
#
# Per skill profile-vllm-v1: vLLM V1 EngineCore is spawned, nsys can't follow.
# Torch profiler is primary; called out in summary.md.
#
# Outputs under $OUT_DIR:
#   <leg>.pt.trace.json.gz                  (gitignored, local-only)
#   <leg>_serve.log                         (committed)
#   <leg>_mem.log                           (committed; host watchdog)
#   <leg>_region_timings.npy                (timing_on only; copied from container)
#   <leg>_host_launch_walls.npy             (timing_on only)
#   capture.log                             (this runner's log)
#
# Usage:
#   bash docs/research/launch_overhead_profile/capture.sh smoke   # 100/64, timing_on only
#   bash docs/research/launch_overhead_profile/capture.sh full    # 200/64, both legs
#
# Override defaults:
#   NVLLM_IMAGE=nvllm:gb10 HF_MODEL=ig1/Qwen3.5-27B-NVFP4 bash ... full

set -euo pipefail

MODE="${1:-full}"
case "$MODE" in
  smoke)   TIMED_N=100; LEGS=("timing_on") ;;
  control) TIMED_N=20;  LEGS=("timing_off") ;;
  full)    TIMED_N=200; LEGS=("timing_on" "timing_off") ;;
  *) echo "Usage: $0 {smoke|control|full}" >&2; exit 2 ;;
esac

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
OUT_DIR="$REPO_ROOT/benchmarks/nvllm/traces/cute_paged_attn/2026-05-16-launch-overhead-profile"
# Default to nvllm:gb10-bprime (2026-05-16 clean build at commit 1953ebbb0):
# contains wo_split=8 production, tier-1 cherry-picks, SSM zero-on-realloc,
# qwen3_5.py sentinel-file workaround, AND PR #17 (B') instrumentation.
# Image SHA: sha256:4fccbd915044a8f5f7db8268b0ec645323eb3d7063fd66233e64b1882e7c2539
IMAGE="${NVLLM_IMAGE:-nvllm:gb10-bprime}"
HF_MODEL="${HF_MODEL:-ig1/Qwen3.5-27B-NVFP4}"
CONTAINER="nvllm"
PORT=8000
WARMUP_N=15        # β-coop JIT covers 16 per-layer compiles
MAX_TOKENS=64      # tight window per skill OOM avoidance + user choice
CONCURRENT=1       # β-coop solo-only at coop launch (num_seqs_2_target memory)

mkdir -p "$OUT_DIR"
CAPTURE_LOG="$OUT_DIR/capture.log"
echo "[capture] mode=$MODE legs=${LEGS[*]} timed=$TIMED_N max_tokens=$MAX_TOKENS" | tee -a "$CAPTURE_LOG"
echo "[capture] commit=$(git -C "$REPO_ROOT" rev-parse --short HEAD)" | tee -a "$CAPTURE_LOG"
echo "[capture] started $(date -Iseconds)" | tee -a "$CAPTURE_LOG"

# Both legs use the same profiler config: torch + CUPTI, narrow per skill
# defaults. active_iterations=200 + max_iterations=200 belt-and-suspenders
# auto-finalize. record_shapes=false / with_stack=false to keep buffer small.
PROFILER_CONFIG='{"profiler":"torch","torch_profiler_dir":"/tmp/profiles","ignore_frontend":true,"delay_iterations":0,"active_iterations":200,"max_iterations":200,"warmup_iterations":0,"torch_profiler_with_stack":false,"torch_profiler_use_gzip":true,"torch_profiler_record_shapes":false}'

run_leg() {
  local LABEL="$1" REGION_TIMING="$2"
  local TRACE_TARGET="$OUT_DIR/${LABEL}.pt.trace.json.gz"
  local SERVE_LOG="$OUT_DIR/${LABEL}_serve.log"
  local MEM_LOG="$OUT_DIR/${LABEL}_mem.log"

  echo "" | tee -a "$CAPTURE_LOG"
  echo "==============================================================" | tee -a "$CAPTURE_LOG"
  echo "=== Leg: $LABEL (CUTE_BETA_REGION_TIMING=$REGION_TIMING)" | tee -a "$CAPTURE_LOG"
  echo "==============================================================" | tee -a "$CAPTURE_LOG"

  docker rm -f "$CONTAINER" 2>/dev/null || true
  sleep 2

  # Sentinel-file workaround for vLLM EngineCore env stripping
  # (feedback_vllm_enginecore_env_strip). Without this, B' env vars don't
  # reach the EngineCore subprocess.
  mkdir -p /tmp/c2_diag
  {
    echo "CUTE_C2_DIAG="
    echo "CUTE_C2_DIAG_INJECT_NOISE="
    echo "CUTE_C2_DIAG_DUMP_DIR="
    echo "CUTE_C2_DIAG_TOL_ATOL="
    echo "CUTE_C2_DIAG_TOL_RTOL="
    echo "CUTE_WO_SPLIT=8"
  } > /tmp/c2_diag/ENV

  docker run -d \
    --name "$CONTAINER" \
    --gpus all \
    --ipc=host \
    --network host \
    --privileged \
    -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
    -v "$HOME/.cache/flashinfer:/root/.cache/flashinfer" \
    -v "$OUT_DIR:/tmp/profiles" \
    -v "/tmp/c2_diag:/tmp/c2_diag" \
    -e VLLM_NVFP4_GEMM_BACKEND=cutlass \
    -e VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
    -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    -e CUTE_MLP_FUSION=1 \
    -e CUTE_ATTN_FUSION=1 \
    -e CUTE_BETA_MIN_FREE_GB=8 \
    -e CUTE_PHASE_E_FUSION=1 \
    -e CUTE_PHASE_E_PATH=coop \
    -e CUTE_PHASE_E_LAYERS=3,7 \
    -e CUTE_BETA_REGION_TIMING="$REGION_TIMING" \
    -e CUTE_WO_SPLIT=8 \
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
    --kernel-config '{"enable_flashinfer_autotune":false}' \
    --compilation-config '{"cudagraph_mode":"PIECEWISE"}' \
    --profiler-config "$PROFILER_CONFIG"

  echo "[$LABEL] container up; waiting for /v1/models (up to 20 min)..." | tee -a "$CAPTURE_LOG"
  local READY=0
  for i in $(seq 1 240); do
    if ! docker ps --filter name="$CONTAINER" --format '{{.Names}}' | grep -q "^${CONTAINER}$"; then
      echo "[$LABEL] ERROR: container died during load" | tee -a "$CAPTURE_LOG"
      docker logs --tail 200 "$CONTAINER" > "$SERVE_LOG" 2>&1 || true
      return 1
    fi
    if curl -sf "http://localhost:$PORT/v1/models" -o /dev/null 2>&1; then
      echo "[$LABEL] server ready at iter=$i (~$((i * 5)) s)" | tee -a "$CAPTURE_LOG"
      READY=1
      break
    fi
    sleep 5
  done
  if [ "$READY" -ne 1 ]; then
    echo "[$LABEL] ERROR: server never became ready" | tee -a "$CAPTURE_LOG"
    docker logs --tail 200 "$CONTAINER" > "$SERVE_LOG" 2>&1 || true
    return 1
  fi

  # Sidecar host watchdog (canary for OOM avoidance per skill).
  ( while :; do
      date -Iseconds
      free -h | head -2
      docker stats --no-stream "$CONTAINER" --format '{{.CPUPerc}} {{.MemUsage}}' 2>/dev/null || true
      echo "---"
      sleep 15
    done ) > "$MEM_LOG" 2>&1 &
  local WATCHDOG_PID=$!
  trap "kill $WATCHDOG_PID 2>/dev/null || true" RETURN

  # Warmup outside profiler window — covers β-coop per-layer JIT recompiles
  # (each layer 3 + layer 7 first-call ≈ ~6 min combined).
  echo "[$LABEL] warmup: $WARMUP_N requests at bs=1 (max_time 600s for JIT)" | tee -a "$CAPTURE_LOG"
  for i in $(seq 1 "$WARMUP_N"); do
    curl -s --max-time 600 "http://localhost:$PORT/v1/completions" \
      -H "Content-Type: application/json" \
      -d "{\"model\":\"default\",\"prompt\":\"Q: Janet has 3 apples and buys 5 more.\\nA:\",\"max_tokens\":$MAX_TOKENS,\"temperature\":0}" \
      > /dev/null 2>&1 || echo "[$LABEL]   warmup $i timed out (ok during JIT)" | tee -a "$CAPTURE_LOG"
  done
  echo "[$LABEL] warmup done" | tee -a "$CAPTURE_LOG"

  # Timed profiled burst via trace_workload.py (start + workload + stop).
  cd "$REPO_ROOT"
  echo "[$LABEL] timed window: $TIMED_N reqs concurrent=$CONCURRENT max_tokens=$MAX_TOKENS" | tee -a "$CAPTURE_LOG"
  .venv/bin/python docs/research/gemm_sweep/trace_workload.py \
    --base-url "http://localhost:$PORT/v1" \
    --model default \
    --warmup 2 --timed "$TIMED_N" --concurrent "$CONCURRENT" \
    --max-tokens "$MAX_TOKENS" \
    --timeout 600 \
    --profile-start "http://localhost:$PORT/start_profile" \
    --profile-stop  "http://localhost:$PORT/stop_profile" \
    2>&1 | tee -a "$CAPTURE_LOG"

  # CUPTI flush — wait for trace serialization.
  echo "[$LABEL] CUPTI flush: sleeping 120s..." | tee -a "$CAPTURE_LOG"
  for i in 1 2 3 4; do
    sleep 30
    local sz
    sz=$(ls -la "$OUT_DIR"/rank*.pt.trace.json.gz 2>/dev/null | awk '{print $5}' | head -1 || echo "?")
    echo "[$LABEL]   [flush +${i}0s] trace size: $sz bytes" | tee -a "$CAPTURE_LOG"
  done

  # Drain B' region timing + host launch walls (timing_on leg only).
  # Sentinel-file dump is one-shot — touch it, then send one tiny request
  # to trigger the next β-coop call to consume the sentinel and write npys.
  if [ "$REGION_TIMING" = "1" ]; then
    echo "[$LABEL] dropping sentinel + drain request" | tee -a "$CAPTURE_LOG"
    docker exec "$CONTAINER" touch /tmp/.dump_region_timings || true
    # Drain request — small max_tokens so it completes fast.
    curl -s --max-time 60 "http://localhost:$PORT/v1/completions" \
      -H "Content-Type: application/json" \
      -d '{"model":"default","prompt":"Q: 1+1?\nA:","max_tokens":8,"temperature":0}' \
      > /dev/null 2>&1 || true
    sleep 5
    # Verify sentinel was consumed.
    if docker exec "$CONTAINER" test -f /tmp/.dump_region_timings 2>/dev/null; then
      echo "[$LABEL] WARN: sentinel still present (drain may have missed)" | tee -a "$CAPTURE_LOG"
    fi
    # Copy npy files out — these are the committed B' evidence.
    docker cp "$CONTAINER":/root/.cache/vllm/region_timings.npy \
      "$OUT_DIR/${LABEL}_region_timings.npy" 2>&1 | tee -a "$CAPTURE_LOG" || \
      echo "[$LABEL] WARN: no region_timings.npy in container" | tee -a "$CAPTURE_LOG"
    docker cp "$CONTAINER":/root/.cache/vllm/host_launch_walls.npy \
      "$OUT_DIR/${LABEL}_host_launch_walls.npy" 2>&1 | tee -a "$CAPTURE_LOG" || \
      echo "[$LABEL] WARN: no host_launch_walls.npy in container" | tee -a "$CAPTURE_LOG"
  fi

  # Stop watchdog before teardown.
  kill "$WATCHDOG_PID" 2>/dev/null || true
  trap - RETURN

  # Collect serve log + teardown.
  docker logs "$CONTAINER" > "$SERVE_LOG" 2>&1
  docker stop "$CONTAINER" >/dev/null
  docker rm "$CONTAINER" >/dev/null

  # Rename torch profiler trace to canonical name.
  local LATEST
  LATEST=$(ls -t "$OUT_DIR"/rank*.pt.trace.json.gz 2>/dev/null | head -n1 || true)
  if [ -z "$LATEST" ]; then
    echo "[$LABEL] ERROR: no fresh trace in $OUT_DIR (CUPTI flush failed?)" | tee -a "$CAPTURE_LOG"
    ls -la "$OUT_DIR" | tee -a "$CAPTURE_LOG"
    return 1
  fi
  mv "$LATEST" "$TRACE_TARGET"
  local BYTES
  BYTES=$(stat -c%s "$TRACE_TARGET")
  echo "[$LABEL] saved: $(basename "$TRACE_TARGET") ($BYTES bytes)" | tee -a "$CAPTURE_LOG"
}

for LEG in "${LEGS[@]}"; do
  if [ "$LEG" = "timing_on" ]; then
    run_leg "$LEG" "1"
  else
    run_leg "$LEG" "0"
  fi
done

echo "" | tee -a "$CAPTURE_LOG"
echo "==============================================================" | tee -a "$CAPTURE_LOG"
echo "=== Capture complete ===" | tee -a "$CAPTURE_LOG"
echo "==============================================================" | tee -a "$CAPTURE_LOG"
ls -la "$OUT_DIR" | tee -a "$CAPTURE_LOG"
echo "[capture] finished $(date -Iseconds)" | tee -a "$CAPTURE_LOG"
