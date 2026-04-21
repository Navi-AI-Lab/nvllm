#!/bin/bash
# Phase A trace capture — Stream-K + CUDA graphs (A.3) or pre-Stream-K baseline (A.4).
#
# Both modes produce the same server config (ig1/Qwen3.5-27B-NVFP4 + triton_attn
# + PIECEWISE CUDA graphs); only the container name and output filename differ.
# A.4 is expected to run this against an older image where the FP4 GEMM
# dispatcher used Stream-K's predecessor (split-K / baseline CUTLASS).
#
# Usage:
#   ./capture_phase_a_trace.sh streamk      # A.3 — current production code path
#   ./capture_phase_a_trace.sh baseline     # A.4 — pre-Stream-K comparison leg
set -euo pipefail

MODE="${1:?must pass 'streamk' or 'baseline'}"
case "$MODE" in
  streamk|baseline) ;;
  *) echo "MODE must be streamk|baseline" >&2; exit 1 ;;
esac

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
OUT_DIR="$REPO_ROOT/benchmarks/nvllm/traces/gemm_stream_k_cudagraph/2026-04-21"
mkdir -p "$OUT_DIR"

CONTAINER="nvllm-gemm-a3-$MODE"
IMAGE="${NVLLM_IMAGE:-nvllm:gb10}"
PORT=8000

# Clean up any prior containers that would hold the port or the name
docker rm -f nvllm nvllm-gemm-a3-streamk nvllm-gemm-a3-baseline 2>/dev/null || true

echo "=== Phase A trace capture: $MODE ==="
echo "  Image:     $IMAGE"
echo "  Container: $CONTAINER (port $PORT)"
echo "  Output:    $OUT_DIR"

# Bounded profiler window — /start_profile kicks it off, then after
# active_iterations model steps the profiler auto-finalizes and flushes.
# This avoids /stop_profile hanging on a huge trace serialization.
# ~600 iters ≈ 2 batches of decode at concurrency 4, enough for stable NVFP4
# GEMM kernel-duration stats across M positions within a batch.
PROFILER_CONFIG='{"profiler":"torch","torch_profiler_dir":"/tmp/profiles","ignore_frontend":true,"delay_iterations":0,"active_iterations":600,"torch_profiler_with_stack":false,"torch_profiler_use_gzip":true}'

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
  "$IMAGE" \
  serve \
  --model ig1/Qwen3.5-27B-NVFP4 \
  --served-model-name default \
  --host 0.0.0.0 --port "$PORT" \
  --kv-cache-dtype auto \
  --attention-backend triton_attn \
  --max-model-len 65536 \
  --max-num-seqs 4 \
  --language-model-only \
  --mamba-cache-mode align \
  --trust-remote-code \
  --gpu-memory-utilization 0.80 \
  --max-num-batched-tokens 65536 \
  --compilation-config '{"cudagraph_mode":"PIECEWISE"}' \
  --profiler-config "$PROFILER_CONFIG"

echo "Container up. Waiting for readiness (up to 20 min)..."

for i in $(seq 1 240); do
  if curl -sf http://localhost:$PORT/v1/models >/dev/null 2>&1; then
    echo "  Server ready at iter=$i (~$((i * 5)) s)."
    break
  fi
  sleep 5
done

if ! curl -sf http://localhost:$PORT/v1/models >/dev/null 2>&1; then
  echo "ERROR: server never became ready — dumping last 200 log lines." >&2
  docker logs --tail 200 "$CONTAINER" >&2
  docker logs "$CONTAINER" > "$OUT_DIR/decode_${MODE}.txt" 2>&1 || true
  docker rm -f "$CONTAINER" || true
  exit 1
fi

echo "=== Running workload (50 warmup + 100 timed @ concurrency 4) ==="
cd "$REPO_ROOT"
.venv/bin/python docs/research/gemm_sweep/trace_workload.py \
    --base-url "http://localhost:$PORT/v1" \
    --model default \
    --warmup 30 --timed 30 --concurrent 4 \
    --max-tokens 256 \
    --profile-start "http://localhost:$PORT/start_profile" \
    --profile-stop  "http://localhost:$PORT/stop_profile"

# Let profiler flush CUPTI buffers and finalize the .pt.trace.json.gz file.
# torch profiler + CUPTI serialization can take 30-90s after the profile
# window closes; truncated traces lose ALL GPU kernel events (cpu_op survives
# the truncation, but cat=kernel does not). Wait until the file size is
# stable for 3 consecutive 5s checks.
# Profiler flush — torch+CUPTI serialization can take 30-90s after the
# active_iterations window closes. Dumb-but-reliable 120s wait (previous
# fancy size-stability loop silently exited on first iter under pipefail —
# Bash subshell gotcha, not worth re-debugging). If a future iteration needs
# dynamic stop, re-add the stability logic but set +e inside the loop body.
echo "Flushing profiler — sleeping 120s for CUPTI finalize..."
for i in 1 2 3 4; do
  sleep 30
  echo "  [flush +${i}0s]"
  ls -la "$OUT_DIR"/rank*.pt.trace.json.gz 2>/dev/null | awk '{print "    current size:", $5}' || true
done

# Collect logs
echo "Collecting logs..."
docker logs "$CONTAINER" > "$OUT_DIR/decode_${MODE}.txt" 2>&1

# Teardown
echo "Stopping container..."
docker stop "$CONTAINER" >/dev/null
docker rm "$CONTAINER" >/dev/null

# Rename the newly-written profiler trace to our canonical name
# Profiler writes <hostname>_<pid>.<timestamp>.pt.trace.json.gz into /tmp/profiles
LATEST_TRACE="$(ls -t "$OUT_DIR"/*.pt.trace.json.gz 2>/dev/null | grep -v -E '(streamk|baseline)_graphs\.pt\.trace\.json\.gz$' | head -n1 || true)"
TARGET="$OUT_DIR/${MODE}_graphs.pt.trace.json.gz"

if [ -z "$LATEST_TRACE" ]; then
  # Maybe it was already renamed from a prior run — check for the target itself
  if [ -f "$TARGET" ]; then
    echo "WARN: no fresh trace artifact; $TARGET exists from previous run. Leaving as-is."
  else
    echo "ERROR: no profiler trace artifact found in $OUT_DIR" >&2
    ls -la "$OUT_DIR" >&2
    exit 1
  fi
else
  mv "$LATEST_TRACE" "$TARGET"
  echo "Renamed $(basename "$LATEST_TRACE") -> $(basename "$TARGET")"
fi

echo ""
echo "=== Trace artifacts in $OUT_DIR ==="
ls -la "$OUT_DIR"
