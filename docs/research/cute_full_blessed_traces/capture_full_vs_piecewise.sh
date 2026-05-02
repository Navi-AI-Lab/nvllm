#!/bin/bash
# Bench harness: FULL+blessed vs PIECEWISE on matching config (lower-8 + n=1).
#
# Per leg, captures three artifacts:
#   1. <leg>.pt.trace.json.gz  — torch profiler kernel trace (gitignored, raw)
#   2. <leg>_kernels.csv        — per-kernel μs from torch profiler (committed)
#   3. <leg>_streaming.json     — single-request TTFT + decode tok/s (committed)
#   4. <leg>_serve.log          — container log (committed)
#   5. <leg>.nsys-rep           — minimal system-wide nsys for AGENTS.md §4 (committed)
#
# Usage:  bash docs/research/cute_full_blessed_traces/capture_full_vs_piecewise.sh
# Wall clock: ~1.5-2 hr on GB10.
#
# Matched config (only cudagraph_mode + bless mount differ):
#   model=ig1/Qwen3.5-27B-NVFP4, kv-cache=fp8_e4m3, attn=CUTE_PAGED,
#   max-model-len=16384, max-num-seqs=1, max-num-batched-tokens=65536,
#   gpu-memory-utilization=0.65, CUTE_PHASE_E_FUSION=1,
#   CUTE_PHASE_E_LAYERS=0,1,2,3,4,5,6,7, MLP_FUSION=1, ATTN_FUSION=1,
#   cudagraph_capture_sizes=[1].
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
source "$REPO_ROOT/scripts/common.sh"

OUT_DIR="${OUT_DIR:-$REPO_ROOT/benchmarks/nvllm/traces/cute_full_blessed/2026-05-01-vs-piecewise}"
mkdir -p "$OUT_DIR"

IMAGE="${NVLLM_IMAGE:-nvllm:gb10}"
HF_MODEL="${HF_MODEL:-ig1/Qwen3.5-27B-NVFP4}"
CONTAINER="nvllm"
TRACE_CONTAINER="nvllm-trace"
PORT=8000
BASE_URL="http://localhost:${PORT}"

# Common config — same on both legs
KV_CACHE="fp8_e4m3"
ATTN_BACKEND="CUTE_PAGED"
MAX_MODEL_LEN=16384
MAX_NUM_SEQS=1
MAX_NUM_BATCHED_TOKENS=65536
GPU_MEM_UTIL=0.65
PHASE_E_LAYERS="0,1,2,3,4,5,6,7"

# CuTe disk cache — bind-mount on both legs so cute.compile is fast
CUTE_COMPILE_HOST_CACHE_DIR="${CUTE_COMPILE_HOST_CACHE_DIR:-/tmp/nvllm-cute-cache}"
mkdir -p "$CUTE_COMPILE_HOST_CACHE_DIR"

# Profiler config: bounded by max_iterations (active_iterations alone is dead
# code without wait/warmup_iterations — wrapper.py:104-116 gates max_iters
# independently of schedule mode). 200 worker steps caps Kineto buffer
# regardless of wall workload.
PROFILER_MAX_ITERATIONS="${PROFILER_MAX_ITERATIONS:-200}"
PROFILER_CONFIG="{\"profiler\":\"torch\",\"torch_profiler_dir\":\"/tmp/profiles\",\"ignore_frontend\":true,\"delay_iterations\":0,\"max_iterations\":${PROFILER_MAX_ITERATIONS},\"torch_profiler_with_stack\":false,\"torch_profiler_use_gzip\":true,\"torch_profiler_record_shapes\":false,\"torch_profiler_with_memory\":false}"

# Nsys host install — bind-mount the host nsys into trace container
NSYS_HOST_VERSION="${NSYS_HOST_VERSION:-2025.6.3}"
NSYS_HOST_PATH="/opt/nvidia/nsight-systems/${NSYS_HOST_VERSION}"
if [ ! -d "$NSYS_HOST_PATH" ]; then
  echo "ERROR: nsys not found at $NSYS_HOST_PATH" >&2
  echo "       Set NSYS_HOST_VERSION to the available version under /opt/nvidia/nsight-systems/" >&2
  exit 1
fi

# Workload params (matched both legs)
WARMUP_N=5
SMOKE_N=2
TIMED_N=30
MAX_TOKENS=256
WORKLOAD_TIMEOUT=600
PROFILE_STOP_TIMEOUT="${PROFILE_STOP_TIMEOUT:-1800}"

# Pre-flight
nvllm_check_image
nvllm_check_free_mem "${NVLLM_MIN_FREE_GB:-90}"

GIT_SHA=$(cd "$REPO_ROOT" && git rev-parse HEAD)
GIT_BRANCH=$(cd "$REPO_ROOT" && git rev-parse --abbrev-ref HEAD)
IMAGE_ID=$(docker image inspect "$IMAGE" --format '{{.Id}}')
HF_REVISION=$(nvllm_resolve_hf_revision "$HF_MODEL")

echo "=== Bench harness: FULL+blessed vs PIECEWISE ==="
echo "  Output dir:    $OUT_DIR"
echo "  Image:         $IMAGE ($IMAGE_ID)"
echo "  Model:         $HF_MODEL @ $HF_REVISION"
echo "  Git:           $GIT_BRANCH @ $GIT_SHA"
echo "  CuTe cache:    $CUTE_COMPILE_HOST_CACHE_DIR"
echo "  nsys:          $NSYS_HOST_PATH"
echo ""

write_meta() {
  local LABEL="$1" CGMODE="$2" BLESS_PATH="$3"
  cat > "$OUT_DIR/${LABEL}_meta.json" <<EOF
{
  "label": "$LABEL",
  "git_sha": "$GIT_SHA",
  "git_branch": "$GIT_BRANCH",
  "image": "$IMAGE",
  "image_id": "$IMAGE_ID",
  "hf_model": "$HF_MODEL",
  "hf_revision": "$HF_REVISION",
  "cudagraph_mode": "$CGMODE",
  "blessed_cache_path": "$BLESS_PATH",
  "kv_cache_dtype": "$KV_CACHE",
  "attn_backend": "$ATTN_BACKEND",
  "max_model_len": $MAX_MODEL_LEN,
  "max_num_seqs": $MAX_NUM_SEQS,
  "max_num_batched_tokens": $MAX_NUM_BATCHED_TOKENS,
  "gpu_memory_utilization": $GPU_MEM_UTIL,
  "cute_phase_e_layers": "$PHASE_E_LAYERS",
  "cute_phase_e_fusion": 1,
  "cute_mlp_fusion": 1,
  "cute_attn_fusion": 1,
  "warmup_n": $WARMUP_N,
  "smoke_n": $SMOKE_N,
  "timed_n": $TIMED_N,
  "max_tokens": $MAX_TOKENS,
  "concurrent": 1,
  "seed": 42,
  "temperature": 0.0,
  "ignore_eos": true,
  "torch_profiler_max_iterations": $PROFILER_MAX_ITERATIONS,
  "torch_profile_scope": "first bounded worker/model iterations only; full workload still runs 30x256 at concurrency=1",
  "prompt_source": "docs/research/gemm_sweep/trace_workload.py FIXED_PROMPT"
}
EOF
}

# Resolve blessed manifest (FULL leg only)
BLESSED_HOST_PATH=""
BLESSED_AOT_SHA=""
BLESSED_MANIFEST_PATH=""
resolve_bless() {
  local CONFIG_HASH
  CONFIG_HASH=$(nvllm_compute_blessed_config_hash \
    "$IMAGE_ID" "$HF_MODEL" "$HF_REVISION" \
    "$KV_CACHE" "$ATTN_BACKEND" "FULL_AND_PIECEWISE" "[1]" \
    "$MAX_NUM_SEQS" "$MAX_MODEL_LEN" "$MAX_NUM_BATCHED_TOKENS" \
    "1" "$PHASE_E_LAYERS" "1" \
    "0" "0" "0" "1" "1")
  echo "[bless] config_hash=$CONFIG_HASH"
  BLESSED_MANIFEST_PATH=$(nvllm_resolve_blessed_manifest "$CONFIG_HASH")
  echo "[bless] manifest=$BLESSED_MANIFEST_PATH"
  if ! nvllm_verify_blessed_cache "$BLESSED_MANIFEST_PATH"; then
    echo "ERROR: blessed cache drift" >&2
    exit 1
  fi
  BLESSED_HOST_PATH=$(jq -r '.mount.host_path' "$BLESSED_MANIFEST_PATH")
  BLESSED_HOST_PATH="${BLESSED_HOST_PATH/#\~/$HOME}"
  BLESSED_AOT_SHA=$(jq -r '.files[] | select(.relative_path | contains("torch_aot_compile")) | .sha256' "$BLESSED_MANIFEST_PATH")
  echo "[bless] host_path=$BLESSED_HOST_PATH"
  echo "[bless] aot_sha=$BLESSED_AOT_SHA"
}

wait_models_ready() {
  local DEADLINE=$((SECONDS + 600))
  while [ "$SECONDS" -lt "$DEADLINE" ]; do
    if curl -sf "$BASE_URL/v1/models" > /dev/null 2>&1; then
      echo "[ready] /v1/models responding at +${SECONDS}s"
      return 0
    fi
    sleep 5
  done
  echo "ERROR: /v1/models did not come up within 600s" >&2
  return 1
}

fire_warmup() {
  local N="$1"
  for i in $(seq 1 "$N"); do
    curl -sf "$BASE_URL/v1/completions" \
      -H "Content-Type: application/json" \
      -d "{\"model\":\"default\",\"prompt\":\"Q: $i + $i =\",\"max_tokens\":16,\"temperature\":0.0,\"seed\":42}" \
      > /dev/null
  done
}

WATCHDOG_PID=""
start_mem_watchdog() {
  local LABEL="$1"
  rm -f "$OUT_DIR/${LABEL}_mem.log"
  (
    while :; do
      date '+%H:%M:%S'
      free -h
      docker stats --no-stream --format 'table {{.Name}}\t{{.MemUsage}}' \
        "$CONTAINER" 2>/dev/null || true
      echo "---"
      sleep 30
    done
  ) > "$OUT_DIR/${LABEL}_mem.log" 2>&1 &
  WATCHDOG_PID=$!
  echo "[$LABEL] watchdog pid=$WATCHDOG_PID -> ${LABEL}_mem.log"
}

stop_mem_watchdog() {
  if [ -n "${WATCHDOG_PID:-}" ]; then
    kill "$WATCHDOG_PID" 2>/dev/null || true
    wait "$WATCHDOG_PID" 2>/dev/null || true
    WATCHDOG_PID=""
  fi
}

trap 'stop_mem_watchdog' EXIT

# Build common docker run args (same for profiler leg AND nsys leg)
build_common_run() {
  local LABEL="$1" CGMODE="$2" BLESS_MOUNT_FLAGS="$3" PROFILER_CONFIG_FLAG="$4"
  COMMON_RUN=(
    docker run -d
    --name "$CONTAINER"
    --gpus all --ipc=host --network host --privileged
    -v "$HOME/.cache/huggingface:/root/.cache/huggingface"
    -v "$HOME/.cache/flashinfer:/root/.cache/flashinfer"
    -v "$CUTE_COMPILE_HOST_CACHE_DIR:/opt/vllm/kernel_cache"
    -v "$OUT_DIR:/tmp/profiles"
  )
  # FULL leg only: blessed cache mount (already includes :ro)
  if [ -n "$BLESS_MOUNT_FLAGS" ]; then
    # shellcheck disable=SC2206
    COMMON_RUN+=( $BLESS_MOUNT_FLAGS )
  fi
  COMMON_RUN+=(
    -e B12X_CUTE_COMPILE_DISK_CACHE=1
    -e B12X_CUTE_COMPILE_CACHE_DIR=/opt/vllm/kernel_cache
    -e VLLM_NVFP4_GEMM_BACKEND=cutlass
    -e VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
    -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    -e CUTE_MLP_FUSION=1
    -e CUTE_ATTN_FUSION=1
    -e CUTE_BETA_MIN_FREE_GB=8
    -e CUTE_PHASE_E_FUSION=1
    -e CUTE_PHASE_E_LAYERS="$PHASE_E_LAYERS"
    -e CUTE_PHASE_E_FALLBACK_RAISE=1
    -e CUTE_FULL_GRAPH_PROBE=0
    -e CUTE_WO_RESET_LOG=0
    -e CUTE_DISPATCH_AUDIT=0
    "$IMAGE"
    serve
    --model "$HF_MODEL"
    --served-model-name default
    --host 0.0.0.0 --port "$PORT"
    --kv-cache-dtype "$KV_CACHE"
    --attention-backend "$ATTN_BACKEND"
    --max-model-len "$MAX_MODEL_LEN"
    --max-num-seqs "$MAX_NUM_SEQS"
    --language-model-only
    --limit-mm-per-prompt '{"image": 0, "video": 0}'
    --mamba-cache-mode align
    --trust-remote-code
    --gpu-memory-utilization "$GPU_MEM_UTIL"
    --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS"
    --kernel-config '{"enable_flashinfer_autotune":false}'
    --compilation-config "{\"cudagraph_mode\":\"$CGMODE\",\"cudagraph_capture_sizes\":[1]}"
  )
  if [ -n "$PROFILER_CONFIG_FLAG" ]; then
    COMMON_RUN+=( --profiler-config "$PROFILER_CONFIG" )
  fi
}

# Phase 1 per leg: torch profiler workload (per-kernel CSV)
run_torch_profiler_leg() {
  local LABEL="$1" CGMODE="$2" BLESS_MOUNT_FLAGS="$3"

  echo ""
  echo "=============================================================="
  echo "=== [$LABEL] Phase 1: torch profiler workload"
  echo "=== cudagraph_mode=$CGMODE bless_mount=${BLESS_MOUNT_FLAGS:-(none)}"
  echo "=============================================================="

  nvllm_cleanup_container "$CONTAINER"
  rm -f \
    "$OUT_DIR"/*rank0*.pt.trace.json.gz \
    "$OUT_DIR"/*rank0*.pt.trace.json \
    "$OUT_DIR/${LABEL}.pt.trace.json.gz" \
    "$OUT_DIR/${LABEL}.pt.trace.json" \
    "$OUT_DIR/profiler_out_0.txt" \
    "$OUT_DIR/${LABEL}_profiler_out_0.txt"
  build_common_run "$LABEL" "$CGMODE" "$BLESS_MOUNT_FLAGS" "yes"
  "${COMMON_RUN[@]}"

  wait_models_ready
  # Memory watchdog (skill OOM avoidance) — every 30s while the leg runs,
  # so if Kineto flush spikes the host we have a trajectory to reconstruct from.
  start_mem_watchdog "$LABEL"

  echo "[$LABEL] warmup ($WARMUP_N requests)…"
  fire_warmup "$WARMUP_N"
  echo "[$LABEL] smoke ($SMOKE_N requests, post-warmup steady state)…"
  fire_warmup "$SMOKE_N"
  echo "[$LABEL] streaming TTFT measurement…"
  "$REPO_ROOT/.venv/bin/python" "$REPO_ROOT/docs/research/cute_full_blessed_traces/streaming_ttft.py" \
    --base-url "$BASE_URL/v1" --model default \
    --max-tokens "$MAX_TOKENS" --seed 42 --timeout "$WORKLOAD_TIMEOUT" \
    --label "$LABEL" \
    --out "$OUT_DIR/${LABEL}_streaming.json"

  echo "[$LABEL] profiled workload (warmup=0 timed=$TIMED_N concurrent=1 max-tokens=$MAX_TOKENS)…"
  "$REPO_ROOT/.venv/bin/python" "$REPO_ROOT/docs/research/gemm_sweep/trace_workload.py" \
    --base-url "$BASE_URL/v1" --model default \
    --warmup 0 --timed "$TIMED_N" --concurrent 1 \
    --max-tokens "$MAX_TOKENS" --timeout "$WORKLOAD_TIMEOUT" \
    --profile-start "$BASE_URL/start_profile" \
    | tee "$OUT_DIR/${LABEL}_profiled_workload.log"

  echo "[$LABEL] stopping torch profiler (timeout=${PROFILE_STOP_TIMEOUT}s)…"
  if ! curl -sS -X POST --max-time "$PROFILE_STOP_TIMEOUT" \
      "$BASE_URL/stop_profile" > "$OUT_DIR/${LABEL}_stop_profile.log" 2>&1; then
    docker logs "$CONTAINER" > "$OUT_DIR/${LABEL}_serve.log" 2>&1 || true
    stop_mem_watchdog
    echo "ERROR: /stop_profile did not complete for $LABEL; leaving $CONTAINER running for inspection" >&2
    return 1
  fi

  docker logs "$CONTAINER" > "$OUT_DIR/${LABEL}_serve.log" 2>&1
  docker stop "$CONTAINER" >/dev/null 2>&1 || true
  docker rm "$CONTAINER" >/dev/null 2>&1 || true
  stop_mem_watchdog

  # Move the rank0.* trace to <leg>.pt.trace.json.gz
  local TRACE_SRC
  TRACE_SRC=$(ls -t "$OUT_DIR"/*rank0*.pt.trace.json.gz "$OUT_DIR"/rank0.*.pt.trace.json.gz 2>/dev/null | head -1 || true)
  if [ -z "$TRACE_SRC" ]; then
    echo "ERROR: no *rank0*.pt.trace.json.gz found in $OUT_DIR" >&2
    return 1
  fi
  mv "$TRACE_SRC" "$OUT_DIR/${LABEL}.pt.trace.json.gz"
  if [ -f "$OUT_DIR/profiler_out_0.txt" ]; then
    mv "$OUT_DIR/profiler_out_0.txt" "$OUT_DIR/${LABEL}_profiler_out_0.txt"
  fi
  echo "[$LABEL] trace: $OUT_DIR/${LABEL}.pt.trace.json.gz"
}

# Phase 2 per leg: minimal nsys system-wide capture for AGENTS.md §4
run_nsys_leg() {
  local LABEL="$1" CGMODE="$2" BLESS_MOUNT_FLAGS="$3"

  echo ""
  echo "=============================================================="
  echo "=== [$LABEL] Phase 2: nsys sidecar (system-wide, 90s window)"
  echo "=============================================================="

  nvllm_cleanup_container "$CONTAINER"
  docker rm -f "$TRACE_CONTAINER" 2>/dev/null || true

  build_common_run "$LABEL" "$CGMODE" "$BLESS_MOUNT_FLAGS" ""
  "${COMMON_RUN[@]}"
  wait_models_ready
  fire_warmup 3

  # Sidecar trace container with host nsys volume-mounted
  echo "[$LABEL] starting nsys sidecar…"
  docker run -d \
    --name "$TRACE_CONTAINER" \
    --gpus all --ipc=host --network host --privileged --pid host \
    -v "$NSYS_HOST_PATH:/opt/nsight-systems:ro" \
    -v "$OUT_DIR:/traces" \
    --entrypoint bash \
    "$IMAGE" \
    -c "/opt/nsight-systems/bin/nsys profile \
          --trace=cuda,nvtx,cublas \
          --cuda-trace-scope=system-wide \
          --cuda-graph-trace=node \
          --sample=none --cpuctxsw=none \
          --force-overwrite=true \
          --delay=20 --duration=60 \
          -o /traces/${LABEL} \
          sleep 90"

  # Drive load into the GPU during the nsys window (delay=20 + duration=60 = 80s)
  echo "[$LABEL] driving load (single 256-token request to overlap nsys window)…"
  sleep 25
  curl -sf "$BASE_URL/v1/completions" \
    -H "Content-Type: application/json" \
    -d "{\"model\":\"default\",\"prompt\":\"Q: Janet has 3 apples and buys 5 more. How many apples does she have now? Think step by step.\\nA:\",\"max_tokens\":256,\"temperature\":0.0,\"seed\":42,\"ignore_eos\":true}" \
    > "$OUT_DIR/${LABEL}_nsys_request.json" || true

  echo "[$LABEL] waiting on nsys sidecar…"
  docker wait "$TRACE_CONTAINER" >/dev/null
  docker logs "$TRACE_CONTAINER" >> "$OUT_DIR/${LABEL}_serve.log" 2>&1
  docker rm "$TRACE_CONTAINER" >/dev/null

  docker stop "$CONTAINER" >/dev/null 2>&1 || true
  docker rm "$CONTAINER" >/dev/null 2>&1 || true

  if [ ! -f "$OUT_DIR/${LABEL}.nsys-rep" ]; then
    echo "WARNING: nsys-rep not produced for $LABEL" >&2
  else
    echo "[$LABEL] nsys: $OUT_DIR/${LABEL}.nsys-rep"
  fi
}

extract_kernels_csv() {
  local LABEL="$1"
  echo ""
  echo "[$LABEL] extracting per-kernel CSV…"
  "$REPO_ROOT/.venv/bin/python" "$REPO_ROOT/docs/research/gemm_sweep/extract_e2e_kernels.py" \
    --trace "$OUT_DIR/${LABEL}.pt.trace.json.gz" \
    --config "$LABEL" \
    --out "$OUT_DIR/${LABEL}_kernels.csv"
  echo "[$LABEL] csv: $OUT_DIR/${LABEL}_kernels.csv"
}

###################################
# Leg 1: PIECEWISE (no bless mount)
###################################
write_meta "piecewise" "PIECEWISE" ""
run_torch_profiler_leg "piecewise" "PIECEWISE" ""
run_nsys_leg "piecewise" "PIECEWISE" ""
extract_kernels_csv "piecewise"

###################################
# Leg 2: FULL+blessed
###################################
resolve_bless
BLESS_MOUNT_FLAGS="-v ${BLESSED_HOST_PATH}:/root/.cache/vllm:ro"
write_meta "full" "FULL_AND_PIECEWISE" "$BLESSED_HOST_PATH"
# Append blessed manifest detail to full_meta.json
jq --arg sha "$BLESSED_AOT_SHA" --arg manifest "$BLESSED_MANIFEST_PATH" \
   '. + {blessed_aot_sha: $sha, blessed_manifest: $manifest}' \
   "$OUT_DIR/full_meta.json" > "$OUT_DIR/full_meta.json.tmp" && \
   mv "$OUT_DIR/full_meta.json.tmp" "$OUT_DIR/full_meta.json"
run_torch_profiler_leg "full" "FULL_AND_PIECEWISE" "$BLESS_MOUNT_FLAGS"
run_nsys_leg "full" "FULL_AND_PIECEWISE" "$BLESS_MOUNT_FLAGS"
extract_kernels_csv "full"

echo ""
echo "=============================================================="
echo "=== Bench complete. Outputs in $OUT_DIR"
echo "=============================================================="
ls -la "$OUT_DIR"
