#!/bin/bash
# Phase-E-tax screening — 3 legs × 2 phases = 6 boots.
#
# Hypothesis: β-coop is currently a tax on lower8 FULL-graph decode
# because layers 3 and 7 are the only β-coop layers in production and
# (per FULL trace) cost ~40.8 ms/layer-token vs ~17.1 ms/layer-token for
# legacy DecodeKernel + external GEMM.
#
# Decision-maker leg: phaseE-off. If it wins ≥5% on per-call kernel
# μs and clears the diagnostic GSM8K floor (no regression vs prior phase
# ~30/50, NOT the ship-gate ≥47/50), the next move is FP4/GEMV
# K-parallel reduction work, not all-β.
#
# Per-leg phases:
#   profile-boot:  FULL+PIECEWISE, --no-blessed-verify (cold capture this
#                  session), --privileged for CUPTI, /tmp/profiles bind
#                  mount, vLLM torch profiler via /start_profile +
#                  /stop_profile + 120s flush.
#   gsm8k-boot:    PIECEWISE only (deterministic), no profiler, normal
#                  /v1/completions traffic via gsm8k_eval_50.py.
#
# Output:
#   benchmarks/nvllm/traces/cute_paged_attn/2026-05-02-phaseE-tax-3leg/
#     {lower8,phaseE-off,all-beta}/
#       profile_serve.log
#       profile.pt.trace.json.gz   (gitignored)
#       profile_kernels.csv
#       profile_DONE
#       gsm8k_serve.log
#       gsm8k.json
#       gsm8k_DONE
#       metadata.json              (one per phase: profile / gsm8k)
#     mem_watchdog.log
#     summary.md                   (written after run, not by this script)
#
# Resume: re-running this script skips legs/phases that already have a
# *_DONE marker. To force re-run: delete the marker(s).
#
# Usage:
#   tmux new -s bench-3leg
#   bash docs/research/2026-05-02-phaseE-tax-3leg/run_3leg.sh
#   <Ctrl-b> d to detach
#   tmux attach -t bench-3leg to resume
#
# Skip a leg via env:
#   SKIP_LOWER8=1 bash ...
#   SKIP_PHASEE_OFF=1 bash ...
#   SKIP_ALL_BETA=1 bash ...
#
# Override image / model:
#   NVLLM_IMAGE=nvllm:gb10 HF_MODEL=ig1/Qwen3.5-27B-NVFP4 bash ...

set -euo pipefail

# ---------------------------------------------------------------------------
# Pre-flight (validate inputs before set -e starts firing real work)
# ---------------------------------------------------------------------------
REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$REPO_ROOT"

# Validate venv (gsm8k_eval_50 + extract_e2e_kernels live in .venv)
if [ ! -x "$REPO_ROOT/.venv/bin/python" ]; then
  echo "ERROR: $REPO_ROOT/.venv/bin/python not found." >&2
  echo "       Run 'uv venv --python 3.12' per AGENTS.md." >&2
  exit 1
fi

# Validate jq (used by common.sh + metadata builder)
if ! command -v jq >/dev/null 2>&1; then
  echo "ERROR: jq not found. apt-get install jq." >&2
  exit 1
fi

# Pull common helpers AFTER venv check (some helpers shell out to .venv).
source "$REPO_ROOT/scripts/common.sh"

NVLLM_IMAGE="${NVLLM_IMAGE:-nvllm:gb10}"
HF_MODEL="${HF_MODEL:-ig1/Qwen3.5-27B-NVFP4}"
CONTAINER="nvllm"
PORT=8000

OUT_ROOT="$REPO_ROOT/benchmarks/nvllm/traces/cute_paged_attn/2026-05-02-phaseE-tax-3leg"
mkdir -p "$OUT_ROOT"

GIT_COMMIT="$(git -C "$REPO_ROOT" rev-parse --short HEAD)"
IMAGE_ID="$(docker image inspect "$NVLLM_IMAGE" --format '{{.Id}}' 2>/dev/null || true)"
if [ -z "$IMAGE_ID" ]; then
  echo "ERROR: image '$NVLLM_IMAGE' not found locally. Build it first." >&2
  exit 1
fi
HF_REVISION="$(nvllm_resolve_hf_revision "$HF_MODEL")" || {
  echo "ERROR: cannot resolve HF revision for $HF_MODEL" >&2; exit 1; }

# Production-matching config (must mirror serve-cute-full.sh fields that
# feed the blessed config_hash, so metadata's hash is comparable).
KV_CACHE_DTYPE="fp8_e4m3"
ATTN_BACKEND_VAL="CUTE_PAGED"
MAX_NUM_SEQS_VAL=1
MAX_MODEL_LEN_VAL=16384
MAX_NUM_BATCHED_TOKENS_VAL=65536
CUDAGRAPH_CAPTURE_SIZES="[1]"
CUTE_PHASE_E_FALLBACK_RAISE_VAL=1
CUTE_FULL_GRAPH_PROBE_VAL=0
CUTE_WO_RESET_LOG_VAL=0
CUTE_DISPATCH_AUDIT_VAL=0
CUTE_MLP_FUSION_VAL=1
CUTE_ATTN_FUSION_VAL=1
SERVE_GPU_UTIL=0.65

# Memory headroom check — abort early on tight memory rather than mid-leg.
nvllm_check_free_mem "${NVLLM_MIN_FREE_GB:-90}"

# Refuse if container exists from a stale run; operator must clean up.
if docker ps -a --filter "name=^${CONTAINER}$" --format '{{.Names}}' | grep -q .; then
  echo "ERROR: container '$CONTAINER' already exists. Remove it first:" >&2
  echo "         docker stop $CONTAINER && docker rm $CONTAINER" >&2
  exit 1
fi

# ---------------------------------------------------------------------------
# Sidecar memory watchdog (host-side bg process). Trapped on exit.
# ---------------------------------------------------------------------------
MEM_LOG="$OUT_ROOT/mem_watchdog.log"
echo "[watchdog] starting, log=$MEM_LOG"
(
  while true; do
    {
      date -Iseconds
      free -h | head -2
      docker stats --no-stream --format 'docker: {{.Name}} cpu={{.CPUPerc}} mem={{.MemUsage}}' "$CONTAINER" 2>/dev/null || true
      echo "---"
    }
    sleep 30
  done
) > "$MEM_LOG" 2>&1 &
WATCHDOG_PID=$!
trap 'echo "[watchdog] stopping (pid=$WATCHDOG_PID)"; kill $WATCHDOG_PID 2>/dev/null || true' EXIT

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# wait_for_ready CONTAINER PORT
#   Poll /v1/models until 200 or container dies. Returns 0 on ready, 1 else.
wait_for_ready() {
  local container="$1" port="$2"
  echo "  [ready] polling http://localhost:$port/v1/models (up to 30 min)..."
  local i
  for i in $(seq 1 360); do
    if ! docker ps --filter "name=^${container}$" --format '{{.Names}}' | grep -q "^${container}$"; then
      echo "  [ready] ERROR: container died while loading" >&2
      return 1
    fi
    if curl -sf "http://localhost:$port/v1/models" -o /dev/null 2>&1; then
      echo "  [ready] up at iter=$i (~$((i * 5))s)."
      return 0
    fi
    sleep 5
  done
  echo "  [ready] ERROR: server never became ready in 30 min" >&2
  return 1
}

# warmup_requests N MAX_TOKENS
#   Fire N sequential POST /v1/completions, ignore result. JIT primer.
#   Tolerant: a request timing out is logged but not fatal.
warmup_requests() {
  local n="$1" max_tokens="$2"
  echo "  [warmup] $n requests at max_tokens=$max_tokens (tolerant)..."
  local i
  for i in $(seq 1 "$n"); do
    if ! curl -s --max-time 600 "http://localhost:$PORT/v1/completions" \
        -H "Content-Type: application/json" \
        -d "{\"model\":\"default\",\"prompt\":\"Q: Janet has 3 apples and buys 5 more.\nA:\",\"max_tokens\":$max_tokens,\"temperature\":0,\"ignore_eos\":true}" \
        > /dev/null 2>&1; then
      echo "    [warmup $i] timeout/err (ok during JIT)"
    fi
  done
}

# write_metadata LEG PHASE OUT_DIR EXTRA_JSON
#   Build metadata.json for a leg×phase. EXTRA_JSON is a jq object literal
#   (or empty) merged into the env block.
write_metadata() {
  local leg="$1" phase="$2" out_dir="$3" cudagraph_mode="$4"
  local config_hash="$5"
  local fusion="$6" layers="$7"
  local timestamp
  timestamp="$(date -Iseconds)"
  local launcher_command="docs/research/2026-05-02-phaseE-tax-3leg/run_3leg.sh (inline docker run)"

  jq -n \
    --arg leg "$leg" \
    --arg phase "$phase" \
    --arg ts "$timestamp" \
    --arg git "$GIT_COMMIT" \
    --arg image "$IMAGE_ID" \
    --arg model "$HF_MODEL" \
    --arg model_rev "$HF_REVISION" \
    --arg cgmode "$cudagraph_mode" \
    --arg cghash "$config_hash" \
    --argjson cgsizes "$CUDAGRAPH_CAPTURE_SIZES" \
    --argjson nseqs "$MAX_NUM_SEQS_VAL" \
    --argjson nlen "$MAX_MODEL_LEN_VAL" \
    --argjson nbatch "$MAX_NUM_BATCHED_TOKENS_VAL" \
    --arg kvdtype "$KV_CACHE_DTYPE" \
    --arg attn "$ATTN_BACKEND_VAL" \
    --arg gpu_util "$SERVE_GPU_UTIL" \
    --arg fusion "$fusion" \
    --arg layers "$layers" \
    --arg launcher "$launcher_command" \
    '{
      leg: $leg,
      phase: $phase,
      timestamp_iso: $ts,
      git_commit: $git,
      image_id: $image,
      model: $model,
      model_revision: $model_rev,
      cudagraph_mode: $cgmode,
      cudagraph_capture_sizes: $cgsizes,
      max_num_seqs: $nseqs,
      max_model_len: $nlen,
      max_num_batched_tokens: $nbatch,
      kv_cache_dtype: $kvdtype,
      attention_backend: $attn,
      gpu_memory_utilization: ($gpu_util | tonumber),
      bless_mounted: false,
      manifest_enforced: false,
      config_hash: $cghash,
      env: {
        CUTE_PHASE_E_FUSION: $fusion,
        CUTE_PHASE_E_LAYERS: $layers,
        CUTE_PHASE_E_FALLBACK_RAISE: "1",
        CUTE_FULL_GRAPH_PROBE: "0",
        CUTE_WO_RESET_LOG: "0",
        CUTE_DISPATCH_AUDIT: "0",
        CUTE_MLP_FUSION: "1",
        CUTE_ATTN_FUSION: "1",
        VLLM_NVFP4_GEMM_BACKEND: "cutlass",
        B12X_CUTE_COMPILE_DISK_CACHE: "1"
      },
      launcher_command: $launcher
    }' > "$out_dir/${phase}_metadata.json"
}

# compute_config_hash FUSION LAYERS CUDAGRAPH_MODE
compute_config_hash() {
  local fusion="$1" layers="$2" cgmode="$3"
  nvllm_compute_blessed_config_hash \
    "$IMAGE_ID" "$HF_MODEL" "$HF_REVISION" \
    "$KV_CACHE_DTYPE" "$ATTN_BACKEND_VAL" "$cgmode" \
    "$CUDAGRAPH_CAPTURE_SIZES" \
    "$MAX_NUM_SEQS_VAL" "$MAX_MODEL_LEN_VAL" "$MAX_NUM_BATCHED_TOKENS_VAL" \
    "$fusion" "$layers" \
    "$CUTE_PHASE_E_FALLBACK_RAISE_VAL" \
    "$CUTE_FULL_GRAPH_PROBE_VAL" "$CUTE_WO_RESET_LOG_VAL" \
    "$CUTE_DISPATCH_AUDIT_VAL" \
    "$CUTE_MLP_FUSION_VAL" "$CUTE_ATTN_FUSION_VAL"
}

# ---------------------------------------------------------------------------
# run_profile_phase LEG FUSION LAYERS WARMUP_N TIMED_N
#   Boot 1: FULL+PIECEWISE cold, profiler enabled, capture trace + serve.log.
# ---------------------------------------------------------------------------
run_profile_phase() {
  local leg="$1" fusion="$2" layers="$3" warmup_n="$4" timed_n="$5"
  local out_dir="$OUT_ROOT/$leg"
  mkdir -p "$out_dir"

  if [ -f "$out_dir/profile_DONE" ]; then
    echo ""
    echo "=== [$leg / profile] SKIP (DONE marker exists)"
    return 0
  fi

  local cudagraph_mode="FULL_AND_PIECEWISE"
  local config_hash
  config_hash="$(compute_config_hash "$fusion" "$layers" "$cudagraph_mode")"

  echo ""
  echo "=============================================================="
  echo "=== [$leg / profile] FULL+PIECEWISE cold (no-blessed-verify)"
  echo "=== fusion=$fusion layers=$layers config_hash=$config_hash"
  echo "=============================================================="

  docker rm -f "$CONTAINER" >/dev/null 2>&1 || true
  sleep 2

  # Profiler config — bounded active_iterations as defensive ceiling, but
  # we explicitly /start_profile and /stop_profile via trace_workload.py
  # plus a 120s host-side flush. Do NOT rely on active_iterations alone
  # (per memory feedback_active_iterations_dead_code).
  local profiler_config='{"profiler":"torch","torch_profiler_dir":"/tmp/profiles","ignore_frontend":true,"delay_iterations":0,"active_iterations":200,"torch_profiler_with_stack":false,"torch_profiler_use_gzip":true,"torch_profiler_record_shapes":false}'

  # β-coop cold-compile cache (shared across all legs).
  mkdir -p /tmp/nvllm-cute-cache

  docker run -d \
    --name "$CONTAINER" \
    --gpus all \
    --ipc=host \
    --network host \
    --privileged \
    -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
    -v "$HOME/.cache/flashinfer:/root/.cache/flashinfer" \
    -v "/tmp/nvllm-cute-cache:/opt/vllm/kernel_cache" \
    -v "$out_dir:/tmp/profiles" \
    -e B12X_CUTE_COMPILE_DISK_CACHE=1 \
    -e B12X_CUTE_COMPILE_CACHE_DIR=/opt/vllm/kernel_cache \
    -e VLLM_NVFP4_GEMM_BACKEND=cutlass \
    -e VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
    -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    -e CUTE_MLP_FUSION="$CUTE_MLP_FUSION_VAL" \
    -e CUTE_ATTN_FUSION="$CUTE_ATTN_FUSION_VAL" \
    -e CUTE_BETA_MIN_FREE_GB="${CUTE_BETA_MIN_FREE_GB:-8}" \
    -e CUTE_PHASE_E_FUSION="$fusion" \
    -e CUTE_PHASE_E_LAYERS="$layers" \
    -e CUTE_PHASE_E_FALLBACK_RAISE="$CUTE_PHASE_E_FALLBACK_RAISE_VAL" \
    -e CUTE_FULL_GRAPH_PROBE="$CUTE_FULL_GRAPH_PROBE_VAL" \
    -e CUTE_WO_RESET_LOG="$CUTE_WO_RESET_LOG_VAL" \
    -e CUTE_DISPATCH_AUDIT="$CUTE_DISPATCH_AUDIT_VAL" \
    "$NVLLM_IMAGE" \
    serve \
    --model "$HF_MODEL" \
    --served-model-name default \
    --host 0.0.0.0 --port "$PORT" \
    --kv-cache-dtype "$KV_CACHE_DTYPE" \
    --attention-backend "$ATTN_BACKEND_VAL" \
    --max-model-len "$MAX_MODEL_LEN_VAL" \
    --max-num-seqs "$MAX_NUM_SEQS_VAL" \
    --language-model-only \
    --limit-mm-per-prompt '{"image": 0, "video": 0}' \
    --mamba-cache-mode align \
    --trust-remote-code \
    --gpu-memory-utilization "$SERVE_GPU_UTIL" \
    --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS_VAL" \
    --kernel-config '{"enable_flashinfer_autotune":false}' \
    --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE","cudagraph_capture_sizes":[1]}' \
    --profiler-config "$profiler_config" \
    >/dev/null

  if ! wait_for_ready "$CONTAINER" "$PORT"; then
    docker logs --tail 200 "$CONTAINER" > "$out_dir/profile_fail.log" 2>&1 || true
    docker rm -f "$CONTAINER" >/dev/null 2>&1 || true
    return 1
  fi

  warmup_requests "$warmup_n" 64

  echo "  [profile] firing timed burst — concurrent=1 timed=$timed_n max_tokens=256"
  if ! .venv/bin/python docs/research/gemm_sweep/trace_workload.py \
      --base-url "http://localhost:$PORT/v1" \
      --model default \
      --warmup 5 --timed "$timed_n" --concurrent 1 \
      --max-tokens 256 \
      --timeout 600 \
      --profile-start "http://localhost:$PORT/start_profile" \
      --profile-stop  "http://localhost:$PORT/stop_profile"; then
    echo "  [profile] trace_workload.py exited non-zero — collecting logs and bailing leg" >&2
    docker logs "$CONTAINER" > "$out_dir/profile_serve.log" 2>&1 || true
    docker rm -f "$CONTAINER" >/dev/null 2>&1 || true
    return 1
  fi

  # CUPTI flush — torch+CUPTI serialization needs 30-90s after the active
  # window closes. 120s is dumb-but-reliable. Show file size every 30s
  # so operator can see progress.
  echo "  [profile] CUPTI flush — sleeping 120s..."
  local f
  for f in 1 2 3 4; do
    sleep 30
    echo "    [+${f}0s]"
    ls -la "$out_dir"/rank*.pt.trace.json.gz 2>/dev/null \
      | awk '{print "      current size:", $5, "bytes — ", $9}' || true
  done

  # Collect logs + teardown
  docker logs "$CONTAINER" > "$out_dir/profile_serve.log" 2>&1
  docker stop "$CONTAINER" >/dev/null
  docker rm "$CONTAINER" >/dev/null

  # Rename profiler trace
  local latest_trace
  latest_trace=$(ls -t "$out_dir"/rank*.pt.trace.json.gz 2>/dev/null | head -n1 || true)
  if [ -z "$latest_trace" ]; then
    echo "  [profile] ERROR: no fresh trace in $out_dir — profiler likely never flushed" >&2
    ls -la "$out_dir" >&2
    return 1
  fi
  mv "$latest_trace" "$out_dir/profile.pt.trace.json.gz"
  local trace_bytes
  trace_bytes=$(stat -c%s "$out_dir/profile.pt.trace.json.gz")
  echo "  [profile] trace saved: profile.pt.trace.json.gz ($trace_bytes bytes)"

  # Extract per-kernel CSV inline (fast, ~30s for ~200MB trace)
  echo "  [profile] extracting per-kernel CSV..."
  if ! .venv/bin/python docs/research/gemm_sweep/extract_e2e_kernels.py \
      --trace "$out_dir/profile.pt.trace.json.gz" \
      --config "$leg" \
      --out "$out_dir/profile_kernels.csv"; then
    echo "  [profile] WARNING: csv extraction failed; trace still available." >&2
  fi

  # Write metadata.json
  write_metadata "$leg" "profile" "$out_dir" "$cudagraph_mode" "$config_hash" "$fusion" "$layers"

  # Defensive: verify artifacts exist before marking DONE. When this function
  # is called via `func || OK=false`, set -e is disabled inside the function
  # body, so silent mv/stat failures upstream can otherwise slip through.
  local missing=()
  [ -f "$out_dir/profile.pt.trace.json.gz" ] || missing+=("profile.pt.trace.json.gz")
  [ -f "$out_dir/profile_serve.log" ] || missing+=("profile_serve.log")
  [ -f "$out_dir/profile_metadata.json" ] || missing+=("profile_metadata.json")
  if [ "${#missing[@]}" -gt 0 ]; then
    echo "  [$leg / profile] FAIL — missing artifacts: ${missing[*]}" >&2
    return 1
  fi

  touch "$out_dir/profile_DONE"
  echo "  [$leg / profile] DONE"
  return 0
}

# ---------------------------------------------------------------------------
# run_gsm8k_phase LEG FUSION LAYERS WARMUP_N
#   Boot 2: PIECEWISE-only deterministic, run gsm8k_eval_50.py.
# ---------------------------------------------------------------------------
run_gsm8k_phase() {
  local leg="$1" fusion="$2" layers="$3" warmup_n="$4"
  local out_dir="$OUT_ROOT/$leg"
  mkdir -p "$out_dir"

  if [ -f "$out_dir/gsm8k_DONE" ]; then
    echo ""
    echo "=== [$leg / gsm8k] SKIP (DONE marker exists)"
    return 0
  fi

  local cudagraph_mode="PIECEWISE"
  local config_hash
  config_hash="$(compute_config_hash "$fusion" "$layers" "$cudagraph_mode")"

  echo ""
  echo "=============================================================="
  echo "=== [$leg / gsm8k] PIECEWISE deterministic"
  echo "=== fusion=$fusion layers=$layers config_hash=$config_hash"
  echo "=============================================================="

  docker rm -f "$CONTAINER" >/dev/null 2>&1 || true
  sleep 2

  mkdir -p /tmp/nvllm-cute-cache

  docker run -d \
    --name "$CONTAINER" \
    --gpus all \
    --ipc=host \
    --network host \
    -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
    -v "$HOME/.cache/flashinfer:/root/.cache/flashinfer" \
    -v "/tmp/nvllm-cute-cache:/opt/vllm/kernel_cache" \
    -e B12X_CUTE_COMPILE_DISK_CACHE=1 \
    -e B12X_CUTE_COMPILE_CACHE_DIR=/opt/vllm/kernel_cache \
    -e VLLM_NVFP4_GEMM_BACKEND=cutlass \
    -e VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
    -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    -e CUTE_MLP_FUSION="$CUTE_MLP_FUSION_VAL" \
    -e CUTE_ATTN_FUSION="$CUTE_ATTN_FUSION_VAL" \
    -e CUTE_BETA_MIN_FREE_GB="${CUTE_BETA_MIN_FREE_GB:-8}" \
    -e CUTE_PHASE_E_FUSION="$fusion" \
    -e CUTE_PHASE_E_LAYERS="$layers" \
    -e CUTE_PHASE_E_FALLBACK_RAISE="$CUTE_PHASE_E_FALLBACK_RAISE_VAL" \
    -e CUTE_FULL_GRAPH_PROBE="$CUTE_FULL_GRAPH_PROBE_VAL" \
    -e CUTE_WO_RESET_LOG="$CUTE_WO_RESET_LOG_VAL" \
    -e CUTE_DISPATCH_AUDIT="$CUTE_DISPATCH_AUDIT_VAL" \
    "$NVLLM_IMAGE" \
    serve \
    --model "$HF_MODEL" \
    --served-model-name default \
    --host 0.0.0.0 --port "$PORT" \
    --kv-cache-dtype "$KV_CACHE_DTYPE" \
    --attention-backend "$ATTN_BACKEND_VAL" \
    --max-model-len "$MAX_MODEL_LEN_VAL" \
    --max-num-seqs "$MAX_NUM_SEQS_VAL" \
    --language-model-only \
    --limit-mm-per-prompt '{"image": 0, "video": 0}' \
    --mamba-cache-mode align \
    --trust-remote-code \
    --gpu-memory-utilization "$SERVE_GPU_UTIL" \
    --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS_VAL" \
    --kernel-config '{"enable_flashinfer_autotune":false}' \
    --compilation-config '{"cudagraph_mode":"PIECEWISE"}' \
    >/dev/null

  if ! wait_for_ready "$CONTAINER" "$PORT"; then
    docker logs --tail 200 "$CONTAINER" > "$out_dir/gsm8k_fail.log" 2>&1 || true
    docker rm -f "$CONTAINER" >/dev/null 2>&1 || true
    return 1
  fi

  warmup_requests "$warmup_n" 64

  echo "  [gsm8k] running gsm8k_eval_50 (n=50, seed=42, /no_think implicit via /v1/completions)..."
  set +e
  .venv/bin/python "$REPO_ROOT/scripts/gsm8k_eval_50.py" \
    --api "http://localhost:$PORT/v1" \
    --model default \
    --n 50 --seed 42 --max-tokens 512 --timeout 180 \
    --label "phaseE-tax-3leg-$leg" \
    --save "$out_dir/gsm8k.json"
  local gsm_rc=$?
  set -e

  if [ $gsm_rc -ne 0 ]; then
    echo "  [gsm8k] WARNING: gsm8k_eval_50 exited rc=$gsm_rc; gsm8k.json may be partial." >&2
  fi

  # Collect logs + teardown
  docker logs "$CONTAINER" > "$out_dir/gsm8k_serve.log" 2>&1
  docker stop "$CONTAINER" >/dev/null
  docker rm "$CONTAINER" >/dev/null

  # Quick stat dump (correct/wrong/errors raw, both gates)
  if [ -f "$out_dir/gsm8k.json" ]; then
    local correct n errors floor_pass ship_pass
    correct=$(jq -r '.correct' "$out_dir/gsm8k.json")
    n=$(jq -r '.n' "$out_dir/gsm8k.json")
    errors=$(jq -r '.errors' "$out_dir/gsm8k.json")
    # Diagnostic floor: ≥30/50 (no regression vs prior phase)
    if [ "$correct" -ge 30 ]; then floor_pass="PASS"; else floor_pass="FAIL"; fi
    # Ship gate: ≥47/50 (matches blessed production)
    if [ "$correct" -ge 47 ]; then ship_pass="PASS"; else ship_pass="FAIL"; fi
    echo "  [gsm8k] correct=$correct/$n errors=$errors  floor(>=30)=$floor_pass  ship(>=47)=$ship_pass"
    # Add gate verdicts to the json (in-place merge)
    jq --arg floor "$floor_pass" --arg ship "$ship_pass" \
       '. + {gate_floor: $floor, gate_ship: $ship}' \
       "$out_dir/gsm8k.json" > "$out_dir/gsm8k.json.tmp" && \
       mv "$out_dir/gsm8k.json.tmp" "$out_dir/gsm8k.json"
  fi

  write_metadata "$leg" "gsm8k" "$out_dir" "$cudagraph_mode" "$config_hash" "$fusion" "$layers"

  # Defensive: see profile-phase note about set -e + `||` callers.
  local missing=()
  [ -f "$out_dir/gsm8k_serve.log" ] || missing+=("gsm8k_serve.log")
  [ -f "$out_dir/gsm8k_metadata.json" ] || missing+=("gsm8k_metadata.json")
  # gsm8k.json is allowed to be missing (eval may have crashed before writing),
  # but gsm8k_serve.log and metadata MUST be there.
  if [ ! -f "$out_dir/gsm8k.json" ]; then
    echo "  [$leg / gsm8k] WARNING — gsm8k.json missing; eval likely crashed mid-run." >&2
  fi
  if [ "${#missing[@]}" -gt 0 ]; then
    echo "  [$leg / gsm8k] FAIL — missing required artifacts: ${missing[*]}" >&2
    return 1
  fi

  touch "$out_dir/gsm8k_DONE"
  echo "  [$leg / gsm8k] DONE"
  return 0
}

# ---------------------------------------------------------------------------
# Leg dispatch
#
# Layer set "0,1,2,3,4,5,6,7"     → β-coop active for layers in set ∩ full-attn
#                                    (Qwen3.5 full-attn = layers 3, 7 in the
#                                    lower-8 window — so β-coop hits 2 layers).
# Layer set "0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15"
#                                  → β-coop for ALL full-attn layers in the
#                                    16-layer block (Qwen3.5 has 4 full-attn
#                                    layers per 8: 3,7,11,15).
# ---------------------------------------------------------------------------

LEG_LOWER8_FUSION="1"
LEG_LOWER8_LAYERS="0,1,2,3,4,5,6,7"
LEG_LOWER8_WARMUP=15      # β-coop on, 2 per-layer JIT shapes
LEG_LOWER8_TIMED=10       # ~5-10 min profiled at concurrent=1, max_tokens=256

LEG_PHASEE_OFF_FUSION="0"
LEG_PHASEE_OFF_LAYERS="0,1,2,3,4,5,6,7"
LEG_PHASEE_OFF_WARMUP=4   # legacy DecodeKernel + external GEMM, no β-coop JIT
LEG_PHASEE_OFF_TIMED=10

LEG_ALL_BETA_FUSION="1"
LEG_ALL_BETA_LAYERS="0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15"
LEG_ALL_BETA_WARMUP=20    # β-coop on for 4 layers, 4 per-layer JIT shapes
LEG_ALL_BETA_TIMED=4      # short/confirmatory; per-token cost is ~3x lower8

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

OK=true

if [ "${SKIP_LOWER8:-0}" = "1" ]; then
  echo "=== [lower8] SKIP (SKIP_LOWER8=1)"
else
  run_profile_phase "lower8" "$LEG_LOWER8_FUSION" "$LEG_LOWER8_LAYERS" \
                    "$LEG_LOWER8_WARMUP" "$LEG_LOWER8_TIMED" || OK=false
  run_gsm8k_phase   "lower8" "$LEG_LOWER8_FUSION" "$LEG_LOWER8_LAYERS" \
                    "$LEG_LOWER8_WARMUP" || OK=false
fi

if [ "${SKIP_PHASEE_OFF:-0}" = "1" ]; then
  echo "=== [phaseE-off] SKIP (SKIP_PHASEE_OFF=1)"
else
  run_profile_phase "phaseE-off" "$LEG_PHASEE_OFF_FUSION" "$LEG_PHASEE_OFF_LAYERS" \
                    "$LEG_PHASEE_OFF_WARMUP" "$LEG_PHASEE_OFF_TIMED" || OK=false
  run_gsm8k_phase   "phaseE-off" "$LEG_PHASEE_OFF_FUSION" "$LEG_PHASEE_OFF_LAYERS" \
                    "$LEG_PHASEE_OFF_WARMUP" || OK=false
fi

if [ "${SKIP_ALL_BETA:-0}" = "1" ]; then
  echo "=== [all-beta] SKIP (SKIP_ALL_BETA=1)"
else
  run_profile_phase "all-beta" "$LEG_ALL_BETA_FUSION" "$LEG_ALL_BETA_LAYERS" \
                    "$LEG_ALL_BETA_WARMUP" "$LEG_ALL_BETA_TIMED" || OK=false
  run_gsm8k_phase   "all-beta" "$LEG_ALL_BETA_FUSION" "$LEG_ALL_BETA_LAYERS" \
                    "$LEG_ALL_BETA_WARMUP" || OK=false
fi

# ---------------------------------------------------------------------------
# Verdict
# ---------------------------------------------------------------------------

echo ""
echo "=============================================================="
if [ "$OK" = "true" ]; then
  echo "=== ok:true — all attempted legs completed; write summary.md next."
else
  echo "=== ok:false — at least one leg failed. Check *_fail.log files." >&2
fi
echo "=============================================================="
echo ""
echo "Output directory: $OUT_ROOT"
ls -la "$OUT_ROOT" 2>/dev/null | head -40

if [ "$OK" = "true" ]; then
  exit 0
else
  exit 1
fi
