#!/usr/bin/env bash
# Sentinel-gated 4-arm SSM zero-on-realloc ablation suite.
#
# Per-arm bind-mounts a per-arm sentinel dir at /run/nvllm :ro into the
# container; the sentinel-gated overlay reads filesystem-existence as the
# toggle (env vars are stripped by vLLM EngineCore subprocess spawn — see
# memory:feedback_vllm_enginecore_env_strip).
#
# Sentinel files (presence == ENABLED, absence == DISABLED):
#   /run/nvllm/zero_ssm_on_realloc.enabled
#   /run/nvllm/kv_zero_for_mamba_ids.enabled
#
# Execution proof comes from a docker-log triad emitted by the sentinel
# overlay (apply scripts/ablation/ssm_sentinel_overlay.patch to a clean
# checkout to build $PATCHED_REPO):
#   nvllm.ablation.sentinel_check name=<n> exists=<b> enabled=<b>
#   nvllm.ablation.first_fire     name=<n> n_block_ids=<N>
#   nvllm.ablation.fire_count     name=<n> count=<N>
#
# Arm matrix:
#   both     - both sentinels present  (full patch active)
#   neither  - no sentinels            (baseline)
#   ssm_only - SSM sentinel only       (mamba zeroer only)
#   kv_only  - KV sentinel only        (KV new-block-ids channel relax only)
#
# Usage:
#   scripts/ablation/run_ssm_ablation_suite.sh           # default 4 arms x 5 runs
#   scripts/ablation/run_ssm_ablation_suite.sh --force   # overwrite OUT_DIR
#
# Env overrides:
#   OUT_DIR             default /tmp/ssm_ablation_suite
#   NVLLM_IMAGE         default nvllm:gb10
#   REPO_ROOT           default git toplevel of this script
#   PATCHED_REPO        default /tmp/nvllm-ssm-sentinel-patched
#                         (must contain the sentinel overlay applied to a clean
#                          checkout; see scripts/ablation/ssm_sentinel_overlay.patch)
#   SENTINELS_ROOT      default /tmp/nvllm-ablation-sentinels
#   N_RUNS              default 5
#   GSM8K_FLOOR         default 45
#   CONTAINER           default nvllm-ssm-ablation
#   READY_TIMEOUT_S     default 600

set +u
FORCE=0
for arg in "$@"; do
  case "$arg" in
    --force) FORCE=1 ;;
    *) echo "ERROR: unknown argument: $arg (only --force is accepted)" >&2; exit 64 ;;
  esac
done

# ---------------------------------------------------------------------------
# Defaults / inputs. Resolved BEFORE set -e per memory:feedback_bash_runner_patterns.
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT_DIR="${OUT_DIR:-/tmp/ssm_ablation_suite}"
NVLLM_IMAGE="${NVLLM_IMAGE:-nvllm:gb10}"
REPO_ROOT="${REPO_ROOT:-$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel 2>/dev/null || echo "")}"
PATCHED_REPO="${PATCHED_REPO:-/tmp/nvllm-ssm-sentinel-patched}"
SENTINELS_ROOT="${SENTINELS_ROOT:-/tmp/nvllm-ablation-sentinels}"
N_RUNS="${N_RUNS:-5}"
GSM8K_N="${GSM8K_N:-50}"
GSM8K_SEED="${GSM8K_SEED:-42}"
GSM8K_MAX_TOKENS="${GSM8K_MAX_TOKENS:-512}"
GSM8K_TIMEOUT="${GSM8K_TIMEOUT:-600}"
GSM8K_FLOOR="${GSM8K_FLOOR:-45}"
CONTAINER="${CONTAINER:-nvllm-ssm-ablation}"
READY_TIMEOUT_S="${READY_TIMEOUT_S:-600}"
API="http://localhost:8000/v1"
METRICS_URL="http://localhost:8000/metrics"
HF_MODEL="${HF_MODEL:-ig1/Qwen3.5-27B-NVFP4}"
SERVED_NAME="${SERVED_NAME:-default}"

# ---------------------------------------------------------------------------
# Validate inputs BEFORE set -e.
# ---------------------------------------------------------------------------
if [ -z "$REPO_ROOT" ] || ! git -C "$REPO_ROOT" rev-parse --git-dir >/dev/null 2>&1; then
  echo "ERROR: REPO_ROOT='$REPO_ROOT' is not a git working tree" >&2; exit 1
fi
if ! docker image inspect "$NVLLM_IMAGE" >/dev/null 2>&1; then
  echo "ERROR: docker image '$NVLLM_IMAGE' not found" >&2; exit 1
fi
for f in vllm/v1/worker/utils.py vllm/v1/worker/gpu_model_runner.py vllm/v1/core/single_type_kv_cache_manager.py; do
  if [ ! -f "$PATCHED_REPO/$f" ]; then
    echo "ERROR: patched file missing: $PATCHED_REPO/$f" >&2
    echo "       Did you apply scripts/ablation/ssm_sentinel_overlay.patch to a clean checkout at PATCHED_REPO?" >&2
    exit 1
  fi
done
GSM8K_SCRIPT="$REPO_ROOT/scripts/gsm8k_eval_50.py"
if [ ! -f "$GSM8K_SCRIPT" ]; then
  echo "ERROR: gsm8k_eval_50.py missing at $GSM8K_SCRIPT" >&2; exit 1
fi
if [ ! -x "$REPO_ROOT/.venv/bin/python" ]; then
  echo "ERROR: $REPO_ROOT/.venv/bin/python not found" >&2; exit 1
fi
if ! grep -q -- '--run-index' "$GSM8K_SCRIPT" || ! grep -q -- '--metrics-url' "$GSM8K_SCRIPT"; then
  echo "ERROR: $GSM8K_SCRIPT missing --run-index / --metrics-url (not instrumented)" >&2; exit 1
fi
# Smoke-test that the patched files actually contain the sentinel markers.
if ! grep -q "_SSM_ZERO_SENTINEL" "$PATCHED_REPO/vllm/v1/worker/utils.py"; then
  echo "ERROR: $PATCHED_REPO/vllm/v1/worker/utils.py missing _SSM_ZERO_SENTINEL marker (overlay not applied?)" >&2; exit 1
fi
if ! grep -q "_KV_ZERO_SENTINEL" "$PATCHED_REPO/vllm/v1/core/single_type_kv_cache_manager.py"; then
  echo "ERROR: $PATCHED_REPO/vllm/v1/core/single_type_kv_cache_manager.py missing _KV_ZERO_SENTINEL marker (overlay not applied?)" >&2; exit 1
fi

# Refuse stale OUT_DIR unless --force.
if [ -d "$OUT_DIR" ] && [ "$(ls -A "$OUT_DIR" 2>/dev/null)" ] && [ "$FORCE" -ne 1 ]; then
  echo "ERROR: $OUT_DIR is non-empty (rerun with --force to overwrite)" >&2; exit 1
fi
rm -rf "$OUT_DIR"
mkdir -p "$OUT_DIR"

set -euo pipefail
log() { printf '[%(%Y-%m-%d %H:%M:%S)T] %s\n' -1 "$*"; }

# Counter helper: awk-based count of lines matching a literal substring.
# Replaces `grep -c PATTERN file || echo 0`, which emitted "0\n0" when grep
# found zero matches (grep prints "0" + exits 1, triggering the || fallback).
count_substr() {
  local pattern="$1"
  local file="$2"
  if [ ! -f "$file" ]; then
    printf '0'
    return
  fi
  awk -v pat="$pattern" 'index($0, pat) { n++ } END { print n+0 }' "$file"
}

# ---------------------------------------------------------------------------
# Per-arm sentinel directories. SENTINELS_ROOT is rebuilt every run so we
# can be sure no stray sentinel from a prior arm leaks in.
# ---------------------------------------------------------------------------
rm -rf "$SENTINELS_ROOT"
mkdir -p "$SENTINELS_ROOT"/{both,neither,ssm_only,kv_only}
touch "$SENTINELS_ROOT/both/zero_ssm_on_realloc.enabled"
touch "$SENTINELS_ROOT/both/kv_zero_for_mamba_ids.enabled"
touch "$SENTINELS_ROOT/ssm_only/zero_ssm_on_realloc.enabled"
touch "$SENTINELS_ROOT/kv_only/kv_zero_for_mamba_ids.enabled"
# 'neither/' stays empty by design.

log "sentinel dirs prepared:"
for arm in both neither ssm_only kv_only; do
  files=$(ls "$SENTINELS_ROOT/$arm" 2>/dev/null | tr '\n' ',' | sed 's/,$//')
  log "  $SENTINELS_ROOT/$arm = [$files]"
done

# Common bind-mounts (patch files are pre-built in $PATCHED_REPO; no apply step).
PATCHED_FILES=(
  "vllm/v1/core/single_type_kv_cache_manager.py"
  "vllm/v1/worker/utils.py"
  "vllm/v1/worker/gpu_model_runner.py"
)
BIND_MOUNTS=()
for f in "${PATCHED_FILES[@]}"; do
  BIND_MOUNTS+=(-v "$PATCHED_REPO/$f:/app/nvllm/$f")
done

GIT_SHA="$(git -C "$REPO_ROOT" rev-parse HEAD)"
IMAGE_ID="$(docker images --format '{{.Repository}}:{{.Tag}}@{{.ID}}' "$NVLLM_IMAGE" | head -n1)"
IMAGE_DIGEST="$(docker inspect --format '{{index .RepoDigests 0}}' "$NVLLM_IMAGE" 2>/dev/null || echo "no-digest")"
HOST_DRIVER="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -n1 || echo "unknown")"
HOST_KERNEL="$(uname -r 2>/dev/null || echo unknown)"
HOST_NAME="$(hostname 2>/dev/null || echo unknown)"
# Deterministic prompt-set identifier: (n, seed, model, served-name).
PROMPT_SET_HASH="$(printf '%s|%s|%s|%s' "$GSM8K_N" "$GSM8K_SEED" "$HF_MODEL" "$SERVED_NAME" | sha256sum | awk '{print $1}')"

# One-time runner manifest written before any arm runs.
{
  echo "{"
  echo "  \"runner\": \"$0\","
  echo "  \"started_utc\": \"$(date -u +%Y-%m-%dT%H:%M:%SZ)\","
  echo "  \"git_sha\": \"$GIT_SHA\","
  echo "  \"image\": \"$NVLLM_IMAGE\","
  echo "  \"image_id\": \"$IMAGE_ID\","
  echo "  \"image_digest\": \"$IMAGE_DIGEST\","
  echo "  \"patched_repo\": \"$PATCHED_REPO\","
  echo "  \"sentinels_root\": \"$SENTINELS_ROOT\","
  echo "  \"host_name\": \"$HOST_NAME\","
  echo "  \"host_driver\": \"$HOST_DRIVER\","
  echo "  \"host_kernel\": \"$HOST_KERNEL\","
  echo "  \"gsm8k_n\": $GSM8K_N,"
  echo "  \"gsm8k_seed\": $GSM8K_SEED,"
  echo "  \"gsm8k_max_tokens\": $GSM8K_MAX_TOKENS,"
  echo "  \"prompt_set_hash\": \"$PROMPT_SET_HASH\","
  echo "  \"hf_model\": \"$HF_MODEL\","
  echo "  \"n_runs\": $N_RUNS,"
  echo "  \"arms\": [\"both\", \"neither\", \"ssm_only\", \"kv_only\"]"
  echo "}"
} > "$OUT_DIR/runner_manifest.json"
log "runner manifest: $OUT_DIR/runner_manifest.json"

# ---------------------------------------------------------------------------
# Arm matrix.
# ---------------------------------------------------------------------------
ARM_NAMES=(both neither ssm_only kv_only)
declare -A ARM_SSM=( [both]=1 [neither]=0 [ssm_only]=1 [kv_only]=0 )
declare -A ARM_KV=(  [both]=1 [neither]=0 [ssm_only]=0 [kv_only]=1 )
declare -a ARM_GATE_PASS
declare -a ARM_GIT_SUMMARY

for arm_idx in "${!ARM_NAMES[@]}"; do
  ARM="${ARM_NAMES[$arm_idx]}"
  SSM_VAL="${ARM_SSM[$ARM]}"
  KV_VAL="${ARM_KV[$ARM]}"
  ARM_DIR="$OUT_DIR/$ARM"
  ARM_SENTINEL_DIR="$SENTINELS_ROOT/$ARM"
  mkdir -p "$ARM_DIR"
  log "========================================================================"
  log "ARM $((arm_idx + 1))/4: $ARM (SSM_sentinel=$SSM_VAL, KV_sentinel=$KV_VAL)"
  log "========================================================================"

  arm_files=$(ls "$ARM_SENTINEL_DIR" 2>/dev/null | tr '\n' ',' | sed 's/,$//')
  log "nvllm.ablation.arm=$ARM host_sentinels_dir=$ARM_SENTINEL_DIR container_sentinels_dir=/run/nvllm files=[$arm_files]"

  docker rm -f "$CONTAINER" >/dev/null 2>&1 || true

  HOST_TRACE_DIR="$ARM_DIR/trace"
  mkdir -p "$HOST_TRACE_DIR"
  CONT_TRACE_PATH="/tmp/ssm_zero_trace/mamba_slot_trace.jsonl"
  SERVE_LOG="$ARM_DIR/serve.log"

  log "boot patched server (arm=$ARM, image=$NVLLM_IMAGE, container=$CONTAINER)"
  # shellcheck disable=SC2086
  docker run -d \
    --name "$CONTAINER" \
    --gpus all \
    --ipc=host \
    --network host \
    --shm-size=8g \
    -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
    -v "$HOME/.cache/flashinfer:/root/.cache/flashinfer" \
    -v "$HOST_TRACE_DIR:/tmp/ssm_zero_trace" \
    -v "$ARM_SENTINEL_DIR:/run/nvllm:ro" \
    "${BIND_MOUNTS[@]}" \
    -e VLLM_NVFP4_GEMM_BACKEND=cutlass \
    -e VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
    -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    -e NVLLM_MAMBA_SLOT_TRACE="$CONT_TRACE_PATH" \
    -e CUTE_PHASE_E_FUSION=1 \
    -e CUTE_PHASE_E_PATH=auto \
    -e CUTE_PHASE_E_LAYERS="3,7" \
    -e CUTE_PHASE_E_FALLBACK_RAISE=1 \
    -e CUTE_WO_SPLIT=8 \
    "$NVLLM_IMAGE" \
    serve \
    --model "$HF_MODEL" \
    --served-model-name "$SERVED_NAME" \
    --host 0.0.0.0 --port 8000 \
    --gpu-memory-utilization 0.85 \
    > "$SERVE_LOG" 2>&1

  # Clear stale .pyc from bind-mounted dirs (memory:feedback_docker_bindmount).
  sleep 2
  docker exec "$CONTAINER" sh -c '
    find /app/nvllm/vllm/v1/core /app/nvllm/vllm/v1/worker \
         -maxdepth 3 -name "__pycache__" -type d \
         -exec rm -rf {} + 2>/dev/null || true
  ' || true

  # Active readiness probe (memory:feedback_active_serve_readiness_probe).
  deadline=$((SECONDS + READY_TIMEOUT_S))
  log "wait for ready on $API/models ..."
  READY=0
  while [ "$SECONDS" -lt "$deadline" ]; do
    if ! docker ps --filter "name=^/${CONTAINER}$" --format '{{.Names}}' | grep -qx "$CONTAINER"; then
      echo "ERROR: container died during boot (arm=$ARM); tail $SERVE_LOG" >&2
      docker logs "$CONTAINER" > "$ARM_DIR/docker.log" 2>&1 || true
      ARM_GATE_PASS[$arm_idx]="boot_fail"
      ARM_GIT_SUMMARY[$arm_idx]="-"
      break
    fi
    if curl -fsS "$API/models" >/dev/null 2>&1; then
      if curl -fsS "$API/completions" -H 'Content-Type: application/json' \
          -d '{"model":"'"$SERVED_NAME"'","prompt":"warmup","max_tokens":8,"temperature":0}' \
          >/dev/null 2>&1; then
        log "ready (~${SECONDS}s)"
        READY=1
        break
      fi
    fi
    sleep 5
  done
  if [ "$READY" -ne 1 ]; then
    log "WARN: arm=$ARM did not become ready within ${READY_TIMEOUT_S}s"
    docker logs "$CONTAINER" > "$ARM_DIR/docker.log" 2>&1 || true
    docker rm -f "$CONTAINER" >/dev/null 2>&1 || true
    {
      echo "{"
      echo "  \"arm\": \"$ARM\","
      echo "  \"ssm_sentinel\": $SSM_VAL,"
      echo "  \"kv_sentinel\": $KV_VAL,"
      echo "  \"git_sha\": \"$GIT_SHA\","
      echo "  \"image\": \"$NVLLM_IMAGE\","
      echo "  \"image_id\": \"$IMAGE_ID\","
      echo "  \"n_runs\": $N_RUNS,"
      echo "  \"gsm8k_floor\": $GSM8K_FLOOR,"
      echo "  \"runs\": [],"
      echo "  \"container_alive_at_end\": false,"
      echo "  \"gate_pass\": false,"
      echo "  \"reason\": \"server_never_ready\""
      echo "}"
    } > "$ARM_DIR/verdict.json"
    ARM_GATE_PASS[$arm_idx]="not_ready"
    ARM_GIT_SUMMARY[$arm_idx]="-"
    continue
  fi

  # Bind-mount proof: sentinel marker present in patched utils.py inside container.
  INSIDE_MARKER=$(docker exec "$CONTAINER" sh -c "grep -c '_SSM_ZERO_SENTINEL' /app/nvllm/vllm/v1/worker/utils.py 2>/dev/null" || printf '0')
  INSIDE_MARKER=${INSIDE_MARKER:-0}
  if [ "$INSIDE_MARKER" -lt 1 ]; then
    echo "FAIL: bind-mount did not land inside container (arm=$ARM)" >&2
    docker logs "$CONTAINER" > "$ARM_DIR/docker.log" 2>&1 || true
    docker rm -f "$CONTAINER" >/dev/null 2>&1 || true
    ARM_GATE_PASS[$arm_idx]="bind_fail"
    ARM_GIT_SUMMARY[$arm_idx]="-"
    continue
  fi
  log "bind-mount verified inside container (marker count=$INSIDE_MARKER)"

  # Sentinel-dir proof: docker exec ls /run/nvllm
  SENTINEL_LIST_INSIDE="$(docker exec "$CONTAINER" sh -c 'ls /run/nvllm 2>/dev/null | tr "\n" "," | sed "s/,$//"' || true)"
  log "sentinels inside container /run/nvllm = [$SENTINEL_LIST_INSIDE]"

  CONTAINER_ID="$(docker inspect --format '{{.Id}}' "$CONTAINER" 2>/dev/null || echo unknown)"
  log "container id: $CONTAINER_ID"
  docker inspect "$CONTAINER" > "$ARM_DIR/docker_inspect.json" 2>/dev/null || true

  declare -a RUN_RESULTS=()

  for run_idx in $(seq 1 "$N_RUNS"); do
    RUN_DIR="$ARM_DIR/run${run_idx}"
    mkdir -p "$RUN_DIR"
    log "==> arm=$ARM run ${run_idx}/${N_RUNS} GSM8K-${GSM8K_N}"
    set +e
    ( cd "$REPO_ROOT" && \
      .venv/bin/python "$GSM8K_SCRIPT" \
        --api "$API" --model "$SERVED_NAME" \
        --n "$GSM8K_N" --seed "$GSM8K_SEED" \
        --max-tokens "$GSM8K_MAX_TOKENS" --timeout "$GSM8K_TIMEOUT" \
        --label "ablation_${ARM}_run${run_idx}" \
        --run-index "$run_idx" \
        --metrics-url "$METRICS_URL" \
        --save "$RUN_DIR/gsm8k.json" ) 2>&1 | tee "$RUN_DIR/gsm8k.log"
    RC_GSM="${PIPESTATUS[0]}"
    set -e
    if [ "$RC_GSM" -ne 0 ]; then
      log "WARN: arm=$ARM run ${run_idx} returned rc=$RC_GSM"
    fi
    if [ ! -f "$RUN_DIR/gsm8k.json" ]; then
      log "FAIL: arm=$ARM run ${run_idx} did not produce gsm8k.json"
      RUN_RESULTS+=("$run_idx fail no-json")
      continue
    fi
    CORRECT="$("$REPO_ROOT/.venv/bin/python" -c "import json; print(json.load(open('$RUN_DIR/gsm8k.json'))['correct'])")"
    ERRORS="$("$REPO_ROOT/.venv/bin/python" -c "import json; print(json.load(open('$RUN_DIR/gsm8k.json'))['errors'])")"
    RUN_RESULTS+=("$run_idx $CORRECT $ERRORS")
    log "<== arm=$ARM run ${run_idx} correct=$CORRECT errors=$ERRORS"
  done

  # Final capture + teardown for this arm.
  docker logs "$CONTAINER" > "$ARM_DIR/docker.log" 2>&1 || true
  CONTAINER_ALIVE="false"
  if docker ps --filter "name=^/${CONTAINER}$" --format '{{.Names}}' | grep -qx "$CONTAINER"; then
    CONTAINER_ALIVE="true"
  fi

  # Extract ablation event triad from docker logs - this is the execution proof.
  ABLATION_EVENTS="$ARM_DIR/ablation_events.log"
  grep -E "nvllm.ablation" "$ARM_DIR/docker.log" > "$ABLATION_EVENTS" || true
  SENTINEL_CHECK_LINES=$(count_substr "nvllm.ablation.sentinel_check" "$ABLATION_EVENTS")
  FIRST_FIRE_LINES=$(count_substr "nvllm.ablation.first_fire" "$ABLATION_EVENTS")
  FIRE_COUNT_LINES=$(count_substr "nvllm.ablation.fire_count" "$ABLATION_EVENTS")
  # Per-gate breakdown: did SSM gate fire? did KV gate fire?
  SSM_FIRST_FIRE=$(count_substr "nvllm.ablation.first_fire name=ssm_zero_on_realloc" "$ABLATION_EVENTS")
  KV_FIRST_FIRE=$(count_substr "nvllm.ablation.first_fire name=kv_zero_for_mamba_ids" "$ABLATION_EVENTS")
  log "ablation events for $ARM: sentinel_check=$SENTINEL_CHECK_LINES first_fire=$FIRST_FIRE_LINES fire_count=$FIRE_COUNT_LINES (ssm_fire=$SSM_FIRST_FIRE kv_fire=$KV_FIRST_FIRE)"

  # Harness validation gate: enabled => first_fire>=1; disabled => first_fire==0.
  HARNESS_PASS="true"
  HARNESS_REASON="ok"
  if [ "$SSM_VAL" -eq 1 ] && [ "$SSM_FIRST_FIRE" -lt 1 ]; then
    HARNESS_PASS="false"; HARNESS_REASON="ssm_enabled_but_no_first_fire"
  fi
  if [ "$SSM_VAL" -eq 0 ] && [ "$SSM_FIRST_FIRE" -gt 0 ]; then
    HARNESS_PASS="false"; HARNESS_REASON="ssm_disabled_but_first_fire_observed"
  fi
  if [ "$KV_VAL" -eq 1 ] && [ "$KV_FIRST_FIRE" -lt 1 ]; then
    HARNESS_PASS="false"; HARNESS_REASON="kv_enabled_but_no_first_fire"
  fi
  if [ "$KV_VAL" -eq 0 ] && [ "$KV_FIRST_FIRE" -gt 0 ]; then
    HARNESS_PASS="false"; HARNESS_REASON="kv_disabled_but_first_fire_observed"
  fi
  log "harness validation for $ARM: pass=$HARNESS_PASS reason=$HARNESS_REASON"

  docker rm -f "$CONTAINER" >/dev/null 2>&1 || true

  TRACE_FILE="$HOST_TRACE_DIR/mamba_slot_trace.jsonl"
  TRACE_LINES=0
  if [ -f "$TRACE_FILE" ]; then
    TRACE_LINES=$(wc -l < "$TRACE_FILE" | tr -d ' ')
  fi

  # Gate evaluation.
  ALL_PASS="true"
  for line in "${RUN_RESULTS[@]}"; do
    read -r idx correct errors <<< "$line"
    if [ "$correct" = "fail" ]; then ALL_PASS="false"; continue; fi
    if [ "$correct" -lt "$GSM8K_FLOOR" ]; then ALL_PASS="false"; fi
    if [ "$errors" -gt 0 ]; then ALL_PASS="false"; fi
  done

  CORRUPT_HITS=0
  if [ -f "$ARM_DIR/docker.log" ]; then
    CORRUPT_HITS=$(awk '/ERROR|FATAL|state.*corrupt/{n++} END{print n+0}' "$ARM_DIR/docker.log")
  fi
  [ "$CORRUPT_HITS" -gt 0 ] && ALL_PASS="false"
  [ "$CONTAINER_ALIVE" != "true" ] && ALL_PASS="false"

  # Per-arm token summary.
  PERQ_FILE="$ARM_DIR/perq.jsonl"
  rm -f "$PERQ_FILE"
  for run_idx in $(seq 1 "$N_RUNS"); do
    if [ -f "$ARM_DIR/run${run_idx}/perq.jsonl" ]; then
      cat "$ARM_DIR/run${run_idx}/perq.jsonl" >> "$PERQ_FILE"
    fi
  done
  TOKEN_SUMMARY="$($REPO_ROOT/.venv/bin/python - <<EOF
import json, statistics
sumc, sump, count = 0, 0, 0
walls, decode_rates = [], []
try:
    with open("$PERQ_FILE") as f:
        for line in f:
            try:
                r = json.loads(line)
            except Exception:
                continue
            count += 1
            sumc += int(r.get("completion_tokens", 0) or 0)
            sump += int(r.get("prompt_tokens", 0) or 0)
            w = float(r.get("wall_time_s", 0) or 0)
            d = float(r.get("decode_tok_s", 0) or 0)
            if w > 0:
                walls.append(w)
            if d > 0:
                decode_rates.append(d)
except FileNotFoundError:
    pass
print(json.dumps({
    "n_questions": count,
    "sum_completion_tokens": sumc,
    "sum_prompt_tokens": sump,
    "median_wall_time_s": (statistics.median(walls) if walls else 0.0),
    "median_decode_tok_s": (statistics.median(decode_rates) if decode_rates else 0.0),
}))
EOF
)"

  {
    echo "{"
    echo "  \"arm\": \"$ARM\","
    echo "  \"ssm_sentinel\": $SSM_VAL,"
    echo "  \"kv_sentinel\": $KV_VAL,"
    echo "  \"hypothesis\": \"ssm_zero_on_realloc_ablation_sentinel_gated\","
    echo "  \"patched_repo\": \"$PATCHED_REPO\","
    echo "  \"sentinel_dir\": \"$ARM_SENTINEL_DIR\","
    echo "  \"sentinel_files_inside\": \"$SENTINEL_LIST_INSIDE\","
    echo "  \"container_id\": \"$CONTAINER_ID\","
    echo "  \"host_driver\": \"$HOST_DRIVER\","
    echo "  \"prompt_set_hash\": \"$PROMPT_SET_HASH\","
    echo "  \"harness_validation\": {\"pass\": $HARNESS_PASS, \"reason\": \"$HARNESS_REASON\", \"ssm_first_fire\": $SSM_FIRST_FIRE, \"kv_first_fire\": $KV_FIRST_FIRE},"
    echo "  \"git_sha\": \"$GIT_SHA\","
    echo "  \"image\": \"$NVLLM_IMAGE\","
    echo "  \"image_id\": \"$IMAGE_ID\","
    echo "  \"phase_e_layers\": \"3,7\","
    echo "  \"phase_e_fusion\": 1,"
    echo "  \"phase_e_path\": \"auto\","
    echo "  \"wo_split\": 8,"
    echo "  \"n_runs\": $N_RUNS,"
    echo "  \"gsm8k_floor\": $GSM8K_FLOOR,"
    echo "  \"mamba_slot_trace_lines\": $TRACE_LINES,"
    echo "  \"ablation_events\": {\"sentinel_check\": $SENTINEL_CHECK_LINES, \"first_fire\": $FIRST_FIRE_LINES, \"fire_count\": $FIRE_COUNT_LINES},"
    echo "  \"token_summary\": $TOKEN_SUMMARY,"
    echo "  \"runs\": ["
    first=1
    for line in "${RUN_RESULTS[@]}"; do
      read -r idx correct errors <<< "$line"
      [ "$first" -eq 0 ] && echo "," || true
      first=0
      if [ "$correct" = "fail" ]; then
        echo -n "    {\"run\": $idx, \"ok\": false, \"reason\": \"no_gsm8k_json\"}"
      else
        pass=true
        [ "$correct" -lt "$GSM8K_FLOOR" ] && pass=false
        [ "$errors" -gt 0 ] && pass=false
        echo -n "    {\"run\": $idx, \"correct\": $correct, \"errors\": $errors, \"pass\": $pass}"
      fi
    done
    echo
    echo "  ],"
    echo "  \"container_alive_at_end\": $CONTAINER_ALIVE,"
    echo "  \"docker_log_corruption_hits\": $CORRUPT_HITS,"
    echo "  \"gate_pass\": $ALL_PASS,"
    echo "  \"harness_pass\": $HARNESS_PASS"
    echo "}"
  } > "$ARM_DIR/verdict.json"

  ARM_GATE_PASS[$arm_idx]="$ALL_PASS"
  SUM=""
  for line in "${RUN_RESULTS[@]}"; do
    read -r idx correct errors <<< "$line"
    SUM+="${correct},"
  done
  ARM_GIT_SUMMARY[$arm_idx]="${SUM%,}"

  log "<== arm=$ARM complete (gate_pass=$ALL_PASS, trace_lines=$TRACE_LINES, sentinel_check=$SENTINEL_CHECK_LINES, first_fire=$FIRST_FIRE_LINES, runs=$SUM)"
done

# ---------------------------------------------------------------------------
# Aggregate comparison.json
# ---------------------------------------------------------------------------
{
  echo "{"
  echo "  \"out_dir\": \"$OUT_DIR\","
  echo "  \"git_sha\": \"$GIT_SHA\","
  echo "  \"image\": \"$NVLLM_IMAGE\","
  echo "  \"n_runs\": $N_RUNS,"
  echo "  \"gsm8k_floor\": $GSM8K_FLOOR,"
  echo "  \"patched_repo\": \"$PATCHED_REPO\","
  echo "  \"sentinels_root\": \"$SENTINELS_ROOT\","
  echo "  \"arms\": ["
  first=1
  for arm_idx in "${!ARM_NAMES[@]}"; do
    ARM="${ARM_NAMES[$arm_idx]}"
    [ "$first" -eq 0 ] && echo "," || true
    first=0
    GP="${ARM_GATE_PASS[$arm_idx]:-unknown}"
    SUM="${ARM_GIT_SUMMARY[$arm_idx]:-unknown}"
    echo -n "    {\"arm\": \"$ARM\", \"ssm_sentinel\": ${ARM_SSM[$ARM]}, \"kv_sentinel\": ${ARM_KV[$ARM]}, \"gate_pass\": \"$GP\", \"correct_per_run\": \"$SUM\", \"verdict\": \"$OUT_DIR/$ARM/verdict.json\"}"
  done
  echo
  echo "  ]"
  echo "}"
} > "$OUT_DIR/comparison.json"

log "ablation suite complete"
log "comparison: $OUT_DIR/comparison.json"
exit 0
