#!/usr/bin/env bash
# D2.3 bisection leg: 12L_3_47 arm with CUTE_PHASE_E_FUSION=0 (vs Stage 2b's 1).
#
# Hypothesis under test (post-D2.2 narrowing):
#   D2.1 (wo_split=1) and D2.2 (path=lite) both reproduced the Stage 2b
#   collapse shape exactly — bug is path-agnostic and the persistent β-coop
#   workspace is falsified as the substrate. D2.3 turns OFF Phase E entirely
#   (CUTE_PHASE_E_FUSION=0): no β-coop and no β-lite fire on any layer;
#   the model falls through to base CuTe paged attention on all 48 layers.
#
#   - If this run is STABLE across 5 GSM8K-50 passes, the Phase E machinery
#     (membership check, dispatch decision, restricted-layer set caching,
#     framework output routing under the Phase E predicates) is the
#     substrate. Hardening + locking 2L_3_7 default is the production fix.
#   - If this run COLLAPSES like Stage 2b, the bug lives in base CuTe path /
#     vLLM scheduling / hybrid mamba machinery — entirely upstream of Phase E.
#     2L_3_7 itself becomes at-risk; D2.5 (2L survival control) escalates.
#
# Same soak shape as Stage 2b: boots scripts/serve-cute.sh ONCE with the
# 12L layer list set but FUSION=0, runs the dispatch audit once (expects
# empty coop and lite layer sets, records present), then runs GSM8K-50 N
# times back-to-back on the same container.
#
# Usage:
#   ./soak_d2_3_runner.sh            # 5 runs, default
#   ./soak_d2_3_runner.sh --force    # overwrite existing soak_d2_3_phaseE_off/ dir
#
# Gate D2.3: all 5 runs ≥45/50, 0 errors, container alive, no docker.log
# corruption hits. Same gate as Stage 2b; verdict.json indicates pass/fail.

set +u
FORCE=0
for arg in "$@"; do
  case "$arg" in
    --force) FORCE=1 ;;
    *) echo "ERROR: unknown argument: $arg (only --force is accepted)" >&2; exit 64 ;;
  esac
done

# ---------------------------------------------------------------------------
# Resolve paths.
# ---------------------------------------------------------------------------
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git -C "$HERE" rev-parse --show-toplevel)"
DOC_DIR="$HERE"
SOAK_DIR="$DOC_DIR/soak_d2_3_phaseE_off"
EXTRACT_DISPATCH="$DOC_DIR/extract_dispatch_log.py"
GSM8K_SCRIPT="$REPO_ROOT/scripts/gsm8k_eval_50.py"

# Worktree guard (matches earlier soak runners).
if [ "${ALLOW_PRIMARY_CHECKOUT:-0}" != "1" ]; then
  case "$REPO_ROOT" in
    *nvllm-beta-layer-sweep-wo8) : ;;  # ok
    *)
      echo "ERROR: refusing to run from $REPO_ROOT" >&2
      echo "       expected a worktree path ending in nvllm-beta-layer-sweep-wo8" >&2
      echo "       (set ALLOW_PRIMARY_CHECKOUT=1 to override)" >&2
      exit 65
      ;;
  esac
fi

if [ ! -x "$REPO_ROOT/.venv/bin/python" ]; then
  echo "ERROR: $REPO_ROOT/.venv/bin/python not found" >&2; exit 1
fi
if [ ! -f "$EXTRACT_DISPATCH" ]; then
  echo "ERROR: extract_dispatch_log.py missing at $EXTRACT_DISPATCH" >&2; exit 1
fi
if [ ! -f "$GSM8K_SCRIPT" ]; then
  echo "ERROR: gsm8k_eval_50.py missing at $GSM8K_SCRIPT" >&2; exit 1
fi

if [ -d "$SOAK_DIR" ] && [ "$(ls -A "$SOAK_DIR" 2>/dev/null)" ] && [ "$FORCE" -ne 1 ]; then
  echo "ERROR: $SOAK_DIR is non-empty (rerun with --force to overwrite)" >&2; exit 1
fi
rm -rf "$SOAK_DIR"
mkdir -p "$SOAK_DIR"

set -euo pipefail
N_RUNS="${N_RUNS:-5}"
GSM8K_N="${GSM8K_N:-50}"
GSM8K_SEED="${GSM8K_SEED:-42}"
GSM8K_MAX_TOKENS="${GSM8K_MAX_TOKENS:-512}"
GSM8K_TIMEOUT="${GSM8K_TIMEOUT:-600}"
GSM8K_FLOOR="${GSM8K_FLOOR:-45}"
READY_TIMEOUT_S="${READY_TIMEOUT_S:-600}"
CONTAINER="${CONTAINER:-nvllm}"
API="http://localhost:8000/v1"
GIT_SHA="$(git -C "$REPO_ROOT" rev-parse HEAD)"
IMAGE_ID="$(docker images --format '{{.Repository}}:{{.Tag}}@{{.ID}}' nvllm:gb10 | head -n1)"

# D2.3 config: same 12L layer list (silently ignored when FUSION=0), FUSION
# OFF. wo_split=8 base (irrelevant when no β kernel fires). Layer list kept
# explicit to ensure only one knob differs from Stage 2b base / D2.0.
ARM="12L_3_47_d2_3_phaseE_off"
PHASE_E_LAYERS="3,7,11,15,19,23,27,31,35,39,43,47"
WO_SPLIT="8"

log() { printf '[%(%Y-%m-%d %H:%M:%S)T] %s\n' -1 "$*"; }

# ---------------------------------------------------------------------------
# Boot the server with chosen arm's env.
# ---------------------------------------------------------------------------
docker rm -f "$CONTAINER" >/dev/null 2>&1 || true

export CUTE_PHASE_E_FUSION=0
export CUTE_PHASE_E_PATH="auto"
export CUTE_PHASE_E_LAYERS="$PHASE_E_LAYERS"
export CUTE_PHASE_E_FALLBACK_RAISE=1
export CUTE_PHASE_E_DISPATCH_LOG=1
export CUTE_BETA_REGION_TIMING=0
export CUTE_WO_SPLIT="$WO_SPLIT"
export NVLLM_BIND_MOUNT_CUTE_PAGED=1
export NVLLM_BIND_MOUNT_QWEN35=1

SERVE_LOG="$SOAK_DIR/serve.log"
log "boot serve-cute.sh (arm=$ARM, layers=[$PHASE_E_LAYERS], FUSION=0, wo=$WO_SPLIT) ..."
set +e
( cd "$REPO_ROOT" && bash scripts/serve-cute.sh ) > "$SERVE_LOG" 2>&1
RC_SERVE="$?"
set -e
if [ "$RC_SERVE" -ne 0 ]; then
  echo "FAIL: serve-cute.sh exit=$RC_SERVE; see $SERVE_LOG" >&2
  exit 1
fi

deadline=$((SECONDS + READY_TIMEOUT_S))
log "wait for ready on $API/models ..."
while [ "$SECONDS" -lt "$deadline" ]; do
  if ! docker ps --filter "name=^/${CONTAINER}$" --format '{{.Names}}' | grep -qx "$CONTAINER"; then
    echo "ERROR: container died during boot; see $SERVE_LOG" >&2; exit 1
  fi
  if curl -fsS "$API/models" >/dev/null 2>&1; then
    if curl -fsS "$API/completions" -H 'Content-Type: application/json' \
        -d '{"model":"default","prompt":"warmup","max_tokens":8,"temperature":0}' \
        >/dev/null 2>&1; then
      log "ready (~${SECONDS}s)"
      break
    fi
  fi
  sleep 5
done

# ---------------------------------------------------------------------------
# One-shot dispatch audit. When FUSION=0, no β-coop and no β-lite layers
# should fire. The dispatch log still emits records (DISPATCH_LOG=1 fires
# inside the decode forward unconditionally on the predicate path), just
# with use_beta_coop=False AND use_beta_lite=False on every layer. The
# --expect-coop-layers "" gate is also fail-closed on lite contamination
# via the existing symmetric guard.
# ---------------------------------------------------------------------------
log "dispatch audit (expect coop_layers=[] AND lite_layers=[], records present)"
set +e
docker logs "$CONTAINER" 2>&1 \
  | "$REPO_ROOT/.venv/bin/python" "$EXTRACT_DISPATCH" \
      --expect-coop-layers "" \
      --json-out "$SOAK_DIR/dispatch_audit.json" \
      --require-records \
  > "$SOAK_DIR/dispatch_audit.stdout" 2> "$SOAK_DIR/dispatch_audit.stderr"
RC_AUDIT="${PIPESTATUS[1]}"
set -e
if [ "$RC_AUDIT" -ne 0 ]; then
  echo "FAIL: dispatch audit (rc=$RC_AUDIT); see $SOAK_DIR/dispatch_audit.stderr" >&2
  docker logs "$CONTAINER" > "$SOAK_DIR/docker.log" 2>&1 || true
  docker rm -f "$CONTAINER" >/dev/null 2>&1 || true
  exit 1
fi
log "dispatch audit OK"

cp /tmp/c2_diag/ENV "$SOAK_DIR/c2_diag_ENV.txt" 2>/dev/null || true
docker inspect "$CONTAINER" > "$SOAK_DIR/docker_inspect.json" 2>/dev/null || true

# ---------------------------------------------------------------------------
# Run N sequential GSM8K-50 passes on the same container.
# ---------------------------------------------------------------------------
declare -a RUN_RESULTS=()
for run_idx in $(seq 1 "$N_RUNS"); do
  RUN_DIR="$SOAK_DIR/run${run_idx}"
  mkdir -p "$RUN_DIR"
  log "==> run ${run_idx}/${N_RUNS} GSM8K-50"
  set +e
  ( cd "$REPO_ROOT" && \
    .venv/bin/python scripts/gsm8k_eval_50.py \
      --api "$API" --model default \
      --n "$GSM8K_N" --seed "$GSM8K_SEED" \
      --max-tokens "$GSM8K_MAX_TOKENS" --timeout "$GSM8K_TIMEOUT" \
      --label "soak_${ARM}_run${run_idx}" \
      --save "$RUN_DIR/gsm8k.json" ) 2>&1 | tee "$RUN_DIR/gsm8k.log"
  RC_GSM="${PIPESTATUS[0]}"
  set -e
  if [ "$RC_GSM" -ne 0 ]; then
    log "WARN: run ${run_idx} returned rc=$RC_GSM"
  fi
  if [ ! -f "$RUN_DIR/gsm8k.json" ]; then
    log "FAIL: run ${run_idx} did not produce gsm8k.json"
    RUN_RESULTS+=("$run_idx fail no-json")
    continue
  fi
  CORRECT="$("$REPO_ROOT/.venv/bin/python" -c "import json; print(json.load(open('$RUN_DIR/gsm8k.json'))['correct'])")"
  ERRORS="$("$REPO_ROOT/.venv/bin/python" -c "import json; print(json.load(open('$RUN_DIR/gsm8k.json'))['errors'])")"
  RUN_RESULTS+=("$run_idx $CORRECT $ERRORS")
  log "<== run ${run_idx} correct=$CORRECT errors=$ERRORS"
done

# ---------------------------------------------------------------------------
# Final docker.log capture + container teardown.
# ---------------------------------------------------------------------------
docker logs "$CONTAINER" > "$SOAK_DIR/docker.log" 2>&1 || true
CONTAINER_ALIVE="false"
if docker ps --filter "name=^/${CONTAINER}$" --format '{{.Names}}' | grep -qx "$CONTAINER"; then
  CONTAINER_ALIVE="true"
fi
docker rm -f "$CONTAINER" >/dev/null 2>&1 || true

# ---------------------------------------------------------------------------
# Gate D2.3 evaluation (same shape as Gate 2b).
# ---------------------------------------------------------------------------
ALL_PASS="true"
ANY_FAIL="false"
TOTAL_ERR=0
for line in "${RUN_RESULTS[@]}"; do
  read -r idx correct errors <<< "$line"
  if [ "$correct" = "fail" ]; then ALL_PASS="false"; ANY_FAIL="true"; continue; fi
  if [ "$correct" -lt "$GSM8K_FLOOR" ]; then ALL_PASS="false"; fi
  if [ "$errors" -gt 0 ]; then ALL_PASS="false"; fi
  TOTAL_ERR=$((TOTAL_ERR + errors))
done

CORRUPT_HITS=0
if [ -f "$SOAK_DIR/docker.log" ]; then
  CORRUPT_HITS=$(grep -cE "ERROR|FATAL|state.*corrupt" "$SOAK_DIR/docker.log" || true)
fi
[ "$CORRUPT_HITS" -gt 0 ] && ALL_PASS="false"
[ "$CONTAINER_ALIVE" != "true" ] && ALL_PASS="false"

{
  echo "{"
  echo "  \"arm\": \"$ARM\","
  echo "  \"git_sha\": \"$GIT_SHA\","
  echo "  \"image_id\": \"$IMAGE_ID\","
  echo "  \"phase_e_layers\": \"$PHASE_E_LAYERS\","
  echo "  \"phase_e_fusion\": 0,"
  echo "  \"wo_split\": $WO_SPLIT,"
  echo "  \"n_runs\": $N_RUNS,"
  echo "  \"gsm8k_floor\": $GSM8K_FLOOR,"
  echo "  \"runs\": ["
  first=1
  for line in "${RUN_RESULTS[@]}"; do
    read -r idx correct errors <<< "$line"
    [ "$first" -eq 0 ] && echo "," || true
    first=0
    if [ "$correct" = "fail" ]; then
      echo -n "    {\"run\": $idx, \"ok\": false, \"reason\": \"no_gsm8k_json\"}"
    else
      gsm8k_pass=true
      [ "$correct" -lt "$GSM8K_FLOOR" ] && gsm8k_pass=false
      [ "$errors" -gt 0 ] && gsm8k_pass=false
      echo -n "    {\"run\": $idx, \"correct\": $correct, \"errors\": $errors, \"pass\": $gsm8k_pass}"
    fi
  done
  echo
  echo "  ],"
  echo "  \"container_alive_at_end\": $CONTAINER_ALIVE,"
  echo "  \"docker_log_corruption_hits\": $CORRUPT_HITS,"
  echo "  \"gate_d2_3_pass\": $ALL_PASS"
  echo "}"
} > "$SOAK_DIR/verdict.json"

log "D2.3 soak complete (gate_d2_3_pass=$ALL_PASS)"
log "artifacts: $SOAK_DIR/"
log "next: build_soak_summary.py for the per-question miss table"

exit 0
