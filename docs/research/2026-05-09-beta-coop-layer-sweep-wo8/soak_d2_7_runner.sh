#!/usr/bin/env bash
# D2.7 bisection leg: pre-PR#10 baseline soak (substrate-entry attribution).
#
# Hypothesis under test (post-D2.6):
#   D2.6 ruled out 9e3a48cd8 as the substrate (Run 4 = 11/50 with the same
#   Stage 2b family shape, but per-question signature drifted by +1
#   collapse onset and +3 recovery onset). The substrate is still
#   unidentified; the next discriminator is whether it lives within
#   PR #10's cherry-pick surface at all.
#
#   D2.7 builds from `a131443ff` — the main tip IMMEDIATELY BEFORE
#   c7614342d merged PR #10. Same 2L_3_7 production-default soak shape.
#
#   - If 5×≥45/50 (stable): substrate IS in PR #10. Cumulative-revert
#     sweep within PR #10 (D2.8 = revert b383774ad, D2.9 = revert
#     f3b4d3d09, D2.10 = revert 884b5ae34) is justified.
#   - If unstable matching Stage 2b shape: substrate PREDATES PR #10.
#     The cherry-pick sweep is dead-end. Investigation widens to older
#     code (β-coop kernel internals, base CuTe paged path, hybrid model
#     machinery, prior cherry-picks before PR #10).
#   - If unstable with a different/structural shape: substrate predates
#     PR #10 AND PR #10 was masking a different bug. Diagnosis arc forks.
#
# Same soak shape as D2.5/D2.6. Image override via NVLLM_IMAGE.
#
# Usage:
#   ./soak_d2_7_runner.sh            # 5 runs, default
#   ./soak_d2_7_runner.sh --force    # overwrite existing soak_d2_7_pre_pr10_baseline/ dir
#
# Gate D2.7: all 5 runs ≥45/50, 0 errors, container alive, no docker.log
# corruption hits. Verdict.json indicates pass/fail.

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
SOAK_DIR="$DOC_DIR/soak_d2_7_pre_pr10_baseline"
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

# Branch guard — D2.7 must run on the pre-PR#10 baseline branch.
EXPECTED_BRANCH="work/d2_7_pre_pr10_baseline"
CURRENT_BRANCH="$(git -C "$REPO_ROOT" rev-parse --abbrev-ref HEAD)"
if [ "$CURRENT_BRANCH" != "$EXPECTED_BRANCH" ] && [ "${ALLOW_BRANCH_OVERRIDE:-0}" != "1" ]; then
  echo "ERROR: refusing to run D2.7 on branch '$CURRENT_BRANCH'" >&2
  echo "       expected '$EXPECTED_BRANCH' (the pre-PR#10 baseline branch at a131443ff)" >&2
  echo "       set ALLOW_BRANCH_OVERRIDE=1 to override" >&2
  exit 66
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

# Image guard — must use the pre-PR#10 build, not the production image.
export NVLLM_IMAGE="${NVLLM_IMAGE:-nvllm:gb10-d2_7}"
if ! docker image inspect "$NVLLM_IMAGE" >/dev/null 2>&1; then
  echo "ERROR: docker image '$NVLLM_IMAGE' not found" >&2
  echo "       build it first via: docker build -f docker/Dockerfile.gb10 -t $NVLLM_IMAGE ." >&2
  echo "       in /tmp/nvllm-d2_7-build (worktree breaks setuptools-scm — clone first)" >&2
  exit 67
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
IMAGE_ID="$(docker images --format '{{.Repository}}:{{.Tag}}@{{.ID}}' "$NVLLM_IMAGE" | head -n1)"

# D2.7 config: identical to D2.5/D2.6 (2L_3_7 production default) except the
# nvllm image is built from the pre-PR#10 main tip a131443ff.
ARM="2L_3_7_d2_7_pre_pr10_baseline"
PHASE_E_LAYERS="3,7"
WO_SPLIT="8"

log() { printf '[%(%Y-%m-%d %H:%M:%S)T] %s\n' -1 "$*"; }

# ---------------------------------------------------------------------------
# Boot the server with chosen arm's env.
# ---------------------------------------------------------------------------
docker rm -f "$CONTAINER" >/dev/null 2>&1 || true

export CUTE_PHASE_E_FUSION=1
export CUTE_PHASE_E_PATH="auto"
export CUTE_PHASE_E_LAYERS="$PHASE_E_LAYERS"
export CUTE_PHASE_E_FALLBACK_RAISE=1
export CUTE_PHASE_E_DISPATCH_LOG=1
export CUTE_BETA_REGION_TIMING=0
export CUTE_WO_SPLIT="$WO_SPLIT"
export NVLLM_BIND_MOUNT_CUTE_PAGED=1
export NVLLM_BIND_MOUNT_QWEN35=1

SERVE_LOG="$SOAK_DIR/serve.log"
log "boot serve-cute.sh (arm=$ARM, image=$NVLLM_IMAGE, layers=[$PHASE_E_LAYERS], FUSION=1, PATH=auto, wo=$WO_SPLIT) ..."
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
# One-shot dispatch audit. β-coop arm — pass --expect-coop-layers="3,7" only;
# the script's inner check at extract_dispatch_log.py:150 already fail-closes
# on any β-lite contamination.
# ---------------------------------------------------------------------------
log "dispatch audit (expect coop_layers=[3,7], records present)"
set +e
docker logs "$CONTAINER" 2>&1 \
  | "$REPO_ROOT/.venv/bin/python" "$EXTRACT_DISPATCH" \
      --expect-coop-layers "3,7" \
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
# Gate D2.7 evaluation (same shape as Gate 2b).
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
  echo "  \"pre_pr10_baseline_commit\": \"a131443ff\","
  echo "  \"image\": \"$NVLLM_IMAGE\","
  echo "  \"image_id\": \"$IMAGE_ID\","
  echo "  \"phase_e_layers\": \"$PHASE_E_LAYERS\","
  echo "  \"phase_e_fusion\": 1,"
  echo "  \"phase_e_path\": \"auto\","
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
  echo "  \"gate_d2_7_pass\": $ALL_PASS"
  echo "}"
} > "$SOAK_DIR/verdict.json"

log "D2.7 soak complete (gate_d2_7_pass=$ALL_PASS)"
log "artifacts: $SOAK_DIR/"

exit 0
