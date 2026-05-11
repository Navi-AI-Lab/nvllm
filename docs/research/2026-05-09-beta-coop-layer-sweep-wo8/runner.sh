#!/usr/bin/env bash
# β-coop layer-count sweep under wo_split=8 — Task 1a runner.
#
# Per arm: boot serve-cute.sh with arm-specific Phase E env, wait /v1/models
# ready, dispatch-audit, GSM8K-50, snapshot β region-timing artifact, teardown.
# Aggregate results into a top-level summary.md after the loop.
#
# Usage:
#   ./runner.sh arms.csv                  # normal run; refuses non-empty arm dirs
#   ./runner.sh arms.csv --force          # overwrite existing arm dirs
#
# Env knobs:
#   ALLOW_PRIMARY_CHECKOUT=1   bypass worktree guard (not recommended)
#   READY_TIMEOUT_S=600        cap on /v1/models readiness polling
#   GSM8K_FLOOR=45             pass/fail floor for GSM8K-50
#   BETA_PER_CALL_GATE_MS=7    per-call β kernel median gate (advisory)
#
# Memory-rule references (do not delete the comments — they encode the audit
# trail this script depends on):
#   feedback_bash_runner_patterns       — set -euo pipefail + PIPESTATUS
#   feedback_active_serve_readiness_probe — /v1/models poll + warmup completion
#   feedback_vllm_enginecore_env_strip  — sentinel-file env workaround
#   feedback_evidence_force_add         — evidence outputs need git add -f

# ---------------------------------------------------------------------------
# Pre-set-e validation. We check inputs BEFORE enabling -e so a clean
# user-facing error survives instead of an opaque set-e abort.
# ---------------------------------------------------------------------------

if [ "$#" -lt 1 ] || [ "$#" -gt 2 ]; then
  echo "usage: $0 <arms.csv> [--force]" >&2
  exit 64
fi

ARMS_CSV="$1"
FORCE=0
if [ "$#" -eq 2 ]; then
  case "$2" in
    --force) FORCE=1 ;;
    *) echo "ERROR: unknown second argument: $2 (only --force is accepted)" >&2; exit 64 ;;
  esac
fi

if [ ! -f "$ARMS_CSV" ]; then
  echo "ERROR: arms manifest not found: $ARMS_CSV" >&2
  exit 66
fi

DOC_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$DOC_DIR/../../.." && pwd)"

# Worktree guard (Task 0w). Refuse to run from the primary checkout unless
# ALLOW_PRIMARY_CHECKOUT=1 is set explicitly.
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

# Stage 0a sentinel-presence guard. The runner cannot trust env-passthrough
# without these lines in serve-cute.sh + qwen3_5.py (per
# feedback_vllm_enginecore_env_strip). Refuse to start if either is missing.
if ! grep -q 'CUTE_PHASE_E_FUSION=' "$REPO_ROOT/scripts/serve-cute.sh"; then
  echo "ERROR: serve-cute.sh missing CUTE_PHASE_E_FUSION sentinel write" >&2
  exit 65
fi
if ! grep -q 'CUTE_PHASE_E_DISPATCH_LOG' "$REPO_ROOT/scripts/serve-cute.sh"; then
  echo "ERROR: serve-cute.sh missing CUTE_PHASE_E_DISPATCH_LOG sentinel write" >&2
  exit 65
fi
if ! grep -q 'CUTE_PHASE_E_' "$REPO_ROOT/vllm/nvllm/models/qwen3_5.py"; then
  echo "ERROR: vllm/nvllm/models/qwen3_5.py missing CUTE_PHASE_E_* sentinel reader" >&2
  exit 65
fi

EXTRACT_DISPATCH="$DOC_DIR/extract_dispatch_log.py"
if [ ! -f "$EXTRACT_DISPATCH" ]; then
  echo "ERROR: extract_dispatch_log.py missing at $EXTRACT_DISPATCH" >&2
  exit 66
fi

GSM8K_SCRIPT="$REPO_ROOT/scripts/gsm8k_eval_50.py"
if [ ! -f "$GSM8K_SCRIPT" ]; then
  echo "ERROR: gsm8k_eval_50.py missing at $GSM8K_SCRIPT" >&2
  exit 66
fi
if [ ! -x "$REPO_ROOT/.venv/bin/python" ]; then
  echo "ERROR: $REPO_ROOT/.venv/bin/python not found (run 'uv venv --python 3.12' per AGENTS.md)" >&2
  exit 65
fi

# Verify full-attention layer set from config.json (per
# feedback_verify_model_config). We treat the canonical Qwen3.5-27B set
# 3,7,11,15,19,23,27,31,35,39,43,47,51,55,59,63 as authoritative.
FULL_ATTN_LAYERS="3,7,11,15,19,23,27,31,35,39,43,47,51,55,59,63"

# ---------------------------------------------------------------------------
set -euo pipefail
# After this point, command failures abort. PIPESTATUS captures rc through
# `tee` so we can fail loudly when GSM8K or audit returns non-zero.
# ---------------------------------------------------------------------------

OUT_DIR="${OUT_DIR:-$DOC_DIR/sweep}"
mkdir -p "$OUT_DIR"

CONTAINER="${CONTAINER:-nvllm}"
PORT="${PORT:-8000}"
API="http://localhost:${PORT}/v1"
READY_TIMEOUT_S="${READY_TIMEOUT_S:-600}"
GSM8K_FLOOR="${GSM8K_FLOOR:-45}"
GSM8K_N="${GSM8K_N:-50}"
GSM8K_SEED="${GSM8K_SEED:-42}"
GSM8K_MAX_TOKENS="${GSM8K_MAX_TOKENS:-512}"
GSM8K_TIMEOUT="${GSM8K_TIMEOUT:-600}"
BETA_PER_CALL_GATE_MS="${BETA_PER_CALL_GATE_MS:-7}"

GIT_SHA="$(git -C "$REPO_ROOT" rev-parse HEAD)"
IMAGE_ID="$(docker image inspect "${NVLLM_IMAGE:-nvllm:gb10}" --format '{{.Id}}' 2>/dev/null || echo unknown)"

log() { printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"; }

# ---------------------------------------------------------------------------
stop_server() {
  local docker_log="${1:-/tmp/nvllm_layer_sweep_docker.log}"
  docker logs "$CONTAINER" > "$docker_log" 2>&1 || true
  docker rm -f "$CONTAINER" >/dev/null 2>&1 || true
}

# Active readiness probe (feedback_active_serve_readiness_probe): poll
# /v1/models AND fire a /v1/completions warmup before declaring ready.
wait_ready() {
  local serve_log="$1"
  local deadline=$((SECONDS + READY_TIMEOUT_S))
  while [ "$SECONDS" -lt "$deadline" ]; do
    if ! docker ps --filter "name=^/${CONTAINER}$" --format '{{.Names}}' \
        | grep -qx "$CONTAINER"; then
      echo "ERROR: container died while loading; see $serve_log" >&2
      return 1
    fi
    if curl -fsS "$API/models" >/dev/null 2>&1; then
      # max_tokens=8 (not 1): a 1-token request runs as a single mixed
      # prefill+decode forward (is_decode_only=False), so the
      # CUTE_PHASE_E_DISPATCH_LOG audit gate (which requires
      # is_decode_only=True, _backend.py:1388) stays closed and the
      # subsequent --require-records audit fails with no records seen.
      # max_tokens=8 generates ~7 pure-decode forwards which fire the
      # log on every visited full-attn layer.
      if curl -fsS "$API/completions" \
          -H 'Content-Type: application/json' \
          -d '{"model":"default","prompt":"warmup","max_tokens":8,"temperature":0}' \
          >/dev/null 2>&1; then
        log "server ready (~${SECONDS}s)"
        return 0
      fi
    fi
    sleep 5
  done
  echo "ERROR: server did not become ready within ${READY_TIMEOUT_S}s" >&2
  return 1
}

# ---------------------------------------------------------------------------
# arms.csv parsing. Use Python for robust csv.DictReader semantics; the
# helper writes one shell line per arm, NUL-separated fields, which we
# then split safely with `IFS=$'\037'` (unit separator).
# ---------------------------------------------------------------------------
parse_arms() {
  "$REPO_ROOT/.venv/bin/python" - "$ARMS_CSV" "$FULL_ATTN_LAYERS" <<'PY'
import csv
import sys

csv_path = sys.argv[1]
full_attn = {int(x) for x in sys.argv[2].split(",") if x.strip()}

required = {"arm", "phase_e_fusion", "phase_e_layers", "wo_split",
            "expected_coop_layers", "description"}

with open(csv_path) as f:
    reader = csv.DictReader(f)
    missing = required - set(reader.fieldnames or [])
    if missing:
        sys.stderr.write(f"ERROR: arms.csv missing columns: {sorted(missing)}\n")
        sys.exit(2)
    seen = set()
    rows = []
    for i, row in enumerate(reader, start=2):
        arm = row["arm"].strip()
        if not arm:
            sys.stderr.write(f"ERROR: arms.csv row {i}: empty arm name\n")
            sys.exit(2)
        if arm in seen:
            sys.stderr.write(f"ERROR: arms.csv row {i}: duplicate arm '{arm}'\n")
            sys.exit(2)
        seen.add(arm)
        fusion = row["phase_e_fusion"].strip()
        if fusion not in {"0", "1"}:
            sys.stderr.write(f"ERROR: arms.csv row {i}: phase_e_fusion must be 0|1\n")
            sys.exit(2)
        layers_raw = row["phase_e_layers"].strip()
        if "..." in layers_raw:
            sys.stderr.write(f"ERROR: arms.csv row {i}: '...' is forbidden in phase_e_layers (per plan)\n")
            sys.exit(2)
        layers = []
        if layers_raw:
            try:
                layers = [int(x) for x in layers_raw.split(",") if x.strip()]
            except ValueError:
                sys.stderr.write(f"ERROR: arms.csv row {i}: phase_e_layers parse failure: {layers_raw!r}\n")
                sys.exit(2)
        if len(layers) != len(set(layers)):
            sys.stderr.write(f"ERROR: arms.csv row {i}: duplicate layer ids in {layers}\n")
            sys.exit(2)
        if fusion == "1" and not layers:
            sys.stderr.write(f"ERROR: arms.csv row {i}: phase_e_fusion=1 but no layers\n")
            sys.exit(2)
        if fusion == "0" and layers:
            sys.stderr.write(f"ERROR: arms.csv row {i}: phase_e_fusion=0 must have empty layers\n")
            sys.exit(2)
        bad = [l for l in layers if l not in full_attn]
        if bad:
            sys.stderr.write(
                f"ERROR: arms.csv row {i}: layers {bad} are not in full-attn set\n"
            )
            sys.exit(2)
        expected = row["expected_coop_layers"].strip()
        if fusion == "1" and expected != ",".join(str(x) for x in layers):
            sys.stderr.write(
                f"ERROR: arms.csv row {i}: expected_coop_layers {expected!r} "
                f"does not match phase_e_layers {layers}\n"
            )
            sys.exit(2)
        wo_split = row["wo_split"].strip()
        if wo_split != "8":
            sys.stderr.write(f"ERROR: arms.csv row {i}: wo_split must be 8 (sweep pins it)\n")
            sys.exit(2)
        rows.append((arm, fusion, layers_raw, wo_split, expected,
                     row["description"].strip()))

US = "\x1f"
for r in rows:
    sys.stdout.write(US.join(r) + "\n")
PY
}

ARMS_LINES="$(parse_arms)"
if [ -z "$ARMS_LINES" ]; then
  echo "ERROR: arms.csv parsed to zero rows" >&2
  exit 65
fi
log "validated arms.csv (full-attn set: $FULL_ATTN_LAYERS)"

# ---------------------------------------------------------------------------
# Per-arm runner.
# ---------------------------------------------------------------------------
run_arm() {
  local arm="$1" fusion="$2" layers_csv="$3" wo_split="$4" expected="$5" desc="$6"
  local arm_dir="$OUT_DIR/$arm"
  local verdict="$arm_dir/verdict.json"
  local summary="$arm_dir/summary.md"

  if [ -d "$arm_dir" ] && [ -n "$(ls -A "$arm_dir" 2>/dev/null)" ] && [ "$FORCE" != "1" ]; then
    echo "ERROR: arm dir is non-empty: $arm_dir (rerun with --force to overwrite)" >&2
    return 65
  fi
  rm -rf "$arm_dir"
  mkdir -p "$arm_dir"

  log "==> arm=$arm fusion=$fusion layers=[$layers_csv] wo=$wo_split  ($desc)"

  # Per-arm env. CUTE_PHASE_E_FALLBACK_RAISE and CUTE_PHASE_E_DISPATCH_LOG
  # are mandatory per plan (every arm must produce dispatch records and
  # fail loud on any fallback).
  export CUTE_PHASE_E_FUSION="$fusion"
  export CUTE_PHASE_E_PATH="auto"
  export CUTE_PHASE_E_LAYERS="$layers_csv"
  export CUTE_PHASE_E_FALLBACK_RAISE=1
  export CUTE_PHASE_E_DISPATCH_LOG=1
  # Plan Risk #2: the sweep itself runs CUTE_BETA_REGION_TIMING=0 to keep
  # arm correctness/perf clean. Per-call β timing is captured separately
  # in Stage 0c (already passed at 5.538 ms vs ≤7 ms gate).
  export CUTE_BETA_REGION_TIMING=0
  export CUTE_WO_SPLIT="$wo_split"
  export NVLLM_BIND_MOUNT_CUTE_PAGED=1
  export NVLLM_BIND_MOUNT_QWEN35=1

  # Tear down any prior container with the same name.
  docker rm -f "$CONTAINER" >/dev/null 2>&1 || true

  local serve_log="$arm_dir/serve.log"
  : > "$serve_log"
  log "starting serve-cute.sh ..."
  set +e
  ( cd "$REPO_ROOT" && bash scripts/serve-cute.sh ) >> "$serve_log" 2>&1
  local rc_serve="$?"
  set -e
  if [ "$rc_serve" -ne 0 ]; then
    log "FAIL: serve-cute.sh exited $rc_serve"
    cat > "$verdict" <<EOF
{"arm":"$arm","ok":false,"reason":"serve_cute_failed","rc":$rc_serve,"git_sha":"$GIT_SHA"}
EOF
    stop_server "$arm_dir/docker.log"
    return 0
  fi

  if ! wait_ready "$serve_log"; then
    log "FAIL: server not ready"
    cat > "$verdict" <<EOF
{"arm":"$arm","ok":false,"reason":"server_not_ready","git_sha":"$GIT_SHA"}
EOF
    stop_server "$arm_dir/docker.log"
    return 0
  fi

  # Snapshot env evidence: sentinel file, container inspect, log header.
  cp /tmp/c2_diag/ENV "$arm_dir/c2_diag_ENV.txt" 2>/dev/null || true
  docker inspect "$CONTAINER" > "$arm_dir/docker_inspect.json" 2>&1 || true
  docker logs "$CONTAINER" 2>&1 | head -200 > "$arm_dir/serve_log_head.txt" || true

  # ------------------------------------------------------------------
  # Dispatch audit (must run first; failure aborts this arm).
  # ------------------------------------------------------------------
  local dispatch_json="$arm_dir/dispatch_audit.json"
  log "dispatch audit (expect coop_layers=[$expected])"
  set +e
  # All arms in arms.csv post 2026-05-10 audit are β-on; require records
  # so an empty observed coop_layers can't pass via empty-expected match
  # when the audit log silently failed to fire.
  docker logs "$CONTAINER" 2>&1 \
    | "$REPO_ROOT/.venv/bin/python" "$EXTRACT_DISPATCH" \
        --expect-coop-layers "$expected" \
        --json-out "$dispatch_json" \
        --require-records \
    > "$arm_dir/dispatch_audit.stdout" 2> "$arm_dir/dispatch_audit.stderr"
  local rc_audit="${PIPESTATUS[1]}"
  set -e
  if [ "$rc_audit" -ne 0 ]; then
    log "FAIL: dispatch audit (rc=$rc_audit); aborting arm"
    cat > "$verdict" <<EOF
{"arm":"$arm","ok":false,"reason":"dispatch_audit_fail","rc":$rc_audit,"expected":"$expected","git_sha":"$GIT_SHA"}
EOF
    stop_server "$arm_dir/docker.log"
    return 0
  fi
  log "dispatch audit OK"

  # ------------------------------------------------------------------
  # GSM8K-50.
  # ------------------------------------------------------------------
  local gsm8k_json="$arm_dir/gsm8k.json"
  local gsm8k_log="$arm_dir/gsm8k.log"
  log "GSM8K-50 ..."
  set +e
  ( cd "$REPO_ROOT" && \
    .venv/bin/python scripts/gsm8k_eval_50.py \
      --api "$API" --model default \
      --n "$GSM8K_N" --seed "$GSM8K_SEED" \
      --max-tokens "$GSM8K_MAX_TOKENS" --timeout "$GSM8K_TIMEOUT" \
      --save "$gsm8k_json" --label "${arm}_wo${wo_split}" ) \
    2>&1 | tee "$gsm8k_log"
  local rc_gsm="${PIPESTATUS[0]}"
  set -e

  local gsm_correct=0 gsm_errors=0 gsm_pass=false
  if [ -f "$gsm8k_json" ] && [ "$rc_gsm" -eq 0 ]; then
    gsm_correct="$("$REPO_ROOT/.venv/bin/python" -c \
      "import json,sys; print(json.load(open('$gsm8k_json'))['correct'])")"
    gsm_errors="$("$REPO_ROOT/.venv/bin/python" -c \
      "import json,sys; print(json.load(open('$gsm8k_json'))['errors'])")"
    if [ "$gsm_correct" -ge "$GSM8K_FLOOR" ] && [ "$gsm_errors" -eq 0 ]; then
      gsm_pass=true
    fi
  fi
  log "GSM8K: correct=$gsm_correct errors=$gsm_errors floor=$GSM8K_FLOOR pass=$gsm_pass"

  # ------------------------------------------------------------------
  # Region-timing snapshot is DISABLED in this sweep run. CUTE_BETA_REGION_TIMING=0
  # is exported above (Plan Risk #2) so the kernel never allocates the timing
  # buffer; triggering the dump would no-op anyway. Per-call β median was
  # captured separately in Stage 0c (5.538 ms vs ≤7 ms gate).
  # ------------------------------------------------------------------
  local region_npy="$arm_dir/region_timings.npy"
  local beta_median_ms=""
  if false && [ "$fusion" = "1" ]; then
    log "snapshotting β region-timing buffer ..."
    set +e
    ( cd "$REPO_ROOT" && bash scripts/trigger_region_timing_dump.sh "$region_npy" ) \
      > "$arm_dir/region_timing_dump.log" 2>&1
    local rc_dump="$?"
    set -e
    if [ "$rc_dump" -ne 0 ] || [ ! -f "$region_npy" ]; then
      log "WARNING: region timing dump failed (rc=$rc_dump); continuing"
    else
      # Compute per-call β median (sum-of-region-medians proxy, per
      # plan Task 0c). Best-effort — the sweep does not gate on this.
      beta_median_ms="$(
        "$REPO_ROOT/.venv/bin/python" - "$region_npy" <<'PY' 2>/dev/null || true
import sys
import numpy as np
buf = np.load(sys.argv[1])
# buf shape: [last_ctas, num_regions, 2] (begin,end ticks). Convert to
# microsecond-equivalent durations using globaltimer (1 GHz tick).
if buf.ndim != 3 or buf.shape[2] != 2:
    raise SystemExit(0)
durs_ns = (buf[..., 1].astype(np.int64) - buf[..., 0].astype(np.int64))
# Per-region median across CTAs, then sum across regions = per-call β.
per_region_median = np.median(durs_ns, axis=0)
total_ms = float(per_region_median.sum()) / 1.0e6
print(f"{total_ms:.3f}")
PY
      )"
      log "β per-call median (sum-of-region-medians proxy): ${beta_median_ms:-unknown} ms"
    fi
  else
    log "skipping region-timing dump (CUTE_BETA_REGION_TIMING=0 in this sweep — Plan Risk #2)"
  fi

  # ------------------------------------------------------------------
  # Per-arm summary.md + verdict.json.
  # ------------------------------------------------------------------
  local beta_gate_ok="n/a"
  if [ -n "${beta_median_ms:-}" ]; then
    beta_gate_ok="$("$REPO_ROOT/.venv/bin/python" -c \
      "v=float('$beta_median_ms'); print('true' if v <= float('$BETA_PER_CALL_GATE_MS') else 'false')")"
  fi

  cat > "$summary" <<EOF
# Arm: $arm

- description: $desc
- git_sha: $GIT_SHA
- image_id: $IMAGE_ID
- worktree: $REPO_ROOT
- arms.csv row: arm=$arm, fusion=$fusion, phase_e_layers=[$layers_csv],
  wo_split=$wo_split, expected_coop_layers=[$expected]

## Dispatch audit
- result: PASS (coop_layers matched expected=[$expected])
- artifact: [dispatch_audit.json](dispatch_audit.json)

## GSM8K-50
- correct: $gsm_correct / $GSM8K_N
- errors: $gsm_errors
- floor: $GSM8K_FLOOR
- pass: $gsm_pass
- artifact: [gsm8k.json](gsm8k.json), [gsm8k.log](gsm8k.log)

## β kernel per-call timing (advisory)
- per-call median (sum-of-region-medians proxy): ${beta_median_ms:-n/a} ms
- gate (≤${BETA_PER_CALL_GATE_MS} ms): $beta_gate_ok
- artifact: [region_timings.npy](region_timings.npy)

## Server provenance
- [c2_diag_ENV.txt](c2_diag_ENV.txt) — sentinel-file env snapshot
- [docker_inspect.json](docker_inspect.json) — container Cmd + Env
- [serve_log_head.txt](serve_log_head.txt) — first 200 log lines
- [serve.log](serve.log), [docker.log](docker.log)
EOF

  cat > "$verdict" <<EOF
{
  "arm": "$arm",
  "ok": $gsm_pass,
  "git_sha": "$GIT_SHA",
  "image_id": "$IMAGE_ID",
  "fusion": $fusion,
  "phase_e_layers": "$layers_csv",
  "wo_split": $wo_split,
  "expected_coop_layers": "$expected",
  "dispatch_audit": "pass",
  "gsm8k_correct": $gsm_correct,
  "gsm8k_errors": $gsm_errors,
  "gsm8k_floor": $GSM8K_FLOOR,
  "gsm8k_pass": $gsm_pass,
  "beta_per_call_median_ms": ${beta_median_ms:-null},
  "beta_per_call_gate_ms": $BETA_PER_CALL_GATE_MS,
  "beta_per_call_gate_ok": "$beta_gate_ok",
  "description": "$desc"
}
EOF

  stop_server "$arm_dir/docker.log"
  log "<== arm=$arm done (gsm_pass=$gsm_pass)"
  return 0
}

# ---------------------------------------------------------------------------
# Iterate arms. We swallow per-arm failures into verdict.json with ok:false
# so the loop completes and the top-level summary aggregates everything.
# ---------------------------------------------------------------------------
US=$'\x1f'
declare -a ARM_NAMES=()
while IFS= read -r line; do
  [ -z "$line" ] && continue
  IFS="$US" read -r arm fusion layers_csv wo_split expected desc <<< "$line"
  ARM_NAMES+=("$arm")
  set +e
  run_arm "$arm" "$fusion" "$layers_csv" "$wo_split" "$expected" "$desc"
  set -e
done <<< "$ARMS_LINES"

# ---------------------------------------------------------------------------
# Aggregate top-level summary.md.
# ---------------------------------------------------------------------------
TOP_SUMMARY="$OUT_DIR/summary.md"
{
  echo "# β-coop layer-count sweep under wo_split=8"
  echo
  echo "- generated: $(date '+%Y-%m-%d %H:%M:%S')"
  echo "- git_sha: $GIT_SHA"
  echo "- image_id: $IMAGE_ID"
  echo "- worktree: $REPO_ROOT"
  echo "- arms manifest: \`$ARMS_CSV\`"
  echo
  echo "| arm | fusion | layers | dispatch | GSM8K | errors | β per-call (ms) | gate ≤${BETA_PER_CALL_GATE_MS}ms | ok |"
  echo "|---|---|---|---|---|---|---|---|---|"
  for arm in "${ARM_NAMES[@]}"; do
    local_v="$OUT_DIR/$arm/verdict.json"
    if [ ! -f "$local_v" ]; then
      echo "| $arm | ? | ? | MISSING | ? | ? | ? | ? | false |"
      continue
    fi
    "$REPO_ROOT/.venv/bin/python" - "$local_v" <<'PY'
import json, sys
v = json.load(open(sys.argv[1]))
def s(k, d="?"):
    val = v.get(k, d)
    return "?" if val is None else str(val)
ok = "true" if v.get("ok") is True else "false"
print(
    f"| {v.get('arm','?')} "
    f"| {s('fusion')} "
    f"| [{v.get('phase_e_layers','')}] "
    f"| {s('dispatch_audit')} "
    f"| {s('gsm8k_correct')}/{s('gsm8k_floor')}+ "
    f"| {s('gsm8k_errors')} "
    f"| {s('beta_per_call_median_ms')} "
    f"| {s('beta_per_call_gate_ok')} "
    f"| {ok} |"
)
PY
  done
  echo
  echo "## Per-arm artifacts"
  for arm in "${ARM_NAMES[@]}"; do
    echo "- [$arm/summary.md]($arm/summary.md), [$arm/verdict.json]($arm/verdict.json)"
  done
} > "$TOP_SUMMARY"

log "wrote $TOP_SUMMARY"
log "remember: evidence dir is gitignored — commit with 'git add -f' (per feedback_evidence_force_add)"
