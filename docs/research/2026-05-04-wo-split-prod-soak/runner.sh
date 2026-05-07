#!/usr/bin/env bash
# Orchestrate the wo_split production soak.
#
# Default run is long (~7h): wo_split=1/2/4/8, five primary replays per arm,
# one supplementary profiler/region pass per arm. Override WO_SPLITS,
# REPLAYS, or PHASES for debug runs.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
DOC_DIR="$REPO_ROOT/docs/research/2026-05-04-wo-split-prod-soak"
OUT_DIR="${OUT_DIR:-$REPO_ROOT/benchmarks/nvllm/traces/wo_split_prod_soak/2026-05-04-soak}"
CONTAINER="${CONTAINER:-nvllm}"
PORT="${PORT:-8000}"
API="http://localhost:${PORT}/v1"

if [ -n "${WO_SPLITS:-}" ]; then
  read -r -a ARMS <<< "$WO_SPLITS"
else
  ARMS=(1 2 4 8)
fi
REPLAYS="${REPLAYS:-5}"
SEED="${SEED:-42}"
GSM8K_N="${GSM8K_N:-50}"
GSM8K_MAX_TOKENS="${GSM8K_MAX_TOKENS:-512}"
GSM8K_TIMEOUT="${GSM8K_TIMEOUT:-600}"
SHAREGPT_MAX_TOKENS="${SHAREGPT_MAX_TOKENS:-128}"
LONGDECODE_MAX_TOKENS="${LONGDECODE_MAX_TOKENS:-2048}"
CONCURRENT_MAX_TOKENS="${CONCURRENT_MAX_TOKENS:-128}"
REPLAY_TIMEOUT="${REPLAY_TIMEOUT:-900}"
# Supplementary pass runs with profiler + region timing on, which slows
# inference enough that a long prompt can blow past the primary timeout
# and starve the EngineCore. Cap by both count and length, and give each
# request a much longer ceiling.
SUPP_REPLAY_TIMEOUT="${SUPP_REPLAY_TIMEOUT:-1800}"
SUPP_LIMIT_REQUESTS="${SUPP_LIMIT_REQUESTS:-4}"
SUPP_MAX_PROMPT_CHARS="${SUPP_MAX_PROMPT_CHARS:-5500}"
PROFILER_FLUSH_SECONDS="${PROFILER_FLUSH_SECONDS:-120}"
TICK_SOURCE="${TICK_SOURCE:-globaltimer}"
PHASES="${PHASES:-primary,supplementary}"
FORCE="${FORCE:-0}"

mkdir -p "$OUT_DIR"

log() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

phase_enabled() {
  case ",$PHASES," in
    *",$1,"*) return 0 ;;
    *) return 1 ;;
  esac
}

stop_server() {
  docker logs "$CONTAINER" > "${1:-/tmp/nvllm_soak_docker.log}" 2>&1 || true
  docker rm -f "$CONTAINER" >/dev/null 2>&1 || true
}

wait_ready() {
  local log_file="$1"
  for i in $(seq 1 240); do
    if ! docker ps --filter "name=^/${CONTAINER}$" --format '{{.Names}}' \
        | grep -qx "$CONTAINER"; then
      echo "ERROR: container died while loading. See $log_file" >&2
      docker logs "$CONTAINER" >> "$log_file" 2>&1 || true
      return 1
    fi
    if curl -fsS "$API/models" >/dev/null 2>&1; then
      if curl -fsS "$API/completions" \
          -H 'Content-Type: application/json' \
          -d '{"model":"default","prompt":"warmup","max_tokens":1,"temperature":0}' \
          >/dev/null 2>&1; then
        log "server ready after ~$((i * 5))s"
        return 0
      fi
    fi
    sleep 5
  done
  echo "ERROR: server did not become ready. See $log_file" >&2
  docker logs "$CONTAINER" >> "$log_file" 2>&1 || true
  return 1
}

start_server() {
  local wo_split="$1"
  local region_timing="$2"
  local profiler="$3"
  local log_file="$4"

  docker rm -f "$CONTAINER" >/dev/null 2>&1 || true
  mkdir -p "$(dirname "$log_file")"
  : > "$log_file"

  log "starting server wo_split=$wo_split region_timing=$region_timing profiler=$profiler"
  (
    cd "$REPO_ROOT"
    CUTE_WO_SPLIT="$wo_split" \
    CUTE_PHASE_E_FUSION=1 \
    CUTE_PHASE_E_LAYERS=0,1,2,3,4,5,6,7 \
    CUTE_PHASE_E_FALLBACK_RAISE=1 \
    CUTE_BETA_REGION_TIMING="$region_timing" \
    NVLLM_BIND_MOUNT_CUTE_PAGED=1 \
    NVLLM_TORCH_PROFILER="$profiler" \
    VLLM_TORCH_PROFILER_DIR=/root/.cache/vllm/profiler \
      bash scripts/serve-cute.sh
  ) >> "$log_file" 2>&1

  wait_ready "$log_file"
}

run_gsm8k() {
  local arm_dir="$1"
  local wo_split="$2"
  local out="$arm_dir/primary/gsm8k.json"
  if [ "$FORCE" != "1" ] && [ -f "$out" ]; then
    log "skip GSM8K wo$wo_split (exists)"
    return 0
  fi
  log "GSM8K wo$wo_split"
  (
    cd "$REPO_ROOT"
    .venv/bin/python scripts/gsm8k_eval_50.py \
      --api "$API" \
      --model default \
      --n "$GSM8K_N" \
      --seed "$SEED" \
      --max-tokens "$GSM8K_MAX_TOKENS" \
      --timeout "$GSM8K_TIMEOUT" \
      --save "$out" \
      --label "wo${wo_split}_primary_gsm8k"
  ) 2>&1 | tee "$arm_dir/primary/gsm8k.log"
}

run_primary_replays() {
  local arm_dir="$1"
  local wo_split="$2"
  for idx in $(seq 1 "$REPLAYS"); do
    local run
    run="$(printf 'run%02d' "$idx")"
    local run_dir="$arm_dir/primary/$run"
    local done_file="$run_dir/DONE"
    if [ "$FORCE" != "1" ] && [ -f "$done_file" ]; then
      log "skip primary wo$wo_split $run (DONE)"
      continue
    fi
    mkdir -p "$run_dir"
    log "ShareGPT wo$wo_split $run"
    "$REPO_ROOT/.venv/bin/python" "$DOC_DIR/_replay.py" \
      --phase sharegpt \
      --api "$API" \
      --model default \
      --out-dir "$run_dir" \
      --sharegpt-slice "$DOC_DIR/sharegpt_slice.jsonl" \
      --max-tokens "$SHAREGPT_MAX_TOKENS" \
      --seed "$SEED" \
      --timeout "$REPLAY_TIMEOUT" \
      2>&1 | tee "$run_dir/sharegpt.log"

    log "Long decode wo$wo_split $run"
    "$REPO_ROOT/.venv/bin/python" "$DOC_DIR/_replay.py" \
      --phase longdecode \
      --api "$API" \
      --model default \
      --out-dir "$run_dir" \
      --longdecode-prompt "$DOC_DIR/longdecode_prompt.txt" \
      --max-tokens "$LONGDECODE_MAX_TOKENS" \
      --seed "$SEED" \
      --timeout "$REPLAY_TIMEOUT" \
      2>&1 | tee "$run_dir/longdecode.log"

    local baseline="$OUT_DIR/wo1/primary/$run/longdecode_output.txt"
    local baseline_arg=()
    if [ "$wo_split" != "1" ] && [ -f "$baseline" ]; then
      baseline_arg=(--baseline "$baseline")
    fi
    "$REPO_ROOT/.venv/bin/python" "$DOC_DIR/coherence_check.py" \
      --input "$run_dir/longdecode_output.txt" \
      "${baseline_arg[@]}" \
      --label "wo${wo_split}/${run}" \
      > "$run_dir/longdecode_coherence.json"
    touch "$done_file"
  done
}

run_concurrent_probe() {
  local arm_dir="$1"
  local wo_split="$2"
  local out_dir="$arm_dir/primary/concurrent"
  if [ "$FORCE" != "1" ] && [ -f "$out_dir/DONE" ]; then
    log "skip concurrent wo$wo_split (DONE)"
    return 0
  fi
  mkdir -p "$out_dir"
  log "2-concurrent probe wo$wo_split"
  "$REPO_ROOT/.venv/bin/python" "$DOC_DIR/_replay.py" \
    --phase concurrent \
    --api "$API" \
    --model default \
    --out-dir "$out_dir" \
    --sharegpt-slice "$DOC_DIR/sharegpt_slice.jsonl" \
    --max-tokens "$CONCURRENT_MAX_TOKENS" \
    --seed "$SEED" \
    --timeout "$REPLAY_TIMEOUT" \
    2>&1 | tee "$out_dir/concurrent.log"
  touch "$out_dir/DONE"
}

profiler_start() {
  curl -fsS -X POST "http://localhost:${PORT}/start_profile" >/dev/null
}

profiler_stop() {
  curl -fsS -X POST "http://localhost:${PORT}/stop_profile" >/dev/null
  log "profiler flush sleep ${PROFILER_FLUSH_SECONDS}s"
  sleep "$PROFILER_FLUSH_SECONDS"
}

prepare_region_dump() {
  docker exec "$CONTAINER" rm -f \
    /root/.cache/vllm/region_timings.npy /tmp/.dump_region_timings
  docker exec "$CONTAINER" touch /tmp/.dump_region_timings
}

copy_region_dump() {
  local out_npy="$1"
  for _ in $(seq 1 30); do
    if docker exec "$CONTAINER" test -f /root/.cache/vllm/region_timings.npy; then
      docker cp "$CONTAINER":/root/.cache/vllm/region_timings.npy "$out_npy"
      return 0
    fi
    sleep 2
  done
  echo "ERROR: region timing dump was not produced: $out_npy" >&2
  return 1
}

copy_profile_trace() {
  local trace_dir="$1"
  rm -rf "$trace_dir"
  mkdir -p "$trace_dir"
  docker cp "$CONTAINER":/root/.cache/vllm/profiler/. "$trace_dir/" 2>/dev/null || true
  local trace
  trace="$(find "$trace_dir" -name 'rank*.pt.trace.json.gz' | sort | head -n1 || true)"
  if [ -z "$trace" ]; then
    trace="$(find "$trace_dir" -name '*.pt.trace.json.gz' | sort | head -n1 || true)"
  fi
  if [ -z "$trace" ]; then
    echo "ERROR: no torch profiler trace found under $trace_dir" >&2
    return 1
  fi
  printf '%s\n' "$trace"
}

extract_profiler_and_regions() {
  local wo_split="$1"
  local phase_name="$2"
  local trace_dir="$3"
  local region_npy="$4"
  local kernels_csv="$5"
  local region_csv="$6"

  local trace
  trace="$(copy_profile_trace "$trace_dir")"
  (
    cd "$REPO_ROOT"
    .venv/bin/python docs/research/gemm_sweep/extract_e2e_kernels.py \
      --trace "$trace" \
      --config "wo${wo_split}_${phase_name}" \
      --out "$kernels_csv"
    .venv/bin/python docs/research/2026-05-02-beta-region-breakdown/extract_regions.py \
      --buf "$region_npy" \
      --kernels "$kernels_csv" \
      --slice-ctas 8 \
      --num-k-tiles 8 \
      --num-seqs 1 \
      --tick-source "$TICK_SOURCE" \
      --wo-split "$wo_split" \
      --num-kv-heads 4 \
      --out "$region_csv"
  )
}

run_supplementary_phase() {
  local arm_dir="$1"
  local wo_split="$2"
  local phase="$3"
  local out_dir="$arm_dir/supplementary"
  mkdir -p "$out_dir"

  local done_file="$out_dir/${phase}_DONE"
  if [ "$FORCE" != "1" ] && [ -f "$done_file" ]; then
    log "skip supplementary $phase wo$wo_split (DONE)"
    return 0
  fi

  docker exec "$CONTAINER" rm -rf /root/.cache/vllm/profiler
  docker exec "$CONTAINER" mkdir -p /root/.cache/vllm/profiler
  prepare_region_dump
  profiler_start

  # The replay command is the only step that can hang for a long time.
  # Run it with `set +e` so a read-timeout/connection failure does NOT
  # abort the runner before profiler_stop + region dump + container
  # cleanup. The original rc is preserved for the caller.
  local replay_rc=0
  set +e
  if [ "$phase" = "sharegpt" ]; then
    "$REPO_ROOT/.venv/bin/python" "$DOC_DIR/_replay.py" \
      --phase sharegpt \
      --api "$API" \
      --model default \
      --out-dir "$out_dir/sharegpt_replay" \
      --sharegpt-slice "$DOC_DIR/sharegpt_slice.jsonl" \
      --max-tokens "$SHAREGPT_MAX_TOKENS" \
      --seed "$SEED" \
      --http-timeout "$SUPP_REPLAY_TIMEOUT" \
      --limit-requests "$SUPP_LIMIT_REQUESTS" \
      --max-prompt-chars "$SUPP_MAX_PROMPT_CHARS" \
      2>&1 | tee "$out_dir/sharegpt_replay.log"
    replay_rc="${PIPESTATUS[0]}"
  elif [ "$phase" = "longdecode" ]; then
    "$REPO_ROOT/.venv/bin/python" "$DOC_DIR/_replay.py" \
      --phase longdecode \
      --api "$API" \
      --model default \
      --out-dir "$out_dir/longdecode_replay" \
      --longdecode-prompt "$DOC_DIR/longdecode_prompt.txt" \
      --max-tokens "$LONGDECODE_MAX_TOKENS" \
      --seed "$SEED" \
      --http-timeout "$SUPP_REPLAY_TIMEOUT" \
      2>&1 | tee "$out_dir/longdecode_replay.log"
    replay_rc="${PIPESTATUS[0]}"
  else
    set -e
    echo "unknown supplementary phase: $phase" >&2
    return 1
  fi
  set -e

  if [ "$replay_rc" -ne 0 ]; then
    log "WARNING: supplementary $phase wo$wo_split replay failed (rc=$replay_rc); attempting best-effort flush + extract"
  fi

  # Best-effort cleanup. None of these failing should mask replay_rc.
  profiler_stop || true

  local region_npy="$out_dir/${phase}_region_timings.npy"
  local region_csv="$out_dir/${phase}_region_timing.csv"
  local trace_name="$phase"
  if [ "$phase" = "sharegpt" ]; then
    trace_name="serve"
  fi
  local trace_dir="$out_dir/${trace_name}_trace"
  local kernels_csv="$out_dir/${phase}_profile_kernels.csv"
  copy_region_dump "$region_npy" || \
    log "WARNING: region dump copy failed for wo$wo_split $phase"
  extract_profiler_and_regions \
    "$wo_split" "$phase" "$trace_dir" "$region_npy" "$kernels_csv" "$region_csv" \
    || log "WARNING: profiler/region extraction failed for wo$wo_split $phase"

  if [ "$replay_rc" -eq 0 ]; then
    touch "$done_file"
  fi
  return "$replay_rc"
}

write_metadata() {
  local meta="$OUT_DIR/metadata.json"
  (
    cd "$REPO_ROOT"
    .venv/bin/python - <<'PY'
import json
import os
import subprocess

def sh(cmd):
    return subprocess.check_output(cmd).decode().strip()

print(json.dumps({
    "commit": sh(["git", "rev-parse", "HEAD"]),
    "branch": sh(["git", "branch", "--show-current"]),
    "image_id": sh(["docker", "image", "inspect", os.environ.get("NVLLM_IMAGE", "nvllm:gb10"), "--format", "{{.Id}}"]),
    "model": os.environ.get("HF_MODEL", "ig1/Qwen3.5-27B-NVFP4"),
    "wo_splits": os.environ.get("WO_SPLITS", "1 2 4 8"),
    "replays": int(os.environ.get("REPLAYS", "5")),
    "sharegpt_max_tokens": int(os.environ.get("SHAREGPT_MAX_TOKENS", "128")),
    "longdecode_max_tokens": int(os.environ.get("LONGDECODE_MAX_TOKENS", "2048")),
}, indent=2))
PY
  ) > "$meta"
}

main() {
  log "output: $OUT_DIR"
  export WO_SPLITS="${ARMS[*]}"
  export REPLAYS SHAREGPT_MAX_TOKENS LONGDECODE_MAX_TOKENS
  write_metadata

  # Phase 1: ALL primary passes first. These are the decision-critical
  # data (wall, TPOT, GSM8K). Supplementary diagnostics are deferred so
  # that a profiler-induced failure on one arm cannot block the verdict
  # for the others.
  if phase_enabled primary; then
    for wo in "${ARMS[@]}"; do
      local arm_dir="$OUT_DIR/wo${wo}"
      mkdir -p "$arm_dir/primary" "$arm_dir/supplementary"
      if [ "$FORCE" != "1" ] && [ -f "$arm_dir/primary_DONE" ]; then
        log "skip primary wo$wo (primary_DONE)"
        continue
      fi
      start_server "$wo" 0 0 "$arm_dir/primary/serve.log"
      run_gsm8k "$arm_dir" "$wo"
      run_primary_replays "$arm_dir" "$wo"
      run_concurrent_probe "$arm_dir" "$wo"
      stop_server "$arm_dir/primary/docker.log"
      touch "$arm_dir/primary_DONE"
    done
  fi

  # Phase 2: ALL supplementary passes. Each phase is fault-tolerant
  # (replay failure is captured, profiler/region extraction runs
  # best-effort) so one arm timing out does not block later arms.
  if phase_enabled supplementary; then
    for wo in "${ARMS[@]}"; do
      local arm_dir="$OUT_DIR/wo${wo}"
      mkdir -p "$arm_dir/supplementary"
      if [ "$FORCE" != "1" ] && [ -f "$arm_dir/supplementary_DONE" ]; then
        log "skip supplementary wo$wo (supplementary_DONE)"
        continue
      fi
      start_server "$wo" 1 1 "$arm_dir/supplementary/serve.log"
      local supp_rc=0
      run_supplementary_phase "$arm_dir" "$wo" sharegpt   || supp_rc=$?
      run_supplementary_phase "$arm_dir" "$wo" longdecode || supp_rc=$?
      stop_server "$arm_dir/supplementary/docker.log"
      if [ "$supp_rc" -eq 0 ]; then
        touch "$arm_dir/supplementary_DONE"
      else
        log "WARNING: supplementary wo$wo finished with rc=$supp_rc; not marking DONE"
      fi
    done
  fi

  (
    cd "$REPO_ROOT"
    .venv/bin/python "$DOC_DIR/parse_results.py" \
      --evidence-dir "$OUT_DIR" \
      --expected-replays "$REPLAYS" \
      --out "$OUT_DIR/summary.md"
  )
  log "done: $OUT_DIR/summary.md"
}

main "$@"
