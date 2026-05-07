#!/usr/bin/env bash
# Region-timing-first profile pass for wo_split=1 baseline.
#
# Goal: produce a same-day, same-bounds region_timings.npy for wo_split=1
# to head-to-head against wo8/supplementary/sharegpt_region_timings.npy.
#
# Design: copy region_timings.npy out of the container BEFORE attempting
# torch profiler stop. The npy is auto-dumped by the kernel after 64 iters,
# so it's on disk by the time replay finishes. Profiler stop is the failure
# mode that killed the wo8 engine — treat it as best-effort.
#
# Pre-conditions:
#   - .venv populated (uv venv --python 3.12 + pip install)
#   - nvllm:gb10 image present
#   - HF cache populated for ig1/Qwen3.5-27B-NVFP4
#
# Output: benchmarks/nvllm/traces/wo_split_prod_soak/2026-05-04-soak/wo1/
#         supplementary_2026-05-07/

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
DOC_DIR="$REPO_ROOT/docs/research/2026-05-04-wo-split-prod-soak"
SOAK_DIR="$REPO_ROOT/benchmarks/nvllm/traces/wo_split_prod_soak/2026-05-04-soak"
OUT_DIR="$SOAK_DIR/wo1/supplementary_2026-05-07"
CONTAINER="nvllm"
PORT=8000
API="http://localhost:${PORT}/v1"

LIMIT_REQUESTS=4
MAX_PROMPT_CHARS=5500
SHAREGPT_MAX_TOKENS=128
SEED=42

mkdir -p "$OUT_DIR" "$OUT_DIR/sharegpt_replay" "$OUT_DIR/serve_trace"
COMMIT="$(cd "$REPO_ROOT" && git rev-parse HEAD)"
echo "{\"commit\": \"$COMMIT\", \"wo_split\": 1, \"limit_requests\": $LIMIT_REQUESTS, \"max_prompt_chars\": $MAX_PROMPT_CHARS, \"sharegpt_max_tokens\": $SHAREGPT_MAX_TOKENS, \"seed\": $SEED}" > "$OUT_DIR/metadata.json"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

cleanup() {
  log "cleanup"
  docker stop "$CONTAINER" >/dev/null 2>&1 || true
  docker logs "$CONTAINER" > "$OUT_DIR/docker.log" 2>&1 || true
  docker rm -f "$CONTAINER" >/dev/null 2>&1 || true
}
trap cleanup EXIT

log "starting server: wo_split=1 region_timing=1 profiler=1"
docker rm -f "$CONTAINER" >/dev/null 2>&1 || true
(
  cd "$REPO_ROOT"
  CUTE_WO_SPLIT=1 \
  CUTE_PHASE_E_FUSION=1 \
  CUTE_PHASE_E_LAYERS=0,1,2,3,4,5,6,7 \
  CUTE_PHASE_E_FALLBACK_RAISE=1 \
  CUTE_BETA_REGION_TIMING=1 \
  NVLLM_BIND_MOUNT_CUTE_PAGED=1 \
  NVLLM_TORCH_PROFILER=1 \
  VLLM_TORCH_PROFILER_DIR=/root/.cache/vllm/profiler \
    bash scripts/serve-cute.sh
) > "$OUT_DIR/serve.log" 2>&1

# Wait for /v1/models (server readiness).
log "waiting for /v1/models"
deadline=$(( $(date +%s) + 600 ))
while :; do
  if curl -fsS "$API/models" >/dev/null 2>&1; then
    log "server ready"
    break
  fi
  if [ "$(date +%s)" -gt "$deadline" ]; then
    log "ERROR: server not ready after 600s"
    exit 1
  fi
  sleep 5
done

# Pre-clean container profiler dir.
docker exec "$CONTAINER" rm -rf /root/.cache/vllm/profiler /root/.cache/vllm/region_timings.npy >/dev/null 2>&1 || true
docker exec "$CONTAINER" mkdir -p /root/.cache/vllm/profiler

# Start torch profiler.
log "POST /start_profile"
curl -fsS -X POST "http://localhost:${PORT}/start_profile" >/dev/null

# Run bounded sharegpt replay.
log "running bounded sharegpt replay (limit=$LIMIT_REQUESTS max_prompt_chars=$MAX_PROMPT_CHARS)"
"$REPO_ROOT/.venv/bin/python" "$DOC_DIR/_replay.py" \
  --phase sharegpt \
  --api "$API" \
  --model default \
  --out-dir "$OUT_DIR/sharegpt_replay" \
  --sharegpt-slice "$DOC_DIR/sharegpt_slice.jsonl" \
  --max-tokens "$SHAREGPT_MAX_TOKENS" \
  --seed "$SEED" \
  --http-timeout 1800 \
  --limit-requests "$LIMIT_REQUESTS" \
  --max-prompt-chars "$MAX_PROMPT_CHARS" \
  > "$OUT_DIR/sharegpt_replay.log" 2>&1

log "replay finished"

# Copy region_timings.npy NOW — before any risky profiler stop.
log "copying region_timings.npy"
if docker exec "$CONTAINER" test -f /root/.cache/vllm/region_timings.npy; then
  docker cp "$CONTAINER":/root/.cache/vllm/region_timings.npy \
    "$OUT_DIR/sharegpt_region_timings.npy"
  log "region_timings.npy copied: $(ls -la "$OUT_DIR/sharegpt_region_timings.npy" | awk '{print $5}') bytes"
else
  log "WARNING: region_timings.npy not present in container"
fi

# Best-effort: stop torch profiler and copy kineto traces. Cap at 60s
# because profiler-stop killed the wo8 engine. Even if engine dies here,
# we've already extracted the npy.
log "best-effort POST /stop_profile (60s cap)"
set +e
timeout 60 curl -fsS -X POST "http://localhost:${PORT}/stop_profile" >/dev/null 2>&1
stop_rc=$?
set -e
log "stop_profile rc=$stop_rc (0=ok, non-zero=timeout/failed)"

# Allow some flush time even on failure.
sleep 30

log "copying any kineto trace files"
docker cp "$CONTAINER":/root/.cache/vllm/profiler/. "$OUT_DIR/serve_trace/" 2>/dev/null || true
ls -la "$OUT_DIR/serve_trace/" || true

log "done — outputs in $OUT_DIR"
