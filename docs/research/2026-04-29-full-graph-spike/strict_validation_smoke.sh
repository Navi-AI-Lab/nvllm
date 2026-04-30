#!/bin/bash
# Strict-validation smoke for the post-G1-detour clean rebuild.
#
# Prereq: nvllm:gb10 has been clean-rebuilt from current main so it contains:
#   - 410d59390 warmup.py prefill k_cache=/v_cache= kwargs (load-bearing
#     for cache_smoke.py + first cold serve)
#   - dd1c4e252 transformers==4.57.6 pin
#   - <fix-commit> Dockerfile delete build-time warmup steps (libcuda absent
#     at build, runtime bind-mount overlays anyway — see commit message and
#     evidence/2026-04-29-1958-strict-validation/build_failure_analysis.md)
#
# Pass criteria (from user, 2026-04-29):
#   1. image builds cleanly                       → covered by ./build success
#   2. transformers==4.57.6 inside image          → image_metadata.txt
#   3. fresh mounted host cache starts empty      → cold_host_cache_pre.txt
#   4. cold serve compiles/stores                 → cold_metrics.txt MISS+STORED >=1
#   5. warm serve hits with no relevant misses    → warm_metrics.txt HIT >=1, MISS=0
#
# Produces evidence under $EVDIR (default = $(date +%Y-%m-%d-%H%M)-strict-validation):
#   run_metadata.txt           # HEAD, host cache path
#   image_metadata.txt         # docker images, transformers version
#   cold_host_cache_pre.txt    # FILE_COUNT before cold (must be 0)
#   cold_serve_launch.log      # serve-cute-full.sh stdout/stderr
#   cold_serve.log             # docker logs from cold serve
#   cold_smoke.json            # /v1/completions response
#   cold_metrics.txt           # TIME_TO_API_READY, MISS/STORED/HIT counts
#   cold_host_cache.txt        # host cache contents AFTER cold
#   warm_serve_launch.log      # serve-cute-full.sh stdout/stderr (warm)
#   warm_serve.log             # docker logs from warm serve
#   warm_smoke.json            # /v1/completions response (warm)
#   warm_metrics.txt           # TIME_TO_API_READY, MISS/STORED/HIT counts
#   verdict.md                 # PASS/FAIL summary against pass criteria

set -uo pipefail   # NOT -e — we want to capture failures, not abort the harness

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$REPO_ROOT"

EVDIR="${EVDIR:-docs/research/2026-04-29-full-graph-spike/evidence/$(date +%Y-%m-%d-%H%M)-strict-validation}"
mkdir -p "$EVDIR"
echo "EVDIR=$EVDIR"

# Use a fresh host cache dir to prove cold serve actually populates it.
HOST_CACHE="/tmp/nvllm-strict-validation-cache-$(date +%s)"
mkdir -p "$HOST_CACHE"
export CUTE_COMPILE_HOST_CACHE_DIR="$HOST_CACHE"
{
  echo "HEAD=$(git rev-parse HEAD)"
  echo "HOST_CACHE=$HOST_CACHE"
  echo "EVDIR=$EVDIR"
  echo "STARTED=$(date -Iseconds)"
} | tee "$EVDIR/run_metadata.txt"

# ---- 1. Image metadata ----------------------------------------------------
{
  echo "=== docker images nvllm:gb10 ==="
  docker images nvllm:gb10 --format 'id={{.ID}} created={{.CreatedAt}} size={{.Size}}'
  echo "=== docker inspect ==="
  docker inspect nvllm:gb10 --format '{{.Created}} {{.Id}}'
} | tee "$EVDIR/image_metadata.txt"

# transformers version inside image (option B: NOT checking image-baked
# /opt/vllm/kernel_cache, since that path is unpopulated by design now)
docker run --rm --entrypoint /workspace/.venv/bin/python nvllm:gb10 \
  -c 'import transformers; print("transformers==", transformers.__version__)' \
  2>&1 | tee -a "$EVDIR/image_metadata.txt"

# ---- 2. Pre-cold: assert host cache starts empty -------------------------
{
  echo "=== host cache BEFORE cold (must be empty) ==="
  find "$HOST_CACHE" -type f
  echo "FILE_COUNT=$(find "$HOST_CACHE" -type f | wc -l)"
} | tee "$EVDIR/cold_host_cache_pre.txt"

# ---- 3. Cold serve --------------------------------------------------------
echo "=== Cold serve start ==="
docker rm -f nvllm 2>/dev/null
COLD_START=$(date +%s)
./scripts/serve-cute-full.sh > "$EVDIR/cold_serve_launch.log" 2>&1
echo "  serve-cute-full.sh launch returned (container detached)"

# Active poll /v1/models — up to 30 min
COLD_READY=0
COLD_END=$COLD_START
for i in $(seq 1 1800); do
  if curl -fsS http://localhost:8000/v1/models > /dev/null 2>&1; then
    COLD_READY=1
    COLD_END=$(date +%s)
    break
  fi
  sleep 1
done
COLD_TIME=$((COLD_END - COLD_START))
echo "  Cold TIME_TO_API_READY=${COLD_TIME}s  ready=$COLD_READY"

if [ "$COLD_READY" -eq 1 ]; then
  curl -sS http://localhost:8000/v1/completions \
    -H 'Content-Type: application/json' \
    -d '{"model":"default","prompt":"Hello","max_tokens":4}' \
    > "$EVDIR/cold_smoke.json"
  echo "  cold_smoke.json: $(cat "$EVDIR/cold_smoke.json" | head -c 200)"
fi

docker logs nvllm > "$EVDIR/cold_serve.log" 2>&1
{
  echo "TIME_TO_API_READY=${COLD_TIME}s"
  echo "READY=$COLD_READY"
  echo "MISS_count=$(grep -c 'CuTe disk cache MISS' "$EVDIR/cold_serve.log" || echo 0)"
  echo "STORED_count=$(grep -c 'CuTe disk cache stored' "$EVDIR/cold_serve.log" || echo 0)"
  echo "HIT_count=$(grep -c 'CuTe disk cache HIT' "$EVDIR/cold_serve.log" || echo 0)"
  echo "=== unique MISS keys ==="
  grep -oE 'MISS key=[a-f0-9]{16}' "$EVDIR/cold_serve.log" | sort -u
  echo "=== unique STORED keys ==="
  grep -oE 'stored \(native\) key=[a-f0-9]{16}' "$EVDIR/cold_serve.log" | sort -u
} | tee "$EVDIR/cold_metrics.txt"

{
  echo "=== host cache AFTER cold ==="
  find "$HOST_CACHE" -type f
  echo "FILE_COUNT=$(find "$HOST_CACHE" -type f | wc -l)"
} | tee "$EVDIR/cold_host_cache.txt"

# Stop container before warm
docker rm -f nvllm 2>/dev/null
sleep 2

# ---- 4. Warm serve --------------------------------------------------------
echo "=== Warm serve start ==="
WARM_START=$(date +%s)
./scripts/serve-cute-full.sh > "$EVDIR/warm_serve_launch.log" 2>&1

WARM_READY=0
WARM_END=$WARM_START
for i in $(seq 1 1200); do  # up to 20 min — warm baseline 261s per prior evidence
  if curl -fsS http://localhost:8000/v1/models > /dev/null 2>&1; then
    WARM_READY=1
    WARM_END=$(date +%s)
    break
  fi
  sleep 1
done
WARM_TIME=$((WARM_END - WARM_START))
echo "  Warm TIME_TO_API_READY=${WARM_TIME}s  ready=$WARM_READY"

if [ "$WARM_READY" -eq 1 ]; then
  curl -sS http://localhost:8000/v1/completions \
    -H 'Content-Type: application/json' \
    -d '{"model":"default","prompt":"Hello","max_tokens":4}' \
    > "$EVDIR/warm_smoke.json"
  echo "  warm_smoke.json: $(cat "$EVDIR/warm_smoke.json" | head -c 200)"
fi

docker logs nvllm > "$EVDIR/warm_serve.log" 2>&1
{
  echo "TIME_TO_API_READY=${WARM_TIME}s"
  echo "READY=$WARM_READY"
  echo "MISS_count=$(grep -c 'CuTe disk cache MISS' "$EVDIR/warm_serve.log" || echo 0)"
  echo "STORED_count=$(grep -c 'CuTe disk cache stored' "$EVDIR/warm_serve.log" || echo 0)"
  echo "HIT_count=$(grep -c 'CuTe disk cache HIT' "$EVDIR/warm_serve.log" || echo 0)"
  echo "=== HIT lines ==="
  grep -E 'CuTe disk cache HIT' "$EVDIR/warm_serve.log" | head -10
  echo "=== unique MISS keys (must be empty for PASS) ==="
  grep -oE 'MISS key=[a-f0-9]{16}' "$EVDIR/warm_serve.log" | sort -u
} | tee "$EVDIR/warm_metrics.txt"

docker rm -f nvllm 2>/dev/null

# ---- 5. Verdict against pass criteria ------------------------------------
COLD_HOST_PRE_COUNT=$(grep -E '^FILE_COUNT=' "$EVDIR/cold_host_cache_pre.txt" | awk -F= '{print $2}')
COLD_HOST_POST_COUNT=$(grep -E '^FILE_COUNT=' "$EVDIR/cold_host_cache.txt" | awk -F= '{print $2}')
COLD_MISS=$(grep -E '^MISS_count=' "$EVDIR/cold_metrics.txt" | awk -F= '{print $2}')
COLD_STORED=$(grep -E '^STORED_count=' "$EVDIR/cold_metrics.txt" | awk -F= '{print $2}')
WARM_HIT=$(grep -E '^HIT_count=' "$EVDIR/warm_metrics.txt" | awk -F= '{print $2}')
WARM_MISS=$(grep -E '^MISS_count=' "$EVDIR/warm_metrics.txt" | awk -F= '{print $2}')
TRANSFORMERS_LINE=$(grep '^transformers==' "$EVDIR/image_metadata.txt" | tail -1)

pass_or_fail() {
  if [ "$1" = "PASS" ]; then echo "✅ PASS"; else echo "❌ FAIL"; fi
}

verdict_image_built=$([ -s "$EVDIR/image_metadata.txt" ] && echo PASS || echo FAIL)
verdict_transformers=$(echo "$TRANSFORMERS_LINE" | grep -q '4.57.6' && echo PASS || echo FAIL)
verdict_cache_empty=$([ "${COLD_HOST_PRE_COUNT:-0}" -eq 0 ] && echo PASS || echo FAIL)
verdict_cold_compile=$([ "${COLD_MISS:-0}" -ge 1 ] && [ "${COLD_STORED:-0}" -ge 1 ] && echo PASS || echo FAIL)
verdict_warm_hit=$([ "${WARM_HIT:-0}" -ge 1 ] && [ "${WARM_MISS:-0}" -eq 0 ] && echo PASS || echo FAIL)

{
  echo "# Strict validation verdict"
  echo
  echo "**HEAD:** $(git rev-parse HEAD)"
  echo "**EVDIR:** $EVDIR"
  echo "**HOST_CACHE:** $HOST_CACHE"
  echo
  echo "## Pass criteria"
  echo
  echo "| # | Criterion | Verdict |"
  echo "|---|---|---|"
  echo "| 1 | image builds cleanly | $(pass_or_fail "$verdict_image_built") |"
  echo "| 2 | transformers==4.57.6 | $(pass_or_fail "$verdict_transformers") (\`$TRANSFORMERS_LINE\`) |"
  echo "| 3 | fresh host cache starts empty | $(pass_or_fail "$verdict_cache_empty") (pre=$COLD_HOST_PRE_COUNT) |"
  echo "| 4 | cold serve compiles+stores | $(pass_or_fail "$verdict_cold_compile") (MISS=$COLD_MISS STORED=$COLD_STORED) |"
  echo "| 5 | warm serve HIT, no relevant MISS | $(pass_or_fail "$verdict_warm_hit") (HIT=$WARM_HIT MISS=$WARM_MISS) |"
  echo
  echo "## Cold serve metrics"
  echo '```'
  cat "$EVDIR/cold_metrics.txt"
  echo '```'
  echo
  echo "## Warm serve metrics"
  echo '```'
  cat "$EVDIR/warm_metrics.txt"
  echo '```'
  echo
  echo "## Files"
  ls -la "$EVDIR" | sed 's/^/    /'
} > "$EVDIR/verdict.md"

echo
echo "=== DONE === EVDIR=$EVDIR"
echo "Verdict summary:"
grep -E '^\| [0-9] \|' "$EVDIR/verdict.md"
