#!/usr/bin/env bash
# Push the spike's host-side Python edits into a freshly-launched
# nvllm container, since the prebuilt image was baked pre-edits.
# Per `feedback_docker_bindmount` + `feedback_rebuild_guard`:
# Python-only edits do NOT require docker build.
#
# Strategy: docker cp BEFORE Python imports the relevant modules.
# `docker run -d` returns immediately; vLLM's worker imports the
# CuTe backend during model load (several seconds in). We have a
# few seconds of race-window to overwrite the .py files before
# they are imported. NO restart required — the model only loads
# once, with the new code already in place.
#
# Usage (called from gate scripts IMMEDIATELY after `docker run -d`
# returns, BEFORE waiting for /v1/models):
#   ./_sync_host_edits.sh
#
# This will:
#   1. Wait briefly for the container to be running (not for /v1/models).
#   2. docker cp the edited files into the container.
#   3. Delete the relevant __pycache__/*.pyc.
#   4. Return immediately. Caller waits for /v1/models afterward.

set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"

# Wait for the container to be up enough to docker exec into (docker run -d
# returns instantly, but the container's filesystem may not be addressable
# for ~1s). Retry 30 times × 1s.
echo "[sync] waiting for container to be addressable..."
for i in $(seq 1 30); do
  if docker exec nvllm test -d /app/nvllm 2>/dev/null; then
    echo "[sync] container addressable after ${i}s"
    break
  fi
  if [ "$i" = "30" ]; then
    echo "[sync] FAIL: container not addressable after 30s" >&2
    docker ps --filter name=nvllm --format '{{.Names}} {{.Status}}' >&2
    exit 1
  fi
  sleep 1
done

# Probe the container's actual install path (editable install normally puts
# the package at /app/nvllm/vllm via the COPY at Dockerfile.gb10:48 + the
# `pip install -e .` at L79, with a .pth pointing to it under
# /usr/local/lib/python3.12/dist-packages/). Push to BOTH paths to be safe;
# whichever is the live one will see the change.
TARGETS=()
for cp in "/app/nvllm/vllm" "/usr/local/lib/python3.12/dist-packages/vllm"; do
  if docker exec nvllm test -d "$cp" 2>/dev/null; then
    TARGETS+=("$cp")
  fi
done

if [ ${#TARGETS[@]} -eq 0 ]; then
  echo "FAIL: neither /app/nvllm/vllm nor dist-packages/vllm found in container"
  exit 1
fi

for t in "${TARGETS[@]}"; do
  echo "[sync] docker cp _backend.py → $t/v1/attention/backends/cute_paged/_backend.py"
  docker cp "$REPO_ROOT/vllm/v1/attention/backends/cute_paged/_backend.py" \
    "nvllm:$t/v1/attention/backends/cute_paged/_backend.py"
  echo "[sync] docker cp gpu/model_runner.py → $t/v1/worker/gpu/model_runner.py"
  docker cp "$REPO_ROOT/vllm/v1/worker/gpu/model_runner.py" \
    "nvllm:$t/v1/worker/gpu/model_runner.py"
  # gpu_model_runner.py (flat path) is the default V1 runner used by this
  # spike. The subdir path above is the V2 runner. Copy both so whichever
  # path the engine selects sees the probe. Ported 2026-04-30 — see
  # feedback_verify_model_class.
  echo "[sync] docker cp gpu_model_runner.py → $t/v1/worker/gpu_model_runner.py"
  docker cp "$REPO_ROOT/vllm/v1/worker/gpu_model_runner.py" \
    "nvllm:$t/v1/worker/gpu_model_runner.py"
  # cache-cache branch additions: disk_cache.py + phase_e_kernel.py.
  # Without these, Gate G2 cannot positively verify HIT lines, and the
  # heartbeat / compile_only kwarg from Tasks 7+8 are absent at serve.
  echo "[sync] docker cp disk_cache.py → $t/v1/attention/backends/cute_paged/disk_cache.py"
  docker cp "$REPO_ROOT/vllm/v1/attention/backends/cute_paged/disk_cache.py" \
    "nvllm:$t/v1/attention/backends/cute_paged/disk_cache.py"
  echo "[sync] docker cp phase_e_kernel.py → $t/v1/attention/backends/cute_paged/phase_e_kernel.py"
  docker cp "$REPO_ROOT/vllm/v1/attention/backends/cute_paged/phase_e_kernel.py" \
    "nvllm:$t/v1/attention/backends/cute_paged/phase_e_kernel.py"
  # v2 patch: new wo_output reset op + qwen3_5 side-effect import
  echo "[sync] docker cp _wo_output_reset_op.py → $t/v1/attention/backends/cute_paged/_wo_output_reset_op.py"
  docker cp "$REPO_ROOT/vllm/v1/attention/backends/cute_paged/_wo_output_reset_op.py" \
    "nvllm:$t/v1/attention/backends/cute_paged/_wo_output_reset_op.py"
  echo "[sync] docker cp qwen3_5.py → $t/nvllm/models/qwen3_5.py"
  docker cp "$REPO_ROOT/vllm/nvllm/models/qwen3_5.py" \
    "nvllm:$t/nvllm/models/qwen3_5.py"
done

echo "[sync] deleting stale pyc"
docker exec nvllm bash -c '
for d in \
  /app/nvllm/vllm/v1/attention/backends/cute_paged/__pycache__ \
  /app/nvllm/vllm/v1/worker/gpu/__pycache__ \
  /app/nvllm/vllm/v1/worker/__pycache__ \
  /usr/local/lib/python3.12/dist-packages/vllm/v1/attention/backends/cute_paged/__pycache__ \
  /usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu/__pycache__ \
  /usr/local/lib/python3.12/dist-packages/vllm/v1/worker/__pycache__ \
  /app/nvllm/vllm/nvllm/models/__pycache__ \
  /usr/local/lib/python3.12/dist-packages/vllm/nvllm/models/__pycache__ ; do
  [ -d "$d" ] && find "$d" -name "*.pyc" -delete
done
true
'

# Verify markers landed.
for t in "${TARGETS[@]}"; do
  if ! docker exec nvllm grep -q "_PHASE_E_FALLBACK_RAISE" \
       "$t/v1/attention/backends/cute_paged/_backend.py"; then
    echo "FAIL: _PHASE_E_FALLBACK_RAISE marker missing in $t after docker cp"
    exit 1
  fi
  if ! docker exec nvllm grep -q "CUTE_FULL_GRAPH_PROBE" \
       "$t/v1/worker/gpu/model_runner.py"; then
    echo "FAIL: CUTE_FULL_GRAPH_PROBE marker missing in $t after docker cp"
    exit 1
  fi
  if ! docker exec nvllm grep -q "CUTE_FULL_GRAPH_PROBE" \
       "$t/v1/worker/gpu_model_runner.py"; then
    echo "FAIL: CUTE_FULL_GRAPH_PROBE marker missing in $t/v1/worker/gpu_model_runner.py after docker cp"
    exit 1
  fi
  if ! docker exec nvllm grep -q "CUTE_DISPATCH_AUDIT" \
       "$t/v1/worker/gpu_model_runner.py"; then
    echo "FAIL: CUTE_DISPATCH_AUDIT marker missing in $t/v1/worker/gpu_model_runner.py after docker cp"
    exit 1
  fi
  # cache-cache markers
  if ! docker exec nvllm grep -q "CuTe disk cache HIT" \
       "$t/v1/attention/backends/cute_paged/disk_cache.py"; then
    echo "FAIL: 'CuTe disk cache HIT' marker missing in $t after docker cp"
    exit 1
  fi
  if ! docker exec nvllm grep -q "_coop_full_compile_heartbeat" \
       "$t/v1/attention/backends/cute_paged/phase_e_kernel.py"; then
    echo "FAIL: _coop_full_compile_heartbeat marker missing in $t after docker cp"
    exit 1
  fi
  # v2 patch sentinels
  if ! docker exec nvllm test -f \
       "$t/v1/attention/backends/cute_paged/_wo_output_reset_op.py"; then
    echo "FAIL: _wo_output_reset_op.py missing in $t after docker cp"
    exit 1
  fi
  if ! docker exec nvllm grep -q "_wo_output_reset_op" \
       "$t/nvllm/models/qwen3_5.py"; then
    echo "FAIL: '_wo_output_reset_op' import marker missing in $t/nvllm/models/qwen3_5.py after docker cp"
    exit 1
  fi
done

echo "[sync] done — caller is responsible for waiting on /v1/models"
