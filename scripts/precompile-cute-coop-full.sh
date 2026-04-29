#!/bin/bash
# nvllm — Offline precompile of the β-coop FULL kernel.
#
# One-shot container that runs scripts/precompile-cute-coop-full.py inside
# nvllm:gb10 with the same B12X disk-cache plumbing as serve-cute-full.sh,
# but no port bind, no engine boot. Logs to evidence dir per spec §6.
#
# Spec: docs/superpowers/specs/2026-04-29-cute-full-compile-cache-design.md §5 Step 2.2.
#
# Usage:
#   ./scripts/precompile-cute-coop-full.sh
#
# IMPORTANT: this can run for >95 minutes on a cold cache. Always launch
# from tmux per feedback_tmux_long_jobs.

set -euo pipefail

source "$(dirname "$0")/common.sh"

CONTAINER="nvllm-precompile"
EVDIR="docs/research/2026-04-29-full-graph-spike/evidence/$(date +%Y-%m-%d-%H%M)"
mkdir -p "$EVDIR"

CUTE_COMPILE_HOST_CACHE_DIR="${CUTE_COMPILE_HOST_CACHE_DIR:-/tmp/nvllm-cute-cache}"
mkdir -p "$CUTE_COMPILE_HOST_CACHE_DIR"

nvllm_check_image
docker rm -f "$CONTAINER" 2>/dev/null || true

echo "=== Offline β-coop FULL precompile ==="
echo "  Image:       $NVLLM_IMAGE"
echo "  Cache dir:   $CUTE_COMPILE_HOST_CACHE_DIR -> /opt/vllm/kernel_cache"
echo "  Evidence:    $EVDIR"
echo "  Heartbeat:   every 5 min, look for '[β-coop compile] t=Xs alive (#N)'"
echo ""

# --entrypoint /workspace/.venv/bin/python — image ENTRYPOINT is
# python3 -m vllm.entrypoints.cli.main, which we don't want here.
docker run --rm --name "$CONTAINER" \
  --gpus all \
  --ipc=host \
  --entrypoint /workspace/.venv/bin/python \
  -v "$PWD:/workspace" -w /workspace \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  -v "$CUTE_COMPILE_HOST_CACHE_DIR:/opt/vllm/kernel_cache" \
  -e B12X_CUTE_COMPILE_DISK_CACHE=1 \
  -e B12X_CUTE_COMPILE_CACHE_DIR=/opt/vllm/kernel_cache \
  -e CUTE_PHASE_E_FUSION=1 \
  -e HF_MODEL="${HF_MODEL:-ig1/Qwen3.5-27B-NVFP4}" \
  "$NVLLM_IMAGE" \
  scripts/precompile-cute-coop-full.py 2>&1 | tee "$EVDIR/precompile_run.log"

echo ""
echo "=== Precompile finished ==="
echo "  Log:           $EVDIR/precompile_run.log"
echo "  Verify cache:  ls -la $CUTE_COMPILE_HOST_CACHE_DIR"
