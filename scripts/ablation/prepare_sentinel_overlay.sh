#!/usr/bin/env bash
# Build a sentinel-overlaid scratch checkout for run_ssm_ablation_suite.sh.
#
# Usage:
#   scripts/ablation/prepare_sentinel_overlay.sh [SCRATCH_DIR]
#
# Defaults SCRATCH_DIR to /tmp/nvllm-ssm-sentinel-patched.
# Clones the current repo HEAD into SCRATCH_DIR, applies the sentinel
# overlay patch, and verifies the marker strings landed.
#
# The runner expects $PATCHED_REPO to point at SCRATCH_DIR.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)"
SCRATCH_DIR="${1:-/tmp/nvllm-ssm-sentinel-patched}"
OVERLAY="$SCRIPT_DIR/ssm_sentinel_overlay.patch"

if [ ! -f "$OVERLAY" ]; then
  echo "ERROR: overlay patch missing: $OVERLAY" >&2
  exit 1
fi

if [ -e "$SCRATCH_DIR" ]; then
  echo "INFO: removing existing $SCRATCH_DIR"
  rm -rf "$SCRATCH_DIR"
fi

CURRENT_SHA="$(git -C "$REPO_ROOT" rev-parse HEAD)"
echo "cloning $REPO_ROOT @ $CURRENT_SHA -> $SCRATCH_DIR"
git clone --no-local "$REPO_ROOT" "$SCRATCH_DIR" >/dev/null
git -C "$SCRATCH_DIR" checkout --detach "$CURRENT_SHA" >/dev/null 2>&1

echo "applying $OVERLAY"
git -C "$SCRATCH_DIR" apply "$OVERLAY"

# Verify markers.
SSM_HITS=$(grep -c _SSM_ZERO_SENTINEL "$SCRATCH_DIR/vllm/v1/worker/utils.py" || echo 0)
KV_HITS=$(grep -c _KV_ZERO_SENTINEL "$SCRATCH_DIR/vllm/v1/core/single_type_kv_cache_manager.py" || echo 0)
if [ "$SSM_HITS" -lt 1 ] || [ "$KV_HITS" -lt 1 ]; then
  echo "ERROR: sentinel markers missing after overlay (SSM=$SSM_HITS KV=$KV_HITS)" >&2
  exit 1
fi

echo "done: PATCHED_REPO=$SCRATCH_DIR ready"
echo "next:  PATCHED_REPO=$SCRATCH_DIR scripts/ablation/run_ssm_ablation_suite.sh"
