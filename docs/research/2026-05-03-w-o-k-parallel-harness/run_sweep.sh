#!/usr/bin/env bash
#
# W_O K-parallel scaling sweep. Runs run_harness.py for each
# wo_split in {1, 2, 4, 8} inside the nvllm:gb10 container.
#
# Outputs land under
#   benchmarks/nvllm/traces/cute_paged_attn/<DATE>-w-o-k-parallel-harness/
#     variant_4cta_scratchpad/    (wo_split=1)
#     variant_8cta_scratchpad/    (wo_split=2)
#     variant_16cta_scratchpad/   (wo_split=4)
#     variant_32cta_scratchpad/   (wo_split=8)
#
# Disk-cache JIT artefacts persist in /tmp/cute_harness_cache_v3 on
# the host (bind-mounted) so subsequent runs HIT cache.
#
# Override LAUNCHES via env (default 50). Example:
#   LAUNCHES=3 bash run_sweep.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
DATE="$(date +%Y-%m-%d)"
OUT_BASE="$REPO_ROOT/benchmarks/nvllm/traces/cute_paged_attn/$DATE-w-o-k-parallel-harness"
mkdir -p "$OUT_BASE"
mkdir -p "/tmp/cute_harness_cache_v3"

LAUNCHES="${LAUNCHES:-50}"

# Resolve git sha on the HOST (the container does not have git
# installed). Pass via env to the harness so config.json records it.
GIT_SHA="$(git -C "$REPO_ROOT" rev-parse HEAD 2>/dev/null || echo unknown)"
IMAGE_ID="$(docker inspect nvllm:gb10 --format '{{.Id}}' 2>/dev/null || echo unknown)"

echo "Sweep config:"
echo "  REPO_ROOT  = $REPO_ROOT"
echo "  OUT_BASE   = $OUT_BASE"
echo "  LAUNCHES   = $LAUNCHES"
echo "  DATE       = $DATE"
echo "  GIT_SHA    = $GIT_SHA"
echo "  IMAGE_ID   = $IMAGE_ID"
echo

for WS in 1 2 4 8; do
  TOTAL=$((4 * WS))
  OUT_DIR="$OUT_BASE/variant_${TOTAL}cta_scratchpad"
  mkdir -p "$OUT_DIR"
  echo "=== wo_split=$WS  total_wo_ctas=$TOTAL  launches=$LAUNCHES ==="

  docker run --rm --gpus all \
    -v "$REPO_ROOT:/work" \
    -v "$REPO_ROOT:/app/nvllm" \
    -v "/tmp/cute_harness_cache_v3:/tmp/cute_harness_cache_v3" \
    -e "NVLLM_HARNESS_GIT_SHA=$GIT_SHA" \
    -e "NVLLM_HARNESS_IMAGE_ID=$IMAGE_ID" \
    --entrypoint /opt/venv/bin/python \
    nvllm:gb10 \
    /work/docs/research/2026-05-03-w-o-k-parallel-harness/run_harness.py \
    --wo-split "$WS" \
    --launches "$LAUNCHES" \
    --out "/work/benchmarks/nvllm/traces/cute_paged_attn/$DATE-w-o-k-parallel-harness/variant_${TOTAL}cta_scratchpad"
done

echo
echo "Sweep complete. Artifacts in $OUT_BASE"
ls -la "$OUT_BASE"
