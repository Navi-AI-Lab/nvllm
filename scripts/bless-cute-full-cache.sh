#!/bin/bash
# nvllm — Bless a torch.compile inductor AOT cache for FULL+β-coop production.
#
# Two-phase fail-closed flow per spec:
#   docs/superpowers/specs/2026-05-01-cute-full-cache-production-workaround-design.md
#
# Usage:
#   ./scripts/bless-cute-full-cache.sh
#   ./scripts/bless-cute-full-cache.sh --rebless
#   NVLLM_BLESS_VALIDATION_TRIALS=10 ./scripts/bless-cute-full-cache.sh
#   ./scripts/bless-cute-full-cache.sh --unsafe-trials 2  # dev only
#
# Env vars:
#   HF_MODEL                          model id (default ig1/Qwen3.5-27B-NVFP4)
#   NVLLM_BLESSED_CACHE_ROOT          host cache root (default ~/.cache/nvllm)
#   NVLLM_BLESS_VALIDATION_TRIALS     raise K (>=5)
#   CUTE_COMPILE_HOST_CACHE_DIR       CuTe DSL cache dir (default /tmp/nvllm-cute-cache)

set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
source "$REPO_ROOT/scripts/common.sh"

# Parse flags.
REBLESS=0
UNSAFE_TRIALS=""
while [ "$#" -gt 0 ]; do
  case "$1" in
    --rebless) REBLESS=1; shift ;;
    --unsafe-trials)
      shift
      UNSAFE_TRIALS="${1:-}"
      [ -z "$UNSAFE_TRIALS" ] && { echo "ERROR: --unsafe-trials needs a number" >&2; exit 1; }
      shift ;;
    -h|--help)
      sed -n '2,17p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; exit 1 ;;
  esac
done

HF_MODEL="${HF_MODEL:-ig1/Qwen3.5-27B-NVFP4}"
PY="$REPO_ROOT/.venv/bin/python"

# 1. Preflight: venv, image, GPU memory, no running container, jq.
if [ ! -x "$PY" ]; then
  echo "ERROR: $PY not found. Run uv venv per AGENTS.md." >&2; exit 1
fi
if ! command -v jq >/dev/null; then
  echo "ERROR: jq not found (apt-get install jq)." >&2; exit 1
fi
if ! command -v flock >/dev/null; then
  echo "ERROR: flock not found (util-linux)." >&2; exit 1
fi
nvllm_check_image
nvllm_refuse_if_container_exists "nvllm" || exit 1
nvllm_check_free_mem "${NVLLM_MIN_FREE_GB:-90}"

# 2. Resolve image ID + HF revision.
IMAGE_ID=$(docker image inspect "$NVLLM_IMAGE" --format '{{.Id}}')
echo "[bless] image: $NVLLM_IMAGE ($IMAGE_ID)"

HF_REVISION=$(nvllm_resolve_hf_revision "$HF_MODEL") || {
  echo "ERROR: cannot resolve HF revision for $HF_MODEL" >&2
  exit 1
}
echo "[bless] HF revision: $HF_REVISION"

# 3. Derive config_hash from the launch defaults baked into Phase 1/2 docker args.
# These MUST match scripts/serve-cute-full.sh post-Phase-2 defaults exactly.
CONFIG_HASH=$(nvllm_compute_blessed_config_hash \
  "$IMAGE_ID" "$HF_MODEL" "$HF_REVISION" \
  "fp8_e4m3" "CUTE_PAGED" "FULL_AND_PIECEWISE" "[1]" \
  1 16384 65536 \
  1 "0,1,2,3,4,5,6,7" 1 \
  0 0 0 \
  1 1)
echo "[bless] config_hash: $CONFIG_HASH"

# 4. Acquire flock on the per-config-hash lock file.
LOCK_DIR="${NVLLM_BLESSED_CACHE_ROOT:-$HOME/.cache/nvllm}/staging"
mkdir -p "$LOCK_DIR"
LOCK_FILE="$LOCK_DIR/${CONFIG_HASH}.lock"
exec 9>"$LOCK_FILE"
if ! flock -n 9; then
  echo "ERROR: another bless is in progress (lock: $LOCK_FILE)" >&2
  exit 1
fi
echo "[bless] flock acquired on $LOCK_FILE"

# 5. Refuse early if manifest exists and no --rebless.
if MANIFEST_PATH=$(nvllm_resolve_blessed_manifest "$CONFIG_HASH" 2>/dev/null); then
  if [ "$REBLESS" -ne 1 ]; then
    echo "ERROR: manifest already exists: $MANIFEST_PATH" >&2
    echo "       Re-run with --rebless to replace (atomic + archived)." >&2
    exit 1
  fi
  echo "[bless] existing manifest will be archived: $MANIFEST_PATH"
fi

# 6. Hand off to Python orchestrator.
ORCH_ARGS=(
  --config-hash "$CONFIG_HASH"
  --image-id "$IMAGE_ID"
  --hf-revision "$HF_REVISION"
)
[ "$REBLESS" -eq 1 ] && ORCH_ARGS+=(--rebless)
[ -n "$UNSAFE_TRIALS" ] && ORCH_ARGS+=(--unsafe-trials "$UNSAFE_TRIALS")

echo "[bless] handing off to .venv/bin/python scripts/bless_cute_full_cache.py …"
exec "$PY" "$REPO_ROOT/scripts/bless_cute_full_cache.py" "${ORCH_ARGS[@]}"
