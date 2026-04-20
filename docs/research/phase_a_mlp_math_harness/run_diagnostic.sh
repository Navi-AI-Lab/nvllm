#!/usr/bin/env bash
# Run the Phase A MLP math harness inside a given nvllm image and
# capture the saved .pt outputs.
#
# Usage: run_diagnostic.sh <image:tag> <host_output_dir>
# Example:
#   run_diagnostic.sh nvllm:gb10-phaseD2e /tmp/mlp_math_dump_d2e

set -euo pipefail

if [[ $# -ne 2 ]]; then
    echo "usage: $0 <image:tag> <host_output_dir>" >&2
    exit 2
fi

IMAGE="$1"
OUTDIR="$2"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
HARNESS="${REPO_ROOT}/docs/research/phase_a_mlp_math_harness/harness.py"

if [[ ! -f "$HARNESS" ]]; then
    echo "harness not found at $HARNESS" >&2
    exit 3
fi

mkdir -p "$OUTDIR"
rm -rf "$OUTDIR"/*

echo "[driver] image:    $IMAGE"
echo "[driver] outdir:   $OUTDIR"
echo "[driver] harness:  $HARNESS"

# Snapshot image metadata.
{
    echo "# Image metadata"
    echo "image: $IMAGE"
    docker image inspect --format 'id: {{.Id}}' "$IMAGE"
    docker image inspect --format 'created: {{.Created}}' "$IMAGE"
    echo
    echo "# nvidia-cutlass-dsl + torch versions"
    docker run --rm --entrypoint bash "$IMAGE" -c \
        "python3 -c 'from importlib.metadata import version; import torch; print(\"cutlass-dsl\", version(\"nvidia-cutlass-dsl\")); print(\"torch\", torch.__version__)'"
    echo
    echo "# vLLM git commit"
    docker run --rm --entrypoint bash "$IMAGE" -c \
        "cat /app/nvllm/.git/HEAD 2>/dev/null; cd /app/nvllm && git rev-parse HEAD 2>/dev/null || echo '(no git in container)'"
} > "$OUTDIR/env.txt" 2>&1 || true

# Run the harness. Bind-mount OUTDIR at /workdir/out so saved .pt
# tensors land there. Intentionally do NOT set CUTE_DSL_NO_CACHE —
# we want the real production cache behavior (match serving path).
docker run --rm --gpus all \
    -v "$OUTDIR:/workdir/out" \
    -v "$HARNESS:/workdir/harness.py:ro" \
    --entrypoint bash "$IMAGE" \
    -c "cd /workdir && python3 harness.py" \
    2>&1 | tee "$OUTDIR/harness_stdout.txt"

echo
echo "[driver] files emitted:"
ls -la "$OUTDIR"
