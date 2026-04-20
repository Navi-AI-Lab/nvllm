#!/usr/bin/env bash
# Run the Phase A PTX-diff harness inside a given nvllm image and
# capture the emitted IR/PTX/CUBIN plus an env snapshot.
#
# Usage: run_diagnostic.sh <image:tag> <host_output_dir>
# Example:
#   run_diagnostic.sh nvllm:gb10-phaseD2e /tmp/ptx_dump_d2e

set -euo pipefail

if [[ $# -ne 2 ]]; then
    echo "usage: $0 <image:tag> <host_output_dir>" >&2
    exit 2
fi

IMAGE="$1"
OUTDIR="$2"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
HARNESS="${REPO_ROOT}/docs/research/phase_a_ptx_diag/harness.py"

if [[ ! -f "$HARNESS" ]]; then
    echo "harness not found at $HARNESS" >&2
    exit 3
fi

mkdir -p "$OUTDIR"
rm -rf "$OUTDIR"/*

echo "[driver] image:    $IMAGE"
echo "[driver] outdir:   $OUTDIR"
echo "[driver] harness:  $HARNESS"

# Snapshot image metadata before running the harness.
{
    echo "# Image metadata"
    echo "image: $IMAGE"
    docker image inspect --format 'id: {{.Id}}' "$IMAGE"
    docker image inspect --format 'created: {{.Created}}' "$IMAGE"
    echo
    echo "# nvidia-cutlass-dsl version"
    docker run --rm --entrypoint bash "$IMAGE" -c \
        "python3 -c 'from importlib.metadata import version; print(version(\"nvidia-cutlass-dsl\"))'"
    echo
    echo "# vLLM git commit (if recorded)"
    docker run --rm --entrypoint bash "$IMAGE" -c \
        "cat /app/nvllm/.git/HEAD 2>/dev/null; cd /app/nvllm && git rev-parse HEAD 2>/dev/null || echo '(no git in container)'"
} > "$OUTDIR/env.txt" 2>&1 || true

# Run the harness. bind-mount OUTDIR at /workdir/ir_dump so DSL dumps land there.
docker run --rm --gpus all \
    -v "$OUTDIR:/workdir/ir_dump" \
    -v "$HARNESS:/workdir/harness.py:ro" \
    -e CUTE_DSL_KEEP_PTX=1 \
    -e CUTE_DSL_KEEP_IR=1 \
    -e CUTE_DSL_KEEP_CUBIN=1 \
    -e CUTE_DSL_DUMP_DIR=/workdir/ir_dump \
    -e CUTE_DSL_NO_CACHE=1 \
    -e CUTE_DSL_JIT_TIME_PROFILING=1 \
    -e CUTE_MLP_FUSION=1 \
    --entrypoint bash "$IMAGE" \
    -c "cd /workdir && python3 harness.py" \
    2>&1 | tee "$OUTDIR/harness_stdout.txt"

echo
echo "[driver] files emitted:"
ls -la "$OUTDIR"
