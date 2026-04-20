#!/usr/bin/env bash
# Reproduce the Phase A gate-2 Q2 math break by serving Qwen3.5-27B
# NVFP4 under a given image, running scripts/gsm8k_sanity.py, and
# capturing the Q2 output.
#
# Usage: run_repro.sh <image:tag> <host_output_dir>
# Example:
#   run_repro.sh nvllm:gb10-phaseD2e $PWD/benchmarks/.../d2e

set -euo pipefail

if [[ $# -ne 2 ]]; then
    echo "usage: $0 <image:tag> <host_output_dir>" >&2
    exit 2
fi

IMAGE="$1"
OUTDIR="$2"
CONTAINER="nvllm-phase-a-repro"
PORT=8000

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
GSM8K_SCRIPT="${REPO_ROOT}/scripts/gsm8k_sanity.py"

mkdir -p "$OUTDIR"
rm -rf "$OUTDIR"/*

echo "[driver] image:    $IMAGE"
echo "[driver] outdir:   $OUTDIR"

# Preflight — port free, no stale container.
if curl -sf "http://localhost:$PORT/v1/models" >/dev/null 2>&1; then
    echo "[driver] ERROR: port $PORT already serving" >&2
    exit 3
fi
docker rm -f "$CONTAINER" nvllm 2>/dev/null || true

# Snapshot image metadata.
{
    echo "# Image metadata"
    echo "image: $IMAGE"
    docker image inspect --format 'id: {{.Id}}' "$IMAGE"
    docker image inspect --format 'created: {{.Created}}' "$IMAGE"
} > "$OUTDIR/env.txt" 2>&1 || true

echo "[driver] launching server..."
docker run -d \
    --name "$CONTAINER" \
    --gpus all \
    --ipc=host \
    --network host \
    --privileged \
    -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
    -v "$HOME/.cache/flashinfer:/root/.cache/flashinfer" \
    -e VLLM_NVFP4_GEMM_BACKEND=cutlass \
    -e VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
    -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    -e CUTE_ATTN_FUSION=1 \
    -e CUTE_MLP_FUSION=1 \
    "$IMAGE" \
    serve \
    --model natfii/Qwen3.5-27B-NVFP4-Opus-GB10 \
    --served-model-name default \
    --host 0.0.0.0 --port "$PORT" \
    --kv-cache-dtype fp8_e4m3 \
    --attention-backend CUTE_PAGED \
    --max-model-len 65536 \
    --max-num-seqs 4 \
    --language-model-only \
    --mamba-cache-mode align \
    --trust-remote-code \
    --gpu-memory-utilization 0.80 \
    --max-num-batched-tokens 65536 \
    --compilation-config '{"cudagraph_mode":"PIECEWISE"}' \
    >"$OUTDIR/container_id.txt" 2>&1

echo "[driver] waiting for readiness (up to 15 min)..."
ready=0
for i in $(seq 1 180); do
    if curl -sf "http://localhost:$PORT/v1/models" >/dev/null 2>&1; then
        ready=1
        echo "[driver] ready at iter=$i (~$((i * 5)) s)"
        break
    fi
    sleep 5
done

if (( ready == 0 )); then
    echo "[driver] ERROR: server never ready" >&2
    docker logs --tail 200 "$CONTAINER" > "$OUTDIR/startup_fail.log" 2>&1 || true
    docker rm -f "$CONTAINER" >/dev/null 2>&1 || true
    exit 4
fi

# Warmup — match D3a investigation pattern.
echo "[driver] warmup..."
for w in 1 2; do
    curl -sf "http://localhost:$PORT/v1/completions" \
        -H "Content-Type: application/json" \
        -d '{"model":"default","prompt":"Warmup.","max_tokens":16,"temperature":0}' \
        >/dev/null || true
done

# GSM8K sanity.
echo "[driver] running GSM8K sanity..."
set +o pipefail
"$REPO_ROOT/.venv/bin/python" "$GSM8K_SCRIPT" \
    --api "http://localhost:$PORT/v1" \
    --model default \
    --label "phase_a_repro_$(basename "$IMAGE")" \
    --save "$OUTDIR/gsm8k.json" \
    2>&1 | tee "$OUTDIR/gsm8k.log"
gsm8k_exit=${PIPESTATUS[0]}
set -o pipefail
echo "$gsm8k_exit" > "$OUTDIR/gsm8k.exit"

# Collect logs.
echo "[driver] collecting docker logs..."
docker logs "$CONTAINER" > "$OUTDIR/decode_log.txt" 2>&1 || true

# Teardown.
echo "[driver] teardown..."
docker rm -f "$CONTAINER" >/dev/null 2>&1 || true

echo
echo "[driver] artifacts:"
ls -la "$OUTDIR"
