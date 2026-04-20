#!/usr/bin/env bash
# Same as run_q2_repeat.sh but with CUTE_*_FUSION=0 env vars —
# disables both the fused MLP and fused attention paths so we exercise
# only upstream vLLM's standard NVFP4 GEMM + unfused attention. If
# Q2 x5 is deterministic here, the non-determinism is isolated to the
# fused CuTe kernel's atomicAdd/cross-CTA sync path.
#
# Usage: run_q2_repeat_unfused.sh <image:tag> <host_output_dir> [N]

set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
    echo "usage: $0 <image:tag> <host_output_dir> [N]" >&2
    exit 2
fi

IMAGE="$1"
OUTDIR="$2"
N="${3:-5}"
CONTAINER="nvllm-q2-repeat-unfused"
PORT=8000

mkdir -p "$OUTDIR"
rm -rf "$OUTDIR"/*

echo "[driver] image: $IMAGE  outdir: $OUTDIR  repeat: $N  FUSION=OFF"

if curl -sf "http://localhost:$PORT/v1/models" >/dev/null 2>&1; then
    echo "[driver] ERROR: port $PORT already serving" >&2
    exit 3
fi
docker rm -f "$CONTAINER" nvllm 2>/dev/null || true

{
    echo "image: $IMAGE"
    echo "fusion: OFF (CUTE_MLP_FUSION=0 CUTE_ATTN_FUSION=0)"
    docker image inspect --format 'id: {{.Id}}' "$IMAGE"
} > "$OUTDIR/env.txt" 2>&1 || true

echo "[driver] launching server (fusion disabled)..."
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
    -e CUTE_ATTN_FUSION=0 \
    -e CUTE_MLP_FUSION=0 \
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

echo "[driver] warmup..."
for w in 1 2; do
    curl -sf "http://localhost:$PORT/v1/completions" \
        -H "Content-Type: application/json" \
        -d '{"model":"default","prompt":"Warmup.","max_tokens":16,"temperature":0}' \
        >/dev/null || true
done

Q2_PROMPT='Q: Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?\nA: 50 min = 50/60 hours. 12 * 50/60 ='

echo "[driver] firing Q2 x$N in same session (unfused)..."
: > "$OUTDIR/q2_results.jsonl"
for i in $(seq 1 "$N"); do
    resp="$(curl -sf "http://localhost:$PORT/v1/completions" \
        -H "Content-Type: application/json" \
        -d "$(jq -nc --arg p "$Q2_PROMPT" \
            '{model:"default", prompt:$p, max_tokens:16, temperature:0}')" || true)"
    text="$(echo "$resp" | jq -r '.choices[0].text // "<ERR>"')"
    echo "[driver]   run $i: $(echo "$text" | head -c 80)"
    echo "$resp" | jq -c --arg run "$i" '. + {run: $run | tonumber}' >> "$OUTDIR/q2_results.jsonl"
done

echo "[driver] collecting docker logs..."
docker logs "$CONTAINER" > "$OUTDIR/decode_log.txt" 2>&1 || true

echo "[driver] teardown..."
docker rm -f "$CONTAINER" >/dev/null 2>&1 || true

echo
echo "[driver] Q2 raw outputs (deduped):"
jq -r '.choices[0].text' "$OUTDIR/q2_results.jsonl" | sort -u | sed 's/^/  /'
