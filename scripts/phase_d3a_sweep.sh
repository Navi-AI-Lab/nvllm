#!/bin/bash
# Phase D3a sweep: for each named preset in the spec, run the D2e
# reference workload (4×128 tok, temperature=0, ignore_eos=true,
# CUTE_ATTN_FUSION=1 CUTE_MLP_FUSION=1, PIECEWISE graphs), capture a
# torch-profiler trace + GSM8K, and append a row to summary.md.
# Image: nvllm:gb10-phaseD3a (built in Task 4).
#
# Invoke: scripts/phase_d3a_sweep.sh
# Runtime: ~15 min per preset × 4 presets ≈ 1 hour.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
IMAGE="${NVLLM_IMAGE:-nvllm:gb10-phaseD3a}"
PORT=8000
# NVLLM_PRESETS override: space-separated preset names; defaults to all 4.
# Used by Phase A to run a single preset (gate 2) before the full sweep.
if [ -n "${NVLLM_PRESETS:-}" ]; then
  read -r -a PRESETS <<< "$NVLLM_PRESETS"
else
  PRESETS=(prefill-legacy decode-balanced decode-small decode-narrow-grid)
fi

# NVLLM_SWEEP_DIR override: absolute path to the sweep output directory.
# Phase A retargets this to a new date-stamped dir; the D3a default is
# preserved for existing D3a reproduction.
SWEEP_DIR="${NVLLM_SWEEP_DIR:-$REPO_ROOT/benchmarks/nvllm/traces/cute_paged_mlp_fusion/2026-04-19-phase-d3a-sweep}"
mkdir -p "$SWEEP_DIR"

# Port preflight: catch an already-occupied port now rather than burning
# 20 minutes per preset on the readiness loop only to find nothing bound.
if curl -sf "http://localhost:$PORT/v1/models" >/dev/null 2>&1; then
  echo "ERROR: port $PORT already serving a /v1/models endpoint — stop the occupant and re-run." >&2
  exit 1
fi

SUMMARY="$SWEEP_DIR/summary.md"
if [ ! -f "$SUMMARY" ]; then
  cat > "$SUMMARY" <<'EOF'
# Phase D3a sweep — MLP decode-tile retune results

**Spec:** `docs/superpowers/specs/2026-04-19-phase-d3a-mlp-decode-retune-design.md`
**Image:** `nvllm:gb10-phaseD3a`
**Model:** `natfii/Qwen3.5-27B-NVFP4-Opus-GB10`
**Config:** `max-model-len=65536, max-num-seqs=4, kv-cache-dtype=fp8_e4m3, cudagraph_mode=PIECEWISE, CUTE_ATTN_FUSION=1, CUTE_MLP_FUSION=1, --language-model-only, --gpu-memory-utilization 0.80`
**Workload:** 4 concurrent × 128 tok, temperature=0, ignore_eos=true.

## Per-preset results

| preset | grid @ nat=4 | CTAs | ~waves @ 96r | slices/CTA | MLP μs/call | MLP Self CUDA (s) | attn fused Self CUDA (s) | s/Q | GSM8K | verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| prefill-legacy | (8,8,4) | 256 | 2.7 | 9 | 68.58 ms | 139.36 | 32.01 | 8.7 | 8/8 | baseline (from D2e) |
EOF
fi

run_preset() {
  local preset="$1"
  local preset_dir="$SWEEP_DIR/$preset"
  mkdir -p "$preset_dir"
  local container="nvllm-phased3a-$preset"
  docker rm -f "$container" 2>/dev/null || true
  # Also drop any container named `nvllm` — the repo's compose-default name
  # and a common collision source when the sweep follows a manual dev run.
  docker rm -f nvllm 2>/dev/null || true

  echo "=== Phase D3a sweep: preset=$preset ==="
  echo "  Image:    $IMAGE"
  echo "  Output:   $preset_dir"

  local profiler_config='{"profiler":"torch","torch_profiler_dir":"/tmp/profiles","ignore_frontend":true,"delay_iterations":3,"active_iterations":30,"torch_profiler_with_stack":false,"torch_profiler_use_gzip":true}'

  docker run -d \
    --name "$container" \
    --gpus all \
    --ipc=host \
    --network host \
    --privileged \
    -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
    -v "$HOME/.cache/flashinfer:/root/.cache/flashinfer" \
    -v "$preset_dir:/tmp/profiles" \
    -e VLLM_NVFP4_GEMM_BACKEND=cutlass \
    -e VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
    -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    -e CUTE_ATTN_FUSION=1 \
    -e CUTE_MLP_FUSION=1 \
    -e CUTE_MLP_TILE="$preset" \
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
    --profiler-config "$profiler_config"

  echo "  Container up. Waiting for readiness..."
  for i in $(seq 1 240); do
    if curl -sf http://localhost:$PORT/v1/models >/dev/null 2>&1; then
      echo "  Server ready at iter=$i (~$((i * 5)) s)."
      break
    fi
    sleep 5
  done

  if ! curl -sf http://localhost:$PORT/v1/models >/dev/null 2>&1; then
    echo "  ERROR: server never ready for preset=$preset — dumping last 100 log lines." >&2
    docker logs --tail 100 "$container" >&2 || true
    docker rm -f "$container" 2>/dev/null || true
    return 1
  fi

  echo "  Warmup..."
  for w in 1 2; do
    curl -sf http://localhost:$PORT/v1/completions \
      -H "Content-Type: application/json" \
      -d '{"model":"default","prompt":"Warmup.","max_tokens":16,"temperature":0}' \
      >/dev/null || true
  done

  echo "  Starting profiler..."
  curl -sf -X POST http://localhost:$PORT/start_profile >/dev/null

  echo "  Running workload (4×128 tok)..."
  for i in 1 2 3 4; do
    curl -sf http://localhost:$PORT/v1/completions \
      -H "Content-Type: application/json" \
      -d '{"model":"default","prompt":"Count 1 to 20: 1,","max_tokens":128,"temperature":0,"ignore_eos":true}' \
      -o "$preset_dir/workload_$i.json" &
  done
  wait

  echo "  Stopping profiler..."
  curl -sf -X POST http://localhost:$PORT/stop_profile >/dev/null
  sleep 3

  echo "  GSM8K sanity check..."
  # Capture the GSM8K harness exit code into a sidecar so Task 7 can
  # distinguish "harness crashed mid-run" (non-zero exit, partial/no JSON)
  # from "model answered some wrong" (zero exit, JSON with fail count).
  set +o pipefail
  "$REPO_ROOT/.venv/bin/python" "$REPO_ROOT/scripts/gsm8k_sanity.py" \
    --api "http://localhost:$PORT/v1" --model default \
    --label "phase_d3a_$preset" \
    --save "$preset_dir/gsm8k_$preset.json" \
    2>&1 | tee "$preset_dir/gsm8k_$preset.log"
  local gsm8k_exit=${PIPESTATUS[0]}
  set -o pipefail
  echo "$gsm8k_exit" > "$preset_dir/gsm8k_$preset.exit"
  if [ "$gsm8k_exit" -ne 0 ]; then
    echo "  WARNING: GSM8K exited $gsm8k_exit — see gsm8k_$preset.log" >&2
  fi

  echo "  Collecting logs..."
  docker logs "$container" > "$preset_dir/decode_log.txt" 2>&1 || true

  echo "  Stopping container..."
  docker rm -f "$container" >/dev/null 2>&1 || true

  echo "  Artifacts in $preset_dir:"
  ls -la "$preset_dir"
  echo
}

for preset in "${PRESETS[@]}"; do
  # NVLLM_SKIP_EXISTING=1 preserves existing preset dirs (D3a reuse); unset
  # means always re-run (Phase A fresh sweep).
  if [ "${NVLLM_SKIP_EXISTING:-0}" = "1" ] \
      && [ -d "$SWEEP_DIR/$preset" ] \
      && [ -n "$(ls -A "$SWEEP_DIR/$preset" 2>/dev/null)" ]; then
    echo "=== Skipping $preset (already captured; NVLLM_SKIP_EXISTING=1) ==="
    continue
  fi
  run_preset "$preset"
done

echo "=== Sweep complete ==="
echo "Summary stub: $SUMMARY"
echo "Per-preset dirs: $SWEEP_DIR/{${PRESETS[*]}}/"
echo "Next: fill in the summary rows by reading profiler_out_0.txt in each subdir."
