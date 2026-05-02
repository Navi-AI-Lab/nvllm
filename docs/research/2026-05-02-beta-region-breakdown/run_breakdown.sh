#!/usr/bin/env bash
# β-coop per-region timing breakdown — single-leg orchestrator.
#
# Two boots:
#   1. Profile boot: CUTE_BETA_REGION_TIMING=1 + torch profiler.
#      Captures region_timings.npy and profile_kernels.csv.
#   2. Sanity boot: CUTE_BETA_REGION_TIMING=0 + GSM8K-50.
#      Confirms timing-off production path still passes 47/50.

set -euo pipefail

OUT_DIR="benchmarks/nvllm/traces/cute_paged_attn/2026-05-02-beta-region-breakdown"
mkdir -p "$OUT_DIR/ncu"

CONTAINER="${CONTAINER:-nvllm}"
HF_MODEL="${HF_MODEL:-ig1/Qwen3.5-27B-NVFP4}"

# ---- Boot 1: Profile boot (timing on) -------------------------------------
echo "=== Boot 1: profile boot (CUTE_BETA_REGION_TIMING=1) ==="

if [ ! -f "$OUT_DIR/profile_DONE" ]; then
  CUTE_BETA_REGION_TIMING=1 \
    bash scripts/serve-cute.sh \
    > "$OUT_DIR/profile_serve.log" 2>&1 &
  SERVE_PID=$!

  # Active readiness probe — poll /v1/models then issue completion
  for i in $(seq 1 60); do
    if curl -fsS http://localhost:8000/v1/models > /dev/null 2>&1; then
      echo "[boot] ready after ${i}s"
      break
    fi
    sleep 5
  done

  # Trigger profiler ON via vLLM endpoint
  curl -s -X POST http://localhost:8000/start_profile \
    -H 'Content-Type: application/json' || true

  # Run a small burst of completions for steady-state
  for i in 1 2 3; do
    curl -s -X POST http://localhost:8000/v1/completions \
      -H 'Content-Type: application/json' \
      -d "{\"model\":\"$HF_MODEL\",\"prompt\":\"capital of france is\",
           \"max_tokens\":50,\"temperature\":0,\"ignore_eos\":true}" \
      > /dev/null
  done

  # Stop profiler and let it flush
  curl -s -X POST http://localhost:8000/stop_profile \
    -H 'Content-Type: application/json' || true
  sleep 120

  # Dump region_timings buffer to disk via the sentinel-file path
  # wired in Task 5b (scripts/trigger_region_timing_dump.sh writes the
  # sentinel + docker cp's the .npy out). One-line, no Python wrapper.
  bash scripts/trigger_region_timing_dump.sh \
    "$OUT_DIR/region_timings.npy"

  # Adjunct: NCU capture while one more completion is in flight
  # NB: env var must be set BEFORE bash, not after — `bash run_ncu.sh
  #     OUT_DIR=...` would pass OUT_DIR=... as $1 (positional), not env.
  OUT_DIR="$OUT_DIR/ncu" \
    bash docs/research/2026-05-02-beta-region-breakdown/run_ncu.sh

  # Pull profile trace + extract kernels CSV using the existing extractor
  # at docs/research/gemm_sweep/extract_e2e_kernels.py (column name is
  # kernel_symbol, not "Kernel Name"). Requires --config to label rows.
  docker cp "$CONTAINER":/root/.cache/vllm/profiler/. "$OUT_DIR/" 2>/dev/null || true
  TRACE_FILE=$(ls "$OUT_DIR"/*.pt.trace.json.gz | head -1)
  if [ -z "$TRACE_FILE" ]; then
    echo "ERROR: no .pt.trace.json.gz found in $OUT_DIR — torch profiler"
    echo "       did not flush. Inspect $OUT_DIR/profile_serve.log."
    exit 1
  fi
  .venv/bin/python docs/research/gemm_sweep/extract_e2e_kernels.py \
    --trace "$TRACE_FILE" \
    --config "beta_region_breakdown_lower8" \
    --out "$OUT_DIR/profile_kernels.csv"

  # Metadata (use .venv/bin/python per AGENTS.md — system python is forbidden)
  .venv/bin/python -c "
import json, hashlib, subprocess, os
git_sha = subprocess.check_output(['git','rev-parse','HEAD']).decode().strip()
img_id = subprocess.check_output(['docker','image','inspect','nvllm:gb10','--format','{{.Id}}']).decode().strip()
print(json.dumps({
  'date': '2026-05-02',
  'commit': git_sha,
  'image_id': img_id,
  'model': '$HF_MODEL',
  'env': {'CUTE_BETA_REGION_TIMING': '1'},
}, indent=2))
" > "$OUT_DIR/metadata.json"

  docker stop "$CONTAINER" || true
  touch "$OUT_DIR/profile_DONE"
fi

# ---- Boot 2: Sanity boot (timing off) -------------------------------------
echo "=== Boot 2: sanity boot (CUTE_BETA_REGION_TIMING=0, GSM8K-50) ==="

if [ ! -f "$OUT_DIR/sanity_DONE" ]; then
  CUTE_BETA_REGION_TIMING=0 \
    bash scripts/serve-cute.sh \
    > "$OUT_DIR/sanity_serve.log" 2>&1 &
  SERVE_PID=$!

  for i in $(seq 1 60); do
    if curl -fsS http://localhost:8000/v1/models > /dev/null 2>&1; then break; fi
    sleep 5
  done

  # gsm8k_eval_50.py CLI: --api (not --base-url), --save (not --out),
  # --model "default" (matches serve-cute.sh's --served-model-name).
  # Use .venv/bin/python per AGENTS.md (no bare python).
  .venv/bin/python scripts/gsm8k_eval_50.py \
    --api http://localhost:8000/v1 \
    --model default \
    --save "$OUT_DIR/sanity_gsm8k.json"

  docker stop "$CONTAINER" || true
  touch "$OUT_DIR/sanity_DONE"
fi

# ---- Reduce ---------------------------------------------------------------
echo "=== Reducing region_timings.npy ==="
TICK_SOURCE="${TICK_SOURCE:-globaltimer}"  # set to clock64 if Task 2 falls back
.venv/bin/python docs/research/2026-05-02-beta-region-breakdown/extract_regions.py \
  --buf "$OUT_DIR/region_timings.npy" \
  --kernels "$OUT_DIR/profile_kernels.csv" \
  --slice-ctas 8 \
  --num-k-tiles 8 \
  --num-seqs 1 \
  --tick-source "$TICK_SOURCE" \
  --out "$OUT_DIR/region_breakdown.csv"

echo ""
echo "=== Done ==="
echo "Artifacts under $OUT_DIR/"
echo "Next: write summary.md against $OUT_DIR/region_breakdown.csv"
