#!/usr/bin/env bash
# Nsight Compute capture for PhaseE_Beta_Kernel — adjunct evidence for
# the per-region breakdown experiment. Reports memory throughput,
# achieved occupancy, L1/L2 hit rates, and roofline model classification
# (memory-bound vs compute-bound).
#
# Usage: bash docs/research/2026-05-02-beta-region-breakdown/run_ncu.sh
#
# Container must already be running (use scripts/serve-cute.sh first).
# This script exec's into the running container and captures ncu output
# while a single completion is in flight.

set -euo pipefail

CONTAINER="${CONTAINER:-nvllm}"
OUT_DIR="${OUT_DIR:-docs/research/2026-05-02-beta-region-breakdown/ncu}"
mkdir -p "$OUT_DIR"

# Verify container is running
if ! docker ps --format '{{.Names}}' | grep -q "^${CONTAINER}$"; then
  echo "ERROR: container '$CONTAINER' is not running. Start with"
  echo "       scripts/serve-cute.sh first." >&2
  exit 1
fi

echo "[ncu] capturing PhaseE_Beta_Kernel to $OUT_DIR/phase_e_beta.ncu-rep"
echo "[ncu] this may take 5-10 minutes; ncu replays each kernel ~50x"

# Trigger one completion in background to keep the kernel in flight
# while ncu attaches.
(
  sleep 5
  # Use served-model-name "default" (set by serve-cute.sh), not the HF id.
  curl -s -X POST http://localhost:8000/v1/completions \
    -H 'Content-Type: application/json' \
    -d '{"model":"default","prompt":"The capital of France is",
         "max_tokens":64,"temperature":0,"ignore_eos":true}' \
    > "$OUT_DIR/completion.json"
) &
COMPLETION_PID=$!

# Attach ncu inside the container
docker exec "$CONTAINER" \
  ncu --target-processes all \
      --kernel-name "regex:PhaseE_Beta_Kernel|cute_kernel" \
      --launch-count 5 \
      --section MemoryWorkloadAnalysis \
      --section ComputeWorkloadAnalysis \
      --section LaunchStats \
      --section Occupancy \
      --section SchedulerStats \
      --csv \
      --log-file /tmp/phase_e_beta_ncu.log \
      --export /tmp/phase_e_beta.ncu-rep \
      --replay-mode kernel \
      ${NCU_EXTRA_ARGS:-} || true

wait "$COMPLETION_PID" || true

# Pull artifacts out of the container
docker cp "$CONTAINER":/tmp/phase_e_beta.ncu-rep "$OUT_DIR/" 2>/dev/null || \
  echo "[ncu] WARN: no .ncu-rep produced (kernel name mismatch?)"
docker cp "$CONTAINER":/tmp/phase_e_beta_ncu.log "$OUT_DIR/" 2>/dev/null || true

# Convert to CSV for committable artifact
docker exec "$CONTAINER" \
  ncu --import /tmp/phase_e_beta.ncu-rep --csv \
  > "$OUT_DIR/phase_e_beta_ncu.csv" 2>/dev/null || \
  echo "[ncu] WARN: CSV export failed"

echo "[ncu] done. Artifacts:"
ls -la "$OUT_DIR/"
