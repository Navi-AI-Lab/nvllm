#!/usr/bin/env bash
# Trigger a one-shot dump of _phase_e_coop_region_timing → numpy file.
#
# Writes the sentinel /tmp/.dump_region_timings inside the container, then
# fires a one-token completion to ensure another forward() runs and
# observes the sentinel. Then docker-cp's the resulting .npy out.

set -euo pipefail

CONTAINER="${CONTAINER:-nvllm}"
OUT="${1:-./region_timings.npy}"

docker exec "$CONTAINER" touch /tmp/.dump_region_timings

# Force a forward() — one completion is enough. Use served-model-name
# "default" (set by serve-cute.sh), NOT the HF id — the OpenAI-compat
# server resolves model-name against served names.
curl -s -X POST http://localhost:8000/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"default","prompt":"x",
       "max_tokens":1,"temperature":0,"ignore_eos":true}' \
  > /dev/null

# Wait briefly for the dump to complete (np.save is sync but the request
# may return before the next forward observes the sentinel).
sleep 2

docker cp "$CONTAINER":/root/.cache/vllm/region_timings.npy "$OUT"
echo "[dump] wrote $OUT ($(stat -c%s "$OUT") bytes)"
