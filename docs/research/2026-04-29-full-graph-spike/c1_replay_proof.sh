#!/usr/bin/env bash
# C1 gate — FULL_AND_PIECEWISE + n=1 + β-coop must reach the FULL
# dispatch branch at vllm/v1/worker/gpu/model_runner.py:1050 for at
# least one decode call.
#
# Source-of-truth: CUTE_FULL_GRAPH_PROBE log from gpu/model_runner.py.
#
# Per spec §3 / C1.

set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
TS="$(date +%Y-%m-%d-%H%M)"
EVIDENCE_DIR="$REPO_ROOT/docs/research/2026-04-29-full-graph-spike/evidence/$TS"
mkdir -p "$EVIDENCE_DIR"

echo "=== C1: FULL_AND_PIECEWISE + n=1 dispatch proof ==="

# Stop any running serve and re-launch via the spike profile.
docker rm -f nvllm 2>/dev/null || true

"$REPO_ROOT/scripts/serve-cute-full.sh" 2>&1 | tee "$EVIDENCE_DIR/c1_serve_launch.txt"

# CRITICAL: docker cp host-side spike edits BEFORE vLLM imports the
# CuTe backend. Without this, the CUTE_FULL_GRAPH_PROBE log will never
# fire (the probe code lives only on host until cp). FULL_AND_PIECEWISE
# graph compilation also takes longer than PIECEWISE — single wait
# below has a 600s ceiling.
"$REPO_ROOT/docs/research/2026-04-29-full-graph-spike/_sync_host_edits.sh" \
  2>&1 | tee "$EVIDENCE_DIR/c1_sync_host_edits.txt"

echo "Waiting for /v1/models to respond (up to 1800s — β-coop full kernel cold-compile)..."
READY=""
for i in $(seq 1 180); do
  if curl -s -o /dev/null -w "%{http_code}" http://localhost:8000/v1/models 2>/dev/null | grep -q 200; then
    echo "Ready after $((i*10))s."
    READY=1
    break
  fi
  sleep 10
done
if [ -z "$READY" ]; then
  echo "FAIL: /v1/models did not respond within 1800s"
  docker logs nvllm 2>&1 | tail -100 > "$EVIDENCE_DIR/c1_docker_logs_timeout.txt"
  exit 1
fi

# Issue a single deterministic decode prompt. Greedy, temp=0, seed=42.
curl -s http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"default","prompt":"Q: What is the capital of France?\nA:","max_tokens":16,"temperature":0,"top_p":1,"seed":42}' \
  > "$EVIDENCE_DIR/c1_response.json"

cat "$EVIDENCE_DIR/c1_response.json"

# Sleep briefly to let logs flush.
sleep 3

# Pull container logs and grep for the probe.
docker logs nvllm 2>&1 > "$EVIDENCE_DIR/c1_docker_logs.txt"
grep "CUTE_FULL_GRAPH_PROBE" "$EVIDENCE_DIR/c1_docker_logs.txt" \
  > "$EVIDENCE_DIR/c1_probe.log" || true

if [ ! -s "$EVIDENCE_DIR/c1_probe.log" ]; then
  echo "C1 FAIL — probe log empty. Either CUTE_FULL_GRAPH_PROBE wasn't"
  echo "  forwarded to container, or no decode call hit gpu/model_runner.py:1050."
  exit 1
fi

# Verify at least one line shows cg_mode=FULL. The probe formats
# CUDAGraphMode.name explicitly, so the literal token to grep is
# `cg_mode=FULL\b` (matches FULL but not FULL_AND_PIECEWISE).
if ! grep -qE "cg_mode=FULL\b" "$EVIDENCE_DIR/c1_probe.log"; then
  echo "C1 FAIL — no probe entry shows cg_mode=FULL."
  echo "  Runner downgraded the mode, or batch-1 decode never reached"
  echo "  the FULL dispatch branch."
  echo
  echo "First 5 probe entries:"
  head -5 "$EVIDENCE_DIR/c1_probe.log"
  exit 1
fi

# Verify the response is non-empty (graph wasn't empty).
PYTEXT=$("$REPO_ROOT/.venv/bin/python" -c "
import json,sys
d=json.load(open('$EVIDENCE_DIR/c1_response.json'))
t=d['choices'][0]['text']
print(repr(t.strip()))
sys.exit(0 if t.strip() else 1)
")
echo "Response text: $PYTEXT"

cat > "$EVIDENCE_DIR/c1_summary.md" <<EOF
# C1 — FULL dispatch proof — PASS

- Date: $TS
- Probe log: c1_probe.log ($(wc -l < "$EVIDENCE_DIR/c1_probe.log") entries)
- At least one entry shows cg_mode=FULL
- Response text: $PYTEXT (non-empty → graph not silently empty)

C2 may now proceed.
EOF

echo "C1 PASS — summary at $EVIDENCE_DIR/c1_summary.md"
