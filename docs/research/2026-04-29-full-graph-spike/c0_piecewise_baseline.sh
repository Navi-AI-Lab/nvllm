#!/usr/bin/env bash
# C0 gate — PIECEWISE + β-coop + CUTE_PHASE_E_FALLBACK_RAISE=1 must pass
# 8/8 GSM8K sanity. Proves the spike flag is inert on the established
# path before FULL is even attempted.
#
# Per spec §3 / C0.

set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
TS="$(date +%Y-%m-%d-%H%M)"
EVIDENCE_DIR="$REPO_ROOT/docs/research/2026-04-29-full-graph-spike/evidence/$TS"
mkdir -p "$EVIDENCE_DIR"

echo "=== C0: PIECEWISE + β-coop + CUTE_PHASE_E_FALLBACK_RAISE=1 ==="
echo "Evidence dir: $EVIDENCE_DIR"

# Stop any existing nvllm container so we can re-launch with the env flag.
docker rm -f nvllm 2>/dev/null || true

# Re-launch using the prod serve-cute.sh, but inject the flags via env.
# serve-cute.sh exports docker -e from CUTE_* env vars.
# CUTE_PHASE_E_FUSION=1 is required: serve-cute.sh defaults it to 0,
# and without β-coop active C0 isn't actually testing what we claim.
CUTE_PHASE_E_FUSION=1 \
CUTE_PHASE_E_FALLBACK_RAISE=1 \
  "$REPO_ROOT/scripts/serve-cute.sh" 2>&1 | tee "$EVIDENCE_DIR/c0_serve_launch.txt"

# Push host-side spike edits into the container BEFORE Python imports the
# CuTe backend. docker cp races against vLLM's startup imports — we win
# because docker cp is faster than the model load + import phase.
"$REPO_ROOT/docs/research/2026-04-29-full-graph-spike/_sync_host_edits.sh" \
  2>&1 | tee "$EVIDENCE_DIR/c0_sync_host_edits.txt"

# Wait for serve readiness — single wait, longer ceiling for graph compile.
echo "Waiting for /v1/models to respond (up to 600s)..."
READY=""
for i in $(seq 1 120); do
  if curl -s -o /dev/null -w "%{http_code}" http://localhost:8000/v1/models 2>/dev/null | grep -q 200; then
    echo "Ready after $((i*5))s."
    READY=1
    break
  fi
  sleep 5
done
if [ -z "$READY" ]; then
  echo "FAIL: /v1/models did not respond within 600s"
  docker logs nvllm 2>&1 | tail -50 > "$EVIDENCE_DIR/c0_docker_logs_timeout.txt"
  exit 1
fi

# Capture container logs to confirm flag is on.
docker logs nvllm 2>&1 > "$EVIDENCE_DIR/c0_docker_logs.txt"
if ! grep -q "CUTE_PHASE_E_FALLBACK_RAISE=1" "$EVIDENCE_DIR/c0_docker_logs.txt"; then
  echo "FAIL: import-time warning not found in container logs"
  echo "  Did serve-cute.sh forward CUTE_PHASE_E_FALLBACK_RAISE to the container?"
  echo "  If not, edit scripts/serve-cute.sh to add '-e CUTE_PHASE_E_FALLBACK_RAISE' to docker run."
  exit 1
fi

# Run the canonical 8/8 sanity.
"$REPO_ROOT/.venv/bin/python" "$REPO_ROOT/scripts/gsm8k_sanity.py" \
  --api http://localhost:8000/v1 \
  --model default \
  --json --save "$EVIDENCE_DIR/c0_gsm8k_sanity.json" \
  | tee "$EVIDENCE_DIR/c0_gsm8k_sanity.txt"

# Parse pass count (sanity.py emits "correct"/"total"; fall back gracefully).
PASS=$("$REPO_ROOT/.venv/bin/python" -c "
import json
d=json.load(open('$EVIDENCE_DIR/c0_gsm8k_sanity.json'))
print(d.get('passed', d.get('correct', 0)))
")
TOTAL=$("$REPO_ROOT/.venv/bin/python" -c "
import json
d=json.load(open('$EVIDENCE_DIR/c0_gsm8k_sanity.json'))
print(d.get('total', 8))
")

echo "C0 result: $PASS / $TOTAL"
if [ "$PASS" != "$TOTAL" ]; then
  echo "C0 FAIL — flag is not inert on PIECEWISE+β-coop. Stop spike."
  exit 1
fi

cat > "$EVIDENCE_DIR/c0_summary.md" <<EOF
# C0 — PIECEWISE + flag inertness — PASS

- Date: $TS
- Result: $PASS / $TOTAL
- Flag: CUTE_PHASE_E_FALLBACK_RAISE=1
- Mode: PIECEWISE (default serve-cute.sh)

Proves the spike flag does not interfere with the established β-coop
path and that β-coop does not rely on fallback for normal operation.

C1 may now proceed.
EOF

echo "C0 PASS — summary at $EVIDENCE_DIR/c0_summary.md"
