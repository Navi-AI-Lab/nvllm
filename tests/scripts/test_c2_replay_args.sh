#!/bin/bash
# Smoke test: c2_replay_coherence.py accepts --json-out / --evidence-dir args.
# Cannot exercise the API path here (no live server); this verifies argparse
# plumbing only. Real path exercise happens in Task 6 integration.

set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
SCRIPT="$REPO_ROOT/docs/research/2026-04-29-full-graph-spike/c2_replay_coherence.py"

echo "[c2-args-test] checking --help mentions both new args"
out=$("$REPO_ROOT/.venv/bin/python" "$SCRIPT" --help 2>&1 || true)
echo "$out" | grep -q -- '--json-out' || { echo "FAIL: --json-out missing from --help"; exit 1; }
echo "$out" | grep -q -- '--evidence-dir' || { echo "FAIL: --evidence-dir missing from --help"; exit 1; }
echo "[c2-args-test] PASS"
