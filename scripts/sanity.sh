#!/bin/bash
# Quick sanity gate — runs GSM8K canary and saves results with raw output.
#
# Usage:
#   ./scripts/sanity.sh                    # wait for server, run, save
#   ./scripts/sanity.sh --label "my_test"  # add a label
#   ./scripts/sanity.sh --no-wait          # skip server readiness check
#
# Output goes to: benchmarks/nvllm/results/sanity/YYYY-MM-DD-HHMMSS.json

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
RESULTS_DIR="$REPO_ROOT/benchmarks/nvllm/results/sanity"
API="${API:-http://localhost:8000/v1}"
NO_WAIT=false
LABEL=""

# Parse args — pass unknown args through to gsm8k_sanity.py
PASSTHROUGH=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --no-wait) NO_WAIT=true; shift ;;
    --label)   LABEL="$2"; shift 2 ;;
    *)         PASSTHROUGH+=("$1"); shift ;;
  esac
done

mkdir -p "$RESULTS_DIR"

TIMESTAMP=$(date +%Y-%m-%d-%H%M%S)
OUTFILE="$RESULTS_DIR/${TIMESTAMP}.json"

# Wait for server readiness
if [ "$NO_WAIT" = false ]; then
  echo "Waiting for $API/models ..."
  for i in $(seq 1 120); do
    if curl -sf --max-time 3 "$API/models" >/dev/null 2>&1; then
      echo "Server ready."
      break
    fi
    if [ "$i" -eq 120 ]; then
      echo "ERROR: Server not ready after 120 attempts." >&2
      exit 1
    fi
    sleep 3
  done
fi

echo "=== GSM8K Sanity Gate ==="
echo "  API:    $API"
echo "  Output: $OUTFILE"
[ -n "$LABEL" ] && echo "  Label:  $LABEL"
echo ""

LABEL_ARG=()
[ -n "$LABEL" ] && LABEL_ARG=(--label "$LABEL")

python3 "$SCRIPT_DIR/gsm8k_sanity.py" \
  --api "$API" \
  --save "$OUTFILE" \
  "${LABEL_ARG[@]}" \
  "${PASSTHROUGH[@]}"

echo ""
echo "Results saved to: $OUTFILE"
