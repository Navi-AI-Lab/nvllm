#!/bin/bash
# Test for progressive output degradation.
# Sends N requests with a trivial prompt and checks for gibberish.
#
# Usage:
#   ./benchmarks/nvllm/test_degradation.sh [num_requests] [api_base]
#
# Each response is checked for the substring "4" (expected answer to "2+2").
# Failures are printed inline so you can see when degradation starts.

set -euo pipefail

N="${1:-50}"
API="${2:-http://localhost:8000/v1}"
MODEL="default"

PASS=0
FAIL=0

echo "=== Degradation test: $N requests to $API ==="
echo ""

for i in $(seq 1 "$N"); do
  RESP=$(curl -s --max-time 120 "$API/chat/completions" \
    -H "Content-Type: application/json" \
    -d '{
      "model": "'"$MODEL"'",
      "messages": [{"role":"user","content":"What is 2+2? Reply with just the number, no explanation."}],
      "max_tokens": 256,
      "temperature": 0,
      "chat_template_kwargs": {"enable_thinking": false}
    }')

  # Extract the assistant content
  CONTENT=$(echo "$RESP" | python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    print(d['choices'][0]['message']['content'])
except Exception as e:
    print(f'ERROR: {e}')
" 2>&1)

  # Check if response contains "4"
  if echo "$CONTENT" | grep -q "4"; then
    STATUS="PASS"
    PASS=$((PASS + 1))
  else
    STATUS="FAIL"
    FAIL=$((FAIL + 1))
  fi

  # Truncate long responses for display
  DISPLAY=$(echo "$CONTENT" | head -c 120 | tr '\n' ' ')
  printf "[%3d/%d] %-4s | %s\n" "$i" "$N" "$STATUS" "$DISPLAY"
done

echo ""
echo "=== Results: $PASS pass, $FAIL fail out of $N ==="
if [ "$FAIL" -gt 0 ]; then
  echo "DEGRADATION DETECTED at request(s) above marked FAIL"
  exit 1
else
  echo "ALL PASSED - no degradation detected"
  exit 0
fi
