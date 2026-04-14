#!/bin/bash
# nvllm-perf -- Benchmark NVFP4 model serving on GB10
#
# Runs vllm bench serve at multiple request rates and saves results.
# Uses the nvllm Docker image as the benchmark client (no host vllm needed).
# Requires a model already being served on the target port.
#
# Usage:
#   ./scripts/local/bench_nvfp4.sh                    # defaults: localhost:8000
#   PORT=8001 ./scripts/local/bench_nvfp4.sh           # custom port
#   ./scripts/local/bench_nvfp4.sh --sweep             # also run GuideLLM sweep
#   ./scripts/local/bench_nvfp4.sh --prompts 1000      # more prompts
#   ./scripts/local/bench_nvfp4.sh --quick             # single rate (8 req/s)

set -euo pipefail

source "$(dirname "$0")/../common.sh"

PORT="${PORT:-8000}"
BASE_URL="http://localhost:${PORT}"
MODEL="default"
TOKENIZER="${TOKENIZER:-}"
NUM_PROMPTS=500
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
RESULT_DIR="${REPO_ROOT}/benchmarks/nvllm/results"
SWEEP=0
QUICK=0
RATES="1 4 8 16"
TIMESTAMP=$(date +%Y%m%d-%H%M%S)

# Parse flags
while [ $# -gt 0 ]; do
  case "$1" in
    --sweep)       SWEEP=1; shift ;;
    --quick)       QUICK=1; RATES="8"; NUM_PROMPTS=200; shift ;;
    --prompts)     NUM_PROMPTS="$2"; shift 2 ;;
    --prompts=*)   NUM_PROMPTS="${1#*=}"; shift ;;
    *) echo "Unknown argument: $1" >&2; exit 1 ;;
  esac
done

mkdir -p "$RESULT_DIR"

# Verify server is up
echo "Checking server at $BASE_URL..."
if ! curl -sf "${BASE_URL}/v1/models" > /dev/null 2>&1; then
  echo "ERROR: No model serving at $BASE_URL" >&2
  echo "Start a model first, e.g.: ./scripts/serve.sh" >&2
  exit 1
fi

# Fetch actual model name and root path (for tokenizer)
MODEL_INFO=$(curl -sf "${BASE_URL}/v1/models")
SERVED_MODEL=$(echo "$MODEL_INFO" | python3 -c "import sys,json; print(json.load(sys.stdin)['data'][0]['id'])" 2>/dev/null || echo "$MODEL")
if [ -z "$TOKENIZER" ]; then
  MODEL_ROOT=$(echo "$MODEL_INFO" | python3 -c "import sys,json; print(json.load(sys.stdin)['data'][0]['root'])" 2>/dev/null || true)
  if [ -n "$MODEL_ROOT" ]; then
    TOKENIZER="$MODEL_ROOT"
  fi
fi
echo "Benchmarking model: $SERVED_MODEL"
if [ -n "$TOKENIZER" ]; then echo "  Tokenizer: $TOKENIZER"; fi
echo ""

# Run vllm bench serve at each request rate via Docker container
for rate in $RATES; do
  echo "=== Request rate: $rate req/s | prompts: $NUM_PROMPTS ==="
  RESULT_FILE="${SERVED_MODEL//\//-}-rate${rate}-${TIMESTAMP}.json"

  TOKENIZER_ARGS=()
  if [ -n "$TOKENIZER" ]; then
    TOKENIZER_ARGS+=(--tokenizer "$TOKENIZER")
  fi

  docker run --rm \
    --gpus all \
    --network host \
    -v "${RESULT_DIR}:/results" \
    -v "$HOME/.cache/huggingface:/root/.cache/huggingface:ro" \
    "$NVLLM_IMAGE" \
    bench serve \
      --backend openai \
      --base-url "$BASE_URL" \
      --model "$SERVED_MODEL" \
      --endpoint /v1/completions \
      --dataset-name random \
      --num-prompts "$NUM_PROMPTS" \
      --request-rate "$rate" \
      --percentile-metrics ttft,tpot,itl \
      --metric-percentiles 50,90,99 \
      --save-result \
      --result-dir /results \
      --result-filename "$RESULT_FILE" \
      "${TOKENIZER_ARGS[@]}"

  echo ""
done

echo "=== Results saved to $RESULT_DIR ==="
ls -lt "$RESULT_DIR"/*.json 2>/dev/null | head -10

# Optional GuideLLM sweep
if [ "$SWEEP" -eq 1 ]; then
  echo ""
  echo "=== GuideLLM sweep ==="
  docker run --rm \
    --gpus all \
    --network host \
    --entrypoint bash \
    -v "${RESULT_DIR}:/results" \
    "$NVLLM_IMAGE" \
    -c "pip install guidellm 2>&1 | tail -3 && guidellm benchmark \
      --target '$BASE_URL' \
      --model '$SERVED_MODEL' \
      --profile sweep \
      --max-seconds 120 \
      --output-dir /results \
      --outputs json,html"
  echo "GuideLLM report saved to $RESULT_DIR"
fi
