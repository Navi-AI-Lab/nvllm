#!/bin/bash
# nvllm -- Evaluate AIME 2024 + 2025 via lm-eval-harness
#
# Runs against the served model's OpenAI-compatible API.
# Requires: lm-eval installed in ~/.venvs/lm-eval
#
# BF16 baseline (Qwen3.5-27B): ~90.8% (model card lists AIME 2026)
#
# Usage:
#   ./scripts/eval_aime.sh
#   PORT=8001 ./scripts/eval_aime.sh

set -euo pipefail

PORT="${PORT:-8000}"
MODEL="${MODEL:-default}"
BASE_URL="http://localhost:${PORT}/v1/chat/completions"
RESULT_DIR="benchmarks/nvllm/results/evals/aime"
LM_EVAL="${LM_EVAL:-$HOME/.venvs/lm-eval/bin/lm_eval}"

# Verify server
if ! curl -sf "http://localhost:${PORT}/v1/models" > /dev/null 2>&1; then
  echo "ERROR: No model serving at http://localhost:${PORT}" >&2
  exit 1
fi

export OPENAI_API_KEY=dummy
mkdir -p "$RESULT_DIR"

echo "=== AIME 2024 + 2025 ==="
echo "  Model:  $MODEL @ localhost:$PORT"
echo "  Output: $RESULT_DIR"
echo ""

"$LM_EVAL" \
  --model local-chat-completions \
  --model_args "model=${MODEL},base_url=${BASE_URL},num_concurrent=4,tokenized_requests=False" \
  --tasks aime24,aime25 \
  --batch_size 1 \
  --apply_chat_template \
  --output_path "$RESULT_DIR" \
  --log_samples

echo ""
echo "=== Results saved to $RESULT_DIR ==="
