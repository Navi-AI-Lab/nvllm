#!/bin/bash
# nvllm -- Evaluate GPQA Diamond (CoT, 0-shot) via lm-eval-harness
#
# Runs against the served model's OpenAI-compatible API.
# Requires: lm-eval installed in ~/.venvs/lm-eval
#
# BF16 baseline (Qwen3.5-27B): 85.5% (from model card, thinking mode)
#
# Usage:
#   ./scripts/eval_gpqa_diamond.sh
#   PORT=8001 ./scripts/eval_gpqa_diamond.sh

set -euo pipefail

PORT="${PORT:-8000}"
MODEL="${MODEL:-default}"
BASE_URL="http://localhost:${PORT}/v1/chat/completions"
RESULT_DIR="benchmarks/nvllm/results/evals/gpqa_diamond"
LM_EVAL="${LM_EVAL:-$HOME/.venvs/lm-eval/bin/lm_eval}"

# Verify server
if ! curl -sf "http://localhost:${PORT}/v1/models" > /dev/null 2>&1; then
  echo "ERROR: No model serving at http://localhost:${PORT}" >&2
  exit 1
fi

export OPENAI_API_KEY=dummy
mkdir -p "$RESULT_DIR"

echo "=== GPQA Diamond (CoT, 0-shot) ==="
echo "  Model:  $MODEL @ localhost:$PORT"
echo "  Output: $RESULT_DIR"
echo ""

"$LM_EVAL" \
  --model local-chat-completions \
  --model_args "model=${MODEL},base_url=${BASE_URL},num_concurrent=4,tokenized_requests=False" \
  --tasks gpqa_diamond_cot_zeroshot \
  --batch_size 1 \
  --apply_chat_template \
  --output_path "$RESULT_DIR" \
  --log_samples

echo ""
echo "=== Results saved to $RESULT_DIR ==="
