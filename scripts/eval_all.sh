#!/bin/bash
# nvllm -- Run all quality evals sequentially
#
# Runs GPQA Diamond, AIME 2024, AIME 2025 against the served model.
# HLE and LiveCodeBench require separate tooling (see docs).
#
# Usage:
#   ./scripts/eval_all.sh
#   PORT=8001 ./scripts/eval_all.sh

set -euo pipefail

SCRIPT_DIR="$(dirname "$0")"
PORT="${PORT:-8000}"
MODEL="${MODEL:-default}"

echo "=== nvllm Quality Eval Suite ==="
echo "  Model: $MODEL @ localhost:$PORT"
echo ""
echo "Evals: GPQA Diamond (CoT 0-shot), AIME 2024, AIME 2025"
echo ""

echo "──────────────────────────────────────────"
echo "1/2  GPQA Diamond"
echo "──────────────────────────────────────────"
PORT="$PORT" MODEL="$MODEL" "$SCRIPT_DIR/eval_gpqa_diamond.sh"
echo ""

echo "──────────────────────────────────────────"
echo "2/2  AIME 2024 + 2025"
echo "──────────────────────────────────────────"
PORT="$PORT" MODEL="$MODEL" "$SCRIPT_DIR/eval_aime.sh"
echo ""

echo "=== All evals complete ==="
echo "Results: benchmarks/nvllm/results/evals/"
