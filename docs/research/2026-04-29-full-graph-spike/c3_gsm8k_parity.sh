#!/usr/bin/env bash
# C3 gate — FULL_AND_PIECEWISE token-level matches PIECEWISE on the
# same checkpoint and seed=42 across 50 GSM8K questions.
#
# Acceptance:
#   - zero answer-level divergence (gsm8k_eval_50 saves `got` + `status`,
#     not full completion text — token-level diff would require modifying
#     the eval script)
#   - FULL accuracy >= PIECEWISE accuracy (parity, not absolute >=90%)
#
# Per spec §3 / C3.

set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
TS="$(date +%Y-%m-%d-%H%M)"
EVIDENCE_DIR="$REPO_ROOT/docs/research/2026-04-29-full-graph-spike/evidence/$TS"
mkdir -p "$EVIDENCE_DIR"

run_gsm8k() {
  local mode="$1"     # "piecewise" or "full"
  local launcher="$2" # script to start the serve
  local out="$EVIDENCE_DIR/c3_gsm8k_${mode}.json"

  echo "=== C3 [$mode]: stopping any existing nvllm container ==="
  docker rm -f nvllm 2>/dev/null || true

  echo "=== C3 [$mode]: launching ${launcher} ==="
  bash "$launcher" 2>&1 | tee "$EVIDENCE_DIR/c3_${mode}_serve_launch.txt"

  # Push host-side spike edits BEFORE Python imports the CuTe backend.
  "$REPO_ROOT/docs/research/2026-04-29-full-graph-spike/_sync_host_edits.sh" \
    2>&1 | tee "$EVIDENCE_DIR/c3_${mode}_sync_host_edits.txt"

  echo "Waiting for /v1/models (up to 900s — FULL graph compile if applicable)..."
  READY=""
  for i in $(seq 1 180); do
    if curl -s -o /dev/null -w "%{http_code}" http://localhost:8000/v1/models 2>/dev/null | grep -q 200; then
      echo "Ready after $((i*5))s."
      READY=1
      break
    fi
    sleep 5
  done
  if [ -z "$READY" ]; then
    echo "FAIL: /v1/models did not respond within 900s for mode=$mode"
    docker logs nvllm 2>&1 | tail -100 > "$EVIDENCE_DIR/c3_${mode}_docker_logs_timeout.txt"
    exit 1
  fi

  # gsm8k_eval_50.py args are --api, --model, --n, --seed, --max-tokens, --save
  # (verified against scripts/gsm8k_eval_50.py argparse).
  "$REPO_ROOT/.venv/bin/python" "$REPO_ROOT/scripts/gsm8k_eval_50.py" \
    --api http://localhost:8000/v1 \
    --model default \
    --n 50 --seed 42 --max-tokens 512 \
    --save "$out" \
    | tee "$EVIDENCE_DIR/c3_gsm8k_${mode}.txt"

  docker logs nvllm 2>&1 > "$EVIDENCE_DIR/c3_${mode}_docker_logs.txt"
}

# For C3 PIECEWISE we want the SAME flags-on profile as C0 (β-coop ON,
# fallback raise ON) — the parity claim is "same checkpoint, same flags,
# only the cudagraph_mode differs."
export CUTE_PHASE_E_FUSION=1
export CUTE_PHASE_E_FALLBACK_RAISE=1

# Run PIECEWISE first (reference).
run_gsm8k piecewise "$REPO_ROOT/scripts/serve-cute.sh"

# Then FULL_AND_PIECEWISE n=1.
run_gsm8k full "$REPO_ROOT/scripts/serve-cute-full.sh"

# Answer-level parity diff.
"$REPO_ROOT/.venv/bin/python" - <<EOF
import json
from pathlib import Path
ev = Path("$EVIDENCE_DIR")
piece = json.loads((ev / "c3_gsm8k_piecewise.json").read_text())
full  = json.loads((ev / "c3_gsm8k_full.json").read_text())

p_results = piece.get("results", [])
f_results = full.get("results",  [])
assert len(p_results) == len(f_results), (
    f"length mismatch: piecewise={len(p_results)} full={len(f_results)}"
)
divergent = []
for i, (p, f) in enumerate(zip(p_results, f_results)):
    if p.get("got", "") != f.get("got", "") or p.get("status") != f.get("status"):
        divergent.append({
            "idx": i,
            "piecewise_got": p.get("got", ""),
            "full_got": f.get("got", ""),
            "piecewise_status": p.get("status"),
            "full_status": f.get("status"),
            "piecewise_raw_tail": p.get("raw_tail", "")[:120],
            "full_raw_tail": f.get("raw_tail", "")[:120],
        })
p_correct = piece.get("correct", -1)
f_correct = full.get("correct", -1)
out = {
    "n_questions": len(p_results),
    "n_divergent_answers": len(divergent),
    "piecewise_correct": p_correct,
    "full_correct": f_correct,
    "parity_pass": (len(divergent) == 0) and (f_correct >= p_correct),
    "divergent_first5": divergent[:5],
    "note": (
        "Answer-level parity (got + status). gsm8k_eval_50 does not save "
        "full completion text — see plan Task 10 step 1 note."
    ),
}
(ev / "c3_diff.json").write_text(json.dumps(out, indent=2))
print(json.dumps(out, indent=2))
EOF

PARITY=$("$REPO_ROOT/.venv/bin/python" -c "
import json
print(json.load(open('$EVIDENCE_DIR/c3_diff.json'))['parity_pass'])
")

cat > "$EVIDENCE_DIR/c3_summary.md" <<EOF
# C3 — math parity vs PIECEWISE — $([ "$PARITY" = "True" ] && echo PASS || echo FAIL)

- Timestamp: $TS
- Diff: c3_diff.json
- PIECEWISE evidence: c3_gsm8k_piecewise.json
- FULL evidence: c3_gsm8k_full.json
EOF

echo "C3 result (parity_pass=$PARITY) — summary at $EVIDENCE_DIR/c3_summary.md"
[ "$PARITY" = "True" ]
