#!/bin/bash
# Z1 cold-restart stability sweep — FULL+CuTe with --no-blessed-verify.
#
# Two modes (keep the signals separate, per 2026-05-19 user directive):
#   SWEEP_MODE=z1            (default) — Z1 / torch-inductor signal.
#                                        /root/.cache/vllm dies with container
#                                        each cycle (truly cold inductor).
#                                        /tmp/nvllm-cute-cache stays warm.
#   SWEEP_MODE=everything    — full-cold stability signal. Same as z1 mode
#                                        PLUS `rm -rf /tmp/nvllm-cute-cache`
#                                        between cycles, so cute.compile() also
#                                        runs cold every cycle.
#
# DO NOT switch modes mid-sweep, and never nuke /tmp/nvllm-cute-cache while a
# container is still using it.
#
# Per cycle:
#   1. docker rm -f nvllm (frees /root/.cache/vllm + nvidia ctx)
#   2. (everything mode only) rm -rf /tmp/nvllm-cute-cache && mkdir -p ...
#   3. ./scripts/serve-qwen35-full.sh --no-blessed-verify
#   4. wait /v1/models ready (cold compile + FULL graph capture)
#   5. warmup probe (4 tokens)
#   6. deterministic probe ("12 × 13", 128 tokens) → record sha16 + wall
#   7. record verdict per cycle
#
# Designed for tmux:
#   tmux new-session -d -s z1_sweep -c $(pwd) "./scripts/z1_sweep.sh"
#
# Runs until either CYCLES is hit or DEADLINE_SECS (default 4h) elapses
# from the FIRST cycle start — whichever comes first.

set -uo pipefail   # NOT -e: per-cycle failures shouldn't kill the sweep

SWEEP_MODE="${SWEEP_MODE:-z1}"
if [[ "$SWEEP_MODE" != "z1" && "$SWEEP_MODE" != "everything" ]]; then
  echo "ERROR: SWEEP_MODE must be 'z1' or 'everything', got: $SWEEP_MODE" >&2
  exit 1
fi
OUT="${OUT:-/tmp/z1_sweep_2026-05-19_${SWEEP_MODE}}"
mkdir -p "$OUT"

CYCLES="${CYCLES:-30}"
READY_TIMEOUT="${READY_TIMEOUT:-1800}"
DEADLINE_SECS="${DEADLINE_SECS:-14400}"   # 4 hours

PROBE_PROMPT='{"model":"default","prompt":"Q: What is 12 times 13?\n\nA:","max_tokens":128,"temperature":0.0}'
SWEEP_T0=$(date +%s)

echo "=== Z1 stability sweep started $(date -Iseconds) ==="
echo "mode=$SWEEP_MODE cycles_cap=$CYCLES deadline=${DEADLINE_SECS}s out=$OUT"
echo ""

for cycle in $(seq 1 "$CYCLES"); do
  EL_TOTAL=$(( $(date +%s) - SWEEP_T0 ))
  if [ "$EL_TOTAL" -ge "$DEADLINE_SECS" ]; then
    echo "##### deadline reached at cycle=$cycle (sweep elapsed ${EL_TOTAL}s) #####"
    break
  fi

  CYCLE_DIR="$OUT/cycle_$(printf '%02d' "$cycle")"
  mkdir -p "$CYCLE_DIR"

  echo ""
  echo "##### CYCLE $cycle / $CYCLES — start $(date -Iseconds) [sweep_el=${EL_TOTAL}s] #####"

  # 1) tear down (container fs gone → /root/.cache/vllm dies → cold inductor)
  docker rm -f nvllm > /dev/null 2>&1 || true
  sleep 2

  # 1a) (everything mode only) wipe cute compile cache so cute.compile()
  #     also runs cold this cycle. Safe to nuke here — container is gone.
  #     IMPORTANT: cute.compile() runs as root in-container and produces
  #     root-owned .o files. Host user can't delete those, so we wipe via
  #     a throwaway root container instead of host `rm -rf`.
  if [ "$SWEEP_MODE" = "everything" ]; then
    docker run --rm -v /tmp/nvllm-cute-cache:/c busybox \
        sh -c 'rm -rf /c/.[!.]* /c/* 2>/dev/null; ls /c | wc -l' \
        > "$CYCLE_DIR/cache_wipe.log" 2>&1
    REMAINING=$(tail -1 "$CYCLE_DIR/cache_wipe.log" | tr -d '[:space:]')
    if [ "$REMAINING" != "0" ]; then
      echo "  WARN: cute-cache wipe left $REMAINING entries (see $CYCLE_DIR/cache_wipe.log)"
    else
      echo "  cute-cache wiped (everything mode, 0 entries remaining)"
    fi
  fi

  # 2) launch cold
  ./scripts/serve-qwen35-full.sh --no-blessed-verify > "$CYCLE_DIR/serve.log" 2>&1
  CONTAINER_STATUS=$(docker ps --filter name=nvllm --format '{{.Status}}' || true)
  echo "  container after serve.sh: $CONTAINER_STATUS"

  # 3) wait ready
  T0=$(date +%s)
  READY=0
  while true; do
    if curl -sf http://localhost:8000/v1/models >/dev/null 2>&1; then
      READY=1; break
    fi
    EL=$(( $(date +%s) - T0 ))
    if [ "$EL" -ge "$READY_TIMEOUT" ]; then break; fi
    if ! docker ps --filter name=nvllm --format '{{.Status}}' | grep -q Up; then
      echo "  container exited at ${EL}s"; break
    fi
    sleep 10
  done
  READY_SECS=$(( $(date +%s) - T0 ))
  echo "  ready=$READY after ${READY_SECS}s"

  if [ "$READY" -ne 1 ]; then
    echo "{\"cycle\": $cycle, \"ok\": false, \"reason\": \"server_never_ready\", \"ready_secs\": $READY_SECS}" > "$CYCLE_DIR/verdict.json"
    docker logs nvllm --tail 100 > "$CYCLE_DIR/docker_tail.log" 2>&1 || true
    continue
  fi

  # 4) warmup probe
  curl -s -o /dev/null http://localhost:8000/v1/completions \
       -H "Content-Type: application/json" \
       -d '{"model":"default","prompt":"Q: 2+2?\nA:","max_tokens":4,"temperature":0.0}' || true

  # 5) deterministic probe
  PROBE_T0=$(date +%s.%N)
  PROBE_RESP=$(curl -s --max-time 180 http://localhost:8000/v1/completions \
                    -H "Content-Type: application/json" \
                    -d "$PROBE_PROMPT")
  PROBE_T1=$(date +%s.%N)
  PROBE_WALL=$(awk "BEGIN{printf \"%.3f\", $PROBE_T1-$PROBE_T0}")

  python3 - "$PROBE_RESP" "$cycle" "$READY_SECS" "$PROBE_WALL" > "$CYCLE_DIR/verdict.json" <<'PY'
import sys, json, hashlib
raw = sys.argv[1]
cycle = int(sys.argv[2])
ready_secs = int(sys.argv[3])
probe_wall = float(sys.argv[4])
ok = True
sha = preview = None
ct = None
finish = None
try:
    d = json.loads(raw)
    c = d['choices'][0]
    txt = c['text']
    sha = hashlib.sha256(txt.encode()).hexdigest()[:16]
    preview = txt[:200]
    ct = d.get('usage', {}).get('completion_tokens')
    finish = c.get('finish_reason')
except Exception as e:
    ok = False
    preview = f'parse_error: {e}'
print(json.dumps({
    'cycle': cycle,
    'ok': ok,
    'ready_secs': ready_secs,
    'probe_wall_s': probe_wall,
    'probe_sha16': sha,
    'probe_preview': preview,
    'completion_tokens': ct,
    'finish_reason': finish,
}, indent=2))
PY
  echo "  probe: $(grep -E 'probe_wall_s|probe_sha16|finish_reason' "$CYCLE_DIR/verdict.json")"
done

# Final summary
echo ""
echo "=== SWEEP DONE $(date -Iseconds) (elapsed $(( $(date +%s) - SWEEP_T0 ))s) ==="
python3 - "$OUT" <<'PY'
import json, os, glob, sys
out = sys.argv[1]
verdicts = sorted(glob.glob(os.path.join(out, "cycle_*", "verdict.json")))
ok_count = fail_count = 0
shas = []
for v in verdicts:
    try:
        d = json.load(open(v))
        if d.get('ok'):
            ok_count += 1
            if d.get('probe_sha16'):
                shas.append(d['probe_sha16'])
        else:
            fail_count += 1
    except Exception:
        fail_count += 1
print(f"cycles_complete: {len(verdicts)}  ok: {ok_count}  fail: {fail_count}")
print(f"unique probe SHA16 across {len(shas)} ok runs: {len(set(shas))}")
print("probe SHAs:", shas)
PY

echo ""
echo "Per-cycle verdicts: $OUT/cycle_*/verdict.json"
echo "Per-cycle serve logs: $OUT/cycle_*/serve.log"
echo "DONE"
