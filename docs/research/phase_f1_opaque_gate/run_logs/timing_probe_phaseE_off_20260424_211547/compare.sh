#!/bin/bash
# Compare probe 2 (PHASE_E=0) timings vs probe 1 (PHASE_E=1) baseline.
set -euo pipefail
P1="/home/natfii/docker/nvllm/docs/research/phase_f1_opaque_gate/run_logs/timing_probe_20260424_210752/timing_lines.txt"
P2="$(dirname "$0")/timing_lines.txt"

avg_full_attn () {
  local f="$1"
  grep "layer=full_attention" "$f" | awk '
    {
      step=""; sa=""; mo=""; se="";
      for (i=1; i<=NF; i++) {
        split($i, p, "=");
        if (p[1] == "step_emit") step=p[2]+0;
        if (p[1] == "self_attn") sa=p[2]+0;
        if (p[1] == "mlp_op")    mo=p[2]+0;
        if (p[1] == "mlp_legacy") mo=p[2]+0;
        if (p[1] == "sync_end")   se=p[2]+0;
      }
      if (step >= 70) {  # skip warmup
        print sa, mo, se
      }
    }
  ' | awk '
    { sa += $1; mo += $2; se += $3; n++ }
    END {
      if (n>0) printf "  avg per fused full_attn layer: self_attn_dispatch=%.0fus mlp_op_dispatch=%.0fus sync_end=%.0fus  (n=%d)\n", sa/n, mo/n, se/n, n
    }
  '
}

avg_linear () {
  local f="$1"
  grep "layer=linear_attention" "$f" | awk '
    {
      step=""; la=""; mo=""; se="";
      for (i=1; i<=NF; i++) {
        split($i, p, "=");
        if (p[1] == "step_emit") step=p[2]+0;
        if (p[1] == "linear_attn") la=p[2]+0;
        if (p[1] == "mlp_legacy")  mo=p[2]+0;
        if (p[1] == "mlp_op")      mo=p[2]+0;
        if (p[1] == "sync_end")    se=p[2]+0;
      }
      if (step >= 70) {
        print la, mo, se
      }
    }
  ' | awk '
    { la += $1; mo += $2; se += $3; n++ }
    END {
      if (n>0) printf "  avg per linear_attention layer: linear_attn_dispatch=%.0fus mlp_dispatch=%.0fus sync_end=%.0fus  (n=%d)\n", la/n, mo/n, se/n, n
    }
  '
}

echo "=== PROBE 1 (PHASE_E=1, MLP=1, ATTN=1) ==="
echo "  $P1"
avg_full_attn "$P1"
avg_linear "$P1"
echo ""
echo "=== PROBE 2 (PHASE_E=0, MLP=1, ATTN=1) ==="
echo "  $P2"
avg_full_attn "$P2"
avg_linear "$P2"
