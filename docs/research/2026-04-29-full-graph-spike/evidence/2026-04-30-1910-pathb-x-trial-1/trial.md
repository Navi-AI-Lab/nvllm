# Path B Step X — Trial 1 (audit-OFF reproducibility)

- **Timestamp:** 2026-04-30-1910
- **git SHA:** d36abf7713dfaaf8c5beb4dd7ee2c0099428e93a
- **Branch:** feat/cute-beta-coop-persistent-buffers
- **Trial command (host):** `CUTE_WO_RESET_LOG=1 bash docs/research/2026-04-29-full-graph-spike/c2_full_layer_bisect.sh '3,7,11,15,19,23,27,31'`
- **Env contract:**
  - `CUTE_WO_RESET_LOG=1` (matches failing Gate 1)
  - `CUTE_DISPATCH_AUDIT=0` (audit OFF — bisect script default)
  - `CUTE_FULL_GRAPH_PROBE=1` (hardcoded in bisect script L72)
- **Time-to-/v1/models:** 196s
- **first-any probe present:** yes
- **first-FULL probe present:** yes
- **cute_wo_reset unique data_ptrs:** 8

## c2_replay_coherence result

- **same_prompt_unique_count:** 1
- **same_prompt_pass:** true
- **cross_prompt_pass:** true
- **overall_pass:** true

## Files
- docker_logs_full.txt
- cute_full_graph_probe.txt
- cute_wo_reset_log.txt
- c2_replay_coherence_stdout.txt
- c2_replay_coherence.md (copied from auto-evidence dir)
- c2_replay_coherence.json (copied from auto-evidence dir)
