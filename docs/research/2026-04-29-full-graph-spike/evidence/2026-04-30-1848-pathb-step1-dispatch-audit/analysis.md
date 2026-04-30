# Path B Step 1 — dispatch_cudagraph audit summary

- **Timestamp:** 2026-04-30-1848
- **Code commit:** `0ef9580ef` (per-call dispatch_cudagraph audit probe)
- **Probes:** `CUTE_DISPATCH_AUDIT=1 + CUTE_WO_RESET_LOG=1 + CUTE_FULL_GRAPH_PROBE=1`
- **Layers:** lower-8 (`3,7,11,15,19,23,27,31`)
- **Compilation config:** `cudagraph_mode=FULL_AND_PIECEWISE, cudagraph_capture_sizes=[1]`
- **C2 result this run:** `unique=1, cross-indep, PASS` (auto evidence at `../2026-04-30-1844/c2_replay_coherence.md`)

## Mode tally — all 100 rows

| Mode | Count |
|---|---|
| FULL | 89 |
| FULL_DECODE_ONLY | 0 |
| PIECEWISE | 2 |
| NONE | 9 |

## Mode tally — decode-only rows (`raw_tokens=1, desc_tokens=1, raw_reqs=1, force_eager=False`)

| Mode | Count |
|---|---|
| FULL | 89 |
| FULL_DECODE_ONLY | 0 |
| PIECEWISE | 2 |
| NONE | 0 |

## Where the non-FULL rows came from

**The 9 NONE rows split into two classes (none of them are steady decode):**

- 6 with `force_eager=True, raw_tokens=1` — eager warmup/probe passes during init.
- 1 with `force_eager=True, raw_tokens=65536` (idx=0) — the memory-profile pass.
- 3 with `force_eager=False, raw_tokens=12, raw_reqs=1, desc_uniform=False` — prefill of the 12-token prompt (`Q: What is the capital of France?\nA:`). Prefill runs eager; this is normal vLLM behavior.

**The 2 PIECEWISE rows happened during FULL graph capture, not steady decode:**

- idx=2 at 22:43:57, idx=6 at 22:44:01. The first-FULL probe fired at 22:43:58.
- Both rows have `uniform_decode=False, desc_uniform=False` — non-uniform shapes that the capture machinery covers via PIECEWISE.
- After idx=8 (a NONE warmup at 22:44:03) there are zero PIECEWISE rows for the rest of the run.

**Every single steady-decode dispatch (89 rows) is `cg_mode=FULL` with `uniform_decode=True, desc_uniform=True`.**

## Verdict

**Hybrid-dispatch hypothesis is RULED OUT for the steady decode path.** During the c2 replay-coherence pattern (8 same-prompt + cross-prompt A + cross-prompt B = 10 generation calls of 32 tokens each = ~80+ steady-decode steps), the dispatcher consistently picked `FULL` for every single decode step.

This eliminates Path B Step 1's working hypothesis (silent hybrid/PIECEWISE dispatch as the cause of v2's Gate 1 FAIL). **Proceed to Step 2: inspect `_beta_coop_op.py` for Python-side capture-time freeze risks.**

## Important sidebar — this run produced unique=1 PASS

Gate 1 at 17:48 gave `unique=4 FAIL`. This run at 18:48 — same code path, same env (mod the new audit probe), same prompts — gave `unique=1 PASS`. The wo_output reset still fires (8 unique data_ptrs) so v2's patch is alive in both runs.

This single PASS run does NOT mean the bug is fixed. Two readings:

1. **The audit probe accidentally perturbed something.** The user explicitly cautioned against state mutation in the dispatch path (per `feedback_no_self_mut_in_cudagraph_dispatch`). The new probe does mutate module-level state (`_CUTE_DISPATCH_AUDIT_COUNT += 1`) and emits log records, both inside the dispatch hot path. This is a known risk shape — the prior counter+setattr variant correlated with a 20+ min capture hang. The pure-int counter pattern is safer in principle but not zero-risk.

2. **The bug is genuinely stochastic.** Gate 1 (unique=4), v1 Gate 1 (unique=2), pre-v1 (unique=3), and now this run (unique=1) are all noise samples from a knife-edge regime. A single c2 trial cannot characterize this distribution. The v2 closeout already framed this; this run reinforces it.

The two readings have very different implications:
- If (1): we have evidence-by-side-effect that *something* state-related in the dispatch path matters for the divergence. That'd be a real new lead.
- If (2): we need multi-seed runs to even characterize what FAIL means here.

**Recommendation for the controller / human:** before doing anything else, run the audit OFF (`CUTE_DISPATCH_AUDIT=0`) but with everything else identical to this run, several times. If c2 still PASSes, the audit probe wasn't load-bearing and the unique=1 was stochastic. If c2 FAILs without the audit, the audit was masking the bug — and we have a new diagnostic surface to chase.

## Files

- `cute_dispatch_audit.txt` — 100 lines, the bounded per-call probe output.
- `cute_full_graph_probe.txt` — first-any (cg_mode=NONE @ profile run) + first-FULL (cg_mode=FULL @ 22:43:58).
- `cute_wo_reset_log.txt` — 8 unique data_ptrs (v2 reset still fires correctly).
- `c2_replay_coherence_stdout.txt` — full stdout from the harness.
- `docker_logs_full.txt` — full container logs.
