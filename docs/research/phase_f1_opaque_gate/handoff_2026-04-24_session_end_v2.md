---
title: Phase E.2 + F.1 — End-of-session handoff v2 (2026-04-24, post-investigation)
date: 2026-04-24
supersedes: handoff_2026-04-24_session_end.md
---

# Phase E.2 + F.1 — End-of-session handoff v2 (2026-04-24)

This supersedes `handoff_2026-04-24_session_end.md`. The original handoff
hypothesized a "deterministic upstream-class wedge" and ranked four
hypotheses. This investigation falsified all four, found the actual crash
root cause, and revealed two major findings the 8-question sanity test
had been masking.

## TL;DR

| Finding | Status |
|---|---|
| Crash root cause | **flashinfer 0.6.7 autotune memory blowup on SM120.** Fix: `--kernel-config '{"enable_flashinfer_autotune":false}'`. The "negative CUDA graph estimate" canary is downstream noise. |
| Disk full (separate problem) | Found `/` at 100% (2.2 GiB free). Cleaned up to 106 GiB free. |
| Fused-path correctness | **48/50 unfused vs 1/24 fused** on 50-Q real GSM8K (seed=42). Quant is fine; fused path is broken. |
| Fused-path throughput | **~25× slower** than unfused at real-workload generation lengths. 8-Q sanity was hiding it (max_tokens=16 doesn't expose the regression). |
| Plan tasks 1–14 + unit tests | Unaffected; 27/27 still green. |
| Plan tasks 15b–19 | **Blocked. Not by autotune.** By the fused-path correctness/perf bug above. |

## Evidence files (all on disk under `docs/research/phase_f1_opaque_gate/run_logs/`)

```
serve_20260424_182227.log        β-coop server log (run #1, autotune-OFF)
gsm8k_20260424_182227.json       β-coop 8-Q sanity #1: 7/8, Q2=120
gsm8k_run2.json                  β-coop 8-Q sanity #2: 7/8, Q2=120 byte-identical
serve_lite_20260424_183939.log   β-lite control server log
gsm8k_lite_20260424_183939.json  β-lite 8-Q sanity: 7/8, Q2=120 byte-identical
serve_unfused_20260424_185345.log    50-Q unfused server log
eval50_unfused_20260424_185345.json  50-Q unfused: **48/50 (96.0%)**, 0 errors, 1164s
serve_fused_20260424_191700.log      50-Q fused server log (partial — killed mid-eval)
summary_*.txt                    human-readable per-run summaries
```

## 1. Crash root cause

### Falsified hypotheses from v1 handoff §5

1. ❌ **vLLM CUDA-graph regression.** `gpu_model_runner.py:5962` last
   changed in our fork on 2026-04-06 (commit `d4a0daa84`); not the
   trigger.
2. ❌ **Flashinfer JIT cache corruption.** Cache rebuilt fresh on each
   run; not corruption.
3. ❌ **Phase E.2 #2 widened the kernel.** All-fusion-OFF crash repro
   (in v1 handoff) already ruled this out. Confirmed again this session
   by reproducing the wedge after disk cleanup with completely fresh
   image state.
4. ❌ **CuTe DSL JIT vs flashinfer competing for memory.** All-fusion-OFF
   crashes too. Not the source.

### Actual root cause

Container log captured this exact sequence (host EDT in [], container UTC):

```
[18:12:07] Initial profiling/warmup run took 86.36 s        ← model loaded fine
[18:12:08–25] CuTe Phase D MLP kernel compiles × 17 layers  ← fusion fine
[18:12:25] Estimated CUDA graph memory: -2.24 GiB total     ← canary
[18:12:25] Available KV cache memory: 42.33 GiB              ← KV alloc proceeds
[18:12:26] flashinfer.jit: Autotuning process starts ...    ← TRIGGER
[18:13:43] systemd-journald: Under memory pressure          ← ~77s into autotune
[18:15:20] systemd-journald: Under memory pressure
[18:17:??] HOST KERNEL PANIC + REBOOT
```

**Diagnosis:** flashinfer 0.6.7's autotune on SM120 allocates workspaces
faster than vLLM's measurement-based accounting tracks. On unified-memory
DGX Spark (system RAM == GPU RAM), the GPU OOM kills the host kernel.
The "Estimated CUDA graph memory: NEGATIVE" canary is a *symptom* (vLLM's
free-memory measurement is corrupted by flashinfer's concurrent allocations
during graph capture profiling), not a cause.

The negative-canary value isn't always exactly -2.24 GiB; observed
-1.66, -1.97, -2.36, -1.72, -2.24, -2.32 across 5+ runs. Variance is
because vLLM measures `before_capture - after_capture` per CUDA graph
shape, and the value depends on what flashinfer has allocated between
those two measurements.

### Fix

`--kernel-config '{"enable_flashinfer_autotune": false}'` — already
documented in `memory:feedback_flashinfer_autotune_sm120`. **This is the
permanent fix on SM120 + flashinfer 0.6.7, not a temporary diagnostic.**

### Side issue: disk full

`/dev/nvme0n1p2` was at 100% (2.2 GiB free of 916 GB) at session start.
Real but separate problem. Symptoms: `kdump_lock` from earlier panics
was 0 bytes (kernel had no room to write the dump).

**Cleanup performed:**
- `docker builder prune --force` → 89.21 GB reclaimed
- Removed 12 phase-debug image tags (most layers shared with active
  `gb10` so net disk freed was small but bookkeeping cleaner)

End state: 106 GB free / 88% used.

## 2. The real bug — fused path is broken

### What the 8-question sanity was hiding

The handoff v1 §2 reported "β-lite GSM8K 8/8 PASS". Re-tested in this
session under the same config: **7/8 (Q2 wrong, byte-identical
"120/12. 2. 2 2" output)**. Either the v1 author misread the script's
permissive `verdict: PASS` (≥6/8) as 8/8, or it was sampling variance.

`docs/research/phase_a_gsm8k_repro/2026-04-20/summary.md` from 4 days ago
investigated this exact thing on the previous fused-path stack and
concluded: "the Phase A 8/8 was a lucky roll … Prior 8/8 claims were
sampling variance." The 8-Q guided-prompt sanity is too noisy and too
short (max_tokens=16) to reveal anything real.

### What the 50-question real-GSM8K eval shows

Same 50 random questions (seed=42), same image, same autotune-OFF, only
difference: `CUTE_*_FUSION=0` vs `CUTE_*_FUSION=1`.

| Leg | Result |
|---|---|
| Unfused (`CUTE_*_FUSION=0`) | **48/50 (96.0%)**, 0 errors, total 1164.2s, ~23s/Q |
| Fused (`CUTE_*_FUSION=1`) | **1/24 answered before kill** — Q10=128 (gold=122, WRONG); Q1–9 + Q11–24 ALL timeout (180s, empty) |

### Why fused timed out

Engine reported 0.8 tokens/s sustained throughput while serving fused.
At 0.8 tok/s the eval's 180s timeout = ~144 tokens; real GSM8K needs
~250 tokens of chain-of-thought. The single Q10 that finished did so in
77s (~62 tokens — a short answer).

This is ~25× slower than unfused (which served ~11–22 tok/s).

The 8-Q sanity uses `max_tokens=16` (very short, by design — guided
prompts give the model nearly all the work). 16 tokens at 0.8 tok/s =
20 seconds per question — same order as unfused. So the perf collapse
was completely invisible to the 8-Q sanity. **This is the answer to
why "8/8" claims persisted.**

### What this falsifies

- "Q2 specifically is the bug" — false. Q2 was a noise sample. Real
  evidence: fused path fails most questions when the timeout-per-question
  exceeds the 0.8 tok/s × 180s budget.
- "It's a Qwen3.5-27B NVFP4 quant problem" — false. Unfused on the
  same model = 96.0%. Quant is healthy.
- "It's a β-coop-specific math bug" — false. β-lite (lite path) under
  fusion-ON shows the same broken behavior; the lite-vs-coop distinction
  is downstream of a more general fused-path issue.

### What this rules in (next-session ranking)

1. **Per-step latency in the fused path.** 0.8 tok/s on a 27B model is
   absurd. Something in the fused critical path is doing dramatically
   more work than the unfused equivalent — a per-token Python op call
   that's not graph-captured, a misconfigured launch grid, or a
   re-compile-per-step regression. nsys profile of one fused decode
   step vs one unfused decode step will pinpoint it in <30 minutes.
2. **The Q10 wrong answer (128 vs 122).** Once perf is back, run the
   50-Q fused leg to completion and see if accuracy is also lower.
   If accuracy ≈ unfused, Q10 was just a single-sample miss; if it's
   <90%, the fused path has a real numerical issue independent of the
   perf bug.
3. **Possible interaction: PIECEWISE + opaque-op + fusion-on.** Phase F.1
   landed opaque ops (Tasks 9–14 this session). Those ops are graph-
   capturable; if fusion-on configuration paths around them are NOT
   graph-captured, every step pays the eager-mode penalty. Bisect:
   try fusion-on with `--enforce-eager` to see if eager has the same
   0.8 tok/s ceiling. If yes, problem is in the fused kernels themselves.
   If no, problem is graph-capture-vs-fusion interaction.
4. **All previous "fused path is correct + faster" claims are now
   suspect.** The 2026-04-23 "Phase E SHIPPED" benchmarks at
   `benchmarks/nvllm/traces/phase_e/2026-04-23-initial/` show 51.7%
   per-layer-decode speedup. That number was per-layer μs from nsys —
   *kernel* latency, not end-to-end throughput. With fusion adding
   per-step Python overhead worth ~25× wall-clock, the per-kernel speedup
   is being dwarfed by something outside the kernel. This matches
   `memory:project_phase_e_phantom_speedup` exactly: kernels run, but
   their outputs are orphaned/not-on-the-critical-path, so end-to-end
   doesn't see the speedup.

## 3. Plan status update

| Plan task | v1 status | v2 status |
|---|---|---|
| 1–14 (math + opaque ops + decoder wiring + unit tests) | ✅ | ✅ unchanged |
| 15a β-lite 8-Q sanity | ✅ "8/8 PASS" claimed | ⚠️ re-tested 7/8; sanity is too noisy to be a gate |
| 15b β-coop 8-Q sanity | ⏸️ blocked | ⚠️ tested 7/8 (Q2 wrong, byte-identical to β-lite — shared bug, not β-coop-specific) |
| **NEW gate: 50-Q real-GSM8K unfused** | n/a | ✅ **96.0%** — quant + model proven healthy |
| **NEW gate: 50-Q real-GSM8K fused** | n/a | ❌ **broken** — engine throughput collapses to 0.8 tok/s |
| 16 kernel-count trace | ⏸️ | **still blocked** by fused-path bug above |
| 17 numerical equivalence | ⏸️ | **upgraded** — needed to compare fused vs unfused activations once perf is back |
| 18 evidence bundle | ⏸️ | still blocked |
| 19 memory updates | ⏸️ | partially executed in this handoff (see §5) |

**Bottom line:** plan is paused at task 15. Tasks 16–19 are blocked
behind a fused-path correctness/perf bug. The 8-Q sanity was the wrong
gate; replace with 50-Q real-GSM8K seed=42.

## 4. What's safe to ship today

- **Unfused path serving** — proven `48/50 (96.0%)` on 50-Q real GSM8K.
  Throughput viable (~23s per long-form question at autotune-OFF).
- **The autotune-OFF workaround as the permanent SM120 + flashinfer
  0.6.7 config** (memory: `feedback_flashinfer_autotune_sm120`).
- **Phase E.2 + F.1 code as committed** — math fixes are correct (verified
  all 4 production γ sites use `(1+γ)`). Code lands as merged but the
  fused-path *runtime* cannot be flipped on for production until the
  fused-perf bug is fixed.

## 5. Memory update directives (do these next session)

- **Update `feedback_flashinfer_autotune_sm120`:** confirm this is the
  permanent fix on SM120 + flashinfer 0.6.7, not a diagnostic-only
  workaround.
- **Add new memory `project_fused_path_perf_collapse`:** at the time
  of writing, fused path serves at ~0.8 tok/s vs unfused ~15 tok/s on
  Qwen3.5-27B-NVFP4, max_tokens=512, max-num-seqs=1, PIECEWISE,
  autotune-OFF. 50-Q real-GSM8K eval: 1/24 answered before kill. Root
  cause TBD; nsys decode-step diff is the recommended first probe.
- **Update `project_phase_e_phantom_speedup`:** the 50-Q evidence
  *strongly* corroborates that earlier "Phase E shipped" numbers were
  per-kernel μs, not end-to-end. Tighten this memory's wording.
- **Replace `feedback_post_quant_sanity`** wording: "GSM8K as fast canary
  after quant; >5% drop = bad quant" — but make explicit that 8-Q is too
  noisy and the canary should be ≥50-Q real GSM8K seed=42.

## 6. Cleanup state

- Container `nvllm-phase_f1` — stopped and removed at session end.
- Image `nvllm:gb10` (`debb0fa1dd29`) — kept (needed for next session).
- Old phase-debug image tags — removed (12 tags).
- New helper script `scripts/gsm8k_eval_50.py` — committed-ready;
  reads canonical `~/.cache/huggingface/datasets/openai___gsm8k/...arrow`
  path with parquet fallback. Use this as the new sanity gate.
- Disk: 106 GB free / 88% used.

## 7. Lessons for the AI (me)

- **Stop trusting one-shot 8-Q sanities.** "8/8 PASS" was reported in
  the v1 handoff and was either misread or sampling-variance. The
  prior `2026-04-20/summary.md` had already warned: "every kernel-quality
  harness must be x5+ with dedup before claiming 'works.'" I missed it.
- **Two earlier hypotheses I committed to before falsifying:**
  - "Disk full is the cause" — partially true (real problem, fixed),
    but didn't fix the wedge. Should have probed before committing.
  - "β-coop has a math bug β-lite doesn't" — falsified by running β-lite
    control. The right test was always "shared infrastructure first,
    path-specific second."
- **Persistent evidence beats tmpfs evidence.** Earlier evidence was
  written to `/tmp` and lost to a panic-reboot. Today's session writes
  to `docs/research/phase_f1_opaque_gate/run_logs/` (on disk) which
  survived all crashes. Make this the default for any
  expected-to-crash investigation.
- **Use the existing investigation record.** `phase_a_gsm8k_repro/`
  from 2026-04-20 had already nailed the "Q2 is sampling variance"
  conclusion. I should have searched the repo for prior Q2 evidence
  before chasing it from scratch.

## 8. Direct next-session checklist

1. nsys-profile one fused decode step vs one unfused decode step.
   Diff per-kernel μs. The kernel(s) >10× slower on fused are the bug.
   Per memory `feedback_nsys_privileged` + `feedback_vllm_profiling`.
2. Try fused-path with `--enforce-eager`. If 0.8 tok/s persists, the
   bug is in the kernel itself (not graph capture). If it's much
   faster eager, the bug is in graph-capture-vs-fusion interaction.
3. Once fused perf is back, rerun 50-Q seed=42 (`scripts/gsm8k_eval_50.py`)
   on fused. Compare to unfused 48/50 = 96.0%. Anything below 90% is
   a real numerical issue worth Plan Task 17 numerical equivalence
   investigation.
4. Update memory entries per §5.
5. Resume plan tasks 16–19.

## 9. Hand-off summary

We thought we were debugging a deterministic CUDA-graph crash plus a
β-coop math bug. We were actually debugging:
(a) a flashinfer 0.6.7 autotune memory leak on SM120 (fixed by config),
(b) a dirty disk that was making symptoms scarier (fixed by cleanup),
(c) a previously-undetected fused-path correctness *and* perf collapse
that the 8-Q sanity test was designed to miss. The fused path on
Qwen3.5-27B-NVFP4 is currently unusable for real workloads at
~0.8 tok/s vs ~15 tok/s unfused. Quant + model are healthy (96.0%
unfused on 50-Q real GSM8K). Plan is paused at task 15. Tasks 16–19
are gated on the fused-path perf bug, which is the highest-priority
next-session item.
