# Path B Step X — audit-OFF reproducibility summary

3 trials at branch HEAD `d36abf7713dfaaf8c5beb4dd7ee2c0099428e93a`, audit-OFF, lower-8 FULL+β-coop.

| Trial | unique | same_pass | cross_pass | overall | evidence |
|---|---|---|---|---|---|
| 1 | 1 | true | true | true (PASS) | `2026-04-30-1910-pathb-x-trial-1/` |
| 2 | 3 | false | false | false (FAIL) | `2026-04-30-1918-pathb-x-trial-2/` |
| 3 | 1 | true | true | true (PASS) | `2026-04-30-1926-pathb-x-trial-3/` |

## Comparison to prior runs

| Run | unique | overall | notes |
|---|---|---|---|
| Pre-v1 (no patch) | 3 | FAIL | evidence/2026-04-30-1311 |
| v1 (1cc51ab95) | 2 | FAIL | evidence/2026-04-30-1548 |
| v2 audit-OFF Gate 1 | 4 | FAIL | evidence/2026-04-30-1805 |
| v2 audit-ON Step 1 | 1 | PASS | evidence/2026-04-30-1848-pathb-step1-dispatch-audit |
| v2 audit-OFF X.1 | 1 | PASS | trial 1 |
| v2 audit-OFF X.2 | 3 | FAIL | trial 2 |
| v2 audit-OFF X.3 | 1 | PASS | trial 3 |

## Verdict logic

- **3/3 FAIL**: bug reproducible; proceed to fresh Z-design with a stable target.
- **Mixed (1-2 PASS)**: run two more and treat as statistical. v3 success criteria become "N-trial PASS rate >= threshold," not single-run.
- **3/3 PASS**: NOT a fix declaration. Treat as baseline instability and pause before any v3 patch.

## This run's verdict

**MIXED (2/3 PASS, 1/3 FAIL).** The audit-OFF Gate 1 failure is **not** consistently reproducible at HEAD `d36abf7713dfaaf8c5beb4dd7ee2c0099428e93a`. The same env contract (`CUTE_WO_RESET_LOG=1`, `CUTE_DISPATCH_AUDIT=0`, `CUTE_FULL_GRAPH_PROBE=1`, lower-8 FULL+β-coop) yielded:
- Trial 1: `unique=1`, overall PASS
- Trial 2: `unique=3`, overall FAIL (cross_prompt also failed: A-after-B drifted from A-first)
- Trial 3: `unique=1`, overall PASS

Per the verdict logic, this is the **mixed** branch: run two more trials and treat as statistical. The original Gate 1 `unique=4` FAIL (evidence/2026-04-30-1805) and the prior v1 `unique=2` FAIL appear to belong to the same statistical distribution — a low-but-nonzero replay-coherence failure rate, **not** a deterministic bug fixed by the v2 persistent-buffer patch and **not** a deterministic regression introduced by it.

Sanity checks (all 3 trials):
- 8 unique `[CUTE_WO_RESET]` data_ptrs — v2 reset still firing
- `first-any` and `first-FULL` probes present in every run
- Time-to-/v1/models: 196 / 220 / 200 s (consistent boot)

Recommended next move per the brief: run trials 4 and 5, then form an N-trial PASS-rate threshold for any v3 patch acceptance criterion. Do not declare a fix and do not begin v3 patch design until the statistical baseline is characterized.
