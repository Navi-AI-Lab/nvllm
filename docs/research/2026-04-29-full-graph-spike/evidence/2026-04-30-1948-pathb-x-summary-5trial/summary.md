# Path B Step X — 5-trial audit-OFF reproducibility summary

5 trials at branch HEAD `68c6ab944a288d81b25da309c42e75f675431341`, audit-OFF, lower-8 FULL+β-coop. X.1-X.3 ran earlier (commit 68c6ab944); X.4-X.5 added by this dispatch.

| Trial | unique | same_pass | cross_pass | overall | evidence |
|---|---|---|---|---|---|
| 1 | 1 | true | true | true | `2026-04-30-1910-pathb-x-trial-1/` |
| 2 | 3 | false | false | false | `2026-04-30-1918-pathb-x-trial-2/` |
| 3 | 1 | true | true | true | `2026-04-30-1926-pathb-x-trial-3/` |
| 4 | 1 | true | true | true | `2026-04-30-1939-pathb-x-trial-4/` |
| 5 | 4 | false | true | false | `2026-04-30-1947-pathb-x-trial-5/` |

## Distribution context

| Run | unique | overall |
|---|---|---|
| Pre-v1 (no patch) | 3 | FAIL |
| v1 | 2 | FAIL |
| v2 Gate 1 (audit-OFF) | 4 | FAIL |
| Step 1 (audit-ON) | 1 | PASS |
| X.1 audit-OFF | 1 | PASS |
| X.2 audit-OFF | 3 | FAIL |
| X.3 audit-OFF | 1 | PASS |
| X.4 audit-OFF | 1 | PASS |
| X.5 audit-OFF | 4 | FAIL |

## Verdict logic (user-specified)

- **0/5 or 1/5 FAIL:** baseline too unstable to justify a patch from this signal alone.
- **2/5 or 3/5 FAIL:** real stochastic bug; v3 needs statistical acceptance, e.g. 5/5 PASS or 9/10 PASS audit-OFF.
- **4/5 or 5/5 FAIL:** strong target for a focused Z patch.

## This run's verdict

**2/5 FAIL** (X.2 unique=3, X.5 unique=4; X.1/X.3/X.4 unique=1 PASS) → **real stochastic bug; v3 needs statistical acceptance, e.g. 5/5 PASS or 9/10 PASS audit-OFF.**

