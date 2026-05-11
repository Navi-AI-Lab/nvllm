# β-coop layer-count sweep under wo_split=8

- generated: 2026-05-10 16:04:22
- git_sha: 9f118cdc571360b83cc3922ca9c72ce04b66c0c5
- image_id: sha256:9c0f1d31c92c29488f66a2c136183950cea787035d735ff95dd6af193740f530
- arms manifest: `/home/natfii/docker/nvllm-beta-layer-sweep-wo8/docs/research/2026-05-09-beta-coop-layer-sweep-wo8/arms.csv`
- region-timing buffer: DISABLED for the sweep (plan Risk #2). Per-call β median was captured separately in Stage 0c (5.538 ms vs ≤7 ms gate).

## Per-arm headline

| arm | fusion | layers | dispatch | GSM8K | errors | wall (s) | ok |
|---|---|---|---|---|---|---|---|
| 2L_3_7 | 1 | [3,7] | pass | 47/50 (≥45) | 0 | 3594 | true |
| 4L_3_15 | 1 | [3,7,11,15] | pass | 47/50 (≥45) | 0 | 3673 | true |
| 8L_3_31 | 1 | [3,7,11,15,19,23,27,31] | pass | 49/50 (≥45) | 0 | 3959 | true |
| 12L_3_47 | 1 | [3,7,11,15,19,23,27,31,35,39,43,47] | pass | 48/50 (≥45) | 0 | 4258 | true |
| 16L_3_63 | 1 | [3,7,11,15,19,23,27,31,35,39,43,47,51,55,59,63] | pass | 36/50 (≥45) | 0 | 8323 | false |

## Per-question miss table

Union of every question any arm got non-OK on. Cells show the model's predicted answer (or `OK` if that arm answered correctly). This table prevents accidental over-claiming from a single arm's accuracy delta — a 49/50 vs 47/50 spread looks impressive in the headline but the miss-pattern below shows whether the spread is layer-correlated quality or just decode variance.

| Q (gold) | 2L_3_7 | 4L_3_15 | 8L_3_31 | 12L_3_47 | 16L_3_63 |
|---|---|---|---|---|---|
| Q0 (gold=2280) | WRONG pred=`2180` | WRONG pred=`2180` | WRONG pred=`2180` | WRONG pred=`2180` | WRONG pred=`300` |
| Q6 (gold=21) | OK | OK | OK | OK | WRONG pred=`2` |
| Q7 (gold=145) | OK | WRONG pred=`70` | OK | OK | OK |
| Q11 (gold=80) | OK | OK | OK | OK | WRONG pred=`8` |
| Q13 (gold=1430) | OK | OK | OK | OK | WRONG pred=`1` |
| Q21 (gold=2000) | WRONG pred=`5000` | OK | OK | OK | OK |
| Q23 (gold=11050) | OK | OK | OK | OK | WRONG pred=`110` |
| Q27 (gold=18) | OK | OK | OK | OK | WRONG pred=`5` |
| Q32 (gold=98) | OK | OK | OK | OK | WRONG pred=`3` |
| Q34 (gold=34) | OK | OK | OK | OK | WRONG pred=`1` |
| Q35 (gold=38) | OK | OK | OK | OK | WRONG pred=`3` |
| Q37 (gold=50) | OK | OK | OK | OK | WRONG pred=`30` |
| Q38 (gold=50) | OK | OK | OK | OK | WRONG pred=`0` |
| Q44 (gold=192) | WRONG pred=`8` | WRONG pred=`8` | OK | WRONG pred=`19` | WRONG pred=`2` |
| Q46 (gold=32) | OK | OK | OK | OK | WRONG pred=`108` |
| Q47 (gold=25) | OK | OK | OK | OK | WRONG pred=`2` |

## Verdict framing (2026-05-10)

- **Q0 (gold=2280, pred=2180 across all β-on arms)** is a stable model/eval miss across β configs, not a regression signal.
- **Q44 and the one-off Q7 / Q21 flips** look like knife-edge decode variance among 2L/4L/8L/12L, not layer-count-correlated quality.
- **The 47–49/50 spread among 2L/4L/8L/12L** should be treated as noise around a passing band, not evidence that 8L is better quality.
- **16L (36/50, 0 errors) is NOT noise.** A 12-question drop on the same seed=42 GSM8K-50 sample is real signal — 16L is **quality-blocked under Stage 1c** and excluded from the dev-baseline pick.
- **Stage 1c verdict:** 2L, 4L, 8L, 12L pass (≥45/50 with 0 errors). 16L fails.
- **Stage 2a dev-baseline pick:** **12L_3_47** (`CUTE_PHASE_E_LAYERS=3,7,11,15,19,23,27,31,35,39,43,47`) — highest β-capable full-attention layer count that passes Stage 1c. Per the plan, perf comparison among passing arms is deferred to the long-soak loop; this loop only proves correctness across the layer ladder.

### Follow-up: bisect the upper-quartet regression

16L = 12L + {51, 55, 59, 63} regressed quality. We are NOT bisecting the offending upper-quartet layer in this loop — scope was "sweep then pick". A future loop should test 13L (12L + 51), 14L (12L + 51, 55), 15L (12L + 51, 55, 59), or rotate which single upper layer is added to 12L, to isolate whether the regression is one bad layer or a cumulative effect.

## Per-arm artifacts

- [2L_3_7/summary.md](2L_3_7/summary.md), [2L_3_7/verdict.json](2L_3_7/verdict.json), [2L_3_7/gsm8k.json](2L_3_7/gsm8k.json), [2L_3_7/dispatch_audit.json](2L_3_7/dispatch_audit.json)
- [4L_3_15/summary.md](4L_3_15/summary.md), [4L_3_15/verdict.json](4L_3_15/verdict.json), [4L_3_15/gsm8k.json](4L_3_15/gsm8k.json), [4L_3_15/dispatch_audit.json](4L_3_15/dispatch_audit.json)
- [8L_3_31/summary.md](8L_3_31/summary.md), [8L_3_31/verdict.json](8L_3_31/verdict.json), [8L_3_31/gsm8k.json](8L_3_31/gsm8k.json), [8L_3_31/dispatch_audit.json](8L_3_31/dispatch_audit.json)
- [12L_3_47/summary.md](12L_3_47/summary.md), [12L_3_47/verdict.json](12L_3_47/verdict.json), [12L_3_47/gsm8k.json](12L_3_47/gsm8k.json), [12L_3_47/dispatch_audit.json](12L_3_47/dispatch_audit.json)
- [16L_3_63/summary.md](16L_3_63/summary.md), [16L_3_63/verdict.json](16L_3_63/verdict.json), [16L_3_63/gsm8k.json](16L_3_63/gsm8k.json), [16L_3_63/dispatch_audit.json](16L_3_63/dispatch_audit.json)
