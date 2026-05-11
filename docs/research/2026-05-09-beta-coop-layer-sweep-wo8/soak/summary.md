# Stage 2b survival soak — 12L_3_47

- generated: 2026-05-11 07:08:29
- arm: 12L_3_47
- git_sha: 9f118cdc571360b83cc3922ca9c72ce04b66c0c5
- image_id: nvllm:gb10@9c0f1d31c92c
- phase_e_layers: `3,7,11,15,19,23,27,31,35,39,43,47`
- wo_split: 8
- n_runs: 5
- gsm8k_floor: 45
- container_alive_at_end: True
- docker_log_corruption_hits: 0
- **gate_2b_pass: False**

## Per-run headline

| run | correct | errors | wall (s) | pass |
|---|---|---|---|---|
| 1 | 48/50 | 0 | 4443 | true |
| 2 | 48/50 | 0 | 4317 | true |
| 3 | 48/50 | 0 | 4388 | true |
| 4 | 11/50 | 0 | 11004 | false |
| 5 | 37/50 | 0 | 6626 | false |

## Per-question miss table

Union of every question any run got non-OK on. Cells show the model's predicted answer (or `OK` if that run answered correctly). Same anti-overclaim discipline as the sweep summary — a stable miss set across all 5 runs is the model's blind spot, not a stability problem. A miss set that grows from run to run is a state-corruption / drift signal.

| Q (gold) | run1 | run2 | run3 | run4 | run5 |
|---|---|---|---|---|---|
| Q0 (gold=2280) | WRONG pred=`2180` | WRONG pred=`2180` | WRONG pred=`2180` | WRONG pred=`2180` | WRONG pred=`(empty)` |
| Q2 (gold=5) | OK | OK | OK | OK | WRONG pred=`1` |
| Q3 (gold=12) | OK | OK | OK | OK | WRONG pred=`(empty)` |
| Q4 (gold=273) | OK | OK | OK | OK | WRONG pred=`(empty)` |
| Q5 (gold=45) | OK | OK | OK | OK | WRONG pred=`(empty)` |
| Q6 (gold=21) | OK | OK | OK | OK | WRONG pred=`(empty)` |
| Q7 (gold=145) | OK | OK | OK | OK | WRONG pred=`(empty)` |
| Q8 (gold=60) | OK | OK | OK | OK | WRONG pred=`1` |
| Q9 (gold=122) | OK | OK | OK | OK | WRONG pred=`(empty)` |
| Q10 (gold=29) | OK | OK | OK | OK | WRONG pred=`1` |
| Q11 (gold=80) | OK | OK | OK | OK | WRONG pred=`(empty)` |
| Q12 (gold=36) | OK | OK | OK | WRONG pred=`1` | WRONG pred=`1` |
| Q13 (gold=1430) | OK | OK | OK | WRONG pred=`(empty)` | WRONG pred=`(empty)` |
| Q14 (gold=5) | OK | OK | OK | WRONG pred=`1` | OK |
| Q15 (gold=5) | OK | OK | OK | WRONG pred=`(empty)` | OK |
| Q16 (gold=5) | OK | OK | OK | WRONG pred=`(empty)` | OK |
| Q17 (gold=66) | OK | OK | OK | WRONG pred=`(empty)` | OK |
| Q18 (gold=15) | OK | OK | OK | WRONG pred=`1` | OK |
| Q19 (gold=40) | OK | OK | OK | WRONG pred=`1` | OK |
| Q20 (gold=93) | OK | OK | OK | WRONG pred=`1` | OK |
| Q21 (gold=2000) | OK | OK | OK | WRONG pred=`1` | OK |
| Q22 (gold=1520) | OK | OK | OK | WRONG pred=`(empty)` | OK |
| Q23 (gold=11050) | OK | OK | OK | WRONG pred=`(empty)` | OK |
| Q24 (gold=90) | OK | OK | OK | WRONG pred=`(empty)` | OK |
| Q25 (gold=40000) | OK | OK | OK | WRONG pred=`1` | OK |
| Q26 (gold=21) | OK | OK | OK | WRONG pred=`(empty)` | OK |
| Q27 (gold=18) | OK | OK | OK | WRONG pred=`(empty)` | OK |
| Q28 (gold=14) | OK | OK | OK | WRONG pred=`1` | OK |
| Q29 (gold=23) | OK | OK | OK | WRONG pred=`1` | OK |
| Q30 (gold=145) | OK | OK | OK | WRONG pred=`(empty)` | OK |
| Q31 (gold=123) | OK | OK | OK | WRONG pred=`1` | OK |
| Q32 (gold=98) | OK | OK | OK | WRONG pred=`(empty)` | OK |
| Q33 (gold=7) | OK | OK | OK | WRONG pred=`(empty)` | OK |
| Q34 (gold=34) | OK | OK | OK | WRONG pred=`(empty)` | OK |
| Q35 (gold=38) | OK | OK | OK | WRONG pred=`(empty)` | OK |
| Q36 (gold=320) | OK | OK | OK | WRONG pred=`(empty)` | OK |
| Q37 (gold=50) | OK | OK | OK | WRONG pred=`(empty)` | OK |
| Q38 (gold=50) | OK | OK | OK | WRONG pred=`(empty)` | OK |
| Q39 (gold=84) | OK | OK | OK | WRONG pred=`1` | OK |
| Q40 (gold=50) | OK | OK | OK | WRONG pred=`1` | OK |
| Q41 (gold=8000) | OK | OK | OK | WRONG pred=`(empty)` | OK |
| Q42 (gold=280) | OK | OK | OK | WRONG pred=`1` | OK |
| Q43 (gold=30) | OK | OK | OK | WRONG pred=`1` | OK |
| Q44 (gold=192) | WRONG pred=`40` | WRONG pred=`20` | WRONG pred=`24` | WRONG pred=`(empty)` | OK |
| Q45 (gold=276) | OK | OK | OK | WRONG pred=`(empty)` | OK |
| Q46 (gold=32) | OK | OK | OK | WRONG pred=`(empty)` | OK |
| Q47 (gold=25) | OK | OK | OK | WRONG pred=`(empty)` | OK |
| Q48 (gold=10) | OK | OK | OK | WRONG pred=`1` | OK |
| Q49 (gold=84) | OK | OK | OK | WRONG pred=`(empty)` | OK |

## Verdict framing

- **Gate 2b spec:** all 5 runs ≥ 45/50 AND 0 errors AND container alive at end AND `docker.log` corruption hits = 0.
- **Observed:** container_alive=True, corruption_hits=0, gate_2b_pass=False.
- **Miss-table read:** Stable miss set across all runs => model/eval blind spot, not soak drift. New questions failing in later runs => state corruption candidate; bisect by run order.

## Per-run artifacts

- [run1/gsm8k.json](run1/gsm8k.json), [run1/gsm8k.log](run1/gsm8k.log)
- [run2/gsm8k.json](run2/gsm8k.json), [run2/gsm8k.log](run2/gsm8k.log)
- [run3/gsm8k.json](run3/gsm8k.json), [run3/gsm8k.log](run3/gsm8k.log)
- [run4/gsm8k.json](run4/gsm8k.json), [run4/gsm8k.log](run4/gsm8k.log)
- [run5/gsm8k.json](run5/gsm8k.json), [run5/gsm8k.log](run5/gsm8k.log)
- [dispatch_audit.json](dispatch_audit.json), [verdict.json](verdict.json), [serve.log](serve.log), [docker.log](docker.log)
