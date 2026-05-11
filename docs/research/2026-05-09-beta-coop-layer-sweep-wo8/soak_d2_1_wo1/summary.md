# D2.1 bisection leg — 12L_3_47 + `CUTE_WO_SPLIT=1`

- generated: 2026-05-11 19:10
- arm: 12L_3_47_d2_1_wo1
- git_sha: 6603532b56acd403b9e166aa04c3c1c3cb270e11
- image_id: nvllm:gb10@9c0f1d31c92c
- phase_e_layers: `3,7,11,15,19,23,27,31,35,39,43,47`
- wo_split: **1** (delta vs Stage 2b base, which was `wo_split=8`)
- n_runs: 5
- gsm8k_floor: 45
- container_alive_at_end: True
- docker_log_corruption_hits: 0
- **gate_d2_1_pass: False**

## Hypothesis tested

If the persistent β-coop workspace corruption substrate lives in the W_O reduction path (pre-WO wait, W_O slot reduction, or `wo_output` slot layout — i.e. the **wo_split-specific** plumbing), then forcing `CUTE_WO_SPLIT=1` (unsplit W_O reduction, the configuration used through 2026-04) should **eliminate or alter** the Stage 2b collapse shape.

Outcome interpretation table:
- **Stable (≥ 45/50 across all 5 runs):** wo_split is the substrate. Move to W_O reduction code-review.
- **Unstable (matches Stage 2b shape):** wo_split is not the substrate. Suspect lives upstream of W_O reduction. Move to D2.2 (`CUTE_PHASE_E_PATH=lite`).

## Per-run headline

| run | correct | errors | wall (s) | pass | notes |
|---|---|---|---|---|---|
| 1 | 48/50 | 0 | 5709 | true | clean |
| 2 | 49/50 | 0 | 5786 | true | clean |
| 3 | 48/50 | 0 | 5615 | true | clean |
| 4 | 11/50 | 0 | 14543 | false | **collapse at Q12, persists through Q49** |
| 5 | 36/50 | 0 | 8917 | false | **inherits collapse Q0–Q13, sharp recovery at Q14** |

Stage 2b reference (for shape comparison, same arm, `wo_split=8`):

| run | correct | wall (s) | shape |
|---|---|---|---|
| 1 | 48/50 | 4443 | clean |
| 2 | 48/50 | 4317 | clean |
| 3 | 48/50 | 4388 | clean |
| 4 | 11/50 | 11004 | collapse at Q12 |
| 5 | 37/50 | 6626 | recovery at Q14 |

## Per-question shape — Run 4 (collapse onset)

Q12 is the **identical** collapse onset index as Stage 2b. Wall-time signature is identical: ~345s/question (max_tokens timeout, no early stop) once collapsed, vs 70–145s/question while healthy.

| Q | gold | got | elapsed | status |
|---|---|---|---|---|
| Q0 | 2280 | 2180 | 104.8s | WRONG (model blind spot, also miss in Stage 2b) |
| Q1–Q11 | — | — | 70–290s | OK |
| **Q12** | 36 | 1 | **346.5s** | **collapse onset** |
| Q13–Q49 | — | `1` or `(empty)` | 345–347s | WRONG (max-tokens degenerates) |

## Per-question shape — Run 5 (inherited + sharp recovery)

Q14 is the **identical** recovery index as Stage 2b. Wall-time recovery is sharp: 345.7s broken at Q13 → 86.0s clean at Q14. Identical signature.

| Q | status | elapsed |
|---|---|---|
| Q0–Q13 | WRONG (inherited collapse from Run 4) | 345–348s |
| **Q14** | **OK** (gold=5, got=5) | **86.0s** |
| Q15–Q49 | all OK except Q44 (single noise miss, also Stage 2b blind spot) | 54–232s |

## Verdict — wo_split is NOT the substrate

The D2.1 leg with `CUTE_WO_SPLIT=1` reproduced the Stage 2b collapse **exactly**:
- Same arm-of-collapse run (Run 4)
- Same collapse onset index (Q12)
- Same recovery index (Q14)
- Same wall-time signatures (~345s broken vs <150s clean)
- Same accuracy bracket (11/50 collapse run, 36–37/50 recovery run)

This **eliminates** the wo_split-specific plumbing as the substrate. Pre-WO-wait, W_O slot reduction, and `wo_output` slot layout are NOT where the persistent corruption lives.

The leading hypothesis narrows to:
- **β-coop kernel's other persistent buffers**: `mlp_partial_fp32`, counters, barriers held in `self._phase_e_coop_*`
- OR the captured `wo_output` reset op (memset path itself, independent of how many split slots downstream code uses)
- OR the inside-kernel MLP-partial reset

## Next leg

**D2.2: `CUTE_PHASE_E_PATH=lite`.** Lite path is the β-lite kernel (paged attn only; no persistent workspace for W_O/MLP partials). If D2.2 is **stable**, the persistent workspace pattern is conclusively the substrate (lite doesn't share it). If D2.2 **collapses**, the suspect lives in phase-layer selection logic, not the coop kernel itself.

## Per-run artifacts

- [run1/gsm8k.json](run1/gsm8k.json), [run1/gsm8k.log](run1/gsm8k.log)
- [run2/gsm8k.json](run2/gsm8k.json), [run2/gsm8k.log](run2/gsm8k.log)
- [run3/gsm8k.json](run3/gsm8k.json), [run3/gsm8k.log](run3/gsm8k.log)
- [run4/gsm8k.json](run4/gsm8k.json), [run4/gsm8k.log](run4/gsm8k.log)
- [run5/gsm8k.json](run5/gsm8k.json), [run5/gsm8k.log](run5/gsm8k.log)
- [dispatch_audit.json](dispatch_audit.json), [verdict.json](verdict.json), [c2_diag_ENV.txt](c2_diag_ENV.txt), [serve.log](serve.log), [docker.log](docker.log)

## Comparison with Stage 2b

See [`../soak/summary.md`](../soak/summary.md) for the base-arm 5-run soak that established the Stage 2b collapse shape (wo_split=8). D2.1 is the wo_split-bisection leg of that diagnosis.
