# D2.5 bisection leg — 2L_3_7 production-default survival control

- generated: 2026-05-13
- arm: `2L_3_7_d2_5_control`
- git_sha: `07de4ebe71891320648547c61f4599e231b21215`
- image_id: `nvllm:gb10@9c0f1d31c92c`
- phase_e_layers: `3,7` (production default)
- phase_e_fusion: **1** (Phase E master enable ON)
- phase_e_path: `auto` → β-coop (production default)
- wo_split: 8 (production default)
- n_runs: 5
- gsm8k_floor: 45
- container_alive_at_end: true
- docker_log_corruption_hits: 0
- **gate_d2_5_pass: false**

## Hypothesis tested

After D2.3 (FUSION=0, 12L) revealed a structurally different failure mode
(deterministic-from-Q0 instead of the Q12-collapse shape), the leading
candidate substrate became **base CuTe paged path on FULL-attention layers**
that Phase E was insulating. D2.5 tests whether the **production default**
(Phase E covering only layers 3 and 7) survives the same sustained-load
shape that broke 12L_3_47.

Discriminator:
- **Stable (5×≥45/50):** cardinality is the gate. The bug is gated on
  raising `CUTE_PHASE_E_LAYERS` above 2L_3_7. Production default is safe.
- **Unstable matching Stage 2b shape:** cardinality is NOT the gate. The
  substrate is shared across cardinalities AND across kernel paths. The
  remaining live hypothesis is the PR #10 cherry-pick surface.

The actual outcome matches the second bullet — definitively.

## Dispatch audit — production-default knobs honored

`dispatch_audit.json` summary across 16 captured records:

| field | value |
|---|---|
| `coop_layers` (aggregate) | `[3, 7]` |
| `lite_layers` (aggregate) | `[]` |
| `enabled` | `true` on layers 3,7; `true` (no-op) on the other 14 records |
| `use_beta_coop` | `true` on layers 3,7; `false` elsewhere |
| `use_beta_lite` | `false` everywhere |
| `use_fusion` | `true` |

Production-default routing confirmed: β-coop fires on layers 3 and 7, every
other restricted-layer membership check correctly skips, no β-lite
contamination.

## Per-run headline

| run | correct | errors | wall (s) | pass | shape |
|---|---|---|---|---|---|
| 1 | 47/50 | 0 | 3,588 | true | clean, identical OK set |
| 2 | 47/50 | 0 | 3,588 | true | identical to Run 1 |
| 3 | 47/50 | 0 | 3,588 | true | identical to Run 1 |
| 4 | 11/50 | 0 | 8,851 | false | **collapse onset Q12, persists through Q49** |
| 5 | 35/50 | 0 | 5,523 | false | **inherits collapse Q0–Q13, sharp recovery at Q14, noise misses Q21+Q44** |

Runs 1-3 wall is 3,588 s each — substantially faster than the 12L arms
(Stage 2b ~4400 s, D2.1/D2.2 ~5700 s) because 2L pushes only 2 layers
through β-coop instead of 12.

## Per-question shape — Run 4 (collapse onset, mirrors Stage 2b exactly)

| Q | gold | got | elapsed | status |
|---|---|---|---|---|
| Q0 | 2280 | 2180 | 58.5s | WRONG (Stage 2b model blind spot) |
| Q1–Q11 | — | — | 42–176s | OK (all 11 indices match) |
| **Q12** | 36 | 1 | **210.8s** | **collapse onset** |
| Q13–Q49 | — | `1` or `(empty)` | 209–211s | WRONG (max-tokens runaway) |

## Per-question shape — Run 5 (inherited + sharp recovery)

| Q | status | elapsed |
|---|---|---|
| Q0–Q13 | WRONG (inherited collapse from Run 4) | 210–213s |
| **Q14** | **OK** (gold=5, got=5) | **52.1s** |
| Q15–Q49 | all OK except Q21 (5/12 noise), Q44 (30/1 noise) | 36–195s |

Two recovery-region noise misses (Q21, Q44) vs prior soaks' single Q44 miss —
the recovery is the same sharp categorical flip with one extra one-off blip.

## Five-soak comparison (canonical bisection table)

| metric | Stage 2b (12L coop wo8) | D2.1 (12L coop wo1) | D2.2 (12L lite wo8) | D2.3 (12L FUSION=0) | **D2.5 (2L coop wo8)** |
|---|---|---|---|---|---|
| Run 1 | 48/50 (4443s) | 48/50 (5709s) | 48/50 (5644s) | 14/50 (17541s) | **47/50 (3588s)** |
| Run 2 | 48/50 (4317s) | 49/50 (5786s) | 48/50 (5652s) | 14/50 (17569s) | **47/50 (3588s)** |
| Run 3 | 48/50 (4388s) | 48/50 (5615s) | 48/50 (5648s) | 14/50 (17641s) | **47/50 (3588s)** |
| Run 4 | 11/50 (11004s) | 11/50 (14543s) | 11/50 (14461s) | 5/50 (19130s) | **11/50 (8851s)** |
| Run 5 | 37/50 (6626s) | 36/50 (8917s) | 36/50 (8799s) | 9/50 (18258s) | **35/50 (5523s)** |
| Collapse onset | Q12 | Q12 | Q12 | **n/a (Q0)** | **Q12** |
| Recovery onset | Q14 | Q14 | Q14 | **n/a** | **Q14** |
| Failure mode | Stage 2b shape | Stage 2b shape | Stage 2b shape | structural | **Stage 2b shape** |
| Recovery noise blip | Q44 | Q44 | Q44 | n/a | **Q21 + Q44** |

**Four-way exact match on Run 4 = 11/50, Q12-onset, Q14-recovery** across:
- 12L β-coop wo8 (Stage 2b)
- 12L β-coop wo1 (D2.1)
- 12L β-lite wo8 (D2.2)
- 2L β-coop wo8 (D2.5 — production default)

## Verdict

1. **Cardinality is NOT the gate.** Production default at 2L_3_7 exhibits the
   *exact* same collapse shape (11/50 Run 4, Q12-onset, Q14-recovery) as
   three 12L_3_47 variants. The bug substrate is not the layer count.
2. **Phase E coverage doesn't mask the bug at 2L either.** Coverage of layers
   3 and 7 was hypothesized to "rescue" the FULL-attention regime that
   broke D2.3; it does not. The Stage 2b sustained-load collapse still fires.
3. **Production default is at-risk under sustained 5-run GSM8K-50 load.**
   Single-run smoke evaluations (the gate used for PR #10) pass cleanly
   (47/50 ≥ 45 floor); the failure only manifests starting at Run 4. See
   [[feedback_default_vs_base_path_coverage]] for the related coverage
   distinction.
4. **The PR #10 cherry-pick surface is the only remaining actionable
   hypothesis.** No other recent change touches base-path code on this
   codebase. D2.6 (single-revert probe of `9e3a48cd8`) is now the
   discriminative next leg — and the contingency on task #19 ("only if
   D2.5 stable") inverts: D2.5 unstable *strengthens* the case for D2.6
   by eliminating the cardinality-gated alternative.

## Methodology lesson (provisional — pending D2.6 attribution)

PR #10 ("cherry-pick/upstream-stabilization-tier1") merged with a smoke-pass
gate: container boots, single GSM8K-50 ≥ 45/50. That gate would have
shown 47/50 on D2.5's Run 1 and passed cleanly — the collapse only starts
at Run 4. If D2.6 attributes the substrate to one of the PR #10 commits,
the lesson is concrete and codifiable:

> Cherry-picks that touch base-path code (cache layout, KV stride,
> backend selection, scheduler hooks) need the full 5×GSM8K-50 sustained
> soak — not just a single-run smoke pass — because the sustained-load
> failure mode is invisible in Run 1.

This becomes a feedback memory + a design tenet candidate once D2.6 lands.

## Next leg (user-gated)

**D2.6 — single-revert probe of `9e3a48cd8`** (KV cache stride canonicalization
for TMA alignment). Revert *only* that one commit; rebuild; repeat the D2.5
shape (2L_3_7 production default, 5×GSM8K-50). If the Run 4 collapse
disappears → `9e3a48cd8` is the substrate; widen to additional reverts only
if it does not. Not a full cherry-pick sweep yet.

## Per-run artifacts

- [run1/gsm8k.json](run1/gsm8k.json), [run1/gsm8k.log](run1/gsm8k.log)
- [run2/gsm8k.json](run2/gsm8k.json), [run2/gsm8k.log](run2/gsm8k.log)
- [run3/gsm8k.json](run3/gsm8k.json), [run3/gsm8k.log](run3/gsm8k.log)
- [run4/gsm8k.json](run4/gsm8k.json), [run4/gsm8k.log](run4/gsm8k.log)
- [run5/gsm8k.json](run5/gsm8k.json), [run5/gsm8k.log](run5/gsm8k.log)
- [dispatch_audit.json](dispatch_audit.json), [verdict.json](verdict.json), [c2_diag_ENV.txt](c2_diag_ENV.txt), [serve.log](serve.log), [docker.log](docker.log)

## Comparisons

- [`../soak/summary.md`](../soak/summary.md) — Stage 2b base soak (12L coop wo8), original Q12-collapse shape.
- [`../soak_d2_1_wo1/summary.md`](../soak_d2_1_wo1/summary.md) — D2.1 (12L coop wo1), eliminated wo_split as substrate.
- [`../soak_d2_2_lite/summary.md`](../soak_d2_2_lite/summary.md) — D2.2 (12L β-lite), falsified persistent β-coop workspace.
- [`../soak_d2_3_phaseE_off/summary.md`](../soak_d2_3_phaseE_off/summary.md) — D2.3 (12L FUSION=0), revealed base-path fragility on FULL-attention layers.
