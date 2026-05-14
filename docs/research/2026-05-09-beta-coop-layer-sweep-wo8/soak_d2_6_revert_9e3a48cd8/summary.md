# D2.6 bisection leg — single-revert probe of `9e3a48cd8`

- generated: 2026-05-14
- arm: `2L_3_7_d2_6_revert_9e3a48cd8`
- git_sha: `6c0ca5824b97b940acbbeb68ff3ab9179da52dfa`
- reverted_commit: `9e3a48cd8` ("port: KV cache stride canonicalization for TMA alignment")
- image_id: `nvllm:gb10-d2_6@b2fe34355d26`
- phase_e_layers: `3,7` (production default)
- phase_e_fusion: **1** (Phase E master enable ON)
- phase_e_path: `auto` → β-coop (production default)
- wo_split: 8 (production default)
- n_runs: 5
- gsm8k_floor: 45
- container_alive_at_end: true
- docker_log_corruption_hits: 0
- **gate_d2_6_pass: false**

## Hypothesis tested

After D2.5's 4-way exact shape match across cardinalities and kernel paths,
the only remaining live hypothesis was the PR Navi-AI-Lab/nvllm#10
(`cherry-pick/upstream-stabilization-tier1`) cherry-pick surface. D2.6
isolates the highest-ranked suspect — `9e3a48cd8` (KV cache stride
canonicalization for TMA alignment) — by reverting *only* that one commit
on top of the D2.5 baseline.

Discriminator:
- **Stable (5×≥45/50):** `9e3a48cd8` is the substrate. Side-port revert into
  a release branch + re-trigger PR #10 with the revert excluded.
- **Unstable matching Stage 2b shape:** `9e3a48cd8` is not the substrate.
  The cherry-pick sweep widens to additional reverts (`b383774ad`,
  `f3b4d3d09`, `884b5ae34`), OR the substrate predates PR #10 entirely
  and the next leg is a **pre-PR#10 baseline** soak.
- **Unstable with a structurally-different shape:** `9e3a48cd8` interacts
  with the substrate but isn't fully responsible. Documented as a partial
  perturbation; widen the sweep.

The actual outcome matches the third bullet — same magnitude (11/50 Run 4),
**shifted per-question signature**.

## Dispatch audit — production-default knobs honored

`dispatch_audit.json` summary across 16 captured records:

| field | value |
|---|---|
| `coop_layers` (aggregate) | `[3, 7]` |
| `lite_layers` (aggregate) | `[]` |
| `use_beta_coop` | `true` on layers 3,7; `false` elsewhere |
| `use_beta_lite` | `false` everywhere |
| `use_fusion` | `true` |

Production-default routing confirmed: β-coop fires on layers 3 and 7, every
other restricted-layer membership check correctly skips, no β-lite
contamination.

## Per-run headline

| run | correct | errors | wall (s) | pass | shape |
|---|---|---|---|---|---|
| 1 | 47/50 | 0 | 3,607 | true | clean |
| 2 | 47/50 | 0 | 3,609 | true | clean (identical to Run 1) |
| 3 | 47/50 | 0 | 3,608 | true | clean (identical to Run 1) |
| 4 | 11/50 | 0 | 8,883 | false | **collapse onset at Q13 (shifted +1 from D2.5)** |
| 5 | 33/50 | 0 | 5,847 | false | **inherited Q3–Q16 collapse, recovery at Q17 (shifted +3 from D2.5), noise at Q1+Q22+Q45** |

## Per-question shape — Run 4 (collapse onset shifted Q12 → Q13)

| Q | gold | got | elapsed | status |
|---|---|---|---|---|
| Q0 | 2280 | 2180 | 58.8s | WRONG (Stage 2b model blind spot) |
| Q1–Q11 | — | — | 42–177s | OK (all 11 indices match) |
| **Q12** | **80** | **80** | **78.7s** | **OK (was WRONG in D2.5)** |
| **Q13** | 36 | 1 | **211.6s** | **collapse onset (max-tokens runaway, shifted +1 from D2.5)** |
| Q14–Q49 | — | `1` or `(empty)` | 210–212s | WRONG (max-tokens runaway) |

The collapse magnitude is identical to D2.5 (11/50 OK), but the boundary
shifted by exactly **one question**. Q12 (`80`) is now in the OK set;
Q13 (`36→1`) becomes the new collapse-onset row.

## Per-question shape — Run 5 (recovery shifted Q14 → Q17)

| Q | status | notes |
|---|---|---|
| Q0 | OK | gold=2280, recovered (model blind spot Q0 still got it) |
| Q1 | WRONG | noise blip (new vs D2.5) |
| Q2 | OK | |
| Q3–Q16 | WRONG | **inherited collapse from Run 4, extended 3 indices later than D2.5** |
| **Q17** | **OK** | **recovery onset (shifted +3 from D2.5's Q14)** |
| Q18–Q49 | mostly OK, except Q22, Q45 | Two recovery-region noise misses |

D2.6 recovery onset is **Q17**, compared to D2.5's Q14 — three indices
later. Recovery-region noise misses also shifted: D2.5 had Q21+Q44;
D2.6 has Q22+Q45 (each shifted +1), plus a new Q1 blip.

## Five-soak comparison (D2.5 vs D2.6 emphasized)

| metric | Stage 2b (12L coop wo8) | D2.1 (12L coop wo1) | D2.2 (12L lite wo8) | **D2.5 (2L coop wo8)** | **D2.6 (revert 9e3a48cd8)** |
|---|---|---|---|---|---|
| Run 1 | 48/50 (4443s) | 48/50 (5709s) | 48/50 (5644s) | 47/50 (3588s) | **47/50 (3607s)** |
| Run 2 | 48/50 (4317s) | 49/50 (5786s) | 48/50 (5652s) | 47/50 (3588s) | **47/50 (3609s)** |
| Run 3 | 48/50 (4388s) | 48/50 (5615s) | 48/50 (5648s) | 47/50 (3588s) | **47/50 (3608s)** |
| Run 4 | 11/50 (11004s) | 11/50 (14543s) | 11/50 (14461s) | 11/50 (8851s) | **11/50 (8883s)** |
| Run 5 | 37/50 (6626s) | 36/50 (8917s) | 36/50 (8799s) | 35/50 (5523s) | **33/50 (5847s)** |
| Collapse onset | Q12 | Q12 | Q12 | Q12 | **Q13 (+1)** |
| Recovery onset | Q14 | Q14 | Q14 | Q14 | **Q17 (+3)** |
| Run 5 noise misses | Q44 | Q44 | Q44 | Q21+Q44 | **Q1+Q22+Q45** |
| Failure mode | Stage 2b shape | Stage 2b shape | Stage 2b shape | Stage 2b shape | **Stage 2b family, shifted** |

**Magnitudes match** (11/50 Run 4 across all five arms) but **D2.6 is the
first soak where the per-question signature drifts**. The Stage 2b
*family* shape is preserved (Run 4 collapse + Run 5 inherited-then-recover),
but the onset/recovery indices shifted by +1 and +3 respectively.

## Verdict

1. **`9e3a48cd8` is NOT the substrate.** Reverting it does not restore
   stability. The Run 4 magnitude (11/50) and Stage 2b family shape both
   persist.
2. **`9e3a48cd8` DOES perturb the substrate.** Run 4 collapse onset
   shifted Q12 → Q13. Run 5 recovery onset shifted Q14 → Q17. This is
   real — the prior four soaks were per-question-identical at these
   indices. The KV stride canonicalization change influences which
   specific tokens land in the collapse region, but does not gate the
   collapse itself.
3. **The substrate lives elsewhere.** Either (a) in the other PR #10
   commits we have not yet reverted (`b383774ad`, `f3b4d3d09`,
   `884b5ae34`), or (b) outside the PR #10 surface entirely. The
   shape-shift suggests it is sensitive to **memory layout / KV-cache
   addressing**, which raises suspicion of `b383774ad` (FLA NULL_BLOCK_ID
   guard, also touches block-table plumbing).
4. **The shape-shift is interpretable evidence about the substrate's
   mechanism.** A bug that responds to KV-stride changes by *shifting*
   rather than *disappearing* is plausibly:
   - a memory-aliasing bug where stride affects which slots collide
   - an off-by-one in block-table addressing where stride changes
     which block gets corrupted first
   - a workspace-zeroing miss where the canonical stride was masking
     a stale write into the next-iteration KV region
   None of these are pinned down — D2.6 narrows the search space, it
   does not close it.

## Next legs (user-gated)

Two viable paths; **the second is more decisive**.

### Option A — continue single-revert sweep within PR #10

D2.7: revert `b383774ad` on top of the D2.6 baseline (cumulative reverts).
D2.8: revert `f3b4d3d09`. D2.9: revert `884b5ae34`. Each leg is a
~9 hour soak. Worst case: all four reverted, still collapsing → 4 wasted
soaks before the next decision point. Best case: a single revert lands
within 1-3 legs.

### Option B — pre-PR#10 baseline soak (recommended)

Build from commit `9f118cdc5` (the commit immediately before
`c7614342d` merged PR #10). Run the same 2L_3_7 production-default soak.

Outcomes:
- **Stable (5×≥45/50):** confirms the substrate entered with PR #10.
  Cumulative-revert sweep within PR #10 is justified.
- **Unstable with Stage 2b shape:** the substrate predates PR #10. The
  cherry-pick sweep is dead-end; the next investigation widens to older
  code (β-coop kernel internals, base CuTe paged path, hybrid model
  machinery). **Saves ~3 soaks of wasted time vs. Option A worst case.**

Recommended: **Option B** (pre-PR#10 baseline) first, then conditionally
Option A.

### Side hardening (independent of bisection arc)

- Tighten `_phase_e_env_config` at `_backend.py:139` to fail closed when
  `CUTE_PHASE_E_LAYERS` is malformed.
- Codify the "5×GSM8K-50 sustained soak required for base-path-adjacent
  cherry-picks" lesson as a feedback memory + design tenet candidate,
  even before final attribution. The 5-soak pattern (Stage 2b → D2.1 →
  D2.2 → D2.5 → D2.6) demonstrates that single-run smoke gates miss
  this failure mode entirely.

## How to reproduce

```bash
# 1. Branch + revert
git checkout work/d2_6_revert_9e3a48cd8   # already at 6c0ca5824

# 2. Build (in /tmp clone — worktree breaks setuptools-scm)
git clone --branch work/d2_6_revert_9e3a48cd8 \
    git@github.com:Navi-AI-Lab/nvllm.git /tmp/nvllm-d2_6-build
cd /tmp/nvllm-d2_6-build
docker build --no-cache -f docker/Dockerfile.gb10 -t nvllm:gb10-d2_6 .

# 3. Soak (in worktree)
cd /home/natfii/docker/nvllm-beta-layer-sweep-wo8
NVLLM_IMAGE=nvllm:gb10-d2_6 \
  bash docs/research/2026-05-09-beta-coop-layer-sweep-wo8/soak_d2_6_runner.sh
```

Env vars honored by serve-cute.sh:
```
CUTE_WO_SPLIT=8
CUTE_PHASE_E_FUSION=1
CUTE_PHASE_E_PATH=auto
CUTE_PHASE_E_LAYERS=3,7
CUTE_PHASE_E_FALLBACK_RAISE=1
CUTE_PHASE_E_DISPATCH_LOG=1
```

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
- [`../soak_d2_3_phaseE_off/summary.md`](../soak_d2_3_phaseE_off/summary.md) — D2.3 (12L FUSION=0), revealed base-path fragility.
- [`../soak_d2_5_2L_3_7_control/summary.md`](../soak_d2_5_2L_3_7_control/summary.md) — D2.5 (2L_3_7 production default), 4-way Stage 2b shape match across cardinalities.
