# D2.3 bisection leg — 12L_3_47 + `CUTE_PHASE_E_FUSION=0`

- generated: 2026-05-13
- arm: `12L_3_47_d2_3_phaseE_off`
- git_sha: `00caa275f94db99633fe14ddffc94808e782ebf6`
- image_id: `nvllm:gb10@9c0f1d31c92c`
- phase_e_layers (env): `3,7,11,15,19,23,27,31,35,39,43,47`
- phase_e_fusion: **0** (Phase E master enable OFF)
- phase_e_path (env): `auto` (irrelevant — fusion=0 prevents either path from firing)
- wo_split: 8 (irrelevant — no β-coop kernel runs)
- n_runs: 5
- gsm8k_floor: 45
- container_alive_at_end: true
- docker_log_corruption_hits: 0
- **gate_d2_3_pass: false**

## Hypothesis tested

With `CUTE_PHASE_E_FUSION=0`, **neither** β-coop nor β-lite fires. Every layer (including the 12 FULL-attention layers that Phase E would otherwise cover) routes through the base CuTe paged attention path.

Outcome interpretation from `DIAGNOSIS.md`:
- **Stable across all 5 runs (≥ 45/50):** the entire Phase E machinery (membership check + dispatch decision + per-layer state shared between coop and lite) is the substrate. Hardening + locking to 2L_3_7 becomes the production fix.
- **Unstable (matches Stage 2b shape):** bug lives even further upstream — base CuTe paged, vLLM scheduling, or hybrid model machinery. Diagnosis arc widens; 2L_3_7 also becomes suspect under sustained load.
- **Unstable with a DIFFERENT shape:** base CuTe paged path itself is broken when forced to cover the FULL-attention-layer regime that Phase E was insulating. A new failure mode, not a milder version of the original.

The actual outcome matches the third bullet — and was not the planned bisection result.

## Dispatch audit — fusion=0 was honored

`dispatch_audit.json` summary across the 16 captured records (all 12 restricted layers × {decode dispatches captured during audit prompt}):

| field | value |
|---|---|
| `enabled` | `{false}` (all 16) |
| `use_beta_coop` | `{false}` (all 16) |
| `use_beta_lite` | `{false}` (all 16) |
| `use_fusion` | `{true}` (env knob seen at module load) |
| `coop_layers` aggregate | `[]` |
| `lite_layers` aggregate | `[]` |

The Phase E dispatch site (`_backend.py:1364–1378`) is reached but every record returns the fall-through verdict. `CUTE_PHASE_E_FALLBACK_RAISE=1` was set; container did not crash, confirming β-coop never tried to fire and silently fall back.

## Per-run headline

| run | correct | errors | wall (s) | wall (h) | pass | shape |
|---|---|---|---|---|---|---|
| 1 | 14/50 | 1 | 17,541 | 4.87 | false | **deterministic 14-OK universe, ERROR at Q14** |
| 2 | 14/50 | 1 | 17,569 | 4.88 | false | **identical to Run 1 — same OK indices, same Q14 ERROR** |
| 3 | 14/50 | 1 | 17,641 | 4.90 | false | **identical to Run 1** |
| 4 | 5/50  | 0 | 19,130 | 5.31 | false | **survival window collapses to early-only `[1,4,7,8,11]`** |
| 5 | 9/50  | 0 | 18,258 | 5.07 | false | **survival window flips to late-only `[1,16,17,22,24,26,31,39,48]`** |

Walls are dominated by max-tokens timeout (~390s per WRONG row) — the model fails to stop generating once it loses coherence. Compare β-coop wo8 baseline (~4400–14460 s walls; WRONG rows there were not timeout-dominated).

## Per-question OK matrix (1 = OK, 0 = WRONG, e = ERROR)

```
i   r1 r2 r3 r4 r5
 1   1  1  1  1  1     <-- only index OK in all 5 runs
 4   1  1  1  1  0
 7   1  1  1  1  0
 8   1  1  1  1  0
11   1  1  1  1  0     <-- run 4 boundary: collapse from here onward
13   1  1  1  0  0
14   e  e  e  0  0
16   1  1  1  0  1
17   1  1  1  0  1
22   1  1  1  0  1
24   1  1  1  0  1
26   1  1  1  0  1     <-- run 5 boundary: only late survivors retained
31   1  1  1  0  1
39   1  1  1  0  1
48   1  1  1  0  1
```

All 35 other indices (Q0, Q2-Q3, Q5-Q6, Q9-Q10, Q12, Q15, Q18-Q21, Q23, Q25, Q27-Q30, Q32-Q38, Q40-Q47, Q49) were WRONG in **every** run. Those constitute the "always-broken" set under FUSION=0.

## Failure-mode signature — different from Stage 2b / D2.1 / D2.2

The Stage 2b family of soaks (β-coop wo8, β-coop wo1, β-lite wo8) all produced:
- Runs 1-3 clean (48/50)
- Run 4 sharp collapse onset at **Q12**, persisting through Q49 with **`got=1` or empty** at max-tokens
- Run 5 inherited collapse through Q13, **sharp recovery at Q14**, single-noise miss at Q44

D2.3 produces a **structurally different** shape:
- Runs 1-3 are deterministic-broken at 14/50 from Q0 — there is no clean window
- WRONG rows in runs 1-3 hit max-tokens cap (~390s per row) — the model rambles instead of converging
- Runs 4-5 reveal cross-run **state dependence** even with FUSION=0: the answerable set is the same 14 indices in runs 1-3, then collapses to early-only in run 4 and flips to late-only in run 5
- The Q12/Q14 collapse-onset/recovery-onset signature is **absent**

This is not a milder version of the original bug. It is a different failure mode with a different temporal shape.

## Four-soak comparison (all 12L_3_47 arms)

| metric | Stage 2b (coop, wo8) | D2.1 (coop, wo1) | D2.2 (lite, wo8) | **D2.3 (fusion=0)** |
|---|---|---|---|---|
| Run 1 correct | 48/50 (4443s) | 48/50 (5709s) | 48/50 (5644s) | **14/50 (17541s)** |
| Run 2 correct | 48/50 (4317s) | 49/50 (5786s) | 48/50 (5652s) | **14/50 (17569s)** |
| Run 3 correct | 48/50 (4388s) | 48/50 (5615s) | 48/50 (5648s) | **14/50 (17641s)** |
| Run 4 correct | 11/50 (11004s) | 11/50 (14543s) | 11/50 (14461s) | **5/50 (19130s)** |
| Run 5 correct | 37/50 (6626s) | 36/50 (8917s) | 36/50 (8799s) | **9/50 (18258s)** |
| Collapse onset | Q12 (run 4) | Q12 (run 4) | Q12 (run 4) | **none — broken from Q0** |
| Recovery onset | Q14 (run 5) | Q14 (run 5) | Q14 (run 5) | **none** |
| Single-noise blip | Q44 (run 5) | Q44 (run 5) | Q44 (run 5) | **absent** |
| WRONG-row elapsed | mostly normal | mostly normal | mostly normal | **~390s max-tokens timeout** |
| Failure mode | sustained-load latent | sustained-load latent | sustained-load latent | **structural / from-Q0** |

## Verdict

1. **Phase E machinery alone is NOT the substrate.** If it were, FUSION=0 would be the safe path — instead it is the worst path.
2. **Base CuTe paged path is broken (or far more numerically fragile) when forced to cover the 12 FULL-attention layers** of Qwen3.5 hybrid. Phase E was, in effect, masking a base-path issue on those layers.
3. **Cross-run state dependence still exists with FUSION=0** (runs 1-3 identical, runs 4-5 different shape) — the substrate has run-to-run-coupled state that survives container-uptime accumulation even without any Phase E buffer.
4. **The Q12-collapse signature was a function of Phase E coverage**, not just of the underlying defect. With Phase E off, the underlying defect manifests as immediate structural breakage rather than latent collapse-then-recovery.

## Coverage caveat (load-bearing for this verdict)

Steady-state passing runs in Stage 2b / D2.1 / D2.2 validated the **default-enabled path**, NOT the base path. In Qwen3.5 hybrid, Phase E covers the 12 FULL-attention layers; base CuTe paged covers the 36 linear-attn (FLA) layers when Phase E is on. With Phase E off, base path now covers FULL-attention layers **for the first time** in our evidence. The 48/50 runs from prior soaks do **not** establish base-path correctness in this regime; D2.3's verdict is consistent with that distinction.

## Cherry-pick suspect surface (off-default regime)

PR Navi-AI-Lab/nvllm#10 (`cherry-pick/upstream-stabilization-tier1`) is the only material recent change to base-path-adjacent code on this codebase. Suspects, ranked by base-path exposure:

1. **`9e3a48cd8` — KV cache stride canonicalization for TMA alignment** *(first revert probe target).* TMA is not present on SM120 but the canonicalized layout may still be exercised by base path on FULL-attention layers that Phase E was bypassing.
2. **`b383774ad` — fix(FLA): tighten write-side guard against NULL_BLOCK_ID=0** — lower suspicion unless shared block/table/null-id plumbing leaks into the FULL-attention path.
3. **`f3b4d3d09` — port: Gemma4 EAGLE-3 mixin + sliding-window cache realignment** — only matters if it touched shared cache layout.
4. **`884b5ae34` — Disable flashinfer autotune** — least suspicious; would only matter if backend selection changed.

## Next legs (user-gated)

**D2.5 — 2L_3_7 survival control soak.** `CUTE_PHASE_E_LAYERS=3,7`, FUSION=1, PATH=auto, WO_SPLIT=8 (production default). Discriminator: does *partial* Phase E coverage rescue the system? If D2.5 is stable across 5 runs, the production default is safe and the regression is gated on raising layer cardinality.

**D2.6 (contingency, single-revert probe).** Triggered only if D2.5 stable AND D2.3 deterministically broken across all 5 runs. Revert **only** `9e3a48cd8`, repeat D2.3 shape. **Not a full cherry-pick sweep** — only widen to additional reverts if the single revert does not move the result.

Side-hardening to land independently of the diagnosis arc: tighten `_phase_e_env_config` at `_backend.py:139` to **fail closed** when `CUTE_PHASE_E_LAYERS` is malformed (currently silently turns into "all layers β-coop").

## Per-run artifacts

- [run1/gsm8k.json](run1/gsm8k.json), [run1/gsm8k.log](run1/gsm8k.log)
- [run2/gsm8k.json](run2/gsm8k.json), [run2/gsm8k.log](run2/gsm8k.log)
- [run3/gsm8k.json](run3/gsm8k.json), [run3/gsm8k.log](run3/gsm8k.log)
- [run4/gsm8k.json](run4/gsm8k.json), [run4/gsm8k.log](run4/gsm8k.log)
- [run5/gsm8k.json](run5/gsm8k.json), [run5/gsm8k.log](run5/gsm8k.log)
- [dispatch_audit.json](dispatch_audit.json), [verdict.json](verdict.json), [c2_diag_ENV.txt](c2_diag_ENV.txt), [serve.log](serve.log), [docker.log](docker.log)

## Comparisons

- [`../soak/summary.md`](../soak/summary.md) — Stage 2b base soak (coop, wo8) establishing the original collapse shape.
- [`../soak_d2_1_wo1/summary.md`](../soak_d2_1_wo1/summary.md) — D2.1 (coop, wo1) eliminating wo_split as substrate.
- [`../soak_d2_2_lite/summary.md`](../soak_d2_2_lite/summary.md) — D2.2 (lite) falsifying the persistent β-coop workspace hypothesis.
