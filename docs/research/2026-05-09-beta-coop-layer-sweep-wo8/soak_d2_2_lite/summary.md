# D2.2 bisection leg — 12L_3_47 + `CUTE_PHASE_E_PATH=lite`

- generated: 2026-05-12 06:20
- arm: 12L_3_47_d2_2_lite
- git_sha: 8718157b7de841001e9850b10327ac94862e5b50
- image_id: nvllm:gb10@9c0f1d31c92c
- phase_e_layers: `3,7,11,15,19,23,27,31,35,39,43,47`
- phase_e_path: **lite** (delta vs Stage 2b base + D2.1, both `auto` → β-coop)
- wo_split: 8 (restored to base; D2.1 already eliminated wo_split as substrate)
- n_runs: 5
- gsm8k_floor: 45
- container_alive_at_end: True
- docker_log_corruption_hits: 0
- **gate_d2_2_pass: False**

## Hypothesis tested

The post-D2.1 leading hypothesis was that the **persistent β-coop workspace buffers** (`self._phase_e_coop_*`: `wo_output`, `mlp_partial_fp32`, counters, barriers) were the corruption substrate. The β-lite path does NOT share this workspace — lite is paged attention only, with no W_O/MLP persistent state. If the workspace pattern were the substrate, lite should remain clean across 5 runs.

Outcome interpretation table from `DIAGNOSIS.md`:
- **Stable (≥ 45/50 across all 5 runs):** β-coop kernel + persistent workspace is the substrate. Lite is the safe path.
- **Unstable (matches Stage 2b shape):** Suspect lives upstream of the kernel path choice — phase-layer selection logic, framework output routing, or something shared between coop and lite. Diagnosis arc widens.

## Per-run headline

| run | correct | errors | wall (s) | pass | shape |
|---|---|---|---|---|---|
| 1 | 48/50 | 0 | 5644 | true | clean |
| 2 | 48/50 | 0 | 5652 | true | clean |
| 3 | 48/50 | 0 | 5648 | true | clean |
| 4 | 11/50 | 0 | 14461 | false | **collapse onset Q12, persists through Q49** |
| 5 | 36/50 | 0 | 8799 | false | **inherits collapse Q0–Q13, sharp recovery at Q14** |

## Three-soak comparison (all 12L_3_47 arms)

| metric | Stage 2b (coop, wo8) | D2.1 (coop, wo1) | **D2.2 (lite, wo8)** |
|---|---|---|---|
| Run 1 | 48/50 (4443s) | 48/50 (5709s) | 48/50 (5644s) |
| Run 2 | 48/50 (4317s) | 49/50 (5786s) | 48/50 (5652s) |
| Run 3 | 48/50 (4388s) | 48/50 (5615s) | 48/50 (5648s) |
| Run 4 | 11/50 (11004s) | 11/50 (14543s) | **11/50 (14461s)** |
| Run 5 | 37/50 (6626s) | 36/50 (8917s) | **36/50 (8799s)** |
| Collapse onset | Q12 | Q12 | **Q12** |
| Recovery onset | Q14 (Run 5) | Q14 (Run 5) | **Q14 (Run 5)** |
| Single-noise blip | Q44 (Run 5) | Q44 (Run 5) | **Q44 (Run 5)** |

Three independent kernel-path configurations — β-coop with two different W_O reduction widths AND β-lite (which doesn't even use the W_O kernel) — collapse at **identical request indices** with **identical recovery indices**. The bug is fully path-agnostic.

## Per-question shape — Run 4 (collapse onset)

| Q | gold | got | elapsed | status |
|---|---|---|---|---|
| Q0 | 2280 | 2180 | 97.6s | WRONG (Stage 2b/D2.1 model blind spot) |
| Q1–Q11 | — | — | 68–282s | OK |
| **Q12** | 36 | 1 | **344.5s** | **collapse onset** |
| Q13–Q49 | — | `1` or `(empty)` | 343–345s | WRONG (max-tokens timeout) |

## Per-question shape — Run 5 (inherited + sharp recovery)

| Q | status | elapsed |
|---|---|---|
| Q0–Q13 | WRONG (inherited collapse from Run 4) | 342–346s |
| **Q14** | **OK** (gold=5, got=5) | **85.3s** |
| Q15–Q49 | all OK except Q44 (single noise miss) | 49–195s |

## Verdict — persistent β-coop workspace is NOT the substrate

The leading hypothesis from `DIAGNOSIS.md` is **falsified**. The β-lite path collapsed with the exact same shape (collapse onset Q12, recovery Q14, accuracy bracket 11/50 → 36/50) as both β-coop runs (Stage 2b wo_split=8, D2.1 wo_split=1).

Lite path does NOT share:
- `self._phase_e_coop_wo_output` buffer
- `self._phase_e_coop_mlp_partial_fp32` buffer
- The atomic-counter / barrier workspace
- The captured `wo_output` memset op (`_wo_output_reset_op.py`)
- The inside-kernel MLP-partial reset path

Yet the collapse fired at the same request index with the same recovery shape. **Therefore the persistent-workspace reset-lifecycle hypothesis is ruled out.**

## What this implies

The bug substrate must live in code paths **shared** between β-coop and β-lite — i.e., **upstream of the kernel-path choice** in `_backend.py:1364–1378`. Candidate surfaces (now elevated):

1. **Phase-layer selection logic** — `_phase_e_active`, `_PHASE_E_LAYERS` membership check, restricted-layer set parsing/caching. The same 12-element layer set is consumed by both paths.
2. **Framework output routing** — both paths exit through the same `output` tensor write path and accept_output_buffer pipeline. A shared zeroing/staging buffer could be the substrate.
3. **KV cache slot machinery under sustained 12-layer Phase E dispatch** — sustained-load triggered, sharp recovery, container alive, no error logs — matches a KV slot reuse race more than a kernel arithmetic bug. The 12-layer cardinality may be the pressure that the 2L default doesn't hit.
4. **Hybrid attention mamba/linear-attn state** (`--mamba-cache-mode align`) — shared across all kernel paths; would be triggered by sustained-load request volume regardless of which CuTe kernel runs.
5. **Pre-kernel state plumbing in `_backend.py` forward** — buffer-stage handoff between layers (per CLAUDE.md tenet "Math correct but output still wrong → audit buffer-stage plumbing").

## Next leg — D2.3 elevated

**D2.3 (`CUTE_PHASE_E_FUSION=0`)** removes the entire Phase E machinery — neither coop nor lite path runs, the model falls through to base CuTe paged attention only.

- **Stable**: Phase E machinery (membership check + dispatch decision + per-layer state) is the substrate. Both coop and lite share it via the dispatch site at `_backend.py:1364–1378`. Hardening the layer-cardinality limit + 2L_3_7 default becomes the production fix; 12L β-coop and 12L β-lite are both blocked.
- **Unstable**: Bug is even further upstream — base CuTe paged path, vLLM scheduling, or hybrid model machinery. Diagnosis arc widens significantly. Would imply 2L_3_7 is ALSO at risk under sustained load; D2.5 (2L survival control) gains urgency.

The 12-layer cardinality vs 2L cardinality hypothesis becomes high-value: if D2.3 is unstable at the same 12L cardinality (no Phase E firing), we've isolated cardinality from kernel path. If D2.3 is stable, we've isolated Phase E machinery as the substrate.

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
