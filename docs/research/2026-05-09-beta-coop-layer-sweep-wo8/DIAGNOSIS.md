# Post-Stage-2b: β-coop sustained-load state corruption — diagnosis arc

## TL;DR

The Stage 2b survival soak at the **12L_3_47** β-coop arm (`CUTE_PHASE_E_LAYERS=3,7,11,15,19,23,27,31,35,39,43,47`, `CUTE_WO_SPLIT=8`) **fails Gate 2b**. Runs 1–3 land 48/50 cleanly; Run 4 collapses at Q12 (summary index) and never recovers (11/50); Run 5 inherits the collapsed state through Q13 then snaps back to clean output at Q14 (37/50). Container stays alive throughout, 0 errors logged, no docker-log corruption hits — the failure shape is *silent quality collapse*, not crash. Evidence: [`soak/summary.md`](soak/summary.md).

**12L_3_47 cannot ship as `serve-cute.sh` default.** 2L stays default until a root-causing fix lands.

## Hypothesis space (after Run 5 evidence)

| Hypothesis | Verdict | Rationale |
|---|---|---|
| Per-run aging (Run 5 starts clean) | **Ruled out** | Run 5's Q1–Q13 are all broken, inheriting Run 4's state. The "run boundary" is just a new GSM8K script invocation — server state is shared. |
| Monotonic degradation (gets worse forever) | **Ruled out** | Run 5 self-recovers at Q14. Bug isn't a one-way ratchet. |
| Process-level corruption, NOT recoverable | **Ruled out** | Same recovery evidence. |
| Process-level corruption, SELF-RECOVERABLE | **Confirmed** | 51 consecutive broken questions across runs 4+5, then sharp single-request recovery at Q14. No external intervention. |
| Specific prompt triggers entry | **Ruled out for Q11** | Q11 (the 220s hard problem) finished cleanly in Run 4; Q12 also passed. Collapse onset is at Run 4 Q12. Could still be Q12-triggered, but every other run handled Q12 fine. More likely accumulated state crossed a threshold during Q12's generation. |

The **sharpness** of the Q13→Q14 recovery (261s broken wall → 64.3s clean wall, with identical infra) is the strongest signal: whatever cleared the corruption flipped a *categorical* state (evicted / rotated / reset), not an analog drift.

## Strongest code-level suspect: persistent β-coop workspace/reset lifecycle

(Updated 2026-05-11 after a focused friend code-review of the launch path.)

β-coop reads its workspace buffers from **persistent impl attributes** — `self._phase_e_coop_*` (`wo_output`, `mlp_partial_fp32`, counters, barriers) — that live across requests because host-side zeroing was avoided to keep CUDA graph capture happy. Resets fire at different points:

- `vllm/v1/attention/backends/cute_paged/_backend.py:~1600` — `wo_output` reset via custom captured memset op pre-launch, then the persistent buffers are passed into β-coop
- `vllm/v1/attention/backends/cute_paged/phase_e_kernel.py:~3036-3038` — counter `zero_()` before launch (inside `run_beta_coop_full`)
- `vllm/v1/attention/backends/cute_paged/phase_e_kernel.py:~4762` — MLP partial rows reset inside-kernel
- `vllm/v1/attention/backends/cute_paged/_wo_output_reset_op.py:~110` — CUDA memset issued via libcudart wrapper (currently dedupes its log emit by `(data_ptr, nat)` — see D3)

If any reset is missed (graph-capture replay ordering quirk, off-by-one in `nat`, request shape that bypasses a path), persistent garbage stays in the buffer and feeds subsequent forward passes — exactly the "silent collapse across many requests, sharp self-recovery later" shape. The Q14 recovery would correspond to a natural overwrite/wrap/reset finally clearing the corruption.

This now leads the suspect list over generic "kernel approximation drift" (the 16L break hypothesis) because:

- The collapse threshold is **sustained-load triggered**, not single-step.
- Recovery is **sharp/categorical**, not gradual.
- The persistent-buffer surface is exactly the right shape: state that lives long, is reset by graph-captured ops at multiple sites, and would manifest as a degenerate output if a single reset misses.

## Bisection plan (D2)

Each leg = a 5-run soak (`soak_runner.sh` with a different `serve-cute.sh` env recipe), ~10h wall each. Reuses the existing infrastructure. Ordering follows the leading hypothesis (cheapest sharp signal first).

| Leg | Config delta vs base | Stable outcome reading | Unstable outcome reading |
|---|---|---|---|
| **D2.0** | none (12L + wo8 + coop) | — | reproduces Stage 2b collapse (control) |
| **D2.1** | `CUTE_WO_SPLIT=1` | suspect lives in the wo_split path (pre-WO wait, W_O slot reduction, or `wo_output` slot layout) | wo_split isn't the substrate — eliminate it |
| **D2.2** | `CUTE_PHASE_E_PATH=lite` | suspect is the **β-coop kernel + persistent workspace** (lite path doesn't share the persistent workspace pattern) | suspect lives in phase-layer selection logic, not the coop kernel itself |
| **D2.3** | `CUTE_PHASE_E_FUSION=0` | base CuTe/serve path is stable; **β-coop is the substrate** | bug is in the base CuTe path, not β-coop — diagnosis arc widens |
| **D2.4** | 8L (`3,7,11,15,19,23,27,31`) + 12L-minus-one-upper variants (`drop=47`, `drop=43`, etc.) | layer-count pressure (more β layers → faster collapse) | a specific upper layer is fragile; identify which |
| **D2.5** | `CUTE_PHASE_E_LAYERS=3,7` (2L), same container, 5 runs | **the current default is survival-safe** — memorialize | the default also collapses; this is much worse than expected |

D2.1 → D2.2 → D2.3 is the cheapest-first ordering on the leading hypothesis. D2.5 is the survival control we owe the 2L default before any user trusts it for long-running serving (e.g., the hermes agent + interactive use per `project_num_seqs_2_target`).

**Demoted from the plan — prefix-caching off:** Per `vllm/config/model.py:1791`, **hybrid attention models default to prefix caching disabled** ("Hybrid models do not support prefix caching since the feature is still experimental"). Qwen3.5 is hybrid (full + linear attention layers). So this is already off in our setup and isn't a meaningful bisection. Replaced with a one-line verification check at boot (logged once via D3 instrumentation).

**Demoted from the plan — eager-mode bisection:** SM120 platform constraint per CLAUDE.md debug protocol step 2 ("`--enforce-eager` is currently broken on SM120 — produces gibberish regardless of kernel correctness"). Eager would inject confound, not isolate.

## Instrumentation plan (D3)

The current `CUTE_WO_RESET_LOG=1` mode in `_wo_output_reset_op.py:~110` dedups by `(data_ptr, nat)` and only emits once per pair per process lifetime. **That can't prove the reset fired at Q14/Q15** — the exact data point we need.

Add a per-request, env-gated diagnostic mode (`CUTE_DIAG_REQUEST_TRACE=1`) that emits one line per request with:

- `request_id`
- question index from the eval script (via `X-Question-Index` header or `?qid=N`)
- `finish_reason` (length vs stop — distinguishes "garbage to max_tokens" from "stopped cleanly")
- `prompt_tokens`, `generated_tokens`
- **output hash** (SHA-1 of first 64 chars of generated text — proves whether identical inputs across the recovery boundary produce identical outputs)
- dispatch path per layer (β-coop / β-lite / paged) — already partially via `[PHASE_E_DISPATCH]`
- `CUTE_WO_SPLIT` value (resolved at runtime)
- `nat` (number of active tokens) at launch
- data pointers for `wo_output`, `mlp_partial_fp32`, counters, barriers
- reset call counts per buffer (cumulative since process start)
- prefix-cache enabled/disabled (resolved at boot — proves the hybrid-default-off assumption)
- free/used KV blocks at request start

Specifically remove the once-per-pair dedup in `CUTE_WO_RESET_LOG` under the diagnostic flag so per-request reset invocations are observable.

Patch locations:

- `vllm/v1/engine/processor.py` or `vllm/v1/engine/output_processor.py` — request-level emit
- `vllm/v1/attention/backends/cute_paged/_wo_output_reset_op.py` — drop dedup under diagnostic flag
- `vllm/v1/attention/backends/cute_paged/_backend.py` — emit data pointers + reset counts per layer
- `scripts/gsm8k_eval_50.py` — add `X-Question-Index` header per request

Keep all markers env-gated per `feedback_keep_debug_harnesses`.

## Side hardening (fail-closed gate)

Per friend review, `_phase_e_env_config()` at `vllm/v1/attention/backends/cute_paged/_backend.py:~139` silently turns a malformed `CUTE_PHASE_E_LAYERS` value into `restricted_layers=None` (= **all** layers β-coop), via the `except ValueError: restricted_layers = None` clause. Production-safe behavior is fail-closed: refuse to start with malformed input rather than enabling β-coop on every full-attention layer (the exact config we just proved breaks). Apply alongside D3 instrumentation.

Runner-side validation (`runner.sh` / `soak_runner.sh`) is good but the backend should not turn a typo into "all β layers."

## EngineCore env-strip workaround — proof obligation

Per friend review and per `feedback_vllm_enginecore_env_strip`: most `docker -e` env vars do not reach pid 146 (EngineCore). The workaround is a sentinel file (`/tmp/c2_diag/ENV`) read at module import in `vllm/nvllm/models/qwen3_5.py:52`. The current commit extends the sentinel-file accept-list to `CUTE_PHASE_E_*` and `CUTE_BETA_REGION_TIMING=` keys (Task 0a infra in commit 1 of this PR).

**Any leg that relies on a new `CUTE_*` env knob must:**

1. Verify `scripts/serve-cute.sh` writes the knob to `/tmp/c2_diag/ENV`.
2. Verify `vllm/nvllm/models/qwen3_5.py` accepts the knob prefix.
3. Verify the knob is honored at the dispatch site (via `[PHASE_E_DISPATCH]` log audit or equivalent).

The existing runner already gates leg start on `extract_dispatch_log.py` reading the audit log. Keep that gate live for every D2 leg.

## Decision: serve-cute defaults

- **2L_3_7 stays default.** Reverts task 3a from "update defaults to 12L" to "leave defaults at 2L, memorialize why."
- `scripts/serve-cute.sh` header CAUTION block (commit 1 of this PR) memorializes the policy.
- D2.5 (2L same-container 5-run) provides the survival evidence the default needs to be trusted under sustained load.

## Open questions

1. **Reset miss mechanism.** If the leading hypothesis is right, which reset is missing? The captured `wo_output` memset (graph-replay ordering)? The pre-launch counter `zero_()`? The inside-kernel MLP-partial reset? D3's reset-call-count instrumentation should isolate.
2. **Trigger.** Sustained-load triggers entry, but is the threshold cumulative tokens, cumulative requests, cumulative resets, time? D2 + D3 evidence should narrow.
3. **Recovery mechanism.** Q14's sharp recovery — natural buffer overwrite, counter wrap, or did some specific reset path finally fire? Output-hash diff across the Q13→Q14 boundary will say whether identical inputs are getting identical outputs (which would mean the corruption was input-conditional, ruled in by the recovery).
4. **Same root cause as the 16L break (sweep)?** D2.2 (`CUTE_PHASE_E_PATH=lite`) provides indirect evidence: if lite-path is stable at sustained load AND lite-path didn't show the 16L quality drop, they share a substrate.

## Tracking

- Plan addendum lives here and in [`docs/superpowers/plans/2026-05-09-beta-coop-layer-sweep-wo8.md`](../../superpowers/plans/2026-05-09-beta-coop-layer-sweep-wo8.md) as a Stage 4 entry.
- Each D2 leg gets its own subdir: `soak_d2_1_wo1/`, `soak_d2_2_lite/`, `soak_d2_3_phaseE_off/`, `soak_d2_4_layer_subsets/`, `soak_d2_5_2L_survival/`.
- Memory updates after each leg lands: append to `project_beta_coop_full_compile_wall.md` or open a new entry per leg as warranted.
- Friend code review (2026-05-11) credit + key code citations preserved inline.
