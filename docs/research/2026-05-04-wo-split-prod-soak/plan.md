# wo_split production soak — plan

**Branch:** `evidence/wo-split-prod-soak`. Force-reset from `feat/wo-split-8-prototype@69c530082` to drop two superseded single-arm evidence/docs commits (`dbd755da1`, `2a812a8da`); the 11 kernel-feature + sentinel + cleanup commits (`751d2f2f8`..`69c530082`) are kept. The PR ships the production W_O K-parallel feature (opt-in via `CUTE_WO_SPLIT`) **plus** the soak evidence proving whether it's safe to default. Default flip remains a conditional follow-up on `feat/wo-split-N-prototype`.
**Date:** 2026-05-04
**Predecessor:** PR #7 (W_O K-parallel harness, Navi-AI-Lab/nvllm) merged 2026-05-04
**Successor (conditional):** `feat/wo-split-N-prototype` where N is the verdict-selected `wo_split` value (typically 4 or 8) — opens only if soak verdict says default candidate

## Goal

Produce committed evidence to decide whether `wo_split` should remain opt-in, become the local default at the verdict-selected value, or be re-investigated. Evidence comes from a real serving workload, not the kernel-tight harness.

## Why now

PR #7 proved correctness (bit-exact at wo_split ∈ {1, 2, 4, 8}, the prod-gated set per commit `13b2337d9`) and a kernel-isolated 8.39× speedup at wo_split=8 (32 W_O CTAs) on the harness microbench. Production behavior is different: at the prod grid (`slice_ctas=8`, num_kv_heads, num_seqs), R11 (pre-W_O wait) and R12 (last-CTA gather) compete for SM occupancy with FC1, FC2, paged attention, and gather. End-to-end wall delta and latency-tail behavior in serving are unknown.

## Scope

**In:**
- 4-arm sweep: `wo_split ∈ {1, 2, 4, 8}`
- N=5 sequential workload replays per arm (one container per arm; cold-restart only between arms / between measurement passes within an arm)
- Four workload phases per arm: GSM8K-50 quality anchor, ShareGPT mixed-length serve, single 2048-token long-decode probe, lightweight 2-concurrent probe
- Two measurement modes per arm: primary `CUTE_BETA_REGION_TIMING=0` for clean wall/TPOT; supplementary `CUTE_BETA_REGION_TIMING=1` for one-shot region breakdown
- Metrics: wall time, token-level TPOT p50/p95/p99, region timing (supplementary pass only), torch-profiler kernel mix via `VLLM_TORCH_PROFILER_DIR` (one representative trace per arm per phase B/C)
- Coherence checks on long-decode output (warning-only, not a gate)
- Verdict doc

**Out (deliberately):**
- Default flip — separate PR if verdict justifies
- `wo_split ∈ {16, 32}` — couples to grid geometry / Phase 3 design, needs its own design pass
- New marker instrumentation — region table uses existing `CUTE_BETA_REGION_TIMING` only
- Multi-user load benchmark beyond the 2-concurrent probe

## Decision criteria

### Quality gate

Per-arm GSM8K-50 score (Phase A, /no_think):
- **Hard floor:** ≥ 30/50 (β-coop baseline ~30-31/50 per `feedback_post_quant_sanity`; below this is catastrophic regression).
- **Pairwise regression:** arm score ≥ baseline (`wo_split=1`) score − 2 (no more than a 2-question drop vs same-environment baseline). Prior /no_think GSM8K-50 results have run 47-48/50, so a 2-question drop is a meaningful signal — the hard floor alone would only catch catastrophic failure.

Either gate failing blocks "default candidate" and routes to "investigate."

### Wall + TPOT bar

**Default candidate** requires all three:
- wall-time mean ≥ 5% improvement vs `wo_split=1` baseline (timing OFF runs only)
- no GSM8K regression per gate above
- TPOT p95 not worse than baseline (within 1× baseline stddev). Token-level p95 within a long-decode run is the authoritative tail metric; run-level p95 from N=5 is "max-with-a-label" (reported as supplementary).

**Keep opt-in:** wall improvement but TPOT p95 worse (wins on aggregate, loses on tail) OR wall improvement marginal (<5%) but no regression.

**Investigate:** GSM8K regression OR wall regression vs baseline.

The 5% wall threshold is a starting point. After the baseline arm runs, revise to `2 × baseline_run_to_run_stddev` if baseline noise exceeds 5%.

### Successor branch selection

Verdict selects the winning `wo_split` value (4 or 8). Successor branch `feat/wo-split-N-prototype` defaults `CUTE_WO_SPLIT=N`. If both 4 and 8 satisfy the criteria, prefer the one with better TPOT p95 (tail behavior wins ties).

## Workload phases per arm

### Phase A — GSM8K-50 quality anchor (1× per arm)

- Runner: `scripts/gsm8k_eval_50.py` (existing canonical, seed=42)
- **Mode: `/no_think`** (per `feedback_eval_think_modes` — canonical for GSM8K). Prior full-think runs were ~60 min; /no_think is ~3 min.
- Sample size: 50; runs **once per arm** (regression gate, not a perf metric — variance not needed)
- Decode params: temperature=0, top_p=1, seed=42 (deterministic)
- Output: `gsm8k.json`, score
- Measurement mode: **`CUTE_BETA_REGION_TIMING=0`** (timing OFF) so the gate measures the prod kernel, not the instrumented one. If a score looks anomalous, fall back to a `CUTE_BETA_REGION_TIMING=1` rerun to disambiguate (instrumentation perturbing numerics).

### Phase B — ShareGPT mixed-length serve (5× primary timing OFF; +1× supplementary timing ON)

- Source: `anon8231489123/ShareGPT_Vicuna_unfiltered`
- **Pinned revision:** `gen_sharegpt_slice.py` records `huggingface_hub` revision SHA + license + filter rules in the slice header before `sharegpt_slice.jsonl` is committed
- Selection: deterministic first ~30 multi-turn conversations, seed=42, filtered to mixed prompt lengths (short ~50, medium ~500, long ~1500-token spans)
- Slice generated **once** by `gen_sharegpt_slice.py` into `sharegpt_slice.jsonl`, committed to `docs/research/2026-05-04-wo-split-prod-soak/`
- Replay: `/v1/completions` per turn, sequential within conversation, non-overlapping requests (per `feedback_eval_completions` — chat triggers thinking mode, breaks extraction)
- **Decode params:** temperature=0, top_p=1, seed=42 across all arms (output diffs reflect kernel behavior, not sampling noise)
- Captured per replay: per-turn TPOT (token-level), per-conv wall, full output

### Phase C — Long-decode probe (5× primary timing OFF; +1× supplementary timing ON)

- Single hand-picked prompt → `ignore_eos=true`, `max_tokens=2048`
- Decode params: temperature=0, top_p=1, seed=42
- Captured: full output tokens, per-token timestamps, TPOT trace
- Coherence checks (warning-only; not a gate):
  - repeated 4-gram rate
  - unique trigram ratio
  - replacement / `U+FFFD` char count
  - same-prefix-across-splits diff (pairwise vs `wo_split=1`)
- Eyeball + full output committed for spot-check; no programmatic gate (per friend feedback: "no perplexity")

### Phase D — Lightweight 2-concurrent probe (1× per arm, timing OFF)

- Two concurrent `/v1/completions` requests (representative steady-state per `project_num_seqs_2_target` — Hermes + interactive use is num_seqs=2)
- Both requests use distinct medium-length prompts from the ShareGPT slice
- Identical decode params + seeds across requests
- Captured: per-request TPOT, total wall, output
- Purpose: tripwire for whether wo_split parallelism degrades under SM contention from a second active sequence; **not** a full multi-user load test

## Measurement modes per arm

Two-pass measurement to avoid the debug-instrumented kernel polluting default-decision data:

| Pass | `CUTE_BETA_REGION_TIMING` | Phases run | Purpose |
|---|---|---|---|
| Primary | `0` (OFF) | A (1×) + B (5×) + C (5×) + D (1×) | Wall, TPOT, GSM8K — feeds the "default candidate" decision |
| Supplementary | `1` (ON) | B (1×, torch profiler ON) + C (1×, torch profiler ON) | Region breakdown across R2/R11/R12; profiler kernel mix |

Container is restarted between primary and supplementary passes within an arm because the timing constexpr is part of the compile-cache key (per `apply_disk_cache_patch`).

## Per-arm container lifecycle

1. Stop existing nvllm container
2. Start container with `CUTE_WO_SPLIT={1,2,4,8}` and `CUTE_BETA_REGION_TIMING=0`
3. Standard prod flags: `--kernel-config '{"enable_flashinfer_autotune":false}'` (per `feedback_flashinfer_autotune_sm120` — without this, host can hard-reboot)
4. Wait for `/v1/models` ready via active probe, not docker-up time (per `feedback_active_serve_readiness_probe`)
5. **Primary pass:** A (1×) → B (5×) → C (5×) → D (1×). All wall/TPOT measurements come from this pass.
6. Stop container; restart with `CUTE_BETA_REGION_TIMING=1` (same `CUTE_WO_SPLIT`)
7. **Supplementary pass:** B (1×, torch profiler ON) → C (1×, torch profiler ON). Region timing CSV + profiler traces come from this pass only.
8. Stop container before next arm.

Cold compile cost: each (`wo_split`, `region_timing_enabled`) constexpr combination has a unique compile-cache key. Two compiles per arm × 4 arms = ~8 × 24 s = ~192 s total. Negligible vs total soak wall.

## Supplementary profiler strategy

Explicit to avoid 40-trace bloat and to avoid relying on `nsys` for vLLM V1
serving. Per `feedback_vllm_profiling`, V1 EngineCore is a spawned subprocess
and CUPTI injection is not inherited reliably, so serving kernel-mix evidence
uses the vLLM torch profiler API with `VLLM_TORCH_PROFILER_DIR`.

- Phase A: no profiler (quality gate, kernel mix uninteresting)
- Phase B (5× primary): **no profiler** — instrumentation perturbs the perf measurement
- Phase B (1× supplementary, timing ON): **torch profiler ON** — single representative trace
- Phase C (5× primary): no profiler
- Phase C (1× supplementary, timing ON): **torch profiler ON** — single representative trace
- Phase D: no profiler (small probe, kernel mix coverage from B suffices)

Total: 4 arms × 2 phases (B + C) = **8 torch-profiler traces**. The nsys
evidence for the W_O kernel path remains the committed PR #7 harness /
production-grid microkernel trace set.

## Evidence layout

```
docs/research/2026-05-04-wo-split-prod-soak/    # pre-run scripts (this dir)
  plan.md                       # this doc
  README.md                     # how to reproduce
  gen_sharegpt_slice.py         # one-shot slice generator + revision pin
  sharegpt_slice.jsonl          # committed deterministic trace (header records revision SHA + license)
  longdecode_prompt.txt         # the 2048-token-target prompt
  runner.sh                     # multi-arm bash driver
  coherence_check.py            # warning heuristics for long decode
  parse_results.py              # raw → summary table

benchmarks/nvllm/traces/wo_split_prod_soak/2026-05-04-soak/   # post-run artifacts
  wo1/  wo2/  wo4/  wo8/        # per arm
    primary/                    # CUTE_BETA_REGION_TIMING=0 pass
      gsm8k.json                # 1× Phase A
      run01/  run02/  run03/  run04/  run05/
        sharegpt_outputs.jsonl  # full output (Phase B)
        sharegpt_wall_tpot.csv  # per-turn wall + per-token TPOT
        longdecode_output.txt   # Phase C output
        longdecode_tpot.csv
        longdecode_coherence.json
      concurrent/               # 1× Phase D
        request_a_output.txt
        request_b_output.txt
        wall_tpot.csv
    supplementary/              # CUTE_BETA_REGION_TIMING=1 pass
      sharegpt_region_timing.csv  # 13-region breakdown (R2/R11/R12 highlighted)
      serve_trace/                # Phase B torch-profiler trace
      longdecode_region_timing.csv
      longdecode_trace/           # Phase C torch-profiler trace
  summary.md                    # verdict, tables, repro commands
```

Per `feedback_benchmarks_evidence_only`: pre-run scripts in `docs/research/`, post-run artifacts in `benchmarks/nvllm/traces/`. Per `feedback_evidence_force_add`: `git add -f` overrides `.gitignore` for evidence dir contents.

## Estimated wall

Per arm:
- Primary pass (`CUTE_BETA_REGION_TIMING=0`):
  - Phase A GSM8K /no_think: ~3 min × 1 = 3 min
  - Phase B ShareGPT 30 convs: ~12 min × 5 = 60 min
  - Phase C long decode: ~3 min × 5 = 15 min
  - Phase D concurrent: ~5 min × 1 = 5 min
  - subtotal: ~83 min
- Supplementary pass (`CUTE_BETA_REGION_TIMING=1`):
  - Phase B ShareGPT: ~12 min × 1 = 12 min
  - Phase C long decode: ~3 min × 1 = 3 min
  - subtotal: ~15 min
- Container restart × 2 + cold compile × 2: ~3 min
- **Total per arm: ~101 min**

4 arms: **~6.7 h** wall. Wrap in tmux per `feedback_tmux_long_jobs`. Phase B is the dominant variable; if first arm overruns, revise estimate before running remaining arms.

## Risks / open items

- **ShareGPT slice quality** — first ~30 convs may include excessive tool-call markers, role tags, or non-English content. Visual inspection of `sharegpt_slice.jsonl` before commit; revision SHA + license + filter rules recorded in slice header.
- **5% wall threshold** — guess based on prior phaseE-tax bench noise. Concrete threshold lands after observing baseline arm run-to-run stddev.
- **TPOT p95 from N=5 run-level samples** is "max-with-a-label." Token-level p95 within a single long-decode run is the authoritative tail metric. Both reported in `summary.md`.
- **Region-table CTA-id remap** — `region_timing.py:75 _phase1_wo_split_cta_ids` already understands `wo_split`; reducer should produce comparable R2/R11/R12 rows across arms. Verify on first supplementary pass before relying on cross-arm comparison.
- **Concurrent probe is small-N** — 1× per arm, 2-concurrent only. Useful as a tripwire, not a full multi-user characterization.

## What this PR is NOT

- A default flip (separate PR, conditional on verdict)
- A `wo_split>8` design exploration
- New marker instrumentation in the kernel
- A multi-user load benchmark
- A quality-bench publication

If verdict says default candidate, `feat/wo-split-N-prototype` opens with a one-line config change (default `CUTE_WO_SPLIT=N` for the verdict-selected N) plus this PR's evidence cited.

## References

- PR #7 harness: `docs/research/2026-05-03-w-o-k-parallel-harness/`
- PR #7 prod gut-check (single-arm wo_split=8): `benchmarks/nvllm/traces/wo_k_parallel_audit/2026-05-03-wo-split-8-prod/`
- Region names: `vllm/v1/attention/backends/cute_paged/region_timing.py:35-49` (commit `69c530082`)
- Region timing kernel gate: `vllm/v1/attention/backends/cute_paged/phase_e_kernel.py:354` (commit `69c530082`)
- W_O K-parallel CTA-id helper: `vllm/v1/attention/backends/cute_paged/region_timing.py:75-91` (commit `69c530082`)
- AGENTS.md §4 evidence standard: nsys + commit hash + reproduce commands required for performance claims; serving soak records torch profiler traces because V1 EngineCore cannot be captured reliably by nsys
- ShareGPT dataset: `anon8231489123/ShareGPT_Vicuna_unfiltered` on HF; revision pinned in `gen_sharegpt_slice.py`
- Concurrent target: `project_num_seqs_2_target` (Hermes + interactive use is num_seqs=2 steady state)
- Eval thinking modes: `feedback_eval_think_modes` (/no_think for GSM8K)
- Strategy context: phaseE-tax bench (lower8 320 vs phaseE-off 656 ms/tok) per `project_strategy_priorities`
