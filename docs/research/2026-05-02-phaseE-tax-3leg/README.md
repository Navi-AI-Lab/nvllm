# Phase-E-tax screening — 3 legs × 2 phases

**Purpose.** Decide whether β-coop (Phase E fusion) is currently a tax on
lower8 FULL-graph decode, in light of the per-layer breakdown of the
existing FULL trace (β-coop ≈ 40.8 ms/layer-token vs legacy DecodeKernel
≈ 17.1 ms/layer-token + external GEMM).

**Design.** Method D from the planning thread: profile in
FULL+PIECEWISE without the blessed-cache mount (cold capture this
session), GSM8K in PIECEWISE for deterministic correctness. Three legs,
two boots each.

This is a **screening experiment, not production proof**. If
PhaseE-off wins materially, the next step is a fresh bless of the
PhaseE-off config and a 2-leg confirmation under blessed FULL. We do
not flip defaults from this run alone.

## Decision matrix

| Outcome on phaseE-off                                      | Action                                                                                            |
| :--------------------------------------------------------- | :------------------------------------------------------------------------------------------------ |
| Per-call kernel μs ≥ 5% better AND GSM8K floor passes      | Proceed to A: bless phaseE-off config, run 2-leg confirmation under FULL+blessed.                 |
| Per-call kernel μs neutral (±5%) AND GSM8K floor passes    | β-coop is not a tax; ship as-is. Move on to NVFP4 GEMV K-parallel reduction work in lower8 layers. |
| Per-call kernel μs better but GSM8K floor fails            | PhaseE-off has a numerics issue independent of perf. Halt; bisect numerics before further work.   |
| Per-call kernel μs worse                                   | β-coop is a win in lower8. Ship as-is, focus on NVFP4 GEMV reduction.                             |

GSM8K gates per leg:
- **Floor (diagnostic):** `correct ≥ 30 / 50` — no regression vs prior phase ~30/50.
- **Ship gate:** `correct ≥ 47 / 50` — matches blessed production. Only required for the *recommendation* downstream of this experiment, not for the diagnostic itself.

## Legs

| Leg          | `CUTE_PHASE_E_FUSION` | `CUTE_PHASE_E_LAYERS`               | β-coop layers (Qwen3.5 full-attn) | Warmup | Timed | Purpose                                                          |
| :----------- | :-------------------- | :---------------------------------- | :-------------------------------- | -----: | ----: | :--------------------------------------------------------------- |
| `lower8`     | `1`                   | `0..7`                              | 3, 7 (2 layers)                   |     15 |    10 | Current blessed production shape (re-measured cold)              |
| `phaseE-off` | `0`                   | `0..7` (irrelevant when fusion=0)   | none — legacy DecodeKernel + GEMM |      4 |    10 | **Decision-maker.** Is β-coop a per-layer tax?                   |
| `all-beta`   | `1`                   | `0..15`                             | 3, 7, 11, 15 (4 layers)           |     20 |     4 | Confirmatory — historical Phase 6a all-β was much slower per token |

All legs share:
- `--max-num-seqs 1`, `--max-model-len 16384`, `--max-num-batched-tokens 65536`
- `--kv-cache-dtype fp8_e4m3`, `--attention-backend CUTE_PAGED`
- `--gpu-memory-utilization 0.65`
- Profile phase: `cudagraph_mode=FULL_AND_PIECEWISE`, `--profiler-config` torch, no blessed-cache mount, `--privileged` for CUPTI.
- GSM8K phase: `cudagraph_mode=PIECEWISE`, no profiler.

## Output layout

```
benchmarks/nvllm/traces/cute_paged_attn/2026-05-02-phaseE-tax-3leg/
  lower8/
    profile_serve.log          # docker logs from EngineCore
    profile.pt.trace.json.gz   # raw torch profiler trace (gitignored)
    profile_kernels.csv        # per-kernel μs stats
    profile_metadata.json      # image/git/env/config_hash for the profile boot
    profile_DONE               # marker (skip-on-resume)
    gsm8k_serve.log
    gsm8k.json                 # raw correct/wrong/errors + gate verdicts
    gsm8k_metadata.json
    gsm8k_DONE
  phaseE-off/   (same layout)
  all-beta/     (same layout)
  mem_watchdog.log             # host free + docker stats every 30s
  summary.md                   # written AFTER run, by post-processing step
```

## Reproduction

```bash
# tmux is required — total wall time ~75-90 min, far past any subagent timeout.
tmux new -s bench-3leg

# Run from repo root.
bash docs/research/2026-05-02-phaseE-tax-3leg/run_3leg.sh

# Detach: Ctrl-b d
# Re-attach: tmux attach -t bench-3leg
# Resume after a partial run: re-running skips legs/phases that have a *_DONE marker.
# Force re-run of a leg: rm benchmarks/nvllm/traces/cute_paged_attn/2026-05-02-phaseE-tax-3leg/lower8/profile_DONE
```

Skip individual legs:

```bash
SKIP_ALL_BETA=1   bash docs/research/2026-05-02-phaseE-tax-3leg/run_3leg.sh   # only lower8 + phaseE-off
SKIP_LOWER8=1     bash ...                                                    # only phaseE-off + all-beta
SKIP_PHASEE_OFF=1 bash ...                                                    # only lower8 + all-beta
```

## Post-run analysis

Before drawing conclusions from the kernel-μs CSVs, run a kernel-inventory
sanity check:

1. **Stable custom kernels (CuTe / cute_paged / fused MLP / GEMMs):** compare per-call mean μs and total ms across legs normally.
2. **Inductor- or Triton-generated one-offs:** report separately in `summary.md` but do NOT let them drive the verdict.
3. **If symbol inventory or call counts differ unexpectedly between legs** (different set of kernel symbols, or 5×+ call-count drift on a custom kernel), method D is **contaminated** by cold FULL-graph variance. Fall back to plan A: bless the phaseE-off config first, then re-measure.

`trace_workload.py` sends `ignore_eos: true` and `temperature: 0`,
so the timed burst per leg fires the same number of forward passes
regardless of model output coherence. The kernel-μs comparison therefore
remains valid even if the cold-captured graph produces gibberish in this
session (which would only matter for the GSM8K phase, which uses
PIECEWISE).

## Why this layout (vs alternatives)

Considered and rejected:

- **A — bless phaseE-off + all-beta first.** Cleanest data, but ~60-90 min of bless work before the screening experiment can start, and the manifests get thrown away if the screening says β-coop is fine.
- **C — all 3 legs PIECEWISE only.** Drops the FULL-graph context the original analysis was built on; loses comparability with the existing lower8 FULL trace.
- **single-boot-per-leg (profile + gsm8k on same server).** Saves a ~5-min model load per leg, but the profiler-active server runs slow (CuTe under profiler is ~16× slower than unprofiled), so the GSM8K wall-clock would be poor and the profiler buffer might still be flushing during quality checks. Cleanest separation: two boots per leg.

## Caveats baked into the run

- **Cold FULL graph variance.** Z1 inductor non-determinism (memory: `project_full_graph_blocked.md`) means each cold capture can produce a different set of inductor pointwise/fused kernels. CuTe custom kernels and CUTLASS GEMMs are stable across captures and are the comparison anchor.
- **β-coop disk cache reuse.** All three profile boots share `/tmp/nvllm-cute-cache` so cute.compile cost is amortized across runs. Cold cute.compile is ~24s (memory: `project_beta_coop_full_compile_wall.md`); cache-warm is sub-second.
- **active_iterations is defensive only.** Per memory `feedback_active_iterations_dead_code`, that field doesn't fire outside a profiler schedule with `wait>0` or `warmup>0`. Real bound is the explicit `/start_profile` + `/stop_profile` pair plus the 120s host-side flush.
- **PIECEWISE GSM8K is for kernel-pathway correctness, not graph-mode correctness.** It tells us "does the phaseE-off code path produce sane outputs?" It does *not* tell us "does FULL+phaseE-off produce sane outputs?" That's a separate question, only answered if we proceed to plan A.

## Wall time budget

| Phase                                | Per leg          | Total (3 legs)    |
| :----------------------------------- | :--------------- | :---------------- |
| Server boot (model load + CUDA graphs) | ~5 min           | ~30 min (6 boots) |
| Warmup (15-20 requests at max_tokens=64) | ~3 min         | ~18 min            |
| Profile burst (concurrent=1, max_tokens=256) | 5-10 min   | ~25 min            |
| CUPTI flush                          | 2 min            | 6 min              |
| GSM8K-50 (max_tokens=512)            | ~10 min          | 30 min             |
| **Total**                            |                  | **~110 min**       |

If the run goes long: the watchdog log shows where memory or container
state went sideways. `docker logs nvllm` from a partial run goes to
`<leg>/<phase>_serve.log`.

## Related work

- Existing FULL lower8 trace: `benchmarks/nvllm/traces/cute_paged_attn/2026-04-30-coop-wo-reset/` (the analysis source for the per-layer breakdown).
- Phase E template (3-leg torch profiler pattern): `benchmarks/nvllm/traces/phase_e/2026-04-23-initial/` and `docs/research/phase_e_traces/capture_all.sh`.
- Bless infrastructure (used in plan A if we proceed): `scripts/bless-cute-full-cache.sh` + `docs/superpowers/specs/2026-05-01-cute-full-cache-production-workaround-design.md`.
