# β-coop layer-count sweep under `CUTE_WO_SPLIT=8`

Re-litigates the 2026-05-02 phaseE-tax verdict (`feedback_phase4_dead`:
"Phase 4 stays dead; β kernel ~40.7 ms/layer ≫ 17 ms decode") now that
`CUTE_WO_SPLIT=8` cuts the W_O GEMV by 8.39× and brings β-coop kernel
total to ~5.5 ms/call (`benchmarks/nvllm/traces/wo_split_prod_soak/2026-05-04-soak/writeup.md`).
At β ≈ 5.5 ms each added β-capable full-attention layer should *save*
~17 − 5.5 = ~11.5 ms vs DecodeKernel + Phase_D_MLP, which would shift
the dev baseline from 2L (current shipped) toward 16L. This sweep does
not prove that perf claim — it proves correctness holds across the
ladder so the long-soak loop can settle the wall-time question.

Worktree: `nvllm-beta-layer-sweep-wo8` (work/beta-layer-sweep-wo8).
The runner refuses the primary checkout unless `ALLOW_PRIMARY_CHECKOUT=1`.

## Arms

Five β-on arms, all pinned to `CUTE_WO_SPLIT=8`. Layer ids are global
decoder-layer indices. Qwen3.5-27B has 64 layers with full-attention at
`3,7,11,15,19,23,27,31,35,39,43,47,51,55,59,63`; linear-attention layers
never attach β-coop and are not counted. The original 0L_off arm was
dropped from the manifest before the sweep ran (the Stage 0a/0b/0c gates
already establish the no-β baseline).

| arm       | β layers | `CUTE_PHASE_E_LAYERS`                                  |
|-----------|---------:|--------------------------------------------------------|
| 2L_3_7    |        2 | `3,7`                                                  |
| 4L_3_15   |        4 | `3,7,11,15`                                            |
| 8L_3_31   |        8 | `3,7,11,15,19,23,27,31`                                |
| 12L_3_47  |       12 | `3,7,11,15,19,23,27,31,35,39,43,47`                    |
| 16L_3_63  |       16 | `3,7,11,15,19,23,27,31,35,39,43,47,51,55,59,63`        |

See [`arms.csv`](arms.csv) for the machine-readable manifest.

## Gates (per arm)

1. **Dispatch audit.** [`extract_dispatch_log.py`](extract_dispatch_log.py)
   parses `[PHASE_E_DISPATCH] …` lines from `docker logs nvllm` and asserts
   that the observed β-coop layer set is exactly `expected_coop_layers`.
   `CUTE_PHASE_E_FALLBACK_RAISE=1` is set on every arm so any silent
   fallback fails the server, not the audit. Audit fail → arm ends with
   `verdict.ok=false`, runner moves on.
2. **GSM8K-50** (`scripts/gsm8k_eval_50.py`, seed=42, `/v1/completions`,
   per `feedback_eval_completions` + `feedback_post_quant_sanity`).
   Pass = ≥ 45/50 correct AND 0 errors.
3. **β kernel per-call median ≤ 7 ms** (advisory). The sweep itself
   exports `CUTE_BETA_REGION_TIMING=0` (per plan Risk #2) so no per-arm
   region-timing dump is taken — the kernel never allocates the timing
   buffer during the sweep. The single per-call β median used to gate
   was captured separately during Stage 0c (5.538 ms vs ≤7 ms gate);
   see `stage0a/verdict.md`. The dump-trigger code path
   (`scripts/trigger_region_timing_dump.sh` + the
   `/tmp/.dump_region_timings` sentinel implemented at
   `vllm/v1/attention/backends/cute_paged/_backend.py:2058`) remains
   wired in `runner.sh` but is hard-disabled (`if false && …`).

## How to run

The sweep is meant to live in a tmux session for the ~75–90 min wall
(per `feedback_tmux_long_jobs`).

```bash
cd /home/natfii/docker/nvllm-beta-layer-sweep-wo8
tmux new -d -s beta_layer_sweep \
  './docs/research/2026-05-09-beta-coop-layer-sweep-wo8/runner.sh \
     docs/research/2026-05-09-beta-coop-layer-sweep-wo8/arms.csv \
     2>&1 | tee docs/research/2026-05-09-beta-coop-layer-sweep-wo8/runner.log'
tmux capture-pane -p -t beta_layer_sweep | tail -40
```

Add `--force` as a second arg to overwrite an existing `sweep/<arm>/`
directory; without it, the runner refuses to start an arm whose
output dir is non-empty.

## Evidence layout

```
docs/research/2026-05-09-beta-coop-layer-sweep-wo8/
  arms.csv
  README.md
  runner.sh
  extract_dispatch_log.py
  stage0a/                          # Stage 0a plumbing-proof artifacts
  sweep/
    summary.md                      # aggregated per-arm table
    <arm>/
      summary.md                    # arm provenance + gate results
      verdict.json                  # machine-readable arm verdict
      dispatch_audit.json           # output of extract_dispatch_log.py
      gsm8k.json, gsm8k.log         # GSM8K-50 results
      c2_diag_ENV.txt               # /tmp/c2_diag/ENV at server boot
      docker_inspect.json           # container Cmd + resolved Env
      serve_log_head.txt            # first 200 lines of server log
      serve.log, docker.log         # full bring-up + post-mortem
```

## Committing evidence

Per `feedback_evidence_force_add`: `*.csv`, `*.json`, `*.log`, and
`*.npy` paths are gitignored at the repo root. **Every evidence path
in this directory needs `-f`** because the gitignore extension match
fires regardless of subdirectory:

```bash
# Tracked-as-source files (.sh / .py / .md only — `git add` works):
git add docs/research/2026-05-09-beta-coop-layer-sweep-wo8/{README.md,runner.sh,soak_runner.sh,extract_dispatch_log.py,build_summary.py}

# Evidence files (gitignored extensions — must use -f):
git add -f docs/research/2026-05-09-beta-coop-layer-sweep-wo8/arms.csv
git add -f docs/research/2026-05-09-beta-coop-layer-sweep-wo8/stage0a/
git add -f docs/research/2026-05-09-beta-coop-layer-sweep-wo8/sweep/
git add -f docs/research/2026-05-09-beta-coop-layer-sweep-wo8/soak/   # post-Stage-2b
git add -f docs/research/2026-05-09-beta-coop-layer-sweep-wo8/runner.log  # if present
```

Pre-run scripts (`runner.sh`, `soak_runner.sh`, `extract_dispatch_log.py`,
`build_summary.py`, `arms.csv`) live here, not under `benchmarks/`
(per `feedback_benchmarks_evidence_only`). The post-sweep canonical
traces dir (`benchmarks/nvllm/traces/cute_paged_attn/2026-05-09-layer-count-sweep-wo8/`)
is created by Stage 2/3 of the plan, not Task 1a.
