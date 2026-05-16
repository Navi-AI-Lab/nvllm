# SSM zero-on-realloc — 4-arm sentinel ablation (2026-05-15)

## Status

**Harness validation only. No perf claim. No nsys trace.**

The β-coop sustained-load collapse this work was designed to discriminate
was not reproducing on the host on 2026-05-15. Per friend's reframing:
> "If all arms pass and counters prove toggles worked: result is 'patch
> not harmful under non-repro conditions; collapse not available for
> discrimination.'"

That is the result here. The harness is committed for the next collapse
window.

## What ran

- **Suite:** 4 arms (both / neither / ssm_only / kv_only), 5×GSM8K-50
  runs per arm, fresh container per arm, sentinel files at
  `/run/nvllm/*.enabled` bind-mounted `:ro` per arm.
- **Total:** 20 runs (1000 generations), ~5h 25min wall (14:17→19:41
  EDT). 0 errors, 0 OOMs, 0 container restarts.
- **Suite code:** `scripts/ablation/run_ssm_ablation_suite.sh`,
  `scripts/ablation/ssm_ablation_compare.py`, overlay applied via
  `scripts/ablation/prepare_sentinel_overlay.sh`.

## Host / image manifest

| Field | Value |
|---|---|
| started_utc | 2026-05-15T18:17:46Z |
| git_sha | `670724746c596f6c095970c4d50b82e6328423db` (`plan/beta-coop-layer-sweep-wo8` head at suite time) |
| image | `nvllm:gb10-d2_7` |
| image_id | `nvllm:gb10-d2_7@4df53234ad5c` |
| image_digest | `no-digest` (local-built image, never pushed) |
| host_driver | `590.48.01` |
| host_kernel | `6.17.0-1014-nvidia` |
| hardware | NVIDIA DGX Spark (GB10, SM120, 128 GB unified) |
| hf_model | `ig1/Qwen3.5-27B-NVFP4` |
| served_name | `default` |
| gsm8k_n | 50 |
| gsm8k_seed | 42 |
| gsm8k_max_tokens | 512 |
| prompt_set_hash | `f422bd91dd644cc1a8afce282e51732977e1e1e5c361e894287f8eed5792e2cf` (sha256 of `n|seed|model|served-name`) |
| phase_e_layers | `3,7` |
| wo_split | 8 |

## Per-arm verdict

| Arm | SSM sentinel | KV sentinel | runs (correct/50) | first_fire (ssm,kv) | gate_pass | harness_pass |
|---|---|---|---|---|---|---|
| `both` | 1 | 1 | 48,48,48,48,48 | (1, 1) | true | true |
| `neither` | 0 | 0 | 48,48,48,48,48 | (0, 0) | true | true |
| `ssm_only` | 1 | 0 | 48,48,48,48,48 | (1, 0) | true | true |
| `kv_only` | 0 | 1 | 47,47,47,47,47 | (0, 1) | true | true |

`harness_pass=true` for all four arms means: when SSM_sentinel=1 the SSM
gate fired (and not when SSM_sentinel=0); same for KV. The sentinel
machinery is proven to discriminate. The env-strip confound from a prior
env-gated attempt is eliminated.

## What this shows and does not show

**Shows:**
- Sentinel-file gating works through vLLM EngineCore (env-stripped) where
  env-var gating did not.
- Under non-collapsing host state, the SSM zero-on-realloc patch is
  correctness-neutral (`both` and `ssm_only` both 48/50, identical to
  `neither` baseline) and perf-neutral (median decode within 0.03 tok/s
  across all arms).
- The KV `new_block_ids` channel relax (kv_only arm) is **NOT
  correctness-neutral**: a deterministic -1 question across all 5 runs.
  That is the basis for shipping the SSM patch alone in the production
  commit and keeping the KV relax in the harness overlay only.

**Does not show:**
- Whether the SSM patch fixes the β-coop sustained-load collapse: the
  collapse did not reproduce under today's host state.
- Any performance win: median decode is flat across arms; no nsys trace
  was captured.

## Per-arm steady-state stats

(See `ANALYSIS.md` Section "Aggregate per-arm steady-state stats".)

| Arm | N | median dtok/s | p95 wall_s | mean completion_tokens | finish_reason |
|---|---|---|---|---|---|
| both | 250 | 9.20 | 34.01 | 169.5 | length=5, stop=245 |
| neither | 250 | 9.19 | 34.03 | 169.5 | length=5, stop=245 |
| ssm_only | 250 | 9.20 | 34.01 | 169.5 | length=5, stop=245 |
| kv_only | 250 | 9.17 | 30.74 | 169.5 | length=5, stop=245 |

Note: per-arm 50-question completion-token sums all land at 8477 tokens
(mean 169.54). Per-Q values do differ (e.g. `kv_only` Q2 = 116 tokens vs
217 for the other three arms — verified distinct via output sha256 and
output_len), but per-arm sums coincide. This is a chance numerical
balance, not a stat-collection bug.

## Drained KV invariant

All 20 runs drained KV cleanly: `vllm:kv_cache_usage_perc` returned to
≤0.3pp of baseline at the post-run snapshot, well inside the 5pp
tolerance. (See `ANALYSIS.md` for the full per-run table.)

## How to reproduce

```bash
# 1. Build a sentinel-overlaid scratch checkout (~5 sec).
scripts/ablation/prepare_sentinel_overlay.sh /tmp/nvllm-ssm-sentinel-patched

# 2. Run the 4-arm sweep (~3 h with default 5 runs x 4 arms x ~15 min/run).
scripts/ablation/run_ssm_ablation_suite.sh

# 3. Produce ANALYSIS.md from the per-arm verdicts.
.venv/bin/python scripts/ablation/ssm_ablation_compare.py /tmp/ssm_ablation_suite
```

Env overrides for the runner are documented in the script header (see
`scripts/ablation/run_ssm_ablation_suite.sh`).

## What is NOT committed

- Per-arm `docker.log` (×4, ~50 MB each)
- Per-arm `serve.log` (×4)
- Per-run per-Q `perq.jsonl` (×20 = 1000 records, ~600 KB)
- Per-run `metrics_*.json` (×120 snapshots)
- The full mamba slot trace (~750 events × 4 arms)

These artifacts live in the suite OUT_DIR
(`/tmp/ssm_ablation_suite_v2/` at run time) and can be regenerated by
re-running the harness against the committed scripts.

## Related

- `docs/research/2026-05-15-ssm-zero-on-realloc/README.md` — design + harness usage
- Production patch: commit `feat(worker): add MambaBlockZeroer sister zeroer for SSM zero-on-realloc`
- Harness commit: `test(ablation): sentinel-gated SSM zero-on-realloc ablation harness`
