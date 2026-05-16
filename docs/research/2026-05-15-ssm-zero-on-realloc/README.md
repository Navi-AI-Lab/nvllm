# SSM zero-on-realloc — design + sentinel ablation harness

## What

The production patch (commit `feat(worker): add MambaBlockZeroer sister
zeroer for SSM zero-on-realloc`) adds an SSM zero-on-realloc guard alongside
the existing full-attention KV zero-on-realloc path.

`KVBlockZeroer.zero_block_ids` now also walks a sister `MambaBlockZeroer` on
the same block-ID list, zeroing recycled `conv_state` / `ssm_state` rows via
`torch.index_fill_` before the next prefill writes into them.

## Why

The existing `KVBlockZeroer` (upstream PR #35219) clears full-attn KV blocks
at request-free / block-realloc time but skips Mamba layers because the conv
/ ssm page sizes differ from the full-attn page size and cannot share the
Triton kernel's uniform `PAGE_SIZE_EL`. `MambaBlockZeroer` covers the
remaining state.

This addresses one half of the suspect set from the Mamba SSM cache
lifecycle audit (memory:`project_mamba_ssm_lifecycle`):
> "what's accumulating in-process between runs that isn't in any cherry-pick"

Hybrid-attention models (Qwen3.5-27B and similar) hold per-block mamba
state in tensors whose leading dim is `num_blocks`. When a block ID is
recycled to a new request, the old request's mamba state in that slot
would otherwise persist as initial state for the new prefill.

## What this commit series does NOT claim

- **No "fixes collapse" claim.** The β-coop sustained-load collapse was not
  reproducing on the host at the time of this work (2026-05-15). The patch
  is shipped because the lifecycle gap is real; the patch's effect under
  the failing host state is unknown.
- **No perf claim.** No nsys trace was captured. The 4-arm sentinel
  ablation (below) shows median decode_tok_s within 0.03 tok/s across all
  arms (perf-neutral under non-collapse load), but that is not a perf win.

## The sentinel ablation harness

The harness in `scripts/ablation/` lets a future operator A/B the patch
under a future collapse window without having to rebuild the image.

It applies a sentinel overlay (`scripts/ablation/ssm_sentinel_overlay.patch`)
to a scratch checkout of the repo, replacing the production
always-on firing path with a filesystem-sentinel gated version. Per-arm,
the runner bind-mounts a per-arm sentinel directory at `/run/nvllm` :ro;
the gate at module-import-time stats the sentinel file and caches the
result.

### Why sentinel files, not env vars

vLLM EngineCore spawns the worker subprocess with most env vars stripped
(memory:`feedback_vllm_enginecore_env_strip`); only `VLLM_TARGET_DEVICE`
and `VLLM_WORKER_MULTIPROC_METHOD` survive. A previous env-gated
ablation (`v1`) was a null A/B because the gate always read empty-string.
Sentinel files survive subprocess spawn because the file system is the
shared substrate.

### Sentinel paths

| Path | Effect when present |
|---|---|
| `/run/nvllm/zero_ssm_on_realloc.enabled` | SSM zero-on-realloc fires |
| `/run/nvllm/kv_zero_for_mamba_ids.enabled` | KV `new_block_ids` channel relaxed for MambaSpec allocations |

The KV channel relax is included in the overlay for completeness but is
NOT shipped in the production patch — the 2026-05-15 4-arm sweep showed it
introduces a deterministic -1 question on `kv_only` (47/50 × 5 vs 48/50 ×
5 on `both`, `neither`, `ssm_only`).

### Per-arm signature

When the harness runs each arm, the docker log triad proves which gates
fired:

| Event | Meaning |
|---|---|
| `nvllm.ablation.sentinel_check name=<n> path=<p> exists=<b> enabled=<b>` | One per gate per worker process, at first call |
| `nvllm.ablation.first_fire name=<n> n_block_ids=<N>` | One per gate the first time the patched branch fires |
| `nvllm.ablation.fire_count name=<n> count=<N>` | Every 100th fire |

`verdict.json` per arm includes:
- `harness_validation.pass` — false if SSM_sentinel=1 but first_fire=0
  (or vice versa), per gate
- `harness_pass` — top-level boolean mirror

### Reproducing the 2026-05-15 sweep

```bash
# Build a sentinel-overlaid scratch checkout (~5 sec).
scripts/ablation/prepare_sentinel_overlay.sh /tmp/nvllm-ssm-sentinel-patched

# Run the 4-arm sweep (~3 h with default 5 runs × 4 arms × ~15 min/run).
scripts/ablation/run_ssm_ablation_suite.sh

# Produce ANALYSIS.md from the per-arm verdicts.
.venv/bin/python scripts/ablation/ssm_ablation_compare.py /tmp/ssm_ablation_suite
```

Env overrides for the runner are documented in the script header.

## Evidence

The 2026-05-15 evidence dir lives at:

```
benchmarks/nvllm/traces/ssm_zero_on_realloc/2026-05-15-sentinel-ablation/
```

See its `summary.md` for the per-arm verdict table, host/image manifest,
and what the run did and did not prove.

## Related memory

- `project_beta_coop_sustained_collapse` — the closed bisection arc
- `project_mamba_ssm_lifecycle` — the lifecycle audit that scoped this fix
- `feedback_substrate_not_cherry_pick` — methodology lesson from D2.x
- `feedback_vllm_enginecore_env_strip` — why env vars aren't reliable
- `feedback_default_vs_base_path_coverage` — why we keep the harness
