# Blessed CuTe FULL+β-coop torch.compile caches

This directory holds the per-configuration **manifests** for blessed
torch.compile inductor AOT caches used by `scripts/serve-cute-full.sh`
(FULL_AND_PIECEWISE + β-coop production).

The manifests are committed to git; the **blessed cache binaries themselves
live host-only** at `~/.cache/nvllm/blessed/<config_hash>/` (override via
`NVLLM_BLESSED_CACHE_ROOT`). Manifests gate which inductor artifact is
trusted at production serve time.

## Why blessed caches exist

Z1 (2026-04-30) proved that `torch.compile`'s inductor produces *non-deterministic*
AOT artifacts for the same input graph (~60% PASS rate per cold compile under
FULL+β-coop replay-coherence). Pinning a known-good artifact gives 5/5 PASS;
locking a known-bad gives 1/5. Until upstream non-determinism is reduced,
production must mount a blessed AOT artifact `:ro`.

Full evidence: [`docs/research/2026-04-29-full-graph-spike/evidence/2026-04-30-2109-pathb-z1-summary/summary.md`](../research/2026-04-29-full-graph-spike/evidence/2026-04-30-2109-pathb-z1-summary/summary.md).

## Active manifests

<!-- BEGIN AUTO-GENERATED TABLE -->

| Filename | Model | cgmode | Layer set | Image ID | Blessed at | Status |
|---|---|---|---|---|---|---|
| `qwen35-27b-nvfp4_fap_lower8_image-a3f3f60_e6d32b4.json` | ig1/Qwen3.5-27B-NVFP4 | FULL_AND_PIECEWISE | 0,1,2,3,4,5,6,7 | sha256:a3f3f60… | 2026-05-01T18:22:45Z | active |

<!-- END AUTO-GENERATED TABLE -->

## Bless protocol summary

Run:

    ./scripts/bless-cute-full-cache.sh

Wall time: ~15 min on GB10 (1 cold-compile RW bootstrap + 5 fresh-container
RO validation trials with c2_replay_coherence + cache-reuse signals).

PASS criterion (per trial — all K must PASS):
- c2 `same_prompt_pass=true` AND `cross_prompt_pass=true` AND `unique=1`
- AOT load marker present in container log
- No `"saved AOT compiled function"` lines in container log
- AOT model sha256 unchanged from Phase-1 expected

K = 5 by default (Z1 empirical 5/5-vs-1/5 discrimination).
`NVLLM_BLESS_VALIDATION_TRIALS` may **raise** K. `--unsafe-trials N` (N<5)
sets `validation.unsafe_dev_trials = true`; production serve **refuses**
such manifests.

To replace an existing blessed manifest+cache:

    ./scripts/bless-cute-full-cache.sh --rebless

This runs the full two-phase flow, archives the prior manifest +
prior blessed cache (with timestamp + old `artifact_sha256[:8]` in name),
and atomically replaces them only if the new K=5 PASS gate holds.

## Refusal-mode FAQ

### "No matching manifest"

`serve-cute-full.sh` derived a `config_hash` for which no manifest exists
in this directory. Caused by: image rebuild (image ID is in `config_hash`),
HF revision change, layer-set/probe override, or first-time bring-up.
**Fix:** run `./scripts/bless-cute-full-cache.sh`.

### "DRIFT DETECTED"

A manifest exists but the on-disk blessed cache has a different sha256
or size for one of the listed files. Caused by: manual edit of
`~/.cache/nvllm/blessed/<hash>/`, disk corruption, or a rebless
where the new cache wasn't completed before the manifest was committed.
**Fix:** `./scripts/bless-cute-full-cache.sh --rebless`.

### "unsafe_dev_trials = true"

The manifest was produced with `--unsafe-trials N` for `N<5`. These
manifests are dev-only and refused by production serve.
**Fix:** `./scripts/bless-cute-full-cache.sh --rebless` (no `--unsafe-trials`).

## Operator notes

- The host blessed cache directory is **not** backed up by this repo. If you
  delete `~/.cache/nvllm/blessed/<hash>/` you must re-bless. Consider
  rsync-ing it to your backup target after every successful bless.
- The same `config_hash` should produce the same manifest filename each time
  (filename = `<human_label>_<config_hash[:7]>.json`). If a re-bless changes
  the filename, that's a configuration drift signal — investigate.
- Filename is for human review only. Lookup gates on the `config_hash` field
  inside the JSON; a renamed-but-stale manifest cannot poison the cache.
