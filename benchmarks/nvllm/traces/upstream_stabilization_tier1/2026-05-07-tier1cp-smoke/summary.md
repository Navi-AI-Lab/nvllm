# Upstream stabilization tier-1 — correctness smoke

## What this is

Apples-to-apples GSM8K + ShareGPT + long-decode + 2-concurrent smoke comparing the
**tier-1 cherry-pick stack** against the wo1 baseline that produced the wo_split
production-soak evidence. **This is a correctness smoke, not a performance claim** —
the 4 picks land in code paths that should not move SM120 decode latency.

## Picks under test (PR #10)

| sha | summary |
|---|---|
| `884b5ae34` | Disable flashinfer autotune temporarily due to correctness issues (vllm-project/vllm#41524) |
| `b383774ad` | fix(FLA): tighten write-side guard against `NULL_BLOCK_ID=0` (partial of upstream `d4cb783c1`) |
| `9e3a48cd8` | KV cache stride canonicalization for TMA alignment (manual port of upstream `66dfee712`) |
| `f3b4d3d09` | Gemma4 EAGLE-3 mixin + sliding-window cache realignment (manual port of upstream `e7cfd7c5b`) |

## Build

| | image | code commit (HEAD) | version |
|---|---|---|---|
| baseline | `nvllm:gb10` | `f79cf418b` (pre-soak main) | `0.3.1.dev69+gf79cf418b` |
| tier1cp | `nvllm:gb10-tier1cp` | `f3b4d3d09` (cherry-pick stack tip) | `0.3.1.dev102+gf3b4d3d09` |

Baseline image was the production image at the time of the wo_split soak.
Tier1cp image was built from the cherry-pick worktree via `/tmp/tier1cp-build-ctx`
(see `feedback_docker_build_worktree`).

## Model & config

- Model: `ig1/Qwen3.5-27B-NVFP4`
- KV cache dtype: `fp8_e4m3`
- Attention backend: CUTE_PAGED
- `CUTE_WO_SPLIT=1` (wo1 arm only — same as baseline)
- `--kernel-config '{"enable_flashinfer_autotune":false}'`
- `--gpu-memory-utilization 0.85`

## GSM8K 50 (seed=42, max_tokens=512, /v1/completions)

| | correct | acc | total wall | mean | median |
|---|---|---|---|---|---|
| baseline (wo1) | 48 | 96.0% | 3737.8 s | 74.8 s | 67.7 s |
| tier1cp        | 48 | 96.0% | 3720.6 s | 74.4 s | 67.4 s |
| Δ              | **0** | **0** | **−17.2 s (−0.46 %)** | −0.4 | −0.3 |

**Answer-level divergences vs baseline: 0/50.** Same two questions wrong (Q1 2280→2180
arithmetic miss; Q45 192→1 reasoning miss), every other answer byte-identical.
Sub-1 % wall delta is well within thermal noise on a workstation GPU and is **not**
claimed as a perf improvement.

## ShareGPT slice (30 conversations, max_tokens=128, seed=42)

Completed 30 conversations with no errors. Per-turn walltimes are recorded in
`wo1/primary/run01/sharegpt.json` and `sharegpt_wall_tpot.csv`.

## Long decode (max_tokens=2048, seed=42)

| | wall | finish | chunks |
|---|---|---|---|
| tier1cp | 986.7 s | length | 2048 |

## 2-concurrent probe

| request | wall | ttft | tpot p95 | finish |
|---|---|---|---|---|
| req_a | 57.37 s | 2882 ms | 433.40 ms | length |
| req_b | 54.81 s | 2882 ms | 433.25 ms | stop   |

## How to reproduce

```bash
# Build (from a non-worktree clone)
git clone --branch cherry-pick/upstream-stabilization-tier1 --single-branch \
  /home/natfii/docker/nvllm /tmp/tier1cp-build-ctx
cd /tmp/tier1cp-build-ctx
docker build -f docker/Dockerfile.gb10 -t nvllm:gb10-tier1cp .

# Smoke (run from main checkout, evidence-PR runner)
cd /home/natfii/docker/nvllm
NVLLM_IMAGE=nvllm:gb10-tier1cp WO_SPLITS=1 REPLAYS=1 PHASES=primary \
  OUT_DIR=/tmp/tier1cp-soak \
  bash docs/research/2026-05-04-wo-split-prod-soak/runner.sh
```

## Verdict

Cherry-pick stack is **bit-clean against the wo1 baseline**: zero answer divergences,
sub-1 % wall delta (within thermal noise), no errors in ShareGPT / long decode /
concurrent. Safe to merge PR #10.

## Caveats

- No nsys trace was captured. AGENTS.md §4 requires nsys for **performance claims**;
  this smoke makes none. If a perf claim is later attached to this stack, capture
  per-kernel evidence at that time.
- `metadata.json` records the host-side script commit (`a131443ff` on `main`),
  not the image-side cherry-pick HEAD. The image-side commit is `f3b4d3d09`,
  surfaced via `vllm --version` inside the container.
