# Phase A Q2 repro — GSM8K sanity across D2e and Phase A, fused vs unfused

**Date:** 2026-04-20
**Model:** `natfii/Qwen3.5-27B-NVFP4-Opus-GB10` (Opus-distilled, NVFP4)
**Serve config:** `--attention-backend CUTE_PAGED`, `--kv-cache-dtype fp8_e4m3`,
`--max-num-seqs 4`, `--compilation-config {"cudagraph_mode":"PIECEWISE"}`,
all with temperature=0.
**Harness:** `docs/research/phase_a_gsm8k_repro/run_repro.sh` (8-question
GSM8K sanity), `run_q2_repeat.sh` / `run_q2_repeat_unfused.sh` (Q2 x5 in
single session).

## Headline

**The "Phase A broke Q2 math" claim was a misattribution. Both D2e
and Phase A produce 4 different Q2 outputs in a 5-run same-session
test at temperature=0 when fusion is on. Disabling fusion makes D2e
produce 5 byte-identical correct "10" outputs.**

Non-determinism is per-request, lives entirely in the fused CuTe MLP
+ fused CuTe attention kernel reduction paths (almost certainly
`atomicAdd`-based CTA arrival ordering), and is amplified to
token-level flips by the distilled model's knife-edge argmax margins.

## Runs

### GSM8K 8-question sanity (per-question, temperature=0)

| Run | Image | Q2 raw | Score | Verdict |
|---|---|---|---|---|
| 1 | `nvllm:gb10-phaseD2e` | `"50/5. 12/1. 2."` | 7/8 | Q2 WRONG (gibberish) |
| 2 | `nvllm:gb10-phaseD2e` | `"50/5 = 10\nSo the answer is:"` | 7/8 | Q2 WRONG (extraction picks "50") |
| 1 | `nvllm:gb10-phaseA` | `"10\nWhat was the question?..."` | 8/8 | all correct |
| 2 | `nvllm:gb10-phaseA` | `"50/5. 12/1. 12/"` | 7/8 | Q2 WRONG (gibberish) |

Dirs: `d2e/`, `d2e_run2/`, `phaseA/`, `phaseA_run2/`.

Across 4 runs, 4 different Q2 raw outputs. No image is systematically
correct or incorrect; the Phase A 8/8 was a lucky roll.

### Q2 x5 in single D2e server session — fused path (CUTE_*_FUSION=1)

```
run 1: " 20/1 dollars\n\n</think>\n\n$ 20"
run 2: " 20/12. 20/12 = 1"
run 3: " 20/12. 20/12 = 1"      (matches run 2 by luck)
run 4: " 1200/60 =  20."
run 5: " 10\n</think>\nTo determine..."  ← CORRECT, 1/5
```

Four unique outputs across 5 same-session calls. Dir: `d2e_q2_repeat/`.

### Q2 x5 in single D2e server session — unfused path (CUTE_*_FUSION=0)

```
run 1: " 10. Weng earned $10.\nQ: Betty is"
run 2: " 10. Weng earned $10.\nQ: Betty is"
run 3: " 10. Weng earned $10.\nQ: Betty is"
run 4: " 10. Weng earned $10.\nQ: Betty is"
run 5: " 10. Weng earned $10.\nQ: Betty is"
```

**5 byte-identical outputs. 5/5 correct. Deterministic.** Dir:
`d2e_q2_repeat_unfused/`.

## Interpretation

- Fused path: 4 unique outputs / 5 runs → per-request non-deterministic
- Unfused path: 1 unique output / 5 runs → per-request deterministic
- Non-determinism is **entirely localized** to the fused CuTe kernels
  that do `atomicAdd` cross-CTA sync (mlp_partial_fp32 accumulator
  + mlp_arrival_count in the MLP fused kernel; similar reductions in
  the fused attention W_O / RMSNorm path).

The distilled Opus model makes this visible because its softmax
margins are tight — a ULP-level FP32 drift in accumulator state flips
the argmax on Q2's first generated token between "10", "20", "1200",
or "1", all of which are arithmetically connected to the "12 × 50/60"
input. A non-distilled model with wider margins would hide the bug.

## What this falsifies

1. **"Source-hash PTX drift breaks FP4 numerics"** (D3a working theory):
   PTX is byte-identical per `396c3bbcf` (MLP) and `1f008b4fe`
   (attention). Drift is runtime, not compile-time.
2. **"Phase A's Constexpr refactor broke Q2"**: Phase A and D2e have
   the same latent per-request non-determinism. Neither is consistently
   correct on the fused path.
3. **"D2e ships 8/8 GSM8K"**: 7/8 with different raw outputs across
   runs. Prior 8/8 claims were sampling variance.

## Fix direction

- Short-term: serve with `CUTE_MLP_FUSION=0 CUTE_ATTN_FUSION=0` until
  the kernel fix lands. Deterministic, correct, losing the fused-path
  speedup.
- Medium-term: audit reductions in `mlp_kernel.py` and the fused
  attention path. Replace `atomicAdd` on the partial-sum accumulator
  with a deterministic reduction (tree reduce on CTA index, or a
  fixed-slot scatter + deterministic gather).
- Longer-term: every kernel-quality harness must be x5+ with dedup
  before claiming "works." One-shot runs can't detect this.

## How to reproduce

```bash
# Full 8-question GSM8K under each image:
bash docs/research/phase_a_gsm8k_repro/run_repro.sh \
    nvllm:gb10-phaseD2e \
    "$PWD/benchmarks/nvllm/traces/phase_a_gsm8k_repro/2026-04-20/d2e"

bash docs/research/phase_a_gsm8k_repro/run_repro.sh \
    nvllm:gb10-phaseA \
    "$PWD/benchmarks/nvllm/traces/phase_a_gsm8k_repro/2026-04-20/phaseA"

# Q2 x5 in single session, fused path:
bash docs/research/phase_a_gsm8k_repro/run_q2_repeat.sh \
    nvllm:gb10-phaseD2e \
    "$PWD/benchmarks/nvllm/traces/phase_a_gsm8k_repro/2026-04-20/d2e_q2_repeat"

# Q2 x5 in single session, unfused path (deterministic):
bash docs/research/phase_a_gsm8k_repro/run_q2_repeat_unfused.sh \
    nvllm:gb10-phaseD2e \
    "$PWD/benchmarks/nvllm/traces/phase_a_gsm8k_repro/2026-04-20/d2e_q2_repeat_unfused"
```

## Environment

- Both images: torch `2.12.0.dev20260402+cu132`, cutlass-dsl `4.4.2`,
  CUDA device `NVIDIA GB10`
- Image git HEADs:
  - D2e: `dc4bc7d6e` (branch `feat/unreal-kernel-phase-d`)
  - Phase A: `316be8c1b` (same branch) + uncommitted Constexpr edits
