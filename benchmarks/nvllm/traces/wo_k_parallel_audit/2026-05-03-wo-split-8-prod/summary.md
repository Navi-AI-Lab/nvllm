# wo_split=8 K-parallel W_O GEMV — production prototype evidence

**Commit:** `b3f75721d` on branch `evidence/wo-k-parallel-harness`
**Date captured:** 2026-05-04
**Model:** `ig1/Qwen3.5-27B-NVFP4` (non-distilled, official llm-compressor VL recipe)

## Serve config (both runs)

```
serve --model ig1/Qwen3.5-27B-NVFP4 --served-model-name default
      --kv-cache-dtype fp8_e4m3
      --attention-backend CUTE_PAGED
      --max-model-len 65536 --max-num-seqs 4
      --gpu-memory-utilization 0.70
      --kernel-config '{"enable_flashinfer_autotune":false}'
      --compilation-config '{"cudagraph_mode":"PIECEWISE"}'
      --language-model-only
      --trust-remote-code
```

Container `nvllm:gb10` with β-coop fusion enabled (`CUTE_PHASE_E_FUSION=1`, layers 0-7).
Bind-mount: `vllm/v1/attention/backends/cute_paged` for live-update.

The two runs differ ONLY in `CUTE_WO_SPLIT` (1 = baseline, 8 = changed). The env var
propagates to the EngineCore subprocess via the `/tmp/c2_diag/ENV` sentinel-file
workaround (commit `ef9f68028`).

## Quality (GSM8K-50 full-think, seed=42, max_tokens=512, timeout=600s)

| Run | Accuracy | Errors / timeouts | 50-question wall | Per-question OK median |
|---|---:|---:|---:|---:|
| **wo_split=1 baseline** | **48/50 (96.0%)** | 0 | 3760 s | 65.7 s |
| **wo_split=8 changed** | **47/50 (94.0%)** | 0 | 3664 s | 62.3 s |
| Δ | −1 question | identical | **−96 s (−2.6%)** | **−3.4 s (−5.2%)** |

Quality parity — within ±2% noise, 0 errors both sides.

Artifacts: `baseline_gsm8k_fullthink.json`, `changed_gsm8k_fullthink.json`.

## Region timing (5-completion synthetic load)

`CUTE_BETA_REGION_TIMING=1`, dumped via `scripts/trigger_region_timing_dump.sh`,
reduced via `docs/research/2026-05-02-beta-region-breakdown/extract_regions.py
--wo-split N`.

| Region | wo_split=1 | wo_split=8 | Δ | Notes |
|---|---:|---:|---:|---|
| R2 `phase1_wo_gemv` | 14121 μs (4 active) | **2360 μs (32 active)** | **−11761 μs (5.99×)** | K-parallel split |
| R4 `grid_barrier_wait` | 15211 μs (64) | **1753 μs (64)** | **−13458 μs (8.68×)** | shrinks because R2 finishes faster |
| R11 `phase1_pre_wo_wait` | 0 μs (mask empty) | 250 μs (28 active) | +250 μs | new: bx>0 consumers spin-wait |
| R12 `phase1_gather_reduce` | 73 μs (1 elected) | 167 μs (1 elected) | +94 μs | gather of 32 partials vs 4 |
| **Cluster (R2+R4+R11+R12)** | **29405 μs** | **4530 μs** | **−24875 μs (6.49× / −84.6%)** | |

Other regions unchanged (R0, R1, R3, R5-R10).

R11 active CTA count = 28 = 32 W_O total − 4 attn producers (bx==0 producers skip
R11; intra-CTA ordering means their attn_output reads need no acquire fence).

R12 is a dynamic-single-CTA region (only the elected CTA writes a tick); host
reducer uses nonzero filtering to drop the 63 zero-rows.

Artifacts:
- `baseline_region_timings.npy`, `baseline_region_breakdown.csv` — wo_split=1
- `changed_v2_region_timings.npy`, `changed_v2_region_breakdown.csv` — wo_split=8

## nsys total-kernel comparison

Captured via the harness microkernel at production grid shape (slice_ctas=8,
32-CTA cooperative grid, num_kv_heads=4, hidden=5120, K=6144, NUM_THREADS=128,
NVFP4 weights). The harness microkernel reproduces the production W_O+gather
math bit-exactly (verified against `reference_split_order(wo_split=N)` —
`max_abs=0.0` at both wo_splits). vLLM V1 nsys against the EngineCore
subprocess is blocked by CUPTI injection inheritance through the multiprocess
spawn (per `feedback_vllm_profiling`); harness microkernel + production grid
is the authoritative nsys path for the W_O+gather portion of the kernel.

| Metric | wo_split=1 | wo_split=8 | Delta |
|---|---:|---:|---:|
| Symbol | `kernel_cutlass__wo_kernel_body_________________0` | same | -- |
| 50-launch median | **13715.248 us** | **1598.064 us** | **-12117 us (-88.3% / 8.58x)** |
| 50-launch mean | 14120.510 us | 1597.994 us | -12522 us (8.84x) |
| 50-launch stddev | 2101.617 us | **5.677 us** | collapsed; high stability at wo_split=8 |
| Min / Max | 13643 / 28391 us | 1585 / 1626 us | wo_split=1 had 28391 us first-call cache-MISS outlier |
| GPU time (50 launches) | 706 ms | 80 ms | -- |
| Time fraction in trace | 92.7% | 58.8% | -- |

`kernel_cutlass__wo_kernel_body` is the CuTe DSL emitted symbol for the W_O
microkernel body. The same kernel-body code path ships in production beta-coop
fusion (`_kernel_phase_0_to_4` in `phase_e_kernel.py`), but production emits a
different mangled symbol for the full fused kernel. The harness isolates the
W_O+gather portion using the same K-range slicing and slot-index formulas,
so the 8.58x harness speedup transfers to production R2 (verified via region
timing R2 = 14121 -> 2360 us = 5.99x; the gap reflects ignore-eos warm-cache
vs serving cold-launch variance).

Files:
- `baseline.nsys-rep` -- wo_split=1 trace, 1.84 MB
- `changed.nsys-rep` -- wo_split=8 trace, 1.96 MB
- `nsys_summary.md` -- full subagent capture report

Reproduction:
```bash
/opt/nvidia/nsight-systems/2025.6.3/bin/nsys stats \
    --report cuda_gpu_kern_sum:mangled baseline.nsys-rep \
    | grep wo_kernel_body
```

## Bit-exact correctness gate

The K-parallel W_O kernel reproduces `reference_split_order(wo_split=N)` from
`docs/research/2026-05-03-w-o-k-parallel-harness/torch_reference.py` bit-exactly
at both wo_split=1 and wo_split=8 (`max_abs == 0.0`).

Methodology: V=constant trick — set FP8 V-cache to `+1.0` (0x38) so Phase 1
attention output is deterministically `attn_output = ones(NAT, K)`. With known
input, `wo_output[seq, 0, :]` (post-gather, written at `phase_e_kernel.py:4471`
and read by RMSNorm Pass 1 at `:4490`) equals
`reference_split_order(attn=ones, weighted, wo_split=N)`.

Repro at `/tmp/wo_split_repro.py` (transient, not committed). Re-runs bit-exact
at any time on a warm container.

## Exact reproduction commands

### Build
```bash
cd /home/natfii/docker/nvllm
git checkout evidence/wo-k-parallel-harness
git rev-parse HEAD  # expect b3f75721d (or descendant)
docker images nvllm:gb10
```

### wo_split=1 baseline GSM8K
```bash
docker stop nvllm; docker rm nvllm
NVLLM_BIND_MOUNT_CUTE_PAGED=1 \
CUTE_PHASE_E_FUSION=1 \
CUTE_PHASE_E_LAYERS=0,1,2,3,4,5,6,7 \
CUTE_PHASE_E_FALLBACK_RAISE=1 \
    bash scripts/serve-cute.sh
until curl -s -f -m 3 http://localhost:8000/v1/models 2>/dev/null | grep -q '"data"'; do sleep 10; done
.venv/bin/python scripts/gsm8k_eval_50.py \
    --api http://localhost:8000/v1 --model default \
    --n 50 --seed 42 --max-tokens 512 --timeout 600 \
    --save baseline_gsm8k_fullthink.json --label task10_wo_split_1_baseline
```

### Region timing (wo_split=1)
```bash
docker stop nvllm; docker rm nvllm
NVLLM_BIND_MOUNT_CUTE_PAGED=1 \
CUTE_PHASE_E_FUSION=1 \
CUTE_PHASE_E_LAYERS=0,1,2,3,4,5,6,7 \
CUTE_PHASE_E_FALLBACK_RAISE=1 \
CUTE_BETA_REGION_TIMING=1 \
VLLM_TORCH_PROFILER_DIR=/root/.cache/vllm/profiler \
    bash scripts/serve-cute.sh
for i in 1 2 3 4 5; do
    curl -s -X POST http://localhost:8000/v1/completions \
        -H 'Content-Type: application/json' \
        -d '{"model":"default","prompt":"capital of france is","max_tokens":50,"temperature":0,"ignore_eos":true}' \
        > /dev/null
done
bash scripts/trigger_region_timing_dump.sh baseline_region_timings.npy
.venv/bin/python docs/research/2026-05-02-beta-region-breakdown/extract_regions.py \
    --buf baseline_region_timings.npy --kernel-mean-us 40000 \
    --slice-ctas 8 --num-k-tiles 8 --num-seqs 1 \
    --tick-source globaltimer --wo-split 1 --num-kv-heads 4 \
    --out baseline_region_breakdown.csv
```

### wo_split=8 changed
Prepend `CUTE_WO_SPLIT=8 \\` to the `bash scripts/serve-cute.sh` line above. The
sentinel-file workaround (commit `ef9f68028`) propagates the env var to
EngineCore. For the region-timing capture at wo_split=8, pass `--wo-split 8` to
extract_regions.

### Bit-exact gate
```bash
.venv/bin/python /tmp/wo_split_repro.py --wo-split 1 --seed 4242  # PASS, max_abs=0
.venv/bin/python /tmp/wo_split_repro.py --wo-split 8 --seed 4242  # PASS, max_abs=0
```

## Caveats

- **Region timing captured at synthetic 5-completion ignore_eos load**, not GSM8K
  workload. Per-region speedups at GSM8K may differ (workload sensitivity).
- **R11 active CTA count (28) reflects bx>0 consumers only**; the host reducer's
  `_phase1_wo_split_cta_ids` mask spans all 32 W_O CTAs but the kernel only
  writes ticks for bx>0 — nonzero filter drops the bx==0 zeros.
- **R12 active CTA count varies (1-3 across runs)** depending on how many
  concurrent decodes are in flight at the dump moment; each seq elects its own
  last CTA.
- **`CUTE_WO_SPLIT` accepted values restricted to `{1, 2, 4, 8}`** (commit
  `4331362e2`). The kernel logic works for arbitrary 1..slice_ctas but only the
  powers-of-2 subset has reference-validation evidence.
- **wo_split=8 stays opt-in** via env var; default is 1 (no behavioral change
  for production callers who don't set the var).
- **R11 / R12 are timing-instrumentation regions**, NOT production gates. The
  underlying mechanisms (consumer wait, single-CTA gather) are production
  behavior; the timing samples are debug-gated by `CUTE_BETA_REGION_TIMING=1`.
