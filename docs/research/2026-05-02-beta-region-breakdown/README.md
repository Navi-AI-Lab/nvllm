# β-coop per-region timing breakdown

**Purpose.** Decide whether NVFP4 GEMV K-parallel reduction (Veitner
pattern) is the right next bet for β-coop by measuring what fraction of
PhaseE_Beta_Kernel runtime is in K-reducible regions (Phase 1 W_O,
Phase 3 stage 3a FC1, Phase 3 stage 3c FC2).

**Design.** Single leg, lower8 production config. Two boots: profile
boot (CUTE_BETA_REGION_TIMING=1, torch profiler ON) + sanity boot
(CUTE_BETA_REGION_TIMING=0, GSM8K-50 to confirm timing path doesn't
regress quality). Plus an NCU adjunct capture.

## Decision matrix

| Outcome (sum of regions 2 + 7 + 9, calibrated frac of kernel)        | Action                                                      |
| :------------------------------------------------------------------- | :---------------------------------------------------------- |
| K-reducible regions ≥ 50% of kernel μs                               | **Strong go.** Prototype Extra Blocks split-K on FC1 first. |
| K-reducible regions 40-50%                                           | **Proceed.** Prototype on FC1; budget 2 weeks before re-evaluate. |
| K-reducible regions 25-40%                                           | **Conditional.** Pursue only if NCU shows memory-bound classification AND no other low-hanging fruit. |
| K-reducible regions < 25%                                            | **No-go for K-parallel alone.** Broader kernel restructuring needed; revisit β architecture. |

GSM8K gates per leg:
- Sanity boot (timing-off): correct ≥ 47/50 (matches blessed production).
- Profile boot (timing-on): not run (instrumented kernel μs is not
  representative of production μs).

## Reproduction

```bash
# tmux required — total wall time ~75-90 min
tmux new -s region-breakdown
bash docs/research/2026-05-02-beta-region-breakdown/run_breakdown.sh
```

## Output layout

```
benchmarks/nvllm/traces/cute_paged_attn/2026-05-02-beta-region-breakdown/
  profile_serve.log
  profile_kernels.csv         # nsys per-kernel μs (denominator for calibration)
  region_timings.npy          # raw (num_ctas, 11, 2) u64 buffer (last launch)
  region_breakdown.csv        # reduced output: per-region mean/p50/p99/frac
  ncu/phase_e_beta_ncu.csv    # NCU adjunct
  metadata.json               # image/git/env/config_hash
  sanity_serve.log            # timing-off boot
  sanity_gsm8k.json           # 47/50 expected
  summary.md                  # written by hand from above artifacts
```

## Caveats baked in

- **Last-launch only.** region_timing buffer captures only the last
  PhaseE_Beta_Kernel launch of the timed burst. Variance across launches
  is not measured; if the captured launch is anomalous (e.g.
  cold-cache), the breakdown is misleading. Mitigation: a "sample
  dimension" extension is documented in the next-step list but not
  built in v1.
- **Phase 0 and Phase 1 are NOT globally sequential.** Only `cta_id ==
  0` (`bx==0 && by==0`) executes Phase 0; the other Phase 1 CTAs
  (`cta_ids {8, 16, 24}` for `slice_ctas=8`) can begin Phase 1 work
  earlier. The breakdown is therefore "per-CTA local-region evidence
  + nsys wall-time denominator," NOT a clean additive timeline. The
  K-reducible *fraction* (regions 2 + 7 + 9 / kernel μs) is robust to
  this; do not present per-region medians as if they sum to the kernel
  μs.
- **Globaltimer overhead is not zero.** Verify against the
  Task 5 overhead measurement before drawing μs-comparison conclusions.
- **Calibration uses nsys mean μs as denominator.** Per-CTA medians give
  the representative single-CTA wall-time contribution within a region
  (concurrent execution → median, not sum). The reported fractions are
  ratio-of-kernel, not a partition that sums to 1.
- **Tick units depend on Task 2 outcome.** If `%globaltimer` smoke
  passes, ticks are nanoseconds (cross-SM synchronized) and `median_us`
  is reported. If we fall back to `%clock64`, ticks are per-SM cycles
  and `median_us` is NaN — only `frac_of_kernel` (computed via ticks
  ratio) is comparable.
- **clockRate is not trusted.** SM clock changes dynamically; even with
  `%clock64`, we never multiply ticks by `props.clockRate` to recover
  μs. nsys is the wall-time anchor.
- **No production gate fired.** The recommendation is which kernel work
  to pursue next; it does NOT flip any default. Default flip requires
  a separate prototype-evaluate-bench cycle for the chosen direction.
