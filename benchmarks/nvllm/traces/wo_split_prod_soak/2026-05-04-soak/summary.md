## Per-arm summary

| arm | gsm8k | replays | wall mean (s) | wall stddev | tpot p50 (ms) | tpot p95 (ms) | tpot p99 (ms) | longdecode p95 |
|-----|-------|---------|---------------|-------------|---------------|---------------|---------------|----------------|
| wo1 | 48/50 | 5 | 8104.75 | 6.74 | 467.98 | 510.73 | 530.38 | 518.54 |
| wo2 | 47/50 | 5 | 7910.47 | 3.39 | 450.43 | 493.07 | 511.96 | 500.75 |
| wo4 | 48/50 | 5 | 7829.37 | 4.22 | 443.63 | 486.66 | 506.36 | 494.26 |
| wo8 | 47/50 | 5 | 7833.98 | 1.98 | 441.94 | 485.21 | 504.69 | 491.69 |

## Pairwise vs baseline (wo1)

| arm | wall % change | tpot p95 Δ (ms) | gsm8k Δ |
|-----|---------------|-----------------|---------|
| wo2 | +2.4% | -17.66 | -1 |
| wo4 | +3.4% | -24.07 | +0 |
| wo8 | +3.3% | -25.52 | -1 |

## Verdicts

_Wall threshold: 5.0% _

- **wo1**: baseline
- **wo2**: keep opt-in (wall +2.4%, tpot p95 -17.66 ms vs baseline noise 0.95)
- **wo4**: keep opt-in (wall +3.4%, tpot p95 -24.07 ms vs baseline noise 0.95)
- **wo8**: keep opt-in (wall +3.3%, tpot p95 -25.52 ms vs baseline noise 0.95)
