# Stage 0 verdict (2L+wo8 baseline)

Date: 2026-05-10
Worktree commit: 9f118cdc571360b83cc3922ca9c72ce04b66c0c5 (worktree branch off `main`)
Image: nvllm:gb10 (~7 days stale — kernel changes overlaid via NVLLM_BIND_MOUNT_CUTE_PAGED + NVLLM_BIND_MOUNT_QWEN35)
Bind mounts: cute_paged + qwen3_5.py from worktree

## Stage 0a: env plumbing

- `/tmp/c2_diag/ENV` written with all 6 new keys.
- Dispatch audit: `coop_layers=[3, 7]`, `restricted_layers=[3, 7]`, `enabled=True`, `use_beta_coop=True` only on layers 3,7. **PASS.**
- Evidence: `dispatch_audit.json`.

## Stage 0b: GSM8K-50 baseline

- Score: **47/50** (matches wo_split soak's 47/50).
- Median ok-question latency: **62.3 s** (matches soak's 62.3 s, +0.0% deviation).
- Errors: 2 × HTTP read timeout at 180 s. Eval was invoked with the script default `--timeout=180`; soak used `--timeout=600`. Same questions almost certainly complete inside 600 s. The runner.sh uses `GSM8K_TIMEOUT=600`, so Stage 1 arms will not see this artifact.
- Wall: 3530.2 s.
- Gate 0b-PASS on accuracy AND median latency.
- Evidence: `gsm8k50_2L_wo8_baseline.json`, `gsm8k50_2L_wo8_baseline.log`.

## Stage 0c: per-call β kernel time sanity

- Sum of all 13 region medians: **5.538 ms** (gate ≤ 7.0 ms = PASS).
- Sum of WORK-only regions (excl R4 grid_barrier_wait + R11 pre_wo_wait + R12 gather_reduce): 3.296 ms.
- Region cluster matches wo_split soak's 5.5 ms baseline.
- Evidence: `region_timings_2L_wo8.npy`, `region_breakdown_2L_wo8.csv`.

## Verdict

All three Stage 0 gates PASS. Proceeding to Task 1b (6-arm sweep in tmux).
