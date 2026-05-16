# Memtester 32G + 64G band — CLEAN (2026-05-14)

## Status

**CLEAN.** All three memtester runs returned `rc=0`, `0 FAILURE` lines.
This rules out RAM in the 4G / 32G / 64G bands as a hardware contributor
to the β-coop sustained-load collapse arc, prior to running D2.7.

D2.7 was subsequently skipped because the SSM zero-on-realloc fix
closed the diagnosis arc (see [#13](https://github.com/Navi-AI-Lab/nvllm/pull/13)
and `memory:project_beta_coop_sustained_collapse`). This evidence is
kept because Spark fleet-level HW caveats are durable and may matter
for future debugging.

## Why this gate exists

NVIDIA forum-confirmed Spark issues that change priors for any host-state
debugging:

- **LPDDR5x on Spark has NO ECC.** Silent bit-flip corruption is possible.
  ([NVIDIA forum, 2026-04-30](https://forums.developer.nvidia.com/t/ecc-on-dgx-spark/354376))
- **Memtester reproducibly fails on the 32-64 GB band on some Spark units**,
  passes ≤24 GB. Exactly where vLLM model weights + KV cache live.
  Unresolved upstream as of 2026-05-14.
  ([NVIDIA forum, 2026-05-05](https://forums.developer.nvidia.com/t/memtester-failures-and-missing-nvidia-dgx-diagnostics-suite/353950))
- **Fieldiag error 082-000-1-020000021139** ("thermal sensor broken or
  miscalibrated") confirmed on at least one Spark unit under vLLM load
  (RMA approved). Means `nvidia-smi` throttle-reasons is NOT a reliable
  HW-health gate on Spark.
  ([NVIDIA forum, 2026-05-12](https://forums.developer.nvidia.com/t/dgx-spark-hangs-under-vllm-load-fieldiag-fails-on-the-thermal-sensor/369381))

The 32-64 GB band is the most-suspected one per the second link, which
is why this gate sweeps three bands (sanity + two suspect bands).

## What ran

| Band | Allocation | Loops | Elapsed | rc | FAILURE count | log |
|---|---|---|---|---|---|---|
| sanity_4G | 4096 MB | 1 | 436s (7m 16s) | 0 | 0 | `memtester_sanity_4G.log` (~1.1 MB) |
| band_32G | 32768 MB | 1 | 3391s (56m 31s) | 0 | 0 | `memtester_band_32G.log` (~8.6 MB) |
| band_64G | 65536 MB | 1 | 6764s (1h 52m 44s) | 0 | 0 | `memtester_band_64G.log` (~17.2 MB) |

Total wall: 2h 56min (19:37:03 → 22:33:35 EDT, 2026-05-14).

Pre-run host: 119 GiB total, 85 GiB free, 0 swap used.
Post-run host: 119 GiB total, 84 GiB free, 0 swap used. No leak.

## Host

| Field | Value |
|---|---|
| host | navi-ai (NVIDIA DGX Spark, GB10) |
| hardware | 128 GB LPDDR5x unified, SM120 |
| host_driver | 590.48.01 (per parallel evidence dir `project_ssm_zero_on_realloc_shipped`) |
| host_kernel | 6.17.0-1014-nvidia |
| memtester version | bundled with Ubuntu (`apt install memtester`) |
| tests run | all default memtester subtests per loop (Stuck Address, Random Value, Compare XOR, Compare SUB, Compare MUL, Compare DIV, Compare OR, Compare AND, Sequential Increment, Solid Bits, Block Sequential, Checkerboard, Bit Spread, Bit Flip, Walking Ones, Walking Zeroes, 8-bit Writes, 16-bit Writes) |

## How to reproduce

```bash
# Install memtester if not present (Ubuntu/Debian).
sudo apt-get install -y memtester

# Run each band as root (memtester locks unswappable pages, needs CAP_IPC_LOCK).
mkdir -p evidence/
free -h | tee evidence/free_pre.txt

# Sanity (4 GB, 1 loop).
sudo memtester 4096 1 > evidence/memtester_sanity_4G.log 2>&1
echo "rc=$?"
grep -c FAILURE evidence/memtester_sanity_4G.log

# 32 GB band (~1h).
sudo memtester 32768 1 > evidence/memtester_band_32G.log 2>&1
echo "rc=$?"
grep -c FAILURE evidence/memtester_band_32G.log

# 64 GB band (~2h).
sudo memtester 65536 1 > evidence/memtester_band_64G.log 2>&1
echo "rc=$?"
grep -c FAILURE evidence/memtester_band_64G.log

free -h | tee evidence/free_post.txt
```

A CLEAN result is `rc=0` AND `grep -c FAILURE` returns `0` for all
three bands. The logs are massive (memtester writes a spinner + per-
subtest "ok" line for every loop) but the verdict is binary.

## What this evidence does NOT prove

- It does not exonerate the host for *transient* corruption under
  thermal or sustained-load stress. Memtester runs in isolation; the
  Spark thermal-sensor errata above means we can't tell if the unit
  was throttling during the run.
- It does not cover the 64-120 GB band (would require >2× the wall time
  and approach the OOM line).
- It does not test write-coherence between CPU and integrated GPU
  domains in unified memory; that's an integrated-platform concern not
  covered by a CPU-only memory test.

For collapse debugging, this gate is a NECESSARY-not-SUFFICIENT check.
A CLEAN result rules out the most-suspected memtester signature; a FAILURE
result would have shortcut directly to RMA.

## Related

- `memory:project_d2_7_hw_followup` — the HW discriminator plan (dormant after SSM fix)
- `memory:project_beta_coop_sustained_collapse` — the collapse arc this gate served
- `benchmarks/nvllm/traces/ssm_zero_on_realloc/2026-05-15-sentinel-ablation/summary.md` — the eventual resolution
