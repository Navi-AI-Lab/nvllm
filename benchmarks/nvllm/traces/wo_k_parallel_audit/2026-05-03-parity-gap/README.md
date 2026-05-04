# wo_k_parallel_audit / 2026-05-03-parity-gap

Machine-readable inventory. One line per artefact; no perf summary.

## Audit axes

| Config | total_grid_ctas | active_wo_ctas | wo_split | slice_ctas |
|--------|-----------------|----------------|----------|------------|
| A      | 64              | 4              | 1        | 16         |
| B      | 32              | 4              | 1        | 8          |
| C      | 32              | 32             | 8        | 8          |

## Harness inputs (shared across configs)

- Hidden 5120, num_kv_heads 4, num_q_heads 24, head_dim 256, K=6144
- B (num_active_tokens) = 1, seed = 4242, launches = 50
- Cooperative launch: True (hardwired in microkernel.py:463)
- Disk-cache: /tmp/cute_harness_cache_v3 (warm)
- nsys: --trace=cuda,nvtx, container --privileged for CUPTI

## File inventory

A_prod_grid_4_active/config.json (1196 bytes) -- harness-emitted run config (slice_ctas, wo_split, gather_ctas, cache_key, cooperative=true)
A_prod_grid_4_active/launch_args.json (817 bytes) -- exact CLI args used to drive the run (audit-spec metadata)
A_prod_grid_4_active/stdout.log (350 bytes) -- harness stdout incl. cache HIT/MISS line and one-line gate summary
A_prod_grid_4_active/timing.csv (2731 bytes) -- 50 device-side CUDA-event launch timings (us) from the plain run
A_prod_grid_4_active/correctness_gate_split_order.json (191 bytes) -- AUTHORITATIVE bit-exact gate vs reference_split_order(wo_split)
A_prod_grid_4_active/correctness_vs_chained.json (106 bytes) -- DIAGNOSTIC vs chained-FMA reference
A_prod_grid_4_active/correctness_vs_matmul.json (125 bytes) -- DIAGNOSTIC vs cuBLAS-tree matmul reference
A_prod_grid_4_active/run.nsys-rep (1843104 bytes) -- nsys CUDA+NVTX trace of the same harness invocation under --privileged
A_prod_grid_4_active/nsys_run/config.json (1196 bytes) -- harness config from inside the nsys profile run
A_prod_grid_4_active/nsys_run/timing.csv (2731 bytes) -- 50 device-side timings from inside the nsys profile run (CUPTI overhead biased)
A_prod_grid_4_active/nsys_run/correctness_gate_split_order.json (191 bytes) -- gate result from inside the nsys profile run
A_prod_grid_4_active/nsys_run/correctness_vs_chained.json (106 bytes) -- diagnostic from inside the nsys profile run
A_prod_grid_4_active/nsys_run/correctness_vs_matmul.json (125 bytes) -- diagnostic from inside the nsys profile run

B_harness_grid_4_active/config.json (1195 bytes) -- harness-emitted run config (slice_ctas, wo_split, gather_ctas, cache_key, cooperative=true)
B_harness_grid_4_active/launch_args.json (824 bytes) -- exact CLI args used to drive the run (audit-spec metadata)
B_harness_grid_4_active/stdout.log (349 bytes) -- harness stdout incl. cache HIT/MISS line and one-line gate summary
B_harness_grid_4_active/timing.csv (2731 bytes) -- 50 device-side CUDA-event launch timings (us) from the plain run
B_harness_grid_4_active/correctness_gate_split_order.json (191 bytes) -- AUTHORITATIVE bit-exact gate vs reference_split_order(wo_split)
B_harness_grid_4_active/correctness_vs_chained.json (106 bytes) -- DIAGNOSTIC vs chained-FMA reference
B_harness_grid_4_active/correctness_vs_matmul.json (125 bytes) -- DIAGNOSTIC vs cuBLAS-tree matmul reference
B_harness_grid_4_active/run.nsys-rep (1205511 bytes) -- nsys CUDA+NVTX trace of the same harness invocation under --privileged
B_harness_grid_4_active/nsys_run/config.json (1195 bytes) -- harness config from inside the nsys profile run
B_harness_grid_4_active/nsys_run/timing.csv (2731 bytes) -- 50 device-side timings from inside the nsys profile run (CUPTI overhead biased)
B_harness_grid_4_active/nsys_run/correctness_gate_split_order.json (191 bytes) -- gate result from inside the nsys profile run
B_harness_grid_4_active/nsys_run/correctness_vs_chained.json (106 bytes) -- diagnostic from inside the nsys profile run
B_harness_grid_4_active/nsys_run/correctness_vs_matmul.json (125 bytes) -- diagnostic from inside the nsys profile run

C_harness_grid_32_active/config.json (1197 bytes) -- harness-emitted run config (slice_ctas, wo_split, gather_ctas, cache_key, cooperative=true)
C_harness_grid_32_active/launch_args.json (891 bytes) -- exact CLI args used to drive the run (audit-spec metadata)
C_harness_grid_32_active/stdout.log (350 bytes) -- harness stdout incl. cache HIT/MISS line and one-line gate summary
C_harness_grid_32_active/timing.csv (2781 bytes) -- 50 device-side CUDA-event launch timings (us) from the plain run
C_harness_grid_32_active/correctness_gate_split_order.json (191 bytes) -- AUTHORITATIVE bit-exact gate vs reference_split_order(wo_split)
C_harness_grid_32_active/correctness_vs_chained.json (131 bytes) -- DIAGNOSTIC vs chained-FMA reference
C_harness_grid_32_active/correctness_vs_matmul.json (124 bytes) -- DIAGNOSTIC vs cuBLAS-tree matmul reference
C_harness_grid_32_active/run.nsys-rep (1837463 bytes) -- nsys CUDA+NVTX trace of the same harness invocation under --privileged
C_harness_grid_32_active/nsys_run/config.json (1197 bytes) -- harness config from inside the nsys profile run
C_harness_grid_32_active/nsys_run/timing.csv (2781 bytes) -- 50 device-side timings from inside the nsys profile run (CUPTI overhead biased)
C_harness_grid_32_active/nsys_run/correctness_gate_split_order.json (191 bytes) -- gate result from inside the nsys profile run
C_harness_grid_32_active/nsys_run/correctness_vs_chained.json (131 bytes) -- diagnostic from inside the nsys profile run
C_harness_grid_32_active/nsys_run/correctness_vs_matmul.json (124 bytes) -- diagnostic from inside the nsys profile run

## Harness modifications for this audit

docs/research/2026-05-03-w-o-k-parallel-harness/run_harness.py
    Added --slice-ctas flag (default 8 for parity with prior sweep). The
    flag plumbs into make_w_o_microkernel(slice_ctas=...) and into the
    run-actual gather_ctas in the effective-bytes formula. Validated via
    args constraint slice_ctas >= wo_split. Cooperative=True remains
    hardwired in microkernel.py:463.

## Provenance

- Branch: evidence/wo-k-parallel-harness
- HEAD at run time: a4c2765607fc0e3c7334f85b7b317857cd37705f
- Image id: sha256:9c0f1d31c92c29488f66a2c136183950cea787035d735ff95dd6af193740f530
- Image tag: nvllm:gb10
- nsys: 2025.6.3.541 from /opt/nvidia/nsight-systems/2025.6.3 (host bind-mount)

## Reproduction

Each <cfg>/launch_args.json contains the exact harness flags used. The
docker invocation pattern matches docs/research/2026-05-03-w-o-k-parallel-
harness/run_sweep.sh except that --slice-ctas is supplied per-run. nsys
invocations bind-mount /opt/nvidia/nsight-systems/2025.6.3 to /opt/nsys
inside the container and run --privileged.

