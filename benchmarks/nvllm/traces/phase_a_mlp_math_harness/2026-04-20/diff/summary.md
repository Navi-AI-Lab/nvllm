# Phase A vs D2e MLP math harness — summary

- d2e dir:    benchmarks/nvllm/traces/phase_a_mlp_math_harness/2026-04-20/d2e
- phaseA dir: benchmarks/nvllm/traces/phase_a_mlp_math_harness/2026-04-20/phaseA
## Headline: ALL MATCH


## Per-case comparison

### zero_nat1

- D2e:    `shape=(1, 5120) dtype=torch.bfloat16 nan=0/5120 absmax_finite=0 md5=1276481102f218c981e0324180bafd9f`
- PhaseA: `shape=(1, 5120) dtype=torch.bfloat16 nan=0/5120 absmax_finite=0 md5=1276481102f218c981e0324180bafd9f`
- **MATCH (md5 identical)**

### seed_nat1

- D2e:    `shape=(1, 5120) dtype=torch.bfloat16 nan=5120/5120 absmax_finite=0 md5=d6d9ed37b4f4a3954c9f88ebaea74100`
- PhaseA: `shape=(1, 5120) dtype=torch.bfloat16 nan=5120/5120 absmax_finite=0 md5=d6d9ed37b4f4a3954c9f88ebaea74100`
- **MATCH (md5 identical)**

### seed_nat8

- D2e:    `shape=(8, 5120) dtype=torch.bfloat16 nan=40960/40960 absmax_finite=0 md5=507ed14f2a11de96296b3e44e990f925`
- PhaseA: `shape=(8, 5120) dtype=torch.bfloat16 nan=40960/40960 absmax_finite=0 md5=507ed14f2a11de96296b3e44e990f925`
- **MATCH (md5 identical)**

### seed_nat1_repeat

- D2e:    `shape=(1, 5120) dtype=torch.bfloat16 nan=5120/5120 absmax_finite=0 md5=d6d9ed37b4f4a3954c9f88ebaea74100`
- PhaseA: `shape=(1, 5120) dtype=torch.bfloat16 nan=5120/5120 absmax_finite=0 md5=d6d9ed37b4f4a3954c9f88ebaea74100`
- **MATCH (md5 identical)**
