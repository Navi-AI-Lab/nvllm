# Arm: 8L_3_31

- description: β-coop on first 8 full-attn layers
- git_sha: 9f118cdc571360b83cc3922ca9c72ce04b66c0c5
- image_id: sha256:9c0f1d31c92c29488f66a2c136183950cea787035d735ff95dd6af193740f530
- worktree: /home/natfii/docker/nvllm-beta-layer-sweep-wo8
- arms.csv row: arm=8L_3_31, fusion=1, phase_e_layers=[3,7,11,15,19,23,27,31],
  wo_split=8, expected_coop_layers=[3,7,11,15,19,23,27,31]

## Dispatch audit
- result: PASS (coop_layers matched expected=[3,7,11,15,19,23,27,31])
- artifact: [dispatch_audit.json](dispatch_audit.json)

## GSM8K-50
- correct: 49 / 50
- errors: 0
- floor: 45
- pass: true
- artifact: [gsm8k.json](gsm8k.json), [gsm8k.log](gsm8k.log)

## β kernel per-call timing (advisory)
- per-call median (sum-of-region-medians proxy): n/a ms
- gate (≤7 ms): n/a
- artifact: [region_timings.npy](region_timings.npy)

## Server provenance
- [c2_diag_ENV.txt](c2_diag_ENV.txt) — sentinel-file env snapshot
- [docker_inspect.json](docker_inspect.json) — container Cmd + Env
- [serve_log_head.txt](serve_log_head.txt) — first 200 log lines
- [serve.log](serve.log), [docker.log](docker.log)
