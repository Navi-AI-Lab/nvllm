# SSM zero-on-realloc ablation: 4-arm comparison

- OUT_DIR: `/tmp/ssm_ablation_suite_v2`
- git_sha: `670724746c596f6c095970c4d50b82e6328423db`
- image: `nvllm:gb10-d2_7`
- N runs per arm: 5
- gsm8k_floor: 45

## Verdict table (run x correct/errors)

| Arm | SSM | KV | Run 1 | Run 2 | Run 3 | Run 4 | Run 5 | Gate |
|-----|-----|----|-------|-------|-------|-------|-------|------|
| both | 1 | 1 | 48/0err | 48/0err | 48/0err | 48/0err | 48/0err | True |
| neither | 0 | 0 | 48/0err | 48/0err | 48/0err | 48/0err | 48/0err | True |
| ssm_only | 1 | 0 | 48/0err | 48/0err | 48/0err | 48/0err | 48/0err | True |
| kv_only | 0 | 1 | 47/0err | 47/0err | 47/0err | 47/0err | 47/0err | True |

## Per-question table - Run 4 (collapse window)

Columns per arm: lat (wall_time_s), ct (completion_tokens), dtok/s (decode_tok_s), fr (finish_reason), ok (correct).

| Q | both:lat | both:ct | both:dtok/s | both:fr | both:ok | neither:lat | neither:ct | neither:dtok/s | neither:fr | neither:ok | ssm_only:lat | ssm_only:ct | ssm_only:dtok/s | ssm_only:fr | ssm_only:ok | kv_only:lat | kv_only:ct | kv_only:dtok/s | kv_only:fr | kv_only:ok |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 16.47 | 153 | 9.29 | stop | N | 16.43 | 153 | 9.31 | stop | N | 16.43 | 153 | 9.31 | stop | N | 16.68 | 155 | 9.29 | stop | N |
| 2 | 23.52 | 217 | 9.23 | stop | Y | 23.53 | 217 | 9.22 | stop | Y | 23.54 | 217 | 9.22 | stop | Y | 12.73 | 116 | 9.11 | stop | Y |
| 3 | 11.60 | 106 | 9.14 | stop | Y | 11.60 | 106 | 9.13 | stop | Y | 11.61 | 106 | 9.13 | stop | Y | 13.88 | 127 | 9.15 | stop | Y |
| 4 | 19.42 | 179 | 9.21 | stop | Y | 19.43 | 179 | 9.21 | stop | Y | 19.44 | 179 | 9.21 | stop | Y | 20.66 | 190 | 9.20 | stop | Y |
| 5 | 20.82 | 192 | 9.22 | stop | Y | 20.81 | 192 | 9.22 | stop | Y | 20.83 | 192 | 9.22 | stop | Y | 16.46 | 151 | 9.18 | stop | Y |
| 6 | 13.74 | 126 | 9.17 | stop | Y | 13.74 | 126 | 9.17 | stop | Y | 13.76 | 126 | 9.16 | stop | Y | 14.43 | 132 | 9.15 | stop | Y |
| 7 | 13.63 | 125 | 9.17 | stop | Y | 13.64 | 125 | 9.17 | stop | Y | 13.65 | 125 | 9.16 | stop | Y | 11.20 | 102 | 9.11 | stop | Y |
| 8 | 19.64 | 181 | 9.22 | stop | Y | 19.65 | 181 | 9.21 | stop | Y | 19.63 | 181 | 9.22 | stop | Y | 15.28 | 140 | 9.16 | stop | N |
| 9 | 10.10 | 92 | 9.11 | stop | Y | 10.10 | 92 | 9.11 | stop | Y | 10.10 | 92 | 9.11 | stop | Y | 10.12 | 92 | 9.09 | stop | Y |
| 10 | 13.21 | 121 | 9.16 | stop | Y | 13.21 | 121 | 9.16 | stop | Y | 13.20 | 121 | 9.16 | stop | Y | 10.88 | 99 | 9.10 | stop | Y |
| 11 | 19.55 | 180 | 9.21 | stop | Y | 19.56 | 180 | 9.21 | stop | Y | 19.55 | 180 | 9.21 | stop | Y | 45.53 | 421 | 9.25 | stop | Y |
| 12 | 21.24 | 196 | 9.23 | stop | Y | 21.25 | 196 | 9.22 | stop | Y | 21.24 | 196 | 9.23 | stop | Y | 22.49 | 207 | 9.21 | stop | Y |
| 13 | 16.96 | 156 | 9.20 | stop | Y | 16.97 | 156 | 9.19 | stop | Y | 16.96 | 156 | 9.20 | stop | Y | 18.29 | 168 | 9.18 | stop | Y |
| 14 | 15.02 | 138 | 9.19 | stop | Y | 15.03 | 138 | 9.18 | stop | Y | 15.02 | 138 | 9.19 | stop | Y | 14.74 | 135 | 9.16 | stop | Y |
| 15 | 14.49 | 133 | 9.18 | stop | Y | 14.50 | 133 | 9.17 | stop | Y | 14.48 | 133 | 9.18 | stop | Y | 16.16 | 148 | 9.16 | stop | Y |
| 16 | 11.28 | 103 | 9.13 | stop | Y | 11.29 | 103 | 9.13 | stop | Y | 11.27 | 103 | 9.14 | stop | Y | 10.98 | 100 | 9.10 | stop | Y |
| 17 | 11.38 | 104 | 9.14 | stop | Y | 11.39 | 104 | 9.13 | stop | Y | 11.38 | 104 | 9.14 | stop | Y | 14.21 | 130 | 9.15 | stop | Y |
| 18 | 13.75 | 126 | 9.17 | stop | Y | 13.75 | 126 | 9.17 | stop | Y | 13.74 | 126 | 9.17 | stop | Y | 13.03 | 119 | 9.13 | stop | Y |
| 19 | 11.17 | 102 | 9.13 | stop | Y | 11.17 | 102 | 9.13 | stop | Y | 11.17 | 102 | 9.13 | stop | Y | 10.65 | 97 | 9.10 | stop | Y |
| 20 | 9.98 | 91 | 9.11 | stop | Y | 9.99 | 91 | 9.11 | stop | Y | 10.00 | 91 | 9.10 | stop | Y | 9.38 | 85 | 9.07 | stop | Y |
| 21 | 13.41 | 123 | 9.17 | stop | Y | 13.42 | 123 | 9.17 | stop | Y | 13.41 | 123 | 9.17 | stop | Y | 13.34 | 122 | 9.14 | stop | Y |
| 22 | 26.52 | 245 | 9.24 | stop | Y | 26.52 | 245 | 9.24 | stop | Y | 26.51 | 245 | 9.24 | stop | Y | 26.26 | 242 | 9.22 | stop | Y |
| 23 | 9.67 | 88 | 9.10 | stop | Y | 9.67 | 88 | 9.10 | stop | Y | 9.67 | 88 | 9.10 | stop | Y | 13.24 | 121 | 9.14 | stop | Y |
| 24 | 15.45 | 142 | 9.19 | stop | Y | 15.46 | 142 | 9.19 | stop | Y | 15.44 | 142 | 9.19 | stop | Y | 15.49 | 142 | 9.17 | stop | Y |
| 25 | 35.08 | 325 | 9.26 | stop | Y | 35.09 | 325 | 9.26 | stop | Y | 35.08 | 325 | 9.27 | stop | Y | 25.08 | 231 | 9.21 | stop | Y |
| 26 | 16.85 | 155 | 9.20 | stop | Y | 16.86 | 155 | 9.19 | stop | Y | 16.85 | 155 | 9.20 | stop | Y | 20.34 | 187 | 9.19 | stop | Y |
| 27 | 14.28 | 131 | 9.17 | stop | Y | 14.27 | 131 | 9.18 | stop | Y | 14.28 | 131 | 9.18 | stop | Y | 9.59 | 87 | 9.07 | stop | Y |
| 28 | 24.69 | 228 | 9.24 | stop | Y | 24.69 | 228 | 9.23 | stop | Y | 24.68 | 228 | 9.24 | stop | Y | 27.98 | 258 | 9.22 | stop | Y |
| 29 | 16.75 | 154 | 9.20 | stop | Y | 16.76 | 154 | 9.19 | stop | Y | 16.75 | 154 | 9.19 | stop | Y | 17.01 | 156 | 9.17 | stop | Y |
| 30 | 11.38 | 104 | 9.14 | stop | Y | 11.37 | 104 | 9.15 | stop | Y | 11.37 | 104 | 9.14 | stop | Y | 10.33 | 94 | 9.10 | stop | Y |
| 31 | 19.22 | 177 | 9.21 | stop | Y | 19.22 | 177 | 9.21 | stop | Y | 19.22 | 177 | 9.21 | stop | Y | 20.03 | 184 | 9.19 | stop | Y |
| 32 | 17.93 | 165 | 9.20 | stop | Y | 17.93 | 165 | 9.20 | stop | Y | 17.93 | 165 | 9.20 | stop | Y | 18.95 | 174 | 9.18 | stop | Y |
| 33 | 14.49 | 133 | 9.18 | stop | Y | 14.50 | 133 | 9.17 | stop | Y | 14.49 | 133 | 9.18 | stop | Y | 18.19 | 167 | 9.18 | stop | Y |
| 34 | 25.86 | 239 | 9.24 | stop | Y | 25.87 | 239 | 9.24 | stop | Y | 25.85 | 239 | 9.25 | stop | Y | 24.64 | 227 | 9.21 | stop | Y |
| 35 | 34.01 | 315 | 9.26 | stop | Y | 34.03 | 315 | 9.26 | stop | Y | 34.02 | 315 | 9.26 | stop | Y | 20.66 | 190 | 9.19 | stop | Y |
| 36 | 14.74 | 135 | 9.16 | stop | Y | 14.72 | 135 | 9.17 | stop | Y | 14.72 | 135 | 9.17 | stop | Y | 16.15 | 148 | 9.16 | stop | Y |
| 37 | 21.80 | 201 | 9.22 | stop | Y | 21.79 | 201 | 9.22 | stop | Y | 21.79 | 201 | 9.22 | stop | Y | 21.95 | 202 | 9.20 | stop | Y |
| 38 | 21.02 | 194 | 9.23 | stop | Y | 21.03 | 194 | 9.22 | stop | Y | 21.03 | 194 | 9.22 | stop | Y | 21.30 | 196 | 9.20 | stop | Y |
| 39 | 16.74 | 154 | 9.20 | stop | Y | 16.75 | 154 | 9.20 | stop | Y | 16.74 | 154 | 9.20 | stop | Y | 17.21 | 158 | 9.18 | stop | Y |
| 40 | 20.17 | 186 | 9.22 | stop | Y | 20.18 | 186 | 9.21 | stop | Y | 20.18 | 186 | 9.22 | stop | Y | 12.38 | 113 | 9.13 | stop | Y |
| 41 | 25.83 | 239 | 9.25 | stop | Y | 25.86 | 239 | 9.24 | stop | Y | 25.88 | 239 | 9.23 | stop | Y | 30.79 | 284 | 9.22 | stop | Y |
| 42 | 17.81 | 164 | 9.21 | stop | Y | 17.82 | 164 | 9.20 | stop | Y | 17.83 | 164 | 9.20 | stop | Y | 17.00 | 156 | 9.17 | stop | Y |
| 43 | 11.60 | 106 | 9.14 | stop | Y | 11.61 | 106 | 9.13 | stop | Y | 11.60 | 106 | 9.13 | stop | Y | 16.25 | 149 | 9.17 | stop | Y |
| 44 | 9.03 | 82 | 9.09 | stop | Y | 9.03 | 82 | 9.08 | stop | Y | 9.02 | 82 | 9.09 | stop | Y | 12.81 | 117 | 9.13 | stop | Y |
| 45 | 55.13 | 512 | 9.29 | length | N | 55.18 | 512 | 9.28 | length | N | 55.15 | 512 | 9.28 | length | N | 55.28 | 512 | 9.26 | length | N |
| 46 | 21.28 | 198 | 9.30 | stop | Y | 21.26 | 198 | 9.31 | stop | Y | 21.26 | 198 | 9.31 | stop | Y | 25.19 | 234 | 9.29 | stop | Y |
| 47 | 20.08 | 185 | 9.21 | stop | Y | 20.08 | 185 | 9.21 | stop | Y | 20.07 | 185 | 9.22 | stop | Y | 20.12 | 185 | 9.19 | stop | Y |
| 48 | 26.50 | 245 | 9.24 | stop | Y | 26.52 | 245 | 9.24 | stop | Y | 26.50 | 245 | 9.24 | stop | Y | 29.47 | 272 | 9.23 | stop | Y |
| 49 | 13.73 | 126 | 9.18 | stop | Y | 13.74 | 126 | 9.17 | stop | Y | 13.74 | 126 | 9.17 | stop | Y | 13.13 | 120 | 9.14 | stop | Y |
| 50 | 22.21 | 205 | 9.23 | stop | Y | 22.22 | 205 | 9.22 | stop | Y | 22.22 | 205 | 9.22 | stop | Y | 14.75 | 135 | 9.15 | stop | Y |

## Aggregate per-arm steady-state stats (concat across runs)

| Arm | N | median dtok/s | p95 wall_s | mean completion_tokens | finish_reason hist |
|-----|---|---------------|------------|------------------------|--------------------|
| both | 250 | 9.20 | 34.01 | 169.5 | length=5, stop=245 |
| neither | 250 | 9.19 | 34.03 | 169.5 | length=5, stop=245 |
| ssm_only | 250 | 9.20 | 34.01 | 169.5 | length=5, stop=245 |
| kv_only | 250 | 9.17 | 30.74 | 169.5 | length=5, stop=245 |

## Friend's interpretation thresholds applied

- 'real pipeline win' iff median decode_tok_s >= 1.30x baseline ('neither') AND mean completion_tokens >= 0.85x baseline
- 'shortened generations' iff decode rate up but completion_tokens < 0.85x baseline

- **both**: no decode win vs baseline (decode 1.00x, compt 1.00x)
- **neither**: baseline
- **ssm_only**: no decode win vs baseline (decode 1.00x, compt 1.00x)
- **kv_only**: no decode win vs baseline (decode 1.00x, compt 1.00x)

## Drained KV invariant (per-run pre vs post)

Tolerance: |delta| <= 0.05 (5 pp) counts as drained.

| Arm | Run | KV pre | KV post | delta | drained |
|-----|-----|--------|---------|-------|---------|
| both | 1 | 0.0000 | 0.0027 | 0.0027 | Y |
| both | 2 | 0.0000 | 0.0027 | 0.0027 | Y |
| both | 3 | 0.0000 | 0.0027 | 0.0027 | Y |
| both | 4 | 0.0000 | 0.0027 | 0.0027 | Y |
| both | 5 | 0.0000 | 0.0027 | 0.0027 | Y |
| neither | 1 | 0.0000 | 0.0027 | 0.0027 | Y |
| neither | 2 | 0.0000 | 0.0027 | 0.0027 | Y |
| neither | 3 | 0.0000 | 0.0027 | 0.0027 | Y |
| neither | 4 | 0.0000 | 0.0027 | 0.0027 | Y |
| neither | 5 | 0.0000 | 0.0027 | 0.0027 | Y |
| ssm_only | 1 | 0.0000 | 0.0027 | 0.0027 | Y |
| ssm_only | 2 | 0.0000 | 0.0027 | 0.0027 | Y |
| ssm_only | 3 | 0.0000 | 0.0027 | 0.0027 | Y |
| ssm_only | 4 | 0.0000 | 0.0027 | 0.0027 | Y |
| ssm_only | 5 | 0.0000 | 0.0027 | 0.0027 | Y |
| kv_only | 1 | 0.0000 | 0.0027 | 0.0027 | Y |
| kv_only | 2 | 0.0000 | 0.0027 | 0.0027 | Y |
| kv_only | 3 | 0.0000 | 0.0027 | 0.0027 | Y |
| kv_only | 4 | 0.0000 | 0.0027 | 0.0027 | Y |
| kv_only | 5 | 0.0000 | 0.0027 | 0.0027 | Y |

