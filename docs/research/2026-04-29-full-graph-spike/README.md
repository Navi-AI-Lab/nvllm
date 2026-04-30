# FULL_AND_PIECEWISE + CuTe (β-coop) — n=1 Spike Evidence

Companion to spec `docs/superpowers/specs/2026-04-29-full-and-piecewise-cute-spike-design.md` and plan `docs/superpowers/plans/2026-04-29-full-and-piecewise-cute-spike.md`.

## Gate sequence

C0 → C1 → C2 → C3. Each gate must pass before the next is meaningful.

| Gate | Script | What it proves | What failure means |
|---|---|---|---|
| C0 | `c0_piecewise_baseline.sh` | PIECEWISE + β-coop + `CUTE_PHASE_E_FALLBACK_RAISE=1` passes 8/8 GSM8K sanity | The flag isn't inert; β-coop relies on fallback in normal operation. Stop. |
| C1 | `c1_replay_proof.sh` | FULL_AND_PIECEWISE + n=1 + flag: live V1 `gpu_model_runner.py` dispatch reports `cg_mode=FULL` | β-coop didn't capture, or runner downgraded the mode. Audit stream wiring + dispatch logs. |
| C2 | `c2_replay_coherence.py` + `c2_single_token_determinism.py` | Two external-arm determinism checks: same-prompt N-replay token-stable; cross-prompt independent; max_tokens=1 deterministic. Tensor-level byte-equality is NOT verified by these — see spec §2.4. | Stale-state contamination evidence; escalate to in-engine instrumentation, identify failing tensor, then pivot to β (persistent buffers). |
| C3 | `c3_gsm8k_parity.sh` | FULL token-level matches PIECEWISE on the same checkpoint/seed, accuracy ≥ PIECEWISE | C2 missed something. Bisect by tensor. |

## Evidence layout

Each run writes to a timestamped subdirectory:

```
docs/research/2026-04-29-full-graph-spike/
  README.md                       (this file)
  c0_piecewise_baseline.sh
  c1_replay_proof.sh
  c2_replay_coherence.py
  c2_single_token_determinism.py
  c3_gsm8k_parity.sh
  evidence/
    YYYY-MM-DD-HHMM/
      c0_gsm8k_sanity.json
      c0_docker_logs.txt
      c1_probe.log
      c1_passes_full.txt
      c2_replay_coherence.txt
      c2_single_token_determinism.txt
      c3_gsm8k_full.json
      c3_gsm8k_piecewise.json
      c3_diff.txt
      summary.md
```

## How to run

Run gates in order from the repo root:

```bash
./docs/research/2026-04-29-full-graph-spike/c0_piecewise_baseline.sh
./docs/research/2026-04-29-full-graph-spike/c1_replay_proof.sh
./docs/research/2026-04-29-full-graph-spike/c2_replay_coherence.py
./docs/research/2026-04-29-full-graph-spike/c2_single_token_determinism.py
./docs/research/2026-04-29-full-graph-spike/c3_gsm8k_parity.sh
```

Each script is idempotent: it stops or restarts the appropriate serve, runs its gate, writes evidence to a fresh `evidence/<timestamp>/` directory, and returns exit code 0 on pass / non-zero on fail.

## What "pass" looks like

- C0 PASS: `gsm8k_sanity.py` reports 8/8.
- C1 PASS: probe log contains `cg_mode=FULL` for at least one decode call in `vllm/v1/worker/gpu_model_runner.py` (not just config).
- C2 PASS: replay-coherence harness reports `n_replays=N, divergent=0` AND single-token-determinism harness reports `text_match=true`. **Caveat:** these are two external-arm checks; they do not prove tensor-level byte-equality of β-coop's output. The tensor-level path requires in-engine instrumentation (spec §2.4) and is run only if these external arms fail or surface ambiguity.
- C3 PASS: zero answer-level divergence (per-question `got` + `status` match) across 50 questions; FULL accuracy ≥ PIECEWISE accuracy. Token-level full-completion diff is not in scope this branch.

A non-passing gate writes a summary explaining what failed and where (which file/tensor/replay).
