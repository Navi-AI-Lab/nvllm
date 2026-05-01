#!/usr/bin/env python
"""C2 gate (external arm 1) — replay-coherence under stale-state pressure.

Two checks, both at the OpenAI-compatible API level (token-stable, NOT
byte-identical at the tensor level — see Task 9 + spec §2.4 note):

  1. Same-prompt repeatability: identical prompt × N replays produces
     identical token sequences.
  2. Cross-prompt independence: prompt A response is unchanged whether
     prompt B was just served or not.

Per spec §3 / C2.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime
import urllib.request

API = "http://localhost:8000/v1/completions"
MODEL = "default"
N_REPLAYS = 8
TIMEOUT = 120

PROMPT_A = "Q: What is the capital of France?\nA:"
PROMPT_B = "Q: What is the boiling point of water in Celsius?\nA:"

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_EVIDENCE_BASE = REPO_ROOT / "docs/research/2026-04-29-full-graph-spike/evidence"


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="C2 replay-coherence external arm.")
    p.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="If set, also write the result JSON to this exact path "
             "(in addition to the evidence-dir copy).",
    )
    p.add_argument(
        "--evidence-dir",
        type=Path,
        default=None,
        help="Override the evidence directory. Default: "
             "<repo>/docs/research/2026-04-29-full-graph-spike/evidence/<timestamp>/",
    )
    return p.parse_args(argv)


def complete(prompt: str) -> str:
    body = json.dumps({
        "model": MODEL,
        "prompt": prompt,
        "max_tokens": 32,
        "temperature": 0,
        "top_p": 1,
        "seed": 42,
    }).encode()
    req = urllib.request.Request(
        API, data=body, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
        d = json.loads(resp.read())
    return d["choices"][0]["text"]


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    ts = datetime.now().strftime("%Y-%m-%d-%H%M")
    evidence = args.evidence_dir or (DEFAULT_EVIDENCE_BASE / ts)
    evidence.mkdir(parents=True, exist_ok=True)

    print(f"=== C2 replay-coherence (n={N_REPLAYS}) ===")
    print(f"Evidence dir: {evidence}")

    # Same-prompt repeatability.
    print("\n[1] Same-prompt repeatability")
    same_outputs: list[str] = []
    for i in range(N_REPLAYS):
        out = complete(PROMPT_A)
        same_outputs.append(out)
        print(f"  replay {i+1}: {out!r}")
    same_unique = set(same_outputs)
    same_pass = len(same_unique) == 1
    print(f"  unique outputs: {len(same_unique)}  pass={same_pass}")

    # Cross-prompt independence.
    print("\n[2] Cross-prompt independence")
    a_first = complete(PROMPT_A)
    _ = complete(PROMPT_B)
    a_after_b = complete(PROMPT_A)
    cross_pass = a_first == a_after_b
    print(f"  A first:   {a_first!r}")
    print(f"  A after B: {a_after_b!r}")
    print(f"  pass={cross_pass}")

    overall = same_pass and cross_pass
    summary = {
        "timestamp": ts,
        "n_replays": N_REPLAYS,
        "same_prompt_outputs": same_outputs,
        "same_prompt_unique_count": len(same_unique),
        "same_prompt_pass": same_pass,
        "cross_prompt_a_first": a_first,
        "cross_prompt_a_after_b": a_after_b,
        "cross_prompt_pass": cross_pass,
        "overall_pass": overall,
    }
    out_json = evidence / "c2_replay_coherence.json"
    out_json.write_text(json.dumps(summary, indent=2))
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(summary, indent=2))

    md = evidence / "c2_replay_coherence.md"
    md.write_text(
        f"# C2 — replay coherence (external) — "
        f"{'PASS' if overall else 'FAIL'}\n\n"
        f"- Timestamp: {ts}\n"
        f"- N replays: {N_REPLAYS}\n"
        f"- Same-prompt repeatable: {same_pass}\n"
        f"- Cross-prompt independent: {cross_pass}\n"
        f"- Full evidence: c2_replay_coherence.json\n"
    )
    print(f"\nOverall: {'PASS' if overall else 'FAIL'}")
    print(f"Summary: {md}")
    return 0 if overall else 1


if __name__ == "__main__":
    sys.exit(main())
