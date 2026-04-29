#!/usr/bin/env python
"""C2 (low-level external arm) — single-token determinism across two
replays of the same captured CUDA graph.

vLLM V1 spawns EngineCore in a separate process, so an in-process
tensor-capture harness would require a temporary monkeypatch inside
the impl (spec §2.4). The external arm here issues two identical
single-token /v1/completions requests; byte-different framework-output
tensor would imply different argmax → different text → fails this check.

Per spec §3 / C2.
"""

from __future__ import annotations

import json
import sys
import urllib.request
from pathlib import Path
from datetime import datetime

API = "http://localhost:8000/v1/completions"
MODEL = "default"
PROMPT = "Q: What is the capital of France?\nA:"
TIMEOUT = 120

REPO_ROOT = Path(__file__).resolve().parents[3]
TS = datetime.now().strftime("%Y-%m-%d-%H%M")
EVIDENCE = REPO_ROOT / "docs/research/2026-04-29-full-graph-spike/evidence" / TS
EVIDENCE.mkdir(parents=True, exist_ok=True)


def complete_one_token() -> dict:
    body = json.dumps({
        "model": MODEL,
        "prompt": PROMPT,
        "max_tokens": 1,
        "temperature": 0,
        "top_p": 1,
        "seed": 42,
    }).encode()
    req = urllib.request.Request(
        API, data=body, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
        return json.loads(resp.read())


def main() -> int:
    print("=== C2 single-token determinism (external arm 2) ===")
    print(f"Evidence dir: {EVIDENCE}")

    r1 = complete_one_token()
    r2 = complete_one_token()

    t1 = r1["choices"][0]["text"]
    t2 = r2["choices"][0]["text"]
    lp1 = r1["choices"][0].get("logprobs")
    lp2 = r2["choices"][0].get("logprobs")

    text_match = t1 == t2
    lp_match = lp1 == lp2
    overall = text_match and (lp1 is None or lp_match)

    summary = {
        "timestamp": TS,
        "text_replay_1": t1,
        "text_replay_2": t2,
        "text_match": text_match,
        "logprobs_replay_1": lp1,
        "logprobs_replay_2": lp2,
        "logprobs_match": lp_match,
        "overall_pass": overall,
        "note": (
            "External arm 2 — single-token determinism. NOT a byte-equality "
            "check on β-coop's output tensor. Internal stale state that does "
            "not change argmax can still pass this. Tensor-level byte gate "
            "is documented in spec §2.4 and deferred."
        ),
    }
    out = EVIDENCE / "c2_single_token_determinism.json"
    out.write_text(json.dumps(summary, indent=2))

    md = EVIDENCE / "c2_single_token_determinism.md"
    md.write_text(
        f"# C2 — single-token determinism (external arm 2) — "
        f"{'PASS' if overall else 'FAIL'}\n\n"
        f"- Timestamp: {TS}\n"
        f"- Text match: {text_match}\n"
        f"- Logprobs match: {lp_match} (None if logprobs not requested)\n"
        f"- Note: NOT byte-equality of β-coop output tensor. Tensor-level "
        f"harness requires in-engine instrumentation (spec §2.4).\n"
    )
    print(f"\nOverall: {'PASS' if overall else 'FAIL'}")
    print(f"Summary: {md}")
    return 0 if overall else 1


if __name__ == "__main__":
    sys.exit(main())
