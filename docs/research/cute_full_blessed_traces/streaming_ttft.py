"""Single streaming completion request for TTFT + decode tok/s.

Companion to trace_workload.py: while trace_workload.py drives the profiled
window for per-kernel μs comparison, this script captures user-felt
wall-clock metrics on a single steady-state request.

Output: JSON to stdout with TTFT, decode tok/s, total latency, n_tokens.

Usage:
    python streaming_ttft.py --base-url http://localhost:8000/v1 \
        --model default --max-tokens 256
"""
from __future__ import annotations

import argparse
import json
import sys
import time

import httpx


FIXED_PROMPT = (
    "Q: Janet has 3 apples and buys 5 more. How many apples does she have now? "
    "Think step by step.\nA:"
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://localhost:8000/v1")
    ap.add_argument("--model", default="default")
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--timeout", type=float, default=600.0)
    ap.add_argument("--out", required=True, help="JSON output path")
    ap.add_argument("--label", default="leg")
    args = ap.parse_args()

    body = {
        "model": args.model,
        "prompt": FIXED_PROMPT,
        "max_tokens": args.max_tokens,
        "temperature": 0.0,
        "seed": args.seed,
        "ignore_eos": True,
        "stream": True,
        "stream_options": {"include_usage": True},
    }

    ttft_s: float | None = None
    last_chunk_s: float | None = None
    n_tokens = 0
    completion_tokens: int | None = None

    t_start = time.monotonic()
    with httpx.Client(timeout=args.timeout) as client:
        with client.stream(
            "POST", f"{args.base_url}/completions", json=body
        ) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if not line or not line.startswith("data: "):
                    continue
                payload = line[len("data: "):].strip()
                if payload == "[DONE]":
                    break
                try:
                    obj = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                now = time.monotonic()
                choices = obj.get("choices") or []
                if choices and choices[0].get("text"):
                    if ttft_s is None:
                        ttft_s = now - t_start
                    last_chunk_s = now - t_start
                    n_tokens += 1
                usage = obj.get("usage")
                if usage and usage.get("completion_tokens") is not None:
                    completion_tokens = usage["completion_tokens"]
    t_end = time.monotonic()

    total_s = t_end - t_start
    decode_tokens = (completion_tokens or n_tokens) - 1
    decode_window_s = (last_chunk_s or total_s) - (ttft_s or 0.0)
    decode_tok_per_s = decode_tokens / decode_window_s if decode_window_s > 0 else 0.0

    result = {
        "label": args.label,
        "ttft_s": ttft_s,
        "total_latency_s": total_s,
        "completion_tokens": completion_tokens,
        "chunks_with_text": n_tokens,
        "decode_window_s": decode_window_s,
        "decode_tok_per_s": decode_tok_per_s,
        "max_tokens_requested": args.max_tokens,
    }
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    json.dump(result, sys.stdout, indent=2)
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
