"""Fixed E2E workload for Phase A + Phase B.2 nsys traces.

Keeps kernel call distribution identical across all config runs so that
per-kernel μs comparisons are apples-to-apples.

Usage:
    python trace_workload.py --base-url http://localhost:8000/v1 \
        --model default --warmup 50 --timed 100 --concurrent 4
"""
from __future__ import annotations

import argparse
import asyncio
import time

import httpx

# Fixed prompt — deterministic, long enough to give the model room to run a
# real decode phase but short enough that 100 timed requests finish in ~3-5 min.
FIXED_PROMPT = (
    "Q: Janet has 3 apples and buys 5 more. How many apples does she have now? "
    "Think step by step.\nA:"
)


async def send_one(client: httpx.AsyncClient, base_url: str, model: str,
                   max_tokens: int) -> dict:
    r = await client.post(
        f"{base_url}/completions",
        json={
            "model": model,
            "prompt": FIXED_PROMPT,
            "max_tokens": max_tokens,
            "temperature": 0.0,
            "seed": 42,
            "ignore_eos": True,
        },
        timeout=120,
    )
    r.raise_for_status()
    return r.json()


async def burst(base_url: str, model: str, n_requests: int, concurrent: int,
                max_tokens: int) -> float:
    start = time.monotonic()
    async with httpx.AsyncClient() as client:
        sem = asyncio.Semaphore(concurrent)

        async def one():
            async with sem:
                return await send_one(client, base_url, model, max_tokens)

        await asyncio.gather(*[one() for _ in range(n_requests)])
    return time.monotonic() - start


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://localhost:8000/v1")
    ap.add_argument("--model", default="default")
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--timed", type=int, default=100)
    ap.add_argument("--concurrent", type=int, default=4)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--profile-start", default=None,
                    help="URL to hit to start vLLM torch profiler before timed burst")
    ap.add_argument("--profile-stop", default=None,
                    help="URL to hit to stop vLLM torch profiler after timed burst")
    args = ap.parse_args()

    print(f"[warmup] {args.warmup} requests at concurrency {args.concurrent}")
    await burst(args.base_url, args.model, args.warmup, args.concurrent, args.max_tokens)

    if args.profile_start:
        async with httpx.AsyncClient() as c:
            r = await c.post(args.profile_start); r.raise_for_status()
        print(f"[profiler] started via {args.profile_start}")

    print(f"[timed]  {args.timed} requests at concurrency {args.concurrent}")
    elapsed = await burst(args.base_url, args.model, args.timed, args.concurrent,
                          args.max_tokens)
    print(f"[timed]  elapsed={elapsed:.2f}s, throughput={args.timed / elapsed:.2f} req/s")

    if args.profile_stop:
        # Fire-and-forget: don't wait on the response. Calling /stop_profile
        # on a running torch profiler can block on CUPTI buffer serialization,
        # and if we await it we either block the caller (delaying teardown)
        # or time out (causing set -e abort). With bounded active_iterations
        # the profiler auto-finalizes anyway; this call is belt-and-suspenders.
        # Caller is responsible for a generous post-workload sleep so CUPTI
        # actually finishes writing the trace to disk before docker stop.
        try:
            async with httpx.AsyncClient(timeout=5) as c:
                await c.post(args.profile_stop)
            print(f"[profiler] /stop_profile sent (fire-and-forget)")
        except Exception as e:
            print(f"[profiler] /stop_profile fire-and-forget: {type(e).__name__} — caller MUST sleep long enough for CUPTI flush")


if __name__ == "__main__":
    asyncio.run(main())
