"""5-turn sequential /v1/completions smoke against a running nvllm server.

Sends 5 back-to-back completion requests with cumulative context (turn N+1's
prompt includes turn N's prompt + response), validates each turn returns a
non-empty coherent string. Catches SSM-state / KV-residual leak between
requests — the failure mode flagged by memory:feedback_post_quant_sanity and
memory:project_model_degradation.

Per memory:feedback_eval_completions: /v1/completions, NOT /v1/chat/completions.
Chat triggers thinking mode and breaks extraction; completions stays predictable.

Usage:
    .venv/bin/python scripts/multi_turn_smoke.py \\
        --api http://localhost:8000/v1 --model default --save smoke.json
"""

import argparse
import json
import sys
import time

import requests


TURNS = [
    "Q: A train leaves Boston at 9:00 AM traveling 60 mph east. "
    "Another train leaves New York City at 10:00 AM traveling 70 mph west. "
    "Boston and NYC are 200 miles apart. At what time (Eastern) do they meet?\n\nA:",
    " Now suppose the eastbound train accidentally stops for 30 minutes at "
    "10:30 AM. How does the meeting time change?\n\nA:",
    " Now ignore that stop. Suppose instead the westbound train's actual "
    "speed is 80 mph. At what time do they meet under this revised scenario?\n\nA:",
    " For that revised scenario, how far from Boston (in miles) is the "
    "meeting point?\n\nA:",
    " Finally, summarize all four answers above as a short bulleted list, "
    "one bullet per question.\n\nA:",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", default="http://localhost:8000/v1")
    ap.add_argument("--model", default="default")
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--timeout", type=int, default=120)
    ap.add_argument("--save", default=None)
    ap.add_argument(
        "--temperature", type=float, default=0.0,
        help="0 for determinism; bump for variety smoke",
    )
    args = ap.parse_args()

    transcript = ""
    turns_out = []
    pass_count = 0
    total_wall = 0.0

    for turn_idx, segment in enumerate(TURNS, start=1):
        prompt = transcript + segment
        body = {
            "model": args.model,
            "prompt": prompt,
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "stop": ["\nQ:"],
        }
        t0 = time.time()
        try:
            r = requests.post(
                f"{args.api}/completions",
                json=body,
                timeout=args.timeout,
            )
            r.raise_for_status()
            wall = time.time() - t0
            data = r.json()
            text = data["choices"][0]["text"]
            finish = data["choices"][0].get("finish_reason", "")
            usage = data.get("usage", {})
        except Exception as e:
            wall = time.time() - t0
            text = ""
            finish = f"error: {type(e).__name__}: {e}"
            usage = {}

        total_wall += wall
        ok = bool(text.strip()) and "error:" not in finish
        if ok:
            pass_count += 1
        turns_out.append({
            "turn": turn_idx,
            "wall_s": round(wall, 3),
            "finish_reason": finish,
            "completion_tokens": usage.get("completion_tokens"),
            "text_preview": text.strip()[:200],
            "ok": ok,
        })
        # Build up the conversation for the next turn.
        transcript = prompt + text

        status = "PASS" if ok else "FAIL"
        print(
            f"[turn {turn_idx}/5] {status} wall={wall:.2f}s "
            f"tok={usage.get('completion_tokens')} "
            f"finish={finish}",
            flush=True,
        )
        print(f"    preview: {text.strip()[:160]!r}", flush=True)

    summary = {
        "pass": pass_count,
        "total": len(TURNS),
        "wall_total_s": round(total_wall, 3),
        "turns": turns_out,
    }
    print(
        f"\n=== multi-turn smoke: {pass_count}/{len(TURNS)} turns passed "
        f"(wall={total_wall:.2f}s) ===",
        flush=True,
    )

    if args.save:
        with open(args.save, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"saved: {args.save}", flush=True)

    # Exit non-zero if any turn returned empty/errored.
    sys.exit(0 if pass_count == len(TURNS) else 1)


if __name__ == "__main__":
    main()
