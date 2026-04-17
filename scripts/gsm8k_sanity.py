#!/usr/bin/env python3
"""GSM8K sanity gate — fast canary for model quality after quant/kernel changes.

NOT a quality benchmark. This is a quick "did we break something?" smoke test.
The guided prompts hand the model 90% of the work — it only needs to finish
the arithmetic. A failure here means something is fundamentally wrong (bad
quant, kernel producing NaN, broken dequant, etc.), not that the model is
underperforming. For real quality evaluation, use the full eval suite.

Uses guided chain-of-thought prompts with short max_tokens to avoid thinking
mode timeout issues. Each question includes partial work so the model only
needs to output the final number, not a full reasoning chain.

Usage:
    python3 scripts/gsm8k_sanity.py                    # defaults
    python3 scripts/gsm8k_sanity.py --api http://localhost:8000/v1
    python3 scripts/gsm8k_sanity.py --model default --timeout 120
    python3 scripts/gsm8k_sanity.py --json              # machine-readable output
    python3 scripts/gsm8k_sanity.py --save results.json # save results to file

Exit code 0 = PASS (>= 50% correct), 1 = FAIL.
"""
import argparse
import json
import re
import sys
import time

import requests

# 8 GSM8K questions with guided chain-of-thought prompts.
# Each prompt includes partial work so the model only needs to complete
# the final arithmetic step. This avoids long <think> chains that blow
# past timeout limits while still testing that the model produces
# correct numerical output end-to-end.
QUESTIONS = [
    {
        "id": "Q1",
        "prompt": (
            "Q: Natalia sold clips to 48 of her friends in April, and then "
            "she sold half as many clips in May. How many clips did Natalia "
            "sell altogether in April and May?\n"
            "A: April: 48, May: 48/2 = 24. Total = 48 + 24 ="
        ),
        "expected": 72,
    },
    {
        "id": "Q2",
        "prompt": (
            "Q: Weng earns $12 an hour for babysitting. Yesterday, she just "
            "did 50 minutes of babysitting. How much did she earn?\n"
            "A: 50 min = 50/60 hours. 12 * 50/60 ="
        ),
        "expected": 10,
    },
    {
        "id": "Q3",
        "prompt": (
            "Q: Betty is saving money for a new wallet which costs $100. "
            "Betty has only half of the money she needs. Her parents decided "
            "to give her $15 for that purpose, and her grandparents twice as "
            "much as her parents. How much more money does Betty need to buy "
            "the wallet?\n"
            "A: Betty has 100/2 = 50. Parents: 15. Grandparents: 30. "
            "Total received: 50+15+30 = 95. Needs: 100 - 95 ="
        ),
        "expected": 5,
    },
    {
        "id": "Q4",
        "prompt": (
            "Q: Julie is reading a 120-page book. Yesterday, she was able to "
            "read 12 pages and today, she read twice as many pages as "
            "yesterday. If she wants to read half of the remaining pages "
            "tomorrow, how many pages should she read?\n"
            "A: Yesterday: 12, Today: 24. Total read: 36. Remaining: "
            "120-36 = 84. Half of remaining: 84/2 ="
        ),
        "expected": 42,
    },
    {
        "id": "Q5",
        "prompt": (
            "Q: James writes a 3-page letter to 2 different friends twice a "
            "week. How many pages does he write a year?\n"
            "A: Per week: 3 * 2 * 2 = 12. Per year: 12 * 52 ="
        ),
        "expected": 624,
    },
    {
        "id": "Q6",
        "prompt": (
            "Q: Mark has a garden with flowers. He planted plants of three "
            "different colors in it. Ten of them are yellow, and there are "
            "80% more of those in purple. There are only 25% as many green "
            "flowers as there are yellow and purple flowers. How many flowers "
            "does Mark have in his garden?\n"
            "A: Yellow: 10. Purple: 10 * 1.8 = 18. Green: (10+18) * 0.25 "
            "= 7. Total: 10+18+7 ="
        ),
        "expected": 35,
    },
    {
        "id": "Q7",
        "prompt": (
            "Q: Albert is wondering how much pizza he can eat in one day. "
            "He buys 2 large pizzas and 2 small pizzas. A large pizza has "
            "16 slices and a small pizza has 8 slices. If he eats it all, "
            "how many pieces does he eat that day?\n"
            "A: Large: 2*16 = 32. Small: 2*8 = 16. Total: 32+16 ="
        ),
        "expected": 48,
    },
    {
        "id": "Q8",
        "prompt": (
            "Q: Ken placed a box on a scale, then poured jelly beans to "
            "bring weight to 2 pounds. Then added brownies to triple the "
            "weight. Next added 2 more pounds of jelly beans. Finally added "
            "gummy worms to double the weight. What was the final weight?\n"
            "A: Start: 2. Triple: 6. Add 2: 8. Double: 8*2 ="
        ),
        "expected": 16,
    },
]


def run_sanity(api_base: str, model: str, timeout: int, max_tokens: int):
    """Run all GSM8K questions and return results."""
    url = f"{api_base}/completions"
    results = []

    for q in QUESTIONS:
        t0 = time.time()
        try:
            r = requests.post(
                url,
                json={
                    "model": model,
                    "prompt": q["prompt"],
                    "max_tokens": max_tokens,
                    "temperature": 0,
                },
                timeout=timeout,
            )
            r.raise_for_status()
            text = r.json()["choices"][0]["text"]
            # Strip any thinking tags
            clean = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
            nums = re.findall(r"[-+]?\d*\.?\d+", clean)
            got = int(float(nums[0])) if nums else None
            ok = got == q["expected"]
            results.append({
                "id": q["id"],
                "expected": q["expected"],
                "got": got,
                "raw": text.strip(),
                "status": "OK" if ok else "WRONG",
                "elapsed": round(time.time() - t0, 1),
            })
        except Exception as e:
            results.append({
                "id": q["id"],
                "expected": q["expected"],
                "got": None,
                "status": "ERROR",
                "error": str(e),
                "elapsed": round(time.time() - t0, 1),
            })

    return results


def main():
    p = argparse.ArgumentParser(description="GSM8K sanity gate")
    p.add_argument("--api", default="http://localhost:8000/v1",
                    help="vLLM API base URL (default: http://localhost:8000/v1)")
    p.add_argument("--model", default="default",
                    help="Model name (default: default)")
    p.add_argument("--timeout", type=int, default=120,
                    help="Per-question HTTP timeout in seconds (default: 120)")
    p.add_argument("--max-tokens", type=int, default=16,
                    help="Max tokens per completion (default: 16)")
    p.add_argument("--json", action="store_true",
                    help="Output machine-readable JSON")
    p.add_argument("--save", metavar="FILE",
                    help="Save results to JSON file")
    p.add_argument("--label", default="",
                    help="Label for this run (e.g. 'phase_c_rmsnorm')")
    args = p.parse_args()

    results = run_sanity(args.api, args.model, args.timeout, args.max_tokens)

    correct = sum(1 for r in results if r["status"] == "OK")
    total = len(results)
    errors = sum(1 for r in results if r["status"] == "ERROR")
    passed = correct >= total // 2

    summary = {
        "test": "gsm8k_sanity_gate",
        "label": args.label or None,
        "model": args.model,
        "api": args.api,
        "correct": correct,
        "total": total,
        "errors": errors,
        "accuracy": f"{correct}/{total} ({100 * correct // total}%)",
        "verdict": "PASS" if passed else "FAIL",
        "questions": results,
    }

    if args.json:
        print(json.dumps(summary, indent=2))
    else:
        for r in results:
            if r["status"] == "ERROR":
                print(f"  {r['id']}: ERROR  {r.get('error', '?')}")
            else:
                print(f"  {r['id']}: {r['status']}  "
                      f"expected={r['expected']}  got={r['got']}  "
                      f"({r['elapsed']}s)")
        print()
        print(f"GSM8K sanity: {correct}/{total} "
              f"({100 * correct // total}%)")
        if errors:
            print(f"  ({errors} errors — check timeout/connectivity)")
        print(f"Verdict: {'PASS' if passed else 'FAIL'}")

    if args.save:
        with open(args.save, "w") as f:
            json.dump(summary, f, indent=2)
        if not args.json:
            print(f"Saved to {args.save}")

    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
