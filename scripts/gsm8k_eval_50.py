"""
GSM8K 50-question random eval against a vLLM server.

Reads cached HF gsm8k test parquet (1319 questions), samples N with a fixed
seed (default 50, seed=42 — reproducible), sends each to /v1/completions at
temperature=0, parses final numeric answer.

Per memory:feedback_eval_completions: /v1/completions, NOT /v1/chat/completions.

Usage:
    .venv/bin/python scripts/gsm8k_eval_50.py \\
        --api http://localhost:8000/v1 --model default \\
        --n 50 --save out.json --label some_run_name
"""

import argparse
import json
import re
import sys
import time

import pyarrow as pa
import pyarrow.ipc as ipc
import pyarrow.parquet as pq
import requests

GSM8K_TEST_ARROW = (
    "/home/natfii/.cache/huggingface/datasets/openai___gsm8k/main/"
    "0.0.0/740312add88f781978c0658806c59bc2815b9866/gsm8k-test.arrow"
)
GSM8K_TEST_BLOB_FALLBACK = (
    "/home/natfii/.cache/huggingface/hub/datasets--openai--gsm8k/blobs/"
    "ee7b8da9e381df27b9e3f7758a159ab2bdaa4dbaa910546cbbc47e0cb44e4f59"
)


def _load_test_split():
    import os
    if os.path.exists(GSM8K_TEST_ARROW):
        with pa.memory_map(GSM8K_TEST_ARROW, "rb") as src:
            return ipc.open_stream(src).read_all().to_pylist()
    return pq.read_table(GSM8K_TEST_BLOB_FALLBACK).to_pylist()


def extract_gold(answer: str) -> str:
    # GSM8K gold answers have the form "...explanation...\n#### 18"
    m = re.search(r"####\s*([-\d.,]+)", answer)
    return m.group(1).replace(",", "") if m else answer.strip()


def extract_predicted(text: str) -> str:
    # Prefer "#### N" if present
    m = re.search(r"####\s*([-\d.,]+)", text)
    if m:
        return m.group(1).replace(",", "")
    # Otherwise last bare number in the text
    nums = re.findall(r"-?\d+(?:[.,]\d+)?", text)
    return nums[-1].replace(",", "") if nums else ""


def normalize(s: str) -> str:
    try:
        return str(int(float(s)))
    except (ValueError, TypeError):
        return s.strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", default="http://localhost:8000/v1")
    ap.add_argument("--model", default="default")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--timeout", type=int, default=180)
    ap.add_argument("--label", default="gsm8k_50")
    ap.add_argument("--save", default=None)
    args = ap.parse_args()

    table = _load_test_split()

    import random
    rng = random.Random(args.seed)
    sample = rng.sample(table, args.n)

    results = []
    correct = 0
    errors = 0
    t0 = time.time()
    for i, row in enumerate(sample):
        q = row["question"]
        gold = normalize(extract_gold(row["answer"]))

        prompt = f"Q: {q}\nA: Let me solve this step by step.\n"
        body = {
            "model": args.model,
            "prompt": prompt,
            "max_tokens": args.max_tokens,
            "temperature": 0.0,
            "stop": ["\nQ:", "\nQuestion:"],
        }
        ts = time.time()
        try:
            r = requests.post(
                f"{args.api}/completions", json=body, timeout=args.timeout
            )
            r.raise_for_status()
            text = r.json()["choices"][0]["text"]
            pred = normalize(extract_predicted(text))
            ok = (pred == gold)
            status = "OK" if ok else "WRONG"
            if ok:
                correct += 1
        except Exception as e:
            text = f"ERROR: {e}"
            pred = ""
            status = "ERROR"
            errors += 1
        elapsed = time.time() - ts

        results.append({
            "i": i,
            "expected": gold,
            "got": pred,
            "status": status,
            "elapsed": round(elapsed, 1),
            "raw_tail": text[-200:] if isinstance(text, str) else "",
            "question": q[:80] + "..." if len(q) > 80 else q,
        })

        # progress on stderr
        sys.stderr.write(
            f"[{i + 1}/{args.n}] {status} (gold={gold} pred={pred}) {elapsed:.1f}s\n"
        )
        sys.stderr.flush()

    total_t = time.time() - t0
    out = {
        "label": args.label,
        "model": args.model,
        "api": args.api,
        "n": args.n,
        "seed": args.seed,
        "correct": correct,
        "errors": errors,
        "accuracy": f"{correct}/{args.n} ({100*correct/args.n:.1f}%)",
        "total_seconds": round(total_t, 1),
        "results": results,
    }
    if args.save:
        with open(args.save, "w") as f:
            json.dump(out, f, indent=2)
    print(json.dumps({k: v for k, v in out.items() if k != "results"}, indent=2))


if __name__ == "__main__":
    main()
