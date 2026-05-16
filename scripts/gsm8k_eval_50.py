"""
GSM8K 50-question random eval against a vLLM server.

Reads cached HF gsm8k test parquet (1319 questions), samples N with a fixed
seed (default 50, seed=42 - reproducible), sends each to /v1/completions at
temperature=0, parses final numeric answer.

Per memory:feedback_eval_completions: /v1/completions, NOT /v1/chat/completions.

Instrumented form (2026-05-15) for the SSM zero-on-realloc ablation suite:
  - per-question JSONL trace at <output-dir>/perq.jsonl (one record per Q)
  - --run-index flag, stamped into every per-Q record
  - usage tokens (prompt/completion/total) + decode_tok/s + finish_reason
  - output sha256 (16 hex), request id, character count
  - --metrics-url flag: snapshots vllm:* prometheus metrics at pre/q10/q20/
    q30/q40/q50/post tags, saved to <output-dir>/metrics_<tag>.json

Timing semantics PRESERVED: wall_time_s is the time from the
requests.post() start to response received, EXCLUDING metrics-snapshot time.

Usage:
    .venv/bin/python scripts/gsm8k_eval_50.py \\
        --api http://localhost:8000/v1 --model default \\
        --n 50 --save out.json --label some_run_name \\
        --run-index 1 \\
        --metrics-url http://localhost:8000/metrics
"""

import argparse
import hashlib
import json
import os
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

# Subset of /metrics lines we extract into the per-snapshot JSON.
METRICS_KEYS = (
    "vllm:num_requests_running",
    "vllm:num_requests_waiting",
    "vllm:kv_cache_usage_perc",
    "vllm:generation_tokens_total",
    "vllm:num_preemptions_total",
)


def _load_test_split():
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


def _snapshot_metrics(metrics_url, tag, perq_dir):
    """Fetch /metrics, extract METRICS_KEYS, write metrics_<tag>.json.

    Best-effort: failures never abort the eval; returns a dict on success
    or None on failure. Called outside the wall_time_s timer.
    """
    if not metrics_url or not perq_dir:
        return None
    try:
        r = requests.get(metrics_url, timeout=10)
        r.raise_for_status()
        body = r.text
    except Exception as e:
        snap = {"tag": tag, "ts": time.time(), "error": repr(e)}
        try:
            with open(os.path.join(perq_dir, f"metrics_{tag}.json"), "w") as f:
                json.dump(snap, f, indent=2)
        except Exception:
            pass
        return snap

    # Prometheus text format: each line starts with the metric name
    # (possibly with {labels}) and a value. We extract the LAST numeric
    # value seen for each desired key (sum across labels for gauges or
    # final total for counters; both behaviors are acceptable here since
    # we mostly care about deltas between snapshots).
    values: dict = {}
    for line in body.splitlines():
        if not line or line.startswith("#"):
            continue
        # Match "metric_name" or "metric_name{labels}"
        for key in METRICS_KEYS:
            if line.startswith(key + " ") or line.startswith(key + "{"):
                # split on whitespace from the right: "<name>{...} <value>"
                parts = line.rsplit(None, 1)
                if len(parts) != 2:
                    continue
                try:
                    v = float(parts[1])
                except ValueError:
                    continue
                # Sum across label sets (gauges total across engines; counters
                # already monotonic, so summing engine_idx labels is correct).
                values[key] = values.get(key, 0.0) + v
                break
    snap = {"tag": tag, "ts": time.time(), "metrics": values}
    try:
        with open(os.path.join(perq_dir, f"metrics_{tag}.json"), "w") as f:
            json.dump(snap, f, indent=2)
    except Exception:
        pass
    return snap


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
    # Instrumented additions:
    ap.add_argument(
        "--run-index", type=int, default=0,
        help="Soak run index, stamped into every per-Q JSONL record",
    )
    ap.add_argument(
        "--metrics-url", default=None,
        help="If set, snapshot /metrics pre / q10..q50 / post into "
             "<output-dir>/metrics_<tag>.json (timing NOT charged to wall).",
    )
    args = ap.parse_args()

    # perq_dir = directory holding gsm8k.json (i.e. <output-dir>)
    perq_dir = None
    perq_fh = None
    if args.save:
        perq_dir = os.path.dirname(os.path.abspath(args.save)) or "."
        try:
            os.makedirs(perq_dir, exist_ok=True)
            perq_fh = open(os.path.join(perq_dir, "perq.jsonl"), "a", buffering=1)
        except Exception as e:
            sys.stderr.write(f"WARN: cannot open perq.jsonl: {e}\n")
            perq_fh = None

    table = _load_test_split()

    import random
    rng = random.Random(args.seed)
    sample = rng.sample(table, args.n)

    # Pre-flight metrics snapshot (NOT charged to any question's wall time).
    _snapshot_metrics(args.metrics_url, "pre", perq_dir)

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

        # Per-question instrumentation defaults (filled in on success).
        usage = {}
        finish_reason = None
        request_id = None
        text = ""

        ts = time.time()
        try:
            r = requests.post(
                f"{args.api}/completions", json=body, timeout=args.timeout
            )
            r.raise_for_status()
            payload = r.json()
            choice0 = payload.get("choices", [{}])[0]
            text = choice0.get("text", "")
            finish_reason = choice0.get("finish_reason")
            usage = payload.get("usage", {}) or {}
            request_id = payload.get("id")
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
            ok = False
        wall_time_s = time.time() - ts

        # Per-Q instrumentation record.
        prompt_tokens = usage.get("prompt_tokens", 0) if isinstance(usage, dict) else 0
        completion_tokens = (
            usage.get("completion_tokens", 0) if isinstance(usage, dict) else 0
        )
        total_tokens = usage.get("total_tokens", 0) if isinstance(usage, dict) else 0
        decode_tok_s = (
            (completion_tokens / wall_time_s)
            if (completion_tokens and wall_time_s > 0) else 0.0
        )
        output_len = len(text) if isinstance(text, str) else 0
        try:
            output_sha256 = hashlib.sha256(
                (text if isinstance(text, str) else "").encode("utf-8", "replace")
            ).hexdigest()[:16]
        except Exception:
            output_sha256 = ""

        perq_rec = {
            "label": args.label,
            "run_index": args.run_index,
            "prompt_index": i + 1,  # 1-based, matches "[N/50]" log format
            "wall_time_s": round(wall_time_s, 4),
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "decode_tok_s": round(decode_tok_s, 3),
            "finish_reason": finish_reason,
            "gold": gold,
            "pred": pred,
            "correct": bool(ok),
            "output_len": output_len,
            "output_sha256": output_sha256,
            "request_id": request_id,
            "ts": ts,
        }
        if perq_fh is not None:
            try:
                perq_fh.write(json.dumps(perq_rec, separators=(",", ":")) + "\n")
            except Exception:
                pass

        # Aggregate JSON results (same shape as before; do not break callers).
        results.append({
            "i": i,
            "expected": gold,
            "got": pred,
            "status": status,
            "elapsed": round(wall_time_s, 1),
            "raw_tail": text[-200:] if isinstance(text, str) else "",
            "question": q[:80] + "..." if len(q) > 80 else q,
            # New non-breaking fields (additive, do not alter existing keys):
            "wall_time_s": round(wall_time_s, 4),
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "decode_tok_s": round(decode_tok_s, 3),
            "finish_reason": finish_reason,
            "output_len": output_len,
            "output_sha256": output_sha256,
            "request_id": request_id,
        })

        # progress on stderr (preserve existing format for log greppability)
        sys.stderr.write(
            f"[{i + 1}/{args.n}] {status} (gold={gold} pred={pred}) "
            f"{wall_time_s:.1f}s ct={completion_tokens} dtok/s={decode_tok_s:.2f} "
            f"fr={finish_reason}\n"
        )
        sys.stderr.flush()

        # Mid-eval metrics snapshots (after Q10/20/30/40/50). Done AFTER
        # wall_time_s is recorded, so snapshot cost is never charged to a
        # question's decode latency.
        if (i + 1) in (10, 20, 30, 40, 50):
            _snapshot_metrics(args.metrics_url, f"q{i + 1}", perq_dir)

    total_t = time.time() - t0

    # Post-eval snapshot (NOT charged to wall).
    _snapshot_metrics(args.metrics_url, "post", perq_dir)

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
        # Additive aggregate fields (won't break existing parsers).
        "run_index": args.run_index,
    }
    if args.save:
        with open(args.save, "w") as f:
            json.dump(out, f, indent=2)
    if perq_fh is not None:
        try:
            perq_fh.close()
        except Exception:
            pass
    print(json.dumps({k: v for k, v in out.items() if k != "results"}, indent=2))


if __name__ == "__main__":
    main()
