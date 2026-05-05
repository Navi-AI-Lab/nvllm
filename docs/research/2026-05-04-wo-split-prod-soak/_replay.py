#!/usr/bin/env python3
"""Replay deterministic serving workloads for the wo_split production soak.

The runner owns container lifecycle. This helper owns OpenAI-compatible
completion requests, streaming timing capture, and output serialization.

Outputs are intentionally simple JSON/CSV so parse_results.py can aggregate
without importing vLLM or pandas.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import requests


DEFAULT_API = "http://localhost:8000/v1"
DEFAULT_MODEL = "default"


def percentile(data: list[float], p: float) -> float:
    if not data:
        return 0.0
    sd = sorted(data)
    k = (len(sd) - 1) * p
    f = int(k)
    c = min(f + 1, len(sd) - 1)
    if f == c:
        return sd[f]
    return sd[f] + (sd[c] - sd[f]) * (k - f)


def stream_completion(
    *,
    api: str,
    model: str,
    prompt: str,
    max_tokens: int,
    seed: int,
    timeout: int,
    ignore_eos: bool = False,
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "top_p": 1.0,
        "seed": seed,
        "stream": True,
    }
    if ignore_eos:
        body["ignore_eos"] = True

    chunks: list[str] = []
    chunk_abs_ms: list[float] = []
    finish_reason: str | None = None
    start = time.perf_counter()

    with requests.post(
        f"{api}/completions",
        json=body,
        stream=True,
        timeout=(10, timeout),
    ) as resp:
        resp.raise_for_status()
        for raw_line in resp.iter_lines(decode_unicode=True):
            if not raw_line:
                continue
            if not raw_line.startswith("data:"):
                continue
            payload = raw_line[len("data:"):].strip()
            if payload == "[DONE]":
                break
            try:
                event = json.loads(payload)
            except json.JSONDecodeError:
                continue
            for choice in event.get("choices", []):
                text = choice.get("text")
                if text is None:
                    delta = choice.get("delta") or {}
                    text = delta.get("content", "")
                if text:
                    chunks.append(text)
                    chunk_abs_ms.append((time.perf_counter() - start) * 1000.0)
                if choice.get("finish_reason") is not None:
                    finish_reason = choice.get("finish_reason")

    wall_s = time.perf_counter() - start
    token_deltas_ms = [
        chunk_abs_ms[i] - chunk_abs_ms[i - 1]
        for i in range(1, len(chunk_abs_ms))
    ]
    output = "".join(chunks)
    return {
        "wall_s": wall_s,
        "ttft_ms": chunk_abs_ms[0] if chunk_abs_ms else None,
        "token_deltas_ms": token_deltas_ms,
        "n_stream_chunks": len(chunks),
        "output_text": output,
        "output_chars": len(output),
        "finish_reason": finish_reason,
    }


def load_sharegpt(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open() as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("_header"):
                continue
            rows.append(row)
    return rows


def first_human_prompt(conv: dict[str, Any]) -> str | None:
    for turn in conv.get("conversations") or []:
        if turn.get("from") == "human":
            value = str(turn.get("value", "")).strip()
            return value or None
    return None


def transcript_prompt(turns: list[dict[str, Any]], human_idx: int) -> str:
    lines: list[str] = []
    for turn in turns[:human_idx + 1]:
        role = turn.get("from")
        value = str(turn.get("value", "")).strip()
        if not value:
            continue
        if role == "human":
            lines.append(f"User: {value}")
        elif role == "gpt":
            lines.append(f"Assistant: {value}")
    lines.append("Assistant:")
    return "\n\n".join(lines)


def write_turn_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "phase",
        "conv_idx",
        "turn_idx",
        "request_id",
        "prompt_chars",
        "output_chars",
        "wall_s",
        "ttft_ms",
        "n_stream_chunks",
        "tpot_p50_ms",
        "tpot_p95_ms",
        "tpot_p99_ms",
        "finish_reason",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fields})


def summarize_for_csv(
    *,
    phase: str,
    prompt: str,
    result: dict[str, Any],
    conv_idx: int | None = None,
    turn_idx: int | None = None,
    request_id: str | None = None,
) -> dict[str, Any]:
    deltas = result["token_deltas_ms"]
    return {
        "phase": phase,
        "conv_idx": "" if conv_idx is None else conv_idx,
        "turn_idx": "" if turn_idx is None else turn_idx,
        "request_id": "" if request_id is None else request_id,
        "prompt_chars": len(prompt),
        "output_chars": result["output_chars"],
        "wall_s": f"{result['wall_s']:.6f}",
        "ttft_ms": "" if result["ttft_ms"] is None else f"{result['ttft_ms']:.3f}",
        "n_stream_chunks": result["n_stream_chunks"],
        "tpot_p50_ms": f"{percentile(deltas, 0.50):.3f}",
        "tpot_p95_ms": f"{percentile(deltas, 0.95):.3f}",
        "tpot_p99_ms": f"{percentile(deltas, 0.99):.3f}",
        "finish_reason": result["finish_reason"] or "",
    }


def run_sharegpt(args: argparse.Namespace) -> int:
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    convs = load_sharegpt(args.sharegpt_slice)
    if args.limit_convs:
        convs = convs[:args.limit_convs]

    turns_out: list[dict[str, Any]] = []
    csv_rows: list[dict[str, Any]] = []
    skipped_long: list[dict[str, Any]] = []
    outputs_path = out_dir / "sharegpt_outputs.jsonl"
    hit_request_limit = False
    with outputs_path.open("w") as outputs:
        for conv_idx, conv in enumerate(convs):
            if hit_request_limit:
                break
            turns = conv.get("conversations") or []
            for turn_idx, turn in enumerate(turns):
                if turn.get("from") != "human":
                    continue
                prompt = transcript_prompt(turns, turn_idx)
                # Length filter applied BEFORE the request-count cap so a
                # pathological prompt cannot consume one of the budgeted slots.
                if args.max_prompt_chars > 0 and len(prompt) > args.max_prompt_chars:
                    skipped_long.append({
                        "conv_idx": conv_idx,
                        "turn_idx": turn_idx,
                        "prompt_chars": len(prompt),
                    })
                    print(
                        f"[sharegpt] skip conv={conv_idx} turn={turn_idx} "
                        f"prompt_chars={len(prompt)} > "
                        f"max_prompt_chars={args.max_prompt_chars}",
                        file=sys.stderr,
                    )
                    continue
                if args.limit_requests > 0 and len(turns_out) >= args.limit_requests:
                    hit_request_limit = True
                    break
                result = stream_completion(
                    api=args.api,
                    model=args.model,
                    prompt=prompt,
                    max_tokens=args.max_tokens,
                    seed=args.seed,
                    timeout=args.timeout,
                    ignore_eos=False,
                )
                turn_record = {
                    "conv_idx": conv_idx,
                    "turn_idx": turn_idx,
                    "prompt_chars": len(prompt),
                    "wall_s": result["wall_s"],
                    "ttft_ms": result["ttft_ms"],
                    "token_deltas_ms": result["token_deltas_ms"],
                    "n_stream_chunks": result["n_stream_chunks"],
                    "finish_reason": result["finish_reason"],
                }
                turns_out.append(turn_record)
                csv_rows.append(summarize_for_csv(
                    phase="sharegpt",
                    prompt=prompt,
                    result=result,
                    conv_idx=conv_idx,
                    turn_idx=turn_idx,
                ))
                outputs.write(json.dumps({
                    **turn_record,
                    "output_text": result["output_text"],
                }, ensure_ascii=False) + "\n")
                outputs.flush()
                print(
                    f"[sharegpt] conv={conv_idx} turn={turn_idx} "
                    f"chunks={result['n_stream_chunks']} "
                    f"wall={result['wall_s']:.2f}s",
                    file=sys.stderr,
                )

    (out_dir / "sharegpt.json").write_text(json.dumps({
        "phase": "sharegpt",
        "api": args.api,
        "model": args.model,
        "seed": args.seed,
        "max_tokens": args.max_tokens,
        "n_conversations": len(convs),
        "limit_requests": args.limit_requests,
        "max_prompt_chars": args.max_prompt_chars,
        "skipped_long": skipped_long,
        "hit_request_limit": hit_request_limit,
        "turns": turns_out,
    }, indent=2))
    write_turn_csv(out_dir / "sharegpt_wall_tpot.csv", csv_rows)
    return 0


def run_longdecode(args: argparse.Namespace) -> int:
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    prompt = args.longdecode_prompt.read_text()
    result = stream_completion(
        api=args.api,
        model=args.model,
        prompt=prompt,
        max_tokens=args.max_tokens,
        seed=args.seed,
        timeout=args.timeout,
        ignore_eos=True,
    )
    (out_dir / "longdecode_output.txt").write_text(result["output_text"])
    turn = {
        "wall_s": result["wall_s"],
        "ttft_ms": result["ttft_ms"],
        "token_deltas_ms": result["token_deltas_ms"],
        "n_stream_chunks": result["n_stream_chunks"],
        "finish_reason": result["finish_reason"],
        "prompt_chars": len(prompt),
        "output_chars": result["output_chars"],
    }
    (out_dir / "longdecode.json").write_text(json.dumps({
        "phase": "longdecode",
        "api": args.api,
        "model": args.model,
        "seed": args.seed,
        "max_tokens": args.max_tokens,
        "ignore_eos": True,
        "turns": [turn],
    }, indent=2))
    write_turn_csv(out_dir / "longdecode_tpot.csv", [
        summarize_for_csv(
            phase="longdecode",
            prompt=prompt,
            result=result,
            request_id="longdecode",
        )
    ])
    print(
        f"[longdecode] chunks={result['n_stream_chunks']} "
        f"wall={result['wall_s']:.2f}s finish={result['finish_reason']}",
        file=sys.stderr,
    )
    return 0


def run_concurrent(args: argparse.Namespace) -> int:
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    convs = load_sharegpt(args.sharegpt_slice)
    prompts: list[str] = []
    for conv in convs:
        prompt = first_human_prompt(conv)
        if prompt:
            prompts.append(f"User: {prompt}\n\nAssistant:")
        if len(prompts) == 2:
            break
    if len(prompts) < 2:
        raise SystemExit("need at least two ShareGPT prompts for concurrent probe")

    def one(idx: int) -> dict[str, Any]:
        return stream_completion(
            api=args.api,
            model=args.model,
            prompt=prompts[idx],
            max_tokens=args.max_tokens,
            seed=args.seed + idx,
            timeout=args.timeout,
            ignore_eos=False,
        )

    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(one, [0, 1]))
    total_wall_s = time.perf_counter() - start

    csv_rows: list[dict[str, Any]] = []
    summary_requests: list[dict[str, Any]] = []
    for idx, result in enumerate(results):
        req_id = "request_a" if idx == 0 else "request_b"
        (out_dir / f"{req_id}_output.txt").write_text(result["output_text"])
        csv_rows.append(summarize_for_csv(
            phase="concurrent",
            prompt=prompts[idx],
            result=result,
            request_id=req_id,
        ))
        summary_requests.append({
            "request_id": req_id,
            "wall_s": result["wall_s"],
            "ttft_ms": result["ttft_ms"],
            "token_deltas_ms": result["token_deltas_ms"],
            "n_stream_chunks": result["n_stream_chunks"],
            "finish_reason": result["finish_reason"],
            "prompt_chars": len(prompts[idx]),
            "output_chars": result["output_chars"],
        })
    write_turn_csv(out_dir / "wall_tpot.csv", csv_rows)
    (out_dir / "summary.json").write_text(json.dumps({
        "phase": "concurrent",
        "api": args.api,
        "model": args.model,
        "seed": args.seed,
        "max_tokens": args.max_tokens,
        "total_wall_s": total_wall_s,
        "requests": summary_requests,
    }, indent=2))
    print(f"[concurrent] wall={total_wall_s:.2f}s", file=sys.stderr)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=("sharegpt", "longdecode", "concurrent"),
                    required=True)
    ap.add_argument("--api", default=DEFAULT_API)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--sharegpt-slice", type=Path,
                    default=Path(__file__).with_name("sharegpt_slice.jsonl"))
    ap.add_argument("--longdecode-prompt", type=Path,
                    default=Path(__file__).with_name("longdecode_prompt.txt"))
    ap.add_argument("--max-tokens", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    # `--timeout` retained as backward-compatible alias for `--http-timeout`
    # (runner.sh and earlier invocations use the original name).
    ap.add_argument("--http-timeout", "--timeout", dest="timeout",
                    type=int, default=900,
                    help="per-request HTTP read timeout (seconds)")
    ap.add_argument("--limit-convs", type=int, default=0,
                    help="debug-only limit for ShareGPT conversations")
    ap.add_argument("--limit-requests", type=int, default=0,
                    help="ShareGPT only: hard cap on issued requests after "
                         "max-prompt-chars filtering. 0 = unlimited.")
    ap.add_argument("--max-prompt-chars", type=int, default=0,
                    help="ShareGPT only: skip turns whose composed prompt "
                         "exceeds this many characters. 0 = unlimited.")
    args = ap.parse_args()

    if args.max_tokens is None:
        if args.phase == "longdecode":
            args.max_tokens = 2048
        elif args.phase == "concurrent":
            args.max_tokens = 128
        else:
            args.max_tokens = 128

    if args.phase == "sharegpt":
        return run_sharegpt(args)
    if args.phase == "longdecode":
        return run_longdecode(args)
    if args.phase == "concurrent":
        return run_concurrent(args)
    raise AssertionError(args.phase)


if __name__ == "__main__":
    sys.exit(main())
