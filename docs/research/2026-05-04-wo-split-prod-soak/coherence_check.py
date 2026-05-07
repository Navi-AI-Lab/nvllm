#!/usr/bin/env python3
"""Coherence heuristics for the long-decode probe output.

Warning-only — eyeball remains the primary gate per plan.md. Computes:
  - repeated_4gram_fraction  (total_4grams - unique_4grams) / total_4grams
  - unique_trigram_ratio     unique_3grams / total_3grams (lower = more repetitive)
  - fffd_count               count of U+FFFD replacement chars

Pairwise (when --baseline is given):
  - first_divergence_token_idx  position where current first differs from baseline

Whitespace tokenization is intentional: coherence is about word-level patterns,
not subword BPE artifacts.

Usage:
    .venv/bin/python coherence_check.py \
        --input  benchmarks/.../wo8/primary/run01/longdecode_output.txt \
        --baseline benchmarks/.../wo1/primary/run01/longdecode_output.txt \
        --label wo8/run01
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path


def ngrams(tokens: list[str], n: int) -> list[tuple]:
    return [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]


def repeated_ngram_fraction(tokens: list[str], n: int) -> float:
    """Fraction of n-gram *occurrences* that are repetitions of an earlier
    n-gram. Equivalent to (total_grams - unique_grams) / total_grams.

    Coherent prose: ~5-15%. Gibberish loops approach 100%.
    """
    grams = ngrams(tokens, n)
    if not grams:
        return 0.0
    return (len(grams) - len(set(grams))) / len(grams)


def unique_ngram_ratio(tokens: list[str], n: int) -> float:
    grams = ngrams(tokens, n)
    if not grams:
        return 1.0
    return len(set(grams)) / len(grams)


def first_divergence(a: list[str], b: list[str]) -> int:
    for i, (av, bv) in enumerate(zip(a, b)):
        if av != bv:
            return i
    return min(len(a), len(b))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, type=Path,
                    help="long-decode output text file")
    ap.add_argument("--baseline", type=Path, default=None,
                    help="optional wo_split=1 baseline for prefix diff")
    ap.add_argument("--label", default="",
                    help="tag for the output record")
    args = ap.parse_args()

    text = args.input.read_text()
    tokens = text.split()

    out: dict = {
        "label": args.label,
        "input": str(args.input),
        "n_chars": len(text),
        "n_tokens": len(tokens),
        "fffd_count": text.count("�"),
        "repeated_4gram_fraction": repeated_ngram_fraction(tokens, 4),
        "unique_trigram_ratio": unique_ngram_ratio(tokens, 3),
    }
    if args.baseline is not None and args.baseline.exists():
        baseline_tokens = args.baseline.read_text().split()
        out["baseline"] = str(args.baseline)
        out["baseline_n_tokens"] = len(baseline_tokens)
        out["first_divergence_token_idx"] = first_divergence(
            tokens, baseline_tokens
        )

    warnings: list[str] = []
    if out["fffd_count"] > 0:
        warnings.append(f"fffd_count={out['fffd_count']}")
    if out["repeated_4gram_fraction"] > 0.5:
        warnings.append(
            f"repeated_4gram_fraction={out['repeated_4gram_fraction']:.3f}")
    if out["unique_trigram_ratio"] < 0.5:
        warnings.append(
            f"unique_trigram_ratio={out['unique_trigram_ratio']:.3f}")
    out["warnings"] = warnings

    json.dump(out, sys.stdout, indent=2)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
