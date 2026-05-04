#!/usr/bin/env python3
"""Generate a deterministic ShareGPT slice for the wo_split production soak.

Source:  anon8231489123/ShareGPT_Vicuna_unfiltered (HuggingFace dataset)
Output:  sharegpt_slice.jsonl in this directory
Header:  first JSONL line is a metadata record (`_header: true`) containing
         the dataset revision SHA, license, and filter rules.

Filter:
  - Multi-turn (>=2 conversation turns)
  - First N convs per length bucket in dataset order (deterministic)
  - 10x short, 10x medium, 10x long → 30 convs total
  - Bucket boundaries are character-length on the first human turn
    (~4 chars/token rule of thumb; exact tokenization happens at serve time)

Usage:
    cd docs/research/2026-05-04-wo-split-prod-soak/
    /home/natfii/docker/nvllm/.venv/bin/python gen_sharegpt_slice.py
"""

import argparse
import json
import sys
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

DATASET_REPO = "anon8231489123/ShareGPT_Vicuna_unfiltered"
SLICE_OUT = Path(__file__).parent / "sharegpt_slice.jsonl"

UNKNOWN_LICENSE = "unknown"

# Bucket boundaries on first human-turn character length.
# 4 chars/token estimate → matches plan.md spec (short ~50, medium ~500, long ~1500 tokens).
BUCKETS: dict[str, tuple[int, int]] = {
    "short":  (40, 250),       # ~10-60 tokens
    "medium": (1600, 2600),    # ~400-650 tokens
    "long":   (5000, 7500),    # ~1250-1875 tokens
}
PER_BUCKET = 10
EXPECTED_TOTAL = PER_BUCKET * len(BUCKETS)


def pin_revision(api: HfApi) -> str:
    """Resolve the current main-branch SHA so the slice is reproducible."""
    refs = api.list_repo_refs(DATASET_REPO, repo_type="dataset")
    for branch in refs.branches:
        if branch.name == "main":
            return branch.target_commit
    raise RuntimeError(f"no 'main' branch on {DATASET_REPO}")


def fetch_license(api: HfApi, revision: str) -> str:
    info = api.dataset_info(DATASET_REPO, revision=revision)
    if info.card_data and "license" in info.card_data:
        return str(info.card_data["license"])
    return UNKNOWN_LICENSE


def load_conversations(cache_dir: Path) -> list[dict]:
    """ShareGPT_Vicuna_unfiltered ships as one or more *.json files.
    Concatenate all top-level JSON arrays (typically a single file)."""
    json_files = sorted(cache_dir.glob("*.json"))
    if not json_files:
        raise RuntimeError(f"no .json files under {cache_dir}")
    convs: list[dict] = []
    for jf in json_files:
        print(f"  loading {jf.name} ({jf.stat().st_size / 1e6:.1f} MB)...",
              file=sys.stderr)
        with open(jf) as f:
            convs.extend(json.load(f))
    return convs


def first_human_prompt(conv: dict) -> str | None:
    turns = conv.get("conversations") or []
    if len(turns) < 2:
        return None
    for t in turns:
        if t.get("from") == "human":
            v = t.get("value", "").strip()
            return v if v else None
    return None


def select_slice(convs: list[dict]) -> list[dict]:
    selected: dict[str, list[dict]] = {b: [] for b in BUCKETS}
    for conv in convs:
        prompt = first_human_prompt(conv)
        if prompt is None:
            continue
        plen = len(prompt)
        for bucket, (lo, hi) in BUCKETS.items():
            if lo <= plen <= hi and len(selected[bucket]) < PER_BUCKET:
                selected[bucket].append(conv)
                break
        if all(len(selected[b]) == PER_BUCKET for b in BUCKETS):
            break

    out: list[dict] = []
    for bucket in ("short", "medium", "long"):
        n = len(selected[bucket])
        print(f"  {bucket:6s}: {n}/{PER_BUCKET}", file=sys.stderr)
        out.extend(selected[bucket])
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--allow-unknown-license",
        action="store_true",
        help="proceed even if the dataset card has no license field; the "
             "header will record 'unknown' and the human committer must "
             "verify the dataset card before merging the slice.",
    )
    args = ap.parse_args()

    api = HfApi()
    print(f"resolving revision for {DATASET_REPO}...", file=sys.stderr)
    revision = pin_revision(api)
    license_str = fetch_license(api, revision)
    print(f"  revision: {revision}", file=sys.stderr)
    print(f"  license:  {license_str}", file=sys.stderr)
    if license_str == UNKNOWN_LICENSE and not args.allow_unknown_license:
        print(
            "ERROR: dataset card has no license field. Re-run with "
            "--allow-unknown-license after manually verifying the dataset's "
            "license, or pick a different ShareGPT mirror.",
            file=sys.stderr,
        )
        return 2

    print("downloading dataset (cached after first run)...", file=sys.stderr)
    cache_dir = Path(snapshot_download(
        repo_id=DATASET_REPO,
        repo_type="dataset",
        revision=revision,
    ))

    print("loading conversations...", file=sys.stderr)
    convs = load_conversations(cache_dir)
    print(f"  total: {len(convs)} convs", file=sys.stderr)

    print("selecting slice...", file=sys.stderr)
    final = select_slice(convs)
    if len(final) != EXPECTED_TOTAL:
        print(
            f"ERROR: selected {len(final)}/{EXPECTED_TOTAL} convs — bucket "
            "boundaries may need widening, or the dataset is too small. "
            "Refusing to write an underpowered slice.",
            file=sys.stderr,
        )
        return 3

    header = {
        "_header": True,
        "source_repo": DATASET_REPO,
        "revision_sha": revision,
        "license": license_str,
        "filter_rules": {
            "min_turns": 2,
            "buckets": {b: {"char_range": list(r), "n_target": PER_BUCKET}
                        for b, r in BUCKETS.items()},
            "selection": "first-N-per-bucket-in-dataset-order",
        },
        "n_convs": len(final),
        "generator": "gen_sharegpt_slice.py",
    }
    with open(SLICE_OUT, "w") as f:
        f.write(json.dumps(header) + "\n")
        for conv in final:
            f.write(json.dumps(conv, ensure_ascii=False) + "\n")
    print(f"wrote {SLICE_OUT} ({SLICE_OUT.stat().st_size / 1024:.1f} KB)",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
