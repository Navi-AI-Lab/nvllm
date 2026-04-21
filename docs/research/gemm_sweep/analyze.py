"""Shortlisting logic for Phase B.1 -> B.2 handoff. Importable by notebooks
and code generators."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable

import pandas as pd

CFG_RE = re.compile(
    r"Cfg_(?P<tm>\d+)x(?P<tn>\d+)x(?P<tk>\d+)_(?P<sched>Auto|TmaWS|TmaWSPing|TmaWSCoop)_(?P<tsched>Pers|SK)"
)

M_BUCKETS = [
    ("1-8",     (1, 8)),
    ("16-32",   (16, 32)),
    ("64-128",  (64, 128)),
    ("192-256", (192, 256)),
]


def load(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = df[df["min_us"] > 0].copy()
    parsed = df["config_id"].str.extract(CFG_RE)
    # smoke_M256 doesn't match the regex — leave those columns NaN for it
    df = pd.concat([df, parsed], axis=1)
    for col in ("tm", "tn", "tk"):
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
    return df


def bucket_of(m: int) -> str | None:
    for name, (lo, hi) in M_BUCKETS:
        if lo <= m <= hi:
            return name
    return None


def shortlist_top3(df: pd.DataFrame) -> dict:
    """Top-3 configs per (shape x M-bucket) by min μs within the bucket.

    Output schema:
        {
          "by_shape": {
              "<shape>": {"<bucket>": ["<cfg1>", "<cfg2>", "<cfg3>"], ...},
              ...
          },
          "all_configs": ["<cfg>", ...],   # deduplicated across shape x bucket
        }
    """
    d = df.copy()
    d["bucket"] = d["M"].apply(bucket_of)
    d = d[d["bucket"].notna()]

    agg = (d.groupby(["shape", "bucket", "config_id"])
             .agg(bucket_min_us=("min_us", "min"),
                  bucket_mean_us=("min_us", "mean"))
             .reset_index())
    agg["rank"] = agg.groupby(["shape", "bucket"])["bucket_min_us"].rank(method="first")
    top3 = agg[agg["rank"] <= 3].sort_values(["shape", "bucket", "rank"])

    by_shape: dict = {}
    for (shape, bucket), group in top3.groupby(["shape", "bucket"]):
        by_shape.setdefault(shape, {})[bucket] = group["config_id"].tolist()

    all_configs = sorted({c for s in by_shape.values() for b in s.values() for c in b})
    return {"by_shape": by_shape, "all_configs": all_configs}


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    df = load(args.csv)
    sl = shortlist_top3(df)
    args.out.write_text(json.dumps(sl, indent=2))
    print(f"Shortlisted {len(sl['all_configs'])} unique configs -> {args.out}")
    print(json.dumps(sl, indent=2))


if __name__ == "__main__":
    main()
