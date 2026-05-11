#!/usr/bin/env python3
"""Parse CUTE_PHASE_E_DISPATCH_LOG output from docker logs.

Each emitted line looks like:

    [PHASE_E_DISPATCH] layer_name=model.layers.3.self_attn.attn layer_idx=3 \
        enabled=True restricted_layers=[3, 7] is_decode_only=True \
        use_fusion=True num_seqs=1 resident_cap=72 \
        use_beta_coop=True use_beta_lite=False

We extract one record per (layer_name, use_beta_coop, use_beta_lite) triple,
emit them as JSON, and (optionally) verify that the observed β-coop layer
set matches an expected `--expect-coop-layers` CSV.

Usage:
    docker logs nvllm 2>&1 | python extract_dispatch_log.py \
        --expect-coop-layers 3,7 --json-out dispatch_audit.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from typing import Any

LINE_RE = re.compile(r"\[PHASE_E_DISPATCH\] (.*)$")
KV_RE = re.compile(r"(\w+)=(\[[^\]]*\]|None|True|False|-?\d+|\S+)")


def _parse_value(raw: str) -> Any:
    if raw == "None":
        return None
    if raw == "True":
        return True
    if raw == "False":
        return False
    if raw.startswith("[") and raw.endswith("]"):
        inner = raw[1:-1].strip()
        if not inner:
            return []
        return [int(x.strip()) for x in inner.split(",")]
    try:
        return int(raw)
    except ValueError:
        return raw


def parse_lines(lines):
    records = []
    for line in lines:
        m = LINE_RE.search(line)
        if not m:
            continue
        kv: dict[str, Any] = {}
        for k, v in KV_RE.findall(m.group(1)):
            kv[k] = _parse_value(v)
        if "layer_name" in kv and "use_beta_coop" in kv:
            records.append(kv)
    return records


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--expect-coop-layers",
        type=str,
        default=None,
        help="Comma-separated layer indices expected to fire β-coop. "
        "Pass empty string for arms that should fire 0 β-coop layers.",
    )
    p.add_argument("--json-out", type=str, default=None)
    p.add_argument("--input", type=str, default=None,
                   help="Read from file instead of stdin")
    p.add_argument(
        "--require-records",
        action="store_true",
        help="Fail (exit 3) if no [PHASE_E_DISPATCH] records were parsed. "
        "Use on β-on arms so an empty observed coop_layers can no longer "
        "look like a pass via empty-expected match — it could mean the "
        "audit log never fired (audit broke).",
    )
    args = p.parse_args()

    if args.input:
        with open(args.input) as f:
            lines = f.readlines()
    else:
        lines = sys.stdin.readlines()

    records = parse_lines(lines)
    coop_layers = sorted({
        r["layer_idx"] for r in records
        if r.get("use_beta_coop") is True and r.get("layer_idx") is not None
    })
    lite_layers = sorted({
        r["layer_idx"] for r in records
        if r.get("use_beta_lite") is True and r.get("layer_idx") is not None
    })
    audit = {
        "coop_layers": coop_layers,
        "lite_layers": lite_layers,
        "records": records,
    }

    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(audit, f, indent=2, sort_keys=True)
    else:
        json.dump(audit, sys.stdout, indent=2, sort_keys=True)
        sys.stdout.write("\n")

    if args.require_records and not records:
        sys.stderr.write(
            "DISPATCH AUDIT FAIL: --require-records set but no "
            "[PHASE_E_DISPATCH] records were parsed. The audit log "
            "did not fire — instrumentation is broken or the dispatch "
            "site was not reached.\n"
        )
        return 3

    if args.expect_coop_layers is not None:
        expected = sorted({
            int(x.strip())
            for x in args.expect_coop_layers.split(",")
            if x.strip()
        })
        if coop_layers != expected:
            sys.stderr.write(
                f"DISPATCH AUDIT FAIL: expected coop_layers={expected} "
                f"got coop_layers={coop_layers}\n"
            )
            return 2
        # When β-coop is the expected path, β-lite must NOT also have
        # fired on any layer — that would mean some restricted layer
        # silently fell back to the lite path and the arm's identity
        # is contaminated. (β-lite is acceptable in arms that explicitly
        # request the lite path; this audit only runs against β-coop
        # arms today, so we fail-closed.)
        if lite_layers:
            sys.stderr.write(
                "DISPATCH AUDIT FAIL: β-coop arm but observed β-lite "
                f"on layers {lite_layers}; the restricted layer set "
                "must not also fire β-lite (fallback contamination).\n"
            )
            return 2
        sys.stderr.write(
            f"DISPATCH AUDIT OK: coop_layers={coop_layers} "
            f"lite_layers={lite_layers}\n"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
