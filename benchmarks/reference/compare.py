#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Compare a native mlxPDLP JSON report with one optional reference report.

Uses only the standard library. Unverified or mismatched cases never get a speed
ratio. Both reports must use the same audit tolerance and serial instance runs.
"""
import argparse
import csv
import json
import math
from pathlib import Path

from common import RESIDUALS


def load(path):
    report = json.loads(Path(path).read_text())
    if report.get("schema") != "mlxpdlp.reference.v1" and report.get("schema_version") != 9:
        raise ValueError(f"unsupported benchmark schema in {path}")
    rows = report["results"]
    if len({r["name"] for r in rows}) != len(rows):
        raise ValueError(f"duplicate instance names in {path}")
    if report["protocol"].get("jobs") != 1:
        raise ValueError(f"{path}: use serial instance runs (--jobs 1 for mlxPDLP)")
    return report, {r["name"]: r for r in rows}


def verified(row, tolerance):
    metrics = row.get("original_model", {})
    return row.get("verified") is True and all(
        isinstance(metrics.get(k), (int, float)) and math.isfinite(metrics[k]) and
        0 <= metrics[k] <= tolerance for k in RESIDUALS)


def comparisons(baseline, reference, tolerance, metric):
    rows = []
    for name in sorted(baseline.keys() | reference.keys()):
        left, right = baseline.get(name), reference.get(name)
        reason, identity = "", "dimensions_only"
        if left is None or right is None:
            reason = "missing_instance"
        elif not left.get("dimensions") or not right.get("dimensions"):
            reason = "missing_dimensions"
        elif any(left["dimensions"][k] != right["dimensions"][k] for k in ("rows", "columns", "nonzeros")):
            reason = "different_dimensions"
        elif left.get("input_sha256") and right.get("input_sha256"):
            identity = "input_sha256"
            if left["input_sha256"] != right["input_sha256"]:
                reason = "different_input"
        if not reason and not (verified(left, tolerance) and verified(right, tolerance)):
            reason = "unverified_solution"
        a = left.get("timing_seconds", {}).get(metric) if left else None
        b = right.get("timing_seconds", {}).get(metric) if right else None
        if not reason and not all(isinstance(v, (float, int)) and math.isfinite(v) and v > 0 for v in (a, b)):
            reason = "missing_timing"
        rows.append({"name": name, "baseline_termination": left.get("termination") if left else None,
                     "reference_termination": right.get("termination") if right else None,
                     "baseline_verified": verified(left, tolerance) if left else False,
                     "reference_verified": verified(right, tolerance) if right else False,
                     "baseline_seconds": a, "reference_seconds": b,
                     "reference_over_baseline": b / a if not reason else None,
                     "input_identity": identity, "reason": reason})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("reference", type=Path)
    parser.add_argument("--metric", choices=("total", "solve"), default="total")
    parser.add_argument("--output", type=Path, help="optional comparison CSV")
    args = parser.parse_args()
    try:
        base, left = load(args.baseline)
        ref, right = load(args.reference)
        tolerance = float(base["protocol"]["tolerance"])
        if not math.isfinite(tolerance) or tolerance <= 0 or not math.isclose(
                tolerance, float(ref["protocol"]["tolerance"]), rel_tol=1e-12):
            raise ValueError("reports must use the same finite positive audit tolerance")
        if base["host"]["name"] != ref["host"]["name"]:
            raise ValueError("reports must be measured on the same host")
        rows = comparisons(left, right, tolerance, args.metric)
    except (KeyError, TypeError, ValueError, OSError) as error:
        parser.error(str(error))
    print(f"{base['solver']} vs {ref['solver']}; {args.metric} wall time; audit={tolerance:g}")
    print("Ratios > 1 mean the baseline is faster. Only independently verified pairs are compared.")
    if any(r["input_identity"] == "dimensions_only" for r in rows):
        print("Input hashes are absent from the native report; confirm identical MPS files. Dimensions are checked.")
    print(f"{'instance':24s} {'baseline s':>12s} {'reference s':>12s} {'ratio':>10s} status")
    for row in rows:
        fmt = lambda v: f"{v:.6g}" if isinstance(v, (float, int)) else "-"
        print(f"{row['name']:24s} {fmt(row['baseline_seconds']):>12s} "
              f"{fmt(row['reference_seconds']):>12s} {fmt(row['reference_over_baseline']):>10s} {row['reason']}")
    print(f"Verified: baseline {sum(r['baseline_verified'] for r in rows)}/{len(rows)}, "
          f"reference {sum(r['reference_verified'] for r in rows)}/{len(rows)}")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", newline="") as stream:
            if rows:
                writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
