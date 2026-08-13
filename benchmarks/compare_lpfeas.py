#!/usr/bin/env python3

# Copyright 2026 Ethan Wang <ethanshurui.wang@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Compare mlxPDLP LPfeas CSV output with the published B200 GPU table."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path


REFERENCE_COLUMNS = {
    "cuopt": "cuopt_26_06_seconds",
    "cupdlpx": "cupdlpx_0_2_9_seconds",
    "coptg": "coptg_seconds",
    "hprlp": "hprlp_0_1_2_seconds",
}


def read_rows(path: Path) -> dict[str, dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return {row["name"]: row for row in csv.DictReader(stream)}


def numeric(value: str) -> float | None:
    try:
        return float(value)
    except ValueError:
        return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=Path, help="CSV from mlxpdlp_lpfeas_benchmark")
    parser.add_argument(
        "--reference",
        choices=sorted(REFERENCE_COLUMNS),
        default="cupdlpx",
        help="published NVIDIA B200 implementation to compare (default: cupdlpx)",
    )
    parser.add_argument(
        "--reference-csv",
        type=Path,
        default=Path(__file__).parent / "data/lpfeas/online_gpu_reference.csv",
    )
    parser.add_argument("--output", type=Path, help="optional machine-readable comparison CSV")
    args = parser.parse_args()

    measured = read_rows(args.results)
    published = read_rows(args.reference_csv)
    reference_column = REFERENCE_COLUMNS[args.reference]
    comparisons: list[dict[str, str]] = []
    shifted_ratios: list[float] = []

    print(f"{'instance':<24} {'mlxPDLP':>10} {'online':>10} {'online/mlx':>12} status")
    for name, row in measured.items():
        reference_row = published.get(name)
        if reference_row is None:
            print(f"{name:<24} {'-':>10} {'missing':>10} {'-':>12} {row['termination']}")
            continue
        measured_seconds = numeric(row["solve_seconds"])
        reference_value = reference_row[reference_column]
        reference_seconds = numeric(reference_value)
        ratio = None
        if measured_seconds is not None and measured_seconds > 0 and reference_seconds is not None:
            ratio = reference_seconds / measured_seconds
            shifted_ratios.append((reference_seconds + 10.0) / (measured_seconds + 10.0))
        print(
            f"{name:<24} {measured_seconds:10.3f} {reference_value:>10} "
            f"{ratio if ratio is not None else '-':>12.3f} {row['termination']}"
            if ratio is not None
            else f"{name:<24} {measured_seconds:10.3f} {reference_value:>10} {'-':>12} {row['termination']}"
        )
        comparisons.append(
            {
                "name": name,
                "mlxPDLP_seconds": row["solve_seconds"],
                "mlxPDLP_termination": row["termination"],
                "mlxPDLP_verified_float64": row["verified"],
                "reference": args.reference,
                "reference_seconds_or_status": reference_value,
                "reference_over_mlxPDLP": "" if ratio is None else f"{ratio:.17g}",
            }
        )

    if shifted_ratios:
        shifted_geomean = math.exp(sum(math.log(value) for value in shifted_ratios) /
                                    len(shifted_ratios))
        print(
            f"\nComparable numeric rows: {len(shifted_ratios)}; "
            f"shifted (10 s) online/mlxPDLP geometric mean: {shifted_geomean:.3f}x"
        )
    else:
        print("\nNo numeric rows were comparable.")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=comparisons[0].keys() if comparisons else [])
            if comparisons:
                writer.writeheader()
                writer.writerows(comparisons)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
