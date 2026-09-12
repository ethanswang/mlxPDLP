#!/usr/bin/env python3
# Copyright 2026 Ethan Wang <ethanshurui.wang@gmail.com>
# SPDX-License-Identifier: Apache-2.0
"""Validate and qualify observational LP traces. No solver or NumPy dependency.

This module defines the trace schema and conservative readiness rule. Hashes only
select candidates: packed, validated CSR bytes establish equality.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
import struct
from typing import Any


SCHEMA = "mlxpdlp.batch_lp.trace.v1"
DATA_FIELDS = (
    "objective", "objective_constant", "variable_lower_bounds",
    "variable_upper_bounds", "constraint_lower_bounds", "constraint_upper_bounds",
    "primal_start", "dual_start", "reduced_cost_start",
)
TIMING_FIELDS = ("construct_ns", "solve_ns", "rescale_ns")
INT32_MAX = 2**31 - 1


class TraceError(ValueError):
    """Invalid trace, with a path identifying the offending field/member."""


def require(condition: bool, field: str, message: str) -> None:
    if not condition:
        raise TraceError(f"{field}: {message}")


def integer(value: Any, field: str, maximum: int | None = None) -> int:
    require(type(value) is int and value >= 0, field, "expected a nonnegative integer")
    require(maximum is None or value <= maximum, field, f"exceeds {maximum}")
    return value


def number(value: Any, field: str, *, bounds: bool = False) -> float:
    if bounds and isinstance(value, str) and value in ("+inf", "-inf"):
        return math.inf if value == "+inf" else -math.inf
    require(type(value) in (int, float), field, "expected a number" +
            (" or '+inf'/'-inf'" if bounds else ""))
    try:
        result = float(value)
    except (ValueError, OverflowError) as exc:
        raise TraceError(f"{field}: not representable in FP64") from exc
    require(math.isfinite(result), field, "expected finite FP64 data")
    return result


def vector(value: Any, length: int, field: str, *, bounds: bool = False) -> list[float]:
    require(isinstance(value, list) and len(value) == length, field,
            f"expected an array of length {length}; broadcasting is not part of the trace format")
    return [number(x, f"{field}[{i}]", bounds=bounds) for i, x in enumerate(value)]


def text_field(value: Any, field: str) -> str:
    require(isinstance(value, str) and bool(value.strip()), field, "expected a nonempty string")
    return value


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    obj: dict[str, Any] = {}
    for key, value in pairs:
        require(key not in obj, key, "duplicate JSON key")
        obj[key] = value
    return obj


def read_trace(path: str | Path) -> dict[str, Any]:
    def reject_constant(value: str) -> None:
        raise TraceError(f"nonstandard JSON number {value}; use '+inf'/'-inf' for bounds")

    with open(path, encoding="utf-8") as stream:
        trace = json.load(stream, object_pairs_hook=_unique_object,
                          parse_constant=reject_constant)
    validate_trace(trace)
    return trace


def write_json(path: str | Path, value: Any) -> None:
    with open(path, "w", encoding="utf-8") as stream:
        json.dump(value, stream, allow_nan=False, indent=2, sort_keys=True)
        stream.write("\n")


def matrix_bytes(matrix: dict[str, Any], field: str = "matrix") -> bytes:
    """Validate raw CSR without sorting, coalescing, or dropping any entry."""
    require(isinstance(matrix, dict), field, "expected an object")
    n = integer(matrix.get("num_variables"), field + ".num_variables", INT32_MAX)
    m = integer(matrix.get("num_constraints"), field + ".num_constraints", INT32_MAX)
    rp, ci, values = (matrix.get(k) for k in ("row_ptr", "col_indices", "values"))
    require(isinstance(rp, list) and len(rp) == m + 1, field + ".row_ptr",
            "expected num_constraints + 1 entries")
    require(isinstance(ci, list) and isinstance(values, list) and len(ci) == len(values),
            field, "col_indices and values must have equal lengths")
    nnz = integer(len(values), field + ".nnz", INT32_MAX)
    for i, value in enumerate(rp):
        integer(value, f"{field}.row_ptr[{i}]", nnz)
        require(i == 0 or rp[i - 1] <= value, field + ".row_ptr", "must be monotone")
    require(rp[0] == 0 and rp[-1] == nnz, field + ".row_ptr", "must span [0, nnz]")
    for i, value in enumerate(ci):
        integer(value, f"{field}.col_indices[{i}]", n - 1)
    vals = vector(values, nnz, field + ".values")
    # Include dimensions and raw duplicate order. Even +0/-0 stay distinct.
    return (struct.pack("<iii", n, m, nnz) + struct.pack(f"<{m + 1}i", *rp) +
            struct.pack(f"<{nnz}i", *ci) + struct.pack(f"<{nnz}d", *vals))


class MatrixRegistry:
    """Own one JSON snapshot and one exact comparison key per distinct matrix."""

    def __init__(self) -> None:
        self.matrices: dict[str, dict[str, Any]] = {}
        self._candidates: dict[str, list[tuple[str, bytes]]] = defaultdict(list)

    def add(self, matrix: dict[str, Any]) -> str:
        packed = matrix_bytes(matrix)
        digest = hashlib.sha256(packed).hexdigest()
        for matrix_id, existing in self._candidates[digest]:
            if packed == existing:  # A hash collision must never combine operators.
                return matrix_id
        matrix_id = f"{digest}:{len(self._candidates[digest])}"
        self.matrices[matrix_id] = json.loads(json.dumps(matrix, allow_nan=False))
        self._candidates[digest].append((matrix_id, packed))
        return matrix_id


def validate_data(data: Any, matrix: dict[str, Any], field: str) -> None:
    require(isinstance(data, dict), field, "expected an object")
    require(set(data) == set(DATA_FIELDS), field, "must contain exactly the documented LP fields")
    n, m = matrix["num_variables"], matrix["num_constraints"]
    vector(data["objective"], n, field + ".objective")
    number(data["objective_constant"], field + ".objective_constant")
    for prefix, length in (("variable", n), ("constraint", m)):
        lo = vector(data[prefix + "_lower_bounds"], length,
                    field + f".{prefix}_lower_bounds", bounds=True)
        hi = vector(data[prefix + "_upper_bounds"], length,
                    field + f".{prefix}_upper_bounds", bounds=True)
        for i, (lower, upper) in enumerate(zip(lo, hi)):
            require(lower != math.inf, field + f".{prefix}_lower_bounds[{i}]", "cannot be +inf")
            require(upper != -math.inf, field + f".{prefix}_upper_bounds[{i}]", "cannot be -inf")
            require(lower <= upper, field + f".{prefix}_bounds[{i}]", "lower exceeds upper")
    for name, length in (("primal_start", n), ("dual_start", m), ("reduced_cost_start", n)):
        if data[name] is not None:
            vector(data[name], length, field + "." + name)


def validate_trace(trace: Any) -> None:
    require(isinstance(trace, dict) and trace.get("schema") == SCHEMA, "schema", f"expected {SCHEMA}")
    for name in ("source", "application", "matrices"):
        require(isinstance(trace.get(name), dict), name, "expected an object")
    source, application = trace["source"], trace["application"]
    require(source.get("kind") in ("application", "synthetic"), "source.kind",
            "expected application or synthetic")
    for name in ("representative", "scheduling_unchanged", "complete_capture", "settings_complete"):
        require(type(source.get(name)) is bool, "source." + name, "expected true or false")
    for name in ("application", "workload_id", "selection", "revision", "solver_revision",
                 "mlx_revision", "hardware", "build", "command"):
        text_field(source.get(name), "source." + name)
    wall = integer(application.get("wall_time_ns"), "application.wall_time_ns")
    require(wall > 0, "application.wall_time_ns", "must be positive")
    for name in ("latency_limit_ns", "memory_limit_bytes"):
        if application.get(name) not in (None, "unlimited"):
            integer(application[name], "application." + name)
            require(application[name] > 0, "application." + name,
                    "must be positive, 'unlimited', or null (unknown)")
    text_field(application.get("scheduling_policy"), "application.scheduling_policy")
    matrices = trace["matrices"]
    for key, matrix in matrices.items():
        text_field(key, "matrices key")
        matrix_bytes(matrix, f"matrices[{key}]")
    requests = trace.get("requests")
    require(isinstance(requests, list), "requests", "expected an array")
    by_id: dict[str, dict[str, Any]] = {}
    for i, req in enumerate(requests):
        field = f"requests[{i}]"
        require(isinstance(req, dict), field, "expected an object")
        rid = text_field(req.get("id"), field + ".id")
        require(rid not in by_id, field + ".id", "duplicate request id")
        by_id[rid] = req
        field += f"({rid})"
        text_field(req.get("family"), field + ".family")
        matrix_id = text_field(req.get("matrix_id"), field + ".matrix_id")
        require(matrix_id in matrices, field + ".matrix_id", "unknown matrix")
        require(req.get("device") in ("cpu", "gpu"), field + ".device", "expected cpu or gpu")
        params = req.get("parameters")
        require(isinstance(params, dict) and type(params.get("presolve")) is bool,
                field + ".parameters.presolve", "must record actual presolve setting")
        criteria = params.get("termination_criteria")
        require(isinstance(criteria, dict), field + ".parameters.termination_criteria",
                "must record requested tolerances and limits")
        for name in ("eps_optimal_relative", "eps_feasible_relative", "eps_infeasible_relative"):
            require(number(criteria.get(name), field + ".parameters." + name) > 0,
                    field + ".parameters." + name, "must be positive")
        integer(criteria.get("iteration_limit"), field + ".parameters.iteration_limit", INT32_MAX)
        time_limit = criteria.get("time_sec_limit")
        require(time_limit == "+inf" or number(time_limit, field + ".parameters.time_sec_limit") >= 0,
                field + ".parameters.time_sec_limit", "must be nonnegative")
        # Ensure even ancillary settings have a portable JSON representation.
        try:
            json.dumps(params, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise TraceError(f"{field}.parameters: not strict JSON") from exc
        start = integer(req.get("start_ns"), field + ".start_ns")
        end = integer(req.get("end_ns"), field + ".end_ns")
        require(start <= end <= wall, field, "execution must be inside the application interval")
        if req.get("ready_ns") is not None:
            ready = integer(req["ready_ns"], field + ".ready_ns")
            require(ready <= start, field + ".ready_ns", "data becomes ready after dispatch")
        deps = req.get("depends_on")
        require(deps is None or (isinstance(deps, list) and all(isinstance(d, str) for d in deps)),
                field + ".depends_on", "expected request ids, [] for independent, or null for unknown")
        if deps is not None:
            require(len(deps) == len(set(deps)), field + ".depends_on", "duplicate dependency")
        for name in TIMING_FIELDS:
            if req.get(name) is not None:
                value = integer(req[name], field + "." + name)
                require(value <= end - start, field + "." + name, "exceeds request execution")
        if all(req.get(name) is not None for name in ("construct_ns", "solve_ns")):
            require(req["construct_ns"] + req["solve_ns"] <= end - start,
                    field, "constructor and solve times overlap")
        if req.get("rescale_ns") is not None and req.get("solve_ns") is not None:
            require(req["rescale_ns"] <= req["solve_ns"], field + ".rescale_ns", "must be inside solve")
        if req.get("iterations") is not None:
            integer(req["iterations"], field + ".iterations")
        text_field(req.get("termination"), field + ".termination")
        validate_data(req.get("data"), matrices[matrix_id], field + ".data")
    children: dict[str, list[str]] = defaultdict(list)
    degrees: dict[str, int] = {}
    for rid, req in by_id.items():
        deps = req.get("depends_on") or []
        degrees[rid] = len(deps)
        for dep in deps:
            require(dep in by_id and dep != rid, f"request {rid}.depends_on", "unknown or self dependency")
            ready = req.get("ready_ns")
            require(by_id[dep]["end_ns"] <= (req["start_ns"] if ready is None else ready),
                    f"request {rid}.ready_ns", f"precedes completion of dependency {dep}")
            children[dep].append(rid)
    queue = [rid for rid, degree in degrees.items() if degree == 0]
    visited = 0
    while queue:
        visited += 1
        for child in children[queue.pop()]:
            degrees[child] -= 1
            if degrees[child] == 0:
                queue.append(child)
    require(visited == len(requests), "requests.depends_on", "dependency cycle")


def distribution(values: list[int]) -> dict[str, Any]:
    if not values:
        return {"count": 0, "min": None, "p50": None, "p95": None, "max": None}
    ordered = sorted(values)
    return {"count": len(values), "min": ordered[0], "p50": statistics.median(ordered),
            "p95": ordered[math.ceil(0.95 * len(ordered)) - 1], "max": ordered[-1]}


def _groups(trace: dict[str, Any], identities: dict[str, str]) -> list[dict[str, Any]]:
    """Disjoint groups available at observed dispatches, with no extra waiting."""
    requests = trace["requests"]
    partitions: dict[tuple[str, ...], list[int]] = defaultdict(list)
    for i, req in enumerate(requests):
        key = (req["family"], identities[req["matrix_id"]], req["device"],
               json.dumps(req["parameters"], sort_keys=True, allow_nan=False))
        partitions[key].append(i)
    groups = []
    limit = trace["application"].get("latency_limit_ns")
    for indices in partitions.values():
        remaining = sorted(indices, key=lambda i: (requests[i]["start_ns"], i))
        while remaining:
            first = remaining[0]
            dispatch = requests[first]["start_ns"]
            eligible = []
            for i in remaining:
                req = requests[i]
                ready = req.get("ready_ns")
                if (ready is not None and req.get("depends_on") is not None and
                        ready <= dispatch and limit is not None and
                        (limit == "unlimited" or dispatch < ready + limit)):
                    eligible.append(i)
            # Do not move a request ahead of a data/control dependency, even for
            # zero-duration requests whose timestamps happen to be identical.
            eligible_ids = {requests[i]["id"] for i in eligible}
            eligible = [i for i in eligible if not
                        eligible_ids.intersection(requests[i]["depends_on"])]
            if first not in eligible:
                eligible = [first]
            taken = set(eligible)
            remaining = [i for i in remaining if i not in taken]
            groups.append({"family": requests[first]["family"], "at_ns": dispatch,
                           "input_indices": eligible, "width": len(eligible)})
    return sorted(groups, key=lambda g: (g["at_ns"], g["input_indices"][0]))


def _wall_accounting(requests: list[dict[str, Any]], compatible: set[int]) -> dict[str, Any]:
    """Integrate wall occupation, dividing overlapping intervals among live LPs."""
    events: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for i, req in enumerate(requests):
        if req["end_ns"] > req["start_ns"]:
            events[req["start_ns"]].append((i, 1))
            events[req["end_ns"]].append((i, -1))
    active: set[int] = set()
    union = 0
    peak = 0
    family_wall: dict[str, float] = defaultdict(float)
    previous = 0
    for timestamp in sorted(events):
        if active:
            elapsed = timestamp - previous
            union += elapsed
            for i in active.intersection(compatible):
                family_wall[requests[i]["family"]] += elapsed / len(active)
        for i, sign in events[timestamp]:
            if sign == -1:
                active.remove(i)
        for i, sign in events[timestamp]:
            if sign == 1:
                active.add(i)
        peak = max(peak, len(active))
        previous = timestamp
    return {"lp_wall_union_ns": union, "peak_concurrent_requests": peak,
            "compatible_family_wall_ns": dict(family_wall)}


def analyze(trace: dict[str, Any], *, allow_synthetic: bool = False) -> dict[str, Any]:
    validate_trace(trace)
    registry = MatrixRegistry()
    identities = {key: registry.add(matrix) for key, matrix in trace["matrices"].items()}
    groups = _groups(trace, identities)
    compatible = {i for group in groups if group["width"] >= 2 for i in group["input_indices"]}
    requests = trace["requests"]
    wall = _wall_accounting(requests, compatible)
    source, application = trace["source"], trace["application"]
    missing = []
    if not (allow_synthetic and source["kind"] == "synthetic") and (
            source["kind"] != "application" or not source["representative"]):
        missing.append("no representative application capture (synthetic/smoke fixtures cannot qualify)")
    if allow_synthetic and source["kind"] == "synthetic" and source.get("timings_measured") is not True:
        missing.append("synthetic qualification requires measured timings, not artificial fixture timestamps")
    for name in ("scheduling_unchanged", "complete_capture", "settings_complete"):
        if not source[name]:
            missing.append(f"source.{name} is false")
    for name in ("latency_limit_ns", "memory_limit_bytes"):
        if application.get(name) is None:
            missing.append(f"application.{name} is unknown")
    for name in ("ready_ns", "depends_on", *TIMING_FIELDS, "iterations"):
        ids = [req["id"] for req in requests if req.get(name) is None]
        if ids:
            missing.append(f"{name} is unknown for {len(ids)} requests: {', '.join(ids[:5])}")
    if not requests or wall["lp_wall_union_ns"] == 0:
        missing.append("no measured LP wall time")
    families = []
    for family in sorted({req["family"] for req in requests}):
        members = [r for r in requests if r["family"] == family]
        widths = [g["width"] for g in groups if g["family"] == family]
        contribution = wall["compatible_family_wall_ns"].get(family, 0.0)
        fraction = contribution / wall["lp_wall_union_ns"] if wall["lp_wall_union_ns"] else 0.0
        sizes = distribution(widths)
        families.append({
            "family": family, "requests": len(members), "ready_group_sizes": sizes,
            "ready_group_histogram": dict(sorted(Counter(widths).items())),
            "compatible_lp_wall_ns": contribution, "compatible_lp_wall_fraction": fraction,
            "numerical_workload_gate_passed": fraction >= 0.25 and sizes["p50"] >= 4,
            "presolve_requested": sum(r["parameters"]["presolve"] for r in members),
            "warm_started": sum(any(r["data"][k] is not None for k in
                                    ("primal_start", "dual_start", "reduced_cost_start")) for r in members),
            "varying_fields": [key for key in DATA_FIELDS if len({
                json.dumps(r["data"][key], sort_keys=True) for r in members}) > 1],
            "settings_variants": len({json.dumps(r["parameters"], sort_keys=True) for r in members}),
            "latency_ns": distribution([r["end_ns"] - r["ready_ns"] for r in members
                                        if r.get("ready_ns") is not None]),
            "queue_ns": distribution([r["start_ns"] - r["ready_ns"] for r in members
                                      if r.get("ready_ns") is not None]),
            "execution_ns": distribution([r["end_ns"] - r["start_ns"] for r in members]),
            "iterations": distribution([r["iterations"] for r in members if r.get("iterations") is not None]),
            "termination_counts": dict(Counter(r["termination"] for r in members)),
            "timing_sums_ns": {name: sum(r[name] for r in members if r.get(name) is not None)
                               for name in TIMING_FIELDS},
        })
    demand_passed = any(family["numerical_workload_gate_passed"] for family in families)
    reasons = list(missing)
    if not demand_passed:
        reasons.append("no family has >=25% of LP wall occupation in compatible groups and median ready width >=4")
    matrix_reports = []
    for identity, matrix in registry.matrices.items():
        members = [r for r in requests if identities[r["matrix_id"]] == identity]
        matrix_reports.append({"identity": identity, "num_variables": matrix["num_variables"],
                               "num_constraints": matrix["num_constraints"], "nnz": len(matrix["values"]),
                               "requests": len(members), "reuse_count": max(0, len(members) - 1),
                               "declared_ids": [key for key, value in identities.items() if value == identity]})
    return {
        "schema": "mlxpdlp.batch_lp.qualification.v1",
        "qualification_policy": "measured-derived-families-allowed" if allow_synthetic else "application-only",
        "decision": "go" if demand_passed and not missing else "defer",
        "evidence_complete": not missing, "reasons": reasons,
        "thresholds": {"compatible_lp_wall_fraction": 0.25, "median_ready_width": 4},
        "readiness_rule": "disjoint compatible groups ready at an observed dispatch; no added waiting",
        "wall_accounting": "union of LP execution intervals; overlaps divided equally among active requests",
        "source": source, "application": application,
        "total_requests": len(requests), "distinct_exact_matrices": len(registry.matrices),
        "lp_wall_union_ns": wall["lp_wall_union_ns"],
        "lp_fraction_of_application_wall": wall["lp_wall_union_ns"] / application["wall_time_ns"],
        "summed_lp_execution_ns": sum(r["end_ns"] - r["start_ns"] for r in requests),
        "peak_concurrent_requests": wall["peak_concurrent_requests"],
        "matrices": matrix_reports, "families": families, "ready_groups": groups,
        "release_gate_evaluated": False,
    }


def write_csv(path: str | Path, report: dict[str, Any]) -> None:
    columns = ("family", "requests", "median_ready_width", "p95_ready_width",
               "compatible_lp_wall_fraction", "median_latency_ns", "p95_latency_ns",
               "construct_ns", "solve_ns", "rescale_ns", "numerical_workload_gate_passed")
    with open(path, "w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for family in report["families"]:
            writer.writerow({
                "family": family["family"], "requests": family["requests"],
                "median_ready_width": family["ready_group_sizes"]["p50"],
                "p95_ready_width": family["ready_group_sizes"]["p95"],
                "compatible_lp_wall_fraction": family["compatible_lp_wall_fraction"],
                "median_latency_ns": family["latency_ns"]["p50"],
                "p95_latency_ns": family["latency_ns"]["p95"],
                **family["timing_sums_ns"],
                "numerical_workload_gate_passed": family["numerical_workload_gate_passed"],
            })


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--csv", type=Path)
    parser.add_argument("--require-go", action="store_true", help="exit 2 on defer")
    parser.add_argument("--allow-synthetic", action="store_true",
                        help="explicitly permit measured derived workload qualification")
    args = parser.parse_args()
    try:
        report = analyze(read_trace(args.trace), allow_synthetic=args.allow_synthetic)
        write_json(args.output, report)
        if args.csv:
            write_csv(args.csv, report)
    except (OSError, ValueError, TypeError) as exc:
        parser.error(str(exc))
    print(f"{report['decision']}: {report['total_requests']} LPs, "
          f"{report['distinct_exact_matrices']} exact matrices")
    for reason in report["reasons"]:
        print(f"  {reason}")
    return 2 if args.require_go and report["decision"] != "go" else 0


if __name__ == "__main__":
    raise SystemExit(main())
