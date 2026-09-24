#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Optional reference benchmark protocol; never imported by mlxPDLP itself.

Only the selected adapter imports its solver. NumPy/SciPy are loaded on demand.
The audit follows benchmarks/lpfeas_support.cpp, independently of solver status.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
AUDIT = "mlxPDLP.original_l2.v1"
RESIDUALS = (
    "relative_primal_residual", "relative_dual_residual", "relative_objective_gap",
    "relative_variable_bound_violation", "relative_dual_bound_violation",
)


@dataclass
class Problem:
    """Original, unscaled LP, normalized to minimization for the audit."""
    matrix: Any
    objective: Any
    variable_lower: Any
    variable_upper: Any
    constraint_lower: Any
    constraint_upper: Any
    offset: float = 0.0
    objective_sign: float = 1.0  # -1 restores a maximization objective.


@dataclass
class Solution:
    primal: Any
    dual: Any
    reduced_cost: Any
    termination: str
    iterations: int
    details: dict
    has_solution: bool = True


def audit(problem: Problem, solution: Solution, tolerance: float,
          reference_objective: float | None = None) -> dict:
    import numpy as np

    m, n = problem.matrix.shape
    x, y, z = (np.asarray(v, dtype=np.float64) for v in
               (solution.primal, solution.dual, solution.reduced_cost))
    metrics = {key: None for key in RESIDUALS}
    metrics.update(finite=False, dimensions_match=False, verified=False)
    if not solution.has_solution or x.shape != (n,) or y.shape != (m,) or z.shape != (n,):
        return metrics
    metrics["dimensions_match"] = True
    if not all(np.isfinite(v).all() for v in (x, y, z, problem.objective, problem.matrix.data)):
        return metrics

    def norm(v):
        return float(np.linalg.norm(v))

    def violation(value, lower, upper):
        return np.maximum(np.maximum(lower - value, value - upper), 0.0)

    def bound_norm(lower, upper):
        return math.hypot(norm(lower[np.isfinite(lower)]), norm(upper[np.isfinite(upper)]))

    def contribution(value, lower, upper):
        positive, negative = value > 0, value < 0
        lo, hi = positive & np.isfinite(lower), negative & np.isfinite(upper)
        objective = float(lower[lo] @ value[lo] + upper[hi] @ value[hi])
        invalid = (positive & ~np.isfinite(lower)) | (negative & ~np.isfinite(upper))
        return objective, norm(value[invalid])

    c = np.asarray(problem.objective, dtype=np.float64)
    vl, vu = problem.variable_lower, problem.variable_upper
    cl, cu = problem.constraint_lower, problem.constraint_upper
    with np.errstate(over="ignore", invalid="ignore"):
        primal = float(c @ x + problem.offset)
        row_objective, bad_y = contribution(y, cl, cu)
        col_objective, bad_z = contribution(z, vl, vu)
        dual = row_objective + col_objective + problem.offset
        metrics.update(
            relative_primal_residual=norm(violation(problem.matrix @ x, cl, cu)) /
                                    (1.0 + bound_norm(cl, cu)),
            relative_dual_residual=norm(c - problem.matrix.T @ y - z) / (1.0 + norm(c)),
            relative_objective_gap=abs(primal - dual) / (1.0 + abs(primal) + abs(dual)),
            relative_variable_bound_violation=norm(violation(x, vl, vu)) /
                                              (1.0 + bound_norm(vl, vu)),
            relative_dual_bound_violation=math.hypot(bad_y, bad_z) / (1.0 + norm(c)),
            primal_objective=problem.objective_sign * primal,
            dual_objective=problem.objective_sign * dual,
            objective_without_constant=problem.objective_sign * (primal - problem.offset),
            dual_objective_without_constant=problem.objective_sign * (dual - problem.offset),
        )
    metrics["finite"] = all(math.isfinite(metrics[k]) for k in
                            (*RESIDUALS, "primal_objective", "dual_objective"))
    metrics["reference_objective"] = reference_objective
    metrics["reference_objective_relative_error"] = None
    reference_ok = True
    if reference_objective is not None:
        error = max(abs(metrics[key] - reference_objective) for key in
                    ("objective_without_constant", "dual_objective_without_constant")) / (
                        1.0 + abs(reference_objective))
        metrics["reference_objective_relative_error"] = error
        reference_ok = math.isfinite(error) and error <= tolerance
    metrics["verified"] = metrics["finite"] and reference_ok and all(
        metrics[key] <= tolerance for key in RESIDUALS)
    return metrics


def positive_float(value):
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("expected a finite positive number")
    return number


def positive_int(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("expected a positive integer")
    return number


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def case_name(path):
    name = Path(path).name
    if name.endswith(".gz"):
        name = name[:-3]
    return name.removesuffix(".mps")


def select_cases(args):
    if args.files:
        if args.manifest or args.instance:
            raise ValueError("positional MPS files cannot be combined with --manifest or --instance")
        cases = [{"name": case_name(p), "path": p.resolve()} for p in args.files]
    else:
        manifest = args.manifest or args.data_dir / "manifest.tsv"
        with manifest.open(newline="") as stream:
            rows = list(csv.DictReader(stream, delimiter="\t"))
        missing = set(args.instance) - {r["name"] for r in rows}
        if missing:
            raise ValueError(f"instances absent from manifest: {', '.join(sorted(missing))}")
        cases = []
        for row in rows:
            name = row["name"]
            if args.instance and name not in args.instance:
                continue
            path = args.data_dir / (name + ".mps.gz")
            if not path.exists():
                path = args.data_dir / (name + ".mps")
            cases.append({"name": name, "path": path.resolve(), "manifest": row})
    if not cases or len({r["name"] for r in cases}) != len(cases):
        raise ValueError("selection must contain at least one case and unique instance names")
    if any(Path(r["name"]).name != r["name"] or r["name"] in (".", "..") for r in cases):
        raise ValueError("instance names must be plain filenames")

    references = args.reference_objectives
    if references is None and not args.files:
        candidate = args.data_dir / "reference_objectives.tsv"
        if candidate.exists():
            references = candidate
    if references:
        with references.open(newline="") as stream:
            values = {r["name"]: float(r["optimal_objective"])
                      for r in csv.DictReader(stream, delimiter="\t")}
        for case in cases:
            value = values[case["name"]]
            if not math.isfinite(value):
                raise ValueError("reference objectives must be finite")
            case["reference_objective"] = value
    return cases


def json_safe(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    return value


def write_reports(prefix, report):
    prefix.parent.mkdir(parents=True, exist_ok=True)
    target = Path(str(prefix) + ".json")
    temporary = target.with_name(target.name + ".tmp")
    temporary.write_text(json.dumps(json_safe(report), indent=2, allow_nan=False) + "\n")
    temporary.replace(target)
    rows = []
    for result in report["results"]:
        row = {key: result.get(key) for key in ("name", "termination", "verified", "iterations", "error")}
        row.update(result.get("dimensions", {}))
        row.update({key + "_seconds": value for key, value in result["timing_seconds"].items()})
        row.update(result.get("original_model", {}))
        rows.append(row)
    target = Path(str(prefix) + ".csv")
    temporary = target.with_name(target.name + ".tmp")
    with temporary.open("w", newline="") as stream:
        fields = list(dict.fromkeys(key for row in rows for key in row))
        if fields:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    temporary.replace(target)


def main(adapter_factory, adapter_file, configure_parser=None):
    directory = Path(adapter_file).resolve().parent
    parser = argparse.ArgumentParser(description="Optional CPU LP reference benchmark; downloads happen only via make setup/run.")
    parser.add_argument("files", nargs="*", type=Path, help="MPS or MPS.gz files; otherwise use the manifest")
    parser.add_argument("--data-dir", "--data", type=Path, default=ROOT / "benchmarks/data/lpfeas")
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--instance", action="append", default=[])
    parser.add_argument("--reference-objectives", type=Path)
    parser.add_argument("--output-prefix", type=Path, default=directory / "results/run")
    parser.add_argument("--tolerance", type=positive_float, default=1e-6, help="independent original-model audit tolerance")
    parser.add_argument("--solver-tolerance", type=positive_float, help="native stopping target (default: --tolerance)")
    parser.add_argument("--time-limit", type=positive_float, default=1000.0)
    parser.add_argument("--iteration-limit", type=positive_int, default=2147483647)
    parser.add_argument("--threads", type=positive_int, default=1)
    parser.add_argument("--presolve", choices=("on", "off"), default="off")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--save-solutions", action="store_true", help="save audited x/y/reduced-cost arrays as NPZ files")
    parser.add_argument("--fail-on-validation", action="store_true")
    if configure_parser:
        configure_parser(parser)
    args = parser.parse_args()
    args.solver_tolerance = args.solver_tolerance or args.tolerance
    try:
        cases = select_cases(args)
        adapter = adapter_factory(args)
    except ImportError as error:
        parser.exit(2, f"{error}\nInstall this reference with: make -C {directory} setup\n"
                       f"Then use {directory}/.venv/bin/python {adapter_file}\n")
    except (ValueError, KeyError, OSError) as error:
        parser.error(str(error))

    report = {
        "schema": "mlxpdlp.reference.v1", "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "solver": adapter.name, "solver_version": adapter.version,
        "host": {"name": platform.node(), "os": platform.platform(), "machine": platform.machine(),
                 "logical_cpus": os.cpu_count()},
        "environment": {"python": sys.version, "executable": sys.executable,
                        "packages": dict(sorted((d.metadata["Name"], d.version)
                                                for d in importlib.metadata.distributions()))},
        "source_sha256": {p.name: digest(p) for p in
                          (Path(__file__), Path(adapter_file), directory / "requirements.txt")},
        "protocol": {"device": "cpu", "arithmetic_precision": "float64", "jobs": 1,
                     "threads": args.threads, "presolve": args.presolve == "on",
                     "tolerance": args.tolerance, "solver_tolerance": args.solver_tolerance,
                     "time_limit_seconds": args.time_limit, "iteration_limit": args.iteration_limit,
                     "audit": AUDIT, "integer_variables": "relaxed", "attempts": 1,
                     "timing_notes": "Wall time; parse includes model conversion. solve includes native presolve/postsolve. "
                                     "total includes parse, setup, solve, extraction and audit; excludes imports, hashing, report/NPZ writes. "
                                     "No external polishing, retries, or warm starts. Limits apply to the native solve."},
        "results": [],
    }
    write_reports(args.output_prefix, report)
    for case in cases:
        record = {"name": case["name"], "path": str(case["path"]), "termination": "ERROR",
                  "verified": False, "error": "", "iterations": 0, "original_model": {},
                  "timing_seconds": {}}
        start = time.perf_counter()
        try:
            problem, state = adapter.read(case["path"])
            parsed = time.perf_counter()
            record["timing_seconds"]["parse"] = parsed - start
            m, n = problem.matrix.shape
            record["dimensions"] = {"rows": m, "columns": n, "nonzeros": int(problem.matrix.nnz)}
            if "reference_objective" in case and "manifest" in case:
                # Netlib's manifest includes the objective row and its entries.
                import numpy as np
                expected = case["manifest"]
                observed = (m + 1, n, problem.matrix.nnz + np.count_nonzero(problem.objective))
                if observed != tuple(int(expected[k]) for k in ("rows", "columns", "nonzeros")):
                    raise ValueError(f"imported dimensions disagree with Netlib manifest: {observed}")
            record["native_parameters"] = adapter.configure(state)
            configured = time.perf_counter()
            record["timing_seconds"]["setup"] = configured - parsed
            solution = adapter.solve(state)
            solved = time.perf_counter()
            record["timing_seconds"]["solve"] = solved - configured
            record.update(termination=solution.termination, iterations=solution.iterations,
                          solver_details=solution.details)
            metrics = audit(problem, solution, args.tolerance, case.get("reference_objective"))
            record["original_model"] = metrics
            record["verified"] = bool(metrics["verified"])
            record["timing_seconds"]["verification"] = time.perf_counter() - solved
            record["timing_seconds"]["total"] = time.perf_counter() - start
            # Hash after timing so provenance work does not prewarm the input.
            record["input_sha256"] = digest(case["path"])
            if args.save_solutions and solution.has_solution:
                import numpy as np
                destination = Path(str(args.output_prefix) + "-solutions")
                destination.mkdir(parents=True, exist_ok=True)
                path = destination / (case["name"] + ".npz")
                np.savez_compressed(path, primal=solution.primal, dual=solution.dual,
                                    reduced_cost=solution.reduced_cost, objective_sign=problem.objective_sign)
                record["solution_file"] = str(path)
        except Exception as error:
            record.update(termination="ERROR", verified=False, error=f"{type(error).__name__}: {error}")
            record["timing_seconds"]["total"] = time.perf_counter() - start
        report["results"].append(record)
        write_reports(args.output_prefix, report)
        print(f"{case['name']}: {record['termination']} audit={'PASS' if record['verified'] else 'FAIL'} "
              f"total={record['timing_seconds']['total']:.4f}s {record['error']}", flush=True)
    return int(any(r["error"] or (args.fail_on_validation and not r["verified"]) for r in report["results"]))
