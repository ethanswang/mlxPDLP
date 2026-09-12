#!/usr/bin/env python3
# Copyright 2026 Ethan Wang <ethanshurui.wang@gmail.com>
# SPDX-License-Identifier: Apache-2.0
"""Opt-in trace recording and independent replay through the existing Python API.

Recording never invents ready times or dependencies; the application supplies
them at its scheduler boundary. This is benchmark support, not a batch API.
"""

from __future__ import annotations

import argparse
import enum
import math
from numbers import Integral, Real
from pathlib import Path
import time
from typing import Any

from batch_lp_trace import (
    DATA_FIELDS, MatrixRegistry, SCHEMA, TraceError, read_trace, require,
    validate_data, validate_trace, write_json,
)


PARAMETER_ALIASES = {"tolerance", "time_limit_seconds", "iteration_limit"}
MATRIX_FIELDS = ("num_variables", "num_constraints", "row_ptr", "col_indices", "values")


def wire(value: Any) -> Any:
    """Make owned strict JSON data, preserving every finite FP64 value."""
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, enum.Enum):
        return wire(value.value)
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, Real):
        value = float(value)
        if math.isnan(value):
            raise TraceError("NaN cannot be captured as LP input or solver metadata")
        return ("+inf" if value > 0 else "-inf") if math.isinf(value) else value
    if isinstance(value, dict):
        return {key: wire(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [wire(item) for item in value]
    if hasattr(value, "tolist"):
        return wire(value.tolist())
    raise TraceError(f"unsupported snapshot type {type(value).__name__}")


def snapshot_parameters(parameters: Any, *, nested: bool = False) -> dict[str, Any]:
    snapshot = {}
    for name in dir(parameters):
        if name.startswith("_") or (not nested and name in PARAMETER_ALIASES):
            continue
        value = getattr(parameters, name)
        if callable(value):
            continue
        snapshot[name] = (snapshot_parameters(value, nested=True)
                          if name in ("termination_criteria", "restart_params") else wire(value))
    return snapshot


def restore_parameters(snapshot: dict[str, Any], parameters: Any, *, nested: bool = False) -> Any:
    current = snapshot_parameters(parameters, nested=nested)
    require(set(snapshot) == set(current), "parameters",
            f"binding settings differ: missing {sorted(set(current) - set(snapshot))}; "
            f"unknown {sorted(set(snapshot) - set(current))}. Use the captured solver build.")
    for name, value in snapshot.items():
        existing = getattr(parameters, name)
        if isinstance(value, dict):
            restore_parameters(value, existing, nested=True)
            # Some bindings return nested parameter structs by value.
            setattr(parameters, name, existing)
        else:
            if value in ("+inf", "-inf"):
                value = math.inf if value == "+inf" else -math.inf
            if isinstance(existing, enum.Enum):
                value = type(existing)(value)
            setattr(parameters, name, value)
    return parameters


def solver_arguments(matrix: dict[str, Any], data: dict[str, Any], parameters: Any,
                     device: str) -> dict[str, Any]:
    import numpy as np

    args = {"num_variables": matrix["num_variables"], "num_constraints": matrix["num_constraints"],
            "parameters": parameters, "device": device}
    for name in ("row_ptr", "col_indices", "values"):
        args[name] = np.asarray(matrix[name], dtype=np.float64 if name == "values" else np.int32)
    for name in DATA_FIELDS:
        value = data[name]
        if name == "objective_constant" or value is None:
            args[name] = value
        else:
            args[name] = np.asarray(value, dtype=np.float64)
    return args


def result_snapshot(result: Any) -> dict[str, Any]:
    # Results with numerical failures can legitimately contain NaN. Preserve it
    # as a string in replay output, which is never accepted as an input trace.
    def result_wire(value: Any) -> Any:
        if hasattr(value, "tolist"):
            value = value.tolist()
        if isinstance(value, (list, tuple)):
            return [result_wire(item) for item in value]
        if isinstance(value, Real) and math.isnan(float(value)):
            return "nan"
        return wire(value)

    fields = ("primal_solution", "dual_solution", "reduced_cost", "termination_reason_name",
              "total_count", "primal_objective_value", "dual_objective_value",
              "relative_primal_residual", "relative_dual_residual", "relative_objective_gap",
              "rescaling_time_sec", "presolve_time", "cumulative_time_sec",
              "host_double_polishing_time", "feasibility_polishing_time")
    return {name: result_wire(getattr(result, name)) for name in fields}


class TraceRecorder:
    """One-process recorder; instrument all relevant LPs, including fallbacks.

    Call ready() only when all request data is available, including warm starts.
    Keep the application's existing dispatch order when calling solve(). A
    solver-entry-only observer should use ready_ns=None and depends_on=None.
    It can instead use observe() around the application's existing solver calls.
    """

    def __init__(self, *, source: dict[str, Any], latency_limit_ns: int | str | None,
                 memory_limit_bytes: int | str | None, scheduling_policy: str) -> None:
        self._epoch = time.perf_counter_ns()
        self._registry = MatrixRegistry()
        self._requests: dict[str, dict[str, Any]] = {}
        self._overhead_ns = 0
        self.source = wire(source)
        self.application = {"latency_limit_ns": latency_limit_ns, "memory_limit_bytes": memory_limit_bytes,
                            "scheduling_policy": scheduling_policy}

    def now_ns(self) -> int:
        return time.perf_counter_ns() - self._epoch

    def ready(self, request_id: str, family: str, arguments: dict[str, Any], *,
              ready_ns: int | None, depends_on: list[str] | None) -> None:
        start = self.now_ns()
        require(request_id not in self._requests, request_id, "duplicate request id")
        import mlxpdlp

        allowed = set(MATRIX_FIELDS) | set(DATA_FIELDS) | {"parameters", "device"}
        require(not set(arguments).difference(allowed), request_id, "unknown solver argument")
        matrix = wire({name: arguments[name] for name in MATRIX_FIELDS})
        matrix_id = self._registry.add(matrix)
        n, m = matrix["num_variables"], matrix["num_constraints"]
        data = {"objective": arguments["objective"], "objective_constant": arguments.get("objective_constant", 0.0)}
        for prefix, length in (("variable", n), ("constraint", m)):
            for side, default in (("lower", -math.inf), ("upper", math.inf)):
                name = f"{prefix}_{side}_bounds"
                value = arguments.get(name)
                data[name] = [default] * length if value is None else value
        for name in ("primal_start", "dual_start", "reduced_cost_start"):
            data[name] = arguments.get(name)
        data = wire(data)
        validate_data(data, matrix, f"request {request_id}.data")
        parameters = arguments.get("parameters")
        if parameters is None:
            parameters = mlxpdlp.Parameters()
        self._requests[request_id] = {
            "id": request_id, "family": family, "matrix_id": matrix_id,
            "data": data, "parameters": snapshot_parameters(parameters),
            "device": arguments.get("device", "cpu"), "ready_ns": ready_ns,
            "depends_on": wire(depends_on),
        }
        self._overhead_ns += self.now_ns() - start

    def observe(self, request_id: str, *, start_ns: int, end_ns: int,
                construct_ns: int, solve_ns: int, result: Any) -> None:
        start = self.now_ns()
        req = self._requests[request_id]
        require("end_ns" not in req, request_id, "request already completed")
        req.update(start_ns=start_ns, end_ns=end_ns, construct_ns=construct_ns, solve_ns=solve_ns,
                   rescale_ns=round(result.rescaling_time_sec * 1e9),
                   iterations=int(result.total_count), termination=str(result.termination_reason_name))
        self._overhead_ns += self.now_ns() - start

    def solve(self, request_id: str) -> Any:
        """Run an already-ready snapshot using the unchanged single-LP API."""
        import mlxpdlp

        req = self._requests[request_id]
        require("end_ns" not in req, request_id, "request already completed")
        preparation = self.now_ns()
        params = restore_parameters(req["parameters"], mlxpdlp.Parameters())
        args = solver_arguments(self._registry.matrices[req["matrix_id"]], req["data"], params, req["device"])
        self._overhead_ns += self.now_ns() - preparation
        start = self.now_ns()
        try:
            solver = mlxpdlp.Solver(**args)
            constructed = self.now_ns()
            result = solver.solve()
        except Exception as exc:
            end = self.now_ns()
            req.update(start_ns=start, end_ns=end, construct_ns=None, solve_ns=None, rescale_ns=None,
                       iterations=None, termination=f"ERROR: {type(exc).__name__}: {exc}")
            raise
        end = self.now_ns()
        self.observe(request_id, start_ns=start, end_ns=end, construct_ns=constructed - start,
                     solve_ns=end - constructed, result=result)
        return result

    def finish(self, *, wall_time_ns: int | None = None) -> dict[str, Any]:
        for rid, req in self._requests.items():
            require("end_ns" in req, rid, "unfinished capture; do not silently drop queued LPs")
        trace = {"schema": SCHEMA, "source": wire(self.source),
                 "application": {**self.application,
                                 "wall_time_ns": self.now_ns() if wall_time_ns is None else wall_time_ns,
                                 "instrumentation_ns": self._overhead_ns},
                 "matrices": wire(self._registry.matrices), "requests": wire(list(self._requests.values()))}
        validate_trace(trace)
        return trace


def replay(trace: dict[str, Any], *, device: str | None = None) -> dict[str, Any]:
    """Replay in original dispatch order; timings never replace application observations."""
    validate_trace(trace)
    import mlxpdlp

    # Validate every parameter snapshot before the first solver/kernel launch.
    parameters = [restore_parameters(req["parameters"], mlxpdlp.Parameters()) for req in trace["requests"]]
    records = []
    start = time.perf_counter_ns()
    for i in sorted(range(len(parameters)), key=lambda i: (trace["requests"][i]["start_ns"], i)):
        req = trace["requests"][i]
        selected = device or req["device"]
        began = time.perf_counter_ns()
        args = solver_arguments(trace["matrices"][req["matrix_id"]], req["data"], parameters[i], selected)
        packed = time.perf_counter_ns()
        solver = mlxpdlp.Solver(**args)
        constructed = time.perf_counter_ns()
        result = solver.solve()
        ended = time.perf_counter_ns()
        records.append({"input_index": i, "id": req["id"], "device": selected,
                        "execution_mode": "independent", "independently_audited": False,
                        "packing_ns": packed - began, "construct_ns": constructed - packed,
                        "solve_ns": ended - constructed, "result": result_snapshot(result)})
    return {"schema": "mlxpdlp.batch_lp.replay.v1", "execution_mode": "independent",
            "schedule": "serial original-dispatch order; observed readiness is not simulated",
            "qualification_evidence": False, "source": trace["source"], "device_override": device,
            "wall_time_ns": time.perf_counter_ns() - start,
            "results": sorted(records, key=lambda item: item["input_index"])}


def main() -> int:
    parser = argparse.ArgumentParser(description="Replay a trace through the existing independent solver (no batching).")
    parser.add_argument("trace", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "gpu"), help="explicitly override the recorded device")
    args = parser.parse_args()
    try:
        result = replay(read_trace(args.trace), device=args.device)
        write_json(args.output, result)
    except (OSError, ValueError, RuntimeError) as exc:
        parser.error(str(exc))
    print(f"Replayed {len(result['results'])} LPs independently; this is not a qualification or throughput result.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
