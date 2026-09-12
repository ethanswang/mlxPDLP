#!/usr/bin/env python3
# Copyright 2026 Ethan Wang <ethanshurui.wang@gmail.com>
# SPDX-License-Identifier: Apache-2.0
"""Measure explicitly synthetic same-matrix families derived from local MPS files.

The experiment's scheduler owns all members before dispatch. This is a measured
derived workload, never a capture of alpha-beta-CROWN or a whole-application gain.
"""
from __future__ import annotations

import argparse
import hashlib
import platform
from pathlib import Path
import subprocess
import sys

from batch_lp_capture import TraceRecorder
from batch_lp_trace import analyze, write_csv, write_json


ROOT = Path(__file__).resolve().parents[1]


def revision(path: Path) -> str:
    return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cases", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--width", type=int, default=8)
    parser.add_argument("--device", choices=("cpu", "gpu"), default="gpu")
    parser.add_argument("--time-limit", type=float, default=5.0,
                        help="experiment solve cap, distinct from unconstrained caller latency")
    parser.add_argument("--iteration-limit", type=int, default=50000)
    parser.add_argument("--no-presolve", action="store_true")
    args = parser.parse_args()
    if args.width < 1:
        parser.error("--width must be positive")
    import mlxpdlp
    import numpy as np

    params = mlxpdlp.Parameters()
    params.presolve = not args.no_presolve
    params.verbose = False
    params.tolerance = 1e-4
    params.time_limit_seconds = args.time_limit
    params.iteration_limit = args.iteration_limit
    source = {
        "kind": "synthetic", "representative": False, "timings_measured": True,
        "application": "Netlib/LPfeas derived objective family",
        "workload_id": "mps-objective-scale-offset-v1",
        "selection": "User-authorized 2026-09-10; positive objective scaling and offsets; all data ready before serial dispatch",
        "scheduling_unchanged": True, "complete_capture": True, "settings_complete": True,
        "revision": revision(ROOT), "solver_revision": revision(ROOT),
        "mlx_revision": revision(ROOT.parent / "mlx"),
        "hardware": subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip(),
        "build": "Release; staged current Python binding; " + platform.platform(),
        "command": " ".join(sys.argv),
        "case_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in args.cases},
        "caller_limits_authority": "User: No info about per-LP latency and resident-memory info, assuming none.",
    }
    recorder = TraceRecorder(source=source, latency_limit_ns="unlimited",
                             memory_limit_bytes="unlimited",
                             scheduling_policy="serial FIFO; entire derived family ready before its first solve")
    for path in args.cases:
        problem = mlxpdlp.load_mps(str(path))
        family = path.name.removesuffix(".gz").removesuffix(".mps")
        base = dict(num_variables=problem.num_variables, num_constraints=problem.num_constraints,
                    row_ptr=problem.row_ptr, col_indices=problem.col_ind, values=problem.values,
                    variable_lower_bounds=problem.variable_lb, variable_upper_bounds=problem.variable_ub,
                    constraint_lower_bounds=problem.constraint_lb, constraint_upper_bounds=problem.constraint_ub,
                    parameters=params, device=args.device)
        members = [dict(base, objective=np.asarray(problem.objective) * (1 + i / args.width),
                        objective_constant=float(problem.objective_constant) + i / 8)
                   for i in range(args.width)]
        ready = recorder.now_ns()
        for i, member in enumerate(members):
            recorder.ready(f"{family}/{i}", family, member, ready_ns=ready, depends_on=[])
        for i in range(args.width):
            result = recorder.solve(f"{family}/{i}")
            print(f"{family}/{i}: {result.termination_reason_name}, {result.total_count} iterations", flush=True)
    trace = recorder.finish()
    report = analyze(trace, allow_synthetic=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, trace)
    write_json(args.output.with_suffix(".qualification.json"), report)
    write_csv(args.output.with_suffix(".qualification.csv"), report)
    print(f"{report['decision']}: {report['total_requests']} measured LPs", flush=True)
    return 0 if report["decision"] == "go" else 2


if __name__ == "__main__":
    raise SystemExit(main())
