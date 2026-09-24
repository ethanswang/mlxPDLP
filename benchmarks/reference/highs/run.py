#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""HiGHS CPU adapter; independent of OR-Tools and the mlxPDLP Python package."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import Problem, Solution, main


class Highs:
    def __init__(self, args):
        import highspy
        import numpy as np
        from scipy import sparse

        self.args, self.api, self.np, self.sparse = args, highspy, np, sparse
        self.name = f"HiGHS {args.solver}"
        self.version = highspy.Highs().version()

    def check(self, status, action):
        if status == self.api.HighsStatus.kError:
            raise ValueError(f"HiGHS failed to {action}")

    def read(self, path):
        h = self.api.Highs()
        h.setOptionValue("output_flag", self.args.verbose)
        self.check(h.readModel(str(path)), f"read {path}")
        if h.getHessianNumNz():
            raise ValueError("this benchmark accepts linear objectives only")
        lp = h.getLp()
        a = lp.a_matrix_
        constructor = self.sparse.csc_matrix if a.format_ == self.api.MatrixFormat.kColwise else self.sparse.csr_matrix
        matrix = constructor((a.value_, a.index_, a.start_), shape=(lp.num_row_, lp.num_col_))
        sign = -1.0 if lp.sense_ == self.api.ObjSense.kMaximize else 1.0
        vector = lambda values: self.np.asarray(values, dtype=self.np.float64)
        problem = Problem(matrix, sign * vector(lp.col_cost_), vector(lp.col_lower_),
                          vector(lp.col_upper_), vector(lp.row_lower_), vector(lp.row_upper_),
                          sign * lp.offset_, sign)
        if sign < 0:
            # Normalize MAX explicitly: HiPDLP 1.15.1 otherwise optimizes the
            # unnegated costs. The audit restores the original objective sense.
            self.check(h.changeObjectiveSense(self.api.ObjSense.kMinimize), "normalize objective sense")
            self.check(h.changeColsCost(lp.num_col_, self.np.arange(lp.num_col_, dtype=self.np.int32),
                                       problem.objective), "normalize objective costs")
            self.check(h.changeObjectiveOffset(problem.offset), "normalize objective offset")
        if lp.integrality_:
            # MIPLIB MPS files in LPfeas are benchmarked as continuous relaxations.
            self.check(h.changeColsIntegrality(lp.num_col_, self.np.arange(lp.num_col_, dtype=self.np.int32),
                       self.np.zeros(lp.num_col_, dtype=self.np.uint8)), "relax integrality")
        return problem, h

    def configure(self, state):
        h = state
        self.settings = {"solver": self.args.solver, "threads": self.args.threads,
                         "presolve": self.args.presolve, "time_limit": self.args.time_limit,
                         "pdlp_iteration_limit": self.args.iteration_limit,
                         "simplex_iteration_limit": self.args.iteration_limit,
                         "ipm_iteration_limit": self.args.iteration_limit,
                         "pdlp_optimality_tolerance": self.args.solver_tolerance,
                         "kkt_tolerance": self.args.solver_tolerance,
                         "ipm_optimality_tolerance": self.args.solver_tolerance,
                         "run_crossover": "off", "random_seed": 0}
        if self.args.solver == "hipdlp":
            self.settings["pdlp_step_size_strategy"] = 3  # HiPDLP's PID strategy.
        for key, value in self.settings.items():
            self.check(h.setOptionValue(key, value), f"set {key}={value}")
        return {"options": self.settings, "objective_normalization": "minimization"}

    def solve(self, state):
        h = state
        self.check(h.run(), "solve")
        status = h.modelStatusToString(h.getModelStatus())
        info, result = h.getInfo(), h.getSolution()
        iterations = (info.pdlp_iteration_count if self.args.solver in ("hipdlp", "pdlp") else
                      info.simplex_iteration_count if self.args.solver == "simplex" else info.ipm_iteration_count)
        details = {key: getattr(info, key) for key in
                   ("objective_function_value", "max_primal_infeasibility", "max_dual_infeasibility",
                    "max_relative_primal_infeasibility", "max_relative_dual_infeasibility",
                    "primal_dual_objective_error", "pdlp_iteration_count", "simplex_iteration_count",
                    "ipm_iteration_count", "crossover_iteration_count")}
        details.update(native_status=status, solver_reported_seconds=h.getRunTime())
        has_solution = result.value_valid and result.dual_valid and status not in (
            "Infeasible", "Unbounded", "Unbounded or infeasible", "Model error", "Solve error")
        return Solution(self.np.asarray(result.col_value), self.np.asarray(result.row_dual),
                        self.np.asarray(result.col_dual), status.upper().replace(" ", "_"),
                        iterations, details, has_solution)


def options(parser):
    parser.add_argument("--solver", choices=("hipdlp", "pdlp", "simplex", "ipm"), default="hipdlp",
                        help="hipdlp=native PDHG; pdlp=cuPDLP-C; simplex/ipm=additional accuracy references")


if __name__ == "__main__":
    raise SystemExit(main(Highs, __file__, options))
