#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Google OR-Tools PDLP CPU adapter. Run --help without installing dependencies."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import Problem, Solution, main


class OrToolsPdlp:
    def __init__(self, args):
        import ortools
        import numpy as np
        from scipy import sparse
        from google.protobuf.json_format import MessageToDict
        from ortools.linear_solver.python import model_builder
        from ortools.pdlp.python import pdlp
        from ortools.pdlp import solvers_pb2, solve_log_pb2

        self.args, self.np, self.model_builder = args, np, model_builder
        # Load sparse conversion and log serialization before per-case timing.
        self.sparse, self.log_to_dict = sparse, MessageToDict
        self.pdlp, self.pb, self.log_pb = pdlp, solvers_pb2, solve_log_pb2
        self.name, self.version = "OR-Tools PDLP", ortools.__version__

    def read(self, path):
        # The checked reader reports malformed MPS instead of aborting the process
        # like read_quadratic_program_or_die(). Integer annotations are relaxed.
        model = self.model_builder.Model()
        if not model.import_from_mps_file(str(path)):
            raise ValueError(f"cannot read MPS: {path}")
        qp = self.pdlp.qp_from_mpmodel_proto(model.export_to_proto(), relax_integer_variables=True)
        if not self.pdlp.is_linear_program(qp):
            raise ValueError("this benchmark accepts linear objectives only")
        problem = Problem(self.sparse.csc_matrix(qp.constraint_matrix, copy=False), qp.objective_vector,
                          qp.variable_lower_bounds, qp.variable_upper_bounds,
                          qp.constraint_lower_bounds, qp.constraint_upper_bounds,
                          qp.objective_offset, qp.objective_scaling_factor)
        return problem, qp

    def configure(self, state):
        p = self.pb.PrimalDualHybridGradientParams()
        p.num_threads = self.args.threads
        p.verbosity_level = 1 if self.args.verbose else 0
        p.presolve_options.use_glop = self.args.presolve == "on"
        p.use_feasibility_polishing = self.args.feasibility_polishing
        p.termination_criteria.optimality_norm = self.pb.OPTIMALITY_NORM_L2
        p.termination_criteria.time_sec_limit = self.args.time_limit
        p.termination_criteria.iteration_limit = self.args.iteration_limit
        criteria = p.termination_criteria.simple_optimality_criteria
        criteria.eps_optimal_absolute = self.args.solver_tolerance
        criteria.eps_optimal_relative = self.args.solver_tolerance
        self.params = p
        return {"text_proto": str(p)}

    def solve(self, qp):
        result = self.pdlp.primal_dual_hybrid_gradient(qp, self.params)
        log = result.solve_log
        status = self.log_pb.TerminationReason.Name(log.termination_reason).removeprefix("TERMINATION_REASON_")
        has_solution = status in {"OPTIMAL", "TIME_LIMIT", "ITERATION_LIMIT", "KKT_MATRIX_PASS_LIMIT", "NUMERICAL_ERROR"}
        # These dual vectors use the normalized minimization objective even for MAX.
        return Solution(result.primal_solution, result.dual_solution, result.reduced_costs,
                        status, log.iteration_count,
                        self.log_to_dict(log, preserving_proto_field_name=True), has_solution)


def options(parser):
    parser.add_argument("--feasibility-polishing", action="store_true",
                        help="enable OR-Tools' native feasibility polishing (default: off)")


if __name__ == "__main__":
    raise SystemExit(main(OrToolsPdlp, __file__, options))
