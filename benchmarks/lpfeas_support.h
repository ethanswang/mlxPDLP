/*
Copyright 2026 Ethan Wang <ethanshurui.wang@gmail.com>

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

#pragma once

#include "mlxPDLP/mps_loader.h"
#include "mlxPDLP/solver.h"

namespace mlxpdlp::benchmark {

struct ValidationMetrics {
    bool dimensions_match = false;
    bool finite = false;
    bool dual_bounds_valid = false;

    double primal_objective = 0.0;
    double dual_objective = 0.0;
    double absolute_primal_residual = 0.0;
    double relative_primal_residual = 0.0;
    double absolute_dual_residual = 0.0;
    double relative_dual_residual = 0.0;
    double objective_gap = 0.0;
    double relative_objective_gap = 0.0;
    double absolute_variable_bound_violation = 0.0;
    double relative_variable_bound_violation = 0.0;
    double absolute_dual_bound_violation = 0.0;
    double relative_dual_bound_violation = 0.0;

    bool satisfies(double tolerance) const;
};

// Recomputes all certificate metrics on the original, unscaled model using
// host double precision. objective/objective_constant must be in the
// minimization convention passed to MlxPdlpSolver.
ValidationMetrics validate_original_problem(const mlxpdlp_mps_problem_t &problem,
                                            const mlxpdlp_result_t &result,
                                            const double *objective,
                                            double objective_constant);

const char *termination_reason_name(termination_reason_t reason);

} // namespace mlxpdlp::benchmark
