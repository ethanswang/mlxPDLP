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

#include <vector>

namespace mlxpdlp::detail {

struct HostProblemView {
    int num_variables;
    int num_constraints;
    int num_nonzeros;
    const int *row_ptr;
    const int *col_ind;
    const double *values;
    const double *variable_lower_bound;
    const double *variable_upper_bound;
    const double *constraint_lower_bound;
    const double *constraint_upper_bound;
    const double *objective;
    double objective_constant;
};

struct PresolveContext;

struct PresolveOutcome {
    PresolveContext *context;
    HostProblemView reduced_problem;
    bool solved;
    int status;
    double elapsed_seconds;
};

struct PresolveOptions {
    bool singleton_columns = false;
    bool doubleton_equations = true;
    bool parallel_rows = true;
    bool parallel_columns = true;
    bool dual_fix = true;
    bool finite_bound_tightening = true;
    bool primal_propagation = false;
};

struct PostsolveSolution {
    std::vector<double> primal;
    std::vector<double> dual;
    std::vector<double> reduced_cost;
};

enum class PresolveTerminalKind {
    none,
    primal_infeasible,
    infeasible_or_unbounded,
};

PresolveOutcome run_presolve(const HostProblemView &problem, double matrix_zero_tol,
                             const PresolveOptions &options, bool verbose);
PostsolveSolution postsolve(PresolveContext *context, const double *primal, const double *dual,
                            const double *reduced_cost,
                            const std::vector<double> &original_variable_lower_bound,
                            const std::vector<double> &original_variable_upper_bound);
void destroy_presolve(PresolveContext *context);
const char *presolve_status_string(int status);
PresolveTerminalKind classify_presolve_status(int status);

} // namespace mlxpdlp::detail
