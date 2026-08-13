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

#include "presolve_adapter.h"

#include "PSLP_API.h"
#include "PSLP_sol.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <map>
#include <stdexcept>
#include <utility>
#include <vector>

namespace mlxpdlp::detail {

struct PresolveContext {
    Settings *settings = nullptr;
    Presolver *presolver = nullptr;
    std::vector<int> row_ptr;
    std::vector<int> col_ind;
    std::vector<double> values;
};

namespace {

void normalize_csr(PresolveContext &context, const HostProblemView &problem,
                   double matrix_zero_tol) {
    context.row_ptr.resize(static_cast<size_t>(problem.num_constraints) + 1);
    context.row_ptr[0] = 0;

    for (int row = 0; row < problem.num_constraints; ++row) {
        std::map<int, double> entries;
        for (int position = problem.row_ptr[row]; position < problem.row_ptr[row + 1]; ++position) {
            int column = problem.col_ind[position];
            if (column < 0 || column >= problem.num_variables) {
                throw std::invalid_argument("CSR column index is outside the problem dimensions");
            }
            entries[column] += problem.values[position];
        }
        for (const auto &[column, value] : entries) {
            if (std::fabs(value) <= matrix_zero_tol)
                continue;
            context.col_ind.push_back(column);
            context.values.push_back(value);
        }
        context.row_ptr[static_cast<size_t>(row) + 1] = static_cast<int>(context.values.size());
    }
}

} // namespace

const char *presolve_status_string(int status) {
    switch (status) {
    case UNCHANGED:
        return "UNCHANGED";
    case REDUCED:
        return "REDUCED";
    case INFEASIBLE:
        return "INFEASIBLE";
    case UNBNDORINFEAS:
        return "INFEASIBLE_OR_UNBOUNDED";
    default:
        return "UNKNOWN";
    }
}

PresolveTerminalKind classify_presolve_status(int status) {
    if (status == INFEASIBLE)
        return PresolveTerminalKind::primal_infeasible;
    if (status == UNBNDORINFEAS)
        return PresolveTerminalKind::infeasible_or_unbounded;
    return PresolveTerminalKind::none;
}

PresolveOutcome run_presolve(const HostProblemView &problem, double matrix_zero_tol,
                             const PresolveOptions &options, bool verbose) {
    auto start = std::chrono::steady_clock::now();
    auto *context = new PresolveContext();
    try {
        normalize_csr(*context, problem, std::max(matrix_zero_tol, 0.0));

        context->settings = default_settings();
        if (!context->settings)
            throw std::bad_alloc();
        context->settings->verbose = false;
        // Propagation materially reduces some structured LPFeas instances, but
        // its inverse map can amplify an approximate reduced dual certificate.
        // Keep it explicit so audited benchmark portfolios can try the
        // aggressive reduction while ordinary callers retain the safe default.
        context->settings->ston_cols = options.singleton_columns;
        context->settings->dton_eq = options.doubleton_equations;
        context->settings->parallel_rows = options.parallel_rows;
        context->settings->parallel_cols = options.parallel_columns;
        context->settings->dual_fix = options.dual_fix;
        context->settings->finite_bound_tightening =
            options.finite_bound_tightening;
        context->settings->primal_propagation = options.primal_propagation;

        context->presolver =
            new_presolver(context->values.data(), context->col_ind.data(), context->row_ptr.data(),
                          static_cast<size_t>(problem.num_constraints),
                          static_cast<size_t>(problem.num_variables), context->values.size(),
                          problem.constraint_lower_bound, problem.constraint_upper_bound,
                          problem.variable_lower_bound, problem.variable_upper_bound,
                          problem.objective, context->settings);
        if (!context->presolver)
            throw std::runtime_error("PSLP failed to initialize the presolver");

        PresolveStatus status = run_presolver(context->presolver);
        double elapsed =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();

        PresolvedProblem *reduced = context->presolver->reduced_prob;
        bool terminal_status = status == INFEASIBLE || status == UNBNDORINFEAS;
        bool solved = terminal_status || reduced->n == 0;

        HostProblemView reduced_view{
            static_cast<int>(reduced->n),
            static_cast<int>(reduced->m),
            static_cast<int>(reduced->nnz),
            reduced->Ap,
            reduced->Ai,
            reduced->Ax,
            reduced->lbs,
            reduced->ubs,
            reduced->lhs,
            reduced->rhs,
            reduced->c,
            problem.objective_constant + reduced->obj_offset,
        };

        if (verbose) {
            std::printf("\nPresolve (PSLP 0.0.8)\n");
            std::printf("  status: %s\n", presolve_status_string(status));
            std::printf("  reduced problem: %d rows, %d columns, %d nonzeros\n",
                        reduced_view.num_constraints, reduced_view.num_variables,
                        reduced_view.num_nonzeros);
            std::printf("  time: %.3g sec\n", elapsed);
        }

        return PresolveOutcome{context, reduced_view, solved, status, elapsed};
    } catch (...) {
        destroy_presolve(context);
        throw;
    }
}

PostsolveSolution postsolve(PresolveContext *context, const double *primal, const double *dual,
                            const double *reduced_cost,
                            const std::vector<double> &original_variable_lower_bound,
                            const std::vector<double> &original_variable_upper_bound) {
    if (!context || !context->presolver)
        throw std::invalid_argument("postsolve requires an initialized PSLP context");

    ::postsolve(context->presolver, primal, dual, reduced_cost);
    const Solution *solution = context->presolver->sol;

    PostsolveSolution output;
    output.primal.assign(solution->x, solution->x + solution->dim_x);
    output.dual.assign(solution->y, solution->y + solution->dim_y);
    output.reduced_cost.assign(solution->z, solution->z + solution->dim_x);

    for (size_t i = 0; i < output.reduced_cost.size(); ++i) {
        if (!std::isfinite(original_variable_lower_bound[i]))
            output.reduced_cost[i] = std::min(output.reduced_cost[i], 0.0);
        if (!std::isfinite(original_variable_upper_bound[i]))
            output.reduced_cost[i] = std::max(output.reduced_cost[i], 0.0);
    }
    return output;
}

void destroy_presolve(PresolveContext *context) {
    if (!context)
        return;
    if (context->presolver)
        free_presolver(context->presolver);
    if (context->settings)
        free_settings(context->settings);
    delete context;
}

} // namespace mlxpdlp::detail
