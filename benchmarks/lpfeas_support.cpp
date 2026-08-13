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

#include "lpfeas_support.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>

namespace mlxpdlp::benchmark {
namespace {

double distance_to_interval(double value, double lower, double upper) {
    if (value < lower)
        return lower - value;
    if (value > upper)
        return value - upper;
    return 0.0;
}

void add_square(long double &sum, double value) {
    const long double wide = value;
    sum += wide * wide;
}

double root(long double sum) {
    return std::sqrt(static_cast<double>(sum));
}

bool all_finite(const double *values, int count) {
    if (!values)
        return false;
    for (int i = 0; i < count; ++i) {
        if (!std::isfinite(values[i]))
            return false;
    }
    return true;
}

} // namespace

bool ValidationMetrics::satisfies(double tolerance) const {
    return dimensions_match && finite && relative_primal_residual <= tolerance &&
           relative_dual_residual <= tolerance &&
           relative_objective_gap <= tolerance &&
           relative_variable_bound_violation <= tolerance &&
           relative_dual_bound_violation <= tolerance;
}

ValidationMetrics validate_original_problem(const mlxpdlp_mps_problem_t &problem,
                                            const mlxpdlp_result_t &result,
                                            const double *objective,
                                            double objective_constant) {
    ValidationMetrics metrics;
    metrics.dimensions_match = result.num_variables == problem.num_variables &&
                               result.num_constraints == problem.num_constraints;
    if (!metrics.dimensions_match || !objective || !problem.row_ptr || !problem.col_ind ||
        !problem.values || !problem.variable_lb || !problem.variable_ub ||
        !problem.constraint_lb || !problem.constraint_ub ||
        !all_finite(result.primal_solution, problem.num_variables) ||
        !all_finite(result.dual_solution, problem.num_constraints) ||
        !all_finite(result.reduced_cost, problem.num_variables)) {
        return metrics;
    }

    const int m = problem.num_constraints;
    const int n = problem.num_variables;
    std::vector<double> ax(static_cast<size_t>(m), 0.0);
    std::vector<double> aty(static_cast<size_t>(n), 0.0);

    for (int row = 0; row < m; ++row) {
        long double row_value = 0.0;
        const double y = result.dual_solution[row];
        for (int entry = problem.row_ptr[row]; entry < problem.row_ptr[row + 1]; ++entry) {
            const int column = problem.col_ind[entry];
            if (column < 0 || column >= n || !std::isfinite(problem.values[entry]))
                return metrics;
            const double coefficient = problem.values[entry];
            row_value += static_cast<long double>(coefficient) * result.primal_solution[column];
            aty[static_cast<size_t>(column)] += coefficient * y;
        }
        ax[static_cast<size_t>(row)] = static_cast<double>(row_value);
    }

    long double primal_residual_sq = 0.0;
    long double constraint_bound_norm_sq = 0.0;
    long double variable_violation_sq = 0.0;
    long double variable_bound_norm_sq = 0.0;
    long double dual_residual_sq = 0.0;
    long double objective_norm_sq = 0.0;
    long double dual_bound_violation_sq = 0.0;
    long double primal_objective = objective_constant;
    long double dual_objective = objective_constant;

    for (int row = 0; row < m; ++row) {
        const double lower = problem.constraint_lb[row];
        const double upper = problem.constraint_ub[row];
        add_square(primal_residual_sq,
                   distance_to_interval(ax[static_cast<size_t>(row)], lower, upper));
        if (std::isfinite(lower))
            add_square(constraint_bound_norm_sq, lower);
        if (std::isfinite(upper))
            add_square(constraint_bound_norm_sq, upper);

        const double y = result.dual_solution[row];
        if (y > 0.0) {
            if (std::isfinite(lower))
                dual_objective += static_cast<long double>(lower) * y;
            else
                add_square(dual_bound_violation_sq, y);
        } else if (y < 0.0) {
            if (std::isfinite(upper))
                dual_objective += static_cast<long double>(upper) * y;
            else
                add_square(dual_bound_violation_sq, y);
        }
    }

    for (int column = 0; column < n; ++column) {
        const double x = result.primal_solution[column];
        const double z = result.reduced_cost[column];
        const double lower = problem.variable_lb[column];
        const double upper = problem.variable_ub[column];
        const double coefficient = objective[column];
        if (!std::isfinite(coefficient))
            return metrics;

        add_square(variable_violation_sq, distance_to_interval(x, lower, upper));
        if (std::isfinite(lower))
            add_square(variable_bound_norm_sq, lower);
        if (std::isfinite(upper))
            add_square(variable_bound_norm_sq, upper);

        primal_objective += static_cast<long double>(coefficient) * x;
        add_square(objective_norm_sq, coefficient);
        add_square(dual_residual_sq, coefficient - aty[static_cast<size_t>(column)] - z);

        if (z > 0.0) {
            if (std::isfinite(lower))
                dual_objective += static_cast<long double>(lower) * z;
            else
                add_square(dual_bound_violation_sq, z);
        } else if (z < 0.0) {
            if (std::isfinite(upper))
                dual_objective += static_cast<long double>(upper) * z;
            else
                add_square(dual_bound_violation_sq, z);
        }
    }

    metrics.primal_objective = static_cast<double>(primal_objective);
    metrics.dual_objective = static_cast<double>(dual_objective);
    metrics.absolute_primal_residual = root(primal_residual_sq);
    metrics.absolute_dual_residual = root(dual_residual_sq);
    metrics.absolute_variable_bound_violation = root(variable_violation_sq);
    metrics.absolute_dual_bound_violation = root(dual_bound_violation_sq);
    metrics.objective_gap = std::fabs(metrics.primal_objective - metrics.dual_objective);

    metrics.relative_primal_residual =
        metrics.absolute_primal_residual / (1.0 + root(constraint_bound_norm_sq));
    metrics.relative_dual_residual =
        metrics.absolute_dual_residual / (1.0 + root(objective_norm_sq));
    metrics.relative_variable_bound_violation =
        metrics.absolute_variable_bound_violation / (1.0 + root(variable_bound_norm_sq));
    metrics.relative_dual_bound_violation =
        metrics.absolute_dual_bound_violation / (1.0 + root(objective_norm_sq));
    metrics.relative_objective_gap =
        metrics.objective_gap /
        (1.0 + std::fabs(metrics.primal_objective) + std::fabs(metrics.dual_objective));
    metrics.dual_bounds_valid = metrics.absolute_dual_bound_violation == 0.0;
    metrics.finite = std::isfinite(metrics.primal_objective) &&
                     std::isfinite(metrics.dual_objective) &&
                     std::isfinite(metrics.relative_primal_residual) &&
                     std::isfinite(metrics.relative_dual_residual) &&
                     std::isfinite(metrics.relative_objective_gap) &&
                     std::isfinite(metrics.relative_variable_bound_violation) &&
                     std::isfinite(metrics.relative_dual_bound_violation);
    return metrics;
}

const char *termination_reason_name(termination_reason_t reason) {
    switch (reason) {
    case TERMINATION_REASON_OPTIMAL:
        return "OPTIMAL";
    case TERMINATION_REASON_PRIMAL_INFEASIBLE:
        return "PRIMAL_INFEASIBLE";
    case TERMINATION_REASON_DUAL_INFEASIBLE:
        return "DUAL_INFEASIBLE";
    case TERMINATION_REASON_INFEASIBLE_OR_UNBOUNDED:
        return "INFEASIBLE_OR_UNBOUNDED";
    case TERMINATION_REASON_TIME_LIMIT:
        return "TIME_LIMIT";
    case TERMINATION_REASON_ITERATION_LIMIT:
        return "ITERATION_LIMIT";
    case TERMINATION_REASON_FEAS_POLISH_SUCCESS:
        return "FEAS_POLISH_SUCCESS";
    case TERMINATION_REASON_HOST_DOUBLE_HANDOFF:
        return "HOST_DOUBLE_HANDOFF";
    case TERMINATION_REASON_UNSPECIFIED:
        return "UNSPECIFIED";
    }
    return "UNKNOWN";
}

} // namespace mlxpdlp::benchmark
