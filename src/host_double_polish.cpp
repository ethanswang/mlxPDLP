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

#include "mlxPDLP/solver.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <limits>
#include <numeric>
#include <vector>

namespace mlxpdlp {
namespace {

using Clock = std::chrono::steady_clock;

double seconds_since(Clock::time_point start) {
    return std::chrono::duration<double>(Clock::now() - start).count();
}

double finite_scaled(double value, double scale) {
    return std::isfinite(value) ? value * scale : value;
}

double vector_norm(const std::vector<double> &values, norm_type_t norm_type) {
    if (norm_type == NORM_TYPE_L_INF) {
        double maximum = 0.0;
        for (double value : values)
            maximum = std::max(maximum, std::fabs(value));
        return maximum;
    }
    long double sum = 0.0L;
    for (double value : values)
        sum += static_cast<long double>(value) * value;
    return std::sqrt(static_cast<double>(sum));
}

double distance_norm(const std::vector<double> &lhs, const std::vector<double> &rhs) {
    long double sum = 0.0L;
    for (size_t i = 0; i < lhs.size(); ++i) {
        const long double difference =
            static_cast<long double>(lhs[i]) - static_cast<long double>(rhs[i]);
        sum += difference * difference;
    }
    return std::sqrt(static_cast<double>(sum));
}

struct HostMetrics {
    double relative_primal = std::numeric_limits<double>::infinity();
    double relative_dual = std::numeric_limits<double>::infinity();
    double relative_gap = std::numeric_limits<double>::infinity();
    double restart_relative_dual = std::numeric_limits<double>::infinity();
    double primal_objective = 0.0;
    double dual_objective = 0.0;

    double kkt() const {
        return std::max({relative_primal, relative_dual, relative_gap});
    }

    double feasibility() const {
        return std::max(relative_primal, relative_dual);
    }

    bool finite() const {
        return std::isfinite(kkt()) && std::isfinite(restart_relative_dual) &&
               std::isfinite(primal_objective) && std::isfinite(dual_objective);
    }
};

bool improves_checkpoint(const HostMetrics &candidate, const HostMetrics &incumbent) {
    if (!candidate.finite())
        return false;
    if (!incumbent.finite())
        return true;

    bool improves = candidate.kkt() < incumbent.kkt();
    constexpr double saturated_gap_floor = 0.99;
    constexpr double saturated_gap_tie = 1e-6;
    if (candidate.kkt() >= saturated_gap_floor && incumbent.kkt() >= saturated_gap_floor &&
        std::fabs(candidate.kkt() - incumbent.kkt()) <= saturated_gap_tie) {
        improves = candidate.feasibility() < incumbent.feasibility();
    }
    return improves;
}

} // namespace

// A bounded fp64 continuation for trajectories that are already close to
// feasible. With presolve it first operates on the reduced model, before
// postsolve can amplify fp32 residuals; any remaining budget may then correct
// the reconstructed original certificate. Returned vectors stay in fp64.
void MlxPdlpSolver::host_double_polish(mlxpdlp_result_t *result,
                                      bool working_model) {
    const auto &criteria = params_.termination_criteria;
    const int m = working_model ? working_num_constraints_
                                : original_num_constraints_;
    const int n = working_model ? working_num_variables_
                                : original_num_variables_;
    const int nnz = working_model ? working_num_nonzeros_
                                  : original_num_nonzeros_;
    const double objective_constant =
        working_model ? working_objective_constant_
                      : original_objective_constant_;
    const auto &row_ptr = working_model ? working_row_ptr_ : original_row_ptr_;
    const auto &col_ind = working_model ? working_col_ind_ : original_col_ind_;
    const auto &matrix_values =
        working_model ? working_matrix_values_ : original_matrix_values_;
    const auto &model_objective =
        working_model ? working_objective_ : original_objective_;
    const auto &model_variable_lower =
        working_model ? working_variable_lower_bound_
                      : original_variable_lower_bound_;
    const auto &model_variable_upper =
        working_model ? working_variable_upper_bound_
                      : original_variable_upper_bound_;
    const auto &model_constraint_lower =
        working_model ? working_constraint_lower_bound_
                      : original_constraint_lower_bound_;
    const auto &model_constraint_upper =
        working_model ? working_constraint_upper_bound_
                      : original_constraint_upper_bound_;
    const int iteration_limit = result
                                    ? std::max(
                                          0,
                                          params_.host_double_polishing_iteration_limit -
                                              result->host_double_polishing_iteration)
                                    : 0;
    const double time_limit = result
                                  ? std::max(
                                        0.0,
                                        params_.host_double_polishing_time_sec_limit -
                                            result->host_double_polishing_time)
                                  : 0.0;
    const double target_feasibility =
        working_model
            ? std::min(criteria.eps_feasible_relative,
                       criteria.eps_feas_polish_relative)
            : criteria.eps_feasible_relative;
    const double target_optimality =
        working_model
            ? std::min(criteria.eps_optimal_relative,
                       criteria.eps_feas_polish_relative)
            : criteria.eps_optimal_relative;
    if (!result || !params_.host_double_polishing ||
        iteration_limit <= 0 || time_limit <= 0.0 ||
        !result->primal_solution || !result->dual_solution ||
        !result->reduced_cost || result->num_variables != n ||
        result->num_constraints != m ||
        row_ptr.size() != static_cast<size_t>(m + 1) ||
        col_ind.size() != static_cast<size_t>(nnz) ||
        matrix_values.size() != static_cast<size_t>(nnz) ||
        model_objective.size() != static_cast<size_t>(n) ||
        model_variable_lower.size() != static_cast<size_t>(n) ||
        model_variable_upper.size() != static_cast<size_t>(n) ||
        model_constraint_lower.size() != static_cast<size_t>(m) ||
        model_constraint_upper.size() != static_cast<size_t>(m)) {
        return;
    }

    // Projection normally keeps PDHG iterates inside the working bounds, but
    // warm starts and postsolve reconstruction can bypass that device-side
    // projection. Repair the host point before the three residual metrics are
    // allowed to short-circuit polishing. The caller re-audits the resulting
    // certificate on the relevant model after this routine returns.
    bool repaired_variable_bound = false;
    for (int column = 0; column < n; ++column) {
        double &primal = result->primal_solution[column];
        if (!std::isfinite(primal))
            continue;
        const double projected =
            std::clamp(primal,
                       model_variable_lower[static_cast<size_t>(column)],
                       model_variable_upper[static_cast<size_t>(column)]);
        if (projected != primal) {
            primal = projected;
            repaired_variable_bound = true;
        }
    }

    const bool already_optimal =
        !repaired_variable_bound &&
        result->relative_primal_residual < target_feasibility &&
        result->relative_dual_residual < target_feasibility &&
        result->relative_objective_gap < target_optimality;
    if (already_optimal)
        return;

    // A continuation is useful near the fp32 floor. Also allow the important
    // postsolve case where x is already feasible but PSLP damaged y/z. It is
    // still not a replacement for the main solve: large/nonfinite dual errors
    // are rejected, as are primal iterates outside the normal fp32 margin.
    // Keep this a correction rather than a replacement solver, but admit the
    // moderate fp32 feasibility floor seen on bounded Netlib models such as
    // BORE3D (~5e-3). An original-model point whose dual and gap already pass
    // is safer to admit more broadly: the objective-neutral phase below moves
    // only x, preserves c'x, and retains a checkpoint only when complete KKT
    // merit improves. This covers primal-only fp32 stalls such as
    // L1_sixm250obs (~1.1e-1 before its fp64 correction) without widening
    // reduced/mixed-error admission.
    const bool original_primal_only =
        !working_model &&
        result->relative_dual_residual <= target_feasibility &&
        result->relative_objective_gap <= target_optimality;
    const double primal_feasibility_gate =
        std::max(original_primal_only ? 2e-1 : 1e-2,
                 (original_primal_only ? 2000.0 : 100.0) *
                     target_feasibility);
    if (!std::isfinite(result->relative_primal_residual) ||
        !std::isfinite(result->relative_dual_residual) ||
        !std::isfinite(result->relative_objective_gap) ||
        result->relative_primal_residual > primal_feasibility_gate ||
        result->relative_dual_residual > 1.0) {
        return;
    }

    if (m <= 0 || n <= 0)
        return;

    const auto polish_start = Clock::now();
    if (params_.verbose) {
        std::printf("  host-double %s polish: start full=(%.6e, %.6e, %.6e), "
                    "limit=%d/%.1fs\n",
                    working_model ? "reduced" : "original",
                    result->relative_primal_residual, result->relative_dual_residual,
                    result->relative_objective_gap, iteration_limit, time_limit);
    }

    std::vector<double> a = matrix_values;
    std::vector<double> row_scale(static_cast<size_t>(m), 1.0);
    std::vector<double> column_scale(static_cast<size_t>(n), 1.0);

    auto apply_diagonal_scaling = [&](const std::vector<double> &rows,
                                      const std::vector<double> &columns) {
        for (int row = 0; row < m; ++row) {
            const double inverse_row = 1.0 / rows[static_cast<size_t>(row)];
            row_scale[static_cast<size_t>(row)] *= rows[static_cast<size_t>(row)];
            for (int entry = row_ptr[static_cast<size_t>(row)];
                 entry < row_ptr[static_cast<size_t>(row + 1)]; ++entry) {
                const int column = col_ind[static_cast<size_t>(entry)];
                a[static_cast<size_t>(entry)] *=
                    inverse_row / columns[static_cast<size_t>(column)];
            }
        }
        for (int column = 0; column < n; ++column)
            column_scale[static_cast<size_t>(column)] *=
                columns[static_cast<size_t>(column)];
    };

    constexpr double scaling_epsilon = 1e-12;
    std::vector<double> row_metric(static_cast<size_t>(m));
    std::vector<double> column_metric(static_cast<size_t>(n));
    std::vector<double> row_update(static_cast<size_t>(m));
    std::vector<double> column_update(static_cast<size_t>(n));
    for (int iteration = 0; iteration < params_.l_inf_ruiz_iterations; ++iteration) {
        std::fill(row_metric.begin(), row_metric.end(), 0.0);
        std::fill(column_metric.begin(), column_metric.end(), 0.0);
        for (int row = 0; row < m; ++row) {
            for (int entry = row_ptr[static_cast<size_t>(row)];
                 entry < row_ptr[static_cast<size_t>(row + 1)]; ++entry) {
                const double magnitude = std::fabs(a[static_cast<size_t>(entry)]);
                if (!std::isfinite(magnitude))
                    continue;
                row_metric[static_cast<size_t>(row)] =
                    std::max(row_metric[static_cast<size_t>(row)], magnitude);
                const int column = col_ind[static_cast<size_t>(entry)];
                column_metric[static_cast<size_t>(column)] =
                    std::max(column_metric[static_cast<size_t>(column)], magnitude);
            }
        }
        for (int row = 0; row < m; ++row) {
            const double metric = row_metric[static_cast<size_t>(row)];
            row_update[static_cast<size_t>(row)] =
                metric < scaling_epsilon ? 1.0 : std::sqrt(metric);
        }
        for (int column = 0; column < n; ++column) {
            const double metric = column_metric[static_cast<size_t>(column)];
            column_update[static_cast<size_t>(column)] =
                metric < scaling_epsilon ? 1.0 : std::sqrt(metric);
        }
        apply_diagonal_scaling(row_update, column_update);
    }

    if (params_.has_pock_chambolle_alpha) {
        const double alpha = params_.pock_chambolle_alpha;
        std::fill(row_metric.begin(), row_metric.end(), 0.0);
        std::fill(column_metric.begin(), column_metric.end(), 0.0);
        for (int row = 0; row < m; ++row) {
            for (int entry = row_ptr[static_cast<size_t>(row)];
                 entry < row_ptr[static_cast<size_t>(row + 1)]; ++entry) {
                const double magnitude = std::fabs(a[static_cast<size_t>(entry)]);
                if (!std::isfinite(magnitude))
                    continue;
                row_metric[static_cast<size_t>(row)] += std::pow(magnitude, alpha);
                const int column = col_ind[static_cast<size_t>(entry)];
                column_metric[static_cast<size_t>(column)] +=
                    std::pow(magnitude, 2.0 - alpha);
            }
        }
        for (int row = 0; row < m; ++row) {
            const double metric = row_metric[static_cast<size_t>(row)];
            row_update[static_cast<size_t>(row)] =
                metric < scaling_epsilon || !std::isfinite(metric) ? 1.0
                                                                    : std::sqrt(metric);
        }
        for (int column = 0; column < n; ++column) {
            const double metric = column_metric[static_cast<size_t>(column)];
            column_update[static_cast<size_t>(column)] =
                metric < scaling_epsilon || !std::isfinite(metric) ? 1.0
                                                                    : std::sqrt(metric);
        }
        apply_diagonal_scaling(row_update, column_update);
    }

    // Build a transpose CSR once. Both sparse products then use contiguous row
    // traversals and double accumulation throughout the continuation.
    std::vector<int> transpose_row_ptr(static_cast<size_t>(n) + 1, 0);
    for (int column : col_ind)
        ++transpose_row_ptr[static_cast<size_t>(column) + 1];
    for (int column = 0; column < n; ++column)
        transpose_row_ptr[static_cast<size_t>(column) + 1] +=
            transpose_row_ptr[static_cast<size_t>(column)];
    std::vector<int> transpose_col_ind(static_cast<size_t>(nnz));
    std::vector<double> transpose_values(static_cast<size_t>(nnz));
    auto transpose_next = transpose_row_ptr;
    for (int row = 0; row < m; ++row) {
        for (int entry = row_ptr[static_cast<size_t>(row)];
             entry < row_ptr[static_cast<size_t>(row + 1)]; ++entry) {
            const int column = col_ind[static_cast<size_t>(entry)];
            const int target = transpose_next[static_cast<size_t>(column)]++;
            transpose_col_ind[static_cast<size_t>(target)] = row;
            transpose_values[static_cast<size_t>(target)] = a[static_cast<size_t>(entry)];
        }
    }

    auto multiply_a = [&](const std::vector<double> &input, std::vector<double> &output) {
        for (int row = 0; row < m; ++row) {
            long double sum = 0.0L;
            for (int entry = row_ptr[static_cast<size_t>(row)];
                 entry < row_ptr[static_cast<size_t>(row + 1)]; ++entry) {
                sum += static_cast<long double>(a[static_cast<size_t>(entry)]) *
                       input[static_cast<size_t>(
                           col_ind[static_cast<size_t>(entry)])];
            }
            output[static_cast<size_t>(row)] = static_cast<double>(sum);
        }
    };
    auto multiply_at = [&](const std::vector<double> &input, std::vector<double> &output) {
        for (int column = 0; column < n; ++column) {
            long double sum = 0.0L;
            for (int entry = transpose_row_ptr[static_cast<size_t>(column)];
                 entry < transpose_row_ptr[static_cast<size_t>(column + 1)]; ++entry) {
                sum += static_cast<long double>(transpose_values[static_cast<size_t>(entry)]) *
                       input[static_cast<size_t>(
                           transpose_col_ind[static_cast<size_t>(entry)])];
            }
            output[static_cast<size_t>(column)] = static_cast<double>(sum);
        }
    };

    std::vector<double> objective(static_cast<size_t>(n));
    std::vector<double> variable_lower(static_cast<size_t>(n));
    std::vector<double> variable_upper(static_cast<size_t>(n));
    std::vector<double> constraint_lower(static_cast<size_t>(m));
    std::vector<double> constraint_upper(static_cast<size_t>(m));
    for (int column = 0; column < n; ++column) {
        objective[static_cast<size_t>(column)] =
            model_objective[static_cast<size_t>(column)] /
            column_scale[static_cast<size_t>(column)];
        variable_lower[static_cast<size_t>(column)] = finite_scaled(
            model_variable_lower[static_cast<size_t>(column)],
            column_scale[static_cast<size_t>(column)]);
        variable_upper[static_cast<size_t>(column)] = finite_scaled(
            model_variable_upper[static_cast<size_t>(column)],
            column_scale[static_cast<size_t>(column)]);
    }
    for (int row = 0; row < m; ++row) {
        constraint_lower[static_cast<size_t>(row)] = finite_scaled(
            model_constraint_lower[static_cast<size_t>(row)],
            1.0 / row_scale[static_cast<size_t>(row)]);
        constraint_upper[static_cast<size_t>(row)] = finite_scaled(
            model_constraint_upper[static_cast<size_t>(row)],
            1.0 / row_scale[static_cast<size_t>(row)]);
    }

    long double original_objective_norm_sq = 0.0L;
    long double original_constraint_norm_sq = 0.0L;
    for (double coefficient : model_objective)
        original_objective_norm_sq += static_cast<long double>(coefficient) * coefficient;
    for (int row = 0; row < m; ++row) {
        const double lower = model_constraint_lower[static_cast<size_t>(row)];
        const double upper = model_constraint_upper[static_cast<size_t>(row)];
        if (std::isfinite(lower))
            original_constraint_norm_sq += static_cast<long double>(lower) * lower;
        if (std::isfinite(upper))
            original_constraint_norm_sq += static_cast<long double>(upper) * upper;
    }
    const double original_objective_norm =
        std::sqrt(static_cast<double>(original_objective_norm_sq));
    const double original_constraint_norm =
        std::sqrt(static_cast<double>(original_constraint_norm_sq));

    double constraint_global_scale = 1.0;
    double objective_global_scale = 1.0;
    if (params_.bound_objective_rescaling) {
        long double bound_norm_sq = 0.0L;
        for (int row = 0; row < m; ++row) {
            const double lower = constraint_lower[static_cast<size_t>(row)];
            const double upper = constraint_upper[static_cast<size_t>(row)];
            if (std::isfinite(lower) &&
                (!std::isfinite(upper) || std::fabs(lower - upper) > 1e-12))
                bound_norm_sq += static_cast<long double>(lower) * lower;
            if (std::isfinite(upper))
                bound_norm_sq += static_cast<long double>(upper) * upper;
        }
        long double objective_norm_sq = 0.0L;
        for (double coefficient : objective)
            objective_norm_sq += static_cast<long double>(coefficient) * coefficient;
        constraint_global_scale =
            1.0 / (std::sqrt(static_cast<double>(bound_norm_sq)) + 1.0);
        objective_global_scale =
            1.0 / (std::sqrt(static_cast<double>(objective_norm_sq)) + 1.0);
        for (double &value : objective)
            value *= objective_global_scale;
        for (double &value : variable_lower)
            value = finite_scaled(value, constraint_global_scale);
        for (double &value : variable_upper)
            value = finite_scaled(value, constraint_global_scale);
        for (double &value : constraint_lower)
            value = finite_scaled(value, constraint_global_scale);
        for (double &value : constraint_upper)
            value = finite_scaled(value, constraint_global_scale);
    }
    const double objective_unscale =
        constraint_global_scale * objective_global_scale;

    std::vector<double> x_initial(static_cast<size_t>(n));
    std::vector<double> x_current(static_cast<size_t>(n));
    std::vector<double> x_pdhg(static_cast<size_t>(n));
    std::vector<double> x_reflected(static_cast<size_t>(n));
    std::vector<double> x_best(static_cast<size_t>(n));
    std::vector<double> y_initial(static_cast<size_t>(m));
    std::vector<double> y_current(static_cast<size_t>(m));
    std::vector<double> y_pdhg(static_cast<size_t>(m));
    std::vector<double> y_reflected(static_cast<size_t>(m));
    std::vector<double> y_best(static_cast<size_t>(m));
    std::vector<double> dual_slack(static_cast<size_t>(n));
    std::vector<double> dual_slack_best(static_cast<size_t>(n));
    for (int column = 0; column < n; ++column) {
        double value = result->primal_solution[column] *
                       column_scale[static_cast<size_t>(column)] *
                       constraint_global_scale;
        value = std::clamp(value, variable_lower[static_cast<size_t>(column)],
                           variable_upper[static_cast<size_t>(column)]);
        x_initial[static_cast<size_t>(column)] = value;
    }
    for (int row = 0; row < m; ++row) {
        double value = result->dual_solution[row] * row_scale[static_cast<size_t>(row)] *
                       objective_global_scale;
        if (!std::isfinite(constraint_lower[static_cast<size_t>(row)]))
            value = std::min(value, 0.0);
        if (!std::isfinite(constraint_upper[static_cast<size_t>(row)]))
            value = std::max(value, 0.0);
        y_initial[static_cast<size_t>(row)] = value;
    }
    x_current = x_initial;
    x_pdhg = x_initial;
    x_best = x_initial;
    y_current = y_initial;
    y_pdhg = y_initial;
    y_best = y_initial;
    for (int column = 0; column < n; ++column) {
        dual_slack[static_cast<size_t>(column)] =
            result->reduced_cost[column] * objective_global_scale /
            column_scale[static_cast<size_t>(column)];
    }
    dual_slack_best = dual_slack;

    std::vector<double> ax(static_cast<size_t>(m));
    std::vector<double> aty(static_cast<size_t>(n));
    std::vector<double> primal_residual(static_cast<size_t>(m));
    std::vector<double> dual_residual(static_cast<size_t>(n));
    std::vector<double> restart_dual_residual(static_cast<size_t>(n));
    std::vector<double> delta_x(static_cast<size_t>(n));
    std::vector<double> delta_y(static_cast<size_t>(m));
    std::vector<double> at_delta_y(static_cast<size_t>(n));

    auto evaluate = [&](const std::vector<double> &x, const std::vector<double> &y,
                        const std::vector<double> &slack) {
        HostMetrics metrics;
        multiply_a(x, ax);
        multiply_at(y, aty);
        long double primal_objective = objective_constant;
        long double dual_objective_scaled = 0.0L;
        for (int row = 0; row < m; ++row) {
            const double activity = ax[static_cast<size_t>(row)];
            const double lower = constraint_lower[static_cast<size_t>(row)];
            const double upper = constraint_upper[static_cast<size_t>(row)];
            primal_residual[static_cast<size_t>(row)] =
                activity - std::clamp(activity, lower, upper);
            const double dual = y[static_cast<size_t>(row)];
            if (dual > 0.0 && std::isfinite(lower))
                dual_objective_scaled += static_cast<long double>(lower) * dual;
            else if (dual < 0.0 && std::isfinite(upper))
                dual_objective_scaled += static_cast<long double>(upper) * dual;
        }
        for (int column = 0; column < n; ++column) {
            const double raw = objective[static_cast<size_t>(column)] -
                               aty[static_cast<size_t>(column)];
            double reduced_cost = slack[static_cast<size_t>(column)];
            if (!std::isfinite(variable_lower[static_cast<size_t>(column)]))
                reduced_cost = std::min(reduced_cost, 0.0);
            if (!std::isfinite(variable_upper[static_cast<size_t>(column)]))
                reduced_cost = std::max(reduced_cost, 0.0);
            dual_residual[static_cast<size_t>(column)] = raw - reduced_cost;
            restart_dual_residual[static_cast<size_t>(column)] = raw - reduced_cost;
            primal_objective +=
                static_cast<long double>(model_objective[static_cast<size_t>(column)]) *
                (x[static_cast<size_t>(column)] /
                 (column_scale[static_cast<size_t>(column)] *
                  constraint_global_scale));
            if (reduced_cost > 0.0 &&
                std::isfinite(variable_lower[static_cast<size_t>(column)]))
                dual_objective_scaled +=
                    static_cast<long double>(variable_lower[static_cast<size_t>(column)]) *
                    reduced_cost;
            else if (reduced_cost < 0.0 &&
                     std::isfinite(variable_upper[static_cast<size_t>(column)]))
                dual_objective_scaled +=
                    static_cast<long double>(variable_upper[static_cast<size_t>(column)]) *
                    reduced_cost;

            dual_residual[static_cast<size_t>(column)] *=
                column_scale[static_cast<size_t>(column)] / objective_global_scale;
            restart_dual_residual[static_cast<size_t>(column)] *=
                column_scale[static_cast<size_t>(column)] / objective_global_scale;
        }
        for (int row = 0; row < m; ++row)
            primal_residual[static_cast<size_t>(row)] *=
                row_scale[static_cast<size_t>(row)] / constraint_global_scale;

        metrics.relative_primal =
            vector_norm(primal_residual, params_.optimality_norm) /
            (1.0 + original_constraint_norm);
        metrics.relative_dual = vector_norm(dual_residual, params_.optimality_norm) /
                                (1.0 + original_objective_norm);
        metrics.restart_relative_dual =
            vector_norm(restart_dual_residual, params_.optimality_norm) /
            (1.0 + original_objective_norm);
        metrics.primal_objective = static_cast<double>(primal_objective);
        metrics.dual_objective =
            static_cast<double>(dual_objective_scaled / objective_unscale) +
            objective_constant;
        metrics.relative_gap =
            std::fabs(metrics.primal_objective - metrics.dual_objective) /
            (1.0 + std::fabs(metrics.primal_objective) +
             std::fabs(metrics.dual_objective));
        return metrics;
    };

    HostMetrics best_metrics = evaluate(x_pdhg, y_pdhg, dual_slack);
    bool best_changed = false;
    if (params_.verbose) {
        long double stationarity_gap = 0.0L;
        long double row_complementarity_gap = 0.0L;
        long double variable_complementarity_gap = 0.0L;
        for (int column = 0; column < n; ++column) {
            const double slack = dual_slack[static_cast<size_t>(column)];
            const double residual = objective[static_cast<size_t>(column)] -
                                    aty[static_cast<size_t>(column)] - slack;
            stationarity_gap +=
                static_cast<long double>(x_pdhg[static_cast<size_t>(column)]) *
                residual;
            if (slack > 0.0 &&
                std::isfinite(variable_lower[static_cast<size_t>(column)])) {
                variable_complementarity_gap +=
                    static_cast<long double>(slack) *
                    (x_pdhg[static_cast<size_t>(column)] -
                     variable_lower[static_cast<size_t>(column)]);
            } else if (slack < 0.0 &&
                       std::isfinite(variable_upper[static_cast<size_t>(column)])) {
                variable_complementarity_gap +=
                    static_cast<long double>(slack) *
                    (x_pdhg[static_cast<size_t>(column)] -
                     variable_upper[static_cast<size_t>(column)]);
            }
        }
        for (int row = 0; row < m; ++row) {
            const double dual = y_pdhg[static_cast<size_t>(row)];
            if (dual > 0.0 &&
                std::isfinite(constraint_lower[static_cast<size_t>(row)])) {
                row_complementarity_gap +=
                    static_cast<long double>(dual) *
                    (ax[static_cast<size_t>(row)] -
                     constraint_lower[static_cast<size_t>(row)]);
            } else if (dual < 0.0 &&
                       std::isfinite(constraint_upper[static_cast<size_t>(row)])) {
                row_complementarity_gap +=
                    static_cast<long double>(dual) *
                    (ax[static_cast<size_t>(row)] -
                     constraint_upper[static_cast<size_t>(row)]);
            }
        }
        const double inverse_objective_unscale = 1.0 / objective_unscale;
        std::printf("  host-double gap decomposition: stationarity=%.6e "
                    "rows=%.6e variables=%.6e sum=%.6e\n",
                    static_cast<double>(stationarity_gap) *
                        inverse_objective_unscale,
                    static_cast<double>(row_complementarity_gap) *
                        inverse_objective_unscale,
                    static_cast<double>(variable_complementarity_gap) *
                        inverse_objective_unscale,
                    static_cast<double>(stationarity_gap +
                                        row_complementarity_gap +
                                        variable_complementarity_gap) *
                        inverse_objective_unscale);
    }

    // Pock-Chambolle diagonal scaling gives ||A||_2 <= 1. Use a small safety
    // margin. Without it, fall back to a guaranteed induced-norm upper bound.
    double step_size = 0.99;
    if (!params_.has_pock_chambolle_alpha) {
        std::vector<double> row_sum(static_cast<size_t>(m), 0.0);
        std::vector<double> column_sum(static_cast<size_t>(n), 0.0);
        for (int row = 0; row < m; ++row) {
            for (int entry = row_ptr[static_cast<size_t>(row)];
                 entry < row_ptr[static_cast<size_t>(row + 1)]; ++entry) {
                const double magnitude = std::fabs(a[static_cast<size_t>(entry)]);
                row_sum[static_cast<size_t>(row)] += magnitude;
                column_sum[static_cast<size_t>(
                    col_ind[static_cast<size_t>(entry)])] += magnitude;
            }
        }
        const double upper_bound =
            std::sqrt(*std::max_element(row_sum.begin(), row_sum.end()) *
                      *std::max_element(column_sum.begin(), column_sum.end()));
        step_size = upper_bound > 1e-14 ? 0.99 / upper_bound : 1.0;
    }

    const int evaluation_frequency =
        std::max(3, params_.termination_evaluation_frequency);
    int total_count = 0;
    bool correction_target_reached = false;

    // A nearly optimal certificate can occasionally be paired with the last
    // few infeasible rows of an fp32 checkpoint (cont11 is a representative
    // example). Joint PDHG moves both sides of the saddle point and can trade
    // that small primal improvement for a much larger objective-gap error.
    // Before doing that, minimize the actual (unscaled) squared feasibility
    // residual in fp64. Project its gradient orthogonally to the objective, so
    // the correction changes Ax while preserving c'x. An exact steepest-
    // descent line search on the current active rows avoids a row-order effect
    // and backtracking handles active-set or bound changes. Variable bounds and
    // the complete KKT checkpoint remain independent safeguards.
    const bool run_objective_neutral_primal =
        !working_model &&
        best_metrics.relative_primal > target_feasibility &&
        best_metrics.relative_dual <= target_feasibility &&
        best_metrics.relative_gap <= target_optimality;
    if (run_objective_neutral_primal) {
        std::vector<double> projection_x = x_initial;
        std::vector<double> projection_best_x = projection_x;
        std::vector<double> projection_candidate(static_cast<size_t>(n));
        std::vector<double> projection_residual(static_cast<size_t>(m));
        std::vector<double> projection_weighted_residual(static_cast<size_t>(m));
        std::vector<double> projection_gradient(static_cast<size_t>(n));
        std::vector<double> projection_direction(static_cast<size_t>(n));
        std::vector<double> projection_ax_direction(static_cast<size_t>(m));
        HostMetrics projection_metrics = best_metrics;
        HostMetrics projection_best_metrics = best_metrics;
        const int projection_limit =
            std::min(8192, std::max(1, iteration_limit / 2));
        const double projection_time_limit =
            std::min(time_limit, std::max(0.05, 0.9 * time_limit));
        int projection_iterations = 0;
        int projection_backtracks = 0;
        int clipped_updates = 0;
        int nonimproving_iterations = 0;
        double last_projection_step = 0.0;

        long double objective_norm_sq = 0.0L;
        for (double value : objective)
            objective_norm_sq += static_cast<long double>(value) * value;

        auto rebuild_projection_residual =
            [&](const std::vector<double> &candidate,
                std::vector<double> &residual) {
                multiply_a(candidate, ax);
                long double residual_norm_sq = 0.0L;
                for (int row = 0; row < m; ++row) {
                    const double activity = ax[static_cast<size_t>(row)];
                    const double value =
                        activity -
                        std::clamp(activity,
                                   constraint_lower[static_cast<size_t>(row)],
                                   constraint_upper[static_cast<size_t>(row)]);
                    residual[static_cast<size_t>(row)] = value;
                    const long double weight =
                        row_scale[static_cast<size_t>(row)] /
                        constraint_global_scale;
                    const long double original_residual = weight * value;
                    residual_norm_sq += original_residual * original_residual;
                }
                return residual_norm_sq;
            };

        while (projection_iterations < projection_limit &&
               seconds_since(polish_start) < projection_time_limit) {
            const long double current_residual_norm_sq =
                rebuild_projection_residual(projection_x,
                                            projection_residual);
            for (int row = 0; row < m; ++row) {
                const long double weight =
                    row_scale[static_cast<size_t>(row)] /
                    constraint_global_scale;
                projection_weighted_residual[static_cast<size_t>(row)] =
                    static_cast<double>(
                        weight * weight *
                        projection_residual[static_cast<size_t>(row)]);
            }
            multiply_at(projection_weighted_residual, projection_gradient);

            long double objective_dot_gradient = 0.0L;
            for (int column = 0; column < n; ++column) {
                objective_dot_gradient +=
                    static_cast<long double>(
                        objective[static_cast<size_t>(column)]) *
                    projection_gradient[static_cast<size_t>(column)];
            }
            const long double objective_component =
                objective_norm_sq > 1e-30L
                    ? objective_dot_gradient / objective_norm_sq
                    : 0.0L;
            long double direction_norm_sq = 0.0L;
            for (int column = 0; column < n; ++column) {
                const double direction =
                    projection_gradient[static_cast<size_t>(column)] -
                    static_cast<double>(
                        objective_component *
                        objective[static_cast<size_t>(column)]);
                projection_direction[static_cast<size_t>(column)] = direction;
                direction_norm_sq +=
                    static_cast<long double>(direction) * direction;
            }
            if (!std::isfinite(static_cast<double>(direction_norm_sq)) ||
                direction_norm_sq <= 1e-30L)
                break;

            multiply_a(projection_direction, projection_ax_direction);
            long double curvature = 0.0L;
            for (int row = 0; row < m; ++row) {
                if (projection_residual[static_cast<size_t>(row)] == 0.0)
                    continue;
                const long double weight =
                    row_scale[static_cast<size_t>(row)] /
                    constraint_global_scale;
                const long double value =
                    weight * projection_ax_direction[static_cast<size_t>(row)];
                curvature += value * value;
            }
            if (!std::isfinite(static_cast<double>(curvature)) ||
                curvature <= 1e-30L)
                break;

            double projection_step = static_cast<double>(
                0.99L * direction_norm_sq / curvature);
            if (!std::isfinite(projection_step) || projection_step <= 0.0)
                break;

            bool accepted_step = false;
            for (int backtrack = 0; backtrack < 24; ++backtrack) {
                int trial_clips = 0;
                for (int column = 0; column < n; ++column) {
                    const double unprojected =
                        projection_x[static_cast<size_t>(column)] -
                        projection_step *
                            projection_direction[static_cast<size_t>(column)];
                    const double projected = std::clamp(
                        unprojected,
                        variable_lower[static_cast<size_t>(column)],
                        variable_upper[static_cast<size_t>(column)]);
                    trial_clips += projected != unprojected;
                    projection_candidate[static_cast<size_t>(column)] = projected;
                }
                const long double candidate_residual_norm_sq =
                    rebuild_projection_residual(projection_candidate,
                                                projection_residual);
                if (std::isfinite(
                        static_cast<double>(candidate_residual_norm_sq)) &&
                    candidate_residual_norm_sq < current_residual_norm_sq) {
                    projection_x.swap(projection_candidate);
                    clipped_updates += trial_clips;
                    last_projection_step = projection_step;
                    accepted_step = true;
                    break;
                }
                projection_step *= 0.5;
                ++projection_backtracks;
            }

            ++projection_iterations;
            ++total_count;
            if (!accepted_step)
                break;

            projection_metrics =
                evaluate(projection_x, y_initial, dual_slack);
            if (projection_metrics.finite() &&
                projection_metrics.kkt() < projection_best_metrics.kkt()) {
                projection_best_metrics = projection_metrics;
                projection_best_x = projection_x;
                nonimproving_iterations = 0;
            } else {
                ++nonimproving_iterations;
            }
            if (improves_checkpoint(projection_metrics, best_metrics)) {
                best_metrics = projection_metrics;
                x_best = projection_x;
                y_best = y_initial;
                dual_slack_best = dual_slack;
                best_changed = true;
            }
            if (best_metrics.relative_primal < target_feasibility &&
                best_metrics.relative_dual < target_feasibility &&
                best_metrics.relative_gap < target_optimality) {
                correction_target_reached = true;
                break;
            }
            if (nonimproving_iterations >= 8)
                break;
        }

        if (projection_best_metrics.kkt() <
            evaluate(x_initial, y_initial, dual_slack).kkt()) {
            x_initial = projection_best_x;
            x_current = projection_best_x;
            x_pdhg = projection_best_x;
        }
        if (params_.verbose) {
            std::printf("  host-double objective-neutral primal: iter=%d "
                        "backtracks=%d step=%.3e clipped=%d "
                        "current=(%.6e, %.6e, %.6e) "
                        "best=(%.6e, %.6e, %.6e)\n",
                        projection_iterations, projection_backtracks,
                        last_projection_step, clipped_updates,
                        projection_metrics.relative_primal,
                        projection_metrics.relative_dual,
                        projection_metrics.relative_gap,
                        best_metrics.relative_primal,
                        best_metrics.relative_dual,
                        best_metrics.relative_gap);
        }
    }

    // When postsolve has preserved a high-quality primal point but damaged its
    // multiplier map, continuing the full saddle-point trajectory needlessly
    // moves x away from feasibility. First solve the homogeneous dual-
    // feasibility problem while holding x fixed. Variable and constraint
    // activity at that point enforce complementarity: interior variables force
    // z=0 and inactive rows force y=0. The activity threshold uses the tighter
    // polishing tolerance rather than the external stopping tolerance. Every
    // checkpoint is still evaluated against the complete model KKT metric, so
    // an imperfect active-set guess can never worsen the returned result.
    const bool run_primal_preserving_dual =
        best_metrics.relative_primal <= target_feasibility &&
        (best_metrics.relative_dual > target_feasibility ||
         best_metrics.relative_gap > target_optimality);
    if (run_primal_preserving_dual) {
        const double infinity = std::numeric_limits<double>::infinity();
        std::vector<double> phase_var_lower(static_cast<size_t>(n), -infinity);
        std::vector<double> phase_var_upper(static_cast<size_t>(n), infinity);
        std::vector<double> phase_con_lower(static_cast<size_t>(m), -infinity);
        std::vector<double> phase_con_upper(static_cast<size_t>(m), infinity);

        int active_columns = 0;
        int active_rows = 0;
        const double activity_epsilon =
            std::max(1e-12, criteria.eps_feas_polish_relative);
        auto active_tolerance = [&](double value) {
            return std::max(1e-9,
                            100.0 * activity_epsilon *
                                (1.0 + std::fabs(value)));
        };
        for (int column = 0; column < n; ++column) {
            const double value = x_initial[static_cast<size_t>(column)];
            const double lower = variable_lower[static_cast<size_t>(column)];
            const double upper = variable_upper[static_cast<size_t>(column)];
            const bool lower_finite = std::isfinite(lower);
            const bool upper_finite = std::isfinite(upper);
            const double tolerance = active_tolerance(value);
            const bool fixed = lower_finite && upper_finite &&
                               std::fabs(upper - lower) <= tolerance;
            const bool at_lower = lower_finite && value <= lower + tolerance;
            const bool at_upper = upper_finite && value >= upper - tolerance;
            if (fixed || (at_lower && at_upper)) {
                phase_var_lower[static_cast<size_t>(column)] = 0.0;
                phase_var_upper[static_cast<size_t>(column)] = 0.0;
                ++active_columns;
            } else if (at_lower) {
                phase_var_lower[static_cast<size_t>(column)] = 0.0;
                ++active_columns;
            } else if (at_upper) {
                phase_var_upper[static_cast<size_t>(column)] = 0.0;
                ++active_columns;
            }
        }

        // evaluate() immediately above left ax=A*x_initial in scaled model
        // coordinates. Refresh it explicitly to keep this phase independent
        // of future metric-evaluation changes.
        multiply_a(x_initial, ax);
        std::array<int, 5> activity_counts{};
        constexpr std::array<double, 5> activity_factors =
            {10.0, 100.0, 1000.0, 10000.0,
             std::numeric_limits<double>::infinity()};
        double maximum_primal_violation = 0.0;
        for (int row = 0; row < m; ++row) {
            const double activity = ax[static_cast<size_t>(row)];
            const double lower = constraint_lower[static_cast<size_t>(row)];
            const double upper = constraint_upper[static_cast<size_t>(row)];
            const bool lower_finite = std::isfinite(lower);
            const bool upper_finite = std::isfinite(upper);
            double violation = 0.0;
            if (lower_finite && activity < lower)
                violation = lower - activity;
            if (upper_finite && activity > upper)
                violation = std::max(violation, activity - upper);
            maximum_primal_violation =
                std::max(maximum_primal_violation, violation);
            for (size_t index = 0; index < activity_factors.size(); ++index) {
                const double candidate_tolerance =
                    std::isfinite(activity_factors[index])
                        ? std::max(1e-9, activity_factors[index] *
                                             activity_epsilon *
                                             (1.0 + std::fabs(activity)))
                        : std::numeric_limits<double>::infinity();
                const bool candidate_equality =
                    lower_finite && upper_finite &&
                    std::fabs(upper - lower) <= candidate_tolerance;
                const bool candidate_lower =
                    lower_finite &&
                    std::fabs(activity - lower) <= candidate_tolerance;
                const bool candidate_upper =
                    upper_finite &&
                    std::fabs(activity - upper) <= candidate_tolerance;
                if (candidate_equality || candidate_lower || candidate_upper)
                    ++activity_counts[index];
            }
            const double tolerance = active_tolerance(activity);
            const bool equality =
                lower_finite && upper_finite &&
                std::fabs(upper - lower) <= tolerance;
            const bool lower_active =
                lower_finite && std::fabs(activity - lower) <= tolerance;
            const bool upper_active =
                upper_finite && std::fabs(activity - upper) <= tolerance;
            if (equality || (lower_active && upper_active)) {
                phase_con_lower[static_cast<size_t>(row)] = 0.0;
                phase_con_upper[static_cast<size_t>(row)] = 0.0;
                ++active_rows;
            } else if (lower_active) {
                phase_con_lower[static_cast<size_t>(row)] = 0.0;
                ++active_rows;
            } else if (upper_active) {
                phase_con_upper[static_cast<size_t>(row)] = 0.0;
                ++active_rows;
            }
        }
        if (params_.verbose) {
            std::printf("  host-double activity: max-violation=%.6e "
                        "rows@10/100/1000/10000/all=%d/%d/%d/%d/%d\n",
                        maximum_primal_violation, activity_counts[0],
                        activity_counts[1], activity_counts[2],
                        activity_counts[3], activity_counts[4]);
        }

        // Reconstruct the dual certificate without moving the already-feasible
        // primal point.  For fixed x this is the smooth convex problem
        //
        //   min_y 1/2 dist(c - A^T y, N_X(x))^2,  y in Y,
        //
        // where Y preserves each row multiplier's globally valid sign. Rows
        // that look inactive are fixed to zero, preventing new
        // complementarity errors away from a bound. The full-gap checkpoint
        // below remains an independent safeguard. Its gradient is -A times
        // the stationarity residual. A
        // block-restarted FISTA iteration uses the same fp64 sparse products
        // as the continuation. Each block must decrease the residual norm;
        // otherwise it is rolled back and the step is halved. Full-model KKT
        // checkpoints provide a second, independent safeguard.
        std::vector<double> certificate_y = y_initial;
        std::vector<double> certificate_previous(static_cast<size_t>(m));
        std::vector<double> certificate_momentum(static_cast<size_t>(m));
        std::vector<double> certificate_next(static_cast<size_t>(m));
        std::vector<double> certificate_accelerated(static_cast<size_t>(m));
        std::vector<double> certificate_block_start(static_cast<size_t>(m));
        std::vector<double> certificate_gradient(static_cast<size_t>(m));
        std::vector<double> certificate_residual(static_cast<size_t>(n));
        std::vector<double> certificate_slack(static_cast<size_t>(n));
        std::vector<double> certificate_work_slack(static_cast<size_t>(n));

        auto project_row_multiplier = [&](double value, int row) {
            const bool adjustable =
                std::isfinite(phase_con_lower[static_cast<size_t>(row)]) ||
                std::isfinite(phase_con_upper[static_cast<size_t>(row)]);
            if (!adjustable)
                return 0.0;
            const bool lower_finite =
                std::isfinite(constraint_lower[static_cast<size_t>(row)]);
            const bool upper_finite =
                std::isfinite(constraint_upper[static_cast<size_t>(row)]);
            if (!lower_finite && !upper_finite)
                return 0.0;
            if (lower_finite && !upper_finite)
                return std::max(value, 0.0);
            if (!lower_finite && upper_finite)
                return std::min(value, 0.0);
            return value;
        };
        auto rebuild_certificate_residual =
            [&](const std::vector<double> &candidate_y,
                std::vector<double> &candidate_slack) {
                multiply_at(candidate_y, aty);
                long double residual_norm_sq = 0.0L;
                for (int column = 0; column < n; ++column) {
                    const double raw = objective[static_cast<size_t>(column)] -
                                       aty[static_cast<size_t>(column)];
                    const bool lower_finite =
                        std::isfinite(phase_var_lower[static_cast<size_t>(column)]);
                    const bool upper_finite =
                        std::isfinite(phase_var_upper[static_cast<size_t>(column)]);
                    double slack = raw;
                    if (!lower_finite && !upper_finite)
                        slack = 0.0;
                    else if (lower_finite && !upper_finite)
                        slack = std::max(raw, 0.0);
                    else if (!lower_finite && upper_finite)
                        slack = std::min(raw, 0.0);
                    candidate_slack[static_cast<size_t>(column)] = slack;
                    const double residual = raw - slack;
                    certificate_residual[static_cast<size_t>(column)] = residual;
                    residual_norm_sq +=
                        static_cast<long double>(residual) * residual;
                }
                return residual_norm_sq;
            };

        for (int row = 0; row < m; ++row) {
            certificate_y[static_cast<size_t>(row)] =
                project_row_multiplier(certificate_y[static_cast<size_t>(row)], row);
        }
        long double certificate_residual_norm_sq =
            rebuild_certificate_residual(certificate_y, certificate_slack);
        const long double certificate_initial_residual_norm_sq =
            certificate_residual_norm_sq;
        HostMetrics certificate_final_metrics =
            evaluate(x_initial, certificate_y, certificate_slack);
        if (improves_checkpoint(certificate_final_metrics, best_metrics)) {
            best_metrics = certificate_final_metrics;
            x_best = x_initial;
            y_best = certificate_y;
            dual_slack_best = certificate_slack;
            best_changed = true;
        }
        if (best_metrics.relative_primal < target_feasibility &&
            best_metrics.relative_dual < target_feasibility &&
            best_metrics.relative_gap < target_optimality) {
            correction_target_reached = true;
        }

        double certificate_step = params_.has_pock_chambolle_alpha
                                      ? 0.99
                                      : step_size * step_size / 0.99;
        if (!std::isfinite(certificate_step) || certificate_step <= 0.0)
            certificate_step = 1e-6;
        // A fixed-primal certificate solve cannot remove an objective gap
        // caused by a feasible but suboptimal x. Reserve half of the remaining
        // iteration and wall-clock budgets for the joint continuation below.
        // The complete KKT checkpoint still protects a genuinely optimal
        // primal point from any regression when the joint phase starts.
        const int remaining_iterations = iteration_limit - total_count;
        const int certificate_budget =
            remaining_iterations <= 1
                ? remaining_iterations
                : std::max(1, remaining_iterations / 2);
        const int certificate_iteration_limit =
            total_count + certificate_budget;
        const double certificate_start_elapsed = seconds_since(polish_start);
        const double certificate_time_limit =
            certificate_start_elapsed +
            0.5 * std::max(0.0, time_limit - certificate_start_elapsed);
        int certificate_count = 0;
        int certificate_backtracks = 0;

        while (!correction_target_reached &&
               total_count < certificate_iteration_limit &&
               certificate_residual_norm_sq > 1e-30L &&
               seconds_since(polish_start) < certificate_time_limit) {
            const int block = std::min(evaluation_frequency,
                                       certificate_iteration_limit - total_count);
            certificate_block_start = certificate_y;
            const long double block_start_residual_norm_sq =
                certificate_residual_norm_sq;
            certificate_previous = certificate_y;
            certificate_momentum = certificate_y;
            double acceleration = 1.0;
            int completed = 0;

            for (; completed < block &&
                   seconds_since(polish_start) < certificate_time_limit;
                 ++completed) {
                rebuild_certificate_residual(certificate_momentum,
                                             certificate_work_slack);
                multiply_a(certificate_residual, certificate_gradient);
                const double next_acceleration =
                    0.5 * (1.0 + std::sqrt(1.0 +
                                           4.0 * acceleration * acceleration));
                const double momentum =
                    (acceleration - 1.0) / next_acceleration;
                for (int row = 0; row < m; ++row) {
                    const double next = project_row_multiplier(
                        certificate_momentum[static_cast<size_t>(row)] +
                            certificate_step *
                                certificate_gradient[static_cast<size_t>(row)],
                        row);
                    certificate_next[static_cast<size_t>(row)] = next;
                    certificate_accelerated[static_cast<size_t>(row)] =
                        next + momentum *
                                   (next - certificate_previous[static_cast<size_t>(row)]);
                }
                certificate_previous.swap(certificate_next);
                certificate_momentum.swap(certificate_accelerated);
                acceleration = next_acceleration;
            }

            total_count += completed;
            certificate_count += completed;
            certificate_y = certificate_previous;
            certificate_residual_norm_sq =
                rebuild_certificate_residual(certificate_y, certificate_slack);
            const bool accept_block =
                std::isfinite(static_cast<double>(certificate_residual_norm_sq)) &&
                certificate_residual_norm_sq <=
                    block_start_residual_norm_sq * (1.0L + 1e-10L);
            if (!accept_block) {
                certificate_y = certificate_block_start;
                certificate_residual_norm_sq = rebuild_certificate_residual(
                    certificate_y, certificate_slack);
                certificate_step *= 0.5;
                ++certificate_backtracks;
            }

            certificate_final_metrics =
                evaluate(x_initial, certificate_y, certificate_slack);
            if (improves_checkpoint(certificate_final_metrics, best_metrics)) {
                best_metrics = certificate_final_metrics;
                x_best = x_initial;
                y_best = certificate_y;
                dual_slack_best = certificate_slack;
                best_changed = true;
            }
            if (best_metrics.relative_primal < target_feasibility &&
                best_metrics.relative_dual < target_feasibility &&
                best_metrics.relative_gap < target_optimality) {
                correction_target_reached = true;
                break;
            }
            if (completed < block || certificate_step < 1e-16)
                break;
        }

        if (params_.verbose) {
            const double initial_residual = std::sqrt(static_cast<double>(
                certificate_initial_residual_norm_sq));
            const double final_residual = std::sqrt(static_cast<double>(
                certificate_residual_norm_sq));
            std::printf("  host-double dual certificate: rows=%d cols=%d iter=%d "
                        "backtracks=%d step=%.3e residual=%.6e->%.6e "
                        "current=(%.6e, %.6e, %.6e) "
                        "best=(%.6e, %.6e, %.6e)\n",
                        active_rows, active_columns, certificate_count,
                        certificate_backtracks, certificate_step,
                        initial_residual, final_residual,
                        certificate_final_metrics.relative_primal,
                        certificate_final_metrics.relative_dual,
                        certificate_final_metrics.relative_gap,
                        best_metrics.relative_primal, best_metrics.relative_dual,
                        best_metrics.relative_gap);
        }

        // A certificate endpoint with a better stationarity residual is also a
        // useful start for the remaining joint continuation, even if its gap
        // has not yet beaten the global checkpoint. The latter remains
        // untouched, so this handoff cannot worsen the returned result.
        x_initial = x_best;
        x_current = x_best;
        x_pdhg = x_best;
        const bool use_certificate_endpoint =
            certificate_final_metrics.finite() &&
            certificate_final_metrics.relative_dual < best_metrics.relative_dual;
        y_initial = use_certificate_endpoint ? certificate_y : y_best;
        y_current = y_initial;
        y_pdhg = y_initial;
        dual_slack =
            use_certificate_endpoint ? certificate_slack : dual_slack_best;
    }

    const HostMetrics joint_start_metrics =
        evaluate(x_initial, y_initial, dual_slack);
    HostMetrics joint_final_metrics = joint_start_metrics;
    const double initial_balance_ratio =
        (joint_start_metrics.relative_dual + 1e-12) /
        (joint_start_metrics.relative_primal + 1e-12);
    double primal_weight =
        std::sqrt(std::clamp(initial_balance_ratio, 1e-6, 1e6));
    double best_primal_weight = primal_weight;
    double best_balance = std::numeric_limits<double>::infinity();
    double error_sum = 0.0;
    double last_error = 0.0;
    double initial_fixed_point_error = std::numeric_limits<double>::infinity();
    double last_trial_fixed_point_error = std::numeric_limits<double>::infinity();
    double fixed_point_error = std::numeric_limits<double>::infinity();
    int inner_count = 0;
    bool initialize_restart_error = false;

    auto compute_fixed_point_error = [&]() {
        long double primal_norm_sq = 0.0L;
        long double dual_norm_sq = 0.0L;
        for (int column = 0; column < n; ++column) {
            const double difference = x_reflected[static_cast<size_t>(column)] -
                                      x_pdhg[static_cast<size_t>(column)];
            delta_x[static_cast<size_t>(column)] = difference;
            primal_norm_sq += static_cast<long double>(difference) * difference;
        }
        for (int row = 0; row < m; ++row) {
            const double difference = y_reflected[static_cast<size_t>(row)] -
                                      y_pdhg[static_cast<size_t>(row)];
            delta_y[static_cast<size_t>(row)] = difference;
            dual_norm_sq += static_cast<long double>(difference) * difference;
        }
        multiply_at(delta_y, at_delta_y);
        long double cross_term = 0.0L;
        for (int column = 0; column < n; ++column)
            cross_term += static_cast<long double>(at_delta_y[static_cast<size_t>(column)]) *
                          delta_x[static_cast<size_t>(column)];
        const long double argument =
            primal_norm_sq * primal_weight + dual_norm_sq / primal_weight +
            2.0L * step_size * cross_term;
        fixed_point_error =
            std::sqrt(static_cast<double>(std::max(argument, 0.0L)));
    };

    while (!correction_target_reached && total_count < iteration_limit &&
           seconds_since(polish_start) < time_limit) {
        const int block = std::min(evaluation_frequency, iteration_limit - total_count);
        const int base_inner_count = inner_count;
        for (int offset = 1; offset <= block; ++offset) {
            const int k = base_inner_count + offset;
            const double averaging_weight =
                static_cast<double>(k) / static_cast<double>(k + 1);
            const double primal_step = step_size / primal_weight;
            const double dual_step = step_size * primal_weight;

            multiply_at(y_current, aty);
            for (int column = 0; column < n; ++column) {
                const double temporary =
                    x_current[static_cast<size_t>(column)] -
                    primal_step * (objective[static_cast<size_t>(column)] -
                                   aty[static_cast<size_t>(column)]);
                const double projected =
                    std::clamp(temporary, variable_lower[static_cast<size_t>(column)],
                               variable_upper[static_cast<size_t>(column)]);
                x_pdhg[static_cast<size_t>(column)] = projected;
                dual_slack[static_cast<size_t>(column)] =
                    (projected - temporary) / primal_step;
                x_reflected[static_cast<size_t>(column)] =
                    2.0 * projected - x_current[static_cast<size_t>(column)];
                const double reflected =
                    params_.reflection_coefficient *
                        x_reflected[static_cast<size_t>(column)] +
                    (1.0 - params_.reflection_coefficient) *
                        x_current[static_cast<size_t>(column)];
                x_current[static_cast<size_t>(column)] =
                    averaging_weight * reflected +
                    (1.0 - averaging_weight) * x_initial[static_cast<size_t>(column)];
            }

            multiply_a(x_reflected, ax);
            for (int row = 0; row < m; ++row) {
                const double temporary =
                    y_current[static_cast<size_t>(row)] / dual_step -
                    ax[static_cast<size_t>(row)];
                const double projected =
                    std::clamp(temporary,
                               -constraint_upper[static_cast<size_t>(row)],
                               -constraint_lower[static_cast<size_t>(row)]);
                const double next = (temporary - projected) * dual_step;
                y_pdhg[static_cast<size_t>(row)] = next;
                y_reflected[static_cast<size_t>(row)] =
                    2.0 * next - y_current[static_cast<size_t>(row)];
                const double reflected =
                    params_.reflection_coefficient *
                        y_reflected[static_cast<size_t>(row)] +
                    (1.0 - params_.reflection_coefficient) *
                        y_current[static_cast<size_t>(row)];
                y_current[static_cast<size_t>(row)] =
                    averaging_weight * reflected +
                    (1.0 - averaging_weight) * y_initial[static_cast<size_t>(row)];
            }

            if (offset == 1 && initialize_restart_error) {
                compute_fixed_point_error();
                initial_fixed_point_error = fixed_point_error;
                initialize_restart_error = false;
            }
        }

        compute_fixed_point_error();
        HostMetrics current_metrics = evaluate(x_pdhg, y_pdhg, dual_slack);
        joint_final_metrics = current_metrics;
        total_count += block;
        inner_count += block;

        if (improves_checkpoint(current_metrics, best_metrics)) {
            best_metrics = current_metrics;
            x_best = x_pdhg;
            y_best = y_pdhg;
            dual_slack_best = dual_slack;
            best_changed = true;
        }
        if (current_metrics.relative_primal < target_feasibility &&
            current_metrics.relative_dual < target_feasibility &&
            current_metrics.relative_gap < target_optimality) {
            best_metrics = current_metrics;
            x_best = x_pdhg;
            y_best = y_pdhg;
            dual_slack_best = dual_slack;
            best_changed = true;
            break;
        }
        if (block < evaluation_frequency)
            break;

        const auto &restart = params_.restart_params;
        bool do_restart = total_count == evaluation_frequency;
        if (!do_restart && total_count > evaluation_frequency) {
            do_restart =
                fixed_point_error <=
                    restart.sufficient_reduction_for_restart *
                        initial_fixed_point_error ||
                (fixed_point_error <=
                     restart.necessary_reduction_for_restart *
                         initial_fixed_point_error &&
                 fixed_point_error > last_trial_fixed_point_error) ||
                inner_count >= restart.artificial_restart_threshold * total_count;
        }
        last_trial_fixed_point_error = fixed_point_error;
        if (!do_restart)
            continue;

        const double primal_distance = distance_norm(x_pdhg, x_initial);
        const double dual_distance = distance_norm(y_pdhg, y_initial);
        const double ratio = current_metrics.relative_primal > 0.0
                                 ? current_metrics.restart_relative_dual /
                                       current_metrics.relative_primal
                                 : std::numeric_limits<double>::infinity();
        if (primal_distance > 1e-16 && dual_distance > 1e-16 &&
            primal_distance < 1e12 && dual_distance < 1e12 && ratio > 1e-8 &&
            ratio < 1e8) {
            const double error = std::log(dual_distance) -
                                 std::log(primal_distance) -
                                 std::log(primal_weight);
            error_sum = error_sum * restart.i_smooth + error;
            const double derivative = error - last_error;
            primal_weight *=
                std::exp(restart.k_p * error + restart.k_i * error_sum +
                         restart.k_d * derivative);
            last_error = error;
            if (!std::isfinite(primal_weight) || primal_weight <= 0.0)
                primal_weight = best_primal_weight;
        } else {
            primal_weight = best_primal_weight;
            error_sum = 0.0;
            last_error = 0.0;
        }
        const double balance =
            ratio > 0.0 && std::isfinite(ratio)
                ? std::fabs(std::log10(ratio))
                : std::numeric_limits<double>::infinity();
        if (balance < best_balance) {
            best_balance = balance;
            best_primal_weight = primal_weight;
        }
        x_initial = x_pdhg;
        x_current = x_pdhg;
        y_initial = y_pdhg;
        y_current = y_pdhg;
        inner_count = 0;
        last_trial_fixed_point_error = std::numeric_limits<double>::infinity();
        initialize_restart_error = true;
    }

    const double elapsed = seconds_since(polish_start);
    result->host_double_polishing_iteration += total_count;
    result->host_double_polishing_time += elapsed;
    result->cumulative_time_sec += elapsed;
    if (best_changed) {
        for (int column = 0; column < n; ++column) {
            result->primal_solution[column] =
                x_best[static_cast<size_t>(column)] /
                (column_scale[static_cast<size_t>(column)] *
                 constraint_global_scale);
        }
        for (int row = 0; row < m; ++row) {
            result->dual_solution[row] =
                y_best[static_cast<size_t>(row)] /
                (row_scale[static_cast<size_t>(row)] * objective_global_scale);
        }
        for (int column = 0; column < n; ++column) {
            result->reduced_cost[column] =
                dual_slack_best[static_cast<size_t>(column)] *
                column_scale[static_cast<size_t>(column)] /
                objective_global_scale;
        }
    }

    if (params_.verbose) {
        std::printf("  host-double %s polish: iter=%d time=%.3fs current=(%.6e, %.6e, "
                    "%.6e) best=(%.6e, %.6e, %.6e) %s\n",
                    working_model ? "reduced" : "original", total_count,
                    elapsed, joint_final_metrics.relative_primal,
                    joint_final_metrics.relative_dual,
                    joint_final_metrics.relative_gap,
                    best_metrics.relative_primal, best_metrics.relative_dual,
                    best_metrics.relative_gap,
                    best_changed ? "accepted" : "unchanged");
    }
}

} // namespace mlxpdlp
