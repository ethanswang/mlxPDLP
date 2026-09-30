// Copyright 2026 Ethan Wang <ethanshurui.wang@gmail.com>
// SPDX-License-Identifier: Apache-2.0
#include "parameter_validation.h"
#include <cmath>
#include <sstream>
#include <stdexcept>

namespace mlxpdlp::detail {
namespace {
template <class T> void require(bool valid, const char *field, const char *rule, T value) {
    if (!valid) {
        std::ostringstream message;
        message << "parameters." << field << " must be " << rule << " (got " << value << ')';
        throw std::invalid_argument(message.str());
    }
}

void nonnegative(int value, const char *field) {
    require(value >= 0, field, "nonnegative", value);
}

void finite_nonnegative(double value, const char *field) {
    require(std::isfinite(value) && value >= 0, field, "finite and nonnegative", value);
}

void finite_positive(double value, const char *field) {
    require(std::isfinite(value) && value > 0, field, "finite and positive", value);
}

void time_budget(double value, const char *field) {
    // Positive infinity is the supported unlimited budget; zero is immediate.
    require(!std::isnan(value) && value >= 0, field, "nonnegative (positive infinity is allowed)",
            value);
}
} // namespace

void validate_parameters(const pdhg_parameters_t &p) {
    const auto &t = p.termination_criteria;
    // Zero disables optimality termination for fixed-work benchmarks. Keep it
    // valid for both single and batch solves, including Python's tolerance alias
    // which also sets the feasibility-polishing tolerance to zero.
    finite_nonnegative(t.eps_optimal_relative, "termination_criteria.eps_optimal_relative");
    finite_nonnegative(t.eps_feasible_relative, "termination_criteria.eps_feasible_relative");
    finite_nonnegative(t.eps_feas_polish_relative, "termination_criteria.eps_feas_polish_relative");
    finite_positive(t.eps_infeasible_relative, "termination_criteria.eps_infeasible_relative");
    nonnegative(t.iteration_limit, "termination_criteria.iteration_limit");
    time_budget(t.time_sec_limit, "termination_criteria.time_sec_limit");

    nonnegative(p.geometric_mean_iterations, "geometric_mean_iterations");
    nonnegative(p.curtis_reid_iterations, "curtis_reid_iterations");
    nonnegative(p.l_inf_ruiz_iterations, "l_inf_ruiz_iterations");
    if (p.has_pock_chambolle_alpha)
        require(std::isfinite(p.pock_chambolle_alpha) && p.pock_chambolle_alpha >= 0 &&
                    p.pock_chambolle_alpha <= 2,
                "pock_chambolle_alpha", "finite and in [0, 2]", p.pock_chambolle_alpha);
    finite_nonnegative(p.matrix_zero_tol, "matrix_zero_tol");

    require(p.termination_evaluation_frequency > 0, "termination_evaluation_frequency", "positive",
            p.termination_evaluation_frequency);
    // Zero selects the existing norm-bound fallback without power iterations.
    nonnegative(p.sv_max_iter, "sv_max_iter");
    finite_positive(p.sv_tol, "sv_tol");
    require(p.optimality_norm == NORM_TYPE_L2 || p.optimality_norm == NORM_TYPE_L_INF,
            "optimality_norm", "L2 (0) or L_INF (1)", static_cast<int>(p.optimality_norm));
    require(p.restart_policy == 0 || p.restart_policy == 1, "restart_policy", "0 (PID) or 1 (HPR)",
            p.restart_policy);
    require(std::isfinite(p.reflection_coefficient) && p.reflection_coefficient > 0 &&
                p.reflection_coefficient <= 1,
            "reflection_coefficient", "finite and in (0, 1]", p.reflection_coefficient);

    const auto &r = p.restart_params;
    require(std::isfinite(r.k_p), "restart_params.k_p", "finite", r.k_p);
    require(std::isfinite(r.k_i), "restart_params.k_i", "finite", r.k_i);
    require(std::isfinite(r.k_d), "restart_params.k_d", "finite", r.k_d);
    require(std::isfinite(r.i_smooth) && r.i_smooth >= 0 && r.i_smooth <= 1,
            "restart_params.i_smooth", "finite and in [0, 1]", r.i_smooth);
    // Preserve custom finite restart thresholds, including diagnostic settings.
    require(std::isfinite(r.artificial_restart_threshold),
            "restart_params.artificial_restart_threshold", "finite",
            r.artificial_restart_threshold);
    require(std::isfinite(r.sufficient_reduction_for_restart),
            "restart_params.sufficient_reduction_for_restart", "finite",
            r.sufficient_reduction_for_restart);
    require(std::isfinite(r.necessary_reduction_for_restart),
            "restart_params.necessary_reduction_for_restart", "finite",
            r.necessary_reduction_for_restart);

    nonnegative(p.host_double_polishing_iteration_limit, "host_double_polishing_iteration_limit");
    time_budget(p.host_double_polishing_time_sec_limit, "host_double_polishing_time_sec_limit");
}
} // namespace mlxpdlp::detail
