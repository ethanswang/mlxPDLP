// Copyright 2026 Ethan Wang <ethanshurui.wang@gmail.com>
// SPDX-License-Identifier: Apache-2.0
#include "mlxPDLP/batch_solver.h"
#include <iostream>
#include <stdexcept>

using namespace mlxpdlp;

namespace {
const int row_ptr[] = {0, 1};
const int col_ind[] = {0};
const double values[] = {1.0};
const double lower[] = {0.0}, upper[] = {1.0}, rhs[] = {0.5};

void check(bool condition, const std::string &message) {
    if (!condition)
        throw std::runtime_error(message);
}

std::unique_ptr<MlxPdlpSolver> solver(const pdhg_parameters_t &params, mx::Device device) {
    return std::make_unique<MlxPdlpSolver>(1, 1, row_ptr, col_ind, values, lower, upper, rhs, rhs,
                                           values, 0.0, &params, device);
}

template <class Action> std::string rejection(Action action, const char *field) {
    try {
        action();
    } catch (const std::invalid_argument &error) {
        const std::string message = error.what();
        check(message.find(field) != std::string::npos, "missing field in error: " + message);
        return message;
    }
    throw std::runtime_error(std::string("accepted invalid ") + field);
}

void test_validation(mx::Device device) {
    pdhg_parameters_t defaults;
    mlxpdlp_set_default_parameters(&defaults);
    defaults.presolve = false;
    defaults.verbose = false;
    SharedMatrixPlan plan(1, 1, row_ptr, col_ind, values, &defaults, device);

    struct InvalidCase {
        const char *field;
        void (*set)(pdhg_parameters_t &);
    };
    const InvalidCase cases[] = {
        {"eps_optimal_relative", [](auto &p) { p.termination_criteria.eps_optimal_relative = -1; }},
        {"eps_feasible_relative",
         [](auto &p) { p.termination_criteria.eps_feasible_relative = NAN; }},
        {"eps_feas_polish_relative",
         [](auto &p) { p.termination_criteria.eps_feas_polish_relative = -1; }},
        {"eps_infeasible_relative",
         [](auto &p) { p.termination_criteria.eps_infeasible_relative = INFINITY; }},
        {"iteration_limit", [](auto &p) { p.termination_criteria.iteration_limit = -1; }},
        {"time_sec_limit", [](auto &p) { p.termination_criteria.time_sec_limit = NAN; }},
        {"optimality_norm", [](auto &p) { p.optimality_norm = static_cast<norm_type_t>(2); }},
        {"geometric_mean_iterations", [](auto &p) { p.geometric_mean_iterations = -1; }},
        {"termination_evaluation_frequency",
         [](auto &p) { p.termination_evaluation_frequency = 0; }},
        {"sv_tol", [](auto &p) { p.sv_tol = INFINITY; }},
        {"reflection_coefficient", [](auto &p) { p.reflection_coefficient = 1.1; }},
        {"k_p", [](auto &p) { p.restart_params.k_p = NAN; }},
        {"necessary_reduction_for_restart",
         [](auto &p) { p.restart_params.necessary_reduction_for_restart = NAN; }},
        {"matrix_zero_tol", [](auto &p) { p.matrix_zero_tol = -1; }},
        {"host_double_polishing_iteration_limit",
         [](auto &p) { p.host_double_polishing_iteration_limit = -1; }},
    };
    for (const auto &test : cases) {
        auto params = defaults;
        test.set(params);
        const auto single = rejection([&] { solver(params, device); }, test.field);
        const auto batch = rejection(
            [&] { SharedMatrixPlan invalid(1, 1, row_ptr, col_ind, values, &params, device); },
            test.field);
        // Even an empty submission must validate before returning or checking
        // whether preparation settings differ from those captured by the plan.
        const auto override_error =
            rejection([&] { plan.solve_batch({}, {}, &params); }, test.field);
        check(single == batch && single == override_error, "entry points disagree on validation");
    }

    BatchProblem problem;
    problem.objective = {1.0};
    problem.variable_lower_bounds = {0.0};
    problem.variable_upper_bounds = {1.0};
    problem.constraint_lower_bounds = problem.constraint_upper_bounds = {0.5};
    const auto valid = plan.solve_batch({problem});
    check(valid.results[0].result->termination_reason == TERMINATION_REASON_OPTIMAL,
          "invalid overrides must leave the plan usable");

    // Zero convergence tolerances are the existing fixed-work convention.
    // Check actual solve behavior, including a partial final evaluation block.
    auto fixed = defaults;
    fixed.termination_criteria.eps_optimal_relative = 0;
    fixed.termination_criteria.eps_feasible_relative = 0;
    fixed.termination_criteria.eps_feas_polish_relative = 0;
    fixed.termination_criteria.time_sec_limit = INFINITY;
    fixed.termination_evaluation_frequency = 3;
    fixed.sv_max_iter = 0; // supported norm-bound fallback
    fixed.host_double_polishing_iteration_limit = 0;
    fixed.host_double_polishing_time_sec_limit = 0;
    for (int iterations : {0, 7}) {
        fixed.termination_criteria.iteration_limit = iterations;
        auto single_solver = solver(fixed, device);
        OwnedSolveResult single(single_solver->solve());
        check(single->termination_reason == TERMINATION_REASON_ITERATION_LIMIT &&
                  single->total_count == iterations,
              "single fixed-work budget");
        SharedMatrixPlan fixed_plan(1, 1, row_ptr, col_ind, values, &fixed, device);
        BatchOptions options;
        options.execution =
            device.type == mx::Device::gpu ? BatchExecution::shared : BatchExecution::independent;
        auto batch = fixed_plan.solve_batch({problem, problem}, options, &fixed);
        for (const auto &member : batch.results)
            check(member.result &&
                      member.result->termination_reason == TERMINATION_REASON_ITERATION_LIMIT &&
                      member.result->total_count == iterations,
                  "batch fixed-work budget");
    }

    // Disabled scaling may carry an unused alpha; a zero time budget is valid.
    fixed.has_pock_chambolle_alpha = false;
    fixed.pock_chambolle_alpha = NAN;
    fixed.termination_criteria.time_sec_limit = 0;
    fixed.host_double_polishing_time_sec_limit = INFINITY;
    solver(fixed, device);
    SharedMatrixPlan zero_budget(1, 1, row_ptr, col_ind, values, &fixed, device);
    zero_budget.solve_batch({}, {}, &fixed);
}
} // namespace

int main(int argc, char **argv) {
    try {
        const bool gpu = argc > 1 && std::string(argv[1]) == "gpu";
        if (gpu && !mx::is_available(mx::Device::gpu))
            return 77;
        test_validation(mx::Device(gpu ? mx::Device::gpu : mx::Device::cpu));
        std::cout << "Parameter validation passed\n";
        return 0;
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
