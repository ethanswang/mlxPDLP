// Copyright 2026 Ethan Wang <ethanshurui.wang@gmail.com>
// SPDX-License-Identifier: Apache-2.0
#include "mlxPDLP/batch_solver.h"
#include <iostream>
#include <stdexcept>

using namespace mlxpdlp;
void check(bool ok, const char *message) { if (!ok) throw std::runtime_error(message); }
int main(int argc, char **argv) {
    try {
        const bool gpu = argc > 1 && std::string(argv[1]) == "gpu";
        if (gpu && !mx::is_available(mx::Device::gpu)) return 77;
        const mx::Device device(gpu ? mx::Device::gpu : mx::Device::cpu);
        pdhg_parameters_t params; mlxpdlp_set_default_parameters(&params);
        params.presolve = false; params.verbose = false;
        params.termination_criteria.eps_optimal_relative = 1e-4;
        params.termination_criteria.eps_feasible_relative = 1e-4;
        params.termination_criteria.iteration_limit = 20000;
        params.termination_criteria.time_sec_limit = 10;
        params.termination_evaluation_frequency = 50;
        int rp[] = {0,3}, ci[] = {0,0,1}; double values[] = {0.5,0.5,1};
        SharedMatrixPlan plan(2,1,rp,ci,values,&params,device);
        check(plan.operator_norm_upper_bound() > 0, "missing norm bound");
        values[0] = 100; // plan owns original numeric matrix
        BatchOptions options; options.execution = gpu ? BatchExecution::shared : BatchExecution::independent;
        for (int width : {0,1,2,3,4,5,8,9,16,17}) {
            std::vector<BatchProblem> problems(width);
            for (int j = 0; j < width; ++j) {
                auto &p = problems[j];
                p.objective = j % 2 ? std::vector<double>{3,1} : std::vector<double>{1,2};
                p.variable_lower_bounds = {0,0}; p.variable_upper_bounds = {1,1};
                p.constraint_lower_bounds = {1}; p.constraint_upper_bounds = {1};
                p.objective_constant = 0.25*j;
                if (j % 3 == 0) { p.primal_start = std::vector<double>{0.5,0.5}; p.dual_start = std::vector<double>{0}; }
            }
            auto result = plan.solve_batch(problems, options);
            check(result.results.size() == size_t(width), "result width");
            if (gpu && width > 1) check(result.execution == BatchExecution::shared, "silent fallback");
            for (int j = 0; j < width; ++j) {
                const auto &member = result.results[j]; const auto &r = *member.result;
                if (r.termination_reason != TERMINATION_REASON_OPTIMAL) {
                    std::cerr << "width=" << width << " member=" << j << " reason=" << r.termination_reason
                              << " iterations=" << r.total_count << " residuals=" << r.relative_primal_residual
                              << ',' << r.relative_dual_residual << ',' << r.relative_objective_gap << '\n';
                }
                check(member.has_solution && member.input_index == size_t(j), "member ownership/order");
                check(r.termination_reason == TERMINATION_REASON_OPTIMAL, "analytic optimal status");
                check(std::abs(r.primal_objective_value - (1+0.25*j)) < 2e-3, "analytic objective");
                check(std::abs(r.primal_solution[0] + r.primal_solution[1] - 1) < 5e-4, "original equality");
            }
            if (width) {
                auto invalid = problems; invalid.back().objective[0] = NAN;
                bool rejected = false;
                try { plan.solve_batch(invalid, options); }
                catch (const std::invalid_argument &e) { rejected = std::string(e.what()).find("objective") != std::string::npos; }
                check(rejected, "invalid final member rejected before execution");
                auto timeout = options; timeout.time_sec_limit = 0;
                auto expired = plan.solve_batch(problems, timeout);
                for (const auto &member : expired.results)
                    check(!member.has_solution && member.result->termination_reason == TERMINATION_REASON_TIME_LIMIT,
                          "unstarted request deadline");
            }
            std::cout << (gpu ? "gpu" : "cpu") << " width=" << width << " passed\n";
        }
        std::cout << "batch solver passed\n";
    } catch (const std::exception &e) { std::cerr << e.what() << '\n'; return 1; }
}
