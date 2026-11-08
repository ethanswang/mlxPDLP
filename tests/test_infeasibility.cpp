// Copyright 2026 Ethan Wang <ethanshurui.wang@gmail.com>
// SPDX-License-Identifier: Apache-2.0
// Regression models adapted from cuPDLPx PR #112 (Apache-2.0).
#include "mlxPDLP/batch_solver.h"
#include "shared_matrix.h"
#include <cmath>
#include <iostream>
#include <random>
#include <stdexcept>

using namespace mlxpdlp;
using Matrix = std::vector<std::vector<double>>;

void require(bool condition, const char *message) {
    if (!condition) throw std::runtime_error(message);
}

OwnedSolveResult solve(const Matrix &a, const std::vector<double> &c,
                      const std::vector<double> &lower, const std::vector<double> &upper,
                      const pdhg_parameters_t &parameters, mx::Device device,
                      const std::vector<double> &x = {}, const std::vector<double> &y = {}) {
    const int n = c.size(), m = a.size();
    std::vector<int> rp{0}, ci;
    std::vector<double> values, lb(n, 0), ub(n, INFINITY);
    for (const auto &row : a) {
        require(row.size() == c.size(), "fixture dimensions");
        for (int j = 0; j < n; ++j) if (row[j] != 0) {
            ci.push_back(j); values.push_back(row[j]);
        }
        rp.push_back(values.size());
    }
    MlxPdlpSolver solver(n, m, rp.data(), ci.data(), values.data(), lb.data(), ub.data(),
                         lower.data(), upper.data(), c.data(), 0, &parameters,
                         x.empty() ? nullptr : x.data(), y.empty() ? nullptr : y.data(), device);
    return OwnedSolveResult(solver.solve());
}

void bounded_models(pdhg_parameters_t p, mx::Device device) {
    for (bool presolve : {false, true}) {
#ifndef MLXPDLP_TEST_HAS_PRESOLVE
        if (presolve) continue;
#endif
        p.presolve = presolve;
        for (bool dummy : {false, true}) {
            Matrix a = dummy ? Matrix{{1,1,0},{0,0,1e10}} : Matrix{{1,1},{-1e10,1e10}};
            std::vector<double> c = dummy ? std::vector<double>{-1e10,-1.001e10,0}
                                         : std::vector<double>{-1e10,-1.001e10};
            auto r = solve(a,c,{-INFINITY,0},{2,dummy ? 0 : INFINITY},p,device);
            require(r->termination_reason == TERMINATION_REASON_OPTIMAL, "large-objective bounded status");
            require(std::abs(r->primal_objective_value / -2.002e10 - 1) < 1e-4, "large-objective optimum");
        }
        for (int seed = 0; seed < 6; ++seed) {
            std::mt19937 rng(seed);
            std::uniform_real_distribution<double> value(0.1,2.0);
            Matrix a(7,std::vector<double>(5,0));
            std::vector<double> c(5,0), lo(7,-INFINITY), hi(7,0);
            for (int i = 0; i < 6; ++i) for (int j = 0; j < 4; ++j) {
                a[i][j] = value(rng); hi[i] += a[i][j];
            }
            double feasible_objective = 0;
            for (int j = 0; j < 4; ++j) { c[j] = -1e10*value(rng); feasible_objective += c[j]; }
            a[6][4] = 1e12; lo[6] = 0;
            auto r = solve(a,c,lo,hi,p,device);
            require(r->termination_reason == TERMINATION_REASON_OPTIMAL, "random bounded status");
            require(r->primal_objective_value <= feasible_objective + 1e-4*std::abs(feasible_objective),
                    "random feasible-point objective bound");
        }
        for (double scale : {1e-6,1.0,1e8}) {
            auto r = solve({{1,2},{0,1},{3,2}},{scale,scale},
                           {10,-INFINITY,-INFINITY},{10,2,8},p,device);
            require(r->termination_reason == TERMINATION_REASON_PRIMAL_INFEASIBLE, "objective-scale invariant detection");
        }
    }
}

void candidates_and_controls(pdhg_parameters_t p, mx::Device device) {
    p.presolve = false;
    p.geometric_mean_iterations = p.l_inf_ruiz_iterations = 0;
    p.has_pock_chambolle_alpha = p.bound_objective_rescaling = false;
    p.termination_evaluation_frequency = 10;
    p.termination_criteria.iteration_limit = 20;
    // Huge warm starts erase small fixed-point deltas even in FP64. The
    // current-iterate candidates still give exact nonzero certificates.
    for (int mode = 0; mode < 3; ++mode) {
        p.infeasibility_detection = mode != 1;
        p.termination_criteria.infeasible_ray_objective_margin = mode == 2 ? 2 : 1e-8;
        auto primal = solve({{1},{1}},{0},{1,2},{1,2},p,device,{1.5},{-1e20,1e20});
        auto dual = solve({{1}},{-1},{0},{INFINITY},p,device,{1e20});
        require(primal->termination_reason == (mode == 0 ? TERMINATION_REASON_PRIMAL_INFEASIBLE
                                                         : TERMINATION_REASON_ITERATION_LIMIT), "dual candidate / control");
        require(dual->termination_reason == (mode == 0 ? TERMINATION_REASON_DUAL_INFEASIBLE
                                                      : TERMINATION_REASON_ITERATION_LIMIT), "primal candidate / control");
    }
    p.infeasibility_detection = true;
    p.termination_criteria.infeasible_ray_objective_margin = 1e-8;
    p.termination_evaluation_frequency = 1;
    p.termination_criteria.iteration_limit = 5;
    auto bounded = solve({{1,1,0},{0,0,1e10}},{-1e16,-1.001e16,0},
                          {-INFINITY,0},{2,0},p,device);
    require(bounded->termination_reason == TERMINATION_REASON_ITERATION_LIMIT,
            "large objective must not hide a violated recession constraint");
    auto zero_primal = solve({{0}},{0},{1},{1},p,device);
    auto zero_dual = solve({{0}},{-1},{0},{0},p,device);
    require(zero_primal->termination_reason == TERMINATION_REASON_PRIMAL_INFEASIBLE &&
            zero_dual->termination_reason == TERMINATION_REASON_DUAL_INFEASIBLE,
            "zero activity certifies only zero violation");

    if (device.type == mx::Device::gpu) for (bool infeasible : {false,true}) {
        const int rp[] = {0,1,2}, ci[] = {0,0};
        const double a[] = {1,1};
        const int m = infeasible ? 2 : 1;
        SharedMatrixPlan plan(1,m,rp,ci,a,&p,device);
        BatchProblem problem;
        problem.objective = {infeasible ? 0.0 : -1.0}; problem.variable_lower_bounds = {0};
        problem.constraint_lower_bounds = infeasible ? std::vector<double>{1,2} : std::vector<double>{0};
        problem.constraint_upper_bounds = infeasible ? std::vector<double>{1,2} : std::vector<double>{INFINITY};
        problem.primal_start = std::vector<double>{infeasible ? 1.5 : 1e20};
        if (infeasible) problem.dual_start = std::vector<double>{-1e20,1e20};
        BatchOptions options; options.execution = BatchExecution::shared;
        for (bool enabled : {true,false}) {
            p.infeasibility_detection = enabled;
            auto result = plan.solve_batch({problem,problem},options,&p);
            for (const auto &member : result.results) {
                const auto expected = enabled ? (infeasible ? TERMINATION_REASON_PRIMAL_INFEASIBLE
                                                            : TERMINATION_REASON_DUAL_INFEASIBLE)
                                              : TERMINATION_REASON_ITERATION_LIMIT;
                require(member.result->termination_reason == expected, "shared detection control");
                const auto &ray = infeasible ? member.dual_ray : member.primal_ray;
                require(enabled ? ray.size() == size_t(m) && ray.back() > 0
                                : ray.empty(), "shared selected ray export");
                if (enabled && infeasible)
                    require(std::abs(ray[0] + ray[1]) < 1e-14 && ray[0] + 2*ray[1] > 0,
                            "exported dual ray certifies the original model");
            }
        }
    }
}

int main(int argc, char **argv) {
    const auto device = argc > 1 && std::string(argv[1]) == "metal" ? mx::Device::gpu : mx::Device::cpu;
    if (device == mx::Device::gpu && !mx::is_available(device)) return 77;
    try {
        const std::vector<int32_t> rp{0,4,6}, ci{1,0,1,1,0,0};
        const auto absolute = detail::absolute_csr_values(rp,ci,{1e10,3,-1e10,1,-4,1});
        const double direction[] = {2,5}, expected[] = {11,6};
        for (int row = 0; row < 2; ++row) {
            double activity = 0;
            for (int k = rp[row]; k < rp[row+1]; ++k) activity += absolute[k]*direction[ci[k]];
            require(activity == expected[row], "duplicate CSR cancellation must not inflate activity");
        }
        pdhg_parameters_t p;
        mlxpdlp_set_default_parameters(&p);
        p.verbose = false;
        p.termination_criteria.iteration_limit = 20000;
        p.termination_criteria.time_sec_limit = 30;
        bounded_models(p,device);
        candidates_and_controls(p,device);
        std::cout << "Infeasibility regression checks passed\n";
    } catch (const std::exception &e) {
        std::cerr << e.what() << '\n'; return 1;
    }
}
