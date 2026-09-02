/*
Copyright 2025 Haihao Lu
Copyright 2026 Ethan Wang <ethanshurui.wang@gmail.com>

This file is derived from cuPDLPx (https://github.com/MIT-Lu-Lab/cuPDLPx),
ported from CUDA to Apple MLX/Metal and modified for the mlxPDLP Metal FP32
and CPU FP64 backends.

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

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

using namespace mlxpdlp;

// Local copy (same as solver.cpp static function)
static const char *term_str(termination_reason_t r) {
    switch (r) {
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
    case TERMINATION_REASON_HOST_DOUBLE_HANDOFF:
        return "HOST_DOUBLE_HANDOFF";
    default:
        return "UNSPECIFIED";
    }
}

// ---------------------------------------------------------------------------
// Helper: test harness
// ---------------------------------------------------------------------------

static int tests_run = 0;
static int tests_passed = 0;

#define TEST(name)                                                                                 \
    do {                                                                                           \
        tests_run++;                                                                               \
        printf("  TEST %s ... ", name);                                                            \
    } while (0)

#define PASS()                                                                                     \
    do {                                                                                           \
        tests_passed++;                                                                            \
        printf("PASS\n");                                                                          \
    } while (0)

#define FAIL(msg)                                                                                  \
    do {                                                                                           \
        printf("FAIL: %s\n", msg);                                                                 \
    } while (0)

#define CHECK(cond, msg)                                                                           \
    do {                                                                                           \
        if (!(cond)) {                                                                             \
            FAIL(msg);                                                                             \
            goto test_cleanup;                                                                     \
        }                                                                                          \
    } while (0)

#define CHECK_CLOSE(a, b, tol, msg)                                                                \
    do {                                                                                           \
        if (std::fabs((a) - (b)) > (tol)) {                                                        \
            printf("FAIL: %s (got %.10f, expected %.10f, diff %.2e)\n", msg, (double)(a),          \
                   (double)(b), std::fabs((double)(a) - (double)(b)));                             \
            goto test_cleanup;                                                                     \
        }                                                                                          \
    } while (0)

// Apply a safe iteration limit to prevent infinite loops
static void set_safe_limits(pdhg_parameters_t *params) {
    params->termination_criteria.iteration_limit = 100000;
    params->termination_criteria.time_sec_limit = 60.0;
    // Numerical solver tests exercise PDHG directly. Presolve has dedicated
    // tests below so it cannot hide regressions by solving tiny fixtures early.
    params->presolve = false;
}

static double distance_to_interval(double value, double lower, double upper) {
    if (value < lower)
        return lower - value;
    if (value > upper)
        return value - upper;
    return 0.0;
}

static double original_relative_primal_residual(int m, const int *row_ptr, const int *col_ind,
                                                const double *values, const double *con_lb,
                                                const double *con_ub,
                                                const mlxpdlp_result_t *result) {
    long double residual_sq = 0.0;
    long double bound_norm_sq = 0.0;
    for (int row = 0; row < m; ++row) {
        long double activity = 0.0;
        for (int entry = row_ptr[row]; entry < row_ptr[row + 1]; ++entry) {
            activity += static_cast<long double>(values[entry]) *
                        result->primal_solution[col_ind[entry]];
        }
        const double violation =
            distance_to_interval(static_cast<double>(activity), con_lb[row], con_ub[row]);
        residual_sq += static_cast<long double>(violation) * violation;
        if (std::isfinite(con_lb[row]))
            bound_norm_sq += static_cast<long double>(con_lb[row]) * con_lb[row];
        if (std::isfinite(con_ub[row]))
            bound_norm_sq += static_cast<long double>(con_ub[row]) * con_ub[row];
    }
    return std::sqrt(static_cast<double>(residual_sq)) /
           (1.0 + std::sqrt(static_cast<double>(bound_norm_sq)));
}

static double original_relative_dual_residual(int m, int n, const int *row_ptr,
                                              const int *col_ind, const double *values,
                                              const double *objective,
                                              const mlxpdlp_result_t *result) {
    auto *aty = new double[static_cast<size_t>(n)]();
    for (int row = 0; row < m; ++row) {
        for (int entry = row_ptr[row]; entry < row_ptr[row + 1]; ++entry) {
            aty[col_ind[entry]] += values[entry] * result->dual_solution[row];
        }
    }
    long double residual_sq = 0.0;
    long double objective_norm_sq = 0.0;
    for (int column = 0; column < n; ++column) {
        const double residual =
            objective[column] - aty[column] - result->reduced_cost[column];
        residual_sq += static_cast<long double>(residual) * residual;
        objective_norm_sq +=
            static_cast<long double>(objective[column]) * objective[column];
    }
    delete[] aty;
    return std::sqrt(static_cast<double>(residual_sq)) /
           (1.0 + std::sqrt(static_cast<double>(objective_norm_sq)));
}

// ---------------------------------------------------------------------------
// Test 0: Default parameters
// ---------------------------------------------------------------------------

static void test_default_parameters() {
    TEST("default parameters");
    pdhg_parameters_t params;
    mlxpdlp_set_default_parameters(&params);
    CHECK(params.geometric_mean_iterations == 12,
          "wrong geometric_mean_iterations");
    CHECK(params.l_inf_ruiz_iterations == 10, "wrong l_inf_ruiz_iterations");
    CHECK(params.termination_evaluation_frequency == 200, "wrong eval_freq");
    CHECK(params.conditional_termination_evaluation,
          "conditional early termination checks should default on");
    CHECK(params.sv_max_iter == 200, "wrong sv_max_iter");
    CHECK(params.sv_tol == 1e-4, "wrong sv_tol");
    CHECK(params.reflection_coefficient == 1.0, "wrong reflection_coeff");
    CHECK(params.restart_params.artificial_restart_threshold == 0.36, "wrong restart threshold");
    CHECK(params.restart_params.k_p == 0.99, "wrong k_p");
    CHECK(params.termination_criteria.eps_optimal_relative == 1e-4, "wrong eps_opt");
    CHECK(params.termination_criteria.eps_feasible_relative == 1e-4, "wrong eps_feas");
    CHECK(params.termination_criteria.eps_feas_polish_relative == 1e-6,
          "wrong eps_feas_polish");
    CHECK(!params.presolve_primal_propagation,
          "primal propagation should default off for stable fp32 postsolve");
    CHECK(!params.host_double_polishing, "host-double polishing should be opt-in");
    CHECK(params.host_double_early_handoff,
          "enabled host-double polishing should hand off from the fp32 floor");
    CHECK(params.host_double_polishing_iteration_limit == 50000,
          "wrong host-double iteration limit");
    CHECK(params.host_double_polishing_time_sec_limit == 30.0,
          "wrong host-double time limit");
    CHECK(!params.presolve_singleton_columns,
          "singleton-column presolve should use the safe opt-in default");
    CHECK(params.presolve_doubleton_equations,
          "doubleton-equation presolve should be enabled");
    CHECK(params.presolve_parallel_rows,
          "parallel-row presolve should be enabled");
    CHECK(params.presolve_parallel_columns,
          "parallel-column presolve should be enabled");
    CHECK(params.presolve_dual_fix, "dual fixing should be enabled");
    CHECK(params.presolve_finite_bound_tightening,
          "finite-bound tightening should be enabled");
#ifdef MLXPDLP_TEST_HAS_PRESOLVE
    CHECK(params.presolve, "presolve should default on when PSLP is built");
#else
    CHECK(!params.presolve, "presolve should default off when PSLP is not built");
#endif
    PASS();
test_cleanup:;
}

// ---------------------------------------------------------------------------
// Geometric-mean scaling must match the alternating Tomlin row/column update
// added to cuPDLPx in #103.
// ---------------------------------------------------------------------------

static void test_geometric_mean_scaling() {
    TEST("geometric-mean scaling factors");

    int row_ptr[] = {0, 2, 4};
    int col_ind[] = {0, 1, 0, 1};
    double vals[] = {1.0, 100.0, 10.0, 1.0};
    double obj[] = {1.0, 1.0};
    double con_lb[] = {0.0, 0.0};
    double con_ub[] = {1.0, 1.0};
    double var_lb[] = {0.0, 0.0};
    double var_ub[] = {INFINITY, INFINITY};

    pdhg_parameters_t params;
    mlxpdlp_set_default_parameters(&params);
    params.verbose = false;
    params.presolve = false;
    params.geometric_mean_iterations = 1;
    params.curtis_reid_iterations = 0;
    params.l_inf_ruiz_iterations = 0;
    params.has_pock_chambolle_alpha = false;
    params.bound_objective_rescaling = false;
    params.sv_max_iter = 2;
    params.termination_evaluation_frequency = 2;
    params.termination_criteria.iteration_limit = 0;
    params.termination_criteria.time_sec_limit = 10.0;

    mlxpdlp_result_t *result = nullptr;
    MlxPdlpSolver solver(2, 2, row_ptr, col_ind, vals, var_lb, var_ub, con_lb, con_ub, obj, 0.0,
                         &params);
    result = solver.solve();

    const auto &state = solver.state();
    const double *scaled = state.A.data<double>();
    const double *row_scale = state.con_rescale.data<double>();
    const double *column_scale = state.var_rescale.data<double>();
    const double small = std::pow(10.0, -0.75);
    const double large = std::pow(10.0, 0.75);

    CHECK_CLOSE(row_scale[0], 10.0, 1e-12, "row scale 0");
    CHECK_CLOSE(row_scale[1], std::sqrt(10.0), 1e-12, "row scale 1");
    CHECK_CLOSE(column_scale[0], std::pow(10.0, -0.25), 1e-12, "column scale 0");
    CHECK_CLOSE(column_scale[1], std::pow(10.0, 0.25), 1e-12, "column scale 1");
    CHECK_CLOSE(scaled[0], small, 1e-12, "scaled A[0,0]");
    CHECK_CLOSE(scaled[1], large, 1e-12, "scaled A[0,1]");
    CHECK_CLOSE(scaled[2], large, 1e-12, "scaled A[1,0]");
    CHECK_CLOSE(scaled[3], small, 1e-12, "scaled A[1,1]");

    mlxpdlp_result_free(result);
    PASS();
    return;

test_cleanup:
    mlxpdlp_result_free(result);
}

// ---------------------------------------------------------------------------
// Test 1: Simple LP — same as CUDA test_basic.py
//
//   minimize  x0 + x1
//   s.t.     1*x0 + 2*x1 = 5    (row 0: l=5, u=5)
//            0*x0 + 1*x1 <= 2   (row 1: l=-inf, u=2)
//            3*x0 + 2*x1 <= 8   (row 2: l=-inf, u=8)
//            x0 >= 0, x1 >= 0   (no upper bound)
//
// Optimal: x* = [1.0, 2.0], y* = [1.0, -1.0, 0.0], obj = 3.0
// ---------------------------------------------------------------------------

static void test_simple_lp() {
    TEST("simple LP (2 vars, 3 cons)");

    int m = 3, n = 2;

    int row_ptr[] = {0, 2, 4, 6};
    int col_ind[] = {0, 1, 0, 1, 0, 1};
    double vals[] = {1.0, 2.0, 0.0, 1.0, 3.0, 2.0};

    double obj[] = {1.0, 1.0};
    double con_lb[] = {5.0, -INFINITY, -INFINITY};
    double con_ub[] = {5.0, 2.0, 8.0};
    double var_lb[] = {0.0, 0.0};
    double var_ub[] = {INFINITY, INFINITY};

    pdhg_parameters_t params;
    mlxpdlp_set_default_parameters(&params);
    params.verbose = false;
    params.termination_evaluation_frequency = 50;
    params.termination_criteria.eps_optimal_relative = 1e-6;
    params.termination_criteria.eps_feasible_relative = 1e-6;
    set_safe_limits(&params);

    MlxPdlpSolver solver(n, m, row_ptr, col_ind, vals, var_lb, var_ub, con_lb, con_ub, obj, 0.0,
                         &params);
    mlxpdlp_result_t *result = solver.solve();

    double tol = 5e-3;
    double expected_x[] = {1.0, 2.0};
    double expected_y[] = {1.0, -1.0, 0.0};
    double expected_obj = 3.0;

    CHECK_CLOSE(result->primal_solution[0], expected_x[0], tol, "x[0]");
    CHECK_CLOSE(result->primal_solution[1], expected_x[1], tol, "x[1]");
    CHECK_CLOSE(result->dual_solution[0], expected_y[0], tol, "y[0]");
    CHECK_CLOSE(result->dual_solution[1], expected_y[1], tol, "y[1]");
    CHECK_CLOSE(result->dual_solution[2], expected_y[2], tol, "y[2]");
    CHECK_CLOSE(result->primal_objective_value, expected_obj, tol, "primal obj");
    CHECK(result->termination_reason == TERMINATION_REASON_OPTIMAL, "should be OPTIMAL");
    printf("(iter=%d) ", result->total_count);

    mlxpdlp_result_free(result);
    PASS();
test_cleanup:;
}

// ---------------------------------------------------------------------------
// Test 2: Medium LP with equality and inequality constraints
//
//   minimize  x0 + 2*x1 + 3*x2
//   s.t.      x0 + x1 + x2 = 6
//             2*x0 + 3*x1 + x2 <= 10
//             x0, x1, x2 >= 0
//
// Optimal: x = [4, 0, 2], obj = 10
// (From equality: x2=6-x0-x1. Sub into ineq: 2x0+3x1+(6-x0-x1)<=10 → x0+2x1<=4.
//  Obj: x0+2x1+3(6-x0-x1)=18-2x0-x1. Min at max of 2x0+x1 s.t. x0+2x1<=4.
//  Max of 2x0+x1 on polytope: at (4,0), value=8, obj=10, x2=2.)
// ---------------------------------------------------------------------------

static void test_medium_lp() {
    TEST("medium LP (3 vars, 2 cons)");

    int m = 2, n = 3;

    int row_ptr[] = {0, 3, 6};
    int col_ind[] = {0, 1, 2, 0, 1, 2};
    double vals[] = {1.0, 1.0, 1.0, 2.0, 3.0, 1.0};

    double obj[] = {1.0, 2.0, 3.0};
    double con_lb[] = {6.0, -INFINITY};
    double con_ub[] = {6.0, 10.0};
    double var_lb[] = {0.0, 0.0, 0.0};
    double var_ub[] = {INFINITY, INFINITY, INFINITY};

    pdhg_parameters_t params;
    mlxpdlp_set_default_parameters(&params);
    params.verbose = false;
    params.termination_evaluation_frequency = 100;
    params.termination_criteria.eps_optimal_relative = 1e-6;
    params.termination_criteria.eps_feasible_relative = 1e-6;
    set_safe_limits(&params);

    MlxPdlpSolver solver(n, m, row_ptr, col_ind, vals, var_lb, var_ub, con_lb, con_ub, obj, 0.0,
                         &params);
    mlxpdlp_result_t *result = solver.solve();

    double tol = 5e-3;
    double expected_x[] = {4.0, 0.0, 2.0};
    double expected_obj = 10.0;
    double ax0; // declared before CHECK macros

    CHECK_CLOSE(result->primal_solution[0], expected_x[0], tol, "x[0]");
    CHECK_CLOSE(result->primal_solution[1], expected_x[1], tol, "x[1]");
    CHECK_CLOSE(result->primal_solution[2], expected_x[2], tol, "x[2]");
    CHECK_CLOSE(result->primal_objective_value, expected_obj, tol, "primal obj");
    CHECK(result->termination_reason == TERMINATION_REASON_OPTIMAL, "should be OPTIMAL");
    printf("(iter=%d) ", result->total_count);

    // Check feasibility: Ax should satisfy bounds
    ax0 = result->primal_solution[0] + result->primal_solution[1] + result->primal_solution[2];
    CHECK_CLOSE(ax0, 6.0, tol, "Ax[0] should = 6");

    mlxpdlp_result_free(result);
    PASS();
test_cleanup:;
}

// ---------------------------------------------------------------------------
// Test 3: Simple LP with tighter tolerances
// ---------------------------------------------------------------------------

static void test_simple_lp_tight() {
    TEST("simple LP with tighter tolerances");

    int m = 3, n = 2;

    int row_ptr[] = {0, 2, 4, 6};
    int col_ind[] = {0, 1, 0, 1, 0, 1};
    double vals[] = {1.0, 2.0, 0.0, 1.0, 3.0, 2.0};

    double obj[] = {1.0, 1.0};
    double con_lb[] = {5.0, -INFINITY, -INFINITY};
    double con_ub[] = {5.0, 2.0, 8.0};
    double var_lb[] = {0.0, 0.0};
    double var_ub[] = {INFINITY, INFINITY};

    pdhg_parameters_t params;
    mlxpdlp_set_default_parameters(&params);
    params.verbose = false;
    params.termination_evaluation_frequency = 50;
    // Use moderate tolerance (1e-8 is too tight for dense float32 path)
    params.termination_criteria.eps_optimal_relative = 1e-7;
    params.termination_criteria.eps_feasible_relative = 1e-7;
    set_safe_limits(&params);

    MlxPdlpSolver solver(n, m, row_ptr, col_ind, vals, var_lb, var_ub, con_lb, con_ub, obj, 0.0,
                         &params);
    mlxpdlp_result_t *result = solver.solve();

    double tol = 1e-4;
    double expected_x[] = {1.0, 2.0};
    double expected_obj = 3.0;

    CHECK_CLOSE(result->primal_solution[0], expected_x[0], tol, "x[0]");
    CHECK_CLOSE(result->primal_solution[1], expected_x[1], tol, "x[1]");
    CHECK_CLOSE(result->primal_objective_value, expected_obj, tol, "primal obj");
    // Accept OPTIMAL or ITERATION_LIMIT (dense matmul may not reach 1e-7)
    CHECK(result->termination_reason == TERMINATION_REASON_OPTIMAL ||
              result->termination_reason == TERMINATION_REASON_ITERATION_LIMIT,
          "should terminate (OPTIMAL or ITERATION_LIMIT)");
    printf("(iter=%d status=%s) ", result->total_count, term_str(result->termination_reason));

    mlxpdlp_result_free(result);
    PASS();
test_cleanup:;
}

// ---------------------------------------------------------------------------
// Test 4: LP with free variables
//
//   minimize  x0
//   s.t.      x0 + x1 = 3
//             x0 - x1 = 1
//             x0, x1 free
//
// Optimal: x = [2, 1], obj = 2
// ---------------------------------------------------------------------------

static void test_free_variables() {
    TEST("LP with free variables");

    int m = 2, n = 2;

    int row_ptr[] = {0, 2, 4};
    int col_ind[] = {0, 1, 0, 1};
    double vals[] = {1.0, 1.0, 1.0, -1.0};

    double obj[] = {1.0, 0.0};
    double con_lb[] = {3.0, 1.0};
    double con_ub[] = {3.0, 1.0};
    double var_lb[] = {-INFINITY, -INFINITY};
    double var_ub[] = {INFINITY, INFINITY};

    pdhg_parameters_t params;
    mlxpdlp_set_default_parameters(&params);
    params.verbose = false;
    params.termination_evaluation_frequency = 50;
    params.termination_criteria.eps_optimal_relative = 1e-6;
    params.termination_criteria.eps_feasible_relative = 1e-6;
    set_safe_limits(&params);

    MlxPdlpSolver solver(n, m, row_ptr, col_ind, vals, var_lb, var_ub, con_lb, con_ub, obj, 0.0,
                         &params);
    mlxpdlp_result_t *result = solver.solve();

    double tol = 5e-3;
    double expected_x[] = {2.0, 1.0};
    double expected_obj = 2.0;

    CHECK_CLOSE(result->primal_solution[0], expected_x[0], tol, "x[0]");
    CHECK_CLOSE(result->primal_solution[1], expected_x[1], tol, "x[1]");
    CHECK_CLOSE(result->primal_objective_value, expected_obj, tol, "primal obj");
    CHECK(result->termination_reason == TERMINATION_REASON_OPTIMAL, "should be OPTIMAL");
    printf("(iter=%d) ", result->total_count);

    mlxpdlp_result_free(result);
    PASS();
test_cleanup:;
}

// ---------------------------------------------------------------------------
// Test 5: Pure equality constraints
//
//   minimize  x0 + x1
//   s.t.      x0 + x1 = 2
//             x0, x1 >= -1, <= 2
//
// Any feasible point has x0+x1=2, obj = 2 always
// ---------------------------------------------------------------------------

static void test_pure_equality() {
    TEST("LP with pure equality and bounds");

    int m = 1, n = 2;

    int row_ptr[] = {0, 2};
    int col_ind[] = {0, 1};
    double vals[] = {1.0, 1.0};

    double obj[] = {1.0, 1.0};
    double con_lb[] = {2.0};
    double con_ub[] = {2.0};
    double var_lb[] = {-1.0, -1.0};
    double var_ub[] = {2.0, 2.0};

    pdhg_parameters_t params;
    mlxpdlp_set_default_parameters(&params);
    params.verbose = false;
    params.termination_evaluation_frequency = 100;
    params.termination_criteria.eps_optimal_relative = 1e-6;
    params.termination_criteria.eps_feasible_relative = 1e-6;
    set_safe_limits(&params);

    MlxPdlpSolver solver(n, m, row_ptr, col_ind, vals, var_lb, var_ub, con_lb, con_ub, obj, 0.0,
                         &params);
    mlxpdlp_result_t *result = solver.solve();

    double tol = 5e-3;
    double ax = result->primal_solution[0] + result->primal_solution[1];
    CHECK_CLOSE(ax, 2.0, tol, "x0+x1 should = 2");
    CHECK(result->primal_solution[0] >= -1.0 - tol, "x0 >= -1");
    CHECK(result->primal_solution[0] <= 2.0 + tol, "x0 <= 2");
    CHECK(result->primal_solution[1] >= -1.0 - tol, "x1 >= -1");
    CHECK(result->primal_solution[1] <= 2.0 + tol, "x1 <= 2");
    CHECK_CLOSE(result->primal_objective_value, 2.0, tol, "obj should be 2");
    CHECK(result->termination_reason == TERMINATION_REASON_OPTIMAL, "should be OPTIMAL");
    printf("(iter=%d) ", result->total_count);

    mlxpdlp_result_free(result);
    PASS();
test_cleanup:;
}

// ---------------------------------------------------------------------------
// Test 6: Result struct validation
// ---------------------------------------------------------------------------

static void test_result_struct() {
    TEST("result struct integrity");

    int m = 1, n = 1;

    int row_ptr[] = {0, 1};
    int col_ind[] = {0};
    double vals[] = {1.0};

    double obj[] = {1.0};
    double con_lb[] = {1.0};
    double con_ub[] = {2.0};
    double var_lb[] = {0.0};
    double var_ub[] = {INFINITY};

    pdhg_parameters_t params;
    mlxpdlp_set_default_parameters(&params);
    params.verbose = false;
    set_safe_limits(&params);

    MlxPdlpSolver solver(n, m, row_ptr, col_ind, vals, var_lb, var_ub, con_lb, con_ub, obj, 0.0,
                         &params);
    mlxpdlp_result_t *result = solver.solve();

    CHECK(result->num_variables == 1, "num_variables");
    CHECK(result->num_constraints == 1, "num_constraints");
    CHECK(result->primal_solution != nullptr, "primal_solution not null");
    CHECK(result->dual_solution != nullptr, "dual_solution not null");
    CHECK(result->reduced_cost != nullptr, "reduced_cost not null");
    CHECK(result->total_count > 0, "total_count > 0");
    CHECK(result->cumulative_time_sec > 0.0, "cumulative_time > 0");
    CHECK(std::isfinite(result->primal_objective_value), "finite primal obj");
    CHECK(result->objective_gap >= 0.0, "objective gap must be absolute");
    CHECK_CLOSE(result->relative_objective_gap,
                std::fabs(result->primal_objective_value - result->dual_objective_value) /
                    (1.0 + std::fabs(result->primal_objective_value) +
                     std::fabs(result->dual_objective_value)),
                1e-12, "CUDA-compatible relative objective gap");
    printf("(iter=%d) ", result->total_count);

    mlxpdlp_result_free(result);
    PASS();
test_cleanup:;
}

static void test_iteration_limit_returns_best_iterate() {
    TEST("iteration limit returns best evaluated iterate");

    int row_ptr[] = {0, 2, 3};
    int col_ind[] = {0, 1, 0};
    double vals[] = {1.0, 1.0, 1.0};
    double obj[] = {1.0, 2.0};
    double con_lb[] = {1.0, -INFINITY};
    double con_ub[] = {1.0, 0.75};
    double var_lb[] = {0.0, 0.0};
    double var_ub[] = {INFINITY, INFINITY};

    pdhg_parameters_t params;
    mlxpdlp_set_default_parameters(&params);
    params.verbose = false;
    params.presolve = false;
    params.termination_evaluation_frequency = 100;
    params.termination_criteria.eps_optimal_relative = 0.0;
    params.termination_criteria.eps_feasible_relative = 0.0;
    params.termination_criteria.iteration_limit = 550;
    params.termination_criteria.time_sec_limit = 60.0;

    MlxPdlpSolver solver(2, 2, row_ptr, col_ind, vals, var_lb, var_ub, con_lb, con_ub, obj, 0.0,
                         &params);
    mlxpdlp_result_t *result = solver.solve();

    const double returned_kkt =
        std::max({result->relative_primal_residual, result->relative_dual_residual,
                  result->relative_objective_gap});
    CHECK(result->termination_reason == TERMINATION_REASON_ITERATION_LIMIT,
          "zero tolerances should force the iteration limit");
    CHECK(result->total_count == 550,
          "a partial final block must stop exactly at the iteration limit");
    CHECK(solver.state().best_iteration >= 0, "a best iterate should be recorded");
    CHECK(solver.state().best_iteration <= result->total_count,
          "best iteration must belong to the completed solve");
    CHECK_CLOSE(returned_kkt, solver.state().best_relative_kkt_error, 1e-7,
                "returned KKT error should match the best checkpoint");

    mlxpdlp_result_free(result);
    PASS();
    return;

test_cleanup:
    mlxpdlp_result_free(result);
}

static void test_conditional_termination_evaluation() {
    TEST("conditional evaluation stops early without changing checkpoints");

    int row_ptr[] = {0, 1};
    int col_ind[] = {0};
    double vals[] = {1.0};
    double obj[] = {1.0};
    double con_lb[] = {1.0};
    double con_ub[] = {1.0};
    double var_lb[] = {0.0};
    double var_ub[] = {INFINITY};
    mlxpdlp_result_t *fixed = nullptr;
    mlxpdlp_result_t *conditional = nullptr;
    mlxpdlp_result_t *strict_fixed = nullptr;
    mlxpdlp_result_t *strict_conditional = nullptr;

    auto solve_case = [&](double tolerance, bool enable_conditional) {
        pdhg_parameters_t params;
        mlxpdlp_set_default_parameters(&params);
        params.verbose = false;
        params.presolve = false;
        params.feasibility_polishing = false;
        params.host_double_polishing = false;
        params.termination_evaluation_frequency = 200;
        params.conditional_termination_evaluation = enable_conditional;
        params.termination_criteria.eps_optimal_relative = tolerance;
        params.termination_criteria.eps_feasible_relative = tolerance;
        params.termination_criteria.iteration_limit = 2000;
        params.termination_criteria.time_sec_limit = 10.0;
        MlxPdlpSolver solver(1, 1, row_ptr, col_ind, vals, var_lb, var_ub,
                             con_lb, con_ub, obj, 0.0, &params,
                             mx::Device::cpu);
        return solver.solve();
    };

    fixed = solve_case(1e-3, false);
    conditional = solve_case(1e-3, true);
    CHECK(fixed->termination_reason == TERMINATION_REASON_OPTIMAL,
          "fixed-cadence control should converge");
    CHECK(conditional->termination_reason == TERMINATION_REASON_OPTIMAL,
          "conditional checkpoint should return a certified solution");
    CHECK(conditional->total_count < fixed->total_count,
          "near-convergence midpoint should stop before the next regular checkpoint");

    // At a stricter tolerance, the iteration-300 midpoint is not scheduled.
    // Both modes must therefore return the identical iteration-400 state.
    strict_fixed = solve_case(1e-4, false);
    strict_conditional = solve_case(1e-4, true);
    CHECK(strict_conditional->total_count == strict_fixed->total_count,
          "conditional checks must not perturb the regular checkpoint trajectory");
    CHECK_CLOSE(strict_conditional->primal_solution[0],
                strict_fixed->primal_solution[0], 1e-14,
                "regular checkpoint primal state should be unchanged");
    CHECK_CLOSE(strict_conditional->dual_solution[0],
                strict_fixed->dual_solution[0], 1e-14,
                "regular checkpoint dual state should be unchanged");

    mlxpdlp_result_free(fixed);
    mlxpdlp_result_free(conditional);
    mlxpdlp_result_free(strict_fixed);
    mlxpdlp_result_free(strict_conditional);
    PASS();
    return;

test_cleanup:
    mlxpdlp_result_free(fixed);
    mlxpdlp_result_free(conditional);
    mlxpdlp_result_free(strict_fixed);
    mlxpdlp_result_free(strict_conditional);
}

// Test 7: Repeated fresh solver instances (no memory leaks)
// ---------------------------------------------------------------------------

static void test_repeat_solve() {
    TEST("repeated fresh solver instances");

    int m = 3, n = 2;

    int row_ptr[] = {0, 2, 4, 6};
    int col_ind[] = {0, 1, 0, 1, 0, 1};
    double vals[] = {1.0, 2.0, 0.0, 1.0, 3.0, 2.0};

    double obj[] = {1.0, 1.0};
    double con_lb[] = {5.0, -INFINITY, -INFINITY};
    double con_ub[] = {5.0, 2.0, 8.0};
    double var_lb[] = {0.0, 0.0};
    double var_ub[] = {INFINITY, INFINITY};

    pdhg_parameters_t params;
    mlxpdlp_set_default_parameters(&params);
    params.verbose = false;
    params.termination_evaluation_frequency = 50;
    params.termination_criteria.eps_optimal_relative = 1e-6;
    params.termination_criteria.eps_feasible_relative = 1e-6;
    set_safe_limits(&params);

    double tol = 5e-3;
    double expected_obj = 3.0;

    for (int rep = 0; rep < 3; rep++) {
        MlxPdlpSolver solver(n, m, row_ptr, col_ind, vals, var_lb, var_ub, con_lb, con_ub, obj, 0.0,
                             &params);
        mlxpdlp_result_t *result = solver.solve();
        CHECK_CLOSE(result->primal_objective_value, expected_obj, tol, "obj");
        CHECK(result->termination_reason == TERMINATION_REASON_OPTIMAL, "should be OPTIMAL");
        mlxpdlp_result_free(result);
    }
    PASS();
test_cleanup:;
}

static void test_solver_rejects_second_solve() {
    TEST("solver rejects a second solve call");

    int row_ptr[] = {0, 2, 4, 6};
    int col_ind[] = {0, 1, 0, 1, 0, 1};
    double vals[] = {1.0, 2.0, 0.0, 1.0, 3.0, 2.0};
    double obj[] = {1.0, 1.0};
    double con_lb[] = {5.0, -INFINITY, -INFINITY};
    double con_ub[] = {5.0, 2.0, 8.0};
    double var_lb[] = {0.0, 0.0};
    double var_ub[] = {INFINITY, INFINITY};

    pdhg_parameters_t params;
    mlxpdlp_set_default_parameters(&params);
    params.verbose = false;
    params.termination_evaluation_frequency = 50;
    params.termination_criteria.eps_optimal_relative = 1e-6;
    params.termination_criteria.eps_feasible_relative = 1e-6;
    set_safe_limits(&params);

    mlxpdlp_result_t *first = nullptr;
    mlxpdlp_result_t *unexpected = nullptr;
    bool rejected = false;
    bool diagnostic_matches = false;
    MlxPdlpSolver solver(2, 3, row_ptr, col_ind, vals, var_lb, var_ub, con_lb,
                         con_ub, obj, 0.0, &params);
    first = solver.solve();
    mlxpdlp_result_free(first);
    first = nullptr;
    try {
        unexpected = solver.solve();
    } catch (const std::logic_error &error) {
        rejected = true;
        diagnostic_matches =
            std::strstr(error.what(), "may only be called once") != nullptr;
    }

    CHECK(rejected, "second solve call must throw std::logic_error");
    CHECK(diagnostic_matches, "second solve diagnostic must explain the single-use contract");

    mlxpdlp_result_free(unexpected);
    PASS();
    return;

test_cleanup:
    mlxpdlp_result_free(first);
    mlxpdlp_result_free(unexpected);
}

// ---------------------------------------------------------------------------
// Test 8: Deterministic random singular-value start avoids a symmetric nullspace
// ---------------------------------------------------------------------------

static void test_singular_value_nullspace_start() {
    TEST("singular-value random start avoids symmetric nullspace");

    int m = 2, n = 1;
    int row_ptr[] = {0, 1, 2};
    int col_ind[] = {0, 0};
    double vals[] = {1.0, -1.0};

    double obj[] = {1.0};
    double con_lb[] = {-INFINITY, -INFINITY};
    double con_ub[] = {INFINITY, INFINITY};
    double var_lb[] = {0.0};
    double var_ub[] = {1.0};

    pdhg_parameters_t params;
    mlxpdlp_set_default_parameters(&params);
    params.verbose = false;
    params.geometric_mean_iterations = 0;
    params.l_inf_ruiz_iterations = 0;
    params.has_pock_chambolle_alpha = false;
    params.bound_objective_rescaling = false;
    params.presolve = false;
    params.sv_max_iter = 20;
    params.sv_tol = 1e-6;
    params.termination_evaluation_frequency = 2;
    params.termination_criteria.iteration_limit = 2;
    params.termination_criteria.time_sec_limit = 10.0;

    MlxPdlpSolver solver(n, m, row_ptr, col_ind, vals, var_lb, var_ub, con_lb, con_ub, obj, 0.0,
                         &params);
    mlxpdlp_result_t *result = solver.solve();
    double step_size = solver.state().step_size;
    mlxpdlp_result_free(result);

    double expected_step_size = 0.998 / std::sqrt(2.0);
    CHECK_CLOSE(step_size, expected_step_size, 1e-6,
                "power method should recover the nonzero singular value");
    PASS();
test_cleanup:;
}

static void test_power_method_stops_on_relative_spectral_change() {
    TEST("power method stops on relative spectral change");

    constexpr int size = 256;
    constexpr int entries_per_row = 4;
    std::vector<int> row_ptr(static_cast<size_t>(size) + 1);
    std::vector<int> col_ind;
    std::vector<double> values;
    col_ind.reserve(static_cast<size_t>(size) * entries_per_row);
    values.reserve(static_cast<size_t>(size) * entries_per_row);
    for (int row = 0; row < size; ++row) {
        row_ptr[static_cast<size_t>(row)] = static_cast<int>(col_ind.size());
        for (int offset = 0; offset < entries_per_row; ++offset) {
            col_ind.push_back((row + offset) % size);
            values.push_back(1.0);
        }
    }
    row_ptr.back() = static_cast<int>(col_ind.size());

    std::vector<double> objective(static_cast<size_t>(size), 0.0);
    std::vector<double> variable_lower(static_cast<size_t>(size), 0.0);
    std::vector<double> variable_upper(static_cast<size_t>(size), INFINITY);
    std::vector<double> constraint_lower(static_cast<size_t>(size), -INFINITY);
    std::vector<double> constraint_upper(static_cast<size_t>(size), INFINITY);

    pdhg_parameters_t params;
    mlxpdlp_set_default_parameters(&params);
    params.verbose = false;
    params.presolve = false;
    params.feasibility_polishing = false;
    params.host_double_polishing = false;
    // One iteration beyond the new default keeps the old absolute-residual
    // implementation bounded while proving the relative criterion fires.
    params.sv_max_iter = 201;
    params.sv_tol = 1e-4;
    params.termination_evaluation_frequency = 2;
    params.termination_criteria.iteration_limit = 0;
    params.termination_criteria.time_sec_limit = 10.0;

    mlxpdlp_result_t *result = nullptr;
    MlxPdlpSolver solver(size, size, row_ptr.data(), col_ind.data(), values.data(),
                         variable_lower.data(), variable_upper.data(),
                         constraint_lower.data(), constraint_upper.data(),
                         objective.data(), 0.0, &params, mx::Device::cpu);
    result = solver.solve();
    const double estimated_sigma = 0.998 / solver.state().step_size;

    CHECK(solver.state().singular_value_iterations < params.sv_max_iter,
          "relative sigma-squared test should stop before the cap");
    CHECK_CLOSE(estimated_sigma, 1.0, 2e-3,
                "early spectral estimate should retain step-size accuracy");

    mlxpdlp_result_free(result);
    PASS();
    return;

test_cleanup:
    mlxpdlp_result_free(result);
}

// ---------------------------------------------------------------------------
// Test 9: Duplicate CSR coordinates represent additive matrix entries
// ---------------------------------------------------------------------------

static void test_duplicate_csr_entries() {
    TEST("duplicate CSR entries are accumulated");

    int m = 1, n = 1;
    int row_ptr[] = {0, 2};
    int col_ind[] = {0, 0};
    double vals[] = {2.0, 3.0};

    double obj[] = {1.0};
    double con_lb[] = {5.0};
    double con_ub[] = {5.0};
    double var_lb[] = {0.0};
    double var_ub[] = {INFINITY};

    pdhg_parameters_t params;
    mlxpdlp_set_default_parameters(&params);
    params.verbose = false;
    params.termination_evaluation_frequency = 50;
    params.termination_criteria.eps_optimal_relative = 1e-6;
    params.termination_criteria.eps_feasible_relative = 1e-6;
    set_safe_limits(&params);

    MlxPdlpSolver solver(n, m, row_ptr, col_ind, vals, var_lb, var_ub, con_lb, con_ub, obj, 0.0,
                         &params);
    mlxpdlp_result_t *result = solver.solve();
    double primal_value = result->primal_solution[0];
    double objective_value = result->primal_objective_value;
    termination_reason_t reason = result->termination_reason;
    mlxpdlp_result_free(result);

    CHECK_CLOSE(primal_value, 1.0, 5e-3, "duplicate coefficients should sum to 5");
    CHECK_CLOSE(objective_value, 1.0, 5e-3, "objective should use the accumulated matrix");
    CHECK(reason == TERMINATION_REASON_OPTIMAL, "should be OPTIMAL");
    PASS();
test_cleanup:;
}

#if defined(__APPLE__)
static void test_large_cpu_problem_uses_sparse_backend() {
    TEST("large sparse CPU problem avoids dense matrix storage");

    constexpr int dimension = 4096;
    std::vector<int> row_ptr(static_cast<size_t>(dimension) + 1);
    std::vector<int> col_ind(static_cast<size_t>(dimension));
    std::vector<double> values(static_cast<size_t>(dimension), 1.0);
    std::vector<double> objective(static_cast<size_t>(dimension), 0.0);
    std::vector<double> lower(static_cast<size_t>(dimension), 0.0);
    std::vector<double> upper(static_cast<size_t>(dimension), 1.0);
    for (int index = 0; index < dimension; ++index) {
        row_ptr[static_cast<size_t>(index)] = index;
        col_ind[static_cast<size_t>(index)] = index;
    }
    row_ptr.back() = dimension;

    pdhg_parameters_t params;
    mlxpdlp_set_default_parameters(&params);
    params.verbose = false;
    params.presolve = false;
    params.geometric_mean_iterations = 0;
    params.curtis_reid_iterations = 0;
    params.l_inf_ruiz_iterations = 0;
    params.has_pock_chambolle_alpha = false;
    params.bound_objective_rescaling = false;
    params.sv_max_iter = 2;
    params.termination_criteria.iteration_limit = 0;
    params.termination_criteria.time_sec_limit = 10.0;

    mlxpdlp_result_t *result = nullptr;
    {
        MlxPdlpSolver solver(
            dimension, dimension, row_ptr.data(), col_ind.data(), values.data(),
            lower.data(), upper.data(), lower.data(), upper.data(), objective.data(),
            0.0, &params, mx::Device::cpu);
        CHECK(solver.expects_sparse_cpu_backend(),
              "large low-density CPU matrix should select sparse storage");
        result = solver.solve();
        CHECK(solver.state().sparse_cpu_active,
              "Accelerate sparse CPU SpMV should be active after preprocessing");
        CHECK(solver.state().cpu_double_precision_active &&
                  solver.state().x_pdhg.dtype() == mx::float64 &&
                  solver.state().Ax.dtype() == mx::float64,
              "sparse CPU matrix and PDHG vectors should use float64");
        CHECK(!solver.state().sparse_metal_active,
              "CPU solve must not activate the Metal sparse backend");
        CHECK(std::isfinite(result->relative_primal_residual) &&
                  std::isfinite(result->relative_dual_residual),
              "sparse CPU residual computation should remain finite");
    }

    mlxpdlp_result_free(result);
    PASS();
    return;

test_cleanup:
    mlxpdlp_result_free(result);
}
#endif

// ---------------------------------------------------------------------------
// Residuals used for termination must be expressed in the original model's
// coordinates after nonuniform geometric-mean/Ruiz/Pock-Chambolle scaling.
// ---------------------------------------------------------------------------

static void test_reported_residuals_use_original_scaling() {
    TEST("reported residuals undo nonuniform scaling");

    int m = 2, n = 2;
    int row_ptr[] = {0, 2, 4};
    int col_ind[] = {0, 1, 0, 1};
    double vals[] = {10000.0, 1.0, 0.01, 2.0};
    double obj[] = {0.01, 100.0};
    double con_lb[] = {10001.0, 2.01};
    double con_ub[] = {10001.0, 2.01};
    double var_lb[] = {0.0, 0.0};
    double var_ub[] = {INFINITY, INFINITY};

    pdhg_parameters_t params;
    mlxpdlp_set_default_parameters(&params);
    params.verbose = false;
    params.presolve = false;
    params.termination_evaluation_frequency = 2;
    params.termination_criteria.eps_optimal_relative = 0.0;
    params.termination_criteria.eps_feasible_relative = 0.0;
    params.termination_criteria.iteration_limit = 2;
    params.termination_criteria.time_sec_limit = 10.0;

    mlxpdlp_result_t *result = nullptr;
    MlxPdlpSolver solver(n, m, row_ptr, col_ind, vals, var_lb, var_ub, con_lb, con_ub, obj, 0.0,
                         &params);
    result = solver.solve();

    const double expected_primal = original_relative_primal_residual(
        m, row_ptr, col_ind, vals, con_lb, con_ub, result);
    const double expected_dual =
        original_relative_dual_residual(m, n, row_ptr, col_ind, vals, obj, result);
    const double primal_error = std::fabs(result->relative_primal_residual - expected_primal);
    const double dual_error = std::fabs(result->relative_dual_residual - expected_dual);
    std::printf("(reported=(%.9e, %.9e) original=(%.9e, %.9e) error=(%.3e, %.3e)) ",
                result->relative_primal_residual, result->relative_dual_residual,
                expected_primal, expected_dual, primal_error, dual_error);

    // The iterate and scaled matrix are float32, while the independent check
    // uses the original double coefficients. Allow their expected rounding
    // difference while still rejecting the orders-of-magnitude scaling bug.
    CHECK(primal_error <= 2e-5 * (1.0 + expected_primal),
          "reported primal residual must use original row scaling");
    CHECK(dual_error <= 2e-5 * (1.0 + expected_dual),
          "reported dual residual must use original column scaling");

    mlxpdlp_result_free(result);
    PASS();
    return;

test_cleanup:
    mlxpdlp_result_free(result);
}

static void test_zero_norm_residual_normalization() {
    TEST("zero objective and bound norms remain zero");

    int row_ptr[] = {0, 1};
    int col_ind[] = {0};
    double vals[] = {1.0};
    double obj[] = {0.0};
    double con_lb[] = {0.0};
    double con_ub[] = {0.0};
    double var_lb[] = {1.0};
    double var_ub[] = {INFINITY};

    pdhg_parameters_t params;
    mlxpdlp_set_default_parameters(&params);
    params.verbose = false;
    params.presolve = false;
    params.feasibility_polishing = false;
    params.termination_evaluation_frequency = 2;
    params.termination_criteria.eps_optimal_relative = 0.0;
    params.termination_criteria.eps_feasible_relative = 0.0;
    params.termination_criteria.iteration_limit = 2;
    params.termination_criteria.time_sec_limit = 10.0;

    mlxpdlp_result_t *result = nullptr;
    MlxPdlpSolver solver(1, 1, row_ptr, col_ind, vals, var_lb, var_ub, con_lb, con_ub, obj, 0.0,
                         &params);
    result = solver.solve();

    const double expected_primal = original_relative_primal_residual(
        1, row_ptr, col_ind, vals, con_lb, con_ub, result);
    CHECK_CLOSE(solver.state().constraint_bound_norm, 0.0, 0.0,
                "zero constraint-bound norm must not be replaced");
    CHECK_CLOSE(solver.state().objective_vector_norm, 0.0, 0.0,
                "zero objective norm must not be replaced");
    CHECK_CLOSE(result->relative_primal_residual, expected_primal, 1e-6,
                "zero-bound residual denominator must be one");
    CHECK(expected_primal >= 0.99,
          "fixture should distinguish the correct denominator from the old factor of two");

    mlxpdlp_result_free(result);
    PASS();
    return;

test_cleanup:
    mlxpdlp_result_free(result);
}

static void test_feasibility_polishing_is_safeguarded() {
    TEST("feasibility polishing cannot worsen full KKT merit");

    int row_ptr[] = {0, 2, 4};
    int col_ind[] = {0, 1, 0, 1};
    double vals[] = {10000.0, 1.0, 0.01, 2.0};
    double obj[] = {0.01, 100.0};
    double con_lb[] = {10001.0, 2.01};
    double con_ub[] = {10001.0, 2.01};
    double var_lb[] = {0.0, 0.0};
    double var_ub[] = {INFINITY, INFINITY};
    mlxpdlp_result_t *baseline = nullptr;
    mlxpdlp_result_t *polished = nullptr;

    auto solve_case = [&](bool enable_polishing) {
        pdhg_parameters_t params;
        mlxpdlp_set_default_parameters(&params);
        params.verbose = false;
        params.presolve = false;
        params.feasibility_polishing = enable_polishing;
        params.termination_evaluation_frequency = 2;
        params.termination_criteria.eps_optimal_relative = 0.0;
        params.termination_criteria.eps_feasible_relative = 0.0;
        params.termination_criteria.eps_feas_polish_relative = 1e-6;
        // Stop before this fixture reaches primal feasibility so the polishing
        // phase is exercised independently of restart-controller tuning.
        params.termination_criteria.iteration_limit = 2;
        params.termination_criteria.time_sec_limit = 10.0;
        MlxPdlpSolver solver(2, 2, row_ptr, col_ind, vals, var_lb, var_ub, con_lb, con_ub,
                             obj, 0.0, &params);
        return solver.solve();
    };

    baseline = solve_case(false);
    polished = solve_case(true);
    const double baseline_merit =
        std::max({baseline->relative_primal_residual, baseline->relative_dual_residual,
                  baseline->relative_objective_gap});
    const double polished_merit =
        std::max({polished->relative_primal_residual, polished->relative_dual_residual,
                  polished->relative_objective_gap});

    CHECK(polished->feasibility_iteration > 0, "enabled polishing should execute a phase");
    CHECK(polished->feasibility_polishing_time > 0.0, "polishing time should be reported");
    CHECK(polished_merit <= baseline_merit + 1e-6 * (1.0 + baseline_merit),
          "polishing safeguard must retain the better full KKT point");

    mlxpdlp_result_free(baseline);
    mlxpdlp_result_free(polished);
    PASS();
    return;

test_cleanup:
    mlxpdlp_result_free(baseline);
    mlxpdlp_result_free(polished);
}

static void test_polish_shares_solve_time_budget() {
    TEST("polishing phases share the solve-level time budget");

    // With a zero time budget the main loop reaches TIME_LIMIT on its first
    // termination check. The polish phases must then be skipped entirely
    // (cuPDLPx measures polish time from the original solve start), rather
    // than each receiving its own full copy of the limit.
    int row_ptr[] = {0, 2, 4};
    int col_ind[] = {0, 1, 0, 1};
    double vals[] = {10000.0, 1.0, 0.01, 2.0};
    double obj[] = {0.01, 100.0};
    double con_lb[] = {10001.0, 2.01};
    double con_ub[] = {10001.0, 2.01};
    double var_lb[] = {0.0, 0.0};
    double var_ub[] = {INFINITY, INFINITY};
    mlxpdlp_result_t *result = nullptr;

    pdhg_parameters_t params;
    mlxpdlp_set_default_parameters(&params);
    params.verbose = false;
    params.presolve = false;
    params.feasibility_polishing = true;
    params.termination_evaluation_frequency = 2;
    params.termination_criteria.eps_optimal_relative = 0.0;
    params.termination_criteria.eps_feasible_relative = 0.0;
    params.termination_criteria.eps_feas_polish_relative = 1e-6;
    params.termination_criteria.iteration_limit = 100000;
    params.termination_criteria.time_sec_limit = 0.0;

    MlxPdlpSolver solver(2, 2, row_ptr, col_ind, vals, var_lb, var_ub, con_lb, con_ub,
                         obj, 0.0, &params);
    result = solver.solve();

    CHECK(result->termination_reason == TERMINATION_REASON_TIME_LIMIT,
          "exhausted budget should terminate with TIME_LIMIT");
    CHECK(result->feasibility_iteration == 0,
          "exhausted budget must skip both polish phases");
    CHECK(result->feasibility_polishing_time == 0.0,
          "skipped polish phases should report zero polishing time");

    mlxpdlp_result_free(result);
    PASS();
    return;

test_cleanup:
    mlxpdlp_result_free(result);
}

static void test_host_double_polishing_is_safeguarded() {
    TEST("host-double polishing is bounded and safeguarded");

    int row_ptr[] = {0, 2, 4};
    int col_ind[] = {0, 1, 0, 1};
    double vals[] = {10000.0, 1.0, 0.01, 2.0};
    double obj[] = {0.01, 100.0};
    double con_lb[] = {10001.0, 2.01};
    double con_ub[] = {10001.0, 2.01};
    double var_lb[] = {0.0, 0.0};
    double var_ub[] = {INFINITY, INFINITY};
    mlxpdlp_result_t *baseline = nullptr;
    mlxpdlp_result_t *polished = nullptr;

    auto solve_case = [&](bool enable_host_polishing) {
        pdhg_parameters_t params;
        mlxpdlp_set_default_parameters(&params);
        params.verbose = false;
        params.presolve = false;
        params.feasibility_polishing = false;
        params.host_double_polishing = enable_host_polishing;
        params.host_double_early_handoff = false;
        params.host_double_polishing_iteration_limit = 200;
        params.host_double_polishing_time_sec_limit = 2.0;
        params.termination_evaluation_frequency = 2;
        params.termination_criteria.eps_optimal_relative = 0.0;
        params.termination_criteria.eps_feasible_relative = 0.0;
        params.termination_criteria.iteration_limit = 200;
        params.termination_criteria.time_sec_limit = 10.0;
        MlxPdlpSolver solver(2, 2, row_ptr, col_ind, vals, var_lb, var_ub, con_lb, con_ub,
                             obj, 0.0, &params, mx::Device::cpu);
        return solver.solve();
    };

    baseline = solve_case(false);
    polished = solve_case(true);
    const double baseline_merit =
        std::max({baseline->relative_primal_residual, baseline->relative_dual_residual,
                  baseline->relative_objective_gap});
    const double polished_merit =
        std::max({polished->relative_primal_residual, polished->relative_dual_residual,
                  polished->relative_objective_gap});

    CHECK(polished->host_double_polishing_iteration > 0,
          "enabled host-double polishing should execute");
    CHECK(polished->host_double_polishing_iteration <= 200,
          "host-double polishing must respect its iteration cap");
    CHECK(polished->host_double_polishing_time > 0.0,
          "host-double polishing time should be reported");
    CHECK(polished_merit <= baseline_merit + 1e-10 * (1.0 + baseline_merit),
          "host-double safeguard must retain the better full KKT point");

    mlxpdlp_result_free(baseline);
    mlxpdlp_result_free(polished);
    PASS();
    return;

test_cleanup:
    mlxpdlp_result_free(baseline);
    mlxpdlp_result_free(polished);
}

static void test_host_double_repairs_primal_without_changing_objective() {
    TEST("host-double objective-neutral descent repairs a primal-only blocker");

    int row_ptr[] = {0, 1};
    int col_ind[] = {0};
    double vals[] = {1.0};
    double obj[] = {0.0};
    double con_lb[] = {1.0};
    double con_ub[] = {1.0};
    double var_lb[] = {0.0};
    double var_ub[] = {INFINITY};
    // The normalized residual is 1.5e-1, beyond the ordinary 1e-2 host gate.
    // A complete original-model certificate with passing dual/gap metrics is
    // deliberately admitted to the safeguarded objective-neutral phase.
    double primal_start[] = {0.7};
    double dual_start[] = {0.0};
    mlxpdlp_result_t *result = nullptr;

    pdhg_parameters_t params;
    mlxpdlp_set_default_parameters(&params);
    params.verbose = false;
    params.presolve = false;
    params.feasibility_polishing = false;
    params.host_double_polishing = true;
    params.host_double_early_handoff = false;
    params.host_double_polishing_iteration_limit = 50;
    params.host_double_polishing_time_sec_limit = 2.0;
    params.termination_evaluation_frequency = 2;
    params.termination_criteria.eps_optimal_relative = 1e-6;
    params.termination_criteria.eps_feasible_relative = 1e-6;
    // Keep the supplied checkpoint untouched by the device loop so this test
    // isolates the fp64 objective-neutral row-projection phase.
    params.termination_criteria.iteration_limit = 0;
    params.termination_criteria.time_sec_limit = 10.0;

    MlxPdlpSolver solver(1, 1, row_ptr, col_ind, vals, var_lb, var_ub,
                         con_lb, con_ub, obj, 0.0, &params, primal_start,
                         dual_start, mx::Device::cpu);
    result = solver.solve();

    CHECK(result->host_double_polishing_iteration > 0,
          "primal-only checkpoint should execute host feasibility descent");
    CHECK(result->host_double_polishing_iteration <= 50,
          "host feasibility descent must share the configured iteration cap");
    CHECK(result->relative_primal_residual < 1e-6,
          "host feasibility descent should repair the equality residual");
    CHECK(result->relative_dual_residual < 1e-12,
          "zero objective certificate should remain dual feasible");
    CHECK(result->relative_objective_gap < 1e-12,
          "objective-neutral correction should preserve the zero gap");

    mlxpdlp_result_free(result);
    PASS();
    return;

test_cleanup:
    mlxpdlp_result_free(result);
}

static void test_fp64_cpu_bypasses_host_double_handoff() {
    TEST("FP64 CPU bypasses the Metal host-double handoff");

    int row_ptr[] = {0, 1};
    int col_ind[] = {0};
    double vals[] = {1.0};
    // This increment is below one FP32 ULP at one. Keeping it in the live CPU
    // state proves that the backend does not merely advertise a double dtype.
    double obj[] = {1.0 + 1e-10};
    double con_lb[] = {1.0};
    double con_ub[] = {1.0};
    double var_lb[] = {0.0};
    double var_ub[] = {INFINITY};
    double primal_start[] = {1.0};
    double dual_start[] = {1.0};
    mlxpdlp_result_t *result = nullptr;

    pdhg_parameters_t params;
    mlxpdlp_set_default_parameters(&params);
    params.verbose = false;
    params.presolve = false;
    params.feasibility_polishing = false;
    params.host_double_polishing = true;
    params.host_double_early_handoff = true;
    params.host_double_polishing_iteration_limit = 200;
    params.host_double_polishing_time_sec_limit = 2.0;
    params.termination_evaluation_frequency = 2;
    // Strict comparisons make zero feasibility tolerance a deterministic way
    // to keep the exact warm start out of the ordinary OPTIMAL branch.
    params.termination_criteria.eps_optimal_relative = 1e-6;
    params.termination_criteria.eps_feasible_relative = 0.0;
    params.termination_criteria.iteration_limit = 200;
    params.termination_criteria.time_sec_limit = 10.0;

    MlxPdlpSolver solver(1, 1, row_ptr, col_ind, vals, var_lb, var_ub,
                         con_lb, con_ub, obj, 0.0, &params, primal_start,
                         dual_start, mx::Device::cpu);
    mx::eval(solver.state().obj);
    CHECK_CLOSE(solver.state().obj.data<double>()[0], obj[0], 1e-13,
                "CPU objective should preserve precision below one FP32 ULP");
    result = solver.solve();

    CHECK(solver.state().cpu_double_precision_active,
          "CPU solver should advertise full float64 arithmetic");
    CHECK(solver.state().x_pdhg.dtype() == mx::float64,
          "CPU primal state should remain float64");
    CHECK(!result->host_double_handoff,
          "a float64 CPU trajectory must not use the float32-to-host handoff");
    CHECK_CLOSE(result->primal_solution[0], 1.0, 1e-9,
                "FP64 CPU should retain the admissible primal point");

    mlxpdlp_result_free(result);
    PASS();
    return;

test_cleanup:
    mlxpdlp_result_free(result);
}

static void test_host_double_reconstructs_dual_without_moving_primal() {
    TEST("host-double polishing reconstructs a dual certificate at fixed primal");

    int row_ptr[] = {0, 1, 2};
    int col_ind[] = {0, 0};
    double vals[] = {1.0, 1.0};
    double obj[] = {1.0};
    double con_lb[] = {1.0, -INFINITY};
    double con_ub[] = {1.0, 2.0};
    double var_lb[] = {0.0};
    double var_ub[] = {INFINITY};
    double primal_start[] = {1.0};
    // The second row is strictly inactive at x=1. Its deliberately invalid
    // multiplier verifies that complementary host reconstruction zeros
    // inactive-row multipliers while recovering the equality multiplier.
    double dual_start[] = {0.0, -1.0};
    mlxpdlp_result_t *baseline = nullptr;
    mlxpdlp_result_t *polished = nullptr;

    auto solve_case = [&](bool enable_host_polishing) {
        pdhg_parameters_t params;
        mlxpdlp_set_default_parameters(&params);
        params.verbose = false;
        params.presolve = false;
        params.feasibility_polishing = false;
        params.host_double_polishing = enable_host_polishing;
        params.host_double_polishing_iteration_limit = 200;
        params.host_double_polishing_time_sec_limit = 2.0;
        params.termination_evaluation_frequency = 10;
        params.termination_criteria.eps_optimal_relative = 1e-6;
        params.termination_criteria.eps_feasible_relative = 1e-6;
        // Preserve the supplied optimal x and leave the deliberately bad y to
        // the host-double certificate reconstruction phase.
        params.termination_criteria.iteration_limit = 0;
        params.termination_criteria.time_sec_limit = 10.0;
        MlxPdlpSolver solver(1, 2, row_ptr, col_ind, vals, var_lb, var_ub,
                             con_lb, con_ub, obj, 0.0, &params, primal_start,
                             dual_start, mx::Device::cpu);
        return solver.solve();
    };

    baseline = solve_case(false);
    polished = solve_case(true);

    CHECK(baseline->relative_objective_gap > 0.1,
          "fixture should begin with a materially bad dual objective");
    CHECK(polished->host_double_polishing_iteration > 0,
          "dual reconstruction should execute");
    CHECK_CLOSE(polished->primal_solution[0], primal_start[0], 1e-9,
                "dual reconstruction must preserve the good primal point");
    CHECK(polished->relative_primal_residual < 1e-9,
          "fixed primal point should remain feasible");
    CHECK(polished->relative_dual_residual < 1e-6,
          "reconstructed certificate should be dual feasible");
    CHECK(polished->relative_objective_gap < 1e-6,
          "reconstructed certificate should close the objective gap");

    mlxpdlp_result_free(baseline);
    mlxpdlp_result_free(polished);
    PASS();
    return;

test_cleanup:
    mlxpdlp_result_free(baseline);
    mlxpdlp_result_free(polished);
}

static void test_host_double_moves_feasible_suboptimal_primal() {
    TEST("host-double continuation moves a feasible suboptimal primal point");

    // minimize x0 subject to x0 + x1 = 1 and x >= 0. The supplied primal
    // point is exactly feasible but suboptimal. Improving only its dual
    // certificate cannot close the gap: the correction must retain budget for
    // a joint primal-dual continuation that moves x toward [0, 1].
    int row_ptr[] = {0, 2};
    int col_ind[] = {0, 1};
    double vals[] = {1.0, 1.0};
    double obj[] = {1.0, 0.0};
    double con_lb[] = {1.0};
    double con_ub[] = {1.0};
    double var_lb[] = {0.0, 0.0};
    double var_ub[] = {INFINITY, INFINITY};
    double primal_start[] = {0.5, 0.5};
    double dual_start[] = {-10.0};
    mlxpdlp_result_t *result = nullptr;

    pdhg_parameters_t params;
    mlxpdlp_set_default_parameters(&params);
    params.verbose = false;
    params.presolve = false;
    params.feasibility_polishing = false;
    params.host_double_polishing = true;
    params.host_double_early_handoff = false;
    params.host_double_polishing_iteration_limit = 5000;
    params.host_double_polishing_time_sec_limit = 2.0;
    params.termination_evaluation_frequency = 10;
    params.termination_criteria.eps_optimal_relative = 1e-6;
    params.termination_criteria.eps_feasible_relative = 1e-6;
    // Preserve the deliberately feasible, suboptimal checkpoint for the host
    // correction so this test isolates its phase transition.
    params.termination_criteria.iteration_limit = 0;
    params.termination_criteria.time_sec_limit = 10.0;

    MlxPdlpSolver solver(2, 1, row_ptr, col_ind, vals, var_lb, var_ub,
                         con_lb, con_ub, obj, 0.0, &params, primal_start,
                         dual_start, mx::Device::cpu);
    result = solver.solve();

    CHECK(result->host_double_polishing_iteration > 0,
          "feasible suboptimal checkpoint should execute host correction");
    CHECK(result->host_double_polishing_iteration <= 5000,
          "host correction must respect its configured iteration cap");
    CHECK(result->relative_primal_residual < 1e-6,
          "joint continuation should preserve primal feasibility");
    CHECK(result->relative_dual_residual < 1e-6,
          "joint continuation should recover dual feasibility");
    CHECK(result->relative_objective_gap < 1e-6,
          "joint continuation should close the objective gap");
    CHECK_CLOSE(result->primal_solution[0], 0.0, 1e-5,
                "joint continuation should move x0 to its optimal bound");
    CHECK_CLOSE(result->primal_solution[1], 1.0, 1e-5,
                "joint continuation should preserve the equality at optimum");

    mlxpdlp_result_free(result);
    PASS();
    return;

test_cleanup:
    mlxpdlp_result_free(result);
}

// ---------------------------------------------------------------------------
// Test 10: Warm starts are accepted in original coordinates and transformed
// through preconditioning.
// ---------------------------------------------------------------------------

static void test_warm_start() {
    TEST("primal and dual warm starts");

    int m = 3, n = 2;
    int row_ptr[] = {0, 2, 4, 6};
    int col_ind[] = {0, 1, 0, 1, 0, 1};
    double vals[] = {1.0, 2.0, 0.0, 1.0, 3.0, 2.0};
    double obj[] = {1.0, 1.0};
    double con_lb[] = {5.0, -INFINITY, -INFINITY};
    double con_ub[] = {5.0, 2.0, 8.0};
    double var_lb[] = {0.0, 0.0};
    double var_ub[] = {INFINITY, INFINITY};
    double primal_start[] = {1.0, 2.0};
    double dual_start[] = {1.0, -1.0, 0.0};

    pdhg_parameters_t params;
    mlxpdlp_set_default_parameters(&params);
    params.verbose = false;
    params.presolve = false;
    params.termination_evaluation_frequency = 50;
    params.termination_criteria.eps_optimal_relative = 1e-6;
    params.termination_criteria.eps_feasible_relative = 1e-6;
    set_safe_limits(&params);

    MlxPdlpSolver cold_solver(n, m, row_ptr, col_ind, vals, var_lb, var_ub, con_lb, con_ub, obj,
                              0.0, &params);
    mlxpdlp_result_t *cold = cold_solver.solve();

    MlxPdlpSolver warm_solver(n, m, row_ptr, col_ind, vals, var_lb, var_ub, con_lb, con_ub, obj,
                              0.0, &params, primal_start, dual_start);
    mlxpdlp_result_t *warm = warm_solver.solve();

    CHECK(warm->termination_reason == TERMINATION_REASON_OPTIMAL, "warm solve should be OPTIMAL");
    CHECK_CLOSE(warm->primal_solution[0], 1.0, 5e-3, "warm x[0]");
    CHECK_CLOSE(warm->primal_solution[1], 2.0, 5e-3, "warm x[1]");
    CHECK_CLOSE(warm->primal_objective_value, 3.0, 5e-3, "warm objective");
    CHECK(warm->total_count < cold->total_count,
          "optimal warm start should take fewer iterations than a cold start");

    mlxpdlp_result_free(cold);
    mlxpdlp_result_free(warm);
    PASS();
    return;

test_cleanup:
    mlxpdlp_result_free(cold);
    mlxpdlp_result_free(warm);
}

static void test_complete_warm_certificate() {
    TEST("complete warm start preserves reduced-cost certificate");

    int row_ptr[] = {0, 1};
    int col_ind[] = {0};
    double vals[] = {1.0};
    double obj[] = {3.0};
    double con_lb[] = {1.0};
    double con_ub[] = {1.0};
    double var_lb[] = {0.0};
    double var_ub[] = {2.0};
    mlxpdlp_result_t *cold = nullptr;
    mlxpdlp_result_t *resumed = nullptr;

    pdhg_parameters_t params;
    mlxpdlp_set_default_parameters(&params);
    params.verbose = false;
    params.presolve = false;
    params.termination_evaluation_frequency = 20;
    params.termination_criteria.eps_optimal_relative = 1e-7;
    params.termination_criteria.eps_feasible_relative = 1e-7;
    set_safe_limits(&params);

    {
        MlxPdlpSolver solver(1, 1, row_ptr, col_ind, vals, var_lb, var_ub,
                             con_lb, con_ub, obj, 0.0, &params);
        cold = solver.solve();
    }
    CHECK(cold->termination_reason == TERMINATION_REASON_OPTIMAL,
          "fixture solve should be optimal");

    params.feasibility_polishing = false;
    params.host_double_polishing = false;
    params.termination_criteria.iteration_limit = 0;
    {
        MlxPdlpSolver solver(1, 1, row_ptr, col_ind, vals, var_lb, var_ub,
                             con_lb, con_ub, obj, 0.0, &params,
                             cold->primal_solution, cold->dual_solution,
                             cold->reduced_cost, mx::Device::cpu);
        resumed = solver.solve();
    }

    CHECK(resumed->total_count == 0,
          "complete certificate resume should not require a PDHG block");
    CHECK_CLOSE(resumed->reduced_cost[0], cold->reduced_cost[0], 1e-10,
                "reduced cost survives scaled warm-start import");
    CHECK_CLOSE(resumed->dual_objective_value, cold->dual_objective_value, 1e-9,
                "dual objective survives complete checkpoint resume");
    CHECK(resumed->relative_primal_residual < 1e-7,
          "resumed certificate remains primal feasible");
    CHECK(resumed->relative_dual_residual < 1e-7,
          "resumed certificate remains dual feasible");
    CHECK(resumed->relative_objective_gap < 1e-7,
          "resumed certificate retains its objective gap");

    mlxpdlp_result_free(cold);
    mlxpdlp_result_free(resumed);
    PASS();
    return;

test_cleanup:
    mlxpdlp_result_free(cold);
    mlxpdlp_result_free(resumed);
}

static void test_warm_start_validation() {
    TEST("warm starts reject non-finite values");

    int row_ptr[] = {0, 1};
    int col_ind[] = {0};
    double vals[] = {1.0};
    double obj[] = {1.0};
    double con_lb[] = {0.0};
    double con_ub[] = {1.0};
    double var_lb[] = {0.0};
    double var_ub[] = {1.0};
    double invalid_primal[] = {NAN};

    pdhg_parameters_t params;
    mlxpdlp_set_default_parameters(&params);
    params.presolve = false;

    bool rejected = false;
    try {
        MlxPdlpSolver solver(1, 1, row_ptr, col_ind, vals, var_lb, var_ub, con_lb, con_ub, obj, 0.0,
                             &params, invalid_primal, nullptr);
    } catch (const std::invalid_argument &) {
        rejected = true;
    }

    CHECK(rejected, "non-finite warm start should throw std::invalid_argument");
    PASS();
test_cleanup:;
}

static void test_cpu_scalar_arithmetic_preserves_fp64() {
    TEST("CPU scalar-array arithmetic preserves FP64 coefficients");

    int row_ptr[] = {0};
    constexpr double objective_coefficient = 0.26741329404626046351;
    double obj[] = {objective_coefficient};
    double var_lb[] = {-10.0};
    double var_ub[] = {10.0};

    pdhg_parameters_t params;
    mlxpdlp_set_default_parameters(&params);
    params.verbose = false;
    params.presolve = false;
    params.geometric_mean_iterations = 0;
    params.curtis_reid_iterations = 0;
    params.l_inf_ruiz_iterations = 0;
    params.has_pock_chambolle_alpha = false;
    params.bound_objective_rescaling = false;
    params.feasibility_polishing = false;
    params.host_double_polishing = false;
    params.termination_evaluation_frequency = 2;
    params.termination_criteria.eps_optimal_relative = 0.0;
    params.termination_criteria.eps_feasible_relative = 0.0;
    params.termination_criteria.iteration_limit = 2;
    params.termination_criteria.time_sec_limit = 2.0;

    MlxPdlpSolver solver(1, 0, row_ptr, nullptr, nullptr, var_lb, var_ub,
                         nullptr, nullptr, obj, 0.0, &params, mx::Device::cpu);
    mlxpdlp_result_t *result = solver.solve();
    const double expected =
        -2.0 * objective_coefficient / (1.0 + std::fabs(objective_coefficient));

    CHECK(result->total_count == 2, "fixture should execute exactly one two-step block");
    CHECK_CLOSE(result->primal_solution[0], expected, 5e-14,
                "CPU iterate uses the full double-precision step coefficient");

    mlxpdlp_result_free(result);
    PASS();
    return;

test_cleanup:
    mlxpdlp_result_free(result);
}

static void test_original_certificate_checks_variable_bounds() {
    TEST("original certificate rejects variable-bound violations");

    int row_ptr[] = {0, 0};
    double var_lb[] = {0.0};
    double var_ub[] = {1.0};
    double con_lb[] = {-INFINITY};
    double con_ub[] = {INFINITY};
    double obj[] = {0.0};
    double primal_start[] = {2.0};
    double dual_start[] = {0.0};

    pdhg_parameters_t params;
    mlxpdlp_set_default_parameters(&params);
    params.verbose = false;
    params.presolve = false;
    params.geometric_mean_iterations = 0;
    params.curtis_reid_iterations = 0;
    params.l_inf_ruiz_iterations = 0;
    params.has_pock_chambolle_alpha = false;
    params.bound_objective_rescaling = false;
    params.feasibility_polishing = false;
    params.host_double_polishing = false;
    params.termination_evaluation_frequency = 2;
    params.termination_criteria.eps_optimal_relative = 1e-8;
    params.termination_criteria.eps_feasible_relative = 1e-8;
    params.termination_criteria.iteration_limit = 0;
    params.termination_criteria.time_sec_limit = 2.0;

    MlxPdlpSolver solver(1, 1, row_ptr, nullptr, nullptr, var_lb, var_ub,
                         con_lb, con_ub, obj, 0.0, &params,
                         primal_start, dual_start, mx::Device::cpu);
    mlxpdlp_result_t *result = solver.solve();

    CHECK(result->termination_reason == TERMINATION_REASON_ITERATION_LIMIT,
          "out-of-bounds warm start must not be promoted to OPTIMAL");
    CHECK_CLOSE(result->primal_solution[0], 2.0, 0.0,
                "disabled host polishing preserves the supplied warm start");

    mlxpdlp_result_free(result);
    PASS();
    return;

test_cleanup:
    mlxpdlp_result_free(result);
}

static void test_original_certificate_demotes_all_failed_metrics() {
    TEST("original certificate demotes every failed audited metric");

    enum class ExpectedFailure { none, primal, variable_bound, dual, gap };
    struct AuditCase {
        const char *name;
        ExpectedFailure failure;
        int nonzeros;
        double matrix_value;
        double objective;
        double variable_lower;
        double variable_upper;
        double constraint_lower;
        double constraint_upper;
        double primal_start;
        double dual_start;
        double reduced_cost_start;
    };
    const AuditCase cases[] = {
        {"valid", ExpectedFailure::none, 0, 0.0, 0.0, 0.0, 1.0,
         -INFINITY, INFINITY, 0.0, 0.0, 0.0},
        {"primal", ExpectedFailure::primal, 1, 1.0, 0.0, 0.0, INFINITY,
         1.0, 1.0, 0.0, 0.0, 0.0},
        {"variable-bound", ExpectedFailure::variable_bound, 0, 0.0, 0.0,
         0.0, 1.0, -INFINITY, INFINITY, 2.0, 0.0, 0.0},
        {"dual", ExpectedFailure::dual, 0, 0.0, 1.0, -INFINITY, INFINITY,
         -INFINITY, INFINITY, 0.0, 0.0, 0.0},
        {"gap", ExpectedFailure::gap, 0, 0.0, 1.0, 0.0, 1.0,
         -INFINITY, INFINITY, 1.0, 0.0, 1.0},
    };

    constexpr double tolerance = 1e-8;
    bool all_valid = true;
    mlxpdlp_result_t *result = nullptr;
    for (const AuditCase &audit_case : cases) {
        int row_ptr[] = {0, audit_case.nonzeros};
        int col_ind[] = {0};
        double values[] = {audit_case.matrix_value};
        double objective[] = {audit_case.objective};
        double var_lb[] = {audit_case.variable_lower};
        double var_ub[] = {audit_case.variable_upper};
        double con_lb[] = {audit_case.constraint_lower};
        double con_ub[] = {audit_case.constraint_upper};
        double primal_start[] = {audit_case.primal_start};
        double dual_start[] = {audit_case.dual_start};
        double reduced_cost_start[] = {audit_case.reduced_cost_start};

        pdhg_parameters_t params;
        mlxpdlp_set_default_parameters(&params);
        params.verbose = false;
        params.presolve = false;
        params.geometric_mean_iterations = 0;
        params.curtis_reid_iterations = 0;
        params.l_inf_ruiz_iterations = 0;
        params.has_pock_chambolle_alpha = false;
        params.bound_objective_rescaling = false;
        params.feasibility_polishing = false;
        params.host_double_polishing = false;
        params.host_double_early_handoff = false;
        params.termination_evaluation_frequency = 2;
        params.termination_criteria.eps_optimal_relative = tolerance;
        params.termination_criteria.eps_feasible_relative = tolerance;
        params.termination_criteria.iteration_limit = 0;
        params.termination_criteria.time_sec_limit = 2.0;

        MlxPdlpSolver solver(
            1, 1, row_ptr,
            audit_case.nonzeros == 0 ? nullptr : col_ind,
            audit_case.nonzeros == 0 ? nullptr : values,
            var_lb, var_ub, con_lb, con_ub, objective, 0.0, &params,
            primal_start, dual_start, reduced_cost_start, mx::Device::cpu);
        // Simulate an internal stopping decision immediately before extraction.
        // With zero iterations, the original-model audit is the only component
        // that may revise this stamp.
        const_cast<MlxPdlpState &>(solver.state()).termination_reason =
            TERMINATION_REASON_OPTIMAL;
        result = solver.solve();

        const bool primal_failed =
            !std::isfinite(result->relative_primal_residual) ||
            result->relative_primal_residual >= tolerance;
        const double primal = result->primal_solution[0];
        const bool variable_bound_failed =
            !std::isfinite(primal) ||
            (std::isfinite(audit_case.variable_lower) &&
             primal < audit_case.variable_lower) ||
            (std::isfinite(audit_case.variable_upper) &&
             primal > audit_case.variable_upper);
        const bool dual_failed =
            !std::isfinite(result->relative_dual_residual) ||
            result->relative_dual_residual >= tolerance;
        const bool gap_failed =
            !std::isfinite(result->relative_objective_gap) ||
            std::fabs(result->relative_objective_gap) >= tolerance;

        const bool expected_primal = audit_case.failure == ExpectedFailure::primal;
        const bool expected_bound =
            audit_case.failure == ExpectedFailure::variable_bound;
        const bool expected_dual = audit_case.failure == ExpectedFailure::dual;
        const bool expected_gap = audit_case.failure == ExpectedFailure::gap;
        const termination_reason_t expected_status =
            audit_case.failure == ExpectedFailure::none
                ? TERMINATION_REASON_OPTIMAL
                : TERMINATION_REASON_UNSPECIFIED;
        const bool valid = primal_failed == expected_primal &&
                           variable_bound_failed == expected_bound &&
                           dual_failed == expected_dual && gap_failed == expected_gap &&
                           result->termination_reason == expected_status;
        if (!valid) {
            std::printf("[%s status=%d failures=(%d,%d,%d,%d)] ", audit_case.name,
                        static_cast<int>(result->termination_reason), primal_failed,
                        variable_bound_failed, dual_failed, gap_failed);
            all_valid = false;
        }
        mlxpdlp_result_free(result);
        result = nullptr;
    }

    CHECK(all_valid, "each failed original-model audit must demote OPTIMAL");
    PASS();
    return;

test_cleanup:
    mlxpdlp_result_free(result);
}

static void test_host_double_polish_repairs_variable_bounds() {
    TEST("host-double polish repairs a bound-only violation");

    int row_ptr[] = {0, 0};
    double var_lb[] = {0.0};
    double var_ub[] = {1.0};
    double con_lb[] = {-INFINITY};
    double con_ub[] = {INFINITY};
    double obj[] = {0.0};
    double primal_start[] = {2.0};
    double dual_start[] = {0.0};

    pdhg_parameters_t params;
    mlxpdlp_set_default_parameters(&params);
    params.verbose = false;
    params.presolve = false;
    params.geometric_mean_iterations = 0;
    params.curtis_reid_iterations = 0;
    params.l_inf_ruiz_iterations = 0;
    params.has_pock_chambolle_alpha = false;
    params.bound_objective_rescaling = false;
    params.feasibility_polishing = false;
    params.host_double_polishing = true;
    params.host_double_polishing_iteration_limit = 4;
    params.host_double_polishing_time_sec_limit = 2.0;
    params.termination_evaluation_frequency = 2;
    params.termination_criteria.eps_optimal_relative = 1e-8;
    params.termination_criteria.eps_feasible_relative = 1e-8;
    params.termination_criteria.iteration_limit = 0;
    params.termination_criteria.time_sec_limit = 2.0;

    MlxPdlpSolver solver(1, 1, row_ptr, nullptr, nullptr, var_lb, var_ub,
                         con_lb, con_ub, obj, 0.0, &params,
                         primal_start, dual_start, mx::Device::cpu);
    mlxpdlp_result_t *result = solver.solve();

    CHECK(result->primal_solution[0] >= var_lb[0] &&
              result->primal_solution[0] <= var_ub[0],
          "host-double polish should clamp the primal point to original bounds");
    CHECK(result->termination_reason == TERMINATION_REASON_OPTIMAL,
          "repaired bound-feasible certificate should be OPTIMAL");

    mlxpdlp_result_free(result);
    PASS();
    return;

test_cleanup:
    mlxpdlp_result_free(result);
}

static void test_pdhg_infeasibility_certificates() {
    TEST("PDHG Farkas certificates terminate infeasible and unbounded LPs");

    // Provably primal-infeasible (x = 3 and x = 2 simultaneously) and
    // provably unbounded (min -x subject to x >= 0) fixtures with presolve
    // disabled, so PDHG itself must certify the status through the Farkas
    // separation ray tests. The FP64 CPU path certifies both.
    mlxpdlp_result_t *result = nullptr;

    {
        int row_ptr[] = {0, 1, 2};
        int col_ind[] = {0, 0};
        double vals[] = {1.0, 1.0};
        double obj[] = {0.0};
        double con_lb[] = {3.0, 2.0};
        double con_ub[] = {3.0, 2.0};
        double var_lb[] = {-INFINITY};
        double var_ub[] = {INFINITY};

        pdhg_parameters_t params;
        mlxpdlp_set_default_parameters(&params);
        params.verbose = false;
        set_safe_limits(&params);

        MlxPdlpSolver solver(1, 2, row_ptr, col_ind, vals, var_lb, var_ub,
                             con_lb, con_ub, obj, 0.0, &params);
        result = solver.solve();
        CHECK(result->termination_reason == TERMINATION_REASON_PRIMAL_INFEASIBLE,
              "PDHG should certify primal infeasibility");
        CHECK(result->total_count < 100000,
              "the certificate should terminate far before the iteration limit");
        mlxpdlp_result_free(result);
        result = nullptr;
    }

    {
        int row_ptr[] = {0, 1};
        int col_ind[] = {0};
        double vals[] = {1.0};
        double obj[] = {-1.0};
        double con_lb[] = {0.0};
        double con_ub[] = {INFINITY};
        double var_lb[] = {-INFINITY};
        double var_ub[] = {INFINITY};

        pdhg_parameters_t params;
        mlxpdlp_set_default_parameters(&params);
        params.verbose = false;
        set_safe_limits(&params);

        MlxPdlpSolver solver(1, 1, row_ptr, col_ind, vals, var_lb, var_ub,
                             con_lb, con_ub, obj, 0.0, &params);
        result = solver.solve();
        CHECK(result->termination_reason == TERMINATION_REASON_DUAL_INFEASIBLE,
              "PDHG should certify dual infeasibility (unbounded primal)");
        CHECK(result->total_count < 100000,
              "the certificate should terminate far before the iteration limit");
        mlxpdlp_result_free(result);
        result = nullptr;
    }

    PASS();
    return;

test_cleanup:
    mlxpdlp_result_free(result);
}

#ifdef MLXPDLP_TEST_HAS_PRESOLVE
static void test_presolve_solves_problem() {
    TEST("presolve solves and postsolves a small LP");

    int row_ptr[] = {0, 2, 4, 6};
    int col_ind[] = {0, 1, 0, 1, 0, 1};
    double vals[] = {1.0, 2.0, 0.0, 1.0, 3.0, 2.0};
    double obj[] = {1.0, 1.0};
    double con_lb[] = {5.0, -INFINITY, -INFINITY};
    double con_ub[] = {5.0, 2.0, 8.0};
    double var_lb[] = {0.0, 0.0};
    double var_ub[] = {INFINITY, INFINITY};

    pdhg_parameters_t params;
    mlxpdlp_set_default_parameters(&params);
    params.verbose = false;
    params.presolve = true;

    MlxPdlpSolver solver(2, 3, row_ptr, col_ind, vals, var_lb, var_ub, con_lb, con_ub, obj, 4.0,
                         &params);
    mlxpdlp_result_t *result = solver.solve();

    CHECK(result->termination_reason == TERMINATION_REASON_OPTIMAL,
          "presolve should report OPTIMAL");
    CHECK(result->total_count == 0, "fully presolved LP should not run PDHG iterations");
    CHECK(result->num_variables == 2, "postsolve should restore original variable count");
    CHECK(result->num_constraints == 3, "postsolve should restore original constraint count");
    CHECK(result->num_reduced_variables == 0, "PSLP should eliminate all variables");
    CHECK_CLOSE(result->primal_solution[0], 1.0, 5e-3, "postsolved x[0]");
    CHECK_CLOSE(result->primal_solution[1], 2.0, 5e-3, "postsolved x[1]");
    CHECK_CLOSE(result->primal_objective_value, 7.0, 5e-3, "postsolved objective");
    CHECK(result->presolve_time >= 0.0, "presolve time should be populated");

    mlxpdlp_result_free(result);
    PASS();
    return;

test_cleanup:
    mlxpdlp_result_free(result);
}

static void test_presolve_then_pdhg_postsolve() {
    TEST("presolve postsolves a reduced PDHG result");

    int row_ptr[] = {0, 3, 6};
    int col_ind[] = {0, 1, 2, 0, 1, 2};
    double vals[] = {1.0, 1.0, 1.0, 2.0, 3.0, 1.0};
    double obj[] = {1.0, 2.0, 3.0};
    double con_lb[] = {6.0, -INFINITY};
    double con_ub[] = {6.0, 10.0};
    double var_lb[] = {0.0, 0.0, 0.0};
    double var_ub[] = {INFINITY, INFINITY, INFINITY};

    pdhg_parameters_t params;
    mlxpdlp_set_default_parameters(&params);
    params.verbose = false;
    params.presolve = true;
    params.termination_evaluation_frequency = 100;
    params.termination_criteria.eps_optimal_relative = 1e-6;
    params.termination_criteria.eps_feasible_relative = 1e-6;
    params.termination_criteria.eps_feas_polish_relative = 1e-12;
    params.termination_criteria.iteration_limit = 100000;
    params.termination_criteria.time_sec_limit = 60.0;
    params.host_double_polishing = true;
    params.host_double_polishing_iteration_limit = 30;
    params.host_double_polishing_time_sec_limit = 2.0;

    MlxPdlpSolver solver(3, 2, row_ptr, col_ind, vals, var_lb, var_ub, con_lb, con_ub, obj, 0.0,
                         &params);
    mlxpdlp_result_t *result = solver.solve();

    CHECK(result->termination_reason == TERMINATION_REASON_OPTIMAL,
          "presolved PDHG solve should be OPTIMAL");
    CHECK(result->total_count > 0, "this fixture should retain a PDHG phase");
    CHECK(result->host_double_polishing_iteration > 0,
          "presolved result should receive reduced-model host correction");
    CHECK(result->host_double_polishing_iteration <= 30,
          "reduced-model host correction should share the configured cap");
    CHECK(result->num_variables == 3, "postsolve should restore original dimensions");
    CHECK_CLOSE(result->primal_solution[0], 4.0, 5e-3, "postsolved x[0]");
    CHECK_CLOSE(result->primal_solution[1], 0.0, 5e-3, "postsolved x[1]");
    CHECK_CLOSE(result->primal_solution[2], 2.0, 5e-3, "postsolved x[2]");
    CHECK_CLOSE(result->primal_objective_value, 10.0, 5e-3, "postsolved objective");

    mlxpdlp_result_free(result);
    PASS();
    return;

test_cleanup:
    mlxpdlp_result_free(result);
}

static void test_presolve_detects_infeasible() {
    TEST("presolve detects primal infeasibility");

    int row_ptr[] = {0, 2, 4, 6};
    int col_ind[] = {0, 1, 0, 1, 0, 1};
    double vals[] = {1.0, 2.0, 0.0, 1.0, 3.0, 2.0};
    double obj[] = {1.0, 1.0};
    double con_lb[] = {10.0, -INFINITY, -INFINITY};
    double con_ub[] = {10.0, 2.0, 8.0};
    double var_lb[] = {0.0, 0.0};
    double var_ub[] = {INFINITY, INFINITY};

    pdhg_parameters_t params;
    mlxpdlp_set_default_parameters(&params);
    params.verbose = false;
    params.presolve = true;

    MlxPdlpSolver solver(2, 3, row_ptr, col_ind, vals, var_lb, var_ub, con_lb, con_ub, obj, 0.0,
                         &params);
    mlxpdlp_result_t *result = solver.solve();

    CHECK(result->termination_reason == TERMINATION_REASON_PRIMAL_INFEASIBLE,
          "PSLP should detect primal infeasibility");
    CHECK(result->total_count == 0, "infeasible presolve should not run PDHG");
    CHECK(std::isinf(result->relative_primal_residual),
          "infeasible result should expose infinite residuals");

    mlxpdlp_result_free(result);
    PASS();
    return;

test_cleanup:
    mlxpdlp_result_free(result);
}

static void test_presolve_detects_unbounded_or_infeasible() {
    TEST("presolve detects unbounded-or-infeasible problem");

    int row_ptr[] = {0};
    double obj[] = {-1.0};
    double var_lb[] = {0.0};
    double var_ub[] = {INFINITY};

    pdhg_parameters_t params;
    mlxpdlp_set_default_parameters(&params);
    params.verbose = false;
    params.presolve = true;

    MlxPdlpSolver solver(1, 0, row_ptr, nullptr, nullptr, var_lb, var_ub, nullptr, nullptr, obj,
                         0.0, &params);
    mlxpdlp_result_t *result = solver.solve();

    CHECK(result->termination_reason == TERMINATION_REASON_INFEASIBLE_OR_UNBOUNDED,
          "PSLP should report its combined unbounded-or-infeasible status");
    CHECK(result->total_count == 0, "terminal presolve should not run PDHG");

    mlxpdlp_result_free(result);
    PASS();
    return;

test_cleanup:
    mlxpdlp_result_free(result);
}

static void test_warm_start_with_presolve_rejected() {
    TEST("presolve rejects warm starts explicitly");

    int row_ptr[] = {0, 1};
    int col_ind[] = {0};
    double vals[] = {1.0};
    double obj[] = {1.0};
    double con_lb[] = {0.0};
    double con_ub[] = {1.0};
    double var_lb[] = {0.0};
    double var_ub[] = {1.0};
    double primal_start[] = {0.5};

    pdhg_parameters_t params;
    mlxpdlp_set_default_parameters(&params);
    params.presolve = true;

    bool rejected = false;
    try {
        MlxPdlpSolver solver(1, 1, row_ptr, col_ind, vals, var_lb, var_ub, con_lb, con_ub, obj, 0.0,
                             &params, primal_start, nullptr);
    } catch (const std::invalid_argument &) {
        rejected = true;
    }

    CHECK(rejected, "warm start plus presolve should throw std::invalid_argument");
    PASS();
test_cleanup:;
}
#else
static void test_unavailable_presolve_rejected() {
    TEST("presolve request is rejected when PSLP is not built");

    int row_ptr[] = {0, 1};
    int col_ind[] = {0};
    double vals[] = {1.0};

    pdhg_parameters_t params;
    mlxpdlp_set_default_parameters(&params);
    params.presolve = true;

    bool rejected = false;
    try {
        MlxPdlpSolver solver(1, 1, row_ptr, col_ind, vals, nullptr, nullptr, nullptr, nullptr,
                             nullptr, 0.0, &params);
    } catch (const std::runtime_error &) {
        rejected = true;
    }

    CHECK(rejected, "unavailable presolve should throw std::runtime_error");
    PASS();
test_cleanup:;
}
#endif

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

int main() {
    printf("mlxPDLP — Solver Test Suite\n");
    printf("========================================\n\n");

    test_default_parameters();
    test_geometric_mean_scaling();
    test_simple_lp();
    test_medium_lp();
    test_simple_lp_tight();
    test_free_variables();
    test_pure_equality();
    test_result_struct();
    test_iteration_limit_returns_best_iterate();
    test_conditional_termination_evaluation();
    test_repeat_solve();
    test_solver_rejects_second_solve();
    test_singular_value_nullspace_start();
    test_power_method_stops_on_relative_spectral_change();
    test_duplicate_csr_entries();
#if defined(__APPLE__)
    test_large_cpu_problem_uses_sparse_backend();
#endif
    test_reported_residuals_use_original_scaling();
    test_zero_norm_residual_normalization();
    test_feasibility_polishing_is_safeguarded();
    test_polish_shares_solve_time_budget();
    test_host_double_polishing_is_safeguarded();
    test_host_double_repairs_primal_without_changing_objective();
    test_fp64_cpu_bypasses_host_double_handoff();
    test_host_double_reconstructs_dual_without_moving_primal();
    test_host_double_moves_feasible_suboptimal_primal();
    test_warm_start();
    test_complete_warm_certificate();
    test_warm_start_validation();
    test_cpu_scalar_arithmetic_preserves_fp64();
    test_original_certificate_checks_variable_bounds();
    test_original_certificate_demotes_all_failed_metrics();
    test_host_double_polish_repairs_variable_bounds();
    test_pdhg_infeasibility_certificates();
#ifdef MLXPDLP_TEST_HAS_PRESOLVE
    test_presolve_solves_problem();
    test_presolve_then_pdhg_postsolve();
    test_presolve_detects_infeasible();
    test_presolve_detects_unbounded_or_infeasible();
    test_warm_start_with_presolve_rejected();
#else
    test_unavailable_presolve_rejected();
#endif

    printf("\n========================================\n");
    printf("Results: %d/%d tests passed\n", tests_passed, tests_run);

    return (tests_passed == tests_run) ? 0 : 1;
}
