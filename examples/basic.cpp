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

using namespace mlxpdlp;

// ---------------------------------------------------------------------------
// Test LP: same as test/test_basic.py
//
//   minimize  x0 + x1
//   s.t.     1*x0 + 2*x1 = 5    (row 0: l=5, u=5)
//            0*x0 + 1*x1 <= 2   (row 1: l=-inf, u=2)
//            3*x0 + 2*x1 <= 8   (row 2: l=-inf, u=8)
//            x0 >= 0, x1 >= 0   (no upper bound)
//
// Optimal: x* = [1.0, 2.0], y* = [1.0, -1.0, 0.0], obj = 3.0
// ---------------------------------------------------------------------------

int main() {
    printf("mlxPDLP — Demo\n");
    printf("===========================\n\n");

    // Build LP in CSR format
    int m = 3; // constraints
    int n = 2; // variables

    // A = [[1, 2], [0, 1], [3, 2]]
    int row_ptr[] = {0, 2, 4, 6};
    int col_ind[] = {0, 1, 0, 1, 0, 1};
    double vals[] = {1.0, 2.0, 0.0, 1.0, 3.0, 2.0};

    double obj[] = {1.0, 1.0};
    double con_lb[] = {5.0, -INFINITY, -INFINITY};
    double con_ub[] = {5.0, 2.0, 8.0};
    double var_lb[] = {0.0, 0.0};
    double var_ub[] = {INFINITY, INFINITY};

    // Set up parameters (verbose for demo)
    pdhg_parameters_t params;
    mlxpdlp_set_default_parameters(&params);
    params.verbose = true;
    params.termination_evaluation_frequency = 50; // smaller blocks for small problem
    params.termination_criteria.eps_optimal_relative = 1e-6;
    params.termination_criteria.eps_feasible_relative = 1e-6;
    params.sv_max_iter = 1000;
    params.sv_tol = 1e-6;

    printf("Problem: %d variables, %d constraints\n", n, m);
    printf("A = [[1,2],[0,1],[3,2]] (CSR, nnz=6)\n");
    printf("Expected: x=[1.0, 2.0], y=[1.0, -1.0, 0.0], obj=3.0\n\n");

    // Create solver
    MlxPdlpSolver solver(n, m, row_ptr, col_ind, vals, var_lb, var_ub, con_lb, con_ub, obj, 0.0,
                         &params);

    // Solve
    mlxpdlp_result_t *result = solver.solve();

    // Print solutions
    printf("\nSolution:\n");
    printf("  Primal x = [");
    for (int i = 0; i < n; ++i) {
        printf("%.8f%s", result->primal_solution[i], i < n - 1 ? ", " : "");
    }
    printf("]\n");

    printf("  Dual y   = [");
    for (int i = 0; i < m; ++i) {
        printf("%.8f%s", result->dual_solution[i], i < m - 1 ? ", " : "");
    }
    printf("]\n");

    printf("  Reduced cost = [");
    for (int i = 0; i < n; ++i) {
        printf("%.8f%s", result->reduced_cost[i], i < n - 1 ? ", " : "");
    }
    printf("]\n");

    // Validate against expected
    double expected_x[] = {1.0, 2.0};
    double expected_y[] = {1.0, -1.0, 0.0};
    double expected_obj = 3.0;
    double tol = 5e-3; // relaxed tolerance for dense float32 implementation

    bool x_ok = true, y_ok = true, obj_ok = true;
    for (int i = 0; i < n; ++i) {
        if (std::fabs(result->primal_solution[i] - expected_x[i]) > tol)
            x_ok = false;
    }
    for (int i = 0; i < m; ++i) {
        if (std::fabs(result->dual_solution[i] - expected_y[i]) > tol)
            y_ok = false;
    }
    if (std::fabs(result->primal_objective_value - expected_obj) > tol)
        obj_ok = false;

    printf("\nValidation (tol=%.1e):\n", tol);
    printf("  Primal solution: %s\n", x_ok ? "PASS" : "FAIL");
    printf("  Dual solution:   %s\n", y_ok ? "PASS" : "FAIL");
    printf("  Objective value: %s\n", obj_ok ? "PASS" : "FAIL");

    bool all_ok = x_ok && y_ok && obj_ok;
    printf("\n%s\n", all_ok ? "ALL CHECKS PASSED" : "SOME CHECKS FAILED");

    mlxpdlp_result_free(result);

    return all_ok ? 0 : 1;
}
