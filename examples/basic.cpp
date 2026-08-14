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
#include <exception>
#include <memory>
#include <stdexcept>

using namespace mlxpdlp;

// ---------------------------------------------------------------------------
// Trivial Metal correctness example. This problem is intentionally too small
// to benchmark a GPU; see metal_acceleration.cpp for a fixed-work comparison.
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
    std::printf("mlxPDLP C++ Metal example\n");
    std::printf("==========================\n\n");

    if (!mx::is_available(mx::Device::gpu)) {
        std::printf("SKIPPED: MLX reports no Metal GPU device.\n");
        std::printf("Build MLX with MLX_BUILD_METAL=ON on Apple Silicon.\n");
        return 77;
    }

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
    // Disable host presolve/correction so this tiny LP demonstrably executes
    // the PDHG iterations on the requested Metal device.
    params.verbose = false;
    params.presolve = false;
    params.feasibility_polishing = false;
    params.host_double_polishing = false;
    params.termination_evaluation_frequency = 50; // smaller blocks for small problem
    params.termination_criteria.eps_optimal_relative = 1e-4;
    params.termination_criteria.eps_feasible_relative = 1e-4;
    params.sv_max_iter = 1000;
    params.sv_tol = 1e-6;

    std::printf("Problem: %d variables, %d constraints, 6 nonzeros\n", n, m);
    std::printf("Expected: x=[1, 2], objective=3\n");
    std::printf("Requested device: mx::Device::gpu (Apple Metal)\n\n");

    try {
        MlxPdlpSolver solver(n, m, row_ptr, col_ind, vals, var_lb, var_ub, con_lb, con_ub, obj, 0.0,
                             &params, mx::Device::gpu);
        const auto &state = solver.state();
        if (state.stream.device != mx::Device::gpu || state.cpu_double_precision_active ||
            state.obj.dtype() != mx::float32) {
            throw std::runtime_error("solver did not retain the requested Metal FP32 device");
        }

        std::unique_ptr<mlxpdlp_result_t, decltype(&mlxpdlp_result_free)> result(
            solver.solve(), mlxpdlp_result_free);
        mx::synchronize(state.stream);

        std::printf("Executed backend: %s\n",
                    state.sparse_metal_active ? "CSR-Metal-FP32" : "dense-MLX-Metal-FP32");
        std::printf("PDHG iterations: %d\n", result->total_count);

        std::printf("\nSolution:\n");
        std::printf("  Primal x = [");
        for (int i = 0; i < n; ++i) {
            std::printf("%.8f%s", result->primal_solution[i], i < n - 1 ? ", " : "");
        }
        std::printf("]\n");

        std::printf("  Dual y   = [");
        for (int i = 0; i < m; ++i) {
            std::printf("%.8f%s", result->dual_solution[i], i < m - 1 ? ", " : "");
        }
        std::printf("]\n");

        std::printf("  Objective = %.8f\n", result->primal_objective_value);

        constexpr double tolerance = 5e-3;
        const bool valid = result->termination_reason == TERMINATION_REASON_OPTIMAL &&
                           std::fabs(result->primal_solution[0] - 1.0) <= tolerance &&
                           std::fabs(result->primal_solution[1] - 2.0) <= tolerance &&
                           std::fabs(result->primal_objective_value - 3.0) <= tolerance;
        std::printf("\nMetal solve validation: %s\n", valid ? "PASS" : "FAIL");
        std::printf("Note: this tiny LP proves Metal execution, not GPU acceleration.\n");
        return valid ? 0 : 1;
    } catch (const std::exception &error) {
        std::fprintf(stderr, "Metal example failed: %s\n", error.what());
        return 1;
    }
}
