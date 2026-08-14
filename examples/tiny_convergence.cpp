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

/*
Fixed-iteration convergence sweep for a two-variable LP with the exact
solution x = [1, 2]. This is a correctness diagnostic, not a performance test.
*/
#include "mlxPDLP/solver.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <exception>
#include <memory>
#include <stdexcept>
#include <vector>

using namespace mlxpdlp;

namespace {

constexpr std::array<int, 7> kIterationSweep = {10, 50, 100, 200, 500, 1000, 5000};

struct ConvergencePoint {
    int iterations;
    double solution_error_inf;
    double primal_residual;
    double dual_residual;
    double duality_gap;
    double objective;
    const char *backend;
};

const char *backend_name(const MlxPdlpState &state) {
    if (state.sparse_metal_active)
        return "CSR-Metal-FP32";
    if (state.sparse_cpu_active)
        return "Accelerate-CSR-FP64";
    return state.cpu_double_precision_active ? "dense-MLX-FP64" : "dense-MLX-Metal-FP32";
}

ConvergencePoint solve_at(int iteration_limit, const mx::Device &device) {
    // minimize x0 + x1
    // subject to x0 + 2*x1 = 5, x1 <= 2, 3*x0 + 2*x1 <= 8, x >= 0.
    // The unique optimum is x* = [1, 2], objective = 3.
    int row_ptr[] = {0, 2, 4, 6};
    int col_ind[] = {0, 1, 0, 1, 0, 1};
    double values[] = {1.0, 2.0, 0.0, 1.0, 3.0, 2.0};
    double variable_lb[] = {0.0, 0.0};
    double variable_ub[] = {INFINITY, INFINITY};
    double constraint_lb[] = {5.0, -INFINITY, -INFINITY};
    double constraint_ub[] = {5.0, 2.0, 8.0};
    double objective[] = {1.0, 1.0};

    pdhg_parameters_t params;
    mlxpdlp_set_default_parameters(&params);
    params.verbose = false;
    params.presolve = false;
    params.feasibility_polishing = false;
    params.host_double_polishing = false;
    params.host_double_early_handoff = false;
    // Keep the restart/evaluation cadence identical at every sweep point so
    // each row samples one trajectory rather than changing the algorithm.
    params.termination_evaluation_frequency = 10;
    params.termination_criteria.eps_optimal_relative = 0.0;
    params.termination_criteria.eps_feasible_relative = 0.0;
    params.termination_criteria.iteration_limit = iteration_limit;
    params.termination_criteria.time_sec_limit = 60.0;
    params.sv_max_iter = 1000;
    params.sv_tol = 1e-6;

    MlxPdlpSolver solver(2, 3, row_ptr, col_ind, values, variable_lb, variable_ub, constraint_lb,
                         constraint_ub, objective, 0.0, &params, device);
    const auto &state = solver.state();
    if (state.stream.device != device) {
        throw std::runtime_error("solver did not retain the requested device");
    }
    const bool expects_fp64 = device.type == mx::Device::cpu;
    if (state.cpu_double_precision_active != expects_fp64 ||
        state.obj.dtype() != (expects_fp64 ? mx::float64 : mx::float32)) {
        throw std::runtime_error("solver did not use the expected device precision");
    }

    std::unique_ptr<mlxpdlp_result_t, decltype(&mlxpdlp_result_free)> result(solver.solve(),
                                                                             mlxpdlp_result_free);
    mx::synchronize(state.stream);
    if (result->total_count != iteration_limit) {
        throw std::runtime_error("solver did not execute the requested fixed iteration count");
    }

    const double solution_error = std::max(std::fabs(result->primal_solution[0] - 1.0),
                                           std::fabs(result->primal_solution[1] - 2.0));
    return ConvergencePoint{
        result->total_count,
        solution_error,
        result->absolute_primal_residual,
        result->absolute_dual_residual,
        std::fabs(result->objective_gap),
        result->primal_objective_value,
        backend_name(state),
    };
}

std::vector<ConvergencePoint> run_sweep(const mx::Device &device) {
    std::vector<ConvergencePoint> points;
    points.reserve(kIterationSweep.size());
    for (int iterations : kIterationSweep) {
        points.push_back(solve_at(iterations, device));
    }
    return points;
}

void print_sweep(const char *label, const std::vector<ConvergencePoint> &points) {
    std::printf("\n%s (%s)\n", label, points.front().backend);
    std::printf("%10s %14s %14s %14s %14s %16s\n", "iteration", "||x-x*||inf", "abs-primal",
                "abs-dual", "abs-gap", "objective");
    for (const auto &point : points) {
        std::printf("%10d %14.6e %14.6e %14.6e %14.6e %16.8f\n", point.iterations,
                    point.solution_error_inf, point.primal_residual, point.dual_residual,
                    point.duality_gap, point.objective);
    }
}

bool finite(const ConvergencePoint &point) {
    return std::isfinite(point.solution_error_inf) && std::isfinite(point.primal_residual) &&
           std::isfinite(point.dual_residual) && std::isfinite(point.duality_gap) &&
           std::isfinite(point.objective);
}

bool converged(const std::vector<ConvergencePoint> &points, double tolerance) {
    if (!std::all_of(points.begin(), points.end(), finite))
        return false;
    const auto &first = points.front();
    const auto &last = points.back();
    return last.solution_error_inf < first.solution_error_inf &&
           last.solution_error_inf <= tolerance && last.primal_residual <= tolerance &&
           last.dual_residual <= tolerance && last.duality_gap <= tolerance;
}

} // namespace

int main() {
    std::printf("mlxPDLP tiny exact-LP convergence sweep\n");
    std::printf("=======================================\n");
    std::printf("Fixed iterations; presolve and polishing disabled.\n");

    try {
        const auto cpu = run_sweep(mx::Device::cpu);
        print_sweep("CPU FP64", cpu);
        if (!converged(cpu, 1e-6)) {
            std::fprintf(stderr, "CPU FP64 convergence validation failed\n");
            return 1;
        }

        if (!mx::is_available(mx::Device::gpu)) {
            std::printf("\nSKIPPED: MLX reports no Metal GPU device.\n");
            return 77;
        }

        const auto metal = run_sweep(mx::Device::gpu);
        print_sweep("Metal FP32", metal);
        if (!converged(metal, 1e-4)) {
            std::fprintf(stderr, "Metal FP32 convergence validation failed\n");
            return 1;
        }

        std::printf("\nConvergence validation: PASS\n");
        std::printf("This diagnostic checks error decay; it does not benchmark either device.\n");
        return 0;
    } catch (const std::exception &error) {
        std::fprintf(stderr, "Tiny convergence example failed: %s\n", error.what());
        return 1;
    }
}
