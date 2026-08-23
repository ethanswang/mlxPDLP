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
Fixed-iteration convergence sweep for Netlib ADLITTLE. The problem is bundled
with the repository and has a published optimal objective, so this example is
reproducible without downloading benchmark data.
*/
#include "mlxPDLP/mps_loader.h"
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

#ifndef NETLIB_ADLITTLE_MPS
#error "NETLIB_ADLITTLE_MPS must identify the bundled ADLITTLE model"
#endif

namespace {

constexpr std::array<int, 7> kIterationSweep = {100, 500, 1000, 2000, 5000, 10000, 20000};
constexpr int kExpectedVariables = 97;
constexpr int kExpectedConstraints = 56;
constexpr double kPublishedObjective = 2.2549496316e5;

struct ProblemDeleter {
    void operator()(mlxpdlp_mps_problem_t *problem) const {
        mlxpdlp_mps_problem_free(problem);
    }
};

struct ConvergencePoint {
    int iterations;
    double objective;
    double objective_relative_error;
    double relative_primal_residual;
    double relative_dual_residual;
    double relative_duality_gap;
    const char *backend;
};

const char *backend_name(const MlxPdlpState &state) {
    if (state.sparse_metal_active)
        return "CSR-Metal-FP32";
    if (state.sparse_cpu_active)
        return "Accelerate-CSR-FP64";
    return state.cpu_double_precision_active ? "dense-MLX-FP64" : "dense-MLX-Metal-FP32";
}

ConvergencePoint solve_at(const mlxpdlp_mps_problem_t &problem,
                          const std::vector<double> &objective, int iteration_limit,
                          const mx::Device &device) {
    pdhg_parameters_t params;
    mlxpdlp_set_default_parameters(&params);
    params.verbose = false;
    params.presolve = false;
    params.feasibility_polishing = false;
    params.host_double_polishing = false;
    params.host_double_early_handoff = false;
    params.termination_evaluation_frequency = 100;
    params.termination_criteria.eps_optimal_relative = 0.0;
    params.termination_criteria.eps_feasible_relative = 0.0;
    params.termination_criteria.iteration_limit = iteration_limit;
    params.termination_criteria.time_sec_limit = 300.0;
    params.sv_max_iter = 100;
    params.sv_tol = 1e-4;

    MlxPdlpSolver solver(problem.num_variables, problem.num_constraints, problem.row_ptr,
                         problem.col_ind, problem.values, problem.variable_lb, problem.variable_ub,
                         problem.constraint_lb, problem.constraint_ub, objective.data(),
                         problem.objective_constant, &params, device);
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

    const double reported_objective =
        problem.maximize ? -result->primal_objective_value : result->primal_objective_value;
    return ConvergencePoint{
        result->total_count,
        reported_objective,
        std::fabs(reported_objective - kPublishedObjective) / std::fabs(kPublishedObjective),
        result->relative_primal_residual,
        result->relative_dual_residual,
        result->relative_objective_gap,
        backend_name(state),
    };
}

std::vector<ConvergencePoint> run_sweep(const mlxpdlp_mps_problem_t &problem,
                                        const std::vector<double> &objective,
                                        const mx::Device &device) {
    std::vector<ConvergencePoint> points;
    points.reserve(kIterationSweep.size());
    for (int iterations : kIterationSweep) {
        points.push_back(solve_at(problem, objective, iterations, device));
    }
    return points;
}

void print_sweep(const char *label, const std::vector<ConvergencePoint> &points) {
    std::printf("\n%s (%s)\n", label, points.front().backend);
    std::printf("%10s %16s %14s %14s %14s %14s\n", "iteration", "objective", "rel-obj-err",
                "rel-primal", "rel-dual", "rel-gap");
    for (const auto &point : points) {
        std::printf("%10d %16.8e %14.6e %14.6e %14.6e %14.6e\n", point.iterations, point.objective,
                    point.objective_relative_error, point.relative_primal_residual,
                    point.relative_dual_residual, point.relative_duality_gap);
    }
}

bool finite(const ConvergencePoint &point) {
    return std::isfinite(point.objective) && std::isfinite(point.objective_relative_error) &&
           std::isfinite(point.relative_primal_residual) &&
           std::isfinite(point.relative_dual_residual) && std::isfinite(point.relative_duality_gap);
}

bool converged(const std::vector<ConvergencePoint> &points) {
    if (!std::all_of(points.begin(), points.end(), finite))
        return false;
    const auto &last = points.back();
    constexpr double tolerance = 1e-5;
    return last.objective_relative_error <= tolerance &&
           last.relative_primal_residual <= tolerance && last.relative_dual_residual <= tolerance &&
           last.relative_duality_gap <= tolerance;
}

} // namespace

int main() {
    std::printf("mlxPDLP Netlib ADLITTLE convergence sweep\n");
    std::printf("=========================================\n");

    try {
        std::unique_ptr<mlxpdlp_mps_problem_t, ProblemDeleter> problem(
            mlxpdlp_mps_problem_load(NETLIB_ADLITTLE_MPS));
        if (!problem) {
            throw std::runtime_error("failed to load the bundled ADLITTLE MPS model");
        }
        if (problem->num_variables != kExpectedVariables ||
            problem->num_constraints != kExpectedConstraints || problem->maximize) {
            throw std::runtime_error("the bundled ADLITTLE model has unexpected metadata");
        }

        std::printf("Model: %s\n", NETLIB_ADLITTLE_MPS);
        std::printf("Parsed: %d variables, %d constraints, %d matrix nonzeros\n",
                    problem->num_variables, problem->num_constraints, problem->num_nonzeros);
        std::printf("Published optimum: %.10e\n", kPublishedObjective);
        std::printf("Fixed iterations; presolve and polishing disabled.\n");

        std::vector<double> objective(problem->objective,
                                      problem->objective + problem->num_variables);
        const auto cpu = run_sweep(*problem, objective, mx::Device::cpu);
        print_sweep("CPU FP64", cpu);
        if (!converged(cpu)) {
            std::fprintf(stderr, "CPU FP64 convergence validation failed\n");
            return 1;
        }

        if (!mx::is_available(mx::Device::gpu)) {
            std::printf("\nSKIPPED: MLX reports no Metal GPU device.\n");
            return 77;
        }

        const auto metal = run_sweep(*problem, objective, mx::Device::gpu);
        print_sweep("Metal FP32", metal);
        if (!converged(metal)) {
            std::fprintf(stderr, "Metal FP32 convergence validation failed\n");
            return 1;
        }

        std::printf("\nConvergence validation at 20,000 iterations: PASS\n");
        std::printf("ADLITTLE is an algorithm diagnostic, not a GPU performance benchmark.\n");
        return 0;
    } catch (const std::exception &error) {
        std::fprintf(stderr, "Netlib convergence example failed: %s\n", error.what());
        return 1;
    }
}
