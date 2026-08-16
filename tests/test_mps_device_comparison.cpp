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
Solve a real MPS benchmark with the same MLX PDLP implementation on CPU and
Metal. The test checks parser dimensions, solution correctness, device
placement, and CPU/GPU agreement; timings are diagnostic only.
*/
#include "mlxPDLP/mps_loader.h"
#include "mlxPDLP/solver.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <exception>
#include <stdexcept>
#include <vector>

using namespace mlxpdlp;

#ifndef NETLIB_ADLITTLE_MPS
#error "NETLIB_ADLITTLE_MPS must identify the bundled benchmark"
#endif

namespace {

constexpr int kSkip = 77;
constexpr int kExpectedVariables = 97;
constexpr int kExpectedConstraints = 56;
constexpr double kPublishedObjective = 2.2549496316e5;
constexpr double kObjectiveRelativeTolerance = 1e-4;
constexpr double kCertificateTolerance = 1e-5;

struct SolveSummary {
    double objective;
    double relative_primal_residual;
    double relative_dual_residual;
    double relative_objective_gap;
    double wall_time_ms;
    int iterations;
    termination_reason_t termination_reason;
    bool sparse_metal_active;
};

void destroy_result(mlxpdlp_result_t *result) {
    mlxpdlp_result_free(result);
}

const char *device_name(const mx::Device &device) {
    return device.type == mx::Device::gpu ? "MLX/GPU" : "MLX/CPU";
}

SolveSummary solve_problem(const mlxpdlp_mps_problem_t &problem,
                           const std::vector<double> &objective, const mx::Device &device) {
    pdhg_parameters_t params;
    mlxpdlp_set_default_parameters(&params);
    params.verbose = false;
    params.termination_evaluation_frequency = 100;
    params.termination_criteria.eps_optimal_relative = kCertificateTolerance;
    params.termination_criteria.eps_feasible_relative = kCertificateTolerance;
    params.termination_criteria.iteration_limit = 250000;
    params.termination_criteria.time_sec_limit = 120.0;
    params.sv_max_iter = 2000;
    params.sv_tol = 1e-5;

    auto start = std::chrono::steady_clock::now();
    MlxPdlpSolver solver(problem.num_variables, problem.num_constraints, problem.row_ptr,
                         problem.col_ind, problem.values, problem.variable_lb, problem.variable_ub,
                         problem.constraint_lb, problem.constraint_ub, objective.data(),
                         problem.objective_constant, &params, device);
    if (solver.state().stream.device != device) {
        throw std::runtime_error("solver did not retain the requested device");
    }

    mlxpdlp_result_t *result = solver.solve();
    mx::synchronize(solver.state().stream);
    auto stop = std::chrono::steady_clock::now();

    SolveSummary summary{
        problem.maximize ? -result->primal_objective_value : result->primal_objective_value,
        result->relative_primal_residual,
        result->relative_dual_residual,
        result->relative_objective_gap,
        std::chrono::duration<double, std::milli>(stop - start).count(),
        result->total_count,
        result->termination_reason,
        solver.state().sparse_metal_active,
    };
    destroy_result(result);
    return summary;
}

bool objective_is_close(double actual, double expected) {
    return std::fabs(actual - expected) <=
           kObjectiveRelativeTolerance * std::max(1.0, std::fabs(expected));
}

void print_summary(const char *label, const SolveSummary &summary) {
    std::printf("%-8s objective=% .8e iterations=%d wall=%9.3f ms "
                "rel_pr=% .2e rel_du=% .2e rel_gap=% .2e sparse_metal=%s\n",
                label, summary.objective, summary.iterations, summary.wall_time_ms,
                summary.relative_primal_residual, summary.relative_dual_residual,
                summary.relative_objective_gap, summary.sparse_metal_active ? "yes" : "no");
}

} // namespace

int main(int argc, char **argv) {
    const char *path = argc > 1 ? argv[1] : NETLIB_ADLITTLE_MPS;
    mlxpdlp_mps_problem_t *problem = mlxpdlp_mps_problem_load(path);
    if (!problem) {
        std::fprintf(stderr, "Failed to parse MPS file: %s\n", path);
        return 1;
    }

    std::printf("mlxPDLP MPS device comparison\n");
    std::printf("==================================\n");
    std::printf("MPS: %s\n", path);
    std::printf("Parsed: %d variables, %d constraints, %d matrix nonzeros\n",
                problem->num_variables, problem->num_constraints, problem->num_nonzeros);
    std::printf("Netlib published optimum: %.10e\n\n", kPublishedObjective);

    if (problem->num_variables != kExpectedVariables ||
        problem->num_constraints != kExpectedConstraints) {
        std::fprintf(stderr, "Unexpected ADLITTLE dimensions\n");
        mlxpdlp_mps_problem_free(problem);
        return 1;
    }

    std::vector<double> objective(problem->objective, problem->objective + problem->num_variables);
    if (problem->maximize) {
        for (double &coefficient : objective) {
            coefficient = -coefficient;
        }
    }

    SolveSummary cpu;
    try {
        cpu = solve_problem(*problem, objective, mx::Device::cpu);
    } catch (const std::exception &error) {
        std::fprintf(stderr, "MLX/CPU solve failed: %s\n", error.what());
        mlxpdlp_mps_problem_free(problem);
        return 1;
    }
    print_summary(device_name(mx::Device::cpu), cpu);

    if (cpu.termination_reason != TERMINATION_REASON_OPTIMAL ||
        !objective_is_close(cpu.objective, kPublishedObjective) || cpu.sparse_metal_active) {
        std::fprintf(stderr, "MLX/CPU did not converge to the Netlib optimum\n");
        mlxpdlp_mps_problem_free(problem);
        return 1;
    }

    if (!mx::is_available(mx::Device::gpu)) {
        std::printf("\nMLX/GPU comparison SKIPPED: no GPU backend available\n");
        mlxpdlp_mps_problem_free(problem);
        return kSkip;
    }

    SolveSummary gpu;
    try {
        gpu = solve_problem(*problem, objective, mx::Device::gpu);
    } catch (const std::exception &error) {
        std::fprintf(stderr, "MLX/GPU solve failed: %s\n", error.what());
        mlxpdlp_mps_problem_free(problem);
        return 1;
    }
    print_summary(device_name(mx::Device::gpu), gpu);

    const bool reported_certificate_passes =
        std::isfinite(gpu.relative_primal_residual) &&
        std::isfinite(gpu.relative_dual_residual) &&
        std::isfinite(gpu.relative_objective_gap) &&
        gpu.relative_primal_residual < kCertificateTolerance &&
        gpu.relative_dual_residual < kCertificateTolerance &&
        std::fabs(gpu.relative_objective_gap) < kCertificateTolerance;
    // The original-model audit can demote an internally optimal Metal result.
    // Preserve this test's objective/device focus, but never accept OPTIMAL
    // alongside a failed metric reported in the same result.
    const bool status_matches_certificate =
        gpu.termination_reason == TERMINATION_REASON_UNSPECIFIED ||
        (gpu.termination_reason == TERMINATION_REASON_OPTIMAL &&
         reported_certificate_passes);
    bool objectives_agree = status_matches_certificate &&
                            objective_is_close(gpu.objective, kPublishedObjective) &&
                            objective_is_close(gpu.objective, cpu.objective) &&
                            gpu.sparse_metal_active;
    mlxpdlp_mps_problem_free(problem);
    if (!objectives_agree) {
        std::fprintf(stderr, "CPU/GPU objective, backend, or certificate status mismatch\n");
        return 1;
    }

    const double metal_speedup = cpu.wall_time_ms / gpu.wall_time_ms;
    if (metal_speedup >= 1.0) {
        std::printf("\nADLITTLE Metal speedup: %.3fx (diagnostic only)\n", metal_speedup);
    } else {
        std::printf("\nADLITTLE Metal slowdown: %.3fx (diagnostic only)\n", 1.0 / metal_speedup);
    }
    return 0;
}
