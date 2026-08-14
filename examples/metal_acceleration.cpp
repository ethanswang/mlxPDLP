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
Self-contained Metal acceleration demonstration.

The program generates a diagonally dominant sparse equality LP with the known
solution x = 1, warms both MLX devices, and times the same fixed number of PDHG
iterations on CPU FP64 and Metal FP32. Fixed work avoids presenting different
convergence trajectories as a hardware speedup.
*/
#include "mlxPDLP/solver.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <exception>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

using namespace mlxpdlp;

namespace {

constexpr int kEntriesPerRow = 64;
constexpr int kEvaluationFrequency = 100;
constexpr int kWarmupIterations = 100;

struct GeneratedProblem {
    int size;
    std::vector<int> row_ptr;
    std::vector<int> col_ind;
    std::vector<double> values;
    std::vector<double> objective;
    std::vector<double> variable_lb;
    std::vector<double> variable_ub;
    std::vector<double> constraint_bound;
};

struct SolveSummary {
    double setup_seconds;
    double solve_seconds;
    int iterations;
    bool sparse_metal_active;
    bool sparse_cpu_active;
    bool cpu_double_precision_active;
};

int parse_positive_integer(const char *text, const char *name) {
    size_t consumed = 0;
    int value = std::stoi(text, &consumed);
    if (text[consumed] != '\0' || value <= 0) {
        throw std::invalid_argument(std::string(name) + " must be a positive integer");
    }
    return value;
}

GeneratedProblem make_problem(int size) {
    GeneratedProblem problem{
        size,
        std::vector<int>(static_cast<size_t>(size) + 1),
        {},
        {},
        std::vector<double>(static_cast<size_t>(size), 0.0),
        std::vector<double>(static_cast<size_t>(size), 0.0),
        std::vector<double>(static_cast<size_t>(size), 2.0),
        std::vector<double>(static_cast<size_t>(size), 0.0),
    };
    problem.col_ind.reserve(static_cast<size_t>(size) * kEntriesPerRow);
    problem.values.reserve(static_cast<size_t>(size) * kEntriesPerRow);

    constexpr double diagonal_value = 2.0;
    constexpr double off_diagonal_value = 0.01;
    const double row_sum = diagonal_value + (kEntriesPerRow - 1) * off_diagonal_value;

    for (int row = 0; row < size; ++row) {
        problem.row_ptr[static_cast<size_t>(row)] = static_cast<int>(problem.col_ind.size());
        for (int entry = 0; entry < kEntriesPerRow; ++entry) {
            const int column = (row + entry) % size;
            problem.col_ind.push_back(column);
            problem.values.push_back(entry == 0 ? diagonal_value : off_diagonal_value);
        }
        problem.constraint_bound[static_cast<size_t>(row)] = row_sum;
    }
    problem.row_ptr.back() = static_cast<int>(problem.col_ind.size());
    return problem;
}

pdhg_parameters_t fixed_work_parameters(int iterations) {
    pdhg_parameters_t params;
    mlxpdlp_set_default_parameters(&params);
    params.verbose = false;
    params.presolve = false;
    params.feasibility_polishing = false;
    params.host_double_polishing = false;
    params.curtis_reid_iterations = 0;
    params.l_inf_ruiz_iterations = 0;
    params.has_pock_chambolle_alpha = false;
    params.bound_objective_rescaling = false;
    params.sv_max_iter = 20;
    params.sv_tol = 1e-4;
    params.termination_evaluation_frequency =
        iterations < kEvaluationFrequency ? iterations : kEvaluationFrequency;
    params.termination_criteria.eps_optimal_relative = 0.0;
    params.termination_criteria.eps_feasible_relative = 0.0;
    params.termination_criteria.iteration_limit = iterations;
    params.termination_criteria.time_sec_limit = 300.0;
    return params;
}

SolveSummary run_fixed_work(const GeneratedProblem &problem, int iterations,
                            const mx::Device &device) {
    pdhg_parameters_t params = fixed_work_parameters(iterations);

    const auto setup_start = std::chrono::steady_clock::now();
    MlxPdlpSolver solver(problem.size, problem.size, problem.row_ptr.data(), problem.col_ind.data(),
                         problem.values.data(), problem.variable_lb.data(),
                         problem.variable_ub.data(), problem.constraint_bound.data(),
                         problem.constraint_bound.data(), problem.objective.data(), 0.0, &params,
                         device);
    if (solver.state().stream.device != device) {
        throw std::runtime_error("solver did not retain the requested MLX device");
    }
    mx::synchronize(solver.state().stream);
    const auto setup_stop = std::chrono::steady_clock::now();

    const auto solve_start = std::chrono::steady_clock::now();
    std::unique_ptr<mlxpdlp_result_t, decltype(&mlxpdlp_result_free)> result(solver.solve(),
                                                                             mlxpdlp_result_free);
    mx::synchronize(solver.state().stream);
    const auto solve_stop = std::chrono::steady_clock::now();

    return SolveSummary{
        std::chrono::duration<double>(setup_stop - setup_start).count(),
        std::chrono::duration<double>(solve_stop - solve_start).count(),
        result->total_count,
        solver.state().sparse_metal_active,
        solver.state().sparse_cpu_active,
        solver.state().cpu_double_precision_active,
    };
}

const char *backend_name(const SolveSummary &summary) {
    if (summary.sparse_metal_active)
        return "CSR-Metal-FP32";
    if (summary.sparse_cpu_active)
        return "Accelerate-CSR-FP64";
    return summary.cpu_double_precision_active ? "dense-MLX-FP64" : "dense-MLX-FP32";
}

void print_summary(const char *label, const SolveSummary &summary) {
    std::printf("%-10s backend=%-21s setup=%7.3f s solve=%7.3f s "
                "iterations=%d\n",
                label, backend_name(summary), summary.setup_seconds, summary.solve_seconds,
                summary.iterations);
}

} // namespace

int main(int argc, char **argv) {
    if (argc > 3) {
        std::fprintf(stderr, "Usage: %s [SIZE=163840] [ITERATIONS=1000]\n", argv[0]);
        return 2;
    }

    try {
        const int size = argc > 1 ? parse_positive_integer(argv[1], "size") : 163840;
        const int iterations = argc > 2 ? parse_positive_integer(argv[2], "iterations") : 1000;
        if (size < 4096) {
            throw std::invalid_argument(
                "size must be at least 4096 to exercise both sparse backends");
        }
        if (iterations < kEvaluationFrequency || iterations % kEvaluationFrequency != 0) {
            throw std::invalid_argument("iterations must be a positive multiple of 100");
        }
        if (!mx::is_available(mx::Device::gpu)) {
            std::printf("SKIPPED: MLX reports no Metal GPU device.\n");
            return 77;
        }

        GeneratedProblem problem = make_problem(size);
        std::printf("mlxPDLP self-contained Metal acceleration example\n");
        std::printf("================================================\n");
        std::printf("Generated equality LP: %d rows, %d columns, %zu nonzeros\n", size, size,
                    problem.values.size());
        std::printf("Timing protocol: %d fixed PDHG iterations per device (not convergence)\n",
                    iterations);
        std::printf("Warming CPU and Metal kernels (not timed)...\n");

        (void)run_fixed_work(problem, kWarmupIterations, mx::Device::cpu);
        (void)run_fixed_work(problem, kWarmupIterations, mx::Device::gpu);

        const SolveSummary cpu = run_fixed_work(problem, iterations, mx::Device::cpu);
        const SolveSummary metal = run_fixed_work(problem, iterations, mx::Device::gpu);
        print_summary("MLX/CPU", cpu);
        print_summary("MLX/Metal", metal);

        if (!cpu.sparse_cpu_active || !metal.sparse_metal_active || cpu.iterations != iterations ||
            metal.iterations != iterations || !std::isfinite(cpu.solve_seconds) ||
            !std::isfinite(metal.solve_seconds)) {
            throw std::runtime_error(
                "fixed-work comparison did not exercise the expected backends");
        }

        const double solve_speedup = cpu.solve_seconds / metal.solve_seconds;
        const double end_to_end_speedup =
            (cpu.setup_seconds + cpu.solve_seconds) / (metal.setup_seconds + metal.solve_seconds);
        std::printf("\nMetal speedup: %.2fx solve-only, %.2fx including setup\n", solve_speedup,
                    end_to_end_speedup);
        if (solve_speedup <= 1.0) {
            std::printf("No acceleration was observed in this run; results depend on Mac model, "
                        "power state, and workload size.\n");
        }
        std::printf("CPU uses FP64; Metal uses FP32. This is equal work, not an accuracy claim.\n");
        return 0;
    } catch (const std::exception &error) {
        std::fprintf(stderr, "Acceleration example failed: %s\n", error.what());
        return 1;
    }
}
