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
Run a fixed number of PDHG iterations on an MPS problem and compare MLX CPU
and Metal wall time. Fixed work, disabled presolve, and zero convergence
tolerances make the device comparison independent of convergence timing.
*/
#include "mlxPDLP/mps_loader.h"
#include "mlxPDLP/solver.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <exception>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

using namespace mlxpdlp;

namespace {

struct ProblemDeleter {
    void operator()(mlxpdlp_mps_problem_t *problem) const {
        mlxpdlp_mps_problem_free(problem);
    }
};

using ProblemPtr = std::unique_ptr<mlxpdlp_mps_problem_t, ProblemDeleter>;

void print_row_profile(const char *label, std::vector<int> lengths) {
    if (lengths.empty()) {
        return;
    }
    const long long nonzeros =
        std::accumulate(lengths.begin(), lengths.end(), 0LL);
    const auto short_rows =
        std::count_if(lengths.begin(), lengths.end(), [](int length) { return length <= 8; });
    const auto empty_rows =
        std::count(lengths.begin(), lengths.end(), 0);
    std::sort(lengths.begin(), lengths.end());
    auto percentile = [&](double fraction) {
        const size_t index = static_cast<size_t>(
            fraction * static_cast<double>(lengths.size() - 1));
        return lengths[index];
    };
    std::printf("%s rows: avg=%.2f p50=%d p90=%d p99=%d max=%d <=8=%.1f%% empty=%.1f%%\n",
                label, static_cast<double>(nonzeros) / lengths.size(), percentile(0.50),
                percentile(0.90), percentile(0.99), lengths.back(),
                100.0 * static_cast<double>(short_rows) / lengths.size(),
                100.0 * static_cast<double>(empty_rows) / lengths.size());
}

struct SolveSummary {
    double setup_seconds;
    double solve_seconds;
    double objective;
    double relative_primal_residual;
    double relative_dual_residual;
    double relative_objective_gap;
    int iterations;
    termination_reason_t termination_reason;
    bool sparse_metal_active;
    bool sparse_cpu_active;
    bool cpu_double_precision_active;
};

enum class DeviceSelection {
    cpu,
    gpu,
    both,
};

int parse_positive_integer(const char *text, const char *name) {
    size_t consumed = 0;
    int value = std::stoi(text, &consumed);
    if (text[consumed] != '\0' || value <= 0) {
        throw std::invalid_argument(std::string(name) + " must be a positive integer");
    }
    return value;
}

DeviceSelection parse_device(const char *text) {
    std::string value(text);
    if (value == "cpu")
        return DeviceSelection::cpu;
    if (value == "gpu" || value == "metal")
        return DeviceSelection::gpu;
    if (value == "both")
        return DeviceSelection::both;
    throw std::invalid_argument("device must be cpu, gpu, metal, or both");
}

const char *device_name(const mx::Device &device) {
    return device.type == mx::Device::gpu ? "MLX/Metal" : "MLX/CPU";
}

const char *termination_name(termination_reason_t reason) {
    switch (reason) {
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
    case TERMINATION_REASON_FEAS_POLISH_SUCCESS:
        return "FEAS_POLISH_SUCCESS";
    case TERMINATION_REASON_HOST_DOUBLE_HANDOFF:
        return "HOST_DOUBLE_HANDOFF";
    case TERMINATION_REASON_UNSPECIFIED:
        return "UNSPECIFIED";
    }
    return "UNKNOWN";
}

SolveSummary solve_problem(const mlxpdlp_mps_problem_t &problem,
                           const std::vector<double> &objective, int iteration_limit,
                           int evaluation_frequency, const mx::Device &device) {
    pdhg_parameters_t params;
    mlxpdlp_set_default_parameters(&params);
    params.verbose = false;
    params.presolve = false;
    params.termination_evaluation_frequency = evaluation_frequency;
    params.termination_criteria.eps_optimal_relative = 0.0;
    params.termination_criteria.eps_feasible_relative = 0.0;
    params.termination_criteria.iteration_limit = iteration_limit;
    params.termination_criteria.time_sec_limit = 3600.0;
    params.sv_max_iter = 100;
    params.sv_tol = 1e-4;

    auto setup_start = std::chrono::steady_clock::now();
    MlxPdlpSolver solver(problem.num_variables, problem.num_constraints, problem.row_ptr,
                         problem.col_ind, problem.values, problem.variable_lb, problem.variable_ub,
                         problem.constraint_lb, problem.constraint_ub, objective.data(),
                         problem.objective_constant, &params, device);
    if (solver.state().stream.device != device) {
        throw std::runtime_error("solver did not retain the requested device");
    }
    mx::synchronize(solver.state().stream);
    auto setup_stop = std::chrono::steady_clock::now();

    auto solve_start = std::chrono::steady_clock::now();
    mlxpdlp_result_t *raw_result = solver.solve();
    std::unique_ptr<mlxpdlp_result_t, decltype(&mlxpdlp_result_free)> result(raw_result,
                                                                             mlxpdlp_result_free);
    mx::synchronize(solver.state().stream);
    auto solve_stop = std::chrono::steady_clock::now();

    return SolveSummary{
        std::chrono::duration<double>(setup_stop - setup_start).count(),
        std::chrono::duration<double>(solve_stop - solve_start).count(),
        problem.maximize ? -result->primal_objective_value : result->primal_objective_value,
        result->relative_primal_residual,
        result->relative_dual_residual,
        result->relative_objective_gap,
        result->total_count,
        result->termination_reason,
        solver.state().sparse_metal_active,
        solver.state().sparse_cpu_active,
        solver.state().cpu_double_precision_active,
    };
}

void print_summary(const char *label, const SolveSummary &summary) {
    double iterations_per_second = summary.iterations / summary.solve_seconds;
    std::printf("%-10s setup=%8.3f s solve=%8.3f s total=%8.3f s "
                "iterations=%d (%8.2f iter/s)\n",
                label, summary.setup_seconds, summary.solve_seconds,
                summary.setup_seconds + summary.solve_seconds, summary.iterations,
                iterations_per_second);
    std::printf("%-10s status=%s objective=% .8e "
                "rel_pr=% .3e rel_du=% .3e rel_gap=% .3e backend=%s\n",
                "", termination_name(summary.termination_reason), summary.objective,
                summary.relative_primal_residual, summary.relative_dual_residual,
                summary.relative_objective_gap,
                summary.sparse_metal_active
                    ? "CSR-Metal-FP32"
                    : (summary.sparse_cpu_active
                           ? "Accelerate-CSR-FP64"
                           : (summary.cpu_double_precision_active ? "dense-MLX-FP64"
                                                                  : "dense-MLX-FP32")));
}

} // namespace

int main(int argc, char **argv) {
    if (argc < 2 || argc > 5) {
        std::fprintf(stderr,
                     "Usage: %s MPS_PATH [ITERATIONS=1000] "
                     "[EVALUATION_FREQUENCY=100] [cpu|gpu|metal|both]\n",
                     argv[0]);
        return 2;
    }

    try {
        int iteration_limit = argc > 2 ? parse_positive_integer(argv[2], "iterations") : 1000;
        int evaluation_frequency =
            argc > 3 ? parse_positive_integer(argv[3], "evaluation frequency") : 100;
        DeviceSelection selection = argc > 4 ? parse_device(argv[4]) : DeviceSelection::both;

        if (evaluation_frequency < 2) {
            throw std::invalid_argument("evaluation frequency must be at least 2");
        }
        if (iteration_limit % evaluation_frequency != 0) {
            throw std::invalid_argument("iterations must be a multiple of evaluation frequency");
        }

        ProblemPtr problem(mlxpdlp_mps_problem_load(argv[1]));
        if (!problem) {
            throw std::runtime_error(std::string("failed to parse MPS file: ") + argv[1]);
        }

        std::vector<double> objective(problem->objective,
                                      problem->objective + problem->num_variables);
        if (problem->maximize) {
            for (double &coefficient : objective)
                coefficient = -coefficient;
        }

        double dense_mib = 2.0 * problem->num_constraints * problem->num_variables * sizeof(float) /
                           (1024.0 * 1024.0);
        double sparse_mib =
            (2.0 * problem->num_nonzeros * (sizeof(float) + sizeof(int32_t)) +
             (problem->num_constraints + problem->num_variables + 2.0) * sizeof(int32_t)) /
            (1024.0 * 1024.0);
        std::printf("mlxPDLP fixed-work MPS benchmark\n");
        std::printf("====================================\n");
        std::printf("MPS: %s\n", argv[1]);
        std::printf("Parsed: %d variables, %d constraints, %d matrix nonzeros\n",
                    problem->num_variables, problem->num_constraints, problem->num_nonzeros);
        std::vector<int> matrix_row_lengths(static_cast<size_t>(problem->num_constraints));
        std::vector<int> transpose_row_lengths(static_cast<size_t>(problem->num_variables), 0);
        for (int row = 0; row < problem->num_constraints; ++row) {
            matrix_row_lengths[static_cast<size_t>(row)] =
                problem->row_ptr[row + 1] - problem->row_ptr[row];
            for (int k = problem->row_ptr[row]; k < problem->row_ptr[row + 1]; ++k) {
                ++transpose_row_lengths[static_cast<size_t>(problem->col_ind[k])];
            }
        }
        print_row_profile("A  ", std::move(matrix_row_lengths));
        print_row_profile("A^T", std::move(transpose_row_lengths));
        std::printf("Dense fallback A + A^T payload: %.1f MiB\n", dense_mib);
        std::printf("Device CSR + transpose CSR payload: %.2f MiB (%.1fx smaller)\n", sparse_mib,
                    dense_mib / sparse_mib);
        std::printf("Sparse Metal preprocessing allocates no dense matrix workspace\n");
        std::printf("Work: %d PDHG iterations, evaluation frequency %d, presolve disabled\n\n",
                    iteration_limit, evaluation_frequency);

        SolveSummary cpu{};
        SolveSummary gpu{};
        bool ran_cpu = selection != DeviceSelection::gpu;
        bool ran_gpu = selection != DeviceSelection::cpu;

        if (ran_cpu) {
            cpu = solve_problem(*problem, objective, iteration_limit, evaluation_frequency,
                                mx::Device::cpu);
            print_summary(device_name(mx::Device::cpu), cpu);
        }

        if (ran_gpu) {
            if (!mx::is_available(mx::Device::gpu)) {
                throw std::runtime_error("MLX Metal device is not available");
            }
            gpu = solve_problem(*problem, objective, iteration_limit, evaluation_frequency,
                                mx::Device::gpu);
            print_summary(device_name(mx::Device::gpu), gpu);
        }

        if (ran_cpu && ran_gpu) {
            std::printf("\nMetal speedup over CPU: %.3fx solve-only, %.3fx end-to-end\n",
                        cpu.solve_seconds / gpu.solve_seconds,
                        (cpu.setup_seconds + cpu.solve_seconds) /
                            (gpu.setup_seconds + gpu.solve_seconds));
        }
        return 0;
    } catch (const std::exception &error) {
        std::fprintf(stderr, "Benchmark failed: %s\n", error.what());
        return 1;
    }
}
