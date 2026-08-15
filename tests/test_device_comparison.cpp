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
Runtime comparison of mlxPDLP on MLX CPU and GPU devices.

This is intentionally a correctness comparison with diagnostic timings, not a
benchmark assertion: a tiny LP is often slower on a GPU because launch and
synchronization overhead dominate.
*/
#include "mlxPDLP/solver.h"

#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <exception>
#include <type_traits>
#include <vector>

using namespace mlxpdlp;

namespace {

constexpr int kSkip = 77;
constexpr double kTolerance = 5e-3;

struct SolveSummary {
    std::array<double, 2> primal;
    double objective;
    double wall_time_ms;
    int iterations;
    termination_reason_t termination_reason;
};

const char *device_name(const mx::Device &device) {
    return device.type == mx::Device::gpu ? "MLX/GPU" : "MLX/CPU";
}

void destroy_result(mlxpdlp_result_t *result) {
    mlxpdlp_result_free(result);
}

SolveSummary solve_example_lp(const mx::Device &device) {
    // minimize x0 + x1
    // subject to x0 + 2*x1 = 5, x1 <= 2, 3*x0 + 2*x1 <= 8, x >= 0.
    // The optimum is x = [1, 2], objective = 3.
    constexpr int m = 3;
    constexpr int n = 2;
    int row_ptr[] = {0, 2, 4, 6};
    int col_ind[] = {0, 1, 0, 1, 0, 1};
    double values[] = {1.0, 2.0, 0.0, 1.0, 3.0, 2.0};
    double objective[] = {1.0, 1.0};
    double constraint_lb[] = {5.0, -INFINITY, -INFINITY};
    double constraint_ub[] = {5.0, 2.0, 8.0};
    double variable_lb[] = {0.0, 0.0};
    double variable_ub[] = {INFINITY, INFINITY};

    pdhg_parameters_t params;
    mlxpdlp_set_default_parameters(&params);
    params.verbose = false;
    params.presolve = false;
    params.termination_evaluation_frequency = 50;
    params.termination_criteria.eps_optimal_relative = 1e-6;
    params.termination_criteria.eps_feasible_relative = 1e-6;
    params.termination_criteria.iteration_limit = 100000;
    params.termination_criteria.time_sec_limit = 60.0;

    auto start = std::chrono::steady_clock::now();
    MlxPdlpSolver solver(n, m, row_ptr, col_ind, values, variable_lb, variable_ub, constraint_lb,
                         constraint_ub, objective, 0.0, &params, device);

    if (solver.state().stream.device != device) {
        throw std::runtime_error("solver did not retain the requested MLX device");
    }
    const bool precision_matches_device =
        device.type == mx::Device::cpu
            ? solver.state().cpu_double_precision_active &&
                  solver.state().obj.dtype() == mx::float64
            : !solver.state().cpu_double_precision_active &&
                  solver.state().obj.dtype() == mx::float32;
    if (!precision_matches_device) {
        throw std::runtime_error("solver arithmetic precision does not match its device");
    }

    mlxpdlp_result_t *result = solver.solve();
    mx::synchronize(solver.state().stream);
    auto stop = std::chrono::steady_clock::now();

    SolveSummary summary{
        {result->primal_solution[0], result->primal_solution[1]},
        result->primal_objective_value,
        std::chrono::duration<double, std::milli>(stop - start).count(),
        result->total_count,
        result->termination_reason,
    };
    destroy_result(result);
    return summary;
}

bool close(double lhs, double rhs) {
    return std::fabs(lhs - rhs) <= kTolerance;
}

bool valid_solution(const SolveSummary &summary) {
    return summary.termination_reason == TERMINATION_REASON_OPTIMAL &&
           close(summary.primal[0], 1.0) && close(summary.primal[1], 2.0) &&
           close(summary.objective, 3.0);
}

bool sparse_duplicate_diagonal_matches(const mx::Device &device) {
    constexpr int size = 64;
    std::vector<int> row_ptr(size + 1);
    std::vector<int> col_ind(2 * size);
    std::vector<double> values(2 * size);
    std::vector<double> objective(size, 0.0);
    std::vector<double> constraint_bound(size);
    std::vector<double> variable_lb(size, -INFINITY);
    std::vector<double> variable_ub(size, INFINITY);

    for (int i = 0; i < size; ++i) {
        row_ptr[i] = 2 * i;
        col_ind[2 * i] = i;
        col_ind[2 * i + 1] = i;
        values[2 * i] = 0.25;
        values[2 * i + 1] = 0.75;
        constraint_bound[i] = 1.0 + 0.01 * i;
    }
    row_ptr[size] = 2 * size;

    pdhg_parameters_t params;
    mlxpdlp_set_default_parameters(&params);
    params.verbose = false;
    params.presolve = false;
    params.termination_evaluation_frequency = 50;
    params.termination_criteria.eps_optimal_relative = 1e-5;
    params.termination_criteria.eps_feasible_relative = 1e-5;
    params.termination_criteria.iteration_limit = 100000;
    params.termination_criteria.time_sec_limit = 60.0;
    params.sv_max_iter = 200;

    MlxPdlpSolver solver(size, size, row_ptr.data(), col_ind.data(), values.data(),
                         variable_lb.data(), variable_ub.data(), constraint_bound.data(),
                         constraint_bound.data(), objective.data(), 0.0, &params, device);
    bool storage_matches_backend =
        device.type == mx::Device::gpu
            ? solver.state().A.size() == 0 && solver.state().AT.size() == 0
            : solver.state().A.size() == size * size && solver.state().AT.size() == size * size;
    mlxpdlp_result_t *result = solver.solve();
    mx::synchronize(solver.state().stream);

    bool expected_backend = device.type == mx::Device::gpu ? solver.state().sparse_metal_active
                                                           : !solver.state().sparse_metal_active;
    bool expected_strategy =
        device.type != mx::Device::gpu ||
        (solver.state().sparse_a_spmv_strategy ==
             SparseMetalSpmvStrategy::scalar_rows &&
         solver.state().sparse_at_spmv_strategy ==
             SparseMetalSpmvStrategy::scalar_rows);
    bool valid = storage_matches_backend && expected_backend &&
                 expected_strategy &&
                 result->termination_reason == TERMINATION_REASON_OPTIMAL;
    for (int i = 0; i < size && valid; ++i) {
        valid = std::fabs(result->primal_solution[i] - constraint_bound[i]) <= 5e-3;
    }
    std::printf("%-8s sparse duplicate-diagonal backend=%s iterations=%d %s\n", device_name(device),
                solver.state().sparse_metal_active ? "CSR-Metal" : "dense-MLX", result->total_count,
                valid ? "PASS" : "FAIL");
    destroy_result(result);
    return valid;
}

bool sparse_mixed_row_lengths_match(const mx::Device &device) {
    // A has two entries per row, while A^T has one 1024-entry row mixed with
    // 1023 single-entry rows. This is the shape that requires both branches of
    // the adaptive Metal SpMV kernel in the same dispatch.
    constexpr int size = 1024;
    std::vector<int> row_ptr(static_cast<size_t>(size) + 1);
    std::vector<int> col_ind;
    std::vector<double> values;
    col_ind.reserve(2 * size - 1);
    values.reserve(2 * size - 1);
    for (int row = 0; row < size; ++row) {
        row_ptr[static_cast<size_t>(row)] = static_cast<int>(col_ind.size());
        col_ind.push_back(0);
        values.push_back(1.0);
        if (row > 0) {
            col_ind.push_back(row);
            values.push_back(1.0);
        }
    }
    row_ptr.back() = static_cast<int>(col_ind.size());

    std::vector<double> objective(static_cast<size_t>(size), 1.0);
    objective[0] = size;
    std::vector<double> constraint_bound(static_cast<size_t>(size), 2.0);
    constraint_bound[0] = 1.0;
    std::vector<double> variable_lb(static_cast<size_t>(size), -INFINITY);
    std::vector<double> variable_ub(static_cast<size_t>(size), INFINITY);
    std::vector<double> primal_start(static_cast<size_t>(size), 1.0);
    std::vector<double> dual_start(static_cast<size_t>(size), 1.0);

    pdhg_parameters_t params;
    mlxpdlp_set_default_parameters(&params);
    params.verbose = false;
    params.presolve = false;
    params.curtis_reid_iterations = 0;
    params.l_inf_ruiz_iterations = 0;
    params.has_pock_chambolle_alpha = false;
    params.bound_objective_rescaling = false;
    params.feasibility_polishing = false;
    params.host_double_polishing = false;
    params.sv_max_iter = 20;
    params.termination_evaluation_frequency = 2;
    params.termination_criteria.eps_optimal_relative = 0.0;
    params.termination_criteria.eps_feasible_relative = 0.0;
    params.termination_criteria.iteration_limit = 0;
    params.termination_criteria.time_sec_limit = 10.0;

    MlxPdlpSolver solver(size, size, row_ptr.data(), col_ind.data(), values.data(),
                         variable_lb.data(), variable_ub.data(), constraint_bound.data(),
                         constraint_bound.data(), objective.data(), 0.0, &params,
                         primal_start.data(), dual_start.data(), device);
    const bool expected_sparse_backend =
        device.type != mx::Device::gpu || solver.expects_sparse_metal_backend();
    mlxpdlp_result_t *result = solver.solve();
    mx::synchronize(solver.state().stream);

    bool valid = expected_sparse_backend &&
                 (device.type != mx::Device::gpu || solver.state().sparse_metal_active) &&
                 (device.type != mx::Device::gpu ||
                  (solver.state().sparse_a_spmv_strategy ==
                       SparseMetalSpmvStrategy::scalar_rows &&
                   solver.state().sparse_at_spmv_strategy ==
                       SparseMetalSpmvStrategy::adaptive)) &&
                 result->relative_primal_residual <= 1e-6 &&
                 result->relative_dual_residual <= 1e-6 &&
                 result->relative_objective_gap <= 1e-6;
    for (int i = 0; i < size && valid; ++i) {
        valid = std::fabs(result->primal_solution[i] - 1.0) <= 1e-6;
    }
    std::printf("%-8s sparse mixed-row backend=%s residuals=(%.2e, %.2e, %.2e) %s\n",
                device_name(device),
                solver.state().sparse_metal_active ? "CSR-Metal" : "dense-MLX",
                result->relative_primal_residual, result->relative_dual_residual,
                result->relative_objective_gap, valid ? "PASS" : "FAIL");
    destroy_result(result);
    return valid;
}

bool sparse_simdgroup_rows_match() {
    // Uniform 65-entry rows sit immediately above the adaptive kernel's
    // scalar-row cutoff. They must use one SIMD group per row rather than a
    // 256-thread, eight-barrier reduction for each row.
    constexpr int size = 512;
    constexpr int entries_per_row = 65;
    constexpr double diagonal_value = 2.0;
    constexpr double off_diagonal_value = 0.01;
    const double row_sum =
        diagonal_value + (entries_per_row - 1) * off_diagonal_value;

    std::vector<int> row_ptr(static_cast<size_t>(size) + 1);
    std::vector<int> col_ind;
    std::vector<double> values;
    col_ind.reserve(static_cast<size_t>(size) * entries_per_row);
    values.reserve(static_cast<size_t>(size) * entries_per_row);
    for (int row = 0; row < size; ++row) {
        row_ptr[static_cast<size_t>(row)] = static_cast<int>(col_ind.size());
        for (int entry = 0; entry < entries_per_row; ++entry) {
            col_ind.push_back((row + entry) % size);
            values.push_back(entry == 0 ? diagonal_value : off_diagonal_value);
        }
    }
    row_ptr.back() = static_cast<int>(col_ind.size());

    std::vector<double> objective(static_cast<size_t>(size), 0.0);
    std::vector<double> constraint_bound(static_cast<size_t>(size), row_sum);
    std::vector<double> variable_lb(static_cast<size_t>(size), -INFINITY);
    std::vector<double> variable_ub(static_cast<size_t>(size), INFINITY);
    std::vector<double> primal_start(static_cast<size_t>(size), 1.0);
    std::vector<double> dual_start(static_cast<size_t>(size), 0.0);

    pdhg_parameters_t params;
    mlxpdlp_set_default_parameters(&params);
    params.verbose = false;
    params.presolve = false;
    params.curtis_reid_iterations = 0;
    params.l_inf_ruiz_iterations = 0;
    params.has_pock_chambolle_alpha = false;
    params.bound_objective_rescaling = false;
    params.feasibility_polishing = false;
    params.host_double_polishing = false;
    params.sv_max_iter = 20;
    params.termination_evaluation_frequency = 2;
    params.termination_criteria.eps_optimal_relative = 0.0;
    params.termination_criteria.eps_feasible_relative = 0.0;
    params.termination_criteria.iteration_limit = 0;
    params.termination_criteria.time_sec_limit = 10.0;

    MlxPdlpSolver solver(size, size, row_ptr.data(), col_ind.data(), values.data(),
                         variable_lb.data(), variable_ub.data(), constraint_bound.data(),
                         constraint_bound.data(), objective.data(), 0.0, &params,
                         primal_start.data(), dual_start.data(), mx::Device::gpu);
    mlxpdlp_result_t *result = solver.solve();
    mx::synchronize(solver.state().stream);

    bool valid = solver.state().sparse_metal_active &&
                 solver.state().sparse_a_spmv_strategy ==
                     SparseMetalSpmvStrategy::simdgroup_rows &&
                 solver.state().sparse_at_spmv_strategy ==
                     SparseMetalSpmvStrategy::simdgroup_rows &&
                 result->relative_primal_residual <= 1e-6 &&
                 result->relative_dual_residual <= 1e-6 &&
                 result->relative_objective_gap <= 1e-6;
    for (int i = 0; i < size && valid; ++i) {
        valid = std::fabs(result->primal_solution[i] - 1.0) <= 1e-6;
    }
    std::printf("MLX/GPU  sparse SIMD-group 65-entry rows residuals=(%.2e, %.2e, %.2e) %s\n",
                result->relative_primal_residual, result->relative_dual_residual,
                result->relative_objective_gap, valid ? "PASS" : "FAIL");
    destroy_result(result);
    return valid;
}

bool sparse_fused_unfused_match() {
    // The fused and unfused Metal iteration paths must produce equivalent
    // trajectories. With the default reflection coefficient both paths execute
    // identical fp32 arithmetic, so a tight elementwise tolerance is possible.
    // The problem mixes a 2048-entry A^T row with short rows so both the
    // adaptive and scalar-row fused kernels run in the same solve.
    constexpr int size = 2048;
    std::vector<int> row_ptr(static_cast<size_t>(size) + 1);
    std::vector<int> col_ind;
    std::vector<double> values;
    col_ind.reserve(2 * size - 1);
    values.reserve(2 * size - 1);
    for (int row = 0; row < size; ++row) {
        row_ptr[static_cast<size_t>(row)] = static_cast<int>(col_ind.size());
        col_ind.push_back(0);
        values.push_back(1.0);
        if (row > 0) {
            col_ind.push_back(row);
            values.push_back(1.0);
        }
    }
    row_ptr.back() = static_cast<int>(col_ind.size());

    std::vector<double> objective(static_cast<size_t>(size), 1.0);
    objective[0] = size;
    std::vector<double> constraint_bound(static_cast<size_t>(size), 2.0);
    constraint_bound[0] = 1.0;
    std::vector<double> variable_lb(static_cast<size_t>(size), -INFINITY);
    std::vector<double> variable_ub(static_cast<size_t>(size), INFINITY);
    std::vector<double> primal_start(static_cast<size_t>(size), 1.0);
    std::vector<double> dual_start(static_cast<size_t>(size), 1.0);

    auto solve_once = [&](bool fused) {
        pdhg_parameters_t params;
        mlxpdlp_set_default_parameters(&params);
        params.verbose = false;
        params.metal_fused_kernels = fused;
        params.presolve = false;
        params.curtis_reid_iterations = 0;
        params.l_inf_ruiz_iterations = 0;
        params.has_pock_chambolle_alpha = false;
        params.bound_objective_rescaling = false;
        params.feasibility_polishing = false;
        params.host_double_polishing = false;
        params.sv_max_iter = 20;
        params.termination_evaluation_frequency = 10;
        params.termination_criteria.eps_optimal_relative = 0.0;
        params.termination_criteria.eps_feasible_relative = 0.0;
        params.termination_criteria.iteration_limit = 40;
        params.termination_criteria.time_sec_limit = 10.0;

        MlxPdlpSolver solver(size, size, row_ptr.data(), col_ind.data(), values.data(),
                             variable_lb.data(), variable_ub.data(),
                             constraint_bound.data(), constraint_bound.data(),
                             objective.data(), 0.0, &params, primal_start.data(),
                             dual_start.data(), mx::Device::gpu);
        return solver.solve();
    };

    mlxpdlp_result_t *fused = solve_once(true);
    mlxpdlp_result_t *unfused = solve_once(false);
    mx::synchronize();

    bool valid = fused && unfused;
    if (valid) {
        for (int i = 0; i < size && valid; ++i) {
            valid = std::fabs(fused->primal_solution[i] - unfused->primal_solution[i]) <=
                        1e-6 * (1.0 + std::fabs(unfused->primal_solution[i])) &&
                    std::fabs(fused->dual_solution[i] - unfused->dual_solution[i]) <=
                        1e-6 * (1.0 + std::fabs(unfused->dual_solution[i]));
        }
        valid = valid && fused->total_count == unfused->total_count &&
                std::fabs(fused->primal_objective_value -
                          unfused->primal_objective_value) <= 1e-6;
    }
    std::printf("MLX/GPU  fused/unfused iteration agreement %s\n",
                valid ? "PASS" : "FAIL");
    destroy_result(fused);
    destroy_result(unfused);
    return valid;
}

bool sparse_fused_unfused_simdgroup_match() {
    // Same agreement check as sparse_fused_unfused_match, but on a uniform
    // 65-entry-row matrix so BOTH matrix directions dispatch to the
    // simdgroup_rows fused kernel (the other agreement test covers the
    // scalar-row and adaptive variants).
    constexpr int size = 512;
    constexpr int entries_per_row = 65;
    constexpr double diagonal_value = 2.0;
    constexpr double off_diagonal_value = 0.01;
    const double row_sum =
        diagonal_value + (entries_per_row - 1) * off_diagonal_value;

    std::vector<int> row_ptr(static_cast<size_t>(size) + 1);
    std::vector<int> col_ind;
    std::vector<double> values;
    col_ind.reserve(static_cast<size_t>(size) * entries_per_row);
    values.reserve(static_cast<size_t>(size) * entries_per_row);
    for (int row = 0; row < size; ++row) {
        row_ptr[static_cast<size_t>(row)] = static_cast<int>(col_ind.size());
        for (int entry = 0; entry < entries_per_row; ++entry) {
            col_ind.push_back((row + entry) % size);
            values.push_back(entry == 0 ? diagonal_value : off_diagonal_value);
        }
    }
    row_ptr.back() = static_cast<int>(col_ind.size());

    std::vector<double> objective(static_cast<size_t>(size), 0.0);
    std::vector<double> constraint_bound(static_cast<size_t>(size), row_sum);
    std::vector<double> variable_lb(static_cast<size_t>(size), -INFINITY);
    std::vector<double> variable_ub(static_cast<size_t>(size), INFINITY);
    std::vector<double> primal_start(static_cast<size_t>(size), 1.0);
    std::vector<double> dual_start(static_cast<size_t>(size), 0.0);

    auto solve_once = [&](bool fused) {
        pdhg_parameters_t params;
        mlxpdlp_set_default_parameters(&params);
        params.verbose = false;
        params.metal_fused_kernels = fused;
        params.presolve = false;
        params.curtis_reid_iterations = 0;
        params.l_inf_ruiz_iterations = 0;
        params.has_pock_chambolle_alpha = false;
        params.bound_objective_rescaling = false;
        params.feasibility_polishing = false;
        params.host_double_polishing = false;
        params.sv_max_iter = 20;
        params.termination_evaluation_frequency = 10;
        params.termination_criteria.eps_optimal_relative = 0.0;
        params.termination_criteria.eps_feasible_relative = 0.0;
        params.termination_criteria.iteration_limit = 40;
        params.termination_criteria.time_sec_limit = 10.0;

        MlxPdlpSolver solver(size, size, row_ptr.data(), col_ind.data(), values.data(),
                             variable_lb.data(), variable_ub.data(),
                             constraint_bound.data(), constraint_bound.data(),
                             objective.data(), 0.0, &params, primal_start.data(),
                             dual_start.data(), mx::Device::gpu);
        mlxpdlp_result_t *result = solver.solve();
        // The dispatch strategies are selected during prepare_sparse_metal_backend
        // inside solve(); inspect them only after the solve.
        if (solver.state().sparse_a_spmv_strategy !=
                SparseMetalSpmvStrategy::simdgroup_rows ||
            solver.state().sparse_at_spmv_strategy !=
                SparseMetalSpmvStrategy::simdgroup_rows) {
            destroy_result(result);
            throw std::runtime_error("fixture did not select the simdgroup-row strategy");
        }
        return result;
    };

    mlxpdlp_result_t *fused = solve_once(true);
    mlxpdlp_result_t *unfused = solve_once(false);
    mx::synchronize();

    bool valid = fused && unfused;
    if (valid) {
        for (int i = 0; i < size && valid; ++i) {
            valid = std::fabs(fused->primal_solution[i] - unfused->primal_solution[i]) <=
                        1e-6 * (1.0 + std::fabs(unfused->primal_solution[i])) &&
                    std::fabs(fused->dual_solution[i] - unfused->dual_solution[i]) <=
                        1e-6 * (1.0 + std::fabs(unfused->dual_solution[i]));
        }
        valid = valid && fused->total_count == unfused->total_count;
    }
    std::printf("MLX/GPU  fused/unfused SIMD-group iteration agreement %s\n",
                valid ? "PASS" : "FAIL");
    destroy_result(fused);
    destroy_result(unfused);
    return valid;
}

bool spmv_simdgroup_threshold_override() {
    // A uniform 32-entry-row matrix lies below the default SIMD-group
    // aggregate-work cutoff (adaptive path), but above a small explicit
    // MLXPDLP_SPMV_SIMD_NNZ_THRESHOLD override (SIMD-group path). The override
    // must actually change the selected dispatch strategy.
    constexpr int size = 512;
    constexpr int entries_per_row = 32;
    std::vector<int> row_ptr(static_cast<size_t>(size) + 1);
    std::vector<int> col_ind;
    std::vector<double> values;
    col_ind.reserve(static_cast<size_t>(size) * entries_per_row);
    values.reserve(static_cast<size_t>(size) * entries_per_row);
    for (int row = 0; row < size; ++row) {
        row_ptr[static_cast<size_t>(row)] = static_cast<int>(col_ind.size());
        for (int entry = 0; entry < entries_per_row; ++entry) {
            col_ind.push_back((row + entry) % size);
            values.push_back(entry == 0 ? 2.0 : 0.01);
        }
    }
    row_ptr.back() = static_cast<int>(col_ind.size());

    std::vector<double> objective(static_cast<size_t>(size), 0.0);
    std::vector<double> constraint_bound(static_cast<size_t>(size), 2.0 + 31 * 0.01);
    std::vector<double> variable_lb(static_cast<size_t>(size), -INFINITY);
    std::vector<double> variable_ub(static_cast<size_t>(size), INFINITY);

    auto solve_with_strategy = [&]() {
        pdhg_parameters_t params;
        mlxpdlp_set_default_parameters(&params);
        params.verbose = false;
        params.presolve = false;
        params.termination_evaluation_frequency = 2;
        params.termination_criteria.iteration_limit = 2;
        params.termination_criteria.time_sec_limit = 10.0;
        MlxPdlpSolver solver(size, size, row_ptr.data(), col_ind.data(), values.data(),
                             variable_lb.data(), variable_ub.data(),
                             constraint_bound.data(), constraint_bound.data(),
                             objective.data(), 0.0, &params, mx::Device::gpu);
        mlxpdlp_result_t *result = solver.solve();
        const auto strategy = solver.state().sparse_a_spmv_strategy;
        destroy_result(result);
        return strategy;
    };

    const bool has_gpu = mx::is_available(mx::Device::gpu);
    if (!has_gpu) {
        return true;
    }

    const SparseMetalSpmvStrategy default_strategy = solve_with_strategy();
    if (default_strategy != SparseMetalSpmvStrategy::adaptive &&
        default_strategy != SparseMetalSpmvStrategy::simdgroup_rows) {
        std::printf("MLX/GPU  SpMV SIMD-group threshold override SKIP (default=%d)\n",
                    static_cast<int>(default_strategy));
        return true;
    }

    setenv("MLXPDLP_SPMV_SIMD_NNZ_THRESHOLD", "1000", 1);
    const SparseMetalSpmvStrategy overridden_strategy = solve_with_strategy();
    unsetenv("MLXPDLP_SPMV_SIMD_NNZ_THRESHOLD");

    const bool valid =
        default_strategy != overridden_strategy &&
        overridden_strategy == SparseMetalSpmvStrategy::simdgroup_rows;
    std::printf("MLX/GPU  SpMV SIMD-group threshold override %s\n",
                valid ? "PASS" : "FAIL");
    return valid;
}

bool infeasibility_certificates_match() {
    // Two synthetic certificates exercise the wired Farkas termination:
    //  - a provably primal-infeasible LP (x = 3 and x = 2 simultaneously),
    //  - a provably unbounded LP (min -x subject to x >= 0).
    // FP64 certifies both cleanly; FP32 ray arithmetic only reliably
    // certifies the unbounded case (see docs/architecture.md).
    auto make_params = []() {
        pdhg_parameters_t params;
        mlxpdlp_set_default_parameters(&params);
        params.verbose = false;
        params.presolve = false;
        params.feasibility_polishing = false;
        params.host_double_polishing = false;
        params.termination_criteria.iteration_limit = 100000;
        params.termination_criteria.time_sec_limit = 60.0;
        return params;
    };

    bool valid = true;

    {
        // Primal infeasible (FP64 CPU path).
        int row_ptr[] = {0, 1, 2};
        int col_ind[] = {0, 0};
        double values[] = {1.0, 1.0};
        double obj[] = {0.0};
        double con_lb[] = {3.0, 2.0};
        double con_ub[] = {3.0, 2.0};
        double var_lb[] = {-INFINITY};
        double var_ub[] = {INFINITY};
        pdhg_parameters_t params = make_params();
        MlxPdlpSolver solver(1, 2, row_ptr, col_ind, values, var_lb, var_ub,
                             con_lb, con_ub, obj, 0.0, &params, mx::Device::cpu);
        mlxpdlp_result_t *result = solver.solve();
        valid = valid && result->termination_reason == TERMINATION_REASON_PRIMAL_INFEASIBLE &&
                result->total_count < 100000;
        std::printf("MLX/CPU  primal-infeasible certificate %s\n",
                    result->termination_reason == TERMINATION_REASON_PRIMAL_INFEASIBLE
                        ? "PASS" : "FAIL");
        destroy_result(result);
    }

    const mx::Device devices[] = {mx::Device::cpu, mx::Device::gpu};
    for (const mx::Device &device : devices) {
        if (device.type == mx::Device::gpu && !mx::is_available(mx::Device::gpu)) {
            continue;
        }
        int row_ptr[] = {0, 1};
        int col_ind[] = {0};
        double values[] = {1.0};
        double obj[] = {-1.0};
        double con_lb[] = {0.0};
        double con_ub[] = {INFINITY};
        double var_lb[] = {-INFINITY};
        double var_ub[] = {INFINITY};
        pdhg_parameters_t params = make_params();
        MlxPdlpSolver solver(1, 1, row_ptr, col_ind, values, var_lb, var_ub,
                             con_lb, con_ub, obj, 0.0, &params, device);
        mlxpdlp_result_t *result = solver.solve();
        const bool ok = result->termination_reason == TERMINATION_REASON_DUAL_INFEASIBLE &&
                        result->total_count < 100000;
        valid = valid && ok;
        std::printf("%-8s dual-infeasible certificate %s\n", device_name(device),
                    ok ? "PASS" : "FAIL");
        destroy_result(result);
    }
    return valid;
}

bool metal_host_double_handoff_uses_saved_checkpoint() {
    int row_ptr[] = {0, 1};
    int col_ind[] = {0};
    double values[] = {1.0};
    double objective[] = {1.0};
    double constraint_bound[] = {1.0};
    double variable_lb[] = {0.0};
    double variable_ub[] = {INFINITY};
    double primal_start[] = {1.0};
    double dual_start[] = {1.0};

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
    params.termination_criteria.eps_optimal_relative = 1e-6;
    // Strict comparisons keep the exact point out of the ordinary OPTIMAL
    // branch so the saved-checkpoint handoff is exercised deterministically.
    params.termination_criteria.eps_feasible_relative = 0.0;
    params.termination_criteria.iteration_limit = 200;
    params.termination_criteria.time_sec_limit = 10.0;

    MlxPdlpSolver solver(1, 1, row_ptr, col_ind, values, variable_lb,
                         variable_ub, constraint_bound, constraint_bound,
                         objective, 0.0, &params, primal_start, dual_start,
                         mx::Device::gpu);
    mlxpdlp_result_t *result = solver.solve();
    mx::synchronize(solver.state().stream);
    const bool valid = !solver.state().cpu_double_precision_active &&
                       solver.state().x_pdhg.dtype() == mx::float32 &&
                       result->host_double_handoff &&
                       result->total_count <
                           params.termination_criteria.iteration_limit &&
                       std::fabs(result->primal_solution[0] - 1.0) <= 1e-9;
    std::printf("MLX/GPU  saved host-double checkpoint %s\n",
                valid ? "PASS" : "FAIL");
    destroy_result(result);
    return valid;
}

void print_summary(const char *label, const SolveSummary &summary) {
    std::printf("%-8s x=[%.8f, %.8f] obj=%.8f iterations=%d wall=%.3f ms\n", label,
                summary.primal[0], summary.primal[1], summary.objective, summary.iterations,
                summary.wall_time_ms);
}

} // namespace

int main() {
    // This compile-time assertion makes the backend dependency explicit: the
    // solver matrix is an mlx::core::array, not a std::vector CPU matrix.
    static_assert(std::is_same_v<std::remove_cvref_t<decltype(MlxPdlpState::A)>, mx::array>);

    std::printf("mlxPDLP device comparison\n");
    std::printf("==============================\n");
    std::printf("LP: min x0+x1 with optimum x=[1,2], objective=3\n\n");

    SolveSummary cpu;
    try {
        cpu = solve_example_lp(mx::Device::cpu);
    } catch (const std::exception &error) {
        std::fprintf(stderr, "MLX/CPU solve failed: %s\n", error.what());
        return 1;
    }
    print_summary(device_name(mx::Device::cpu), cpu);
    if (!valid_solution(cpu)) {
        std::fprintf(stderr, "MLX/CPU returned an incorrect LP solution\n");
        return 1;
    }

    if (!mx::is_available(mx::Device::gpu)) {
        std::printf("\nMLX/GPU comparison SKIPPED: this MLX library has no GPU "
                    "backend (for this checkout, inspect MLX_BUILD_METAL in "
                    "MLX's CMakeCache.txt).\n");
        return kSkip;
    }

    SolveSummary gpu;
    try {
        gpu = solve_example_lp(mx::Device::gpu);
    } catch (const std::exception &error) {
        std::fprintf(stderr, "\nMLX/GPU solve failed despite an available GPU: %s\n", error.what());
        return 1;
    }
    print_summary(device_name(mx::Device::gpu), gpu);

    if (!valid_solution(gpu) || !close(cpu.primal[0], gpu.primal[0]) ||
        !close(cpu.primal[1], gpu.primal[1]) || !close(cpu.objective, gpu.objective)) {
        std::fprintf(stderr, "MLX/CPU and MLX/GPU solutions differ\n");
        return 1;
    }

    if (!metal_host_double_handoff_uses_saved_checkpoint()) {
        std::fprintf(stderr, "Metal host-double handoff regression failed\n");
        return 1;
    }

    if (!sparse_duplicate_diagonal_matches(mx::Device::cpu) ||
        !sparse_duplicate_diagonal_matches(mx::Device::gpu)) {
        std::fprintf(stderr, "sparse duplicate-coordinate regression failed\n");
        return 1;
    }
    if (!sparse_mixed_row_lengths_match(mx::Device::gpu)) {
        std::fprintf(stderr, "adaptive sparse mixed-row regression failed\n");
        return 1;
    }
    if (!sparse_simdgroup_rows_match()) {
        std::fprintf(stderr, "sparse SIMD-group row regression failed\n");
        return 1;
    }
    if (!sparse_fused_unfused_match()) {
        std::fprintf(stderr, "fused/unfused Metal iteration regression failed\n");
        return 1;
    }
    if (!sparse_fused_unfused_simdgroup_match()) {
        std::fprintf(stderr, "fused/unfused SIMD-group Metal iteration regression failed\n");
        return 1;
    }
    if (!spmv_simdgroup_threshold_override()) {
        std::fprintf(stderr, "SpMV SIMD-group threshold override regression failed\n");
        return 1;
    }
    if (!infeasibility_certificates_match()) {
        std::fprintf(stderr, "infeasibility certificate regression failed\n");
        return 1;
    }

    const double metal_speedup = cpu.wall_time_ms / gpu.wall_time_ms;
    if (metal_speedup >= 1.0) {
        std::printf("\nTiny-LP Metal speedup: %.3fx (diagnostic only)\n", metal_speedup);
    } else {
        std::printf("\nTiny-LP Metal slowdown: %.3fx; launch overhead dominates "
                    "(diagnostic only)\n",
                    1.0 / metal_speedup);
    }
    return 0;
}
