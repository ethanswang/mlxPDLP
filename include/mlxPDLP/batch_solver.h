// Copyright 2026 Ethan Wang <ethanshurui.wang@gmail.com>
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "mlxPDLP/solver.h"
#include <optional>

namespace mlxpdlp {

enum class BatchExecution { automatic, shared, independent };

// Owned data in original coordinates. Empty bound vectors mean unbounded.
// Objectives have exactly n entries; optional starts have exactly n/m entries.
struct BatchProblem {
    std::vector<double> objective;
    double objective_constant = 0.0;
    std::vector<double> variable_lower_bounds, variable_upper_bounds;
    std::vector<double> constraint_lower_bounds, constraint_upper_bounds;
    std::optional<std::vector<double>> primal_start, dual_start, reduced_cost_start;
};

struct BatchOptions {
    BatchExecution execution = BatchExecution::automatic;
    double time_sec_limit = std::numeric_limits<double>::infinity();
    size_t resident_memory_budget_bytes = 0; // zero: unconstrained
    int lp_tile_width = 4;                  // 4 or 8
    int iteration_batch_size = 16;          // K; independent of request width B
};

struct SolveResultDeleter {
    void operator()(mlxpdlp_result_t *value) const { mlxpdlp_result_free(value); }
};
using OwnedSolveResult = std::unique_ptr<mlxpdlp_result_t, SolveResultDeleter>;

struct BatchMemberResult {
    size_t input_index = 0;
    bool has_solution = false; // false only when the member was never admitted
    OwnedSolveResult result;
    std::vector<double> primal_ray, dual_ray; // independently audited original-coordinate rays
    double queue_time_sec = 0.0;
    double execution_time_sec = 0.0;
    int step_size_reductions = 0;
    int original_audit_failures = 0;
};

struct BatchResult {
    std::vector<BatchMemberResult> results;
    BatchExecution execution = BatchExecution::independent;
    std::string fallback_reason;
    double wall_time_sec = 0.0;
    double packing_time_sec = 0.0;
    double construction_time_sec = 0.0;
    double initialization_time_sec = 0.0;
    double pdhg_time_sec = 0.0;
    double checkpoint_time_sec = 0.0;
    double audit_time_sec = 0.0;
    double deadline_overrun_sec = 0.0;
    size_t estimated_peak_resident_bytes = 0;
    size_t groups = 0;
    size_t max_active_width = 0;
    int lp_tile_width = 0;
    int iteration_batch_size = 0;
    bool native_iteration_batching_active = false;
};

// Owns exact original FP64 CSR and immutable matrix preparation. Submissions
// through one plan are serialized. Results own their certificates independently
// of this plan and of subsequent submissions. Presolve uses independent solvers.
class SharedMatrixPlan {
  public:
    SharedMatrixPlan(int num_variables, int num_constraints,
                     const int *row_ptr, const int *col_indices, const double *values,
                     const pdhg_parameters_t *parameters = nullptr,
                     mx::Device device = mx::Device::cpu);
    ~SharedMatrixPlan();
    SharedMatrixPlan(SharedMatrixPlan &&) noexcept;
    SharedMatrixPlan &operator=(SharedMatrixPlan &&) noexcept;
    SharedMatrixPlan(const SharedMatrixPlan &) = delete;
    SharedMatrixPlan &operator=(const SharedMatrixPlan &) = delete;

    BatchResult solve_batch(const std::vector<BatchProblem> &problems,
                            const BatchOptions &options = {},
                            const pdhg_parameters_t *parameters = nullptr);
    int num_variables() const;
    int num_constraints() const;
    int num_nonzeros() const;
    size_t resident_bytes() const;
    double preparation_time_sec() const;
    double operator_norm_upper_bound() const;

  private:
    std::unique_ptr<detail::BatchDriver> impl_;
};
} // namespace mlxpdlp
