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

#include "mlx/allocator.h"
#include "mlx/backend/cpu/encoder.h"
#include "mlx/backend/gpu/device_info.h"
#include "mlx/fast.h"
#include "mlx/primitives.h"

#ifdef MLXPDLP_HAS_ACCELERATE_SPARSE
#include <Accelerate/Accelerate.h>
#endif

#ifdef MLXPDLP_HAS_PRESOLVE
#include "presolve_adapter.h"
#endif

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <random>
#include <stdexcept>
#include <utility>
#include <variant>

namespace mlxpdlp {

namespace detail {

#ifdef MLXPDLP_HAS_ACCELERATE_SPARSE
struct CpuSparseMatrix {
    explicit CpuSparseMatrix(int rows, int columns)
        : handle(sparse_matrix_create_double(rows, columns)) {
        if (!handle) {
            throw std::runtime_error("Accelerate failed to create a sparse matrix");
        }
    }

    ~CpuSparseMatrix() {
        sparse_matrix_destroy(handle);
    }

    CpuSparseMatrix(const CpuSparseMatrix &) = delete;
    CpuSparseMatrix &operator=(const CpuSparseMatrix &) = delete;

    sparse_matrix_double handle = nullptr;
    int64_t nonzeros = 0;
};
#else
struct CpuSparseMatrix {};
#endif

} // namespace detail

namespace {

#ifdef MLXPDLP_HAS_ACCELERATE_SPARSE
class CpuSparseMatvecPrimitive final : public mx::UnaryPrimitive {
  public:
    CpuSparseMatvecPrimitive(mx::Stream stream,
                             std::shared_ptr<detail::CpuSparseMatrix> matrix,
                             CBLAS_TRANSPOSE transpose, int rows)
        : mx::UnaryPrimitive(stream), matrix_(std::move(matrix)),
          transpose_(transpose), rows_(rows) {}

    void eval_cpu(const std::vector<mx::array> &inputs, mx::array &output) override {
        if (inputs.size() != 1 || inputs[0].dtype() != mx::float64) {
            throw std::invalid_argument("sparse CPU SpMV requires one float64 vector");
        }
        output.set_data(mx::allocator::malloc(output.nbytes()));
        const double *input = inputs[0].data<double>();
        double *result = output.data<double>();
        auto matrix = matrix_;
        const auto transpose = transpose_;
        const int rows = rows_;

        auto &encoder = mx::cpu::get_command_encoder(stream());
        encoder.set_input_array(inputs[0]);
        encoder.set_output_array(output);
        encoder.dispatch([matrix = std::move(matrix), transpose, rows, input, result]() {
            std::fill_n(result, rows, 0.0);
            const sparse_status status = sparse_matrix_vector_product_dense_double(
                transpose, 1.0, matrix->handle, input, 1, result, 1);
            if (status != SPARSE_SUCCESS) {
                throw std::runtime_error("Accelerate sparse matrix-vector product failed");
            }
        });
    }

    void eval_gpu(const std::vector<mx::array> &, mx::array &) override {
        throw std::runtime_error("sparse CPU SpMV cannot run on Metal");
    }

    const char *name() const override {
        return "MlxPdlpCpuSparseMatvec";
    }

  private:
    std::shared_ptr<detail::CpuSparseMatrix> matrix_;
    CBLAS_TRANSPOSE transpose_;
    int rows_;
};
#endif

} // namespace

// ---------------------------------------------------------------------------
// Utility helpers
// ---------------------------------------------------------------------------

// Used by MlxPdlpState default constructor to initialize mx::array members.
// Defined here (not in the header) to avoid pulling MLX headers into the header.
mx::array _mlx_empty_array() {
    // State construction happens before the solver can establish its requested
    // StreamContext. Keep this placeholder itself on CPU rather than inheriting
    // MLX's process-wide default device.
    static mx::array empty = mx::zeros({0}, mx::float32, mx::Device::cpu);
    return empty;
}

static double inf() {
    return std::numeric_limits<double>::infinity();
}
static bool is_finite(double x) {
    return std::isfinite(x);
}

using SteadyClock = std::chrono::steady_clock;

static double elapsed_seconds(SteadyClock::time_point start) {
    return std::chrono::duration<double>(SteadyClock::now() - start).count();
}

// cuOpt Stable3 adds denser "conditional major" iterations while the solve is
// young, then backs off logarithmically. Starting at 10 iterations is too eager
// for MLX's lazy Metal graphs, so use the same shape shifted by one decade:
// every 100 iterations below 10,000, then every 1,000 below 100,000, and so on.
// Restart decisions retain their configured cadence so an extra Metal
// synchronization cannot alter the PDHG trajectory.
static int conditional_evaluation_interval(int total_count) {
    int64_t step = 100;
    int64_t threshold = 10000;
    while (static_cast<int64_t>(total_count) >= threshold &&
           step <= INT32_MAX / 10 && threshold <= INT32_MAX / 10) {
        step *= 10;
        threshold *= 10;
    }
    return static_cast<int>(step);
}

static int iterations_to_next_evaluation(int total_count, int regular_frequency,
                                         bool conditional) {
    const int regular = std::max(1, regular_frequency);
    int distance = regular - total_count % regular;
    if (conditional) {
        const int early = conditional_evaluation_interval(total_count);
        distance = std::min(distance, early - total_count % early);
    }
    return std::max(1, distance);
}

static mx::array mlx_array_from_doubles(const double *host_ptr, int size,
                                        mx::Dtype dtype) {
    if (dtype == mx::float64) {
        return mx::array(host_ptr, {size}, mx::float64);
    }
    std::vector<float> values(size);
    for (int i = 0; i < size; ++i) {
        values[i] = static_cast<float>(host_ptr[i]);
    }
    return mx::array(values.data(), {size}, mx::float32);
}

static double mlx_scalar_as_double(const mx::array &value) {
    if (value.dtype() == mx::float64) {
        return value.item<double>();
    }
    return static_cast<double>(value.item<float>());
}

// MLX intentionally maps an implicitly constructed C++ double scalar to
// float32. Bind host-computed coefficients to the array compute dtype so CPU
// expressions retain FP64 while Metal stays FP32.
static mx::array mlx_scalar_like(double value, const mx::array &reference) {
    return mx::array(value, reference.dtype());
}

static const char *termination_reason_str(termination_reason_t r) {
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
    case TERMINATION_REASON_FEAS_POLISH_SUCCESS:
        return "FEAS_POLISH_SUCCESS";
    case TERMINATION_REASON_HOST_DOUBLE_HANDOFF:
        return "HOST_DOUBLE_HANDOFF";
    default:
        return "UNSPECIFIED";
    }
}

// ---------------------------------------------------------------------------
// Default parameters
// ---------------------------------------------------------------------------

void mlxpdlp_set_default_parameters(pdhg_parameters_t *p) {
    p->geometric_mean_iterations = 12;
    p->curtis_reid_iterations = 0;
    p->l_inf_ruiz_iterations = 10;
    p->has_pock_chambolle_alpha = true;
    p->pock_chambolle_alpha = 1.0;
    p->bound_objective_rescaling = true;
    p->verbose = true;
    p->termination_evaluation_frequency = 200;
    p->feasibility_polishing = false;
    p->host_double_polishing = false;
    p->host_double_early_handoff = true;
    p->host_double_polishing_iteration_limit = 50000;
    p->host_double_polishing_time_sec_limit = 30.0;
    p->reflection_coefficient = 1.0;
    p->sv_max_iter = 200;
    p->sv_tol = 1e-4;
    p->termination_criteria.eps_optimal_relative = 1e-4;
    p->termination_criteria.eps_feasible_relative = 1e-4;
    p->termination_criteria.eps_infeasible_relative = 1e-14;
    p->termination_criteria.time_sec_limit = 3600.0;
    p->termination_criteria.iteration_limit = INT32_MAX;
    // Feasibility polishing is an internal certificate-improvement phase. Its
    // tighter target provides margin between scaled backend residuals and an
    // original-model float64 audit; the externally required solution accuracy
    // remains eps_feasible_relative (1e-4 by default).
    p->termination_criteria.eps_feas_polish_relative = 1e-6;
    p->restart_params.artificial_restart_threshold = 0.36;
    p->restart_params.sufficient_reduction_for_restart = 0.2;
    p->restart_params.necessary_reduction_for_restart = 0.5;
    p->restart_params.k_p = 0.99;
    p->restart_params.k_i = 0.01;
    p->restart_params.k_d = 0.0;
    p->restart_params.i_smooth = 0.3;
    p->restart_policy = 0;
    p->optimality_norm = NORM_TYPE_L2;
#ifdef MLXPDLP_HAS_PRESOLVE
    p->presolve = true;
#else
    p->presolve = false;
#endif
    // PSLP's singleton-column inverse map can amplify a modest reduced-space
    // error into an original-model primal failure (Netlib BORE3D). Keep the
    // reduction available explicitly, but prefer the safer default.
    p->presolve_singleton_columns = false;
    p->presolve_doubleton_equations = true;
    p->presolve_parallel_rows = true;
    p->presolve_parallel_columns = true;
    p->presolve_dual_fix = true;
    p->presolve_finite_bound_tightening = true;
    p->presolve_primal_propagation = false;
    p->matrix_zero_tol = 1e-9;
    p->metal_fused_kernels = true;
    p->conditional_termination_evaluation = true;
}

void mlxpdlp_result_free(mlxpdlp_result_t *result) {
    if (!result)
        return;
    delete[] result->primal_solution;
    delete[] result->dual_solution;
    delete[] result->reduced_cost;
    delete result;
}

void MlxPdlpSolver::set_default_parameters(pdhg_parameters_t *p) {
    mlxpdlp_set_default_parameters(p);
}

void MlxPdlpSolver::capture_sparse_matrix(int rows, int cols, const int *row_ptr,
                                          const int *col_ind, const double *values) {
    if (!row_ptr || rows < 0 || cols < 0) {
        throw std::invalid_argument("invalid CSR matrix dimensions or row pointers");
    }

    int nnz = row_ptr[rows];
    if (nnz < 0 || (nnz > 0 && (!col_ind || !values))) {
        throw std::invalid_argument("invalid CSR nonzero storage");
    }

    sparse_a_row_ptr_host_.resize(static_cast<size_t>(rows) + 1);
    for (int row = 0; row <= rows; ++row) {
        if (row_ptr[row] < 0 || (row > 0 && row_ptr[row] < row_ptr[row - 1]) ||
            row_ptr[row] > nnz) {
            throw std::invalid_argument("CSR row pointers must be monotone and in range");
        }
        sparse_a_row_ptr_host_[static_cast<size_t>(row)] = row_ptr[row];
    }

    sparse_a_col_ind_host_.resize(nnz);
    sparse_a_values_host_.resize(nnz);
    for (int k = 0; k < nnz; ++k) {
        if (col_ind[k] < 0 || col_ind[k] >= cols) {
            throw std::invalid_argument("CSR column index is out of range");
        }
        sparse_a_col_ind_host_[static_cast<size_t>(k)] = col_ind[k];
        sparse_a_values_host_[static_cast<size_t>(k)] = values[k];
    }

    // Build A^T in CSR order. Its values are populated after preconditioning
    // from sparse_at_source_index_, so duplicate coordinates remain additive
    // exactly as they are in the input CSR matrix.
    sparse_at_row_ptr_host_.assign(static_cast<size_t>(cols) + 1, 0);
    for (int32_t col : sparse_a_col_ind_host_) {
        ++sparse_at_row_ptr_host_[static_cast<size_t>(col) + 1];
    }
    for (int row = 0; row < cols; ++row) {
        sparse_at_row_ptr_host_[static_cast<size_t>(row) + 1] +=
            sparse_at_row_ptr_host_[static_cast<size_t>(row)];
    }

    sparse_at_col_ind_host_.resize(nnz);
    sparse_at_source_index_.resize(nnz);
    auto next = sparse_at_row_ptr_host_;
    for (int row = 0; row < rows; ++row) {
        for (int32_t k = sparse_a_row_ptr_host_[static_cast<size_t>(row)];
             k < sparse_a_row_ptr_host_[static_cast<size_t>(row) + 1]; ++k) {
            int col = sparse_a_col_ind_host_[static_cast<size_t>(k)];
            int32_t target = next[static_cast<size_t>(col)]++;
            sparse_at_col_ind_host_[static_cast<size_t>(target)] = row;
            sparse_at_source_index_[static_cast<size_t>(target)] = k;
        }
    }
}

// ---------------------------------------------------------------------------
// Constructor
// ---------------------------------------------------------------------------

MlxPdlpSolver::MlxPdlpSolver(int num_vars, int num_cons, const int *csr_row_ptr,
                             const int *csr_col_ind, const double *csr_vals, const double *var_lb,
                             const double *var_ub, const double *con_lb, const double *con_ub,
                             const double *objective, double objective_constant,
                             const pdhg_parameters_t *params, mx::Device device)
    : MlxPdlpSolver(num_vars, num_cons, csr_row_ptr, csr_col_ind, csr_vals, var_lb, var_ub, con_lb,
                    con_ub, objective, objective_constant, params, nullptr, nullptr, device) {}

MlxPdlpSolver::MlxPdlpSolver(int num_vars, int num_cons, const int *csr_row_ptr,
                             const int *csr_col_ind, const double *csr_vals, const double *var_lb,
                             const double *var_ub, const double *con_lb, const double *con_ub,
                             const double *objective, double objective_constant,
                             const pdhg_parameters_t *params, const double *primal_start,
                             const double *dual_start, mx::Device device)
    : MlxPdlpSolver(num_vars, num_cons, csr_row_ptr, csr_col_ind, csr_vals, var_lb,
                    var_ub, con_lb, con_ub, objective, objective_constant, params,
                    primal_start, dual_start, nullptr, device) {}

MlxPdlpSolver::MlxPdlpSolver(int num_vars, int num_cons, const int *csr_row_ptr,
                             const int *csr_col_ind, const double *csr_vals,
                             const double *var_lb, const double *var_ub,
                             const double *con_lb, const double *con_ub,
                             const double *objective, double objective_constant,
                             const pdhg_parameters_t *params,
                             const double *primal_start, const double *dual_start,
                             const double *reduced_cost_start, mx::Device device) {
    // Operations without an explicit StreamOrDevice use MLX's current default.
    // Keep that default scoped to this constructor so arrays and preprocessing
    // are placed on the device requested by the caller.
    s_.stream = mx::default_stream(device);
    s_.cpu_double_precision_active = device.type == mx::Device::cpu;
    mx::StreamContext stream_context(s_.stream);

    // Copy parameters
    if (params)
        params_ = *params;
    else
        set_default_parameters(&params_);

    if (!std::isfinite(params_.termination_criteria.eps_infeasible_relative) ||
        params_.termination_criteria.eps_infeasible_relative <= 0.0) {
        throw std::invalid_argument("eps_infeasible_relative must be finite and positive");
    }

    if (num_vars < 0 || num_cons < 0 || !csr_row_ptr) {
        throw std::invalid_argument("invalid LP dimensions or CSR row pointers");
    }

    original_num_variables_ = num_vars;
    original_num_constraints_ = num_cons;
    original_num_nonzeros_ = csr_row_ptr[num_cons];
    original_objective_constant_ = objective_constant;
    if (original_num_nonzeros_ < 0 ||
        (original_num_nonzeros_ > 0 && (!csr_col_ind || !csr_vals))) {
        throw std::invalid_argument("invalid CSR column indices or values");
    }

    auto copy_or_fill = [](const double *values, int size, double fill_value) {
        std::vector<double> result(static_cast<size_t>(size), fill_value);
        if (values)
            std::copy(values, values + size, result.begin());
        return result;
    };
    original_objective_ = copy_or_fill(objective, num_vars, 0.0);
    original_variable_lower_bound_ = copy_or_fill(var_lb, num_vars, -inf());
    original_variable_upper_bound_ = copy_or_fill(var_ub, num_vars, inf());
    original_constraint_lower_bound_ = copy_or_fill(con_lb, num_cons, -inf());
    original_constraint_upper_bound_ = copy_or_fill(con_ub, num_cons, inf());
    original_row_ptr_.assign(csr_row_ptr, csr_row_ptr + num_cons + 1);
    if (original_num_nonzeros_ > 0) {
        original_col_ind_.assign(csr_col_ind, csr_col_ind + original_num_nonzeros_);
        original_matrix_values_.assign(csr_vals, csr_vals + original_num_nonzeros_);
    }

    int working_num_vars = num_vars;
    int working_num_cons = num_cons;
    int working_num_nonzeros = original_num_nonzeros_;
    const int *working_row_ptr = csr_row_ptr;
    const int *working_col_ind = csr_col_ind;
    const double *working_values = csr_vals;
    const double *working_var_lb = var_lb;
    const double *working_var_ub = var_ub;
    const double *working_con_lb = con_lb;
    const double *working_con_ub = con_ub;
    const double *working_objective = objective;
    double working_objective_constant = objective_constant;

#ifdef MLXPDLP_HAS_PRESOLVE
    std::unique_ptr<detail::PresolveContext, decltype(&detail::destroy_presolve)> presolve_guard(
        nullptr, detail::destroy_presolve);
    if (params_.presolve) {
        if (primal_start || dual_start || reduced_cost_start) {
            throw std::invalid_argument(
                "warm starts are not supported when PSLP presolve is enabled");
        }

        auto original_con_lb = copy_or_fill(con_lb, num_cons, -inf());
        auto original_con_ub = copy_or_fill(con_ub, num_cons, inf());
        detail::HostProblemView original_problem{
            num_vars,
            num_cons,
            original_num_nonzeros_,
            csr_row_ptr,
            csr_col_ind,
            csr_vals,
            original_variable_lower_bound_.data(),
            original_variable_upper_bound_.data(),
            original_con_lb.data(),
            original_con_ub.data(),
            original_objective_.data(),
            objective_constant,
        };
        const detail::PresolveOptions presolve_options{
            params_.presolve_singleton_columns,
            params_.presolve_doubleton_equations,
            params_.presolve_parallel_rows,
            params_.presolve_parallel_columns,
            params_.presolve_dual_fix,
            params_.presolve_finite_bound_tightening,
            params_.presolve_primal_propagation,
        };
        auto outcome = detail::run_presolve(original_problem, params_.matrix_zero_tol,
                                            presolve_options, params_.verbose);
        presolve_guard.reset(outcome.context);
        presolve_status_ = outcome.status;
        presolve_time_sec_ = outcome.elapsed_seconds;
        presolve_solved_ = outcome.solved;

        working_num_vars = outcome.reduced_problem.num_variables;
        working_num_cons = outcome.reduced_problem.num_constraints;
        working_num_nonzeros = outcome.reduced_problem.num_nonzeros;
        working_row_ptr = outcome.reduced_problem.row_ptr;
        working_col_ind = outcome.reduced_problem.col_ind;
        working_values = outcome.reduced_problem.values;
        working_var_lb = outcome.reduced_problem.variable_lower_bound;
        working_var_ub = outcome.reduced_problem.variable_upper_bound;
        working_con_lb = outcome.reduced_problem.constraint_lower_bound;
        working_con_ub = outcome.reduced_problem.constraint_upper_bound;
        working_objective = outcome.reduced_problem.objective;
        working_objective_constant = outcome.reduced_problem.objective_constant;

        if (presolve_solved_) {
            s_.n = working_num_vars;
            s_.m = working_num_cons;
            s_.nnz = working_num_nonzeros;
            presolve_context_ = presolve_guard.release();
            return;
        }
    }
#else
    if (params_.presolve) {
        throw std::runtime_error(
            "presolve requested, but mlxPDLP was built with MLXPDLP_BUILD_PRESOLVE=OFF");
    }
#endif

    // Preserve the unscaled reduced model before sparse preprocessing mutates
    // its own host copy. Non-presolved solves use the already-owned original
    // CSR directly and avoid duplicating potentially very large inputs.
    if (params_.presolve) {
        working_num_variables_ = working_num_vars;
        working_num_constraints_ = working_num_cons;
        working_num_nonzeros_ = working_num_nonzeros;
        working_objective_constant_ = working_objective_constant;
        working_objective_ =
            copy_or_fill(working_objective, working_num_vars, 0.0);
        working_variable_lower_bound_ =
            copy_or_fill(working_var_lb, working_num_vars, -inf());
        working_variable_upper_bound_ =
            copy_or_fill(working_var_ub, working_num_vars, inf());
        working_constraint_lower_bound_ =
            copy_or_fill(working_con_lb, working_num_cons, -inf());
        working_constraint_upper_bound_ =
            copy_or_fill(working_con_ub, working_num_cons, inf());
        working_row_ptr_.assign(working_row_ptr,
                                working_row_ptr + working_num_cons + 1);
        if (working_num_nonzeros > 0) {
            working_col_ind_.assign(working_col_ind,
                                    working_col_ind + working_num_nonzeros);
            working_matrix_values_.assign(working_values,
                                          working_values + working_num_nonzeros);
        }
    }

    // Dimensions
    s_.m = working_num_cons;
    s_.n = working_num_vars;

    // Count nonzeros
    s_.nnz = working_num_nonzeros;

    capture_sparse_matrix(working_num_cons, working_num_vars, working_row_ptr, working_col_ind,
                          working_values);

    const int64_t dense_elements = static_cast<int64_t>(s_.m) * s_.n;
    constexpr int64_t min_sparse_metal_dense_elements = 4096;
    // Keep the well-tested dense CPU path for ordinary models. Above this
    // threshold FP64 A + A^T alone would consume at least 256 MiB, and sparse SpMV
    // becomes essential rather than merely an optimization.
    constexpr int64_t min_sparse_cpu_dense_elements = 16 * 1024 * 1024;
    constexpr double max_sparse_density = 0.25;
    sparse_metal_candidate_ = device.type == mx::Device::gpu && s_.nnz > 0 &&
                              dense_elements >= min_sparse_metal_dense_elements &&
                              static_cast<double>(s_.nnz) <= max_sparse_density * dense_elements;
#ifdef MLXPDLP_HAS_ACCELERATE_SPARSE
    sparse_cpu_candidate_ = device.type == mx::Device::cpu && s_.nnz > 0 &&
                            dense_elements >= min_sparse_cpu_dense_elements &&
                            static_cast<double>(s_.nnz) <=
                                max_sparse_density * dense_elements;
#else
    (void)min_sparse_cpu_dense_elements;
#endif

    sparse_con_rescale_host_.assign(static_cast<size_t>(s_.m), 1.0);
    sparse_var_rescale_host_.assign(static_cast<size_t>(s_.n), 1.0);

    // Sparse candidates remain in CSR throughout preprocessing and iteration.
    // Dense storage is retained only by small/dense CPU and Metal fallbacks.
    if (!sparse_metal_candidate_ && !sparse_cpu_candidate_) {
        s_.A = csr_to_dense(working_num_cons, working_num_vars, working_row_ptr, working_col_ind,
                            working_values);
        s_.AT = mx::transpose(s_.A);
        mx::eval(s_.A, s_.AT);
    }

    // Build bound arrays (host → mx::array)
    const mx::Dtype compute_dtype =
        s_.cpu_double_precision_active ? mx::float64 : mx::float32;
    auto make_arr = [compute_dtype](const double *host_ptr, int size) -> mx::array {
        if (host_ptr) {
            return mlx_array_from_doubles(host_ptr, size, compute_dtype);
        }
        return mx::full({size}, -inf(), compute_dtype);
    };
    auto make_arr_ub = [compute_dtype](const double *host_ptr, int size) -> mx::array {
        if (host_ptr) {
            return mlx_array_from_doubles(host_ptr, size, compute_dtype);
        }
        return mx::full({size}, inf(), compute_dtype);
    };

    s_.var_lb = make_arr(working_var_lb, working_num_vars);
    s_.var_ub = make_arr_ub(working_var_ub, working_num_vars);
    s_.con_lb = make_arr(working_con_lb, working_num_cons);
    s_.con_ub = make_arr_ub(working_con_ub, working_num_cons);
    s_.obj = working_objective
                 ? mlx_array_from_doubles(working_objective, working_num_vars,
                                          compute_dtype)
                 : mx::zeros({working_num_vars}, compute_dtype);
    s_.objective_constant = working_objective_constant;

    // Preprocess bounds (finite-safe + masks)
    preprocess_bounds(working_var_lb, working_var_ub, working_con_lb, working_con_ub);

    // Initialize rescaling to identity
    s_.var_rescale = mx::ones({working_num_vars}, compute_dtype);
    s_.con_rescale = mx::ones({working_num_cons}, compute_dtype);
    s_.con_bound_rescale = 1.0;
    s_.obj_vec_rescale = 1.0;

    auto validate_start = [](const double *start, int size, const char *name) {
        if (!start)
            return;
        for (int i = 0; i < size; ++i) {
            if (!std::isfinite(start[i])) {
                throw std::invalid_argument(std::string(name) +
                                            " warm start must contain only finite values");
            }
        }
    };
    validate_start(primal_start, working_num_vars, "primal");
    validate_start(dual_start, working_num_cons, "dual");
    validate_start(reduced_cost_start, working_num_vars, "reduced-cost");
    has_reduced_cost_start_ = reduced_cost_start != nullptr;
    if (has_reduced_cost_start_) {
        warm_reduced_cost_.assign(reduced_cost_start,
                                  reduced_cost_start + working_num_vars);
    }
    has_warm_start_ = primal_start != nullptr || dual_start != nullptr ||
                      has_reduced_cost_start_;

    // Initialize solution vectors. Warm starts are provided in original problem
    // coordinates and are transformed alongside the problem during scaling.
    s_.x_cur = primal_start
                   ? mlx_array_from_doubles(primal_start, working_num_vars,
                                            compute_dtype)
                   : mx::zeros({working_num_vars}, compute_dtype);
    s_.x_pdhg = mx::zeros({working_num_vars}, compute_dtype);
    s_.x_ref = mx::zeros({working_num_vars}, compute_dtype);
    s_.x_init = mx::zeros({working_num_vars}, compute_dtype);
    s_.x_best = mx::zeros({working_num_vars}, compute_dtype);
    s_.ATy = mx::zeros({working_num_vars}, compute_dtype);
    s_.dual_slack = mx::zeros({working_num_vars}, compute_dtype);
    s_.dual_slack_best = mx::zeros({working_num_vars}, compute_dtype);
    s_.dual_res = mx::zeros({working_num_vars}, compute_dtype);
    s_.delta_x = mx::zeros({working_num_vars}, compute_dtype);

    s_.y_cur = dual_start
                   ? mlx_array_from_doubles(dual_start, working_num_cons,
                                            compute_dtype)
                   : mx::zeros({working_num_cons}, compute_dtype);
    s_.y_pdhg = mx::zeros({working_num_cons}, compute_dtype);
    s_.y_ref = mx::zeros({working_num_cons}, compute_dtype);
    s_.y_init = mx::zeros({working_num_cons}, compute_dtype);
    s_.y_best = mx::zeros({working_num_cons}, compute_dtype);
    s_.Ax = mx::zeros({working_num_cons}, compute_dtype);
    s_.primal_res = mx::zeros({working_num_cons}, compute_dtype);
    s_.delta_y = mx::zeros({working_num_cons}, compute_dtype);

    // Iteration state
    s_.step_size = 1.0;
    s_.primal_weight = 1.0;
    s_.step_size_primal = 1.0;
    s_.step_size_dual = 1.0;
    s_.inner_count = 0;
    s_.total_count = 0;
    s_.fixed_point_error = inf();
    s_.initial_fixed_point_error = inf();
    s_.last_trial_fixed_point_error = inf();
    s_.best_primal_dual_residual_gap = inf();
    s_.best_relative_kkt_error = inf();
    s_.best_relative_feasibility_error = inf();
    s_.best_iteration = -1;
    s_.primal_weight_error_sum = 0.0;
    s_.primal_weight_last_error = 0.0;
    s_.best_primal_weight = 1.0;
    s_.termination_reason = TERMINATION_REASON_UNSPECIFIED;

    // Residuals
    s_.absolute_primal_residual = inf();
    s_.relative_primal_residual = inf();
    s_.absolute_dual_residual = inf();
    s_.relative_dual_residual = inf();
    s_.primal_objective_value = 0.0;
    s_.dual_objective_value = 0.0;
    s_.objective_gap = inf();
    s_.relative_objective_gap = inf();

    s_.max_primal_ray_infeasibility = 0.0;
    s_.max_dual_ray_infeasibility = 0.0;
    s_.primal_ray_linear_objective = 0.0;
    s_.dual_ray_objective = 0.0;

    // Norms
    s_.objective_vector_norm = 0.0;
    s_.constraint_bound_norm = 0.0;

    // Timing
    s_.start_time = SteadyClock::now();
    s_.cumulative_time_sec = 0.0;
    s_.rescaling_time_sec = 0.0;

#ifdef MLXPDLP_HAS_PRESOLVE
    presolve_context_ = presolve_guard.release();
#endif
}

MlxPdlpSolver::~MlxPdlpSolver() {
#ifdef MLXPDLP_HAS_PRESOLVE
    detail::destroy_presolve(presolve_context_);
#endif
    // mx::array destructors handle the remaining cleanup automatically.
}

// ---------------------------------------------------------------------------
// CSR → dense conversion (host-side, then imported into an MLX array)
// ---------------------------------------------------------------------------

mx::array MlxPdlpSolver::csr_to_dense(int rows, int cols, const int *row_ptr, const int *col_ind,
                                      const double *vals) {
    if (s_.cpu_double_precision_active) {
        std::vector<double> dense(static_cast<size_t>(rows) * cols, 0.0);
        for (int i = 0; i < rows; ++i) {
            for (int k = row_ptr[i]; k < row_ptr[i + 1]; ++k) {
                const int j = col_ind[k];
                dense[static_cast<size_t>(i) * cols + j] += vals[k];
            }
        }
        return mx::array(dense.data(), {rows, cols}, mx::float64);
    }
    std::vector<float> dense(static_cast<size_t>(rows) * cols, 0.0f);
    for (int i = 0; i < rows; ++i) {
        for (int k = row_ptr[i]; k < row_ptr[i + 1]; ++k) {
            int j = col_ind[k];
            // Duplicate CSR coordinates are additive. The bundled MPS loader
            // can preserve repeated entries, so assigning here would silently
            // discard every coefficient except the last one.
            dense[static_cast<size_t>(i) * cols + j] += static_cast<float>(vals[k]);
        }
    }
    return mx::array(dense.data(), {rows, cols}, mx::float32);
}

// ---------------------------------------------------------------------------
// Bound preprocessing
// ---------------------------------------------------------------------------

void MlxPdlpSolver::preprocess_bounds(const double *host_var_lb, const double *host_var_ub,
                                      const double *host_con_lb, const double *host_con_ub) {
    int m = s_.m, n = s_.n;
    const mx::Dtype compute_dtype =
        s_.cpu_double_precision_active ? mx::float64 : mx::float32;

    // Only the infinite-bound masks are cached. Finite-safe bound values are
    // computed fresh where needed (dual objective and infeasibility rays)
    // because the cached copies would go stale after Pock-Chambolle and
    // bound/objective scaling.
    auto make_inf_mask = [compute_dtype](const double *host, int size,
                                         double inf_val) -> mx::array {
        std::vector<double> mask(size);
        for (int i = 0; i < size; ++i) {
            double v = host ? host[i] : inf_val;
            mask[i] = (v == inf_val || v == -inf_val) ? 1.0 : 0.0;
        }
        return mlx_array_from_doubles(mask.data(), size, compute_dtype);
    };

    s_.var_lb_inf_mask = make_inf_mask(host_var_lb, n, -inf());
    s_.var_ub_inf_mask = make_inf_mask(host_var_ub, n, inf());
    s_.con_lb_inf_mask = make_inf_mask(host_con_lb, m, -inf());
    s_.con_ub_inf_mask = make_inf_mask(host_con_ub, m, inf());
}

// ---------------------------------------------------------------------------
// Core linear algebra
// ---------------------------------------------------------------------------

void MlxPdlpSolver::prepare_sparse_metal_backend() {
    if (s_.sparse_metal_active || !sparse_metal_candidate_) {
        return;
    }

    // The sparse values are already fully preconditioned. Compute the
    // Frobenius norm of the effective matrix for the power method's
    // degenerate-start fallback. Aggregate duplicate coordinates without
    // constructing a dense matrix.
    double squared_frobenius_norm = 0.0;
    std::vector<double> row_col_sums(static_cast<size_t>(s_.n), 0.0);
    std::vector<int> last_seen_row(static_cast<size_t>(s_.n), -1);
    std::vector<int> touched_columns;
    for (int row = 0; row < s_.m; ++row) {
        touched_columns.clear();
        for (int32_t k = sparse_a_row_ptr_host_[static_cast<size_t>(row)];
             k < sparse_a_row_ptr_host_[static_cast<size_t>(row) + 1]; ++k) {
            int col = sparse_a_col_ind_host_[static_cast<size_t>(k)];
            if (last_seen_row[static_cast<size_t>(col)] != row) {
                last_seen_row[static_cast<size_t>(col)] = row;
                row_col_sums[static_cast<size_t>(col)] = 0.0;
                touched_columns.push_back(col);
            }
            row_col_sums[static_cast<size_t>(col)] += sparse_a_values_host_[static_cast<size_t>(k)];
        }
        for (int col : touched_columns) {
            double value = row_col_sums[static_cast<size_t>(col)];
            if (std::isfinite(value)) {
                squared_frobenius_norm += value * value;
            }
        }
    }
    sparse_frobenius_norm_ = std::sqrt(squared_frobenius_norm);

    const int sparse_nnz = static_cast<int>(sparse_a_values_host_.size());
    std::vector<float> matrix_values(static_cast<size_t>(sparse_nnz));
    std::vector<float> transpose_values(static_cast<size_t>(sparse_nnz));
    for (int k = 0; k < sparse_nnz; ++k) {
        matrix_values[static_cast<size_t>(k)] =
            static_cast<float>(sparse_a_values_host_[static_cast<size_t>(k)]);
        transpose_values[static_cast<size_t>(k)] = static_cast<float>(
            sparse_a_values_host_[static_cast<size_t>(sparse_at_source_index_[static_cast<size_t>(k)])]);
    }

    sparse_a_row_ptr_ = mx::array(sparse_a_row_ptr_host_.data(), {s_.m + 1}, mx::int32);
    sparse_a_col_ind_ = mx::array(sparse_a_col_ind_host_.data(), {sparse_nnz}, mx::int32);
    sparse_a_values_ = mx::array(matrix_values.data(), {sparse_nnz}, mx::float32);

    sparse_at_row_ptr_ = mx::array(sparse_at_row_ptr_host_.data(), {s_.n + 1}, mx::int32);
    sparse_at_col_ind_ = mx::array(sparse_at_col_ind_host_.data(), {sparse_nnz}, mx::int32);
    sparse_at_values_ = mx::array(transpose_values.data(), {sparse_nnz}, mx::float32);

    struct AdaptiveWork {
        std::vector<int32_t> offsets{0};
        std::vector<int32_t> rows;
        int item_count = 0;
    };

    struct RowProfile {
        int rows = 0;
        int nonzeros = 0;
        int max_row_nonzeros = 0;
        int tiny_rows = 0;
        int rows_over_adaptive_cutoff = 0;
    };

    auto profile_rows = [](const std::vector<int32_t> &row_ptr, int row_count) {
        constexpr int scalar_row_max_nonzeros = 16;
        constexpr int adaptive_short_row_max_nonzeros = 64;
        RowProfile profile;
        profile.rows = row_count;
        profile.nonzeros = row_ptr.empty() ? 0 : row_ptr.back();
        for (int row = 0; row < row_count; ++row) {
            const int length = row_ptr[static_cast<size_t>(row) + 1] -
                               row_ptr[static_cast<size_t>(row)];
            profile.max_row_nonzeros = std::max(profile.max_row_nonzeros, length);
            if (length <= scalar_row_max_nonzeros) {
                ++profile.tiny_rows;
            }
            if (length > adaptive_short_row_max_nonzeros) {
                ++profile.rows_over_adaptive_cutoff;
            }
        }
        return profile;
    };

    auto select_strategy = [](const RowProfile &profile) {
        constexpr int scalar_row_max_nonzeros = 16;
        // At 64 entries the scalar-row mapping is still competitive on small
        // matrices. Once there is enough aggregate work, a SIMD-group per row
        // wins by coalescing the CSR stream. Rows just above 64 need that path
        // even sooner: the adaptive fallback otherwise assigns all 256 threads
        // and eight barriers to only 65 products.
        //
        // The aggregate-work cutoff is device-tuned: the validated 8M-nonzero
        // default was measured on an M3 Max, and wider/newer GPUs reach the
        // SIMD-group crossover at smaller aggregate work while narrower ones
        // stay on the adaptive path longer. MLXPDLP_SPMV_SIMD_NNZ_THRESHOLD
        // overrides the derived value explicitly.
        auto simdgroup_nnz_threshold = []() -> int64_t {
            const char *override_value = std::getenv("MLXPDLP_SPMV_SIMD_NNZ_THRESHOLD");
            if (override_value && *override_value) {
                const double parsed = std::atof(override_value);
                if (parsed > 0.0) {
                    return static_cast<int64_t>(parsed);
                }
            }
            // Parse the Apple Silicon family generation from the device name
            // ("Apple M1" through "Apple M5 Max"). The validated 8M-nonzero
            // default corresponds to the M3; each newer generation reaches the
            // SIMD-group crossover at smaller aggregate work. Unrecognized
            // names keep the M3-tuned default.
            double family_factor = 1.0;
            try {
                const auto &info = mx::gpu::device_info(0);
                auto name_it = info.find("device_name");
                if (name_it != info.end()) {
                    const std::string &name = std::get<std::string>(name_it->second);
                    const size_t marker = name.find('M');
                    if (marker != std::string::npos && marker + 1 < name.size() &&
                        std::isdigit(static_cast<unsigned char>(name[marker + 1]))) {
                        switch (name[marker + 1] - '0') {
                        case 1:
                            family_factor = 1.5;
                            break;
                        case 2:
                            family_factor = 1.25;
                            break;
                        case 3:
                            family_factor = 1.0;
                            break;
                        case 4:
                            family_factor = 0.75;
                            break;
                        default:
                            // M5 and newer.
                            family_factor = 0.5;
                            break;
                        }
                    }
                }
            } catch (...) {
                // Keep the validated default when device introspection fails.
            }
            return static_cast<int64_t>(8.0 * 1024.0 * 1024.0 * family_factor);
        }();
        if (profile.max_row_nonzeros <= scalar_row_max_nonzeros) {
            return SparseMetalSpmvStrategy::scalar_rows;
        }

        const int64_t rows = profile.rows;
        const int64_t nonzeros = profile.nonzeros;
        const bool medium_workload =
            nonzeros >= 32 * rows && nonzeros <= 512 * rows &&
            static_cast<int64_t>(profile.tiny_rows) * 2 < rows;
        const bool many_rows_cross_adaptive_cutoff =
            static_cast<int64_t>(profile.rows_over_adaptive_cutoff) * 4 >=
            static_cast<int64_t>(profile.rows);
        const bool large_uniform_short_matrix = nonzeros >= simdgroup_nnz_threshold;
        if (medium_workload &&
            (many_rows_cross_adaptive_cutoff || large_uniform_short_matrix)) {
            return SparseMetalSpmvStrategy::simdgroup_rows;
        }
        return SparseMetalSpmvStrategy::adaptive;
    };

    auto build_adaptive_work = [](const std::vector<int32_t> &row_ptr, int row_count) {
        constexpr int short_row_max_nonzeros = 64;
        constexpr int medium_row_max_nonzeros = 4096;
        constexpr int medium_row_marker = 1 << 30;
        constexpr size_t simdgroup_width = 32;
        constexpr size_t threadgroup_width = 256;
        constexpr size_t simdgroups_per_threadgroup =
            threadgroup_width / simdgroup_width;
        std::vector<int32_t> short_rows;
        std::vector<int32_t> medium_rows;
        std::vector<int32_t> long_rows;
        short_rows.reserve(static_cast<size_t>(row_count));
        for (int row = 0; row < row_count; ++row) {
            const int length = row_ptr[static_cast<size_t>(row) + 1] -
                               row_ptr[static_cast<size_t>(row)];
            if (length <= short_row_max_nonzeros) {
                short_rows.push_back(row);
            } else if (length <= medium_row_max_nonzeros) {
                medium_rows.push_back(row);
            } else {
                long_rows.push_back(row);
            }
        }

        AdaptiveWork work;
        work.rows.reserve(short_rows.size() + medium_rows.size() + long_rows.size());
        work.offsets.reserve((short_rows.size() + threadgroup_width - 1) /
                                 threadgroup_width +
                             (medium_rows.size() + simdgroups_per_threadgroup - 1) /
                                 simdgroups_per_threadgroup +
                             long_rows.size() + 1);
        for (size_t begin = 0; begin < short_rows.size(); begin += threadgroup_width) {
            const size_t end =
                std::min(begin + threadgroup_width, short_rows.size());
            work.rows.insert(work.rows.end(), short_rows.begin() + begin,
                             short_rows.begin() + end);
            work.offsets.push_back(static_cast<int32_t>(work.rows.size()));
        }
        for (size_t begin = 0; begin < medium_rows.size();
             begin += simdgroups_per_threadgroup) {
            const size_t end =
                std::min(begin + simdgroups_per_threadgroup, medium_rows.size());
            for (size_t index = begin; index < end; ++index) {
                // A second negative range marks packs where each SIMD group
                // cooperatively reduces one medium row. This keeps -row-1
                // available for the full-threadgroup long-row reducer.
                work.rows.push_back(-medium_row_marker - medium_rows[index] - 1);
            }
            work.offsets.push_back(static_cast<int32_t>(work.rows.size()));
        }
        for (int32_t row : long_rows) {
            // Negative descriptors distinguish a cooperatively reduced long
            // row from a pack of independently accumulated short rows.
            work.rows.push_back(-row - 1);
            work.offsets.push_back(static_cast<int32_t>(work.rows.size()));
        }
        work.item_count = static_cast<int>(work.offsets.size()) - 1;
        return work;
    };

    s_.sparse_a_spmv_strategy =
        select_strategy(profile_rows(sparse_a_row_ptr_host_, s_.m));
    s_.sparse_at_spmv_strategy =
        select_strategy(profile_rows(sparse_at_row_ptr_host_, s_.n));

    auto prepare_adaptive_work = [&](const std::vector<int32_t> &row_ptr, int row_count,
                                     SparseMetalSpmvStrategy strategy,
                                     mx::array &work_offsets, mx::array &work_rows,
                                     int &work_item_count) {
        if (strategy != SparseMetalSpmvStrategy::adaptive) {
            return;
        }
        const AdaptiveWork work = build_adaptive_work(row_ptr, row_count);
        work_offsets = mx::array(work.offsets.data(),
                                 {static_cast<int>(work.offsets.size())}, mx::int32);
        work_rows = mx::array(work.rows.data(),
                              {static_cast<int>(work.rows.size())}, mx::int32);
        work_item_count = work.item_count;
    };
    prepare_adaptive_work(sparse_a_row_ptr_host_, s_.m, s_.sparse_a_spmv_strategy,
                          sparse_a_work_offsets_, sparse_a_work_rows_,
                          sparse_a_work_item_count_);
    prepare_adaptive_work(sparse_at_row_ptr_host_, s_.n, s_.sparse_at_spmv_strategy,
                          sparse_at_work_offsets_, sparse_at_work_rows_,
                          sparse_at_work_item_count_);

    s_.sparse_metal_active = true;
    mx::synchronize(s_.stream);

    if (params_.verbose) {
        auto strategy_name = [](SparseMetalSpmvStrategy strategy) {
            switch (strategy) {
            case SparseMetalSpmvStrategy::scalar_rows:
                return "scalar-row";
            case SparseMetalSpmvStrategy::simdgroup_rows:
                return "SIMD-group-row";
            case SparseMetalSpmvStrategy::adaptive:
                return "adaptive";
            }
            return "unknown";
        };
        double sparse_mib = (2.0 * sparse_nnz * (sizeof(float) + sizeof(int32_t)) +
                             (static_cast<double>(s_.m) + s_.n + 2.0) * sizeof(int32_t)) /
                            (1024.0 * 1024.0);
        printf("  sparse Metal SpMV enabled (CSR + transpose CSR: %.2f MiB; "
               "strategies A=%s, A^T=%s; adaptive work items A=%d, A^T=%d; "
               "fused iteration batch=%d)\n",
               sparse_mib, strategy_name(s_.sparse_a_spmv_strategy),
               strategy_name(s_.sparse_at_spmv_strategy), sparse_a_work_item_count_,
               sparse_at_work_item_count_, fused_eval_batch_size());
    }
}

void MlxPdlpSolver::prepare_sparse_cpu_backend() {
#ifdef MLXPDLP_HAS_ACCELERATE_SPARSE
    if (s_.sparse_cpu_active || !sparse_cpu_candidate_) {
        return;
    }

    auto matrix = std::make_shared<detail::CpuSparseMatrix>(s_.m, s_.n);
    std::vector<std::pair<int32_t, double>> entries;
    std::vector<double> row_values;
    std::vector<sparse_index> row_columns;
    double squared_frobenius_norm = 0.0;

    for (int row = 0; row < s_.m; ++row) {
        entries.clear();
        for (int32_t k = sparse_a_row_ptr_host_[static_cast<size_t>(row)];
             k < sparse_a_row_ptr_host_[static_cast<size_t>(row) + 1]; ++k) {
            const double value = sparse_a_values_host_[static_cast<size_t>(k)];
            if (!std::isfinite(value)) {
                throw std::runtime_error("non-finite coefficient in sparse CPU matrix");
            }
            entries.emplace_back(sparse_a_col_ind_host_[static_cast<size_t>(k)], value);
        }
        std::sort(entries.begin(), entries.end(),
                  [](const auto &lhs, const auto &rhs) { return lhs.first < rhs.first; });

        row_values.clear();
        row_columns.clear();
        for (size_t begin = 0; begin < entries.size();) {
            size_t end = begin + 1;
            double value = entries[begin].second;
            while (end < entries.size() && entries[end].first == entries[begin].first) {
                value += entries[end].second;
                ++end;
            }
            if (value != 0.0) {
                row_columns.push_back(static_cast<sparse_index>(entries[begin].first));
                row_values.push_back(value);
                squared_frobenius_norm += value * value;
            }
            begin = end;
        }

        if (!row_values.empty()) {
            const sparse_status status = sparse_insert_row_double(
                matrix->handle, static_cast<sparse_index>(row),
                static_cast<sparse_dimension>(row_values.size()), row_values.data(),
                row_columns.data());
            if (status != SPARSE_SUCCESS) {
                throw std::runtime_error("Accelerate failed to insert a sparse matrix row");
            }
            matrix->nonzeros += static_cast<int64_t>(row_values.size());
        }
    }
    if (sparse_commit(matrix->handle) != SPARSE_SUCCESS) {
        throw std::runtime_error("Accelerate failed to commit the sparse matrix");
    }

    sparse_frobenius_norm_ = std::sqrt(squared_frobenius_norm);
    sparse_cpu_matrix_ = std::move(matrix);
    s_.sparse_cpu_active = true;

    if (params_.verbose) {
        const double sparse_mib =
            (sparse_cpu_matrix_->nonzeros * (sizeof(double) + sizeof(sparse_index)) +
             (static_cast<int64_t>(s_.m) + 1) * sizeof(sparse_index)) /
            (1024.0 * 1024.0);
        printf("  sparse CPU Accelerate FP64 SpMV enabled (approximately %.2f MiB)\n",
               sparse_mib);
    }
#else
    if (sparse_cpu_candidate_) {
        throw std::runtime_error("sparse CPU backend requires Apple Accelerate");
    }
#endif
}

mx::array MlxPdlpSolver::sparse_matvec(const mx::array &row_ptr, const mx::array &col_ind,
                                       const mx::array &values,
                                       const mx::array &work_offsets,
                                       const mx::array &work_rows, const mx::array &x,
                                       int rows, int work_item_count,
                                       SparseMetalSpmvStrategy strategy) {
    constexpr int threadgroup_width = 256;

    if (strategy == SparseMetalSpmvStrategy::scalar_rows) {
        static const auto scalar_kernel = mx::fast::metal_kernel(
            "mlxpdlp_csr_spmv_scalar_rows",
            {"row_starts", "column_indices", "nonzeros", "vector"},
            {"output"},
            R"(
                uint row = thread_position_in_grid.x;
                float total = 0.0f;
                int begin = row_starts[row];
                int end = row_starts[row + 1];
                for (int k = begin; k < end; ++k) {
                    total = fma(nonzeros[k], vector[column_indices[k]], total);
                }
                output[row] = total;
            )");
        return scalar_kernel({row_ptr, col_ind, values, x}, {{rows}}, {mx::float32},
                             {rows, 1, 1}, {threadgroup_width, 1, 1}, {},
                             std::nullopt, false, s_.stream)[0];
    }

    if (strategy == SparseMetalSpmvStrategy::simdgroup_rows) {
        static const auto simdgroup_kernel = mx::fast::metal_kernel(
            "mlxpdlp_csr_spmv_simdgroup_rows",
            {"row_starts", "column_indices", "nonzeros", "vector"},
            {"output"},
            R"(
                constexpr uint simdgroup_width = 32;
                uint lane = thread_index_in_simdgroup;
                uint row = thread_position_in_grid.x / simdgroup_width;
                float partial = 0.0f;
                int begin = row_starts[row];
                int end = row_starts[row + 1];
                for (int k = begin + int(lane); k < end;
                     k += int(simdgroup_width)) {
                    partial = fma(nonzeros[k], vector[column_indices[k]], partial);
                }
                float total = simd_sum(partial);
                if (lane == 0) {
                    output[row] = total;
                }
            )");
        const int grid = rows * 32;
        return simdgroup_kernel({row_ptr, col_ind, values, x}, {{rows}},
                                {mx::float32}, {grid, 1, 1},
                                {threadgroup_width, 1, 1}, {}, std::nullopt,
                                false, s_.stream)[0];
    }

    static const auto adaptive_kernel = mx::fast::metal_kernel(
        "mlxpdlp_csr_spmv_adaptive",
        {"row_starts", "column_indices", "nonzeros", "work_offsets", "work_rows", "vector"},
        {"output"},
        R"(
            uint local_thread = thread_index_in_threadgroup;
            uint threadgroup_width = threads_per_threadgroup.x;
            uint work_item = threadgroup_position_in_grid.x;
            int descriptor_begin = work_offsets[work_item];
            int descriptor_end = work_offsets[work_item + 1];
            int first_row = work_rows[descriptor_begin];
            threadgroup float partials[256];

            constexpr int medium_row_marker = 1073741824;
            if (first_row < -medium_row_marker) {
                uint simd_lane = local_thread & 31;
                uint simd_group = local_thread >> 5;
                int descriptor = descriptor_begin + int(simd_group);
                if (descriptor < descriptor_end) {
                    uint row = uint(-work_rows[descriptor] - medium_row_marker - 1);
                    float total = 0.0f;
                    int begin = row_starts[row];
                    int end = row_starts[row + 1];
                    for (int k = begin + int(simd_lane); k < end; k += 32) {
                        total = fma(nonzeros[k], vector[column_indices[k]], total);
                    }
                    total = simd_sum(total);
                    if (simd_lane == 0) {
                        output[row] = total;
                    }
                }
            } else if (first_row < 0) {
                uint row = uint(-first_row - 1);
                float partial = 0.0f;
                int begin = row_starts[row];
                int end = row_starts[row + 1];
                for (int k = begin + int(local_thread); k < end;
                     k += int(threadgroup_width)) {
                    partial = fma(nonzeros[k], vector[column_indices[k]], partial);
                }
                partials[local_thread] = partial;
                threadgroup_barrier(mem_flags::mem_threadgroup);
                for (uint stride = 128; stride > 0; stride >>= 1) {
                    if (local_thread < stride) {
                        partials[local_thread] += partials[local_thread + stride];
                    }
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                }
                if (local_thread == 0) {
                    output[row] = partials[0];
                }
            } else {
                int descriptor = descriptor_begin + int(local_thread);
                if (descriptor < descriptor_end) {
                    uint row = uint(work_rows[descriptor]);
                    float total = 0.0f;
                    int begin = row_starts[row];
                    int end = row_starts[row + 1];
                    for (int k = begin; k < end; ++k) {
                        total = fma(nonzeros[k], vector[column_indices[k]], total);
                    }
                    output[row] = total;
                }
            }
        )");

    int grid = work_item_count * threadgroup_width;
    int threadgroup = threadgroup_width;
    return adaptive_kernel({row_ptr, col_ind, values, work_offsets, work_rows, x},
                           {{rows}}, {mx::float32}, {grid, 1, 1},
                           {threadgroup, 1, 1}, {}, std::nullopt, false,
                           s_.stream)[0];
}

// ---------------------------------------------------------------------------
// Fused sparse Metal PDHG half-steps
// ---------------------------------------------------------------------------
//
// Each fused kernel performs one full half-step of the PDHG iteration:
// CSR SpMV + scaled gradient step + bound projection + reflection +
// Halpern weighting, writing x_cur/x_ref (primal) or y_cur/y_ref (dual)
// plus the major-iteration snapshot arrays in a single dispatch. Minor kernel
// variants expose only the two live state outputs, so MLX does not allocate
// snapshot buffers that the caller will discard. The update
// bodies replicate the MLX expression sequences in
// mlx_compute_next_primal/_dual so the fused path stays numerically
// comparable with the unfused one. Evaluation of both half-step kernels of
// one iteration is deferred to a single mx::eval in mlx_compute_next_dual.

namespace {

// MSL header shared by all fused kernels. The scalar block is
// {step, reflection_coefficient, halpern_weight} in float32.
const char *fused_step_header() {
    // MLX selects constant or device address space from each input's element
    // count. Template the complete read-pointer type so either signature can
    // call the shared update helpers.
    static const std::string header = R"(
struct FusedPrimalValues {
    float x_cur;
    float x_ref;
    float x_pdhg;
    float temp;
};

template <typename PrimalInputPtr>
inline FusedPrimalValues fused_primal_values(
    uint row,
    float product,
    PrimalInputPtr x_cur, PrimalInputPtr x_init,
    PrimalInputPtr obj, PrimalInputPtr var_lb, PrimalInputPtr var_ub,
    const constant float* scalars) {
    float step = scalars[0];
    float rc = scalars[1];
    float weight = scalars[2];
    float temp = x_cur[row] - step * (obj[row] - product);
    float proj = min(max(temp, var_lb[row]), var_ub[row]);
    float xref = 2.0f * proj - x_cur[row];
    float reflected = rc * xref + (1.0f - rc) * x_cur[row];
    FusedPrimalValues result;
    result.x_cur = weight * reflected + (1.0f - weight) * x_init[row];
    result.x_ref = xref;
    result.x_pdhg = proj;
    result.temp = temp;
    return result;
}

template <typename PrimalInputPtr>
inline void fused_primal_major_update(
    uint row,
    float product,
    PrimalInputPtr x_cur, PrimalInputPtr x_init,
    PrimalInputPtr obj, PrimalInputPtr var_lb, PrimalInputPtr var_ub,
    const constant float* scalars,
    device float* x_cur_out, device float* x_ref_out,
    device float* x_pdhg_out, device float* dual_slack_out) {
    FusedPrimalValues result = fused_primal_values(
        row, product, x_cur, x_init, obj, var_lb, var_ub, scalars);
    x_cur_out[row] = result.x_cur;
    x_ref_out[row] = result.x_ref;
    x_pdhg_out[row] = result.x_pdhg;
    dual_slack_out[row] = (result.x_pdhg - result.temp) / scalars[0];
}

template <typename PrimalInputPtr>
inline void fused_primal_minor_update(
    uint row,
    float product,
    PrimalInputPtr x_cur, PrimalInputPtr x_init,
    PrimalInputPtr obj, PrimalInputPtr var_lb, PrimalInputPtr var_ub,
    const constant float* scalars,
    device float* x_cur_out, device float* x_ref_out) {
    FusedPrimalValues result = fused_primal_values(
        row, product, x_cur, x_init, obj, var_lb, var_ub, scalars);
    x_cur_out[row] = result.x_cur;
    x_ref_out[row] = result.x_ref;
}

struct FusedDualValues {
    float y_cur;
    float y_ref;
    float y_pdhg;
};

template <typename DualInputPtr>
inline FusedDualValues fused_dual_values(
    uint row,
    float product,
    DualInputPtr y_cur, DualInputPtr y_init,
    DualInputPtr con_lb, DualInputPtr con_ub,
    const constant float* scalars) {
    float step = scalars[0];
    float rc = scalars[1];
    float weight = scalars[2];
    float temp = y_cur[row] / step - product;
    float proj = min(max(temp, -con_ub[row]), -con_lb[row]);
    float ycand = (temp - proj) * step;
    float yref = 2.0f * ycand - y_cur[row];
    float reflected = rc * yref + (1.0f - rc) * y_cur[row];
    FusedDualValues result;
    result.y_cur = weight * reflected + (1.0f - weight) * y_init[row];
    result.y_ref = yref;
    result.y_pdhg = ycand;
    return result;
}

template <typename DualInputPtr>
inline void fused_dual_major_update(
    uint row,
    float product,
    DualInputPtr y_cur, DualInputPtr y_init,
    DualInputPtr con_lb, DualInputPtr con_ub,
    const constant float* scalars,
    device float* y_cur_out, device float* y_ref_out,
    device float* y_pdhg_out) {
    FusedDualValues result = fused_dual_values(
        row, product, y_cur, y_init, con_lb, con_ub, scalars);
    y_cur_out[row] = result.y_cur;
    y_ref_out[row] = result.y_ref;
    y_pdhg_out[row] = result.y_pdhg;
}

template <typename DualInputPtr>
inline void fused_dual_minor_update(
    uint row,
    float product,
    DualInputPtr y_cur, DualInputPtr y_init,
    DualInputPtr con_lb, DualInputPtr con_ub,
    const constant float* scalars,
    device float* y_cur_out) {
    FusedDualValues result = fused_dual_values(
        row, product, y_cur, y_init, con_lb, con_ub, scalars);
    y_cur_out[row] = result.y_cur;
}
)";
    return header.c_str();
}
std::string replace_all(std::string text, const std::string &from, const std::string &to) {
    size_t pos = 0;
    while ((pos = text.find(from, pos)) != std::string::npos) {
        text.replace(pos, from.size(), to);
        pos += to.size();
    }
    return text;
}

// Assemble the kernel body for one SpMV dispatch strategy. The RS/CI/VALS
// tokens name the CSR arrays, VEC the operand vector, WO/WR the adaptive work
// arrays, UPD the update call using register accumulator acc, and UPD_LONG
// the update call using the reduced threadgroup value partials[0].
std::string fused_body(const char *rs, const char *ci, const char *vals,
                       const char *wo, const char *wr, const char *vec,
                       SparseMetalSpmvStrategy strategy,
                       const std::string &update_acc,
                       const std::string &update_partials) {
    std::string body;
    switch (strategy) {
    case SparseMetalSpmvStrategy::scalar_rows:
        body = R"(
            uint row = thread_position_in_grid.x;
            float acc = 0.0f;
            int begin = $RS$[row];
            int end = $RS$[row + 1];
            for (int k = begin; k < end; ++k) {
                acc = fma($VALS$[k], $VEC$[$CI$[k]], acc);
            }
            $UPD$
        )";
        break;
    case SparseMetalSpmvStrategy::simdgroup_rows:
        body = R"(
            constexpr uint simdgroup_width = 32;
            uint lane = thread_index_in_simdgroup;
            uint row = thread_position_in_grid.x / simdgroup_width;
            float partial = 0.0f;
            int begin = $RS$[row];
            int end = $RS$[row + 1];
            for (int k = begin + int(lane); k < end; k += int(simdgroup_width)) {
                partial = fma($VALS$[k], $VEC$[$CI$[k]], partial);
            }
            float acc = simd_sum(partial);
            if (lane == 0) {
                $UPD$
            }
        )";
        break;
    case SparseMetalSpmvStrategy::adaptive:
        body = R"(
            uint local_thread = thread_index_in_threadgroup;
            uint threadgroup_width = threads_per_threadgroup.x;
            uint work_item = threadgroup_position_in_grid.x;
            int descriptor_begin = $WO$[work_item];
            int descriptor_end = $WO$[work_item + 1];
            int first_row = $WR$[descriptor_begin];
            threadgroup float partials[256];
            constexpr int medium_row_marker = 1073741824;
            if (first_row < -medium_row_marker) {
                uint simd_lane = local_thread & 31;
                uint simd_group = local_thread >> 5;
                int descriptor = descriptor_begin + int(simd_group);
                if (descriptor < descriptor_end) {
                    uint row = uint(-$WR$[descriptor] - medium_row_marker - 1);
                    float acc = 0.0f;
                    int begin = $RS$[row];
                    int end = $RS$[row + 1];
                    for (int k = begin + int(simd_lane); k < end; k += 32) {
                        acc = fma($VALS$[k], $VEC$[$CI$[k]], acc);
                    }
                    acc = simd_sum(acc);
                    if (simd_lane == 0) {
                        $UPD$
                    }
                }
            } else if (first_row < 0) {
                uint row = uint(-first_row - 1);
                float partial = 0.0f;
                int begin = $RS$[row];
                int end = $RS$[row + 1];
                for (int k = begin + int(local_thread); k < end;
                     k += int(threadgroup_width)) {
                    partial = fma($VALS$[k], $VEC$[$CI$[k]], partial);
                }
                partials[local_thread] = partial;
                threadgroup_barrier(mem_flags::mem_threadgroup);
                for (uint stride = 128; stride > 0; stride >>= 1) {
                    if (local_thread < stride) {
                        partials[local_thread] += partials[local_thread + stride];
                    }
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                }
                if (local_thread == 0) {
                    $UPD_LONG$
                }
            } else {
                int descriptor = descriptor_begin + int(local_thread);
                if (descriptor < descriptor_end) {
                    uint row = uint($WR$[descriptor]);
                    float acc = 0.0f;
                    int begin = $RS$[row];
                    int end = $RS$[row + 1];
                    for (int k = begin; k < end; ++k) {
                        acc = fma($VALS$[k], $VEC$[$CI$[k]], acc);
                    }
                    $UPD$
                }
            }
        )";
        break;
    }
    body = replace_all(std::move(body), "$RS$", rs);
    body = replace_all(std::move(body), "$CI$", ci);
    body = replace_all(std::move(body), "$VALS$", vals);
    body = replace_all(std::move(body), "$WO$", wo);
    body = replace_all(std::move(body), "$WR$", wr);
    body = replace_all(std::move(body), "$VEC$", vec);
    body = replace_all(std::move(body), "$UPD_LONG$", update_partials);
    body = replace_all(std::move(body), "$UPD$", update_acc);
    return body;
}
const std::string primal_major_update_acc =
    "fused_primal_major_update(row, acc, x_cur, x_init, obj, var_lb, var_ub, scalars, "
    "x_cur_out, x_ref_out, x_pdhg_out, dual_slack_out);";
const std::string primal_major_update_partials =
    "fused_primal_major_update(row, partials[0], x_cur, x_init, obj, var_lb, var_ub, "
    "scalars, x_cur_out, x_ref_out, x_pdhg_out, dual_slack_out);";
const std::string primal_minor_update_acc =
    "fused_primal_minor_update(row, acc, x_cur, x_init, obj, var_lb, var_ub, scalars, "
    "x_cur_out, x_ref_out);";
const std::string primal_minor_update_partials =
    "fused_primal_minor_update(row, partials[0], x_cur, x_init, obj, var_lb, var_ub, "
    "scalars, x_cur_out, x_ref_out);";
const std::string dual_major_update_acc =
    "fused_dual_major_update(row, acc, y_cur, y_init, con_lb, con_ub, scalars, "
    "y_cur_out, y_ref_out, y_pdhg_out);";
const std::string dual_major_update_partials =
    "fused_dual_major_update(row, partials[0], y_cur, y_init, con_lb, con_ub, scalars, "
    "y_cur_out, y_ref_out, y_pdhg_out);";
const std::string dual_minor_update_acc =
    "fused_dual_minor_update(row, acc, y_cur, y_init, con_lb, con_ub, scalars, "
    "y_cur_out);";
const std::string dual_minor_update_partials =
    "fused_dual_minor_update(row, partials[0], y_cur, y_init, con_lb, con_ub, scalars, "
    "y_cur_out);";

struct FusedKernelVariants {
    mx::fast::CustomKernelFunction major;
    mx::fast::CustomKernelFunction minor;
};

FusedKernelVariants make_primal_fused_kernels(
    const std::string &base_name, const std::vector<std::string> &input_names,
    const char *row_starts, const char *column_indices, const char *values,
    const char *work_offsets, const char *work_rows, const char *vector,
    SparseMetalSpmvStrategy strategy) {
    return {
        mx::fast::metal_kernel(
            base_name + "_major", input_names,
            {"x_cur_out", "x_ref_out", "x_pdhg_out", "dual_slack_out"},
            fused_body(row_starts, column_indices, values, work_offsets,
                       work_rows, vector, strategy, primal_major_update_acc,
                       primal_major_update_partials),
            fused_step_header()),
        mx::fast::metal_kernel(
            base_name + "_minor", input_names, {"x_cur_out", "x_ref_out"},
            fused_body(row_starts, column_indices, values, work_offsets,
                       work_rows, vector, strategy, primal_minor_update_acc,
                       primal_minor_update_partials),
            fused_step_header()),
    };
}

FusedKernelVariants make_dual_fused_kernels(
    const std::string &base_name, const std::vector<std::string> &input_names,
    const char *row_starts, const char *column_indices, const char *values,
    const char *work_offsets, const char *work_rows, const char *vector,
    SparseMetalSpmvStrategy strategy) {
    return {
        mx::fast::metal_kernel(
            base_name + "_major", input_names,
            {"y_cur_out", "y_ref_out", "y_pdhg_out"},
            fused_body(row_starts, column_indices, values, work_offsets,
                       work_rows, vector, strategy, dual_major_update_acc,
                       dual_major_update_partials),
            fused_step_header()),
        mx::fast::metal_kernel(
            base_name + "_minor", input_names, {"y_cur_out"},
            fused_body(row_starts, column_indices, values, work_offsets,
                       work_rows, vector, strategy, dual_minor_update_acc,
                       dual_minor_update_partials),
            fused_step_header()),
    };
}

} // namespace

std::vector<mx::array> MlxPdlpSolver::fused_primal_step(
    const mx::array &scalars, bool is_major) {
    constexpr int threadgroup_width = 256;
    const FusedKernelVariants *kernels = nullptr;
    std::vector<mx::array> inputs;
    int grid_size = 0;

    switch (s_.sparse_at_spmv_strategy) {
    case SparseMetalSpmvStrategy::scalar_rows: {
        static const auto variants = make_primal_fused_kernels(
            "mlxpdlp_fused_primal_scalar_rows",
            {"at_row_starts", "at_col_ind", "at_values", "y_cur", "x_cur",
             "x_init", "obj", "var_lb", "var_ub", "scalars"},
            "at_row_starts", "at_col_ind", "at_values", "at_work_offsets",
            "at_work_rows", "y_cur", SparseMetalSpmvStrategy::scalar_rows);
        kernels = &variants;
        inputs = {sparse_at_row_ptr_, sparse_at_col_ind_, sparse_at_values_,
                  s_.y_cur, s_.x_cur, s_.x_init, s_.obj, s_.var_lb,
                  s_.var_ub, scalars};
        grid_size = s_.n;
        break;
    }
    case SparseMetalSpmvStrategy::simdgroup_rows: {
        static const auto variants = make_primal_fused_kernels(
            "mlxpdlp_fused_primal_simdgroup_rows",
            {"at_row_starts", "at_col_ind", "at_values", "y_cur", "x_cur",
             "x_init", "obj", "var_lb", "var_ub", "scalars"},
            "at_row_starts", "at_col_ind", "at_values", "at_work_offsets",
            "at_work_rows", "y_cur", SparseMetalSpmvStrategy::simdgroup_rows);
        kernels = &variants;
        inputs = {sparse_at_row_ptr_, sparse_at_col_ind_, sparse_at_values_,
                  s_.y_cur, s_.x_cur, s_.x_init, s_.obj, s_.var_lb,
                  s_.var_ub, scalars};
        grid_size = s_.n * 32;
        break;
    }
    case SparseMetalSpmvStrategy::adaptive: {
        static const auto variants = make_primal_fused_kernels(
            "mlxpdlp_fused_primal_adaptive",
            {"at_row_starts", "at_col_ind", "at_values", "at_work_offsets",
             "at_work_rows", "y_cur", "x_cur", "x_init", "obj", "var_lb",
             "var_ub", "scalars"},
            "at_row_starts", "at_col_ind", "at_values", "at_work_offsets",
            "at_work_rows", "y_cur", SparseMetalSpmvStrategy::adaptive);
        kernels = &variants;
        inputs = {sparse_at_row_ptr_, sparse_at_col_ind_, sparse_at_values_,
                  sparse_at_work_offsets_, sparse_at_work_rows_, s_.y_cur,
                  s_.x_cur, s_.x_init, s_.obj, s_.var_lb, s_.var_ub, scalars};
        grid_size = sparse_at_work_item_count_ * threadgroup_width;
        break;
    }
    }
    if (!kernels) {
        throw std::logic_error(
            "unknown sparse Metal SpMV strategy for fused primal step");
    }
    if (is_major) {
        return kernels->major(
            inputs, {{s_.n}, {s_.n}, {s_.n}, {s_.n}},
            {mx::float32, mx::float32, mx::float32, mx::float32},
            {grid_size, 1, 1}, {threadgroup_width, 1, 1}, {}, std::nullopt,
            false, s_.stream);
    }
    return kernels->minor(inputs, {{s_.n}, {s_.n}},
                          {mx::float32, mx::float32}, {grid_size, 1, 1},
                          {threadgroup_width, 1, 1}, {}, std::nullopt, false,
                          s_.stream);
}

std::vector<mx::array> MlxPdlpSolver::fused_dual_step(
    const mx::array &scalars, bool is_major) {
    constexpr int threadgroup_width = 256;
    const FusedKernelVariants *kernels = nullptr;
    std::vector<mx::array> inputs;
    int grid_size = 0;

    switch (s_.sparse_a_spmv_strategy) {
    case SparseMetalSpmvStrategy::scalar_rows: {
        static const auto variants = make_dual_fused_kernels(
            "mlxpdlp_fused_dual_scalar_rows",
            {"a_row_starts", "a_col_ind", "a_values", "x_ref", "y_cur",
             "y_init", "con_lb", "con_ub", "scalars"},
            "a_row_starts", "a_col_ind", "a_values", "a_work_offsets",
            "a_work_rows", "x_ref", SparseMetalSpmvStrategy::scalar_rows);
        kernels = &variants;
        inputs = {sparse_a_row_ptr_, sparse_a_col_ind_, sparse_a_values_, s_.x_ref,
                  s_.y_cur, s_.y_init, s_.con_lb, s_.con_ub, scalars};
        grid_size = s_.m;
        break;
    }
    case SparseMetalSpmvStrategy::simdgroup_rows: {
        static const auto variants = make_dual_fused_kernels(
            "mlxpdlp_fused_dual_simdgroup_rows",
            {"a_row_starts", "a_col_ind", "a_values", "x_ref", "y_cur",
             "y_init", "con_lb", "con_ub", "scalars"},
            "a_row_starts", "a_col_ind", "a_values", "a_work_offsets",
            "a_work_rows", "x_ref", SparseMetalSpmvStrategy::simdgroup_rows);
        kernels = &variants;
        inputs = {sparse_a_row_ptr_, sparse_a_col_ind_, sparse_a_values_, s_.x_ref,
                  s_.y_cur, s_.y_init, s_.con_lb, s_.con_ub, scalars};
        grid_size = s_.m * 32;
        break;
    }
    case SparseMetalSpmvStrategy::adaptive: {
        static const auto variants = make_dual_fused_kernels(
            "mlxpdlp_fused_dual_adaptive",
            {"a_row_starts", "a_col_ind", "a_values", "a_work_offsets",
             "a_work_rows", "x_ref", "y_cur", "y_init", "con_lb", "con_ub",
             "scalars"},
            "a_row_starts", "a_col_ind", "a_values", "a_work_offsets",
            "a_work_rows", "x_ref", SparseMetalSpmvStrategy::adaptive);
        kernels = &variants;
        inputs = {sparse_a_row_ptr_, sparse_a_col_ind_, sparse_a_values_,
                  sparse_a_work_offsets_, sparse_a_work_rows_, s_.x_ref,
                  s_.y_cur, s_.y_init, s_.con_lb, s_.con_ub, scalars};
        grid_size = sparse_a_work_item_count_ * threadgroup_width;
        break;
    }
    }
    if (!kernels) {
        throw std::logic_error(
            "unknown sparse Metal SpMV strategy for fused dual step");
    }
    if (is_major) {
        return kernels->major(
            inputs, {{s_.m}, {s_.m}, {s_.m}},
            {mx::float32, mx::float32, mx::float32}, {grid_size, 1, 1},
            {threadgroup_width, 1, 1}, {}, std::nullopt, false, s_.stream);
    }
    return kernels->minor(inputs, {{s_.m}},
                          {mx::float32}, {grid_size, 1, 1},
                          {threadgroup_width, 1, 1}, {}, std::nullopt, false,
                          s_.stream);
}

int MlxPdlpSolver::fused_eval_batch_size() const {
    if (!params_.metal_fused_kernels || !s_.sparse_metal_active) {
        return 1;
    }
    // Each fused minor iteration materializes (2n + m) floats plus a
    // 3-float scalar block; major iterations add (2n + m) snapshot floats.
    // Bound the lazy batch graph to a fixed intermediate footprint so large
    // problems do not accumulate hundreds of iteration outputs in memory.
    constexpr double target_bytes = 256.0 * 1024.0 * 1024.0;
    constexpr int max_batch = 16;
    const double per_iteration_bytes =
        static_cast<double>(2LL * s_.n + s_.m) * sizeof(float);
    if (per_iteration_bytes <= 0.0) {
        return 1;
    }
    int batch = static_cast<int>(target_bytes / per_iteration_bytes);
    return std::clamp(batch, 1, max_batch);
}

mx::array MlxPdlpSolver::sparse_cpu_matvec(const mx::array &x, bool transpose,
                                           int rows) {
#ifdef MLXPDLP_HAS_ACCELERATE_SPARSE
    if (!sparse_cpu_matrix_) {
        throw std::logic_error("sparse CPU matrix is not prepared");
    }
    auto input = mx::contiguous(x, false, s_.stream);
    auto primitive = std::make_shared<CpuSparseMatvecPrimitive>(
        s_.stream, sparse_cpu_matrix_, transpose ? CblasTrans : CblasNoTrans, rows);
    return mx::array(mx::Shape{rows}, mx::float64, std::move(primitive),
                     std::vector<mx::array>{std::move(input)});
#else
    (void)x;
    (void)transpose;
    (void)rows;
    throw std::runtime_error("sparse CPU backend requires Apple Accelerate");
#endif
}

mx::array MlxPdlpSolver::mat_Ax(const mx::array &x) {
    if (s_.sparse_metal_active) {
        return sparse_matvec(sparse_a_row_ptr_, sparse_a_col_ind_, sparse_a_values_,
                             sparse_a_work_offsets_, sparse_a_work_rows_, x, s_.m,
                             sparse_a_work_item_count_, s_.sparse_a_spmv_strategy);
    }
    if (s_.sparse_cpu_active) {
        return sparse_cpu_matvec(x, false, s_.m);
    }

    // A is [m, n], x is [n] → result is [m]
    // matmul requires 2D inputs: reshape x to [n, 1], result is [m, 1], squeeze to [m]
    auto x_2d = mx::reshape(x, {s_.n, 1});
    auto y_2d = mx::matmul(s_.A, x_2d);
    return mx::reshape(y_2d, {s_.m});
}

mx::array MlxPdlpSolver::mat_ATx(const mx::array &y) {
    if (s_.sparse_metal_active) {
        return sparse_matvec(sparse_at_row_ptr_, sparse_at_col_ind_, sparse_at_values_,
                             sparse_at_work_offsets_, sparse_at_work_rows_, y, s_.n,
                             sparse_at_work_item_count_, s_.sparse_at_spmv_strategy);
    }
    if (s_.sparse_cpu_active) {
        return sparse_cpu_matvec(y, true, s_.n);
    }

    // AT is [n, m], y is [m] → result is [n]
    auto y_2d = mx::reshape(y, {s_.m, 1});
    auto x_2d = mx::matmul(s_.AT, y_2d);
    return mx::reshape(x_2d, {s_.n});
}

// ---------------------------------------------------------------------------
// Scalar reductions
// ---------------------------------------------------------------------------

double MlxPdlpSolver::mlx_dot(const mx::array &a, const mx::array &b) {
    auto prod = a * b;
    auto s = mx::sum(prod);
    mx::eval(s);
    return mlx_scalar_as_double(s);
}

double MlxPdlpSolver::mlx_norm2(const mx::array &v) {
    auto n = mx::linalg::norm(v);
    mx::eval(n);
    return mlx_scalar_as_double(n);
}

double MlxPdlpSolver::mlx_norm_inf(const mx::array &v) {
    auto n = mx::max(mx::abs(v));
    mx::eval(n);
    return mlx_scalar_as_double(n);
}

double MlxPdlpSolver::mlx_sum(const mx::array &v) {
    auto s = mx::sum(v);
    mx::eval(s);
    return mlx_scalar_as_double(s);
}

// ---------------------------------------------------------------------------
// Preconditioning
// ---------------------------------------------------------------------------

void MlxPdlpSolver::apply_sparse_scaling(const std::vector<double> &con_scale,
                                         const std::vector<double> &var_scale) {
    if (con_scale.size() != static_cast<size_t>(s_.m) ||
        var_scale.size() != static_cast<size_t>(s_.n)) {
        throw std::invalid_argument("sparse scaling dimensions do not match the matrix");
    }

    std::vector<double> inv_con_scale(static_cast<size_t>(s_.m));
    std::vector<double> inv_var_scale(static_cast<size_t>(s_.n));
    for (int row = 0; row < s_.m; ++row) {
        double scale = con_scale[static_cast<size_t>(row)];
        if (!(scale > 0.0) || !std::isfinite(scale)) {
            throw std::runtime_error("invalid constraint scaling for sparse Metal matrix");
        }
        inv_con_scale[static_cast<size_t>(row)] = 1.0 / scale;
        sparse_con_rescale_host_[static_cast<size_t>(row)] *= scale;
    }
    for (int col = 0; col < s_.n; ++col) {
        double scale = var_scale[static_cast<size_t>(col)];
        if (!(scale > 0.0) || !std::isfinite(scale)) {
            throw std::runtime_error("invalid variable scaling for sparse Metal matrix");
        }
        inv_var_scale[static_cast<size_t>(col)] = 1.0 / scale;
        sparse_var_rescale_host_[static_cast<size_t>(col)] *= scale;
    }

    for (int row = 0; row < s_.m; ++row) {
        double row_multiplier = inv_con_scale[static_cast<size_t>(row)];
        for (int32_t k = sparse_a_row_ptr_host_[static_cast<size_t>(row)];
             k < sparse_a_row_ptr_host_[static_cast<size_t>(row) + 1]; ++k) {
            int col = sparse_a_col_ind_host_[static_cast<size_t>(k)];
            sparse_a_values_host_[static_cast<size_t>(k)] *=
                row_multiplier * inv_var_scale[static_cast<size_t>(col)];
        }
    }

    auto con_scale_array =
        mlx_array_from_doubles(con_scale.data(), s_.m, s_.obj.dtype());
    auto inv_con_scale_array =
        mlx_array_from_doubles(inv_con_scale.data(), s_.m, s_.obj.dtype());
    auto var_scale_array =
        mlx_array_from_doubles(var_scale.data(), s_.n, s_.obj.dtype());
    auto inv_var_scale_array =
        mlx_array_from_doubles(inv_var_scale.data(), s_.n, s_.obj.dtype());

    s_.con_lb = s_.con_lb * inv_con_scale_array;
    s_.con_ub = s_.con_ub * inv_con_scale_array;
    s_.y_cur = s_.y_cur * con_scale_array;
    s_.obj = s_.obj * inv_var_scale_array;
    s_.var_lb = s_.var_lb * var_scale_array;
    s_.var_ub = s_.var_ub * var_scale_array;
    s_.x_cur = s_.x_cur * var_scale_array;

    mx::eval(s_.con_lb, s_.con_ub, s_.y_cur, s_.obj, s_.var_lb, s_.var_ub, s_.x_cur);
}

void MlxPdlpSolver::publish_sparse_rescaling() {
    s_.con_rescale = mlx_array_from_doubles(
        sparse_con_rescale_host_.data(), s_.m, s_.obj.dtype());
    s_.var_rescale = mlx_array_from_doubles(
        sparse_var_rescale_host_.data(), s_.n, s_.obj.dtype());
    mx::eval(s_.con_rescale, s_.var_rescale);
}

void MlxPdlpSolver::mlx_geometric_mean_scaling(int num_iters) {
    if (num_iters <= 0 || s_.nnz == 0)
        return;

    // Match cuPDLPx's Tomlin geometric-mean preconditioner. Multipliers are
    // computed against the current matrix without modifying it between
    // alternating row and column updates; the converged diagonal scaling is
    // applied once at the end.
    constexpr double multiplier_max = 1e20;
    constexpr double multiplier_min = 1.0 / multiplier_max;
    const double infinity = std::numeric_limits<double>::infinity();

    std::vector<double> row_multiplier(static_cast<size_t>(s_.m), 1.0);
    std::vector<double> col_multiplier(static_cast<size_t>(s_.n), 1.0);
    std::vector<double> row_min(static_cast<size_t>(s_.m));
    std::vector<double> row_max(static_cast<size_t>(s_.m));
    std::vector<double> col_min(static_cast<size_t>(s_.n));
    std::vector<double> col_max(static_cast<size_t>(s_.n));

    for (int iter = 0; iter < num_iters; ++iter) {
        std::fill(row_min.begin(), row_min.end(), infinity);
        std::fill(row_max.begin(), row_max.end(), 0.0);
        for (int row = 0; row < s_.m; ++row) {
            for (int32_t k = sparse_a_row_ptr_host_[static_cast<size_t>(row)];
                 k < sparse_a_row_ptr_host_[static_cast<size_t>(row) + 1]; ++k) {
                const int col = sparse_a_col_ind_host_[static_cast<size_t>(k)];
                const double scaled = std::abs(sparse_a_values_host_[static_cast<size_t>(k)]) *
                                      col_multiplier[static_cast<size_t>(col)];
                if (!(scaled > 0.0) || !std::isfinite(scaled))
                    continue;
                row_min[static_cast<size_t>(row)] =
                    std::min(row_min[static_cast<size_t>(row)], scaled);
                row_max[static_cast<size_t>(row)] =
                    std::max(row_max[static_cast<size_t>(row)], scaled);
            }
            if (row_max[static_cast<size_t>(row)] > 0.0) {
                const double updated = 1.0 / (std::sqrt(row_min[static_cast<size_t>(row)]) *
                                              std::sqrt(row_max[static_cast<size_t>(row)]));
                row_multiplier[static_cast<size_t>(row)] =
                    std::clamp(updated, multiplier_min, multiplier_max);
            }
        }

        std::fill(col_min.begin(), col_min.end(), infinity);
        std::fill(col_max.begin(), col_max.end(), 0.0);
        for (int row = 0; row < s_.m; ++row) {
            for (int32_t k = sparse_a_row_ptr_host_[static_cast<size_t>(row)];
                 k < sparse_a_row_ptr_host_[static_cast<size_t>(row) + 1]; ++k) {
                const int col = sparse_a_col_ind_host_[static_cast<size_t>(k)];
                const double scaled = std::abs(sparse_a_values_host_[static_cast<size_t>(k)]) *
                                      row_multiplier[static_cast<size_t>(row)];
                if (!(scaled > 0.0) || !std::isfinite(scaled))
                    continue;
                col_min[static_cast<size_t>(col)] =
                    std::min(col_min[static_cast<size_t>(col)], scaled);
                col_max[static_cast<size_t>(col)] =
                    std::max(col_max[static_cast<size_t>(col)], scaled);
            }
        }
        for (int col = 0; col < s_.n; ++col) {
            if (col_max[static_cast<size_t>(col)] > 0.0) {
                const double updated = 1.0 / (std::sqrt(col_min[static_cast<size_t>(col)]) *
                                              std::sqrt(col_max[static_cast<size_t>(col)]));
                col_multiplier[static_cast<size_t>(col)] =
                    std::clamp(updated, multiplier_min, multiplier_max);
            }
        }
    }

    std::vector<double> con_scale(static_cast<size_t>(s_.m), 1.0);
    std::vector<double> var_scale(static_cast<size_t>(s_.n), 1.0);
    for (int row = 0; row < s_.m; ++row)
        con_scale[static_cast<size_t>(row)] = 1.0 / row_multiplier[static_cast<size_t>(row)];
    for (int col = 0; col < s_.n; ++col)
        var_scale[static_cast<size_t>(col)] = 1.0 / col_multiplier[static_cast<size_t>(col)];

    if (sparse_metal_candidate_ || sparse_cpu_candidate_) {
        apply_sparse_scaling(con_scale, var_scale);
        publish_sparse_rescaling();
        return;
    }

    auto con_scale_array = mlx_array_from_doubles(con_scale.data(), s_.m, s_.obj.dtype());
    auto var_scale_array = mlx_array_from_doubles(var_scale.data(), s_.n, s_.obj.dtype());
    auto inv_con_scale = 1.0 / con_scale_array;
    auto inv_var_scale = 1.0 / var_scale_array;
    s_.con_rescale = s_.con_rescale * con_scale_array;
    s_.var_rescale = s_.var_rescale * var_scale_array;
    s_.A = s_.A * mx::reshape(inv_con_scale, {s_.m, 1}) * mx::reshape(inv_var_scale, {1, s_.n});
    s_.con_lb = s_.con_lb * inv_con_scale;
    s_.con_ub = s_.con_ub * inv_con_scale;
    s_.y_cur = s_.y_cur * con_scale_array;
    s_.obj = s_.obj * inv_var_scale;
    s_.var_lb = s_.var_lb * var_scale_array;
    s_.var_ub = s_.var_ub * var_scale_array;
    s_.x_cur = s_.x_cur * var_scale_array;
    s_.AT = mx::transpose(s_.A);
    mx::eval(s_.A, s_.AT, s_.con_lb, s_.con_ub, s_.y_cur, s_.obj, s_.var_lb, s_.var_ub, s_.x_cur,
             s_.con_rescale, s_.var_rescale);
}

void MlxPdlpSolver::mlx_curtis_reid_scaling(int num_iters) {
    if (num_iters <= 0 || s_.nnz == 0)
        return;

    std::vector<double> row_log_scale(static_cast<size_t>(s_.m), 0.0);
    std::vector<double> col_log_scale(static_cast<size_t>(s_.n), 0.0);
    std::vector<double> row_sum(static_cast<size_t>(s_.m), 0.0);
    std::vector<double> col_sum(static_cast<size_t>(s_.n), 0.0);
    std::vector<int> row_count(static_cast<size_t>(s_.m), 0);
    std::vector<int> col_count(static_cast<size_t>(s_.n), 0);

    for (int row = 0; row < s_.m; ++row) {
        for (int32_t k = sparse_a_row_ptr_host_[static_cast<size_t>(row)];
             k < sparse_a_row_ptr_host_[static_cast<size_t>(row) + 1]; ++k) {
            const double magnitude =
                std::abs(static_cast<double>(sparse_a_values_host_[static_cast<size_t>(k)]));
            if (!(magnitude > 0.0) || !std::isfinite(magnitude))
                continue;
            ++row_count[static_cast<size_t>(row)];
            ++col_count[static_cast<size_t>(
                sparse_a_col_ind_host_[static_cast<size_t>(k)])];
        }
    }

    // Alternating log-space row/column updates solve the Curtis-Reid
    // least-squares equilibration equations. Compute them on the host in
    // double precision. Final device arrays retain the backend dtype; Metal
    // matrix values are rounded when its sparse buffers are materialized.
    for (int iter = 0; iter < num_iters; ++iter) {
        std::fill(row_sum.begin(), row_sum.end(), 0.0);
        for (int row = 0; row < s_.m; ++row) {
            for (int32_t k = sparse_a_row_ptr_host_[static_cast<size_t>(row)];
                 k < sparse_a_row_ptr_host_[static_cast<size_t>(row) + 1]; ++k) {
                const double magnitude =
                    std::abs(static_cast<double>(sparse_a_values_host_[static_cast<size_t>(k)]));
                if (!(magnitude > 0.0) || !std::isfinite(magnitude))
                    continue;
                const int col = sparse_a_col_ind_host_[static_cast<size_t>(k)];
                row_sum[static_cast<size_t>(row)] +=
                    -std::log(magnitude) - col_log_scale[static_cast<size_t>(col)];
            }
            if (row_count[static_cast<size_t>(row)] > 0) {
                row_log_scale[static_cast<size_t>(row)] =
                    row_sum[static_cast<size_t>(row)] /
                    static_cast<double>(row_count[static_cast<size_t>(row)]);
            }
        }

        std::fill(col_sum.begin(), col_sum.end(), 0.0);
        for (int row = 0; row < s_.m; ++row) {
            for (int32_t k = sparse_a_row_ptr_host_[static_cast<size_t>(row)];
                 k < sparse_a_row_ptr_host_[static_cast<size_t>(row) + 1]; ++k) {
                const double magnitude =
                    std::abs(static_cast<double>(sparse_a_values_host_[static_cast<size_t>(k)]));
                if (!(magnitude > 0.0) || !std::isfinite(magnitude))
                    continue;
                const int col = sparse_a_col_ind_host_[static_cast<size_t>(k)];
                col_sum[static_cast<size_t>(col)] +=
                    -std::log(magnitude) - row_log_scale[static_cast<size_t>(row)];
            }
        }
        for (int col = 0; col < s_.n; ++col) {
            if (col_count[static_cast<size_t>(col)] > 0) {
                col_log_scale[static_cast<size_t>(col)] =
                    col_sum[static_cast<size_t>(col)] /
                    static_cast<double>(col_count[static_cast<size_t>(col)]);
            }
        }
    }

    constexpr double max_abs_log_scale = 30.0;
    std::vector<double> con_scale(static_cast<size_t>(s_.m), 1.0);
    std::vector<double> var_scale(static_cast<size_t>(s_.n), 1.0);
    for (int row = 0; row < s_.m; ++row) {
        const double multiplier =
            std::exp(std::clamp(row_log_scale[static_cast<size_t>(row)],
                                -max_abs_log_scale, max_abs_log_scale));
        con_scale[static_cast<size_t>(row)] = 1.0 / multiplier;
    }
    for (int col = 0; col < s_.n; ++col) {
        const double multiplier =
            std::exp(std::clamp(col_log_scale[static_cast<size_t>(col)],
                                -max_abs_log_scale, max_abs_log_scale));
        var_scale[static_cast<size_t>(col)] = 1.0 / multiplier;
    }

    if (sparse_metal_candidate_ || sparse_cpu_candidate_) {
        apply_sparse_scaling(con_scale, var_scale);
        publish_sparse_rescaling();
        return;
    }

    auto con_scale_array =
        mlx_array_from_doubles(con_scale.data(), s_.m, s_.obj.dtype());
    auto var_scale_array =
        mlx_array_from_doubles(var_scale.data(), s_.n, s_.obj.dtype());
    auto inv_con_scale = 1.0 / con_scale_array;
    auto inv_var_scale = 1.0 / var_scale_array;
    s_.con_rescale = s_.con_rescale * con_scale_array;
    s_.var_rescale = s_.var_rescale * var_scale_array;
    s_.A = s_.A * mx::reshape(inv_con_scale, {s_.m, 1}) *
           mx::reshape(inv_var_scale, {1, s_.n});
    s_.con_lb = s_.con_lb * inv_con_scale;
    s_.con_ub = s_.con_ub * inv_con_scale;
    s_.y_cur = s_.y_cur * con_scale_array;
    s_.obj = s_.obj * inv_var_scale;
    s_.var_lb = s_.var_lb * var_scale_array;
    s_.var_ub = s_.var_ub * var_scale_array;
    s_.x_cur = s_.x_cur * var_scale_array;
    s_.AT = mx::transpose(s_.A);
    mx::eval(s_.A, s_.AT, s_.con_lb, s_.con_ub, s_.y_cur, s_.obj, s_.var_lb, s_.var_ub,
             s_.x_cur, s_.con_rescale, s_.var_rescale);
}

void MlxPdlpSolver::sparse_ruiz_scaling(int num_iters) {
    constexpr double eps = 1e-12;
    std::vector<double> row_absmax(static_cast<size_t>(s_.m));
    std::vector<double> col_absmax(static_cast<size_t>(s_.n));
    std::vector<double> con_scale(static_cast<size_t>(s_.m));
    std::vector<double> var_scale(static_cast<size_t>(s_.n));

    for (int iter = 0; iter < num_iters; ++iter) {
        std::fill(row_absmax.begin(), row_absmax.end(), 0.0);
        std::fill(col_absmax.begin(), col_absmax.end(), 0.0);

        // Match the CUDA CSR preconditioner: both metrics are measured from
        // the same matrix state, then row and column scaling are applied
        // together. Duplicate coordinates contribute as distinct CSR entries.
        for (int row = 0; row < s_.m; ++row) {
            for (int32_t k = sparse_a_row_ptr_host_[static_cast<size_t>(row)];
                 k < sparse_a_row_ptr_host_[static_cast<size_t>(row) + 1]; ++k) {
                double magnitude = std::abs(sparse_a_values_host_[static_cast<size_t>(k)]);
                if (!std::isfinite(magnitude)) {
                    continue;
                }
                row_absmax[static_cast<size_t>(row)] =
                    std::max(row_absmax[static_cast<size_t>(row)], magnitude);
                int col = sparse_a_col_ind_host_[static_cast<size_t>(k)];
                col_absmax[static_cast<size_t>(col)] =
                    std::max(col_absmax[static_cast<size_t>(col)], magnitude);
            }
        }

        for (int row = 0; row < s_.m; ++row) {
            double metric = row_absmax[static_cast<size_t>(row)];
            con_scale[static_cast<size_t>(row)] = metric < eps ? 1.0 : std::sqrt(metric);
        }
        for (int col = 0; col < s_.n; ++col) {
            double metric = col_absmax[static_cast<size_t>(col)];
            var_scale[static_cast<size_t>(col)] = metric < eps ? 1.0 : std::sqrt(metric);
        }

        apply_sparse_scaling(con_scale, var_scale);
    }

    publish_sparse_rescaling();
}

void MlxPdlpSolver::sparse_pock_chambolle_scaling(double alpha) {
    constexpr double eps = 1e-12;
    std::vector<double> row_powsum(static_cast<size_t>(s_.m), 0.0);
    std::vector<double> col_powsum(static_cast<size_t>(s_.n), 0.0);
    std::vector<double> con_scale(static_cast<size_t>(s_.m), 1.0);
    std::vector<double> var_scale(static_cast<size_t>(s_.n), 1.0);

    for (int row = 0; row < s_.m; ++row) {
        for (int32_t k = sparse_a_row_ptr_host_[static_cast<size_t>(row)];
             k < sparse_a_row_ptr_host_[static_cast<size_t>(row) + 1]; ++k) {
            double magnitude =
                std::abs(static_cast<double>(sparse_a_values_host_[static_cast<size_t>(k)]));
            if (!std::isfinite(magnitude)) {
                continue;
            }
            row_powsum[static_cast<size_t>(row)] += std::pow(magnitude, alpha);
            int col = sparse_a_col_ind_host_[static_cast<size_t>(k)];
            col_powsum[static_cast<size_t>(col)] += std::pow(magnitude, 2.0 - alpha);
        }
    }

    for (int row = 0; row < s_.m; ++row) {
        double metric = row_powsum[static_cast<size_t>(row)];
        if (metric >= eps && std::isfinite(metric)) {
            con_scale[static_cast<size_t>(row)] = std::sqrt(metric);
        }
    }
    for (int col = 0; col < s_.n; ++col) {
        double metric = col_powsum[static_cast<size_t>(col)];
        if (metric >= eps && std::isfinite(metric)) {
            var_scale[static_cast<size_t>(col)] = std::sqrt(metric);
        }
    }

    apply_sparse_scaling(con_scale, var_scale);
    publish_sparse_rescaling();
}

void MlxPdlpSolver::mlx_ruiz_scaling(int num_iters) {
    if (sparse_metal_candidate_ || sparse_cpu_candidate_) {
        sparse_ruiz_scaling(num_iters);
        return;
    }

    int m = s_.m, n = s_.n;
    double eps = 1e-12;

    auto con_rescale_acc = s_.con_rescale;
    auto var_rescale_acc = s_.var_rescale;

    for (int iter = 0; iter < num_iters; ++iter) {
        // Match cuPDLPx: measure row and column maxima from the same matrix
        // state, then apply both inverse scales together.
        auto row_absmax = mx::max(mx::abs(s_.A), std::vector<int>{1}, false); // [m]
        auto col_absmax = mx::max(mx::abs(s_.A), std::vector<int>{0}, false); // [n]
        auto row_scale = mx::where(row_absmax < mx::array(eps, s_.obj.dtype()),
                                   mx::ones({m}, s_.obj.dtype()),
                                   mx::sqrt(row_absmax));
        auto col_scale = mx::where(col_absmax < mx::array(eps, s_.obj.dtype()),
                                   mx::ones({n}, s_.obj.dtype()),
                                   mx::sqrt(col_absmax));
        auto inv_row_scale = 1.0 / row_scale; // [m]
        auto inv_col_scale = 1.0 / col_scale; // [n]
        con_rescale_acc = con_rescale_acc * row_scale;
        var_rescale_acc = var_rescale_acc * col_scale;

        s_.A = s_.A * mx::reshape(inv_row_scale, {m, 1}) * mx::reshape(inv_col_scale, {1, n});
        s_.con_lb = s_.con_lb * inv_row_scale;
        s_.con_ub = s_.con_ub * inv_row_scale;
        s_.y_cur = s_.y_cur * row_scale;
        s_.obj = s_.obj * inv_col_scale;
        s_.var_lb = s_.var_lb * col_scale;
        s_.var_ub = s_.var_ub * col_scale;
        s_.x_cur = s_.x_cur * col_scale;

        mx::eval(s_.A, s_.con_lb, s_.con_ub, s_.obj, s_.var_lb, s_.var_ub, s_.x_cur, s_.y_cur);
    }

    s_.con_rescale = con_rescale_acc;
    s_.var_rescale = var_rescale_acc;

    // Eager-eval everything before continuing
    mx::eval(s_.A, s_.con_lb, s_.con_ub, s_.obj, s_.var_lb, s_.var_ub);
    mx::eval(s_.con_rescale, s_.var_rescale);

    if (!sparse_metal_candidate_ && !sparse_cpu_candidate_) {
        s_.AT = mx::transpose(s_.A);
        mx::eval(s_.AT);
    }
}

void MlxPdlpSolver::mlx_pock_chambolle_scaling(double alpha) {
    if (!params_.has_pock_chambolle_alpha)
        return;

    if (sparse_metal_candidate_ || sparse_cpu_candidate_) {
        sparse_pock_chambolle_scaling(alpha);
        return;
    }

    int m = s_.m, n = s_.n;
    double eps = 1e-12;

    // CUDA reference: compute_csr_row_powsum with degree alpha for constraints
    // For dense: sum of |A_ij|^alpha per row
    auto row_powsum =
        mx::sum(mx::power(mx::abs(s_.A), mx::array(alpha, s_.obj.dtype())),
                std::vector<int>{1}, false); // [m]
    auto con_scale =
        mx::where(row_powsum < mx::array(eps, s_.obj.dtype()),
                  mx::ones({m}, s_.obj.dtype()),
                  mx::sqrt(row_powsum));
    auto inv_con_scale = 1.0 / con_scale;
    s_.con_rescale = s_.con_rescale * con_scale;

    // Compute_csr_row_powsum with degree (2.0 - alpha) for variables
    auto col_powsum = mx::sum(
        mx::power(mx::abs(s_.A), mx::array(2.0 - alpha, s_.obj.dtype())),
        std::vector<int>{0}, false); // [n]
    auto var_scale =
        mx::where(col_powsum < mx::array(eps, s_.obj.dtype()),
                  mx::ones({n}, s_.obj.dtype()),
                  mx::sqrt(col_powsum));
    auto inv_var_scale = 1.0 / var_scale;
    s_.var_rescale = s_.var_rescale * var_scale;

    // Scale problem: A *= inv_var * inv_con
    s_.A = s_.A * mx::reshape(inv_con_scale, {m, 1});
    s_.A = s_.A * mx::reshape(inv_var_scale, {1, n});

    // scale_variables_kernel: obj *= inv_var, var_lb *= var, var_ub *= var
    s_.obj = s_.obj * inv_var_scale;
    s_.var_lb = s_.var_lb * var_scale;
    s_.var_ub = s_.var_ub * var_scale;
    s_.x_cur = s_.x_cur * var_scale;

    // scale_constraints_kernel: con_lb *= inv_con, con_ub *= inv_con
    s_.con_lb = s_.con_lb * inv_con_scale;
    s_.con_ub = s_.con_ub * inv_con_scale;
    s_.y_cur = s_.y_cur * con_scale;

    mx::eval(s_.A, s_.obj, s_.var_lb, s_.var_ub, s_.con_lb, s_.con_ub, s_.x_cur, s_.y_cur);
    mx::eval(s_.con_rescale, s_.var_rescale);

    if (!sparse_metal_candidate_ && !sparse_cpu_candidate_) {
        s_.AT = mx::transpose(s_.A);
        mx::eval(s_.AT);
    }
}

void MlxPdlpSolver::mlx_bound_objective_scaling() {
    if (!params_.bound_objective_rescaling)
        return;

    // CUDA reference: compute_bound_contrib_kernel
    // Sum of: (finite lb != ub ? lb² : 0) + (finite ub ? ub² : 0)
    auto con_lb_safe = mx::where(mx::isfinite(s_.con_lb), s_.con_lb, mx::zeros_like(s_.con_lb));
    auto con_ub_safe = mx::where(mx::isfinite(s_.con_ub), s_.con_ub, mx::zeros_like(s_.con_ub));

    // contrib[i] = (fL && (!fU || |Li-Ui| > eps) ? Li² : 0) + (fU ? Ui² : 0)
    // We approximate this: include lb² if finite and not equal to ub, else ub² if finite
    auto lb_finite = mx::isfinite(s_.con_lb);
    auto ub_finite = mx::isfinite(s_.con_ub);
    auto lb_ne_ub = mx::abs(s_.con_lb - s_.con_ub) >
                    mx::array(1e-12, s_.obj.dtype());
    auto use_lb = lb_finite && (mx::logical_not(ub_finite) || lb_ne_ub);
    auto contrib = mx::where(use_lb, mx::square(con_lb_safe), mx::zeros_like(con_lb_safe)) +
                   mx::where(ub_finite, mx::square(con_ub_safe), mx::zeros_like(con_ub_safe));

    auto bnd_norm_sq = mx::sum(contrib);
    mx::eval(bnd_norm_sq);
    double bnd_norm = std::sqrt(mlx_scalar_as_double(bnd_norm_sq));

    // objective norm
    double obj_norm = mlx_norm2(s_.obj);

    double constraint_scale = 1.0 / (bnd_norm + 1.0);
    double objective_scale = 1.0 / (obj_norm + 1.0);

    // scale_bounds_kernel: con_lb *= constraint_scale, con_ub *= constraint_scale
    auto constraint_scale_scalar = mlx_scalar_like(constraint_scale, s_.obj);
    auto objective_scale_scalar = mlx_scalar_like(objective_scale, s_.obj);
    s_.con_lb = s_.con_lb * constraint_scale_scalar;
    s_.con_ub = s_.con_ub * constraint_scale_scalar;
    s_.y_cur = s_.y_cur * objective_scale_scalar;

    // scale_objective_kernel: var_lb *= constraint_scale, var_ub *= constraint_scale,
    // obj *= objective_scale
    s_.var_lb = s_.var_lb * constraint_scale_scalar;
    s_.var_ub = s_.var_ub * constraint_scale_scalar;
    s_.obj = s_.obj * objective_scale_scalar;
    s_.x_cur = s_.x_cur * constraint_scale_scalar;

    // Update accumulated rescaling
    s_.con_bound_rescale = constraint_scale;
    s_.obj_vec_rescale = objective_scale;

    mx::eval(s_.obj, s_.con_lb, s_.con_ub, s_.var_lb, s_.var_ub, s_.x_cur, s_.y_cur);
}

// ---------------------------------------------------------------------------
// Singular value estimation (power method)
// ---------------------------------------------------------------------------

double MlxPdlpSolver::mlx_estimate_max_singular_value() {
    int m = s_.m;
    constexpr int convergence_window = 10;

    // Match the CUDA references' deterministic random-normal start. Compared
    // with an all-ones vector this avoids symmetry-induced nullspace starts and
    // gives every dominant eigenspace a reproducible nonzero projection while
    // keeping CPU and Metal runs comparable.
    std::mt19937 generator(1);
    // cuPDLPx draws doubles on the host. Preserve them on CPU and round only
    // for the Metal float32 trajectory.
    std::normal_distribution<double> normal(0.0, 1.0);
    std::vector<double> eigen_host(static_cast<size_t>(m));
    for (double &value : eigen_host) {
        value = normal(generator);
    }
    auto eigen = mlx_array_from_doubles(eigen_host.data(), m, s_.obj.dtype());
    auto compute_frobenius_norm = [this]() {
        if (s_.sparse_metal_active || s_.sparse_cpu_active) {
            return sparse_frobenius_norm_;
        }
        auto flat_A = mx::reshape(s_.A, {s_.m * s_.n});
        return mlx_norm2(flat_A);
    };
    double eigen_norm = mlx_norm2(eigen);
    if (eigen_norm < 1e-14) {
        return compute_frobenius_norm();
    }
    eigen = eigen / mlx_scalar_like(eigen_norm, eigen);

    std::array<double, convergence_window> sigma_sq_history{};
    double sigma_sq = 0.0;
    bool has_estimate = false;
    bool debug = params_.verbose;

    for (int i = 0; i < params_.sv_max_iter; ++i) {
        s_.singular_value_iterations = i + 1;
        // eigen is normalized on entry. Compute the next power iterate and
        // evaluate its Rayleigh quotient and norm together, requiring one host
        // synchronization per iteration instead of four separate reductions.
        auto y = mat_ATx(eigen);
        auto eigen_new = mat_Ax(y);
        auto sigma_sq_value = mx::sum(eigen * eigen_new);
        auto eigen_new_norm_value = mx::linalg::norm(eigen_new);
        mx::eval(sigma_sq_value, eigen_new_norm_value);
        sigma_sq = mlx_scalar_as_double(sigma_sq_value);
        eigen_norm = mlx_scalar_as_double(eigen_new_norm_value);

        if (!is_finite(sigma_sq) || !is_finite(eigen_norm)) {
            sigma_sq = std::numeric_limits<double>::quiet_NaN();
            break;
        }
        if (eigen_norm < 1e-14) {
            double frobenius_norm = compute_frobenius_norm();
            if (debug) {
                printf("  SV: start vector has no projection through A^T; "
                       "using Frobenius norm %.6f\n",
                       frobenius_norm);
            }
            return frobenius_norm;
        }
        has_estimate = true;

        double relative_sigma_sq_change = std::numeric_limits<double>::infinity();
        if (i >= convergence_window) {
            const double previous_sigma_sq =
                sigma_sq_history[static_cast<size_t>(i % convergence_window)];
            relative_sigma_sq_change =
                std::fabs(sigma_sq - previous_sigma_sq) /
                std::max(std::fabs(sigma_sq), 1e-30);
        }
        sigma_sq_history[static_cast<size_t>(i % convergence_window)] = sigma_sq;

        if (debug && (i < 5 || i % 100 == 0)) {
            printf("  SV iter %3d: sigma_sq=%.6f sigma=%.6f rel_change_%d=%.2e\n",
                   i, sigma_sq, std::sqrt(std::fabs(sigma_sq)),
                   convergence_window, relative_sigma_sq_change);
        }

        if (relative_sigma_sq_change < params_.sv_tol) {
            if (debug) {
                printf("  SV: converged at iter %d (relative sigma^2 change %.2e)\n",
                       i, relative_sigma_sq_change);
            }
            break;
        }
        eigen = eigen_new / mlx_scalar_like(eigen_norm, eigen_new);
    }

    if (!has_estimate || !is_finite(sigma_sq)) {
        double frobenius_norm = compute_frobenius_norm();
        if (debug) {
            printf("  SV: power method produced no finite estimate; "
                   "using Frobenius norm %.6f\n",
                   frobenius_norm);
        }
        return frobenius_norm;
    }

    return std::sqrt(std::fabs(sigma_sq));
}

// ---------------------------------------------------------------------------
// PDHG iteration sub-steps
// ---------------------------------------------------------------------------

void MlxPdlpSolver::mlx_compute_next_primal(int k_offset, bool is_major) {
    int k = s_.inner_count + k_offset;
    double weight = static_cast<double>(k) / static_cast<double>(k + 1);
    double rc = params_.reflection_coefficient;

    if (params_.metal_fused_kernels && s_.sparse_metal_active) {
        // One fused Metal kernel computes A^T*y plus the whole primal step.
        // Evaluation is deferred to mlx_compute_next_dual so a single eval
        // covers both half-step kernels of this iteration.
        const double scalars_host[3] = {s_.step_size_primal, rc, weight};
        auto scalars = mlx_array_from_doubles(scalars_host, 3, mx::float32);
        auto out = fused_primal_step(scalars, is_major);
        s_.x_cur = out[0];
        s_.x_ref = out[1];
        if (is_major) {
            s_.x_pdhg = out[2];
            s_.dual_slack = out[3];
        }
        return;
    }

    // ATy = A^T * y_cur
    s_.ATy = mat_ATx(s_.y_cur);

    // temp = x_cur - step_primal * (obj - ATy)
    auto step_size_primal = mlx_scalar_like(s_.step_size_primal, s_.x_cur);
    auto temp = s_.x_cur - step_size_primal * (s_.obj - s_.ATy);

    // temp_proj = clip(temp, var_lb, var_ub)
    auto temp_proj = mx::clip(temp, s_.var_lb, s_.var_ub);

    // x_ref = 2 * temp_proj - x_cur
    s_.x_ref = mlx_scalar_like(2.0, temp_proj) * temp_proj - s_.x_cur;

    // reflected = rc * x_ref + (1 - rc) * x_cur
    auto reflected = mlx_scalar_like(rc, s_.x_ref) * s_.x_ref +
                     mlx_scalar_like(1.0 - rc, s_.x_cur) * s_.x_cur;

    // x_cur = weight * reflected + (1 - weight) * x_init
    s_.x_cur = mlx_scalar_like(weight, reflected) * reflected +
               mlx_scalar_like(1.0 - weight, s_.x_init) * s_.x_init;

    if (is_major) {
        s_.x_pdhg = temp_proj;
        // dual_slack = (x_pdhg - temp) / step_size_primal
        s_.dual_slack = (s_.x_pdhg - temp) / step_size_primal;
    }

    mx::eval(s_.ATy, s_.x_cur);
    if (is_major) {
        mx::eval(s_.x_pdhg, s_.dual_slack);
    }
}

void MlxPdlpSolver::mlx_compute_next_dual(int k_offset, bool is_major,
                                        bool eval_now) {
    int k = s_.inner_count + k_offset;
    double weight = static_cast<double>(k) / static_cast<double>(k + 1);
    double rc = params_.reflection_coefficient;

    if (params_.metal_fused_kernels && s_.sparse_metal_active) {
        // One fused Metal kernel computes A*x_ref plus the whole dual step.
        // When eval_now is set, one mx::eval materializes both half-step
        // kernels of the current batch; otherwise they stay in the lazy graph.
        const double scalars_host[3] = {s_.step_size_dual, rc, weight};
        auto scalars = mlx_array_from_doubles(scalars_host, 3, mx::float32);
        auto out = fused_dual_step(scalars, is_major);
        s_.y_cur = out[0];
        if (is_major) {
            s_.y_ref = out[1];
            s_.y_pdhg = out[2];
        }
        if (eval_now) {
            if (is_major) {
                mx::eval(s_.x_cur, s_.x_ref, s_.x_pdhg, s_.dual_slack, s_.y_cur,
                         s_.y_ref, s_.y_pdhg);
            } else {
                mx::eval(s_.x_cur, s_.x_ref, s_.y_cur);
            }
        }
        return;
    }

    // Ax = A * x_ref
    s_.Ax = mat_Ax(s_.x_ref);

    // temp = y_cur / step_dual - Ax
    auto step_size_dual = mlx_scalar_like(s_.step_size_dual, s_.y_cur);
    auto temp = s_.y_cur / step_size_dual - s_.Ax;

    // temp_proj = clip(temp, -con_ub, -con_lb)
    auto temp_proj = mx::clip(temp, -s_.con_ub, -s_.con_lb);

    // y_pdhg_iter = (temp - temp_proj) * step_dual
    auto y_pdhg_iter = (temp - temp_proj) * step_size_dual;

    // y_ref = 2 * y_pdhg_iter - y_cur
    s_.y_ref = mlx_scalar_like(2.0, y_pdhg_iter) * y_pdhg_iter - s_.y_cur;

    // reflected = rc * y_ref + (1 - rc) * y_cur
    auto reflected = mlx_scalar_like(rc, s_.y_ref) * s_.y_ref +
                     mlx_scalar_like(1.0 - rc, s_.y_cur) * s_.y_cur;

    // y_cur = weight * reflected + (1 - weight) * y_init
    s_.y_cur = mlx_scalar_like(weight, reflected) * reflected +
               mlx_scalar_like(1.0 - weight, s_.y_init) * s_.y_init;

    if (is_major) {
        s_.y_pdhg = y_pdhg_iter;
    }

    mx::eval(s_.Ax, s_.y_cur);
    if (is_major) {
        mx::eval(s_.y_pdhg);
    }
}

void MlxPdlpSolver::mlx_compute_fixed_point_error() {
    // CUDA reference: delta = reflected - pdhg (NOT pdhg - initial)
    // delta_x = x_ref - x_pdhg
    s_.delta_x = s_.x_ref - s_.x_pdhg;
    // delta_y = y_ref - y_pdhg
    s_.delta_y = s_.y_ref - s_.y_pdhg;

    auto primal_norm_value = mx::linalg::norm(s_.delta_x);
    auto dual_norm_value = mx::linalg::norm(s_.delta_y);

    // Cross term: <A^T * delta_y, delta_x>
    // Compute A^T * delta_y
    auto AT_delta_y = mat_ATx(s_.delta_y);
    auto cross_term_value = mx::sum(AT_delta_y * s_.delta_x);
    mx::eval(primal_norm_value, dual_norm_value, cross_term_value);
    const double primal_norm = mlx_scalar_as_double(primal_norm_value);
    const double dual_norm = mlx_scalar_as_double(dual_norm_value);
    const double cross_term = mlx_scalar_as_double(cross_term_value);

    // movement = primal_norm² * primal_weight + dual_norm² / primal_weight
    double movement =
        primal_norm * primal_norm * s_.primal_weight + dual_norm * dual_norm / s_.primal_weight;

    // interaction = 2 * step_size * cross_term
    double interaction = 2.0 * s_.step_size * cross_term;

    // FP_error = sqrt(movement + interaction)
    // Guard against negative under sqrt (can happen with negative interaction)
    double fp_arg = movement + interaction;
    s_.fixed_point_error = std::sqrt(std::max(fp_arg, 0.0));
}

void MlxPdlpSolver::mlx_compute_residual() {
    // Recompute Ax = A * x_pdhg for residual computation
    s_.Ax = mat_Ax(s_.x_pdhg);
    // Recompute ATy = A^T * y_pdhg
    s_.ATy = mat_ATx(s_.y_pdhg);

    // primal_residual = Ax - clip(Ax, con_lb, con_ub)
    auto clamped_ax = mx::clip(s_.Ax, s_.con_lb, s_.con_ub);
    s_.primal_res = s_.Ax - clamped_ax;

    // The projection multiplier is the complementary reduced-cost certificate
    // produced by PDHG. It has the correct bound-dual signs by construction;
    // stationarity is approximate and is measured explicitly below. Replacing
    // it with projected c-A^T y can destroy the dual objective when finite
    // bounds are merely large sentinels (for example 1e9 in shs1023).
    auto reduced_cost_raw = s_.obj - s_.ATy;
    // Termination and restart trajectories follow PDHG's complementary
    // projection multiplier on both original and presolved models. Using the
    // exact c-A^T y seed here can look stationary long before complementarity
    // has converged, causing aggressive presolve to stop hundreds of thousands
    // of iterations before the reference trajectory. Exact and projection
    // seeds are compared only after postsolve on the original model.
    mx::array certificate_source = s_.dual_slack;
    auto restart_dual_res = reduced_cost_raw - s_.dual_slack;
    auto reduced_cost_lb_adjusted =
        mx::where(s_.var_lb_inf_mask,
                  mx::minimum(certificate_source, mx::zeros_like(certificate_source)),
                  certificate_source);
    auto reduced_cost =
        mx::where(s_.var_ub_inf_mask,
                  mx::maximum(reduced_cost_lb_adjusted,
                              mx::zeros_like(reduced_cost_lb_adjusted)),
                  reduced_cost_lb_adjusted);
    s_.dual_res = reduced_cost_raw - reduced_cost;

    // Undo every preconditioner before taking norms. After geometric-mean,
    // Ruiz, and Pock-Chambolle scaling,
    // and the global bound/objective scaling, the working residuals are
    //
    //   r_p' = con_bound_rescale * diag(con_rescale)^-1 * r_p
    //   r_d' = obj_vec_rescale  * diag(var_rescale)^-1 * r_d.
    //
    // Applying only the global scalar factors here used to under-report
    // nonuniformly scaled residuals and could terminate several times too
    // early on LPFeas instances.
    auto con_bound_rescale = mlx_scalar_like(s_.con_bound_rescale, s_.primal_res);
    auto obj_vec_rescale = mlx_scalar_like(s_.obj_vec_rescale, s_.dual_res);
    auto primal_res_original = s_.primal_res * s_.con_rescale / con_bound_rescale;
    auto dual_res_original = s_.dual_res * s_.var_rescale / obj_vec_rescale;
    auto restart_dual_res_original =
        restart_dual_res * s_.var_rescale / obj_vec_rescale;
    auto primal_residual_norm_value =
        params_.optimality_norm == NORM_TYPE_L_INF
            ? mx::max(mx::abs(primal_res_original))
            : mx::linalg::norm(primal_res_original);
    auto dual_residual_norm_value =
        params_.optimality_norm == NORM_TYPE_L_INF
            ? mx::max(mx::abs(dual_res_original))
            : mx::linalg::norm(dual_res_original);
    auto restart_dual_residual_norm_value =
        params_.optimality_norm == NORM_TYPE_L_INF
            ? mx::max(mx::abs(restart_dual_res_original))
            : mx::linalg::norm(restart_dual_res_original);

    // Relative residuals (CUDA reference: divide by 1.0 + norm)
    double obj_norm = s_.objective_vector_norm;
    double con_norm = s_.constraint_bound_norm;
    // Objective values (CUDA reference: divide by rescaling factors)
    auto primal_obj_value = mx::sum(s_.obj * s_.x_pdhg);

    // Dual objective: sum over constraints of con_lb * y_pdhg or con_ub * y_pdhg.
    // Use current bounds with inf→0 replacement to avoid NaN from inf*0.
    auto con_lb_safe = mx::where(mx::isfinite(s_.con_lb), s_.con_lb, mx::zeros_like(s_.con_lb));
    auto con_ub_safe = mx::where(mx::isfinite(s_.con_ub), s_.con_ub, mx::zeros_like(s_.con_ub));
    auto dual_contrib =
        mx::where(s_.y_pdhg > 0.0, con_lb_safe * s_.y_pdhg, con_ub_safe * s_.y_pdhg);
    auto dual_obj_bnd_value = mx::sum(dual_contrib);

    // Variable-bound contribution from the exported reduced-cost certificate.
    // This remains valid away from complementarity, unlike dot(z, x), and
    // therefore makes the internal gap match the independently audited gap.
    auto var_lb_safe =
        mx::where(mx::isfinite(s_.var_lb), s_.var_lb, mx::zeros_like(s_.var_lb));
    auto var_ub_safe =
        mx::where(mx::isfinite(s_.var_ub), s_.var_ub, mx::zeros_like(s_.var_ub));
    auto dual_var_contrib = mx::where(reduced_cost > 0.0, var_lb_safe * reduced_cost,
                                      var_ub_safe * reduced_cost);
    auto dual_var_obj_value = mx::sum(dual_var_contrib);

    // All block-level residual and objective reductions share one evaluation.
    // Reading the already-materialized scalars below does not trigger further
    // device synchronization.
    mx::eval(primal_residual_norm_value, dual_residual_norm_value,
             restart_dual_residual_norm_value, primal_obj_value,
             dual_obj_bnd_value, dual_var_obj_value);
    s_.absolute_primal_residual =
        mlx_scalar_as_double(primal_residual_norm_value);
    s_.absolute_dual_residual =
        mlx_scalar_as_double(dual_residual_norm_value);
    s_.relative_primal_residual = s_.absolute_primal_residual / (1.0 + con_norm);
    s_.relative_dual_residual = s_.absolute_dual_residual / (1.0 + obj_norm);
    const double restart_absolute_dual_residual =
        mlx_scalar_as_double(restart_dual_residual_norm_value);
    s_.restart_relative_dual_residual =
        restart_absolute_dual_residual / (1.0 + obj_norm);

    double primal_obj = mlx_scalar_as_double(primal_obj_value);
    primal_obj = primal_obj / (s_.con_bound_rescale * s_.obj_vec_rescale) +
                 s_.objective_constant;
    const double dual_obj_bnd = mlx_scalar_as_double(dual_obj_bnd_value);
    const double dual_var_obj = mlx_scalar_as_double(dual_var_obj_value);
    double dual_obj =
        (dual_obj_bnd + dual_var_obj) / (s_.con_bound_rescale * s_.obj_vec_rescale) +
        s_.objective_constant;

    s_.primal_objective_value = primal_obj;
    s_.dual_objective_value = dual_obj;
    s_.objective_gap = std::fabs(primal_obj - dual_obj);
    s_.relative_objective_gap =
        s_.objective_gap / (1.0 + std::fabs(primal_obj) + std::fabs(dual_obj));
}

void MlxPdlpSolver::mlx_save_best_iterate() {
    const double kkt_error = std::max(
        {s_.relative_primal_residual, s_.relative_dual_residual, s_.relative_objective_gap});
    const double feasibility_error =
        std::max(s_.relative_primal_residual, s_.relative_dual_residual);
    if (!std::isfinite(kkt_error) || !std::isfinite(feasibility_error)) {
        return;
    }

    bool improves_checkpoint = kkt_error < s_.best_relative_kkt_error;
    // Large finite variable bounds can turn a tiny stationarity error into a
    // dual objective many orders of magnitude away from the primal objective.
    // In that regime the normalized gap rounds to (or oscillates immediately
    // below) one and is not useful for ordering checkpoints. Treat gaps within
    // fp32 noise of one as tied and retain the more feasible point. As soon as
    // either checkpoint has an informative gap, use the strict full-KKT order.
    constexpr double saturated_gap_floor = 0.99;
    constexpr double saturated_gap_tie = 1e-6;
    if (s_.best_iteration >= 0 && kkt_error >= saturated_gap_floor &&
        s_.best_relative_kkt_error >= saturated_gap_floor &&
        std::fabs(kkt_error - s_.best_relative_kkt_error) <= saturated_gap_tie) {
        improves_checkpoint = feasibility_error < s_.best_relative_feasibility_error;
    }
    if (!improves_checkpoint) {
        return;
    }

    s_.best_relative_kkt_error = kkt_error;
    s_.best_relative_feasibility_error = feasibility_error;
    s_.best_iteration = s_.total_count;
    s_.x_best = s_.x_pdhg;
    s_.y_best = s_.y_pdhg;
    s_.dual_slack_best = s_.dual_slack;
    mx::eval(s_.x_best, s_.y_best, s_.dual_slack_best);
}

void MlxPdlpSolver::mlx_restore_best_iterate() {
    if (s_.best_iteration < 0) {
        return;
    }
    s_.x_pdhg = s_.x_best;
    s_.y_pdhg = s_.y_best;
    s_.dual_slack = s_.dual_slack_best;
    mx::eval(s_.x_pdhg, s_.y_pdhg, s_.dual_slack);
}

void MlxPdlpSolver::mlx_primal_feasibility_polish() {
    const auto &criteria = params_.termination_criteria;
    const double host_primal_gate =
        std::max(1e-2, 100.0 * criteria.eps_feasible_relative);
    const bool host_can_continue =
        params_.host_double_polishing &&
        params_.host_double_polishing_iteration_limit > 0 &&
        params_.host_double_polishing_time_sec_limit > 0.0 &&
        std::isfinite(s_.relative_primal_residual) &&
        std::isfinite(s_.relative_dual_residual) &&
        std::isfinite(s_.relative_objective_gap) &&
        s_.relative_primal_residual <= host_primal_gate &&
        s_.relative_dual_residual <= 1.0;
    if (!params_.feasibility_polishing ||
        s_.relative_primal_residual <= criteria.eps_feas_polish_relative ||
        host_can_continue) {
        if (params_.verbose && params_.feasibility_polishing && host_can_continue &&
            s_.relative_primal_residual > criteria.eps_feas_polish_relative) {
            printf("  primal polish: deferred to bounded host-double correction\n");
        }
        return;
    }

    // The polish phases share the solve-level wall-clock budget: cuPDLPx's
    // check_feas_polishing_termination_criteria measures total_time_sec from
    // the ORIGINAL state's start time, so main loop plus both polish phases
    // together stay within time_sec_limit instead of each phase receiving its
    // own full copy of the budget.
    if (elapsed_seconds(s_.start_time) >= criteria.time_sec_limit) {
        if (params_.verbose) {
            printf("  primal polish: skipped, shared time budget exhausted\n");
        }
        return;
    }

    const auto polish_start = SteadyClock::now();
    const auto original_obj = s_.obj;
    const auto baseline_x = s_.x_pdhg;
    const auto baseline_y = s_.y_pdhg;
    const auto baseline_dual_slack = s_.dual_slack;
    const double baseline_merit = std::max(
        {s_.relative_primal_residual, s_.relative_dual_residual, s_.relative_objective_gap});
    const termination_reason_t baseline_reason = s_.termination_reason;
    const int main_total_count = s_.total_count;
    const int main_inner_count = s_.inner_count;
    const double main_primal_weight = s_.primal_weight;
    const double main_best_primal_weight = s_.best_primal_weight;
    const double main_step_size_primal = s_.step_size_primal;
    const double main_step_size_dual = s_.step_size_dual;
    const double main_fixed_point_error = s_.fixed_point_error;
    const double main_initial_fixed_point_error = s_.initial_fixed_point_error;
    const double main_last_trial_fixed_point_error = s_.last_trial_fixed_point_error;
    if (params_.verbose) {
        printf("  primal polish: starting from rel-primal %.6e (target %.1e)\n",
               s_.relative_primal_residual, criteria.eps_feas_polish_relative);
    }

    // Follow cuPDLPx's primal-polishing formulation: remove the objective,
    // retain the incumbent primal point, reset the dual point, and solve only
    // for constraint feasibility. Match the reference implementation by giving
    // the polish phase its own iteration counter and wall-clock budget.
    s_.obj = mx::zeros_like(original_obj);
    s_.x_init = baseline_x;
    s_.x_cur = baseline_x;
    s_.x_pdhg = baseline_x;
    s_.y_init = mx::zeros_like(baseline_y);
    s_.y_cur = mx::zeros_like(baseline_y);
    s_.y_pdhg = mx::zeros_like(baseline_y);
    s_.dual_slack = mx::zeros_like(baseline_dual_slack);
    s_.inner_count = 0;
    s_.total_count = 0;
    s_.primal_weight = params_.bound_objective_rescaling
                           ? 1.0
                           : (s_.objective_vector_norm + 1.0) /
                                 (s_.constraint_bound_norm + 1.0);
    s_.best_primal_weight = s_.primal_weight;
    s_.step_size_primal = s_.step_size / s_.primal_weight;
    s_.step_size_dual = s_.step_size * s_.primal_weight;
    s_.fixed_point_error = inf();
    s_.initial_fixed_point_error = inf();
    s_.last_trial_fixed_point_error = inf();
    mx::eval(s_.obj, s_.x_init, s_.x_cur, s_.x_pdhg, s_.y_init, s_.y_cur, s_.y_pdhg,
             s_.dual_slack);

    auto compute_primal_fixed_point_error = [this]() {
        s_.delta_x = s_.x_ref - s_.x_pdhg;
        mx::eval(s_.delta_x);
        const double primal_norm = mlx_norm2(s_.delta_x);
        // The reference compares this squared weighted distance directly.
        s_.fixed_point_error = primal_norm * primal_norm * s_.primal_weight;
    };
    auto compute_primal_residual = [this]() {
        s_.Ax = mat_Ax(s_.x_pdhg);
        s_.primal_res = s_.Ax - mx::clip(s_.Ax, s_.con_lb, s_.con_ub);
        auto original_residual =
            s_.primal_res * s_.con_rescale /
            mlx_scalar_like(s_.con_bound_rescale, s_.primal_res);
        mx::eval(s_.primal_res, original_residual);
        s_.absolute_primal_residual =
            params_.optimality_norm == NORM_TYPE_L_INF ? mlx_norm_inf(original_residual)
                                                       : mlx_norm2(original_residual);
        s_.relative_primal_residual =
            s_.absolute_primal_residual / (1.0 + s_.constraint_bound_norm);
    };

    auto polished_x = baseline_x;
    double best_primal_residual = s_.relative_primal_residual;
    bool do_restart = false;
    const int eval_freq = params_.termination_evaluation_frequency;
    while (s_.total_count < criteria.iteration_limit &&
           elapsed_seconds(s_.start_time) < criteria.time_sec_limit) {
        mlx_compute_next_primal(1, true);
        mlx_compute_next_dual(1, true);
        if (do_restart) {
            compute_primal_fixed_point_error();
            s_.initial_fixed_point_error = s_.fixed_point_error;
            do_restart = false;
        }
        for (int i = 2; i <= eval_freq - 1; ++i) {
            mlx_compute_next_primal(i, false);
            mlx_compute_next_dual(i, false);
        }
        mlx_compute_next_primal(eval_freq, true);
        mlx_compute_next_dual(eval_freq, true);
        compute_primal_fixed_point_error();
        compute_primal_residual();
        s_.inner_count += eval_freq;
        s_.total_count += eval_freq;

        if (std::isfinite(s_.relative_primal_residual) &&
            s_.relative_primal_residual < best_primal_residual) {
            best_primal_residual = s_.relative_primal_residual;
            polished_x = s_.x_pdhg;
            mx::eval(polished_x);
        }
        if (best_primal_residual <= criteria.eps_feas_polish_relative) {
            break;
        }

        const auto &restart = params_.restart_params;
        if (s_.total_count == eval_freq) {
            do_restart = true;
        } else {
            do_restart = s_.fixed_point_error <=
                             restart.sufficient_reduction_for_restart *
                                 s_.initial_fixed_point_error ||
                         (s_.fixed_point_error <=
                              restart.necessary_reduction_for_restart *
                                  s_.initial_fixed_point_error &&
                          s_.fixed_point_error > s_.last_trial_fixed_point_error) ||
                         s_.inner_count >= restart.artificial_restart_threshold * s_.total_count;
        }
        s_.last_trial_fixed_point_error = s_.fixed_point_error;
        if (do_restart) {
            // Primal polishing deliberately leaves the dual trajectory alone.
            s_.x_init = s_.x_pdhg;
            s_.x_cur = s_.x_pdhg;
            s_.inner_count = 0;
            s_.last_trial_fixed_point_error = inf();
            mx::eval(s_.x_init, s_.x_cur);
        }
    }

    const int polish_iterations = s_.total_count;
    s_.obj = original_obj;
    s_.primal_weight = main_primal_weight;
    s_.best_primal_weight = main_best_primal_weight;
    s_.step_size_primal = main_step_size_primal;
    s_.step_size_dual = main_step_size_dual;
    s_.fixed_point_error = main_fixed_point_error;
    s_.initial_fixed_point_error = main_initial_fixed_point_error;
    s_.last_trial_fixed_point_error = main_last_trial_fixed_point_error;
    s_.total_count = main_total_count;
    s_.inner_count = main_inner_count;
    s_.termination_reason = baseline_reason;
    s_.y_pdhg = baseline_y;
    s_.dual_slack = baseline_dual_slack;
    mx::eval(s_.obj, s_.y_pdhg, s_.dual_slack);

    // Feasibility polishing ignores the objective, so its endpoint can move
    // farther than needed along a good feasibility direction. Search the
    // convex segment back to the incumbent and accept only the point with the
    // best full KKT merit. Bounds and row constraints are convex, and this
    // safeguard also protects objective quality in fp32.
    auto combined_x = baseline_x;
    double combined_merit = baseline_merit;
    double combined_alpha = 0.0;
    constexpr int blend_intervals = 64;
    for (int trial = 1; trial <= blend_intervals; ++trial) {
        const double alpha = static_cast<double>(trial) / blend_intervals;
        s_.x_pdhg = mlx_scalar_like(1.0 - alpha, baseline_x) * baseline_x +
                    mlx_scalar_like(alpha, polished_x) * polished_x;
        mx::eval(s_.x_pdhg);
        mlx_compute_residual();
        const double merit = std::max(
            {s_.relative_primal_residual, s_.relative_dual_residual, s_.relative_objective_gap});
        if (std::isfinite(merit) && merit < combined_merit) {
            combined_merit = merit;
            combined_alpha = alpha;
            combined_x = s_.x_pdhg;
            mx::eval(combined_x);
        }
    }
    s_.x_pdhg = combined_x;
    mx::eval(s_.x_pdhg);
    mlx_compute_residual();

    const bool accept_polish = combined_alpha > 0.0;
    if (params_.verbose) {
        printf("  primal polish: iter=%d best-feas=%.6e alpha=%.6f "
               "full=(%.6e, %.6e, %.6e) %s\n",
               polish_iterations, best_primal_residual, combined_alpha,
               s_.relative_primal_residual, s_.relative_dual_residual, s_.relative_objective_gap,
               accept_polish ? "accepted" : "rejected");
    }
    if (accept_polish) {
        s_.x_best = s_.x_pdhg;
        s_.y_best = s_.y_pdhg;
        s_.dual_slack_best = s_.dual_slack;
        s_.best_relative_kkt_error = combined_merit;
        s_.best_relative_feasibility_error =
            std::max(s_.relative_primal_residual, s_.relative_dual_residual);
        mx::eval(s_.x_best, s_.y_best, s_.dual_slack_best);
        if (s_.relative_primal_residual < criteria.eps_feasible_relative &&
            s_.relative_dual_residual < criteria.eps_feasible_relative &&
            std::fabs(s_.relative_objective_gap) < criteria.eps_optimal_relative) {
            s_.termination_reason = TERMINATION_REASON_OPTIMAL;
        }
    } else {
        s_.x_pdhg = baseline_x;
        s_.y_pdhg = baseline_y;
        s_.dual_slack = baseline_dual_slack;
        mx::eval(s_.x_pdhg, s_.y_pdhg, s_.dual_slack);
        mlx_compute_residual();
    }

    s_.x_init = s_.x_pdhg;
    s_.x_cur = s_.x_pdhg;
    s_.y_init = s_.y_pdhg;
    s_.y_cur = s_.y_pdhg;
    mx::eval(s_.x_init, s_.x_cur, s_.y_init, s_.y_cur);
    s_.feasibility_iteration += polish_iterations;
    s_.feasibility_polishing_time_sec += elapsed_seconds(polish_start);
}

void MlxPdlpSolver::mlx_dual_feasibility_polish() {
    const auto &criteria = params_.termination_criteria;
    const double host_primal_gate =
        std::max(1e-2, 100.0 * criteria.eps_feasible_relative);
    const bool host_can_continue =
        params_.host_double_polishing &&
        params_.host_double_polishing_iteration_limit > 0 &&
        params_.host_double_polishing_time_sec_limit > 0.0 &&
        std::isfinite(s_.relative_primal_residual) &&
        std::isfinite(s_.relative_dual_residual) &&
        std::isfinite(s_.relative_objective_gap) &&
        s_.relative_primal_residual <= host_primal_gate &&
        s_.relative_dual_residual <= 1.0;
    if (!params_.feasibility_polishing ||
        s_.relative_dual_residual <= criteria.eps_feas_polish_relative ||
        host_can_continue) {
        if (params_.verbose && params_.feasibility_polishing && host_can_continue &&
            s_.relative_dual_residual > criteria.eps_feas_polish_relative) {
            printf("  dual polish: deferred to bounded host-double correction\n");
        }
        return;
    }

    // Share the solve-level wall-clock budget exactly like the primal polish
    // phase: cuPDLPx measures polish termination time from the ORIGINAL
    // state's start, so the main loop and both polish phases together stay
    // within time_sec_limit.
    if (elapsed_seconds(s_.start_time) >= criteria.time_sec_limit) {
        if (params_.verbose) {
            printf("  dual polish: skipped, shared time budget exhausted\n");
        }
        return;
    }

    const auto polish_start = SteadyClock::now();
    const auto original_var_lb = s_.var_lb;
    const auto original_var_ub = s_.var_ub;
    const auto original_con_lb = s_.con_lb;
    const auto original_con_ub = s_.con_ub;
    const auto baseline_x = s_.x_pdhg;
    const auto baseline_y = s_.y_pdhg;
    const auto baseline_dual_slack = s_.dual_slack;
    const double baseline_merit = std::max(
        {s_.relative_primal_residual, s_.relative_dual_residual, s_.relative_objective_gap});
    const termination_reason_t baseline_reason = s_.termination_reason;
    const int main_total_count = s_.total_count;
    const int main_inner_count = s_.inner_count;
    const double main_primal_weight = s_.primal_weight;
    const double main_best_primal_weight = s_.best_primal_weight;
    const double main_step_size_primal = s_.step_size_primal;
    const double main_step_size_dual = s_.step_size_dual;
    const double main_fixed_point_error = s_.fixed_point_error;
    const double main_initial_fixed_point_error = s_.initial_fixed_point_error;
    const double main_last_trial_fixed_point_error = s_.last_trial_fixed_point_error;
    if (params_.verbose) {
        printf("  dual polish: starting from rel-dual %.6e (target %.1e)\n",
               s_.relative_dual_residual, criteria.eps_feas_polish_relative);
    }

    // cuPDLPx's dual-feasibility problem keeps the objective and the finite/
    // infinite bound pattern, but replaces every finite variable and row bound
    // by zero. Start its primal surrogate at zero and retain the incumbent dual.
    s_.var_lb = mx::where(mx::isfinite(original_var_lb), mx::zeros_like(original_var_lb),
                          original_var_lb);
    s_.var_ub = mx::where(mx::isfinite(original_var_ub), mx::zeros_like(original_var_ub),
                          original_var_ub);
    s_.con_lb = mx::where(mx::isfinite(original_con_lb), mx::zeros_like(original_con_lb),
                          original_con_lb);
    s_.con_ub = mx::where(mx::isfinite(original_con_ub), mx::zeros_like(original_con_ub),
                          original_con_ub);
    s_.x_init = mx::zeros_like(baseline_x);
    s_.x_cur = mx::zeros_like(baseline_x);
    s_.x_pdhg = mx::zeros_like(baseline_x);
    s_.y_init = baseline_y;
    s_.y_cur = baseline_y;
    s_.y_pdhg = baseline_y;
    s_.dual_slack = baseline_dual_slack;
    s_.inner_count = 0;
    s_.total_count = 0;
    s_.primal_weight = params_.bound_objective_rescaling
                           ? 1.0
                           : (s_.objective_vector_norm + 1.0) /
                                 (s_.constraint_bound_norm + 1.0);
    s_.best_primal_weight = s_.primal_weight;
    s_.step_size_primal = s_.step_size / s_.primal_weight;
    s_.step_size_dual = s_.step_size * s_.primal_weight;
    s_.fixed_point_error = inf();
    s_.initial_fixed_point_error = inf();
    s_.last_trial_fixed_point_error = inf();
    mx::eval(s_.var_lb, s_.var_ub, s_.con_lb, s_.con_ub, s_.x_init, s_.x_cur, s_.x_pdhg,
             s_.y_init, s_.y_cur, s_.y_pdhg, s_.dual_slack);

    auto compute_dual_fixed_point_error = [this]() {
        s_.delta_y = s_.y_ref - s_.y_pdhg;
        mx::eval(s_.delta_y);
        const double dual_norm = mlx_norm2(s_.delta_y);
        s_.fixed_point_error = dual_norm * dual_norm / s_.primal_weight;
    };
    auto compute_dual_residual = [this]() {
        s_.ATy = mat_ATx(s_.y_pdhg);
        auto reduced_cost_raw = s_.obj - s_.ATy;
        mx::array certificate_source = s_.dual_slack;
        auto reduced_cost_lb_adjusted =
            mx::where(s_.var_lb_inf_mask,
                      mx::minimum(certificate_source,
                                  mx::zeros_like(certificate_source)),
                      certificate_source);
        auto reduced_cost =
            mx::where(s_.var_ub_inf_mask,
                      mx::maximum(reduced_cost_lb_adjusted,
                                  mx::zeros_like(reduced_cost_lb_adjusted)),
                      reduced_cost_lb_adjusted);
        s_.dual_res = reduced_cost_raw - reduced_cost;
        auto original_residual =
            s_.dual_res * s_.var_rescale /
            mlx_scalar_like(s_.obj_vec_rescale, s_.dual_res);
        mx::eval(s_.ATy, s_.dual_res, original_residual);
        s_.absolute_dual_residual =
            params_.optimality_norm == NORM_TYPE_L_INF ? mlx_norm_inf(original_residual)
                                                       : mlx_norm2(original_residual);
        s_.relative_dual_residual =
            s_.absolute_dual_residual / (1.0 + s_.objective_vector_norm);
    };

    auto polished_y = baseline_y;
    auto polished_dual_slack = baseline_dual_slack;
    double best_dual_residual = s_.relative_dual_residual;
    bool do_restart = false;
    const int eval_freq = params_.termination_evaluation_frequency;
    while (s_.total_count < criteria.iteration_limit &&
           elapsed_seconds(s_.start_time) < criteria.time_sec_limit) {
        mlx_compute_next_primal(1, true);
        mlx_compute_next_dual(1, true);
        if (do_restart) {
            compute_dual_fixed_point_error();
            s_.initial_fixed_point_error = s_.fixed_point_error;
            do_restart = false;
        }
        for (int i = 2; i <= eval_freq - 1; ++i) {
            mlx_compute_next_primal(i, false);
            mlx_compute_next_dual(i, false);
        }
        mlx_compute_next_primal(eval_freq, true);
        mlx_compute_next_dual(eval_freq, true);
        compute_dual_fixed_point_error();
        compute_dual_residual();
        s_.inner_count += eval_freq;
        s_.total_count += eval_freq;

        if (std::isfinite(s_.relative_dual_residual) &&
            s_.relative_dual_residual < best_dual_residual) {
            best_dual_residual = s_.relative_dual_residual;
            polished_y = s_.y_pdhg;
            polished_dual_slack = s_.dual_slack;
            mx::eval(polished_y, polished_dual_slack);
        }
        if (best_dual_residual <= criteria.eps_feas_polish_relative)
            break;

        const auto &restart = params_.restart_params;
        if (s_.total_count == eval_freq) {
            do_restart = true;
        } else {
            do_restart = s_.fixed_point_error <=
                             restart.sufficient_reduction_for_restart *
                                 s_.initial_fixed_point_error ||
                         (s_.fixed_point_error <=
                              restart.necessary_reduction_for_restart *
                                  s_.initial_fixed_point_error &&
                          s_.fixed_point_error > s_.last_trial_fixed_point_error) ||
                         s_.inner_count >= restart.artificial_restart_threshold * s_.total_count;
        }
        s_.last_trial_fixed_point_error = s_.fixed_point_error;
        if (do_restart) {
            s_.y_init = s_.y_pdhg;
            s_.y_cur = s_.y_pdhg;
            s_.inner_count = 0;
            s_.last_trial_fixed_point_error = inf();
            mx::eval(s_.y_init, s_.y_cur);
        }
    }

    const int polish_iterations = s_.total_count;
    s_.var_lb = original_var_lb;
    s_.var_ub = original_var_ub;
    s_.con_lb = original_con_lb;
    s_.con_ub = original_con_ub;
    s_.x_pdhg = baseline_x;
    s_.y_pdhg = baseline_y;
    s_.dual_slack = baseline_dual_slack;
    s_.primal_weight = main_primal_weight;
    s_.best_primal_weight = main_best_primal_weight;
    s_.step_size_primal = main_step_size_primal;
    s_.step_size_dual = main_step_size_dual;
    s_.fixed_point_error = main_fixed_point_error;
    s_.initial_fixed_point_error = main_initial_fixed_point_error;
    s_.last_trial_fixed_point_error = main_last_trial_fixed_point_error;
    s_.total_count = main_total_count;
    s_.inner_count = main_inner_count;
    s_.termination_reason = baseline_reason;
    mx::eval(s_.var_lb, s_.var_ub, s_.con_lb, s_.con_ub, s_.x_pdhg, s_.y_pdhg,
             s_.dual_slack);

    // As with primal polishing, search the segment back to the incumbent and
    // accept only a point that improves the complete primal/dual/gap KKT merit.
    auto combined_y = baseline_y;
    double combined_merit = baseline_merit;
    double combined_alpha = 0.0;
    constexpr int blend_intervals = 64;
    for (int trial = 1; trial <= blend_intervals; ++trial) {
        const double alpha = static_cast<double>(trial) / blend_intervals;
        s_.y_pdhg = mlx_scalar_like(1.0 - alpha, baseline_y) * baseline_y +
                    mlx_scalar_like(alpha, polished_y) * polished_y;
        s_.dual_slack =
            mlx_scalar_like(1.0 - alpha, baseline_dual_slack) *
                baseline_dual_slack +
            mlx_scalar_like(alpha, polished_dual_slack) * polished_dual_slack;
        mx::eval(s_.y_pdhg, s_.dual_slack);
        mlx_compute_residual();
        const double merit = std::max(
            {s_.relative_primal_residual, s_.relative_dual_residual, s_.relative_objective_gap});
        if (std::isfinite(merit) && merit < combined_merit) {
            combined_merit = merit;
            combined_alpha = alpha;
            combined_y = s_.y_pdhg;
            mx::eval(combined_y);
        }
    }
    s_.y_pdhg = combined_y;
    s_.dual_slack =
        mlx_scalar_like(1.0 - combined_alpha, baseline_dual_slack) *
            baseline_dual_slack +
        mlx_scalar_like(combined_alpha, polished_dual_slack) *
            polished_dual_slack;
    mx::eval(s_.y_pdhg, s_.dual_slack);
    mlx_compute_residual();

    const bool accept_polish = combined_alpha > 0.0;
    if (params_.verbose) {
        printf("  dual polish: iter=%d best-feas=%.6e alpha=%.6f "
               "full=(%.6e, %.6e, %.6e) %s\n",
               polish_iterations, best_dual_residual, combined_alpha,
               s_.relative_primal_residual, s_.relative_dual_residual,
               s_.relative_objective_gap, accept_polish ? "accepted" : "rejected");
    }
    if (accept_polish) {
        s_.x_best = s_.x_pdhg;
        s_.y_best = s_.y_pdhg;
        s_.dual_slack_best = s_.dual_slack;
        s_.best_relative_kkt_error = combined_merit;
        s_.best_relative_feasibility_error =
            std::max(s_.relative_primal_residual, s_.relative_dual_residual);
        mx::eval(s_.x_best, s_.y_best, s_.dual_slack_best);
        if (s_.relative_primal_residual < criteria.eps_feasible_relative &&
            s_.relative_dual_residual < criteria.eps_feasible_relative &&
            std::fabs(s_.relative_objective_gap) < criteria.eps_optimal_relative) {
            s_.termination_reason = TERMINATION_REASON_OPTIMAL;
        }
    } else {
        s_.x_pdhg = baseline_x;
        s_.y_pdhg = baseline_y;
        s_.dual_slack = baseline_dual_slack;
        mx::eval(s_.x_pdhg, s_.y_pdhg, s_.dual_slack);
        mlx_compute_residual();
    }

    s_.x_init = s_.x_pdhg;
    s_.x_cur = s_.x_pdhg;
    s_.y_init = s_.y_pdhg;
    s_.y_cur = s_.y_pdhg;
    mx::eval(s_.x_init, s_.x_cur, s_.y_init, s_.y_cur);
    s_.feasibility_iteration += polish_iterations;
    s_.feasibility_polishing_time_sec += elapsed_seconds(polish_start);
}

void MlxPdlpSolver::mlx_compute_infeasibility_information() {
    ++s_.infeasibility_check_count;

    // Without constraints there can be no primal-infeasibility certificate and
    // without variables no dual-infeasibility certificate; the rays also
    // degenerate to empty arrays that MLX reductions reject.
    if (s_.m == 0 || s_.n == 0) {
        working_dual_ray_objective_ = 0.0;
        working_primal_ray_objective_ = 0.0;
        s_.max_primal_ray_infeasibility = 0.0;
        s_.max_dual_ray_infeasibility = 0.0;
        s_.primal_ray_linear_objective = 0.0;
        s_.dual_ray_objective = 0.0;
        return;
    }

    // Infeasibility certificates via Farkas separation on the box-constrained
    // formulation, computed on the sign-projected fixed-point deltas. The
    // residual measures recession-cone membership, and the objective is the
    // full separation gap, so the certificate is valid (and silent) on
    // feasible problems even when every bound mask is vacuous.
    //
    // Primal ray r (dual-infeasibility certificate): r must lie in the
    // recession cone of the variable box and A r in the recession cone of the
    // constraint box, with c^T r < 0. For a box [l, u] the recession cone is
    //   r_j >= 0 if (u_j = +inf and l_j > -inf),
    //   r_j <= 0 if (l_j = -inf and u_j < +inf),
    //   r_j  = 0 if both bounds are finite, free if both are infinite.
    auto primal_ray_inf_norm_value = mx::max(mx::abs(s_.delta_x));
    auto primal_ray_denominator =
        mx::where(primal_ray_inf_norm_value > 0.0, primal_ray_inf_norm_value,
                  mx::ones_like(primal_ray_inf_norm_value));
    auto primal_ray = s_.delta_x / primal_ray_denominator;
    auto A_pr = mat_Ax(primal_ray);
    auto primal_ray_objective_value = mx::sum(s_.obj * primal_ray);

    auto var_upper_only = s_.var_ub_inf_mask * (1.0 - s_.var_lb_inf_mask);
    auto var_lower_only = s_.var_lb_inf_mask * (1.0 - s_.var_ub_inf_mask);
    auto var_both_finite = (1.0 - s_.var_lb_inf_mask) * (1.0 - s_.var_ub_inf_mask);
    auto r_var_viol =
        mx::maximum(-primal_ray, mx::zeros_like(primal_ray)) * var_upper_only +
        mx::maximum(primal_ray, mx::zeros_like(primal_ray)) * var_lower_only +
        mx::abs(primal_ray) * var_both_finite;
    auto con_upper_only = s_.con_ub_inf_mask * (1.0 - s_.con_lb_inf_mask);
    auto con_lower_only = s_.con_lb_inf_mask * (1.0 - s_.con_ub_inf_mask);
    auto con_both_finite = (1.0 - s_.con_lb_inf_mask) * (1.0 - s_.con_ub_inf_mask);
    auto r_con_viol =
        mx::maximum(-A_pr, mx::zeros_like(A_pr)) * con_upper_only +
        mx::maximum(A_pr, mx::zeros_like(A_pr)) * con_lower_only +
        mx::abs(A_pr) * con_both_finite;
    auto r_var_viol_norm_value = mx::max(mx::abs(r_var_viol));
    auto r_con_viol_norm_value = mx::max(mx::abs(r_con_viol));

    // Dual ray y (primal-infeasibility certificate): the separation gap
    //   min_s y^T s - max_x (A^T y)^T x
    // over the constraint box s and variable box x must be strictly positive.
    // Finiteness of min_s requires y_i > 0 only where con_lb_i is finite and
    // y_i < 0 only where con_ub_i is finite; finiteness of max_x requires
    // (A^T y)_j > 0 only where var_ub_j is finite and < 0 only where
    // var_lb_j is finite. Feasible problems satisfy the gap <= 0 for every y
    // by weak duality, so this test cannot fire on them in exact arithmetic.
    auto dual_ray_inf_norm_value = mx::max(mx::abs(s_.delta_y));
    auto dual_ray_denominator =
        mx::where(dual_ray_inf_norm_value > 0.0, dual_ray_inf_norm_value,
                  mx::ones_like(dual_ray_inf_norm_value));
    auto dual_ray = s_.delta_y / dual_ray_denominator;
    auto AT_dr = mat_ATx(dual_ray);

    auto y_con_viol = mx::maximum(dual_ray, mx::zeros_like(dual_ray)) * s_.con_lb_inf_mask +
                      mx::maximum(-dual_ray, mx::zeros_like(dual_ray)) * s_.con_ub_inf_mask;
    auto y_var_viol = mx::maximum(AT_dr, mx::zeros_like(AT_dr)) * s_.var_ub_inf_mask +
                      mx::maximum(-AT_dr, mx::zeros_like(AT_dr)) * s_.var_lb_inf_mask;
    auto y_con_viol_norm_value = mx::max(mx::abs(y_con_viol));
    auto y_var_viol_norm_value = mx::max(mx::abs(y_var_viol));

    // Finite-safe bound values must be derived from the CURRENT scaled bounds:
    // the cached finite-safe arrays are only refreshed after Ruiz and go stale
    // after Pock-Chambolle and bound/objective scaling. Geometric-mean scaling
    // runs before that refresh.
    auto con_lb_safe = mx::where(mx::isfinite(s_.con_lb), s_.con_lb, mx::zeros_like(s_.con_lb));
    auto con_ub_safe = mx::where(mx::isfinite(s_.con_ub), s_.con_ub, mx::zeros_like(s_.con_ub));
    auto var_lb_safe = mx::where(mx::isfinite(s_.var_lb), s_.var_lb, mx::zeros_like(s_.var_lb));
    auto var_ub_safe = mx::where(mx::isfinite(s_.var_ub), s_.var_ub, mx::zeros_like(s_.var_ub));
    auto min_s = mx::maximum(dual_ray, mx::zeros_like(dual_ray)) * con_lb_safe +
                 mx::minimum(dual_ray, mx::zeros_like(dual_ray)) * con_ub_safe;
    auto max_x = mx::maximum(AT_dr, mx::zeros_like(AT_dr)) * var_ub_safe +
                 mx::minimum(AT_dr, mx::zeros_like(AT_dr)) * var_lb_safe;
    auto dual_ray_objective_value = mx::sum(min_s) - mx::sum(max_x);

    // Materialize both rays, both sparse matvecs, and every certificate
    // reduction together. Reading the six scalar results below then costs
    // one device synchronization instead of a sequence of round trips.
    mx::eval(primal_ray_objective_value, r_var_viol_norm_value,
             r_con_viol_norm_value, y_con_viol_norm_value,
             y_var_viol_norm_value, dual_ray_objective_value);
    working_primal_ray_objective_ =
        mlx_scalar_as_double(primal_ray_objective_value);
    s_.primal_ray_linear_objective =
        working_primal_ray_objective_ / (s_.con_bound_rescale * s_.obj_vec_rescale);
    s_.max_primal_ray_infeasibility =
        std::max(mlx_scalar_as_double(r_var_viol_norm_value),
                 mlx_scalar_as_double(r_con_viol_norm_value));
    s_.max_dual_ray_infeasibility =
        std::max(mlx_scalar_as_double(y_con_viol_norm_value),
                 mlx_scalar_as_double(y_var_viol_norm_value));
    working_dual_ray_objective_ = mlx_scalar_as_double(dual_ray_objective_value);
    s_.dual_ray_objective =
        working_dual_ray_objective_ / (s_.con_bound_rescale * s_.obj_vec_rescale);
}

// ---------------------------------------------------------------------------
// Restart logic
// ---------------------------------------------------------------------------

void MlxPdlpSolver::mlx_perform_restart() {
    // CUDA reference: compute delta = pdhg - initial for distance-based PID
    s_.delta_x = s_.x_pdhg - s_.x_init;
    s_.delta_y = s_.y_pdhg - s_.y_init;
    auto primal_dist_value = mx::linalg::norm(s_.delta_x);
    auto dual_dist_value = mx::linalg::norm(s_.delta_y);
    mx::eval(primal_dist_value, dual_dist_value);
    const double primal_dist = mlx_scalar_as_double(primal_dist_value);
    const double dual_dist = mlx_scalar_as_double(dual_dist_value);

    double ratio_infeas =
        s_.restart_relative_dual_residual / s_.relative_primal_residual;
    const double old_primal_weight = s_.primal_weight;

    if (params_.restart_policy == 1) {
        // HPR-LP sigma update (src/HPRLP.cu update_sigma). In this solver's
        // symmetric step coordinates the HPR movement-ratio target
        // sigma = (||dx||/||dy||)/sqrt(lambda_max) maps to the primal weight
        // w = 0.998 * ||dy||/||dx|| (the same balance the PID aims at), but
        // HPR blends it with the weight that achieved the best fixed-point
        // error so far instead of accumulating an integral term:
        //   fact  = exp(-0.05 * fp / best_fp)
        //   w_new = exp(fact * log(w_ratio) + (1-fact) * log(w_best))
        // and near convergence (fp/gap/residual floor <= 9e-10) rescales by
        // kappa = clamp(Rd/Rp or sqrt(Rd/Rp), 1e-2, 100) to rebalance the
        // tail. The movement guard matches the PID branch.
        if (primal_dist > 1e-16 && dual_dist > 1e-16 && primal_dist < 1e12 &&
            dual_dist < 1e12 && std::isfinite(s_.relative_primal_residual) &&
            std::isfinite(s_.restart_relative_dual_residual)) {
            double w_ratio = 0.998 * dual_dist / primal_dist;
            double best_gap = s_.hpr_best_gap > 0.0 ? s_.hpr_best_gap : s_.fixed_point_error;
            double fact = std::exp(-0.05 * s_.fixed_point_error / best_gap);
            double sigma_candidate =
                std::exp(fact * std::log(w_ratio) +
                         (1.0 - fact) * std::log(s_.hpr_best_weight));
            const double temp1 = std::max(
                std::min(s_.restart_relative_dual_residual, s_.relative_primal_residual),
                std::min(s_.relative_objective_gap, s_.fixed_point_error));
            double kappa = 1.0;
            if (temp1 <= 9e-10 && s_.relative_primal_residual > 0.0) {
                const double ratio_infeas_hpr =
                    s_.restart_relative_dual_residual / s_.relative_primal_residual;
                const double raw =
                    temp1 > 5e-10 ? std::sqrt(ratio_infeas_hpr) : ratio_infeas_hpr;
                kappa = std::max(std::min(raw, 100.0), 1e-2);
            }
            s_.primal_weight = std::clamp(kappa * sigma_candidate, 1e-12, 1e12);
        } else {
            s_.primal_weight = s_.hpr_best_weight;
        }
        s_.hpr_last_gap = s_.fixed_point_error;
    } else {
    // Keep the PID controller on cuPDLPx's projection-slack residual. Exported
    // certificate metrics are stricter and appropriate for stopping/auditing,
    // but feeding them back here changes the reference trajectory.
    // Guard conditions from CUDA reference
    if (primal_dist > 1e-16 && dual_dist > 1e-16 && primal_dist < 1e12 && dual_dist < 1e12 &&
        ratio_infeas > 1e-8 && ratio_infeas < 1e8) {
        double error = std::log(dual_dist) - std::log(primal_dist) - std::log(s_.primal_weight);
        // Apply integral smoothing
        s_.primal_weight_error_sum *= params_.restart_params.i_smooth;
        s_.primal_weight_error_sum += error;
        double delta_error = error - s_.primal_weight_last_error;
        s_.primal_weight *= std::exp(params_.restart_params.k_p * error +
                                     params_.restart_params.k_i * s_.primal_weight_error_sum +
                                     params_.restart_params.k_d * delta_error);
        s_.primal_weight_last_error = error;
    } else {
        s_.primal_weight = s_.best_primal_weight;
        s_.primal_weight_error_sum = 0.0;
        s_.primal_weight_last_error = 0.0;
    }
    }

    double primal_dual_residual_gap = std::fabs(std::log10(ratio_infeas));
    if (primal_dual_residual_gap < s_.best_primal_dual_residual_gap) {
        s_.best_primal_dual_residual_gap = primal_dual_residual_gap;
        s_.best_primal_weight = s_.primal_weight;
    }

    // Reset initial solutions to current pdhg solutions
    s_.x_init = s_.x_pdhg;
    s_.y_init = s_.y_pdhg;
    s_.x_cur = s_.x_pdhg;
    s_.y_cur = s_.y_pdhg;
    s_.inner_count = 0;
    s_.last_trial_fixed_point_error = inf();

    // Update step sizes
    s_.step_size_primal = s_.step_size / s_.primal_weight;
    s_.step_size_dual = s_.step_size * s_.primal_weight;

    if (params_.verbose) {
        printf("    restart weight: %.6e -> %.6e, ratio=%.6e, dist=(%.6e, %.6e)\n",
               old_primal_weight, s_.primal_weight, ratio_infeas, primal_dist, dual_dist);
    }

    mx::eval(s_.x_init, s_.y_init, s_.x_cur, s_.y_cur);
}

// ---------------------------------------------------------------------------
// Termination check
// ---------------------------------------------------------------------------

bool MlxPdlpSolver::mlx_check_termination(bool full_evaluation) {
    const auto &tc = params_.termination_criteria;

    double feas_tol = tc.eps_feasible_relative;
    double opt_tol = tc.eps_optimal_relative;

    // Check optimality first. The gap is compared in absolute value: a
    // numerically negative gap (primal below dual) is noise, not a proof of
    // optimality, and must not pass a one-sided comparison.
    if (s_.relative_dual_residual < feas_tol && s_.relative_primal_residual < feas_tol &&
        std::fabs(s_.relative_objective_gap) < opt_tol) {
        s_.termination_reason = TERMINATION_REASON_OPTIMAL;
        return true;
    }

    // ---- Infeasibility certification ----
    // Farkas separation certificates (see mlx_compute_infeasibility_information).
    // A dual ray with a strictly positive separation gap certifies primal
    // infeasibility; a primal ray with a negative linear objective certifies
    // dual infeasibility (an unbounded primal). The gap must be significant
    // relative to the problem data, and the recession-cone residual small
    // relative to the gap. On feasible problems the dual-ray gap is <= 0 for
    // every y by weak duality, so only fp noise could trip it, and the
    // significance floor absorbs that noise. Honor the independent requested
    // residual ratio on both devices: silently relaxing it on Metal can turn
    // an approximate ray into a false infeasibility status. Keep a separate
    // FP32 gap floor, since even an exactly zero computed ray residual cannot
    // validate a separation gap at the level of roundoff.
    const double infeas_tol = tc.eps_infeasible_relative;
    const double gap_tol =
        s_.cpu_double_precision_active ? infeas_tol : std::max(infeas_tol, 1e-3);
    // Sparse Metal certificate checks carry two additional SpMVs. Check the
    // first block so easy certificates still terminate immediately, then at a
    // bounded iteration cadence. A limit block is always checked to preserve
    // certificate precedence over iteration/time-limit statuses. CPU and
    // dense paths retain the historical every-block behavior.
    constexpr int infeasibility_check_interval = 1000;
    const int eval_frequency = std::max(1, params_.termination_evaluation_frequency);
    const int blocks_per_infeasibility_check =
        1 + (infeasibility_check_interval - 1) / eval_frequency;
    const int evaluation_block = s_.total_count / eval_frequency;
    const bool iteration_limit_reached = s_.total_count >= tc.iteration_limit;
    const bool time_limit_reached =
        elapsed_seconds(s_.start_time) >= tc.time_sec_limit;
    const bool should_check_infeasibility =
        (full_evaluation &&
         (!s_.sparse_metal_active || evaluation_block <= 1 ||
          evaluation_block % blocks_per_infeasibility_check == 0)) ||
        iteration_limit_reached || time_limit_reached;
    if (should_check_infeasibility) {
        mlx_compute_infeasibility_information();
        // Significance floors use the original-unit gaps; the residual ratio
        // tests use the working-unit gaps so both sides share units.
        if (s_.dual_ray_objective >
                gap_tol * (1.0 + s_.constraint_bound_norm) &&
            s_.max_dual_ray_infeasibility <=
                infeas_tol * working_dual_ray_objective_) {
            s_.termination_reason = TERMINATION_REASON_PRIMAL_INFEASIBLE;
            return true;
        }
        if (s_.primal_ray_linear_objective <
                -gap_tol * (1.0 + s_.objective_vector_norm) &&
            s_.max_primal_ray_infeasibility <=
                -infeas_tol * working_primal_ray_objective_) {
            s_.termination_reason = TERMINATION_REASON_DUAL_INFEASIBLE;
            return true;
        }
    }

    // Once the fp32 trajectory is close enough for the independently bounded
    // fp64 continuation, continuing on Metal can become pure stagnation. This
    // remains opt-in through host_double_polishing, and the distinct reason
    // preserves truthful status if continuation cannot finish within its caps.
    // Early transfer is deliberately stricter than the correction routine's
    // post-limit admission gate. BORE3D needs the wider 1e-2 recovery region
    // after fp32 is exhausted, while handing it off near 9e-3 is too early;
    // FORPLAN likewise needs more objective progress before fp64 can polish it.
    const double host_primal_gate = std::max(5e-3, 50.0 * feas_tol);
    // The host phase is a correction/polishing path, not the optimizer of
    // record. Require fp32 to establish objective proximity with a 2x audit
    // margin before early transfer; post-limit recovery may use the wider gate.
    const double host_gap_gate = 0.5 * opt_tol;
    const bool host_handoff_admissible =
        full_evaluation && !s_.cpu_double_precision_active &&
        params_.host_double_polishing &&
        params_.host_double_early_handoff &&
        // Do not abandon fp32 for a continuation budget too small to complete
        // even one normal residual-evaluation block.
        params_.host_double_polishing_iteration_limit >=
            std::max(3, params_.termination_evaluation_frequency) &&
        params_.host_double_polishing_time_sec_limit > 0.0 &&
        std::isfinite(s_.relative_primal_residual) &&
        std::isfinite(s_.relative_dual_residual) &&
        std::isfinite(s_.relative_objective_gap) &&
        s_.relative_primal_residual <= host_primal_gate &&
        s_.relative_dual_residual <= 1.0 &&
        s_.relative_objective_gap <= host_gap_gate;
    if (host_handoff_admissible) {
        const double current_kkt =
            std::max({s_.relative_primal_residual, s_.relative_dual_residual,
                      s_.relative_objective_gap});
        constexpr double substantial_reduction = 0.5;
        if (host_double_handoff_checkpoint_iteration_ < 0 ||
            current_kkt < substantial_reduction *
                               host_double_handoff_checkpoint_kkt_) {
            host_double_handoff_checkpoint_iteration_ = s_.total_count;
            host_double_handoff_checkpoint_kkt_ = current_kkt;
            host_double_handoff_x_ = s_.x_pdhg;
            host_double_handoff_y_ = s_.y_pdhg;
            host_double_handoff_dual_slack_ = s_.dual_slack;
            mx::eval(host_double_handoff_x_, host_double_handoff_y_,
                     host_double_handoff_dual_slack_);
        }
    }
    if (full_evaluation && host_double_handoff_checkpoint_iteration_ >= 0) {
        const int stagnation_window =
            25 * std::max(1, params_.termination_evaluation_frequency);
        if (s_.total_count - host_double_handoff_checkpoint_iteration_ >=
            stagnation_window) {
            s_.x_pdhg = host_double_handoff_x_;
            s_.y_pdhg = host_double_handoff_y_;
            s_.dual_slack = host_double_handoff_dual_slack_;
            mx::eval(s_.x_pdhg, s_.y_pdhg, s_.dual_slack);
            mlx_compute_residual();
            s_.termination_reason = TERMINATION_REASON_HOST_DOUBLE_HANDOFF;
            if (params_.verbose) {
                printf("  host-double handoff @ %d from checkpoint %d: "
                       "rel=(%.6e, %.6e, %.6e)\n",
                       s_.total_count, host_double_handoff_checkpoint_iteration_,
                       s_.relative_primal_residual, s_.relative_dual_residual,
                       s_.relative_objective_gap);
            }
            return true;
        }
    }

    if (s_.total_count >= tc.iteration_limit) {
        s_.termination_reason = TERMINATION_REASON_ITERATION_LIMIT;
        return true;
    }

    s_.cumulative_time_sec = elapsed_seconds(s_.start_time);
    if (s_.cumulative_time_sec >= tc.time_sec_limit) {
        s_.termination_reason = TERMINATION_REASON_TIME_LIMIT;
        return true;
    }

    return false;
}

// ---------------------------------------------------------------------------
// Adaptive restart check
// ---------------------------------------------------------------------------

bool MlxPdlpSolver::mlx_should_adaptive_restart() {
    const auto &rp = params_.restart_params;
    bool do_restart = false;
    const char *reason = nullptr;

    if (params_.restart_policy == 1) {
        // HPR-LP restart rules (src/HPRLP.cu check_restart):
        //  1. the first completed evaluation block always restarts;
        //  2. sufficient: fixed-point error <= 0.2 * the error measured at
        //     the previous restart (HPR's last_gap);
        //  3. long: inner_count >= 0.2 * total_count (HPR's artificial
        //     restart ratio, which dominates once the movement stalls).
        // HPR's "necessary" condition compares the same stale movement
        // quantity against itself between restarts and never fires in this
        // solver's block structure, so it is intentionally omitted.
        // Track the best fixed-point error and its weight at every
        // checkpoint (including the first), mirroring HPR's
        // best_gap/best_sigma bookkeeping.
        if (s_.fixed_point_error < s_.hpr_best_gap) {
            s_.hpr_best_gap = s_.fixed_point_error;
            s_.hpr_best_weight = s_.primal_weight;
        }
        if (s_.total_count == params_.termination_evaluation_frequency) {
            do_restart = true;
            reason = "initial";
        } else if (s_.total_count > params_.termination_evaluation_frequency) {
            if (s_.fixed_point_error <= 0.2 * s_.hpr_last_gap) {
                do_restart = true;
                reason = "sufficient";
            }
            if (s_.inner_count >= 0.2 * s_.total_count) {
                do_restart = true;
                reason = "long";
            }
        }
        s_.last_trial_fixed_point_error = s_.fixed_point_error;
        if (do_restart && params_.verbose) {
            printf("  restart @ %d: %s (inner=%d fp=%.6e last=%.6e best=%.6e)\n",
                   s_.total_count, reason ? reason : "adaptive", s_.inner_count,
                   s_.fixed_point_error, s_.hpr_last_gap, s_.hpr_best_gap);
        }
        return do_restart;
    }

    // CUDA reference: first evaluation block always restarts
    if (s_.total_count == params_.termination_evaluation_frequency) {
        do_restart = true;
        reason = "initial";
    } else if (s_.total_count > params_.termination_evaluation_frequency) {
        // Sufficient reduction: FP_error <= sufficient_reduction * initial_FP_error
        if (s_.fixed_point_error <=
            rp.sufficient_reduction_for_restart * s_.initial_fixed_point_error) {
            do_restart = true;
            reason = "sufficient";
        }
        // Necessary reduction + FP error increased: restart
        if (s_.fixed_point_error <=
            rp.necessary_reduction_for_restart * s_.initial_fixed_point_error) {
            if (s_.fixed_point_error > s_.last_trial_fixed_point_error) {
                do_restart = true;
                reason = "necessary/increase";
            }
        }
        // Artificial restart: inner_count >= threshold * total_count
        if (s_.inner_count >= rp.artificial_restart_threshold * s_.total_count) {
            do_restart = true;
            reason = "artificial";
        }
    }

    s_.last_trial_fixed_point_error = s_.fixed_point_error;
    if (do_restart && params_.verbose) {
        printf("  restart @ %d: %s (inner=%d fp=%.6e initial=%.6e)\n", s_.total_count,
               reason ? reason : "adaptive", s_.inner_count, s_.fixed_point_error,
               s_.initial_fixed_point_error);
    }
    return do_restart;
}

// ---------------------------------------------------------------------------
// Logging
// ---------------------------------------------------------------------------

void MlxPdlpSolver::mlx_display_header() {
    if (!params_.verbose)
        return;
    printf("mlxPDLP (Apple Silicon)\n");
    printf("Problem: %d variables, %d constraints, %d nonzeros\n", s_.n, s_.m, s_.nnz);
    if (s_.sparse_metal_active) {
        printf("Sparse Metal CSR SpMV\n");
    } else if (s_.sparse_cpu_active) {
        printf("Sparse CPU Accelerate SpMV\n");
    } else {
        printf("Dense MLX matmul fallback for SpMV (A = %.1f MB)\n",
               static_cast<double>(s_.m) * s_.n * sizeof(float) / 1e6);
    }
    printf("Parameters: eps_opt=%.1e eps_feas=%.1e eps_infeas=%.1e eval_freq=%d "
           "time_limit=%.0fs iter_limit=%d\n",
           params_.termination_criteria.eps_optimal_relative,
           params_.termination_criteria.eps_feasible_relative,
           params_.termination_criteria.eps_infeasible_relative,
           params_.termination_evaluation_frequency, params_.termination_criteria.time_sec_limit,
           params_.termination_criteria.iteration_limit);
    printf("%8s %14s %14s %14s %14s %14s %12s %12s\n", "Iter", "PrimalRes",
           "CertDualRes", "CtrlDualRes", "ObjGap", "FPError", "StepSize", "PrimalWgt");
}

void MlxPdlpSolver::mlx_display_iteration_stats() {
    if (!params_.verbose)
        return;
    // Keep long LPFeas traces readable while retaining every early checkpoint.
    if (s_.total_count > 10000 && s_.total_count % 1000 != 0)
        return;
    printf("%8d %14.6e %14.6e %14.6e %14.6e %14.6e %12.6f %12.6f\n",
           s_.total_count, s_.relative_primal_residual, s_.relative_dual_residual,
           s_.restart_relative_dual_residual, s_.relative_objective_gap, s_.fixed_point_error,
           s_.step_size, s_.primal_weight);
}

void MlxPdlpSolver::mlx_display_final_log() {
    if (!params_.verbose)
        return;
    printf("\n");
    printf("=== mlxPDLP Results ===\n");
    printf("Status: %s\n", termination_reason_str(s_.termination_reason));
    printf("Iterations: %d\n", s_.total_count);
    printf("Runtime: %.3f sec\n", s_.cumulative_time_sec);
    printf("Primal Objective: %.10f\n", s_.primal_objective_value);
    printf("Dual Objective:   %.10f\n", s_.dual_objective_value);
    printf("Objective Gap:    %.6e\n", s_.objective_gap);
    printf("Relative Gap:     %.6e\n", s_.relative_objective_gap);
    printf("Primal Residual:  %.6e (rel), %.6e (abs)\n", s_.relative_primal_residual,
           s_.absolute_primal_residual);
    printf("Dual Residual:    %.6e (rel), %.6e (abs)\n", s_.relative_dual_residual,
           s_.absolute_dual_residual);
    printf("====================================\n");
}

// ---------------------------------------------------------------------------
// Result extraction
// ---------------------------------------------------------------------------

double MlxPdlpSolver::recompute_original_certificate(mlxpdlp_result_t *result) {
    auto demote_unverifiable_optimal = [result]() {
        if (result && result->termination_reason == TERMINATION_REASON_OPTIMAL)
            result->termination_reason = TERMINATION_REASON_UNSPECIFIED;
    };
    if (!result || !result->primal_solution || !result->dual_solution ||
        !result->reduced_cost || result->num_variables != original_num_variables_ ||
        result->num_constraints != original_num_constraints_ ||
        original_row_ptr_.size() != static_cast<size_t>(original_num_constraints_ + 1) ||
        original_col_ind_.size() != static_cast<size_t>(original_num_nonzeros_) ||
        original_matrix_values_.size() != static_cast<size_t>(original_num_nonzeros_)) {
        demote_unverifiable_optimal();
        return inf();
    }

    const int m = original_num_constraints_;
    const int n = original_num_variables_;

    // Project row and variable-bound multipliers onto their exact original-
    // model sign domains. The reduced costs remain PDHG's complementary
    // projection multipliers; stationarity is measured rather than silently
    // forced by replacing them with c-A^T y.
    for (int row = 0; row < m; ++row) {
        double y = result->dual_solution[row];
        if (!std::isfinite(original_constraint_lower_bound_[static_cast<size_t>(row)]))
            y = std::min(y, 0.0);
        if (!std::isfinite(original_constraint_upper_bound_[static_cast<size_t>(row)]))
            y = std::max(y, 0.0);
        result->dual_solution[row] = y;
    }
    for (int column = 0; column < n; ++column) {
        double reduced_cost = result->reduced_cost[column];
        if (!std::isfinite(original_variable_lower_bound_[static_cast<size_t>(column)]))
            reduced_cost = std::min(reduced_cost, 0.0);
        if (!std::isfinite(original_variable_upper_bound_[static_cast<size_t>(column)]))
            reduced_cost = std::max(reduced_cost, 0.0);
        result->reduced_cost[column] = reduced_cost;
    }

    // Re-evaluate Ax and A^T y from the original double CSR. Long-double
    // accumulators remove scaled-device and reduction-order effects from the
    // reported original-model residuals and objectives.
    std::vector<long double> ax(static_cast<size_t>(m), 0.0L);
    std::vector<long double> aty(static_cast<size_t>(n), 0.0L);
    for (int row = 0; row < m; ++row) {
        const long double y = result->dual_solution[row];
        for (int entry = original_row_ptr_[static_cast<size_t>(row)];
             entry < original_row_ptr_[static_cast<size_t>(row + 1)]; ++entry) {
            const int column = original_col_ind_[static_cast<size_t>(entry)];
            if (column < 0 || column >= n) {
                demote_unverifiable_optimal();
                return inf();
            }
            const long double coefficient =
                original_matrix_values_[static_cast<size_t>(entry)];
            ax[static_cast<size_t>(row)] +=
                coefficient * result->primal_solution[column];
            aty[static_cast<size_t>(column)] += coefficient * y;
        }
    }

    long double primal_residual_sq = 0.0L;
    long double constraint_bound_norm_sq = 0.0L;
    long double variable_bound_violation_sq = 0.0L;
    long double variable_bound_norm_sq = 0.0L;
    long double dual_residual_sq = 0.0L;
    long double exact_dual_residual_sq = 0.0L;
    long double objective_norm_sq = 0.0L;
    long double primal_objective = original_objective_constant_;
    long double dual_objective = original_objective_constant_;
    long double exact_dual_objective = original_objective_constant_;
    std::vector<double> exact_reduced_cost(static_cast<size_t>(n));

    auto add_square = [](long double &sum, long double value) { sum += value * value; };
    for (int row = 0; row < m; ++row) {
        const double lower = original_constraint_lower_bound_[static_cast<size_t>(row)];
        const double upper = original_constraint_upper_bound_[static_cast<size_t>(row)];
        const long double activity = ax[static_cast<size_t>(row)];
        long double violation = 0.0L;
        if (activity < lower)
            violation = static_cast<long double>(lower) - activity;
        else if (activity > upper)
            violation = activity - static_cast<long double>(upper);
        add_square(primal_residual_sq, violation);
        if (std::isfinite(lower))
            add_square(constraint_bound_norm_sq, lower);
        if (std::isfinite(upper))
            add_square(constraint_bound_norm_sq, upper);

        const double y = result->dual_solution[row];
        if (y > 0.0) {
            dual_objective += static_cast<long double>(lower) * y;
            exact_dual_objective += static_cast<long double>(lower) * y;
        } else if (y < 0.0) {
            dual_objective += static_cast<long double>(upper) * y;
            exact_dual_objective += static_cast<long double>(upper) * y;
        }
    }

    for (int column = 0; column < n; ++column) {
        const double coefficient = original_objective_[static_cast<size_t>(column)];
        const double lower = original_variable_lower_bound_[static_cast<size_t>(column)];
        const double upper = original_variable_upper_bound_[static_cast<size_t>(column)];
        const double primal = result->primal_solution[column];
        if (!std::isfinite(primal)) {
            variable_bound_violation_sq =
                std::numeric_limits<long double>::infinity();
        } else {
            long double violation = 0.0L;
            if (primal < lower)
                violation = static_cast<long double>(lower) - primal;
            else if (primal > upper)
                violation = static_cast<long double>(primal) - upper;
            add_square(variable_bound_violation_sq, violation);
        }
        if (std::isfinite(lower))
            add_square(variable_bound_norm_sq, lower);
        if (std::isfinite(upper))
            add_square(variable_bound_norm_sq, upper);

        const long double raw = static_cast<long double>(coefficient) -
                                aty[static_cast<size_t>(column)];
        const long double reduced_cost = result->reduced_cost[column];
        long double exact_cost = raw;
        if (!std::isfinite(lower))
            exact_cost = std::min(exact_cost, 0.0L);
        if (!std::isfinite(upper))
            exact_cost = std::max(exact_cost, 0.0L);
        exact_reduced_cost[static_cast<size_t>(column)] =
            static_cast<double>(exact_cost);

        add_square(dual_residual_sq, raw - reduced_cost);
        add_square(exact_dual_residual_sq, raw - exact_cost);
        add_square(objective_norm_sq, coefficient);
        primal_objective +=
            static_cast<long double>(coefficient) * result->primal_solution[column];
        if (reduced_cost > 0.0L)
            dual_objective += static_cast<long double>(lower) * reduced_cost;
        else if (reduced_cost < 0.0L)
            dual_objective += static_cast<long double>(upper) * reduced_cost;
        if (exact_cost > 0.0L)
            exact_dual_objective += static_cast<long double>(lower) * exact_cost;
        else if (exact_cost < 0.0L)
            exact_dual_objective += static_cast<long double>(upper) * exact_cost;
    }

    const double objective_norm =
        std::sqrt(static_cast<double>(objective_norm_sq));
    const double projection_relative_dual =
        std::sqrt(static_cast<double>(dual_residual_sq)) / (1.0 + objective_norm);
    const double exact_relative_dual =
        std::sqrt(static_cast<double>(exact_dual_residual_sq)) / (1.0 + objective_norm);
    const double primal_objective_value = static_cast<double>(primal_objective);
    const double projection_dual_objective = static_cast<double>(dual_objective);
    const double exact_dual_objective_value = static_cast<double>(exact_dual_objective);
    auto relative_gap = [primal_objective_value](double candidate_dual_objective) {
        return std::fabs(primal_objective_value - candidate_dual_objective) /
               (1.0 + std::fabs(primal_objective_value) +
                std::fabs(candidate_dual_objective));
    };
    const double projection_relative_gap = relative_gap(projection_dual_objective);
    const double exact_relative_gap = relative_gap(exact_dual_objective_value);
    if (std::max(exact_relative_dual, exact_relative_gap) <
        std::max(projection_relative_dual, projection_relative_gap)) {
        std::copy(exact_reduced_cost.begin(), exact_reduced_cost.end(),
                  result->reduced_cost);
        dual_residual_sq = exact_dual_residual_sq;
        dual_objective = exact_dual_objective;
    }

    result->absolute_primal_residual =
        std::sqrt(static_cast<double>(primal_residual_sq));
    result->relative_primal_residual =
        result->absolute_primal_residual /
        (1.0 + std::sqrt(static_cast<double>(constraint_bound_norm_sq)));
    result->absolute_dual_residual = std::sqrt(static_cast<double>(dual_residual_sq));
    result->relative_dual_residual =
        result->absolute_dual_residual / (1.0 + objective_norm);
    result->primal_objective_value = static_cast<double>(primal_objective);
    result->dual_objective_value = static_cast<double>(dual_objective);
    result->objective_gap =
        std::fabs(result->primal_objective_value - result->dual_objective_value);
    result->relative_objective_gap =
        result->objective_gap /
        (1.0 + std::fabs(result->primal_objective_value) +
         std::fabs(result->dual_objective_value));
    const double relative_variable_bound_violation =
        std::sqrt(static_cast<double>(variable_bound_violation_sq)) /
        (1.0 + std::sqrt(static_cast<double>(variable_bound_norm_sq)));

    const auto &criteria = params_.termination_criteria;
    const bool certificate_is_optimal =
        std::isfinite(result->relative_primal_residual) &&
        std::isfinite(result->relative_dual_residual) &&
        std::isfinite(result->relative_objective_gap) &&
        std::isfinite(relative_variable_bound_violation) &&
        result->relative_primal_residual < criteria.eps_feasible_relative &&
        relative_variable_bound_violation < criteria.eps_feasible_relative &&
        result->relative_dual_residual < criteria.eps_feasible_relative &&
        std::fabs(result->relative_objective_gap) < criteria.eps_optimal_relative;
    if (certificate_is_optimal) {
        result->termination_reason = TERMINATION_REASON_OPTIMAL;
    } else if (result->termination_reason == TERMINATION_REASON_OPTIMAL) {
        // An OPTIMAL status must agree with every original-model metric stored
        // in the result: primal feasibility, variable bounds, dual feasibility,
        // and objective gap.
        result->termination_reason = TERMINATION_REASON_UNSPECIFIED;
    }
    return relative_variable_bound_violation;
}

void MlxPdlpSolver::polish_original_dual_certificate(mlxpdlp_result_t *result) {
    recompute_original_certificate(result);
    host_double_polish(result);
    recompute_original_certificate(result);
}

mlxpdlp_result_t *MlxPdlpSolver::extract_result() {
    auto *result = new mlxpdlp_result_t();
    std::memset(result, 0, sizeof(*result));

    result->num_variables = s_.n;
    result->num_constraints = s_.m;
    result->num_nonzeros = s_.nnz;

    // Unscale solutions: x = x_pdhg / var_rescale / con_bound_rescale
    auto x_unscaled =
        s_.x_pdhg / s_.var_rescale /
        mlx_scalar_like(s_.con_bound_rescale, s_.x_pdhg);
    // y = y_pdhg / con_rescale / obj_vec_rescale
    auto y_unscaled =
        s_.y_pdhg / s_.con_rescale /
        mlx_scalar_like(s_.obj_vec_rescale, s_.y_pdhg);
    // Export PDHG's complementary projection multiplier. When presolve is
    // active apply_postsolve independently rebuilds the exact c-A^T y seed in
    // host precision, so a host-corrected reduced multiplier is never replaced
    // by stale fp32 device state.
    auto rc_raw =
        s_.dual_slack * s_.var_rescale /
        mlx_scalar_like(s_.obj_vec_rescale, s_.dual_slack);

    // CUDA reference: clamp reduced cost for free variables
    // If var_lb is -inf: rc = min(rc, 0)
    // If var_ub is +inf: rc = max(rc, 0)
    auto rc_lb_adjusted =
        mx::where(s_.var_lb_inf_mask, mx::minimum(rc_raw, mx::zeros_like(rc_raw)), rc_raw);
    auto rc =
        mx::where(s_.var_ub_inf_mask, mx::maximum(rc_lb_adjusted, mx::zeros_like(rc_lb_adjusted)),
                  rc_lb_adjusted);
    mx::eval(x_unscaled, y_unscaled, rc);

    // Copy to host
    auto copy_to_host = [](const mx::array &arr, int size) -> double * {
        auto *host = new double[size];
        mx::eval(arr);
        if (arr.dtype() == mx::float64) {
            std::copy_n(arr.data<double>(), size, host);
        } else {
            const float *device_values = arr.data<float>();
            for (int i = 0; i < size; ++i) {
                host[i] = static_cast<double>(device_values[i]);
            }
        }
        return host;
    };

    result->primal_solution = copy_to_host(x_unscaled, s_.n);
    result->dual_solution = copy_to_host(y_unscaled, s_.m);
    result->reduced_cost = copy_to_host(rc, s_.n);

    result->total_count = s_.total_count;
    result->cumulative_time_sec = s_.cumulative_time_sec;
    result->rescaling_time_sec = s_.rescaling_time_sec;
    result->feasibility_polishing_time = s_.feasibility_polishing_time_sec;
    result->feasibility_iteration = s_.feasibility_iteration;

    result->absolute_primal_residual = s_.absolute_primal_residual;
    result->relative_primal_residual = s_.relative_primal_residual;
    result->absolute_dual_residual = s_.absolute_dual_residual;
    result->relative_dual_residual = s_.relative_dual_residual;
    result->primal_objective_value = s_.primal_objective_value;
    result->dual_objective_value = s_.dual_objective_value;
    result->objective_gap = s_.objective_gap;
    result->relative_objective_gap = s_.relative_objective_gap;

    result->max_primal_ray_infeasibility = s_.max_primal_ray_infeasibility;
    result->max_dual_ray_infeasibility = s_.max_dual_ray_infeasibility;
    result->primal_ray_linear_objective = s_.primal_ray_linear_objective;
    result->dual_ray_objective = s_.dual_ray_objective;

    result->termination_reason = s_.termination_reason;
    result->host_double_handoff =
        s_.termination_reason == TERMINATION_REASON_HOST_DOUBLE_HANDOFF;

#ifdef MLXPDLP_HAS_PRESOLVE
    if (presolve_context_) {
        host_double_polish(result, true);
        apply_postsolve(result);
    }
#endif

    polish_original_dual_certificate(result);

    return result;
}

#ifdef MLXPDLP_HAS_PRESOLVE
void MlxPdlpSolver::apply_postsolve(mlxpdlp_result_t *result) {
    // PSLP can reconstruct eliminated multipliers from either an exact
    // stationarity seed (c-A^T y) or PDHG's complementary projection
    // multiplier. Approximate fp32 iterates make different transformations
    // prefer different seeds, especially propagation-derived bound changes.
    // Replay the deterministic postsolve with both and select by a full
    // original-model fp64 certificate audit.
    std::vector<long double> reduced_at_y(static_cast<size_t>(working_num_variables_),
                                          0.0L);
    for (int row = 0; row < working_num_constraints_; ++row) {
        const long double dual = result->dual_solution[row];
        for (int entry = working_row_ptr_[static_cast<size_t>(row)];
             entry < working_row_ptr_[static_cast<size_t>(row + 1)]; ++entry) {
            const int column = working_col_ind_[static_cast<size_t>(entry)];
            reduced_at_y[static_cast<size_t>(column)] +=
                static_cast<long double>(
                    working_matrix_values_[static_cast<size_t>(entry)]) *
                dual;
        }
    }
    std::vector<double> exact_rc_host(static_cast<size_t>(working_num_variables_));
    for (int column = 0; column < working_num_variables_; ++column) {
        long double reduced_cost =
            static_cast<long double>(
                working_objective_[static_cast<size_t>(column)]) -
            reduced_at_y[static_cast<size_t>(column)];
        if (!std::isfinite(
                working_variable_lower_bound_[static_cast<size_t>(column)]))
            reduced_cost = std::min(reduced_cost, 0.0L);
        if (!std::isfinite(
                working_variable_upper_bound_[static_cast<size_t>(column)]))
            reduced_cost = std::max(reduced_cost, 0.0L);
        exact_rc_host[static_cast<size_t>(column)] =
            static_cast<double>(reduced_cost);
    }

    auto exact_solution = detail::postsolve(
        presolve_context_, result->primal_solution, result->dual_solution,
        exact_rc_host.data(), original_variable_lower_bound_,
        original_variable_upper_bound_);

    auto projection_solution = detail::postsolve(
        presolve_context_, result->primal_solution, result->dual_solution,
        result->reduced_cost, original_variable_lower_bound_,
        original_variable_upper_bound_);

    delete[] result->primal_solution;
    delete[] result->dual_solution;
    delete[] result->reduced_cost;

    auto copy_vector = [](const std::vector<double> &values) {
        auto *copy = new double[values.size()];
        std::copy(values.begin(), values.end(), copy);
        return copy;
    };
    result->num_reduced_variables = s_.n;
    result->num_reduced_constraints = s_.m;
    result->num_reduced_nonzeros = s_.nnz;
    result->num_variables = original_num_variables_;
    result->num_constraints = original_num_constraints_;
    result->num_nonzeros = original_num_nonzeros_;
    result->presolve_status = presolve_status_;
    result->presolve_time = presolve_time_sec_;

    auto install_solution = [&](const detail::PostsolveSolution &solution) {
        delete[] result->primal_solution;
        delete[] result->dual_solution;
        delete[] result->reduced_cost;
        result->primal_solution = copy_vector(solution.primal);
        result->dual_solution = copy_vector(solution.dual);
        result->reduced_cost = copy_vector(solution.reduced_cost);
    };
    struct CandidateAudit {
        double merit;
        double feasibility;
    };
    auto audited_merit = [&]() {
        const double variable_bound_violation =
            recompute_original_certificate(result);
        return CandidateAudit{
            std::max({result->relative_primal_residual,
                      variable_bound_violation,
                      result->relative_dual_residual,
                      result->relative_objective_gap}),
            std::max({result->relative_primal_residual,
                      variable_bound_violation,
                      result->relative_dual_residual})};
    };

    // The pointers were deleted above and are installed afresh for each
    // candidate. Null them before the first install so the helper remains
    // safe and symmetric.
    result->primal_solution = nullptr;
    result->dual_solution = nullptr;
    result->reduced_cost = nullptr;
    install_solution(exact_solution);
    const CandidateAudit exact_audit = audited_merit();
    install_solution(projection_solution);
    const CandidateAudit projection_audit = audited_merit();
    const double exact_merit = exact_audit.merit;
    const double exact_feasibility = exact_audit.feasibility;
    const double projection_merit = projection_audit.merit;
    const double projection_feasibility = projection_audit.feasibility;
    bool use_projection =
        std::isfinite(projection_merit) &&
        (!std::isfinite(exact_merit) || projection_merit < exact_merit);
    // Differences below the requested accuracy are operationally tied. In
    // that regime retain the more feasible certificate instead of exchanging
    // a large dual-residual regression for an insignificant gap reduction.
    if (std::isfinite(exact_merit) && std::isfinite(projection_merit) &&
        std::fabs(exact_merit - projection_merit) <=
            params_.termination_criteria.eps_optimal_relative) {
        use_projection = projection_feasibility < exact_feasibility;
    }
    if (!use_projection) {
        install_solution(exact_solution);
        recompute_original_certificate(result);
    }
    if (params_.verbose) {
        std::printf("  postsolve certificate seeds: exact=%.6e projection=%.6e "
                    "selected=%s\n",
                    exact_merit, projection_merit,
                    use_projection ? "projection" : "exact");
    }
}

mlxpdlp_result_t *MlxPdlpSolver::extract_presolve_result() {
    auto *result = new mlxpdlp_result_t();
    std::memset(result, 0, sizeof(*result));
    result->num_variables = original_num_variables_;
    result->num_constraints = original_num_constraints_;
    result->num_nonzeros = original_num_nonzeros_;
    result->num_reduced_variables = s_.n;
    result->num_reduced_constraints = s_.m;
    result->num_reduced_nonzeros = s_.nnz;
    result->presolve_status = presolve_status_;
    result->presolve_time = presolve_time_sec_;

    auto allocate_zeros = [](int size) {
        auto *values = new double[static_cast<size_t>(size)]();
        return values;
    };

    auto terminal_kind = detail::classify_presolve_status(presolve_status_);
    if (terminal_kind == detail::PresolveTerminalKind::primal_infeasible ||
        terminal_kind == detail::PresolveTerminalKind::infeasible_or_unbounded) {
        result->termination_reason =
            terminal_kind == detail::PresolveTerminalKind::primal_infeasible
                ? TERMINATION_REASON_PRIMAL_INFEASIBLE
                : TERMINATION_REASON_INFEASIBLE_OR_UNBOUNDED;
        result->absolute_primal_residual = inf();
        result->relative_primal_residual = inf();
        result->absolute_dual_residual = inf();
        result->relative_dual_residual = inf();
        result->primal_objective_value = inf();
        result->dual_objective_value = -inf();
        result->objective_gap = inf();
        result->relative_objective_gap = inf();
        result->primal_solution = allocate_zeros(original_num_variables_);
        result->dual_solution = allocate_zeros(original_num_constraints_);
        result->reduced_cost = allocate_zeros(original_num_variables_);
        return result;
    }

    auto solution =
        detail::postsolve(presolve_context_, nullptr, nullptr, nullptr,
                          original_variable_lower_bound_, original_variable_upper_bound_);
    auto copy_vector = [](const std::vector<double> &values) {
        auto *copy = new double[values.size()];
        std::copy(values.begin(), values.end(), copy);
        return copy;
    };
    result->primal_solution = copy_vector(solution.primal);
    result->dual_solution = copy_vector(solution.dual);
    result->reduced_cost = copy_vector(solution.reduced_cost);

    double objective = original_objective_constant_;
    for (int i = 0; i < original_num_variables_; ++i)
        objective += original_objective_[static_cast<size_t>(i)] * result->primal_solution[i];
    result->primal_objective_value = objective;
    result->dual_objective_value = objective;
    result->termination_reason = TERMINATION_REASON_OPTIMAL;
    polish_original_dual_certificate(result);
    return result;
}
#else
void MlxPdlpSolver::apply_postsolve(mlxpdlp_result_t *) {}

mlxpdlp_result_t *MlxPdlpSolver::extract_presolve_result() {
    throw std::logic_error("presolve support is not built");
}
#endif

// ---------------------------------------------------------------------------
// Main solve() — the full PDHG algorithm
// ---------------------------------------------------------------------------

mlxpdlp_result_t *MlxPdlpSolver::solve() {
    if (solve_called_) {
        throw std::logic_error(
            "MlxPdlpSolver::solve() may only be called once; construct a new solver instance");
    }
    solve_called_ = true;

    // Every MLX operation in the solve path omits an explicit stream, so scope
    // MLX's defaults to the stream selected in the constructor.
    mx::StreamContext stream_context(s_.stream);
    s_.start_time = SteadyClock::now();

    if (presolve_solved_)
        return extract_presolve_result();

    // ---- Phase 1: Preconditioning ----
    const auto rescale_start = SteadyClock::now();

    // Compute initial norms (before rescaling, for relative residual denominators)
    s_.objective_vector_norm = mlx_norm2(s_.obj);
    {
        auto con_lb_safe = mx::where(mx::isfinite(s_.con_lb), s_.con_lb, mx::zeros_like(s_.con_lb));
        auto con_ub_safe = mx::where(mx::isfinite(s_.con_ub), s_.con_ub, mx::zeros_like(s_.con_ub));
        auto con_norm_sq = mx::square(con_lb_safe) + mx::square(con_ub_safe);
        auto con_norm = mx::sqrt(mx::sum(con_norm_sq));
        mx::eval(con_norm);
        s_.constraint_bound_norm = mlx_scalar_as_double(con_norm);
    }
    // Zero is a valid norm.  Relative residuals already use a `1 + norm`
    // denominator, so replacing zero by one halves the residual and also changes
    // the adaptive primal-weight trajectory on homogeneous models.

    // Apply preconditioning (geometric mean, optional Curtis-Reid, then Ruiz,
    // Pock-Chambolle, and bound-objective scaling).
    if (params_.geometric_mean_iterations > 0 && s_.nnz > 0) {
        mlx_geometric_mean_scaling(params_.geometric_mean_iterations);
    }
    if (params_.curtis_reid_iterations > 0 && s_.nnz > 0) {
        mlx_curtis_reid_scaling(params_.curtis_reid_iterations);
    }
    if (params_.l_inf_ruiz_iterations > 0 && s_.nnz > 0) {
        mlx_ruiz_scaling(params_.l_inf_ruiz_iterations);
        // Recompute finite-safe arrays and inf-masks after Ruiz changes bounds
        auto var_lb_host = new double[s_.n];
        auto var_ub_host = new double[s_.n];
        auto con_lb_host = new double[s_.m];
        auto con_ub_host = new double[s_.m];
        mx::eval(s_.var_lb, s_.var_ub, s_.con_lb, s_.con_ub);
        if (s_.var_lb.dtype() == mx::float64) {
            std::copy_n(s_.var_lb.data<double>(), s_.n, var_lb_host);
            std::copy_n(s_.var_ub.data<double>(), s_.n, var_ub_host);
            std::copy_n(s_.con_lb.data<double>(), s_.m, con_lb_host);
            std::copy_n(s_.con_ub.data<double>(), s_.m, con_ub_host);
        } else {
            const float *var_lb_values = s_.var_lb.data<float>();
            const float *var_ub_values = s_.var_ub.data<float>();
            const float *con_lb_values = s_.con_lb.data<float>();
            const float *con_ub_values = s_.con_ub.data<float>();
            for (int i = 0; i < s_.n; ++i) {
                var_lb_host[i] = static_cast<double>(var_lb_values[i]);
                var_ub_host[i] = static_cast<double>(var_ub_values[i]);
            }
            for (int i = 0; i < s_.m; ++i) {
                con_lb_host[i] = static_cast<double>(con_lb_values[i]);
                con_ub_host[i] = static_cast<double>(con_ub_values[i]);
            }
        }
        preprocess_bounds(var_lb_host, var_ub_host, con_lb_host, con_ub_host);
        delete[] var_lb_host;
        delete[] var_ub_host;
        delete[] con_lb_host;
        delete[] con_ub_host;
    }
    if (params_.has_pock_chambolle_alpha) {
        mlx_pock_chambolle_scaling(params_.pock_chambolle_alpha);
    }
    if (params_.bound_objective_rescaling) {
        mlx_bound_objective_scaling();
    }

    // Scaling is now final. Materialize the selected sparse backend before the
    // power method and iterative PDHG hot path.
    prepare_sparse_metal_backend();
    prepare_sparse_cpu_backend();

    // After all rescaling, copy initial solutions
    s_.x_init = s_.x_cur;
    s_.y_init = s_.y_cur;
    s_.x_pdhg = s_.x_cur;
    s_.y_pdhg = s_.y_cur;
    mx::eval(s_.x_init, s_.y_init, s_.x_pdhg, s_.y_pdhg);

    s_.rescaling_time_sec = elapsed_seconds(rescale_start);

    // ---- Phase 2: Step size initialization ----
    if (s_.nnz > 0) {
        double max_sv = mlx_estimate_max_singular_value();
        if (max_sv < 1e-14) {
            if (params_.verbose) {
                printf("  WARNING: max_sv ≈ 0, using fallback step size\n");
            }
            s_.step_size = 1.0;
        } else {
            s_.step_size = 0.998 / max_sv;
        }
    } else {
        s_.step_size = 1.0;
    }

    // Curtis-Reid can make the post-scaled objective and bound norms differ by
    // many orders of magnitude. Match HPR-LP's initial primal step in that
    // mode, expressed in this solver's symmetric step/weight coordinates.
    if (params_.curtis_reid_iterations > 0) {
        auto con_lb_safe =
            mx::where(mx::isfinite(s_.con_lb), mx::abs(s_.con_lb), mx::zeros_like(s_.con_lb));
        auto con_ub_safe =
            mx::where(mx::isfinite(s_.con_ub), mx::abs(s_.con_ub), mx::zeros_like(s_.con_ub));
        double scaled_bound_norm = mlx_norm2(mx::maximum(con_lb_safe, con_ub_safe));
        double scaled_objective_norm = mlx_norm2(s_.obj);
        if (scaled_bound_norm > 1e-8 && scaled_objective_norm > 1e-8) {
            double hpr_primal_step = scaled_bound_norm / scaled_objective_norm;
            s_.primal_weight =
                std::clamp(s_.step_size / hpr_primal_step, 1e-12, 1e12);
        } else {
            s_.primal_weight = 1.0;
        }
        if (params_.verbose) {
            printf("  scaled bound norm=%.6e objective norm=%.6e\n", scaled_bound_norm,
                   scaled_objective_norm);
        }
    } else if (params_.bound_objective_rescaling) {
        s_.primal_weight = 1.0;
    } else {
        s_.primal_weight = (s_.objective_vector_norm + 1.0) / (s_.constraint_bound_norm + 1.0);
    }
    s_.best_primal_weight = s_.primal_weight;
    s_.step_size_primal = s_.step_size / s_.primal_weight;
    s_.step_size_dual = s_.step_size * s_.primal_weight;

    // Make the supplied point eligible for the best-iterate safeguard before
    // the first averaged block can move away from it. A complete checkpoint
    // retains its projection multiplier z; x/y-only callers fall back to the
    // exact sign-projected certificate reconstructed from y.
    if (has_warm_start_) {
        s_.ATy = mat_ATx(s_.y_pdhg);
        if (has_reduced_cost_start_) {
            auto reduced_cost_unscaled = mlx_array_from_doubles(
                warm_reduced_cost_.data(), s_.n, s_.obj.dtype());
            s_.dual_slack =
                reduced_cost_unscaled *
                mlx_scalar_like(s_.obj_vec_rescale, reduced_cost_unscaled) /
                s_.var_rescale;
        } else {
            auto reduced_cost_raw = s_.obj - s_.ATy;
            auto reduced_cost_lb_adjusted = mx::where(
                s_.var_lb_inf_mask,
                mx::minimum(reduced_cost_raw, mx::zeros_like(reduced_cost_raw)),
                reduced_cost_raw);
            s_.dual_slack = mx::where(
                s_.var_ub_inf_mask,
                mx::maximum(reduced_cost_lb_adjusted,
                            mx::zeros_like(reduced_cost_lb_adjusted)),
                reduced_cost_lb_adjusted);
        }
        mx::eval(s_.ATy, s_.dual_slack);
        mlx_compute_residual();
        mlx_save_best_iterate();

        // A correction warm start often has an excellent primal point and a
        // much poorer dual certificate (or vice versa). Preserve the stable
        // PDHG step product while biasing the first block toward the side that
        // needs work. Adaptive restarts take over after this initialization.
        if (std::isfinite(s_.relative_primal_residual) &&
            std::isfinite(s_.relative_dual_residual)) {
            const double ratio =
                (s_.relative_dual_residual + 1e-12) /
                (s_.relative_primal_residual + 1e-12);
            const double balance =
                std::sqrt(std::clamp(ratio, 1e-6, 1e6));
            s_.primal_weight =
                std::clamp(s_.primal_weight * balance, 1e-6, 1e6);
            s_.best_primal_weight = s_.primal_weight;
            s_.step_size_primal = s_.step_size / s_.primal_weight;
            s_.step_size_dual = s_.step_size * s_.primal_weight;
            if (params_.verbose) {
                printf("  warm-start balance=%.6e residual-ratio=%.6e\n",
                       s_.primal_weight, ratio);
            }
        }
    }

    if (params_.verbose) {
        printf("  max_sv=%.6f step_size=%.6f primal_weight=%.6f rescale_time=%.3fs\n",
               s_.nnz > 0 ? 0.998 / s_.step_size : 0.0, s_.step_size, s_.primal_weight,
               s_.rescaling_time_sec);
    }

    // ---- Phase 3: Main PDHG loop ----
    mlx_display_header();

    const int eval_freq = std::max(1, params_.termination_evaluation_frequency);
    // A residual checkpoint reads A and A^T again and drains MLX's lazy graph.
    // Keep cuOpt-style conditional checks to models whose two fp32 CSR copies
    // fit in roughly 4 MiB; larger sparse LPs benchmark better at the regular
    // cadence (notably s250r10 and s82 from LPfeas).
    constexpr int max_conditional_evaluation_nnz = 1 << 18;
    const bool use_conditional_evaluations =
        params_.conditional_termination_evaluation &&
        s_.nnz <= max_conditional_evaluation_nnz;
    bool conditional_evaluation_admissible = false;
    bool do_restart = false;
    // On the fused sparse-Metal path, minor iterations accumulate in the lazy
    // graph and are materialized every fused_batch iterations; major
    // iterations always evaluate so restart/residual checks see live data.
    const int fused_batch = fused_eval_batch_size();

    while (s_.total_count < params_.termination_criteria.iteration_limit) {
        const int remaining =
            params_.termination_criteria.iteration_limit - s_.total_count;
        const int block_iterations = std::min(
            remaining,
            iterations_to_next_evaluation(
                s_.total_count, eval_freq,
                use_conditional_evaluations &&
                    conditional_evaluation_admissible));
        const int checkpoint_iteration = s_.total_count + block_iterations;
        const bool restart_checkpoint = checkpoint_iteration % eval_freq == 0;
        const bool limit_checkpoint = checkpoint_iteration >=
                                      params_.termination_criteria.iteration_limit;

        // A restart needs a materialized first PDHG candidate to seed its new
        // fixed-point baseline. Otherwise the first iteration can use the
        // cheaper minor kernel; only the checkpoint itself needs snapshots.
        const bool first_is_major = do_restart || block_iterations == 1;
        mlx_compute_next_primal(1, first_is_major);
        mlx_compute_next_dual(1, first_is_major,
                              first_is_major || fused_batch == 1);

        // After first iteration, check if we need a restart
        if (do_restart) {
            mlx_compute_fixed_point_error();
            s_.initial_fixed_point_error = s_.fixed_point_error;
            do_restart = false;
        }

        // --- Minor iterations 2 through the penultimate iteration ---
        for (int i = 2; i < block_iterations; ++i) {
            mlx_compute_next_primal(i, false);
            mlx_compute_next_dual(i, false, i % fused_batch == 0);
        }

        // --- Last iteration (major) ---
        if (block_iterations > 1) {
            mlx_compute_next_primal(block_iterations, true);
            mlx_compute_next_dual(block_iterations, true, true);
        }

        // --- Periodic checks ---
        if (restart_checkpoint) {
            mlx_compute_fixed_point_error();
        }
        mlx_compute_residual();

        s_.inner_count += block_iterations;
        s_.total_count += block_iterations;

        // fp32 trajectories can improve and later regress substantially,
        // especially after a primal-weight restart. Preserve the best fully
        // evaluated KKT point so a time/iteration limit returns useful work.
        mlx_save_best_iterate();

        if (restart_checkpoint) {
            mlx_display_iteration_stats();
        }

        // Termination check
        if (mlx_check_termination(restart_checkpoint || limit_checkpoint)) {
            break;
        }

        if (restart_checkpoint && use_conditional_evaluations) {
            // Speculative residual checks only pay off close to convergence.
            // Normalize each component independently because feasibility and
            // optimality tolerances need not match.
            constexpr double conditional_tolerance_factor = 10.0;
            const auto &criteria = params_.termination_criteria;
            conditional_evaluation_admissible =
                std::isfinite(s_.relative_primal_residual) &&
                std::isfinite(s_.relative_dual_residual) &&
                std::isfinite(s_.relative_objective_gap) &&
                s_.relative_primal_residual <=
                    conditional_tolerance_factor *
                        criteria.eps_feasible_relative &&
                s_.relative_dual_residual <=
                    conditional_tolerance_factor *
                        criteria.eps_feasible_relative &&
                std::fabs(s_.relative_objective_gap) <=
                    conditional_tolerance_factor *
                        criteria.eps_optimal_relative;
        }

        // Adaptive restart check
        if (restart_checkpoint && mlx_should_adaptive_restart()) {
            mlx_perform_restart();
            do_restart = true;
        }
    }

    // If loop ended without explicit termination reason, set to iteration limit
    if (s_.termination_reason == TERMINATION_REASON_UNSPECIFIED) {
        s_.termination_reason = TERMINATION_REASON_ITERATION_LIMIT;
    }

    // ---- Phase 4: Final residual computation ----
    if (s_.termination_reason == TERMINATION_REASON_TIME_LIMIT ||
        s_.termination_reason == TERMINATION_REASON_ITERATION_LIMIT) {
        mlx_restore_best_iterate();
    }
    mlx_compute_residual();
    mlx_primal_feasibility_polish();
    mlx_dual_feasibility_polish();
    s_.cumulative_time_sec = elapsed_seconds(s_.start_time);

    mlx_display_final_log();

    // ---- Phase 5: Extract result ----
    return extract_result();
}

} // namespace mlxpdlp
