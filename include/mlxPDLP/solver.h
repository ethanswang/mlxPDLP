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

#pragma once

#include <cmath>
#include <cstdint>
#include <chrono>
#include <cstring>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include "mlx/mlx.h"
#include "mlxPDLP/version.h"

namespace mlxpdlp {

namespace mx = mlx::core;

namespace detail {
struct CpuSparseMatrix;
struct PresolveContext;
}

// ---------------------------------------------------------------------------
// Standalone PDLP parameter and result types (no CUDA dependency).
// ---------------------------------------------------------------------------

typedef enum {
    TERMINATION_REASON_UNSPECIFIED = 0,
    TERMINATION_REASON_OPTIMAL = 1,
    TERMINATION_REASON_PRIMAL_INFEASIBLE = 2,
    TERMINATION_REASON_DUAL_INFEASIBLE = 3,
    TERMINATION_REASON_INFEASIBLE_OR_UNBOUNDED = 4,
    TERMINATION_REASON_TIME_LIMIT = 5,
    TERMINATION_REASON_ITERATION_LIMIT = 6,
    TERMINATION_REASON_FEAS_POLISH_SUCCESS = 7,
    // The fp32 iterate entered the bounded admission region for the optional
    // fp64 continuation. A successful continuation rewrites this to OPTIMAL;
    // otherwise callers can distinguish an intentional handoff from a limit.
    TERMINATION_REASON_HOST_DOUBLE_HANDOFF = 8
} termination_reason_t;

typedef enum { NORM_TYPE_L2 = 0, NORM_TYPE_L_INF = 1 } norm_type_t;

typedef struct {
    double artificial_restart_threshold;
    double sufficient_reduction_for_restart;
    double necessary_reduction_for_restart;
    double k_p;
    double k_i;
    double k_d;
    double i_smooth;
} restart_parameters_t;

typedef struct {
    double eps_optimal_relative;
    double eps_feasible_relative;
    double eps_feas_polish_relative;
    double time_sec_limit;
    int iteration_limit;
} termination_criteria_t;

typedef struct {
    int curtis_reid_iterations;
    int l_inf_ruiz_iterations;
    bool has_pock_chambolle_alpha;
    double pock_chambolle_alpha;
    bool bound_objective_rescaling;
    bool verbose;
    int termination_evaluation_frequency;
    int sv_max_iter;
    double sv_tol;
    termination_criteria_t termination_criteria;
    restart_parameters_t restart_params;
    // 0 = cuPDLPx PID primal-weight restart (default); 1 = HPR-LP-style
    // movement-ratio sigma update with best-gap anchoring (HPR-LP-C
    // src/HPRLP.cu update_sigma/check_restart).
    int restart_policy;
    double reflection_coefficient;
    bool feasibility_polishing;
    bool host_double_polishing;
    bool host_double_early_handoff;
    int host_double_polishing_iteration_limit;
    double host_double_polishing_time_sec_limit;
    norm_type_t optimality_norm;
    bool presolve;
    bool presolve_singleton_columns;
    bool presolve_doubleton_equations;
    bool presolve_parallel_rows;
    bool presolve_parallel_columns;
    bool presolve_dual_fix;
    bool presolve_finite_bound_tightening;
    bool presolve_primal_propagation;
    double matrix_zero_tol;
    // Fuse each PDHG half-step (SpMV + scaled projection + reflection +
    // Halpern weighting) into a single Metal kernel on the sparse Metal path.
    // Disable to fall back to the unfused MLX-expression formulation for A/B
    // comparison or diagnostics. Only affects the sparse Metal backend.
    bool metal_fused_kernels;
} pdhg_parameters_t;

typedef struct {
    int num_variables;
    int num_constraints;
    int num_nonzeros;
    int num_reduced_variables;
    int num_reduced_constraints;
    int num_reduced_nonzeros;
    double *primal_solution;
    double *dual_solution;
    double *reduced_cost;
    int total_count;
    double rescaling_time_sec;
    double cumulative_time_sec;
    double presolve_time;
    int presolve_status;
    double absolute_primal_residual;
    double relative_primal_residual;
    double absolute_dual_residual;
    double relative_dual_residual;
    double primal_objective_value;
    double dual_objective_value;
    double objective_gap;
    double relative_objective_gap;
    double max_primal_ray_infeasibility;
    double max_dual_ray_infeasibility;
    double primal_ray_linear_objective;
    double dual_ray_objective;
    termination_reason_t termination_reason;
    double feasibility_polishing_time;
    int feasibility_iteration;
    double host_double_polishing_time;
    int host_double_polishing_iteration;
    bool host_double_handoff;
} mlxpdlp_result_t;

// ---------------------------------------------------------------------------
// Solver state — numerical data lives in lazy MLX arrays. The execution device
// is selected by the solver's stream (CPU by default; Metal requires a
// GPU-enabled MLX build and a GPU-supported dtype).
// ---------------------------------------------------------------------------

// Forward declare helper used in MlxPdlpState default ctor.
// Defined in solver.cpp.
mx::array _mlx_empty_array();

enum class SparseMetalSpmvStrategy : uint8_t {
    scalar_rows,
    simdgroup_rows,
    adaptive,
};

struct MlxPdlpState {
    // Default constructor — initializes all mx::array fields to empty arrays
    // since mx::array has no default constructor.
    MlxPdlpState()
        : m(0), n(0), nnz(0), sparse_metal_active(false), sparse_cpu_active(false),
          sparse_a_spmv_strategy(SparseMetalSpmvStrategy::adaptive),
          sparse_at_spmv_strategy(SparseMetalSpmvStrategy::adaptive),
          cpu_double_precision_active(false),
          A(_mlx_empty_array()),
          AT(_mlx_empty_array()), var_lb(_mlx_empty_array()), var_ub(_mlx_empty_array()),
          con_lb(_mlx_empty_array()), con_ub(_mlx_empty_array()), obj(_mlx_empty_array()),
          objective_constant(0.0), var_lb_finite(_mlx_empty_array()),
          var_ub_finite(_mlx_empty_array()), con_lb_finite(_mlx_empty_array()),
          con_ub_finite(_mlx_empty_array()), var_lb_inf_mask(_mlx_empty_array()),
          var_ub_inf_mask(_mlx_empty_array()), con_lb_inf_mask(_mlx_empty_array()),
          con_ub_inf_mask(_mlx_empty_array()), var_rescale(_mlx_empty_array()),
          con_rescale(_mlx_empty_array()), con_bound_rescale(1.0), obj_vec_rescale(1.0),
          x_cur(_mlx_empty_array()), x_pdhg(_mlx_empty_array()), x_ref(_mlx_empty_array()),
          x_init(_mlx_empty_array()), x_best(_mlx_empty_array()), ATy(_mlx_empty_array()),
          y_cur(_mlx_empty_array()), y_pdhg(_mlx_empty_array()), y_ref(_mlx_empty_array()),
          y_init(_mlx_empty_array()), y_best(_mlx_empty_array()), Ax(_mlx_empty_array()),
          primal_slack(_mlx_empty_array()), dual_slack(_mlx_empty_array()),
          dual_slack_best(_mlx_empty_array()),
          primal_res(_mlx_empty_array()), dual_res(_mlx_empty_array()), delta_x(_mlx_empty_array()),
          delta_y(_mlx_empty_array()), step_size(1.0), primal_weight(1.0), step_size_primal(1.0),
          step_size_dual(1.0), inner_count(0), total_count(0), absolute_primal_residual(0.0),
          relative_primal_residual(0.0), absolute_dual_residual(0.0), relative_dual_residual(0.0),
          restart_relative_dual_residual(0.0),
          primal_objective_value(0.0), dual_objective_value(0.0), objective_gap(0.0),
          relative_objective_gap(0.0), fixed_point_error(0.0), initial_fixed_point_error(0.0),
          last_trial_fixed_point_error(0.0), best_primal_dual_residual_gap(0.0),
          best_relative_kkt_error(0.0), best_relative_feasibility_error(0.0),
          best_iteration(-1),
          primal_weight_error_sum(0.0), primal_weight_last_error(0.0), best_primal_weight(1.0),
          hpr_last_gap(std::numeric_limits<double>::infinity()),
          hpr_best_gap(std::numeric_limits<double>::infinity()), hpr_best_weight(1.0),
          max_primal_ray_infeasibility(0.0), max_dual_ray_infeasibility(0.0),
          primal_ray_linear_objective(0.0), dual_ray_objective(0.0),
          termination_reason(TERMINATION_REASON_UNSPECIFIED), objective_vector_norm(0.0),
          constraint_bound_norm(0.0), start_time(std::chrono::steady_clock::now()),
          cumulative_time_sec(0.0), rescaling_time_sec(0.0),
          feasibility_polishing_time_sec(0.0), feasibility_iteration(0),
          stream(0, mx::Device::cpu) {}

    // Dimensions
    int m; // num_constraints
    int n; // num_variables
    int nnz;
    bool sparse_metal_active;
    bool sparse_cpu_active;
    // A and A^T are profiled independently because their CSR row-length
    // distributions can require different Metal thread mappings.
    SparseMetalSpmvStrategy sparse_a_spmv_strategy;
    SparseMetalSpmvStrategy sparse_at_spmv_strategy;
    // CPU execution deliberately uses float64 throughout PDHG. Metal remains
    // float32 because Apple GPU kernels do not support float64 arithmetic.
    bool cpu_double_precision_active;

    // Dense matrix storage used by small/dense CPU and Metal fallbacks. Sparse
    // backend candidates keep both arrays empty throughout preprocessing and
    // iteration.
    mx::array A;  // [m, n], or empty for a sparse backend
    mx::array AT; // [n, m], or empty for a sparse backend

    // Bounds & objective
    mx::array var_lb; // [n]
    mx::array var_ub; // [n]
    mx::array con_lb; // [m]
    mx::array con_ub; // [m]
    mx::array obj;    // [n] objective coefficients
    double objective_constant;

    // Finite-safe bound sentinels (inf → 0 for multiplication contexts)
    mx::array var_lb_finite; // [n]
    mx::array var_ub_finite; // [n]
    mx::array con_lb_finite; // [m]
    mx::array con_ub_finite; // [m]

    // Boolean masks for bound conditions
    mx::array var_lb_inf_mask; // [n] true where var_lb == -inf
    mx::array var_ub_inf_mask; // [n] true where var_ub == +inf
    mx::array con_lb_inf_mask; // [m] true where con_lb == -inf
    mx::array con_ub_inf_mask; // [m] true where con_ub == +inf

    // Rescaling factors
    mx::array var_rescale; // [n]
    mx::array con_rescale; // [m]
    double con_bound_rescale;
    double obj_vec_rescale;

    // Solution state — primal (n-vectors)
    mx::array x_cur;  // current_primal
    mx::array x_pdhg; // pdhg_primal (ergodic avg, stored at major iterations)
    mx::array x_ref;  // reflected_primal
    mx::array x_init; // initial_primal (restart anchor)
    mx::array x_best; // best evaluated primal iterate
    mx::array ATy;    // dual_product = A^T * y

    // Solution state — dual (m-vectors)
    mx::array y_cur;  // current_dual
    mx::array y_pdhg; // pdhg_dual
    mx::array y_ref;  // reflected_dual
    mx::array y_init; // initial_dual
    mx::array y_best; // best evaluated dual iterate
    mx::array Ax;     // primal_product = A * x

    // Working arrays
    mx::array primal_slack; // [m]
    mx::array dual_slack;   // [n]
    mx::array dual_slack_best; // [n], paired with x_best/y_best
    mx::array primal_res;   // [m]
    mx::array dual_res;     // [n]
    mx::array delta_x;      // [n]
    mx::array delta_y;      // [m]

    // Step sizes (scalars on host, used in expressions)
    double step_size;
    double primal_weight;
    double step_size_primal; // = step_size / primal_weight
    double step_size_dual;   // = step_size * primal_weight

    // Iteration counters
    int inner_count;
    int total_count;

    // Residual / fixed-point scalars
    double absolute_primal_residual;
    double relative_primal_residual;
    double absolute_dual_residual;
    double relative_dual_residual;
    double restart_relative_dual_residual;
    double primal_objective_value;
    double dual_objective_value;
    double objective_gap;
    double relative_objective_gap;
    double fixed_point_error;
    double initial_fixed_point_error;
    double last_trial_fixed_point_error;
    double best_primal_dual_residual_gap;
    double best_relative_kkt_error;
    double best_relative_feasibility_error;
    int best_iteration;

    // Primal weight PID state
    double primal_weight_error_sum;
    double primal_weight_last_error;
    double best_primal_weight;
    // HPR-LP-style restart state (restart_policy == 1)
    double hpr_last_gap;
    double hpr_best_gap;
    double hpr_best_weight;

    // Ray infeasibility
    double max_primal_ray_infeasibility;
    double max_dual_ray_infeasibility;
    double primal_ray_linear_objective;
    double dual_ray_objective;

    // Termination
    termination_reason_t termination_reason;

    // Norms for relative computations
    double objective_vector_norm;
    double constraint_bound_norm;

    // Timing
    std::chrono::steady_clock::time_point start_time;
    double cumulative_time_sec;
    double rescaling_time_sec;
    double feasibility_polishing_time_sec;
    int feasibility_iteration;

    // Stream (MLX uses lazy eval; we synchronize explicitly)
    mx::Stream stream;
};

// ---------------------------------------------------------------------------
// MLX Backend Solver
// ---------------------------------------------------------------------------

class MlxPdlpSolver {
  public:
    // Construct from CSR-format LP data.
    MlxPdlpSolver(int num_vars, int num_cons, const int *csr_row_ptr, const int *csr_col_ind,
                  const double *csr_vals, const double *var_lb, const double *var_ub,
                  const double *con_lb, const double *con_ub, const double *objective,
                  double objective_constant, const pdhg_parameters_t *params,
                  mx::Device device = mx::Device::cpu);

    // Construct with optional warm starts in the original, unscaled problem
    // coordinates. A non-null primal_start must contain num_vars finite values;
    // a non-null dual_start must contain num_cons finite values.
    MlxPdlpSolver(int num_vars, int num_cons, const int *csr_row_ptr, const int *csr_col_ind,
                  const double *csr_vals, const double *var_lb, const double *var_ub,
                  const double *con_lb, const double *con_ub, const double *objective,
                  double objective_constant, const pdhg_parameters_t *params,
                  const double *primal_start, const double *dual_start,
                  mx::Device device = mx::Device::cpu);

    // Construct with a complete warm certificate. reduced_cost_start contains
    // num_vars finite values in original, unscaled problem coordinates. This
    // preserves bounded-variable dual objectives across saved checkpoints;
    // callers that only have x/y may use the overload above.
    MlxPdlpSolver(int num_vars, int num_cons, const int *csr_row_ptr, const int *csr_col_ind,
                  const double *csr_vals, const double *var_lb, const double *var_ub,
                  const double *con_lb, const double *con_ub, const double *objective,
                  double objective_constant, const pdhg_parameters_t *params,
                  const double *primal_start, const double *dual_start,
                  const double *reduced_cost_start,
                  mx::Device device = mx::Device::cpu);

    ~MlxPdlpSolver();

    // Run the full PDHG algorithm. Returns heap-allocated result.
    mlxpdlp_result_t *solve();

    // Accessors for debugging / inspection
    const MlxPdlpState &state() const {
        return s_;
    }
    bool expects_sparse_metal_backend() const {
        return sparse_metal_candidate_;
    }
    bool expects_sparse_cpu_backend() const {
        return sparse_cpu_candidate_;
    }

  private:
    MlxPdlpState s_;
    pdhg_parameters_t params_;
    [[maybe_unused]] detail::PresolveContext *presolve_context_ = nullptr;
    int original_num_variables_ = 0;
    int original_num_constraints_ = 0;
    int original_num_nonzeros_ = 0;
    [[maybe_unused]] int presolve_status_ = 0;
    [[maybe_unused]] double presolve_time_sec_ = 0.0;
    bool presolve_solved_ = false;
    bool has_warm_start_ = false;
    bool has_reduced_cost_start_ = false;
    std::vector<double> warm_reduced_cost_;
    double original_objective_constant_ = 0.0;
    std::vector<int> original_row_ptr_;
    std::vector<int> original_col_ind_;
    std::vector<double> original_matrix_values_;
    std::vector<double> original_objective_;
    std::vector<double> original_variable_lower_bound_;
    std::vector<double> original_variable_upper_bound_;
    std::vector<double> original_constraint_lower_bound_;
    std::vector<double> original_constraint_upper_bound_;

    // Unscaled problem actually handed to PDHG. When presolve is enabled this
    // owns the reduced model independently of PSLP's context, allowing the
    // fp64 host continuation to run before postsolve reconstruction.
    int working_num_variables_ = 0;
    int working_num_constraints_ = 0;
    int working_num_nonzeros_ = 0;
    double working_objective_constant_ = 0.0;
    std::vector<int> working_row_ptr_;
    std::vector<int> working_col_ind_;
    std::vector<double> working_matrix_values_;
    std::vector<double> working_objective_;
    std::vector<double> working_variable_lower_bound_;
    std::vector<double> working_variable_upper_bound_;
    std::vector<double> working_constraint_lower_bound_;
    std::vector<double> working_constraint_upper_bound_;

    // CSR data and a CSR representation of its transpose. Sparse Metal values
    // are preconditioned on the host in double precision, then rounded once
    // when the final Metal buffers are materialized. The transpose-source map
    // mirrors those final values into A^T without atomics.
    std::vector<int32_t> sparse_a_row_ptr_host_;
    std::vector<int32_t> sparse_a_col_ind_host_;
    std::vector<double> sparse_a_values_host_;
    std::vector<int32_t> sparse_at_row_ptr_host_;
    std::vector<int32_t> sparse_at_col_ind_host_;
    std::vector<int32_t> sparse_at_source_index_;
    std::vector<double> sparse_con_rescale_host_;
    std::vector<double> sparse_var_rescale_host_;

    mx::array sparse_a_row_ptr_ = _mlx_empty_array();
    mx::array sparse_a_col_ind_ = _mlx_empty_array();
    mx::array sparse_a_values_ = _mlx_empty_array();
    mx::array sparse_a_work_offsets_ = _mlx_empty_array();
    mx::array sparse_a_work_rows_ = _mlx_empty_array();
    int sparse_a_work_item_count_ = 0;
    mx::array sparse_at_row_ptr_ = _mlx_empty_array();
    mx::array sparse_at_col_ind_ = _mlx_empty_array();
    mx::array sparse_at_values_ = _mlx_empty_array();
    mx::array sparse_at_work_offsets_ = _mlx_empty_array();
    mx::array sparse_at_work_rows_ = _mlx_empty_array();
    int sparse_at_work_item_count_ = 0;
    double sparse_frobenius_norm_ = 0.0;
    bool sparse_metal_candidate_ = false;
    bool sparse_cpu_candidate_ = false;
    std::shared_ptr<detail::CpuSparseMatrix> sparse_cpu_matrix_;

    // Early fp64 continuation is triggered only after fp32 first reaches its
    // stricter admission region and then fails to make a substantial KKT
    // reduction over a bounded window. The admissible primal/dual certificate
    // is retained explicitly, so later fp32 oscillations cannot discard the
    // point that was already safe to polish.
    int host_double_handoff_checkpoint_iteration_ = -1;
    double host_double_handoff_checkpoint_kkt_ =
        std::numeric_limits<double>::infinity();
    mx::array host_double_handoff_x_ = _mlx_empty_array();
    mx::array host_double_handoff_y_ = _mlx_empty_array();
    mx::array host_double_handoff_dual_slack_ = _mlx_empty_array();

    // Infeasibility-ray gaps in working (scaled) units. The ratio tests in
    // mlx_check_termination compare residuals and gaps in these consistent
    // units; the public state fields carry the original-unit values.
    double working_dual_ray_objective_ = 0.0;
    double working_primal_ray_objective_ = 0.0;

    // ---- Core linear algebra ----
    mx::array mat_Ax(const mx::array &x);  // A * x
    mx::array mat_ATx(const mx::array &y); // A^T * y
    mx::array sparse_matvec(const mx::array &row_ptr, const mx::array &col_ind,
                            const mx::array &values, const mx::array &work_offsets,
                            const mx::array &work_rows, const mx::array &x, int rows,
                            int work_item_count, SparseMetalSpmvStrategy strategy);
    void capture_sparse_matrix(int rows, int cols, const int *row_ptr, const int *col_ind,
                               const double *values);
    void apply_sparse_scaling(const std::vector<double> &con_scale,
                              const std::vector<double> &var_scale);
    void sparse_ruiz_scaling(int num_iters);
    void sparse_pock_chambolle_scaling(double alpha);
    void publish_sparse_rescaling();
    void prepare_sparse_metal_backend();
    void prepare_sparse_cpu_backend();
    mx::array sparse_cpu_matvec(const mx::array &x, bool transpose, int rows);

    // Fused sparse-Metal PDHG half-steps. `scalars` is a float32 array of
    // {step, reflection_coefficient, halpern_weight, major_flag}.
    std::vector<mx::array> fused_primal_step(const mx::array &scalars);
    std::vector<mx::array> fused_dual_step(const mx::array &scalars);

    // ---- Scalar reductions ----
    double mlx_dot(const mx::array &a, const mx::array &b);
    double mlx_norm2(const mx::array &v);
    double mlx_norm_inf(const mx::array &v);
    double mlx_sum(const mx::array &v);

    // ---- CSR → dense conversion ----
    mx::array csr_to_dense(int rows, int cols, const int *row_ptr, const int *col_ind,
                           const double *vals);

    // ---- Bound preprocessing ----
    void preprocess_bounds(const double *host_var_lb, const double *host_var_ub,
                           const double *host_con_lb, const double *host_con_ub);

    // ---- Preconditioning ----
    void mlx_curtis_reid_scaling(int num_iters);
    void mlx_ruiz_scaling(int num_iters);
    void mlx_pock_chambolle_scaling(double alpha);
    void mlx_bound_objective_scaling();

    // ---- Singular value estimation (power method) ----
    double mlx_estimate_max_singular_value();

    // ---- PDHG iteration sub-steps ----
    void mlx_compute_next_primal(int k_offset, bool is_major);
    // eval_now controls the fused sparse-Metal path: when false, the half-step
    // kernels are only appended to the lazy graph and are materialized by a
    // later eval, batching consecutive iterations into one evaluation.
    void mlx_compute_next_dual(int k_offset, bool is_major, bool eval_now = true);
    // Number of fused iterations to accumulate between evaluations. Keeps the
    // batch graph within a bounded memory footprint.
    int fused_eval_batch_size() const;
    void mlx_compute_fixed_point_error();
    void mlx_compute_residual();
    void mlx_save_best_iterate();
    void mlx_restore_best_iterate();
    void mlx_primal_feasibility_polish();
    void mlx_dual_feasibility_polish();
    void mlx_compute_infeasibility_information();
    void mlx_perform_restart();

    // ---- Termination / restart checks ----
    bool mlx_check_termination();
    bool mlx_should_adaptive_restart();

    // ---- Logging ----
    void mlx_display_header();
    void mlx_display_iteration_stats();
    void mlx_display_final_log();

    // ---- Default parameters ----
    static void set_default_parameters(pdhg_parameters_t *p);

    // ---- Result extraction ----
    mlxpdlp_result_t *extract_result();
    mlxpdlp_result_t *extract_presolve_result();
    void apply_postsolve(mlxpdlp_result_t *result);
    void polish_original_dual_certificate(mlxpdlp_result_t *result);
    double recompute_original_certificate(mlxpdlp_result_t *result);
    void host_double_polish(mlxpdlp_result_t *result, bool working_model = false);
};

// Initialize a parameter struct with mlxPDLP defaults.
void mlxpdlp_set_default_parameters(pdhg_parameters_t *params);

// Release a result returned by MlxPdlpSolver::solve(). Accepts nullptr.
void mlxpdlp_result_free(mlxpdlp_result_t *result);

} // namespace mlxpdlp
