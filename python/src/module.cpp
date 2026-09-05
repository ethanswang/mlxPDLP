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

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>

#include "mlxPDLP/mps_loader.h"
#include "mlxPDLP/solver.h"
#include "mlxPDLP/version.h"

#include <mlx/mlx.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace nb = nanobind;
using namespace mlxpdlp;

using F64Arr = nb::ndarray<const double, nb::shape<-1>, nb::c_contig>;
using I32Arr = nb::ndarray<const int32_t, nb::shape<-1>, nb::c_contig>;

// ---------------------------------------------------------------------------
// numpy helpers
// ---------------------------------------------------------------------------

template <typename Scalar, typename Src>
static nb::object to_numpy(const Src *src, size_t n, const char *dtype) {
    nb::object numpy = nb::module_::import_("numpy");
    nb::object array =
        numpy.attr("empty")(n, nb::arg("dtype") = dtype);
    nb::ndarray<Scalar, nb::numpy, nb::shape<-1>, nb::c_contig> view =
        nb::cast<nb::ndarray<Scalar, nb::numpy, nb::shape<-1>, nb::c_contig>>(
            array);
    if (n > 0)
        std::memcpy(view.data(), src, n * sizeof(Scalar));
    return array;
}

static nb::object to_numpy_f64(const double *src, size_t n) {
    return to_numpy<double>(src, n, "float64");
}

static nb::object to_numpy_i32(const int32_t *src, size_t n) {
    return to_numpy<int32_t>(src, n, "int32");
}

static std::vector<int32_t> as_i32_vector(const I32Arr &array) {
    return std::vector<int32_t>(array.data(), array.data() + array.size());
}

static mx::Device parse_device(const std::string &device) {
    if (device == "cpu")
        return mx::Device::cpu;
    if (device == "gpu" || device == "metal")
        return mx::Device::gpu;
    throw nb::value_error(
        ("invalid device '" + device + "'; expected 'cpu', 'gpu', or 'metal'")
            .c_str());
}

// ---------------------------------------------------------------------------
// Parameters
// ---------------------------------------------------------------------------

static void bind_parameters(nb::module_ &m) {
    nb::class_<termination_criteria_t>(m, "TerminationCriteria",
                                       "Convergence tolerances and budgets.")
        .def("__init__", [](termination_criteria_t *self) {
            pdhg_parameters_t defaults;
            mlxpdlp_set_default_parameters(&defaults);
            new (self) termination_criteria_t(defaults.termination_criteria);
        })
        .def_rw("eps_optimal_relative",
                &termination_criteria_t::eps_optimal_relative,
                "Relative optimality (objective gap) tolerance.")
        .def_rw("eps_feasible_relative",
                &termination_criteria_t::eps_feasible_relative,
                "Relative primal/dual feasibility tolerance.")
        .def_rw("eps_feas_polish_relative",
                &termination_criteria_t::eps_feas_polish_relative,
                "Feasibility-polishing phase tolerance.")
        .def_rw("eps_infeasible_relative",
                &termination_criteria_t::eps_infeasible_relative,
                "Relative ray-certificate residual tolerance (default 1e-14); "
                "must be finite and positive. Independent of Parameters.tolerance.")
        .def_rw("time_sec_limit", &termination_criteria_t::time_sec_limit,
                "Wall-clock budget in seconds.")
        .def_rw("iteration_limit", &termination_criteria_t::iteration_limit,
                "PDHG iteration budget.");

    nb::class_<restart_parameters_t>(m, "RestartParameters",
                                     "Adaptive-restart controller gains.")
        .def(nb::init<>())
        .def_rw("artificial_restart_threshold",
                &restart_parameters_t::artificial_restart_threshold)
        .def_rw("sufficient_reduction_for_restart",
                &restart_parameters_t::sufficient_reduction_for_restart)
        .def_rw("necessary_reduction_for_restart",
                &restart_parameters_t::necessary_reduction_for_restart)
        .def_rw("k_p", &restart_parameters_t::k_p)
        .def_rw("k_i", &restart_parameters_t::k_i)
        .def_rw("k_d", &restart_parameters_t::k_d)
        .def_rw("i_smooth", &restart_parameters_t::i_smooth);

    nb::enum_<norm_type_t>(m, "NormType", nb::is_arithmetic())
        .value("L2", NORM_TYPE_L2)
        .value("L_INF", NORM_TYPE_L_INF);

    nb::class_<pdhg_parameters_t>(m, "Parameters",
                                  "PDHG solver parameters (defaults filled by "
                                  "mlxpdlp_set_default_parameters).")
        .def("__init__", [](pdhg_parameters_t *self) {
            new (self) pdhg_parameters_t();
            mlxpdlp_set_default_parameters(self);
        })
        .def_rw("geometric_mean_iterations",
                &pdhg_parameters_t::geometric_mean_iterations,
                "Geometric-mean scaling passes (0 disables; default 12).")
        .def_rw("curtis_reid_iterations",
                &pdhg_parameters_t::curtis_reid_iterations,
                "Curtis-Reid scaling passes (0 disables).")
        .def_rw("l_inf_ruiz_iterations",
                &pdhg_parameters_t::l_inf_ruiz_iterations,
                "L-infinity Ruiz scaling iterations.")
        .def_rw("has_pock_chambolle_alpha",
                &pdhg_parameters_t::has_pock_chambolle_alpha)
        .def_rw("pock_chambolle_alpha",
                &pdhg_parameters_t::pock_chambolle_alpha)
        .def_rw("bound_objective_rescaling",
                &pdhg_parameters_t::bound_objective_rescaling)
        .def_rw("verbose", &pdhg_parameters_t::verbose)
        .def_rw("termination_evaluation_frequency",
                &pdhg_parameters_t::termination_evaluation_frequency)
        .def_rw("sv_max_iter", &pdhg_parameters_t::sv_max_iter,
                "Power-method singular-value estimation budget.")
        .def_rw("sv_tol", &pdhg_parameters_t::sv_tol,
                "Relative sigma-squared change tolerance over ten iterations.")
        .def_prop_rw(
            "termination_criteria",
            [](pdhg_parameters_t &p) -> termination_criteria_t & {
                return p.termination_criteria;
            },
            [](pdhg_parameters_t &p, const termination_criteria_t &value) {
                p.termination_criteria = value;
            })
        .def_prop_rw(
            "restart_params",
            [](pdhg_parameters_t &p) -> restart_parameters_t & {
                return p.restart_params;
            },
            [](pdhg_parameters_t &p, const restart_parameters_t &value) {
                p.restart_params = value;
            })
        .def_rw("restart_policy", &pdhg_parameters_t::restart_policy,
                "Primal-weight restart policy: 0 = cuPDLPx PID (default), "
                "1 = HPR-LP-style sigma update, 2 = frozen-weight diagnostic.")
        .def_rw("conditional_termination_evaluation",
                &pdhg_parameters_t::conditional_termination_evaluation,
                "Use Metal-adapted cuOpt Stable3-style early termination "
                "checkpoints near convergence for working models up to 262,144 "
                "nonzeros, without changing the configured restart cadence.")
        .def_rw("reflection_coefficient",
                &pdhg_parameters_t::reflection_coefficient)
        .def_rw("feasibility_polishing",
                &pdhg_parameters_t::feasibility_polishing)
        .def_rw("host_double_polishing",
                &pdhg_parameters_t::host_double_polishing)
        .def_rw("host_double_early_handoff",
                &pdhg_parameters_t::host_double_early_handoff)
        .def_rw("host_double_polishing_iteration_limit",
                &pdhg_parameters_t::host_double_polishing_iteration_limit)
        .def_rw("host_double_polishing_time_sec_limit",
                &pdhg_parameters_t::host_double_polishing_time_sec_limit)
        .def_rw("optimality_norm", &pdhg_parameters_t::optimality_norm)
        .def_rw("presolve", &pdhg_parameters_t::presolve,
                "Enable PSLP presolve (incompatible with warm starts).")
        .def_rw("presolve_singleton_columns",
                &pdhg_parameters_t::presolve_singleton_columns)
        .def_rw("presolve_doubleton_equations",
                &pdhg_parameters_t::presolve_doubleton_equations)
        .def_rw("presolve_parallel_rows",
                &pdhg_parameters_t::presolve_parallel_rows)
        .def_rw("presolve_parallel_columns",
                &pdhg_parameters_t::presolve_parallel_columns)
        .def_rw("presolve_dual_fix", &pdhg_parameters_t::presolve_dual_fix)
        .def_rw("presolve_finite_bound_tightening",
                &pdhg_parameters_t::presolve_finite_bound_tightening)
        .def_rw("presolve_primal_propagation",
                &pdhg_parameters_t::presolve_primal_propagation)
        .def_rw("matrix_zero_tol", &pdhg_parameters_t::matrix_zero_tol)
        .def_rw("metal_fused_kernels", &pdhg_parameters_t::metal_fused_kernels)
        .def_prop_rw(
            "tolerance",
            [](pdhg_parameters_t &p) {
                return p.termination_criteria.eps_optimal_relative;
            },
            [](pdhg_parameters_t &p, double value) {
                p.termination_criteria.eps_optimal_relative = value;
                p.termination_criteria.eps_feasible_relative = value;
                p.termination_criteria.eps_feas_polish_relative =
                    std::min(value, 1e-6);
            },
            "Set the optimality and feasibility tolerances together.")
        .def_prop_rw(
            "time_limit_seconds",
            [](pdhg_parameters_t &p) {
                return p.termination_criteria.time_sec_limit;
            },
            [](pdhg_parameters_t &p, double value) {
                p.termination_criteria.time_sec_limit = value;
            })
        .def_prop_rw(
            "iteration_limit",
            [](pdhg_parameters_t &p) {
                return p.termination_criteria.iteration_limit;
            },
            [](pdhg_parameters_t &p, int value) {
                p.termination_criteria.iteration_limit = value;
            });
}

// ---------------------------------------------------------------------------
// SolveResult
// ---------------------------------------------------------------------------

struct SolveResult {
    nb::object primal_solution;
    nb::object dual_solution;
    nb::object reduced_cost;
    int num_variables = 0;
    int num_constraints = 0;
    int num_nonzeros = 0;
    int num_reduced_variables = 0;
    int num_reduced_constraints = 0;
    int num_reduced_nonzeros = 0;
    int total_count = 0;
    double rescaling_time_sec = 0.0;
    double cumulative_time_sec = 0.0;
    double presolve_time = 0.0;
    int presolve_status = 0;
    double absolute_primal_residual = 0.0;
    double relative_primal_residual = 0.0;
    double absolute_dual_residual = 0.0;
    double relative_dual_residual = 0.0;
    double primal_objective_value = 0.0;
    double dual_objective_value = 0.0;
    double objective_gap = 0.0;
    double relative_objective_gap = 0.0;
    double max_primal_ray_infeasibility = 0.0;
    double max_dual_ray_infeasibility = 0.0;
    double primal_ray_linear_objective = 0.0;
    double dual_ray_objective = 0.0;
    int termination_reason = 0;
    double feasibility_polishing_time = 0.0;
    int feasibility_iteration = 0;
    double host_double_polishing_time = 0.0;
    int host_double_polishing_iteration = 0;
    bool host_double_handoff = false;

    static SolveResult from_result(const mlxpdlp_result_t *r) {
        SolveResult out;
        out.primal_solution = to_numpy_f64(r->primal_solution, r->num_variables);
        out.dual_solution = to_numpy_f64(r->dual_solution, r->num_constraints);
        out.reduced_cost = to_numpy_f64(r->reduced_cost, r->num_variables);
        out.num_variables = r->num_variables;
        out.num_constraints = r->num_constraints;
        out.num_nonzeros = r->num_nonzeros;
        out.num_reduced_variables = r->num_reduced_variables;
        out.num_reduced_constraints = r->num_reduced_constraints;
        out.num_reduced_nonzeros = r->num_reduced_nonzeros;
        out.total_count = r->total_count;
        out.rescaling_time_sec = r->rescaling_time_sec;
        out.cumulative_time_sec = r->cumulative_time_sec;
        out.presolve_time = r->presolve_time;
        out.presolve_status = r->presolve_status;
        out.absolute_primal_residual = r->absolute_primal_residual;
        out.relative_primal_residual = r->relative_primal_residual;
        out.absolute_dual_residual = r->absolute_dual_residual;
        out.relative_dual_residual = r->relative_dual_residual;
        out.primal_objective_value = r->primal_objective_value;
        out.dual_objective_value = r->dual_objective_value;
        out.objective_gap = r->objective_gap;
        out.relative_objective_gap = r->relative_objective_gap;
        out.max_primal_ray_infeasibility = r->max_primal_ray_infeasibility;
        out.max_dual_ray_infeasibility = r->max_dual_ray_infeasibility;
        out.primal_ray_linear_objective = r->primal_ray_linear_objective;
        out.dual_ray_objective = r->dual_ray_objective;
        out.termination_reason = static_cast<int>(r->termination_reason);
        out.feasibility_polishing_time = r->feasibility_polishing_time;
        out.feasibility_iteration = r->feasibility_iteration;
        out.host_double_polishing_time = r->host_double_polishing_time;
        out.host_double_polishing_iteration = r->host_double_polishing_iteration;
        out.host_double_handoff = r->host_double_handoff;
        return out;
    }
};

static const char *termination_reason_name(int reason) {
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
    default:
        return "UNSPECIFIED";
    }
}

static void bind_result(nb::module_ &m) {
    nb::class_<SolveResult>(m, "SolveResult",
                            "Solution certificate and termination diagnostics.")
        .def_ro("primal_solution", &SolveResult::primal_solution,
                "Primal variables (numpy.ndarray of shape (num_variables,)).")
        .def_ro("dual_solution", &SolveResult::dual_solution,
                "Dual variables (numpy.ndarray of shape (num_constraints,)).")
        .def_ro("reduced_cost", &SolveResult::reduced_cost,
                "Reduced costs (numpy.ndarray of shape (num_variables,)).")
        .def_ro("num_variables", &SolveResult::num_variables)
        .def_ro("num_constraints", &SolveResult::num_constraints)
        .def_ro("num_nonzeros", &SolveResult::num_nonzeros)
        .def_ro("num_reduced_variables", &SolveResult::num_reduced_variables,
                "Variables after PSLP presolve (original size when presolve "
                "is disabled).")
        .def_ro("num_reduced_constraints", &SolveResult::num_reduced_constraints)
        .def_ro("num_reduced_nonzeros", &SolveResult::num_reduced_nonzeros)
        .def_ro("total_count", &SolveResult::total_count, "PDHG iterations.")
        .def_ro("rescaling_time_sec", &SolveResult::rescaling_time_sec)
        .def_ro("cumulative_time_sec", &SolveResult::cumulative_time_sec,
                "Solver-reported cumulative solve time in seconds.")
        .def_ro("presolve_time", &SolveResult::presolve_time)
        .def_ro("presolve_status", &SolveResult::presolve_status)
        .def_ro("absolute_primal_residual",
                &SolveResult::absolute_primal_residual)
        .def_ro("relative_primal_residual",
                &SolveResult::relative_primal_residual)
        .def_ro("absolute_dual_residual", &SolveResult::absolute_dual_residual)
        .def_ro("relative_dual_residual", &SolveResult::relative_dual_residual)
        .def_prop_rw(
            "primal_objective_value",
            [](SolveResult &r) { return r.primal_objective_value; },
            [](SolveResult &r, double value) { r.primal_objective_value = value; },
            "Primal objective (sign-corrected for maximize models by "
            "solve_mps).")
        .def_prop_rw(
            "dual_objective_value",
            [](SolveResult &r) { return r.dual_objective_value; },
            [](SolveResult &r, double value) { r.dual_objective_value = value; })
        .def_ro("objective_gap", &SolveResult::objective_gap)
        .def_ro("relative_objective_gap", &SolveResult::relative_objective_gap)
        .def_ro("max_primal_ray_infeasibility",
                &SolveResult::max_primal_ray_infeasibility)
        .def_ro("max_dual_ray_infeasibility",
                &SolveResult::max_dual_ray_infeasibility)
        .def_ro("primal_ray_linear_objective",
                &SolveResult::primal_ray_linear_objective)
        .def_ro("dual_ray_objective", &SolveResult::dual_ray_objective)
        .def_ro("termination_reason", &SolveResult::termination_reason,
                "Numeric termination reason (see mlxpdlp.TerminationReason).")
        .def_prop_ro(
            "termination_reason_name",
            [](const SolveResult &r) {
                return termination_reason_name(r.termination_reason);
            },
            "Human-readable termination reason.")
        .def_ro("feasibility_polishing_time",
                &SolveResult::feasibility_polishing_time)
        .def_ro("feasibility_iteration", &SolveResult::feasibility_iteration)
        .def_ro("host_double_polishing_time",
                &SolveResult::host_double_polishing_time)
        .def_ro("host_double_polishing_iteration",
                &SolveResult::host_double_polishing_iteration)
        .def_ro("host_double_handoff", &SolveResult::host_double_handoff)
        .def("__repr__", [](const SolveResult &r) {
            char buffer[256];
            std::snprintf(buffer, sizeof(buffer),
                          "SolveResult(termination=%s, primal_obj=%.8e, "
                          "dual_obj=%.8e, iterations=%d, rel_pr=%.3e, "
                          "rel_du=%.3e, rel_gap=%.3e)",
                          termination_reason_name(r.termination_reason),
                          r.primal_objective_value, r.dual_objective_value,
                          r.total_count, r.relative_primal_residual,
                          r.relative_dual_residual, r.relative_objective_gap);
            return std::string(buffer);
        });
}

// ---------------------------------------------------------------------------
// Solver
// ---------------------------------------------------------------------------

static void bind_solver(nb::module_ &m) {
    nb::enum_<termination_reason_t>(m, "TerminationReason", nb::is_arithmetic())
        .value("UNSPECIFIED", TERMINATION_REASON_UNSPECIFIED)
        .value("OPTIMAL", TERMINATION_REASON_OPTIMAL)
        .value("PRIMAL_INFEASIBLE", TERMINATION_REASON_PRIMAL_INFEASIBLE)
        .value("DUAL_INFEASIBLE", TERMINATION_REASON_DUAL_INFEASIBLE)
        .value("INFEASIBLE_OR_UNBOUNDED",
               TERMINATION_REASON_INFEASIBLE_OR_UNBOUNDED)
        .value("TIME_LIMIT", TERMINATION_REASON_TIME_LIMIT)
        .value("ITERATION_LIMIT", TERMINATION_REASON_ITERATION_LIMIT)
        .value("FEAS_POLISH_SUCCESS", TERMINATION_REASON_FEAS_POLISH_SUCCESS)
        .value("HOST_DOUBLE_HANDOFF", TERMINATION_REASON_HOST_DOUBLE_HANDOFF);

    nb::class_<MlxPdlpSolver>(m, "Solver",
                              "PDHG solver for minimization problems\n\n"
                              "    minimize    c^T x + constant\n"
                              "    subject to  constraint_lb <= A x <= "
                              "constraint_ub\n"
                              "                variable_lb   <= x <= "
                              "variable_ub\n\n"
                              "CPU execution is float64 throughout; Metal is "
                              "float32 with optional host-float64 polishing.")
        .def(
            "__init__",
            [](MlxPdlpSolver *self, int num_variables, int num_constraints,
               const I32Arr &row_ptr, const I32Arr &col_indices,
               const F64Arr &values,
               const std::optional<F64Arr> &variable_lower_bounds,
               const std::optional<F64Arr> &variable_upper_bounds,
               const std::optional<F64Arr> &constraint_lower_bounds,
               const std::optional<F64Arr> &constraint_upper_bounds,
               const F64Arr &objective, double objective_constant,
               pdhg_parameters_t *parameters,
               const std::optional<F64Arr> &primal_start,
               const std::optional<F64Arr> &dual_start,
               const std::optional<F64Arr> &reduced_cost_start,
               const std::string &device) {
                // --- input validation (mirrors the C++ constructor checks,
                // but with Python-friendly errors before any reads) ---
                if (num_variables < 0 || num_constraints < 0)
                    throw nb::value_error(
                        "num_variables and num_constraints must be non-negative");
                if (row_ptr.size() != static_cast<size_t>(num_constraints) + 1)
                    throw nb::value_error(
                        "row_ptr must have num_constraints + 1 entries");
                if (row_ptr(0) != 0)
                    throw nb::value_error("row_ptr[0] must be 0");
                const int64_t nnz = row_ptr(num_constraints);
                if (nnz < 0)
                    throw nb::value_error("row_ptr must be non-decreasing");
                if (col_indices.size() != static_cast<size_t>(nnz) ||
                    values.size() != static_cast<size_t>(nnz))
                    throw nb::value_error(
                        "col_indices and values must both have row_ptr[-1] "
                        "entries");
                for (size_t i = 0; i < row_ptr.size() - 1; ++i)
                    if (row_ptr(i) > row_ptr(i + 1))
                        throw nb::value_error(
                            "row_ptr must be non-decreasing");
                if (objective.size() != static_cast<size_t>(num_variables))
                    throw nb::value_error(
                        "objective must have num_variables entries");
                auto check_bounds =
                    [&](const std::optional<F64Arr> &array, size_t size,
                        const char *name) {
                        if (array && array->size() != size)
                            throw nb::value_error(
                                (std::string(name) +
                                 " must have " + std::to_string(size) +
                                 " entries")
                                    .c_str());
                    };
                check_bounds(variable_lower_bounds, num_variables,
                             "variable_lower_bounds");
                check_bounds(variable_upper_bounds, num_variables,
                             "variable_upper_bounds");
                check_bounds(constraint_lower_bounds, num_constraints,
                             "constraint_lower_bounds");
                check_bounds(constraint_upper_bounds, num_constraints,
                             "constraint_upper_bounds");
                check_bounds(primal_start, num_variables, "primal_start");
                check_bounds(dual_start, num_constraints, "dual_start");
                check_bounds(reduced_cost_start, num_variables,
                             "reduced_cost_start");
                if (parameters && parameters->geometric_mean_iterations < 0)
                    throw nb::value_error(
                        "geometric_mean_iterations must be non-negative");
                const bool has_warm_start =
                    primal_start || dual_start || reduced_cost_start;
                if (parameters && parameters->presolve && has_warm_start)
                    throw nb::value_error(
                        "warm starts cannot be combined with PSLP presolve; "
                        "set parameters.presolve = False");
                const mx::Device mx_device = parse_device(device);
                if (mx_device.type == mx::Device::gpu &&
                    !mx::is_available(mx::Device::gpu))
                    throw std::runtime_error(
                        "the Metal GPU device is not available in this MLX "
                        "build; use device='cpu'");

                std::vector<int32_t> csr_row_ptr = as_i32_vector(row_ptr);
                std::vector<int32_t> csr_col_ind = as_i32_vector(col_indices);
                const double *var_lb = variable_lower_bounds
                                           ? variable_lower_bounds->data()
                                           : nullptr;
                const double *var_ub = variable_upper_bounds
                                           ? variable_upper_bounds->data()
                                           : nullptr;
                const double *con_lb = constraint_lower_bounds
                                           ? constraint_lower_bounds->data()
                                           : nullptr;
                const double *con_ub = constraint_upper_bounds
                                           ? constraint_upper_bounds->data()
                                           : nullptr;
                const double *ps = primal_start ? primal_start->data() : nullptr;
                const double *ds = dual_start ? dual_start->data() : nullptr;
                const double *rcs =
                    reduced_cost_start ? reduced_cost_start->data() : nullptr;

                if (rcs)
                    new (self) MlxPdlpSolver(
                        num_variables, num_constraints, csr_row_ptr.data(),
                        csr_col_ind.data(), values.data(), var_lb, var_ub,
                        con_lb, con_ub, objective.data(), objective_constant,
                        parameters, ps, ds, rcs, mx_device);
                else if (ps || ds)
                    new (self) MlxPdlpSolver(
                        num_variables, num_constraints, csr_row_ptr.data(),
                        csr_col_ind.data(), values.data(), var_lb, var_ub,
                        con_lb, con_ub, objective.data(), objective_constant,
                        parameters, ps, ds, mx_device);
                else
                    new (self) MlxPdlpSolver(
                        num_variables, num_constraints, csr_row_ptr.data(),
                        csr_col_ind.data(), values.data(), var_lb, var_ub,
                        con_lb, con_ub, objective.data(), objective_constant,
                        parameters, mx_device);
            },
            nb::arg("num_variables"), nb::arg("num_constraints"),
            nb::arg("row_ptr"), nb::arg("col_indices"), nb::arg("values"),
            nb::arg("variable_lower_bounds") = nb::none(),
            nb::arg("variable_upper_bounds") = nb::none(),
            nb::arg("constraint_lower_bounds") = nb::none(),
            nb::arg("constraint_upper_bounds") = nb::none(),
            nb::arg("objective"), nb::arg("objective_constant") = 0.0,
            nb::arg("parameters") = nb::none(),
            nb::arg("primal_start") = nb::none(),
            nb::arg("dual_start") = nb::none(),
            nb::arg("reduced_cost_start") = nb::none(),
            nb::arg("device") = "cpu")
        .def("solve",
             [](MlxPdlpSolver &self) {
                 mlxpdlp_result_t *raw = self.solve();
                 if (!raw)
                     throw std::runtime_error("solver returned no result");
                 std::unique_ptr<mlxpdlp_result_t,
                                 decltype(&mlxpdlp_result_free)>
                     guard(raw, mlxpdlp_result_free);
                 // Re-acquire the GIL: the result conversion below uses the
                 // Python C API (numpy arrays).
                 nb::gil_scoped_acquire gil;
                 return SolveResult::from_result(raw);
             },
             nb::call_guard<nb::gil_scoped_release>(),
             "Run PDHG and return a SolveResult.")
        .def("expects_sparse_metal_backend",
             &MlxPdlpSolver::expects_sparse_metal_backend,
             "True when this problem shape selects the CSR Metal backend.")
        .def("expects_sparse_cpu_backend",
             &MlxPdlpSolver::expects_sparse_cpu_backend,
             "True when this problem shape selects the Accelerate sparse CPU "
             "backend.");
}

// ---------------------------------------------------------------------------
// MPS loading
// ---------------------------------------------------------------------------

struct MpsProblem {
    std::unique_ptr<mlxpdlp_mps_problem_t, decltype(&mlxpdlp_mps_problem_free)>
        handle{nullptr, mlxpdlp_mps_problem_free};
    nb::object row_ptr;
    nb::object col_ind;
    nb::object values;
    nb::object variable_lb;
    nb::object variable_ub;
    nb::object constraint_lb;
    nb::object constraint_ub;
    nb::object objective;
    int num_variables = 0;
    int num_constraints = 0;
    int num_nonzeros = 0;
    int maximize = 0;
    double objective_constant = 0.0;
};

static void bind_mps(nb::module_ &m) {
    nb::class_<MpsProblem>(
        m, "MpsProblem",
        "Parsed MPS model: CSR arrays, bounds, and objective as numpy "
        "arrays. The solve_mps helper handles the maximize convention.")
        .def_ro("num_variables", &MpsProblem::num_variables)
        .def_ro("num_constraints", &MpsProblem::num_constraints)
        .def_ro("num_nonzeros", &MpsProblem::num_nonzeros)
        .def_ro("maximize", &MpsProblem::maximize,
                "1 when the model objective is a maximization.")
        .def_ro("objective_constant", &MpsProblem::objective_constant)
        .def_ro("row_ptr", &MpsProblem::row_ptr,
                "CSR row pointers (int32, num_constraints + 1).")
        .def_ro("col_ind", &MpsProblem::col_ind, "CSR column indices (int32).")
        .def_ro("values", &MpsProblem::values, "CSR values (float64).")
        .def_ro("variable_lb", &MpsProblem::variable_lb)
        .def_ro("variable_ub", &MpsProblem::variable_ub)
        .def_ro("constraint_lb", &MpsProblem::constraint_lb)
        .def_ro("constraint_ub", &MpsProblem::constraint_ub)
        .def_ro("objective", &MpsProblem::objective)
        .def("__repr__", [](const MpsProblem &p) {
            char buffer[192];
            std::snprintf(buffer, sizeof(buffer),
                          "MpsProblem(variables=%d, constraints=%d, "
                          "nonzeros=%d, maximize=%d)",
                          p.num_variables, p.num_constraints, p.num_nonzeros,
                          p.maximize);
            return std::string(buffer);
        });

    m.def(
        "load_mps",
        [](const std::string &path) {
            mlxpdlp_mps_problem_t *raw = mlxpdlp_mps_problem_load(path.c_str());
            if (!raw)
                throw std::runtime_error("failed to load MPS file: " + path);
            auto problem = std::make_shared<MpsProblem>();
            problem->handle.reset(raw);
            problem->num_variables = raw->num_variables;
            problem->num_constraints = raw->num_constraints;
            problem->num_nonzeros = raw->num_nonzeros;
            problem->maximize = raw->maximize;
            problem->objective_constant = raw->objective_constant;
            problem->row_ptr = to_numpy_i32(raw->row_ptr, raw->num_constraints + 1);
            problem->col_ind = to_numpy_i32(raw->col_ind, raw->num_nonzeros);
            problem->values = to_numpy_f64(raw->values, raw->num_nonzeros);
            problem->variable_lb = to_numpy_f64(raw->variable_lb, raw->num_variables);
            problem->variable_ub = to_numpy_f64(raw->variable_ub, raw->num_variables);
            problem->constraint_lb = to_numpy_f64(raw->constraint_lb, raw->num_constraints);
            problem->constraint_ub = to_numpy_f64(raw->constraint_ub, raw->num_constraints);
            problem->objective = to_numpy_f64(raw->objective, raw->num_variables);
            return problem;
        },
        nb::arg("path"), "Load a plain or gzip-compressed MPS file.");
}

// ---------------------------------------------------------------------------
// Module
// ---------------------------------------------------------------------------

NB_MODULE(_core, m) {
    m.doc() = "mlxPDLP core extension: a PDHG linear-programming solver on "
              "Apple MLX devices (CPU float64 / Metal float32).";

    m.attr("__version__") = MLXPDLP_VERSION_STRING;

    m.def("has_gpu", []() { return mx::is_available(mx::Device::gpu); },
          "True when this MLX build exposes a usable GPU (Metal) device.");
    m.def("version", []() { return std::string(MLXPDLP_VERSION_STRING); },
          "mlxPDLP library version.");

    bind_parameters(m);
    bind_result(m);
    bind_solver(m);
    bind_mps(m);
}
