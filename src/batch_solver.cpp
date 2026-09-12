// Copyright 2026 Ethan Wang <ethanshurui.wang@gmail.com>
// SPDX-License-Identifier: Apache-2.0
#include "mlxPDLP/batch_solver.h"
#include "shared_matrix.h"
#include "metal_spmm.h"
#include "metal_minor_batch.h"
#include "pdhg_control.h"
#include <algorithm>
#include <climits>
#include <mutex>
#include <stdexcept>

namespace mlxpdlp {
namespace {
using Clock = std::chrono::steady_clock;
double seconds(Clock::time_point t) { return std::chrono::duration<double>(Clock::now() - t).count(); }
const double *data(const std::vector<double> &v) { return v.empty() ? nullptr : v.data(); }
const double *data(const std::optional<std::vector<double>> &v) { return v ? v->data() : nullptr; }
size_t checked_add(size_t a, size_t b) {
    if (a > SIZE_MAX - b) throw std::length_error("batch resident memory estimate overflow");
    return a + b;
}
size_t checked_mul(size_t a, size_t b) {
    if (b && a > SIZE_MAX / b) throw std::length_error("batch resident memory estimate overflow");
    return a * b;
}
void validate_member(const BatchProblem &p, int n, int m, size_t index, bool presolve) {
    const std::string prefix = "member[" + std::to_string(index) + "].";
    auto vector = [&](const std::vector<double> &v, int size, const char *name,
                      bool optional, int bound) {
        if (optional && v.empty()) return;
        if (v.size() != static_cast<size_t>(size))
            throw std::invalid_argument(prefix + name + ": incorrect length");
        for (double value : v) {
            if (std::isnan(value) || (bound == 0 && !std::isfinite(value)) ||
                (bound < 0 && value == INFINITY) || (bound > 0 && value == -INFINITY))
                throw std::invalid_argument(prefix + name + ": invalid value");
        }
    };
    vector(p.objective, n, "objective", false, 0);
    if (!std::isfinite(p.objective_constant))
        throw std::invalid_argument(prefix + "objective_constant: expected finite value");
    vector(p.variable_lower_bounds, n, "variable_lower_bounds", true, -1);
    vector(p.variable_upper_bounds, n, "variable_upper_bounds", true, 1);
    vector(p.constraint_lower_bounds, m, "constraint_lower_bounds", true, -1);
    vector(p.constraint_upper_bounds, m, "constraint_upper_bounds", true, 1);
    auto bounds = [&](const auto &lo, const auto &hi, const char *field) {
        if (!lo.empty() && !hi.empty()) for (size_t i = 0; i < lo.size(); ++i)
            if (lo[i] > hi[i]) throw std::invalid_argument(prefix + field + ": lower exceeds upper");
    };
    bounds(p.variable_lower_bounds, p.variable_upper_bounds, "variable_bounds");
    bounds(p.constraint_lower_bounds, p.constraint_upper_bounds, "constraint_bounds");
    if (p.primal_start) vector(*p.primal_start, n, "primal_start", false, 0);
    if (p.dual_start) vector(*p.dual_start, m, "dual_start", false, 0);
    if (p.reduced_cost_start) vector(*p.reduced_cost_start, n, "reduced_cost_start", false, 0);
    if (presolve && (p.primal_start || p.dual_start || p.reduced_cost_start))
        throw std::invalid_argument(prefix + "warm_start: incompatible with presolve");
}
void validate_parameters(const pdhg_parameters_t &p) {
    const auto &t = p.termination_criteria;
    for (double value : {t.eps_optimal_relative, t.eps_feasible_relative, t.eps_infeasible_relative,
                         t.eps_feas_polish_relative})
        if (!std::isfinite(value) || value <= 0) throw std::invalid_argument("parameters: invalid tolerance");
    if (t.iteration_limit < 0 || std::isnan(t.time_sec_limit) || t.time_sec_limit < 0 ||
        p.geometric_mean_iterations < 0 || p.curtis_reid_iterations < 0 || p.l_inf_ruiz_iterations < 0 ||
        p.termination_evaluation_frequency <= 0 || p.sv_max_iter < 0 ||
        !std::isfinite(p.sv_tol) || p.sv_tol <= 0 || p.restart_policy < 0 || p.restart_policy > 1 ||
        (p.optimality_norm != NORM_TYPE_L2 && p.optimality_norm != NORM_TYPE_L_INF) ||
        !std::isfinite(p.reflection_coefficient) || p.reflection_coefficient <= 0 || p.reflection_coefficient > 1)
        throw std::invalid_argument("parameters: invalid algorithm setting");
    const auto &r = p.restart_params;
    for (double value : {r.k_p, r.k_i, r.k_d, r.i_smooth, r.artificial_restart_threshold,
                         r.sufficient_reduction_for_restart, r.necessary_reduction_for_restart})
        if (!std::isfinite(value)) throw std::invalid_argument("parameters.restart_params: non-finite value");
    if (r.i_smooth < 0 || r.i_smooth > 1 || (p.has_pock_chambolle_alpha &&
        (!std::isfinite(p.pock_chambolle_alpha) || p.pock_chambolle_alpha < 0 || p.pock_chambolle_alpha > 2)))
        throw std::invalid_argument("parameters: invalid scaling or restart setting");
}
bool same_preparation(const pdhg_parameters_t &a, const pdhg_parameters_t &b) {
    return a.geometric_mean_iterations == b.geometric_mean_iterations &&
           a.curtis_reid_iterations == b.curtis_reid_iterations &&
           a.l_inf_ruiz_iterations == b.l_inf_ruiz_iterations &&
           a.has_pock_chambolle_alpha == b.has_pock_chambolle_alpha &&
           (!a.has_pock_chambolle_alpha || a.pock_chambolle_alpha == b.pock_chambolle_alpha) &&
           a.sv_max_iter == b.sv_max_iter && a.sv_tol == b.sv_tol &&
           a.conservative_step_size == b.conservative_step_size;
}
OwnedSolveResult unstarted(int n, int m, int nnz) {
    OwnedSolveResult r(new mlxpdlp_result_t{});
    r->num_variables = r->num_reduced_variables = n;
    r->num_constraints = r->num_reduced_constraints = m;
    r->num_nonzeros = r->num_reduced_nonzeros = nnz;
    r->termination_reason = TERMINATION_REASON_TIME_LIMIT;
    r->relative_primal_residual = r->absolute_primal_residual = INFINITY;
    r->relative_dual_residual = r->absolute_dual_residual = INFINITY;
    r->relative_objective_gap = r->objective_gap = INFINITY;
    return r;
}
mx::array lane(const mx::array &a, int j) {
    // A contiguous owned lane prevents best certificates from retaining a full
    // historical B-wide allocation for each independently improving member.
    return mx::copy(mx::reshape(mx::slice(a, {0, j}, {a.shape(0), j + 1}), {a.shape(0)}));
}
mx::array pack(const std::vector<std::unique_ptr<MlxPdlpSolver>> &solvers,
               mx::array MlxPdlpState::*field, int padded) {
    std::vector<mx::array> values;
    for (const auto &p : solvers) values.push_back(p->state().*field);
    for (int j = static_cast<int>(values.size()); j < padded; ++j)
        values.push_back(mx::zeros_like(values[0]));
    return mx::contiguous(mx::stack(values, 1));
}
struct Workspace {
    mx::array x, xr, xp, xi, z, y, yr, yp, yi, vl, vu, cl, cu, obj;
    Workspace(const std::vector<std::unique_ptr<MlxPdlpSolver>> &s, int width)
        : x(pack(s, &MlxPdlpState::x_cur, width)), xr(pack(s, &MlxPdlpState::x_ref, width)),
          xp(pack(s, &MlxPdlpState::x_pdhg, width)), xi(pack(s, &MlxPdlpState::x_init, width)),
          z(pack(s, &MlxPdlpState::dual_slack, width)), y(pack(s, &MlxPdlpState::y_cur, width)),
          yr(pack(s, &MlxPdlpState::y_ref, width)), yp(pack(s, &MlxPdlpState::y_pdhg, width)),
          yi(pack(s, &MlxPdlpState::y_init, width)), vl(pack(s, &MlxPdlpState::var_lb, width)),
          vu(pack(s, &MlxPdlpState::var_ub, width)), cl(pack(s, &MlxPdlpState::con_lb, width)),
          cu(pack(s, &MlxPdlpState::con_ub, width)), obj(pack(s, &MlxPdlpState::obj, width)) {}
};
} // namespace

namespace detail {
struct BatchDriver {
    int n, m, nnz;
    pdhg_parameters_t parameters;
    mx::Device device;
    std::unique_ptr<MlxPdlpSolver> prepared;
    mx::Stream submission_stream{0, mx::Device::cpu};
    std::timed_mutex mutex;
    double preparation_sec;
    size_t matrix_bytes;
    int max_row_a = 0, max_row_at = 0;

    BatchDriver(int n_, int m_, const int *rp, const int *ci, const double *v,
                 const pdhg_parameters_t *p, mx::Device d) : n(n_), m(m_), nnz(0), device(d) {
        const auto start = Clock::now();
        if (p) parameters = *p; else mlxpdlp_set_default_parameters(&parameters);
        validate_parameters(parameters);
        if (n < 0 || m < 0 || !rp || rp[0] != 0) throw std::invalid_argument("matrix: invalid dimensions or row_ptr");
        nnz = rp[m];
        if (nnz < 0 || (nnz && (!ci || !v))) throw std::invalid_argument("matrix: invalid CSR arrays");
        for (int r = 0; r < m; ++r) if (rp[r] < 0 || rp[r] > rp[r+1] || rp[r+1] > nnz)
            throw std::invalid_argument("matrix.row_ptr: must be monotone in [0, nnz]");
        for (int k = 0; k < nnz; ++k) if (ci[k] < 0 || ci[k] >= n || !std::isfinite(v[k]))
            throw std::invalid_argument("matrix: invalid column index or coefficient");
        auto prep_params = parameters;
        prep_params.presolve = false;
        prep_params.bound_objective_rescaling = false;
        prep_params.verbose = false;
        prep_params.host_double_residual_evaluation = false;
        prepared.reset(new MlxPdlpSolver(n, m, rp, ci, v, nullptr, nullptr, nullptr, nullptr,
                        nullptr, 0, &prep_params, nullptr, nullptr, nullptr, device, nullptr, true));
        mx::StreamContext context(prepared->s_.stream);
        prepared->initialize_solve();
        prepared->mlx_operator_norm_upper_bound();
        auto profile = [](const auto &rows) {
            int longest = 0;
            for (size_t i = 1; i < rows.size(); ++i) longest = std::max(longest, rows[i] - rows[i-1]);
            return longest;
        };
        max_row_a = profile(prepared->matrix_->sparse_a_row_ptr_host_);
        max_row_at = profile(prepared->matrix_->sparse_at_row_ptr_host_);
        matrix_bytes = prepared->matrix_->resident_bytes() + prepared->s_.A.nbytes() + prepared->s_.AT.nbytes();
        // The preparation solver retains only O(n+m) scratch, counted here.
        matrix_bytes = checked_add(matrix_bytes, checked_mul(size_t(n) + m, 256));
        preparation_sec = seconds(start);
    }

    std::unique_ptr<MlxPdlpSolver> make_solver(const BatchProblem &p, const pdhg_parameters_t &params,
                                              bool reuse) {
        auto &a = *prepared->matrix_;
        return std::unique_ptr<MlxPdlpSolver>(new MlxPdlpSolver(n, m, a.original_row_ptr_.data(),
            a.original_col_ind_.data(), a.original_matrix_values_.data(), data(p.variable_lower_bounds),
            data(p.variable_upper_bounds), data(p.constraint_lower_bounds), data(p.constraint_upper_bounds),
            p.objective.data(), p.objective_constant, &params, data(p.primal_start), data(p.dual_start),
            data(p.reduced_cost_start), device, reuse ? prepared.get() : nullptr, false));
    }

    int reduction(bool transpose, int tile) const {
        const int max_length = transpose ? max_row_at : max_row_a;
        return max_length <= 16 ? 1 : 32 / tile;
    }
    mx::array product(const mx::array &v, const mx::array &active, bool transpose, int tile) {
        auto &a = *prepared->matrix_;
        return metal_spmm(transpose ? a.sparse_at_row_ptr_ : a.sparse_a_row_ptr_,
                          transpose ? a.sparse_at_col_ind_ : a.sparse_a_col_ind_,
                          transpose ? a.sparse_at_values_ : a.sparse_a_values_, v, active,
                          transpose ? n : m, tile, reduction(transpose, tile), submission_stream);
    }
    void advance(Workspace &w, const mx::array &active, const mx::array &coeff, bool major, int tile) {
        auto &a = *prepared->matrix_;
        std::vector<mx::array> input{a.sparse_at_row_ptr_, a.sparse_at_col_ind_, a.sparse_at_values_, w.y, active,
                                    w.x, w.xi, w.xr, w.vl, w.vu, w.obj, coeff};
        if (major) { input.push_back(w.xp); input.push_back(w.z); }
        auto x = metal_spmm_dispatch(input, n, tile, reduction(true, tile), 1, major, submission_stream);
        w.x = x[0]; w.xr = x[1];
        if (major) { w.xp = x[2]; w.z = x[3]; }
        input = {a.sparse_a_row_ptr_, a.sparse_a_col_ind_, a.sparse_a_values_, w.xr, active,
                 w.y, w.yi, w.yr, w.cl, w.cu, w.obj, coeff};
        if (major) input.push_back(w.yp);
        auto y = metal_spmm_dispatch(input, m, tile, reduction(false, tile), 2, major, submission_stream);
        w.y = y[0]; w.yr = y[1];
        if (major) w.yp = y[2];
    }

    bool audit_optimal(MlxPdlpSolver &solver) {
        std::vector<double> x(n), y(m), z(n);
        mlxpdlp_result_t result{};
        result.num_variables = n; result.num_constraints = m; result.num_nonzeros = nnz;
        result.primal_solution = x.data(); result.dual_solution = y.data(); result.reduced_cost = z.data();
        result.termination_reason = TERMINATION_REASON_OPTIMAL;
        solver.copy_unscaled_certificate(x.data(), y.data(), z.data());
        solver.recompute_original_certificate(&result, false, false);
        return result.termination_reason == TERMINATION_REASON_OPTIMAL;
    }

    bool audit_ray(MlxPdlpSolver &solver, BatchMemberResult &member, bool primal) {
        auto &s = solver.s_;
        mx::eval(s.delta_x, s.delta_y);
        std::vector<double> ray(primal ? n : m);
        const auto &delta = primal ? s.delta_x : s.delta_y;
        const auto &scale = primal ? prepared->matrix_->sparse_var_rescale_host_ : prepared->matrix_->sparse_con_rescale_host_;
        const double normalization = primal ? s.con_bound_rescale : s.obj_vec_rescale;
        double largest = 0;
        for (size_t i=0; i<ray.size(); ++i) {
            ray[i] = (delta.dtype()==mx::float64 ? delta.data<double>()[i] : delta.data<float>()[i]) / scale[i] / normalization;
            if (!std::isfinite(ray[i])) return false;
            largest = std::max(largest, std::abs(ray[i]));
        }
        if (!(largest > 0)) return false;
        for (auto &value : ray) value /= largest;
        const auto &a = *prepared->matrix_;
        std::vector<long double> product_values(primal ? m : n, 0);
        for (int row=0; row<m; ++row) for (int k=a.original_row_ptr_[row]; k<a.original_row_ptr_[row+1]; ++k) {
            const int col=a.original_col_ind_[k];
            if (primal) product_values[row] += static_cast<long double>(a.original_matrix_values_[k])*ray[col];
            else product_values[col] += static_cast<long double>(a.original_matrix_values_[k])*ray[row];
        }
        long double gap=0, violation=0;
        auto recession = [](long double value, double lo, double hi) {
            long double v=0;
            if (std::isfinite(lo)) v=std::max(v,-value);
            if (std::isfinite(hi)) v=std::max(v,value);
            return v;
        };
        if (primal) {
            for (int i=0;i<n;++i) {
                gap += static_cast<long double>(solver.original_objective_[i])*ray[i];
                violation=std::max(violation,recession(ray[i],solver.original_variable_lower_bound_[i],solver.original_variable_upper_bound_[i]));
            }
            for (int i=0;i<m;++i)
                violation=std::max(violation,recession(product_values[i],solver.original_constraint_lower_bound_[i],solver.original_constraint_upper_bound_[i]));
            gap=-gap;
        } else {
            for (int i=0;i<m;++i) if (ray[i]!=0) {
                const double bound=ray[i]>0?solver.original_constraint_lower_bound_[i]:solver.original_constraint_upper_bound_[i];
                if (!std::isfinite(bound)) return false;
                gap += static_cast<long double>(bound)*ray[i];
            }
            for (int i=0;i<n;++i) if (product_values[i]!=0) {
                const double bound=product_values[i]>0?solver.original_variable_upper_bound_[i]:solver.original_variable_lower_bound_[i];
                if (!std::isfinite(bound)) { violation=std::max(violation,std::abs(product_values[i])); continue; }
                gap -= static_cast<long double>(bound)*product_values[i];
            }
        }
        const double tolerance=solver.params_.termination_criteria.eps_infeasible_relative;
        const double norm=primal?s.objective_vector_norm:s.constraint_bound_norm;
        if (!std::isfinite(gap)||!std::isfinite(violation)||gap<=tolerance*(1+norm)||violation>tolerance*gap) return false;
        if (primal) {
            member.primal_ray=std::move(ray);
            s.primal_ray_linear_objective=-double(gap);s.max_primal_ray_infeasibility=double(violation);
            s.termination_reason=TERMINATION_REASON_DUAL_INFEASIBLE;
        } else {
            member.dual_ray=std::move(ray);
            s.dual_ray_objective=double(gap);s.max_dual_ray_infeasibility=double(violation);
            s.termination_reason=TERMINATION_REASON_PRIMAL_INFEASIBLE;
        }
        return true;
    }

    void run_shared(std::vector<std::unique_ptr<MlxPdlpSolver>> &solvers, BatchResult &output,
                     const BatchOptions &options, Clock::time_point entry, size_t begin) {
        const int count = static_cast<int>(solvers.size());
        const int width = ((count + options.lp_tile_width - 1) / options.lp_tile_width) * options.lp_tile_width;
        const int frequency = solvers[0]->params_.termination_evaluation_frequency;
        std::vector<int> active(width, 0), fresh(count, 0), conditional(count, 0);
        for (int j = 0; j < count; ++j) active[j] = 1;
        auto active_array = mx::array(active.data(), {width}, mx::int32);
        while (std::any_of(active.begin(), active.end(), [](int v) { return v != 0; })) {
            for (int j=0;j<count;++j) if (active[j] &&
                (seconds(entry)>=options.time_sec_limit ||
                 seconds(solvers[j]->s_.start_time)>=solvers[j]->params_.termination_criteria.time_sec_limit)) {
                solvers[j]->s_.termination_reason=TERMINATION_REASON_TIME_LIMIT;
                active[j]=0;
                finish_member(*solvers[j],output,begin+j,entry);
            }
            if (std::none_of(active.begin(),active.end(),[](int v) {return v!=0;})) break;
            active_array=mx::array(active.data(),{width},mx::int32);
            int distance = INT_MAX;
            std::vector<int> member_due(count, INT_MAX);
            for (int j = 0; j < count; ++j) if (active[j]) {
                auto &s = solvers[j]->s_;
                const auto &params = solvers[j]->params_;
                int due = frequency - s.total_count % frequency;
                if (conditional[j]) {
                    int stride = 100;
                    for (int threshold = 10000; threshold <= s.total_count && stride <= 10000000; threshold *= 10) {
                        stride *= 10;
                        if (threshold > INT_MAX / 10) break;
                    }
                    due = std::min(due, stride - s.total_count % stride);
                }
                member_due[j] = std::min(due, params.termination_criteria.iteration_limit - s.total_count);
                distance = std::min(distance, member_due[j]);
            }
            if (distance <= 0) break;
            const auto packing = Clock::now();
            Workspace w(solvers, width);
            mx::eval(w.x, w.xr, w.xp, w.xi, w.z, w.y, w.yr, w.yp, w.yi, w.vl, w.vu, w.cl, w.cu, w.obj);
            output.packing_time_sec += seconds(packing);
            const auto iteration_start = Clock::now();
            int advanced = 0;
            auto coefficients = [&](int k) {
                std::vector<float> coeff(4 * width, 0.0f);
                for (int j = 0; j < width; ++j) { coeff[j] = 1; coeff[width+j] = 1; }
                for (int j = 0; j < count; ++j) if (active[j]) {
                    const auto &s = solvers[j]->s_;
                    coeff[j] = float(s.step_size_primal); coeff[width+j] = float(s.step_size_dual);
                    coeff[2*width+j] = float(solvers[j]->params_.reflection_coefficient);
                    const double epoch = double(s.inner_count) + k;
                    coeff[3*width+j] = float(epoch / (epoch + 1));
                }
                return coeff;
            };
            for (int k = 1; k <= distance; ++k) {
                auto coeff = coefficients(k);
                const bool baseline = k == 1 && std::any_of(fresh.begin(), fresh.end(), [](int v) { return v != 0; });
                const bool major = baseline || k == distance;
                bool native = false;
#ifdef MLXPDLP_HAS_METAL_BATCHING
                if (!major && width <= 256 && n > 0 && m > 0 && options.iteration_batch_size > 1 &&
                    solvers[0]->params_.metal_iteration_batching) {
                    const int count = std::min({distance-k, max_metal_batch_iterations,
                        options.iteration_batch_size-(k-1)%options.iteration_batch_size});
                    Workspace prototype = w;
                    advance(prototype, active_array, mx::array(coeff.data(), {4,width}, mx::float32), false, options.lp_tile_width);
                    std::vector<std::vector<float>> block;
                    for (int offset=0; offset<count; ++offset) block.push_back(coefficients(k+offset));
                    auto next = metal_spmm_minor_batch({prototype.x,prototype.xr}, {prototype.y,prototype.yr},
                                                       std::move(block), submission_stream);
                    w.x=next[0];w.xr=next[1];w.y=next[2];w.yr=next[3];
                    k += count-1;
                    native = output.native_iteration_batching_active = true;
                }
#endif
                if (!native) advance(w, active_array, mx::array(coeff.data(), {4, width}, mx::float32), major, options.lp_tile_width);
                advanced = k;
                if (baseline) {
                    auto dx = w.xr - w.xp;
                    auto dy = w.yr - w.yp;
                    auto cross = product(dy, active_array, true, options.lp_tile_width);
                    auto metrics = mx::stack({mx::sqrt(mx::sum(mx::square(dx), 0)),
                                               mx::sqrt(mx::sum(mx::square(dy), 0)), mx::sum(cross * dx, 0)});
                    mx::eval(metrics, w.x, w.xr, w.xp, w.z, w.y, w.yr, w.yp);
                    const auto *values = metrics.data<float>();
                    for (int j = 0; j < count; ++j) if (active[j] && fresh[j]) {
                        solvers[j]->publish_fixed_point_metrics(values[j], values[width+j], values[2*width+j]);
                        solvers[j]->s_.initial_fixed_point_error = solvers[j]->s_.fixed_point_error;
                        fresh[j] = 0;
                    }
                } else if (major || k % options.iteration_batch_size == 0) {
                    mx::eval(w.x, w.xr, w.y, w.xp, w.z, w.yr, w.yp);
                }
                if (k % options.iteration_batch_size == 0) {
                    bool expired = seconds(entry) >= options.time_sec_limit;
                    for (int j = 0; j < count; ++j) if (active[j])
                        expired = expired || seconds(solvers[j]->s_.start_time) >= solvers[j]->params_.termination_criteria.time_sec_limit;
                    if (expired) break;
                }
            }
            output.pdhg_time_sec += seconds(iteration_start);
            const auto checkpoint = Clock::now();
            auto ax = product(w.xp, active_array, false, options.lp_tile_width);
            auto aty = product(w.yp, active_array, true, options.lp_tile_width);
            auto dx = w.xr - w.xp, dy = w.yr - w.yp;
            auto cross = product(dy, active_array, true, options.lp_tile_width);
            auto ray = [&](const mx::array &v) {
                auto norm = v.shape(0) == 0 ? mx::zeros({width}, mx::float32) : mx::max(mx::abs(v), 0);
                return v / mx::where(norm > 0, norm, mx::ones_like(norm));
            };
            auto primal_ray_product = product(ray(dx), active_array, false, options.lp_tile_width);
            auto dual_ray_product = product(ray(dy), active_array, true, options.lp_tile_width);
            auto fp = mx::stack({mx::sqrt(mx::sum(mx::square(dx), 0)), mx::sqrt(mx::sum(mx::square(dy), 0)),
                                 mx::sum(cross * dx, 0)});
            auto distances = mx::stack({mx::sqrt(mx::sum(mx::square(w.xp - w.xi), 0)),
                                        mx::sqrt(mx::sum(mx::square(w.yp - w.yi), 0))});
            std::vector<mx::array> metrics, ray_metrics, evaluated{fp, distances};
            for (int j = 0; j < count; ++j) {
                auto &s = solvers[j]->s_;
                if (active[j]) {
                    s.x_cur = lane(w.x, j); s.x_ref = lane(w.xr, j); s.x_pdhg = lane(w.xp, j);
                    s.dual_slack = lane(w.z, j); s.y_cur = lane(w.y, j); s.y_ref = lane(w.yr, j); s.y_pdhg = lane(w.yp, j);
                    s.Ax = lane(ax, j); s.ATy = lane(aty, j);
                    s.delta_x = s.x_ref - s.x_pdhg; s.delta_y = s.y_ref - s.y_pdhg;
                    evaluated.insert(evaluated.end(), {s.x_cur, s.x_ref, s.x_pdhg, s.dual_slack, s.y_cur, s.y_ref, s.y_pdhg});
                }
                metrics.push_back(solvers[j]->build_residual_metrics());
                auto ar = lane(primal_ray_product, j), atr = lane(dual_ray_product, j);
                ray_metrics.push_back(solvers[j]->build_infeasibility_metrics(&ar, &atr));
            }
            auto packed_metrics = mx::contiguous(mx::stack(metrics));
            auto packed_ray_metrics = mx::contiguous(mx::stack(ray_metrics));
            evaluated.push_back(packed_metrics);
            evaluated.push_back(packed_ray_metrics);
            mx::eval(evaluated);
            const auto *values = packed_metrics.data<float>();
            const auto *ray_values = packed_ray_metrics.data<float>();
            const auto *fp_values = fp.data<float>();
            const auto *distance_values = distances.data<float>();
            for (int j = 0; j < count; ++j) if (active[j]) {
                auto &solver = *solvers[j]; auto &s = solver.s_;
                const auto &params = solver.params_;
                s.total_count += advanced; s.inner_count += advanced;
                const bool regular = s.total_count % frequency == 0;
                const bool limit = s.total_count >= params.termination_criteria.iteration_limit;
                const bool expired = seconds(entry) >= options.time_sec_limit ||
                    seconds(s.start_time) >= params.termination_criteria.time_sec_limit;
                // A speculative check due for another member changes neither
                // this member's controller history nor its best certificate.
                if (advanced < member_due[j] && !limit && !expired) continue;
                if (advanced == distance && advanced >= member_due[j]) {
                    double member_metrics[6];
                    for (int a = 0; a < 6; ++a) member_metrics[a] = values[6*j+a];
                    solver.publish_residual_metrics(member_metrics);
                    solver.publish_fixed_point_metrics(fp_values[j], fp_values[width+j], fp_values[2*width+j]);
                    for (int a = 0; a < 6; ++a) member_metrics[a] = ray_values[6*j+a];
                    solver.publish_infeasibility_metrics(member_metrics);
                    solver.infeasibility_metrics_ready_ = true;
                }
                if (solver.invalid_fixed_point_metric_ || !std::isfinite(s.relative_primal_residual) ||
                    !std::isfinite(s.relative_dual_residual) || !std::isfinite(s.relative_objective_gap)) {
                    if (!solver.mlx_recover_numerical_failure()) s.termination_reason = TERMINATION_REASON_NUMERICAL_ERROR;
                    else fresh[j] = 1;
                } else {
                    solver.mlx_host_double_feedback();
                    solver.mlx_save_best_iterate();
                    solver.mlx_check_termination(regular || limit);
                    if (s.termination_reason == TERMINATION_REASON_PRIMAL_INFEASIBLE ||
                        s.termination_reason == TERMINATION_REASON_DUAL_INFEASIBLE) {
                        if (!audit_ray(solver, output.results[begin+j], s.termination_reason == TERMINATION_REASON_DUAL_INFEASIBLE))
                        {
                            ++output.results[begin+j].original_audit_failures;
                            s.termination_reason = TERMINATION_REASON_UNSPECIFIED;
                        }
                    } else if ((n == 0 || m == 0) && s.termination_reason != TERMINATION_REASON_OPTIMAL) {
                        if (!audit_ray(solver, output.results[begin+j], false)) audit_ray(solver, output.results[begin+j], true);
                    }
                    if (s.termination_reason == TERMINATION_REASON_OPTIMAL) {
                        const auto audit = Clock::now();
                        if (!audit_optimal(solver)) {
                            ++output.results[begin+j].original_audit_failures;
                            s.termination_reason = TERMINATION_REASON_UNSPECIFIED;
                            solver.mlx_host_double_feedback(true);
                        }
                        output.audit_time_sec += seconds(audit);
                    }
                    if (s.termination_reason == TERMINATION_REASON_UNSPECIFIED && regular) {
                        const auto &tc = params.termination_criteria;
                        conditional[j] = params.conditional_termination_evaluation && nnz <= (1 << 18) &&
                            s.relative_primal_residual <= 10 * tc.eps_feasible_relative &&
                            s.relative_dual_residual <= 10 * tc.eps_feasible_relative &&
                            s.relative_objective_gap <= 10 * tc.eps_optimal_relative;
                        if (solver.mlx_should_adaptive_restart()) {
                            solver.mlx_perform_restart(distance_values[j], distance_values[width+j]);
                            fresh[j] = 1;
                        }
                    }
                }
                if (s.termination_reason == TERMINATION_REASON_UNSPECIFIED) {
                    if (seconds(entry) >= options.time_sec_limit || seconds(s.start_time) >= params.termination_criteria.time_sec_limit)
                        s.termination_reason = TERMINATION_REASON_TIME_LIMIT;
                    else if (limit) s.termination_reason = TERMINATION_REASON_ITERATION_LIMIT;
                }
                if (s.termination_reason != TERMINATION_REASON_UNSPECIFIED) {
                    active[j] = 0;
                    fresh[j] = 0;
                    finish_member(solver, output, begin + j, entry);
                }
            }
            active_array = mx::array(active.data(), {width}, mx::int32);
            output.checkpoint_time_sec += seconds(checkpoint);
        }
        for (auto &solver : solvers) if (solver->s_.termination_reason == TERMINATION_REASON_UNSPECIFIED)
            solver->s_.termination_reason = TERMINATION_REASON_ITERATION_LIMIT;
    }

    void finish_member(MlxPdlpSolver &solver, BatchResult &output, size_t index, Clock::time_point entry) {
        auto &member = output.results[index];
        if (member.result) return;
        const auto audit = Clock::now();
        member.result.reset(solver.finish_solve());
        if (member.result->termination_reason == TERMINATION_REASON_UNSPECIFIED)
            member.result->termination_reason = solver.s_.termination_reason == TERMINATION_REASON_OPTIMAL ?
                TERMINATION_REASON_NUMERICAL_ERROR : solver.s_.termination_reason;
        member.execution_time_sec = seconds(entry) - member.queue_time_sec;
        member.step_size_reductions = solver.s_.step_size_reductions;
        output.audit_time_sec += seconds(audit);
    }

    BatchResult solve(const std::vector<BatchProblem> &input, const BatchOptions &options,
                       const pdhg_parameters_t *override_parameters) {
        const auto entry = Clock::now();
        auto params = override_parameters ? *override_parameters : parameters;
        validate_parameters(params);
        if (!same_preparation(params, parameters)) throw std::invalid_argument("parameters: matrix preparation differs from plan");
        if (std::isnan(options.time_sec_limit) || options.time_sec_limit < 0 ||
            (options.lp_tile_width != 4 && options.lp_tile_width != 8) ||
            options.iteration_batch_size < 1 || options.iteration_batch_size > 64)
            throw std::invalid_argument("batch options: invalid deadline, LP tile width, or iteration batch size");
        if (input.size() > INT_MAX - 8) throw std::length_error("batch width exceeds int32 capacity");
        for (size_t i = 0; i < input.size(); ++i) validate_member(input[i], n, m, i, params.presolve);
        BatchResult result;
        if (input.empty()) return result;
        bool shared = options.execution == BatchExecution::shared;
        if (shared && input.size() > 1 && (params.presolve || device.type != mx::Device::gpu || !params.metal_fused_kernels))
            throw std::invalid_argument("shared execution requires GPU, presolve=False, and metal_fused_kernels=True");
        if (input.size() == 1) { shared = false; result.fallback_reason = "B=1 uses the existing single-LP route"; }
        else if (options.execution == BatchExecution::automatic) {
            result.fallback_reason = params.presolve ? "presolve requires independent solves" :
                "automatic shared selection awaits hardware-scoped performance qualification; use shared explicitly";
        }
        result.execution = shared ? BatchExecution::shared : BatchExecution::independent;
        result.lp_tile_width = shared ? options.lp_tile_width : 0;
        result.iteration_batch_size = shared ? options.iteration_batch_size : 0;
        size_t input_bytes = 0;
        for (const auto &p : input) {
            for (const auto *v : {&p.objective, &p.variable_lower_bounds, &p.variable_upper_bounds,
                                 &p.constraint_lower_bounds, &p.constraint_upper_bounds}) input_bytes = checked_add(input_bytes, checked_mul(v->size(), sizeof(double)));
            for (const auto *v : {&p.primal_start, &p.dual_start, &p.reduced_cost_start})
                if (*v) input_bytes = checked_add(input_bytes, checked_mul((*v)->size(), sizeof(double)));
        }
        const size_t correction_matrix_bytes = (params.presolve || params.host_double_polishing) ? checked_mul(matrix_bytes, 4) : 0;
        const size_t base_bytes = checked_add(checked_add(matrix_bytes, correction_matrix_bytes), checked_add(input_bytes,
            checked_mul(input.size(), checked_add(sizeof(BatchMemberResult), checked_mul(size_t(3)*n + size_t(2)*m, sizeof(double))))));
        // Includes packed bounds, live vectors, snapshots, detached member lanes,
        // bounded K graphs, host audit/correction work, and allocator headroom.
        const size_t lane_bytes = checked_mul(size_t(n) + m + 1,
            size_t(768 + 48 * options.iteration_batch_size));
        size_t group_width = shared ? input.size() : 1;
        if (options.resident_memory_budget_bytes) {
            const size_t budget = options.resident_memory_budget_bytes;
            if (budget <= base_bytes || (budget - base_bytes) / lane_bytes < (shared ? size_t(options.lp_tile_width) : 1))
                throw std::length_error("resident_memory_budget_bytes: insufficient for plan, request, results, and one workspace");
            size_t available = (budget - base_bytes) / lane_bytes;
            if (shared) available -= available % options.lp_tile_width;
            group_width = std::min(group_width, available);
        }
        const size_t padded_width = shared ? ((group_width + options.lp_tile_width - 1) / options.lp_tile_width) * options.lp_tile_width : 1;
        result.estimated_peak_resident_bytes = checked_add(base_bytes, checked_mul(padded_width, lane_bytes));
        const std::vector<BatchProblem> problems = input;
        result.results.resize(problems.size());
        for (size_t i = 0; i < problems.size(); ++i) result.results[i].input_index = i;
        std::unique_lock<std::timed_mutex> lock(mutex, std::defer_lock);
        bool acquired;
        if (std::isfinite(options.time_sec_limit)) {
            const double horizon = std::chrono::duration<double>(Clock::time_point::max() - entry).count();
            const auto deadline = options.time_sec_limit >= horizon ? Clock::time_point::max() :
                entry + std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double>(options.time_sec_limit));
            acquired = lock.try_lock_until(deadline);
        }
        else { lock.lock(); acquired = true; }
        if (acquired) {
            submission_stream = mx::default_stream(device);
            mx::StreamContext context(submission_stream);
            for (size_t begin = 0; begin < problems.size() && seconds(entry) < options.time_sec_limit; begin += group_width) {
                const size_t end = std::min(problems.size(), begin + group_width);
                ++result.groups;
                std::vector<std::unique_ptr<MlxPdlpSolver>> solvers;
                for (size_t i = begin; i < end && seconds(entry) < options.time_sec_limit; ++i) {
                    const auto admission = Clock::now();
                    auto member_params = params;
                    member_params.termination_criteria.time_sec_limit = std::min(params.termination_criteria.time_sec_limit,
                        std::max(0.0, options.time_sec_limit - seconds(entry)));
                    auto solver = make_solver(problems[i], member_params, !params.presolve && problems.size() > 1);
                    result.construction_time_sec += seconds(admission);
                    auto &member = result.results[i];
                    member.queue_time_sec = std::chrono::duration<double>(admission - entry).count();
                    member.has_solution = true;
                    if (shared) {
                        solver->s_.start_time = admission;
                        solver->solve_called_ = true;
                        const auto initializing = Clock::now();
                        solver->initialize_solve();
                        result.initialization_time_sec += seconds(initializing);
                        solvers.push_back(std::move(solver));
                    } else {
                        solver->params_.termination_criteria.time_sec_limit = std::max(0.0,
                            member_params.termination_criteria.time_sec_limit - seconds(admission));
                        member.result.reset(solver->solve());
                        member.step_size_reductions = solver->s_.step_size_reductions;
                        member.execution_time_sec = seconds(admission);
                    }
                }
                if (shared && !solvers.empty()) {
                    result.max_active_width = std::max(result.max_active_width, solvers.size());
                    run_shared(solvers, result, options, entry, begin);
                    for (size_t j = 0; j < solvers.size(); ++j) {
                        finish_member(*solvers[j], result, begin+j, entry);
                    }
                }
            }
        }
        for (auto &member : result.results) if (!member.has_solution) {
            member.result = unstarted(n, m, nnz);
            member.queue_time_sec = seconds(entry);
        }
        result.wall_time_sec = seconds(entry);
        result.deadline_overrun_sec = std::max(0.0, result.wall_time_sec - options.time_sec_limit);
        return result;
    }
};
} // namespace detail

SharedMatrixPlan::SharedMatrixPlan(int n, int m, const int *rp, const int *ci, const double *v,
                                   const pdhg_parameters_t *p, mx::Device d)
    : impl_(std::make_unique<detail::BatchDriver>(n, m, rp, ci, v, p, d)) {}
SharedMatrixPlan::~SharedMatrixPlan() = default;
SharedMatrixPlan::SharedMatrixPlan(SharedMatrixPlan &&) noexcept = default;
SharedMatrixPlan &SharedMatrixPlan::operator=(SharedMatrixPlan &&) noexcept = default;
BatchResult SharedMatrixPlan::solve_batch(const std::vector<BatchProblem> &p, const BatchOptions &o,
                                         const pdhg_parameters_t *settings) { return impl_->solve(p, o, settings); }
int SharedMatrixPlan::num_variables() const { return impl_->n; }
int SharedMatrixPlan::num_constraints() const { return impl_->m; }
int SharedMatrixPlan::num_nonzeros() const { return impl_->nnz; }
size_t SharedMatrixPlan::resident_bytes() const { return impl_->matrix_bytes; }
double SharedMatrixPlan::preparation_time_sec() const { return impl_->preparation_sec; }
double SharedMatrixPlan::operator_norm_upper_bound() const { return impl_->prepared->state().operator_norm_upper_bound; }
} // namespace mlxpdlp
