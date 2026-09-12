#include "mlxPDLP/solver.h"
#include "mlxPDLP/batch_solver.h"
#include "pdhg_control.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
using namespace mlxpdlp;

static void require(bool condition, const char *message) {
    if (!condition)
        throw std::runtime_error(message);
}

// Recovery can wait for non-finite arithmetic; rounding and restart decisions
// shift that checkpoint. Leave room for convergence afterwards, while the
// separate 200-iteration case checks that recovery respects a short budget.
static constexpr int spectral_convergence_limit = 5000;

static void spectral_regression(mx::Device device, bool conservative,
                                int iteration_limit = spectral_convergence_limit,
                                size_t direction_rank = 0) {
    constexpr int size = 16;
    constexpr double small_eigenvalue = 0.99;
    const double infinity = std::numeric_limits<double>::infinity();

    // Match the production power method's start, using the same C++ library.
    std::mt19937 generator(1);
    std::normal_distribution<double> normal(0.0, 1.0);
    std::vector<double> start(size);
    for (double &value : start) {
        value = normal(generator);
    }

    // Rank signed uniform unit vectors almost orthogonal to that start.
    // A short exhaustive search makes this portable across normal_distribution
    // implementations, whose outputs are not specified by the C++ standard.
    std::vector<std::pair<double, unsigned>> directions;
    directions.reserve(1u << (size - 1));
    for (unsigned mask = 0; mask < (1u << (size - 1)); ++mask) {
        double dot = start[0];
        for (int j = 1; j < size; ++j) {
            dot += (((mask >> (j - 1)) & 1u) ? 1.0 : -1.0) * start[j];
        }
        directions.emplace_back(std::abs(dot), mask);
    }
    std::sort(directions.begin(), directions.end());
    const unsigned best_mask = directions.at(direction_rank).second;
    std::vector<double> exact(size, 1.0 / std::sqrt(size));
    for (int j = 1; j < size; ++j) {
        exact[j] *= ((best_mask >> (j - 1)) & 1u) ? 1.0 : -1.0;
    }

    // A = .99 I + .01 vv^T has eigenvalues 1 and .99. Its absolute row and
    // column sums are all one. Every geometric/Ruiz scaling pass is uniform;
    // Pock-Chambolle returns the matrix to this scale (up to rounding).
    // min 0 subject to Ax=v, x free, has the unique solution x=v, y=0.
    std::vector<int> row_ptr(size + 1), col_ind;
    std::vector<double> matrix;
    for (int i = 0; i < size; ++i) {
        row_ptr[i] = static_cast<int>(matrix.size());
        for (int j = 0; j < size; ++j) {
            col_ind.push_back(j);
            matrix.push_back((i == j ? small_eigenvalue : 0.0) +
                             (1.0 - small_eigenvalue) * exact[i] * exact[j]);
        }
    }
    row_ptr[size] = static_cast<int>(matrix.size());
    std::vector<double> lower(size, -infinity), upper(size, infinity), objective(size, 0.0);

    pdhg_parameters_t params;
    mlxpdlp_set_default_parameters(&params);
    params.verbose = false;
    params.presolve = false;
    params.conservative_step_size = conservative;
    params.termination_criteria.iteration_limit = iteration_limit;
    params.termination_criteria.time_sec_limit = 10.0;
    MlxPdlpSolver solver(size, size, row_ptr.data(), col_ind.data(), matrix.data(), lower.data(),
                         upper.data(), exact.data(), exact.data(), objective.data(), 0.0, &params,
                         device);
    auto *result = solver.solve();
    const auto &state = solver.state();
    bool finite = true;
    double error = 0.0;
    for (int j = 0; j < size; ++j) {
        finite = finite && std::isfinite(result->primal_solution[j]) &&
                 std::isfinite(result->dual_solution[j]) && std::isfinite(result->reduced_cost[j]);
        error = std::max(error, std::fabs(result->primal_solution[j] - exact[j]));
    }
    const auto reason = result->termination_reason;
    std::cout << (device.type == mx::Device::cpu ? "CPU" : "Metal")
              << " conservative=" << conservative << " direction_rank=" << direction_rank
              << " iterations=" << result->total_count
              << " recoveries=" << state.step_size_reductions << " error=" << error
              << " reason=" << static_cast<int>(reason)
              << " primal=" << result->relative_primal_residual
              << " dual=" << result->relative_dual_residual
              << " gap=" << result->relative_objective_gap << '\n';
    const int count = result->total_count;
    mlxpdlp_result_free(result);
    require(finite, "spectral underestimation returned non-finite certificate");
    require(count <= iteration_limit, "recovery exceeded the iteration budget");
    if (iteration_limit > 200) {
        require(reason == TERMINATION_REASON_OPTIMAL, "spectral adversary did not converge");
        require(error < 1e-3, "spectral adversary has incorrect solution");
    } else {
        return; // A short budget must retain a finite cold-start checkpoint.
    }
    require(state.operator_norm_upper_bound > 0.0 &&
                state.step_size * state.operator_norm_upper_bound < 1.0,
            "recovered/conservative step violates certified bound");
    if (conservative) {
        require(state.step_size_reductions == 0, "conservative mode needed recovery");
        require(state.singular_value_iterations == 0, "conservative mode ran power iteration");
    } else {
        require(state.step_size_reductions > 0, "adversary did not exercise numerical recovery");
    }
    if (device.type == mx::Device::gpu && !conservative) {
        SharedMatrixPlan plan(size,size,row_ptr.data(),col_ind.data(),matrix.data(),&params,device);
        std::vector<BatchProblem> members(4);
        for (auto &member : members) {
            member.objective=objective;
            member.constraint_lower_bounds=member.constraint_upper_bounds=std::vector<double>(size,0);
        }
        members[1].constraint_lower_bounds=members[1].constraint_upper_bounds=exact;
        BatchOptions options;options.execution=BatchExecution::shared;
        auto batch=plan.solve_batch(members,options);
        require(batch.results[1].step_size_reductions>0,"shared adversary did not exercise recovery");
        for (int member=0;member<4;++member) {
            const auto &r=batch.results[member];
            require(r.result->termination_reason==TERMINATION_REASON_OPTIMAL,"shared recovery failed to converge");
            require(r.result->total_count<=iteration_limit,"shared recovery exceeded the member limit");
            if (member!=1) require(r.step_size_reductions==0,"one member's recovery changed its neighbor");
            for (int j=0;j<size;++j)
                require(std::abs(r.result->primal_solution[j]-(member==1?exact[j]:0))<1e-3,
                        "shared recovery returned an incorrect original-model solution");
        }
    }
}

static void controller_regressions() {
    // The source HPR primal step sigma must increase by 100 here. Since
    // sigma=eta/w, w must decrease by 100 at the identical fixed-point target.
    const double corrected =
        detail::hpr_weight(1.0 / 0.998, 1e-12, 1e-12, 1.0, 1e-12, 1e-10, 1e-12);
    require(std::fabs(corrected - 0.01) < 1e-14, "HPR tail sigma correction is inverted");
    const double opposite = detail::hpr_weight(1.0 / 0.998, 1e-12, 1e-12, 1.0, 1e-10, 1e-12, 1e-12);
    require(std::fabs(opposite - 100.0) < 1e-12, "HPR tail correction failed opposite imbalance");
    require(detail::hpr_necessary_restart(5.0, 10.0, 4.0), "HPR necessary restart omitted");
    require(!detail::hpr_necessary_restart(3.0, 10.0, 4.0), "HPR restarted on decreasing movement");
    require(!detail::hpr_necessary_restart(7.0, 10.0, 4.0), "HPR ignored required reduction");
    require(std::isfinite(detail::hpr_weight(1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0)),
            "zero fixed-point error generated a non-finite HPR weight");
    double integral = 0.0, previous = 0.0;
    const double weight =
        detail::pid_weight(1.0, 10.0, 1000.0, 1.0, 0.0, 0.3, integral, previous, 2.0);
    require(std::isfinite(weight) && weight > 0.0 && weight <= 1e12,
            "finite PID gains overflowed the weight");
    const double fallback =
        detail::pid_weight(weight, 10.0, 1e308, 0.0, 0.0, 0.3, integral, previous, 2.0);
    require(fallback == 2.0 && integral == 0.0 && previous == 0.0,
            "non-finite PID update did not reset controller memory");
    require(detail::invalid_fixed_point_metric(1.0, -1.01, 1e-7), "negative metric was hidden");
    require(!detail::invalid_fixed_point_metric(1.0, -1.0 - 1e-8, 1e-7),
            "rounding-scale cancellation triggered recovery");
    require(detail::invalid_fixed_point_metric(1e308, 1e308, 1e-7),
            "overflow of the metric sum escaped recovery");
    const double optimal = detail::certificate_merit(1e-5, 1e-5, 1e-9, 1e-4, 1e-8);
    const double missed_gap = detail::certificate_merit(1e-7, 1e-7, 1e-7, 1e-4, 1e-8);
    require(optimal < 1.0 && missed_gap > 1.0 && optimal < missed_gap,
            "certificate selection discarded a converged point with unequal tolerances");
}

static void parameter_validation_regression() {
    int rows[] = {0, 1}, columns[] = {0};
    double values[] = {1.0}, lower[] = {0.0}, upper[] = {1.0};
    for (int invalid = 0; invalid < 6; ++invalid) {
        pdhg_parameters_t params;
        mlxpdlp_set_default_parameters(&params);
        params.verbose = params.presolve = false;
        switch (invalid) {
        case 0:
            params.restart_params.k_p = std::numeric_limits<double>::quiet_NaN();
            break;
        case 1:
            params.restart_params.i_smooth = 1.1;
            break;
        case 2:
            params.reflection_coefficient = 2.0;
            break;
        case 3:
            params.pock_chambolle_alpha = 2.1;
            break;
        case 4:
            params.termination_evaluation_frequency = 0;
            break;
        case 5:
            params.restart_policy = 2;
            break;
        }
        bool rejected = false;
        try {
            MlxPdlpSolver solver(1, 1, rows, columns, values, lower, upper, lower, upper, values,
                                 0.0, &params, mx::Device::cpu);
        } catch (const std::invalid_argument &) {
            rejected = true;
        }
        require(rejected, "invalid controller parameters were accepted");
    }
}

static void fp64_feedback_regression() {
    // These distinct original coefficients round to the same Metal matrix.
    // At a tighter tolerance, a zero FP32 residual cannot certify the LP.
    int row_ptr[] = {0, 1}, columns[] = {0};
    double values[] = {1.0 + 2e-8}, bounds[] = {1.0}, objective[] = {0.0};
    double lower[] = {-INFINITY}, upper[] = {INFINITY};
    pdhg_parameters_t params;
    mlxpdlp_set_default_parameters(&params);
    params.verbose = params.presolve = false;
    params.geometric_mean_iterations = params.l_inf_ruiz_iterations = 0;
    params.has_pock_chambolle_alpha = params.bound_objective_rescaling = false;
    params.termination_criteria.eps_feasible_relative = 1e-10;
    params.termination_criteria.eps_optimal_relative = 1e-10;
    params.termination_criteria.iteration_limit = 1200;
    int counts[2]{};
    for (int enabled = 0; enabled < 2; ++enabled) {
        params.host_double_residual_evaluation = enabled != 0;
        MlxPdlpSolver solver(1, 1, row_ptr, columns, values, lower, upper, bounds, bounds,
                             objective, 0.0, &params, mx::Device::gpu);
        auto *result = solver.solve();
        counts[enabled] = result->total_count;
        const bool audited = solver.state().host_double_audit_count > 0;
        const auto reason = result->termination_reason;
        const double independent_residual = std::fabs(values[0] * result->primal_solution[0] - 1.0);
        mlxpdlp_result_free(result);
        require(audited == (enabled != 0), "FP64 feedback switch was ignored");
        require(reason != TERMINATION_REASON_OPTIMAL || independent_residual < 3e-10,
                "FP32 rounding generated false optimality");
    }
    require(counts[1] == params.termination_criteria.iteration_limit && counts[0] < counts[1],
            "FP64 disagreement did not prevent premature device stopping");
}

static void accelerated_continuation_regression() {
    // 1,024 independent two-variable LPs with 1,048,576 raw entries. Exact
    // binary-fraction duplicates exercise native construction/coalescing and
    // both product orientations without requiring a large solution vector.
    constexpr int m = 1024, n = 2 * m, repeats = 512;
    std::vector<int> rows(m + 1), columns;
    std::vector<double> values, objective(n), lower(n, 0.0), upper(n, 1.0), bounds(m, 1.0),
        primal(n), dual(m, 0.0);
    for (int row = 0; row < m; ++row) {
        rows[row] = static_cast<int>(values.size());
        for (int repeat = 0; repeat < repeats; ++repeat) {
            columns.push_back(2 * row + 1);
            columns.push_back(2 * row);
            values.push_back(1.0 / repeats);
            values.push_back(1.0 / repeats);
        }
        objective[2 * row + 1] = 1.0;
        primal[2 * row] = 0.49;
        primal[2 * row + 1] = 0.5;
    }
    rows[m] = static_cast<int>(values.size());
    pdhg_parameters_t params;
    mlxpdlp_set_default_parameters(&params);
    params.verbose = params.presolve = false;
    params.geometric_mean_iterations = params.l_inf_ruiz_iterations = params.sv_max_iter = 0;
    params.has_pock_chambolle_alpha = params.bound_objective_rescaling = false;
    params.host_double_polishing = true;
    params.host_double_polishing_iteration_limit = 5000;
    params.host_double_polishing_time_sec_limit = 10.0;
    params.termination_criteria.iteration_limit = 0;
    params.termination_criteria.eps_feasible_relative = 1e-6;
    params.termination_criteria.eps_optimal_relative = 1e-6;
    MlxPdlpSolver solver(n, m, rows.data(), columns.data(), values.data(), lower.data(),
                         upper.data(), bounds.data(), bounds.data(), objective.data(), 0.0, &params,
                         primal.data(), dual.data(), mx::Device::cpu);
    auto *result = solver.solve();
    double error = 0.0;
    for (int row = 0; row < m; ++row) {
        error = std::max(error, std::fabs(result->primal_solution[2 * row] - 1.0));
        error = std::max(error, std::fabs(result->primal_solution[2 * row + 1]));
    }
    const bool passed = result->termination_reason == TERMINATION_REASON_OPTIMAL &&
                        result->host_double_polishing_iteration > 0 &&
                        result->host_double_polishing_iteration <= 5000 && error < 1e-4;
    mlxpdlp_result_free(result);
    require(passed, "native FP64 continuation failed the independent analytic optimum");
}

int main(int argc, char **argv) {
    try {
        controller_regressions();
        parameter_validation_regression();
        const mx::Device device =
            argc > 1 && std::string(argv[1]) == "metal" ? mx::Device::gpu : mx::Device::cpu;
        if (device.type == mx::Device::gpu && !mx::metal::is_available())
            return 77;
        spectral_regression(device, false);
        // This equivalent direction reproduced CI's delayed recovery locally:
        // one recovery, a small primal error, but an unconverged dual certificate.
        if (device.type == mx::Device::cpu)
            spectral_regression(device, false, spectral_convergence_limit, 7);
        spectral_regression(device, true);
        spectral_regression(device, false, 200);
        if (device.type == mx::Device::gpu)
            fp64_feedback_regression();
        else
            accelerated_continuation_regression();
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
    return 0;
}
