// Compare the complete solver trajectory with the original fused lazy graph.
// Synthetic fixtures keep this regression independent of downloaded datasets.
#include "mlxPDLP/solver.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

using namespace mlxpdlp;

namespace {

void require(bool condition, const char *message) {
    if (!condition)
        throw std::runtime_error(message);
}

struct Problem {
    int m, n;
    std::vector<int> rows, columns;
    std::vector<double> values, lower, upper, rhs, objective;
    bool expect_sparse = true;
};

Problem problem(int m, int n, int degree, bool long_row = false) {
    Problem p{m,
              n,
              {0},
              {},
              {},
              std::vector<double>(n, 0.0),
              std::vector<double>(n, 1.0),
              std::vector<double>(m, 0.0),
              std::vector<double>(n)};
    for (int row = 0; row < m; ++row) {
        const int length = row == m - 1 ? 0 : long_row && row == 0 ? 4097 : degree;
        for (int k = 0; k < length; ++k) {
            p.columns.push_back((row * 17 + k * 29) % n);
            const double value = (1 + (k * 3 + row) % 7) * 0.125;
            p.values.push_back(value);
            p.rhs[row] += 0.25 * value; // A known feasible point is x = 1/4.
        }
        p.rows.push_back(static_cast<int>(p.values.size()));
    }
    for (int col = 0; col < n; ++col)
        p.objective[col] = -0.01 * (col % 11);
    return p;
}

struct Snapshot {
    std::vector<float> state;
    std::vector<double> solution;
    double primal_step, dual_step;
    int inner_count, total_count;
    termination_reason_t reason;
    SparseMetalSpmvStrategy a_strategy, at_strategy;
};

Snapshot solve(const Problem &p, int limit, int frequency, bool batch) {
    pdhg_parameters_t params;
    mlxpdlp_set_default_parameters(&params);
    require(params.metal_iteration_batching, "Metal batching should default on");
    require(!params.host_double_residual_evaluation, "periodic FP64 feedback default changed");
    params.verbose = params.presolve = params.feasibility_polishing = false;
    params.host_double_polishing = params.conditional_termination_evaluation = false;
    params.conservative_step_size = true;
    params.metal_iteration_batching = batch;
    params.restart_params.artificial_restart_threshold = 0.1;
    params.termination_evaluation_frequency = frequency;
    params.termination_criteria.iteration_limit = limit;
    params.termination_criteria.time_sec_limit = 60.0;
    params.termination_criteria.eps_feasible_relative = 0.0;
    params.termination_criteria.eps_optimal_relative = 0.0;
    MlxPdlpSolver solver(p.n, p.m, p.rows.data(), p.columns.data(), p.values.data(), p.lower.data(),
                         p.upper.data(), p.rhs.data(), p.rhs.data(), p.objective.data(), 0.0,
                         &params, mx::Device::gpu);
    std::unique_ptr<mlxpdlp_result_t, decltype(&mlxpdlp_result_free)> result(solver.solve(),
                                                                             mlxpdlp_result_free);
    const auto &state = solver.state();
    require(state.sparse_metal_active == p.expect_sparse, "fixture selected the wrong backend");
#ifdef MLXPDLP_EXPECT_METAL_BATCHING
    require(state.metal_iteration_batching_active == (batch && p.expect_sparse),
            "native batch path did not honor the runtime switch");
#else
    require(!state.metal_iteration_batching_active,
            "unsupported build selected native Metal batching");
#endif
    require(result->total_count == limit, "Metal batching changed the iteration cap");
    require(result->termination_reason == TERMINATION_REASON_ITERATION_LIMIT,
            "fixed-work fixture stopped for an unexpected reason");
    Snapshot snapshot{{},
                      {},
                      state.step_size_primal,
                      state.step_size_dual,
                      state.inner_count,
                      result->total_count,
                      result->termination_reason,
                      state.sparse_a_spmv_strategy,
                      state.sparse_at_spmv_strategy};
    // Include restart anchors and major snapshots: buffer reuse must preserve
    // these even when their values coincide with an initial current state.
    std::vector<mx::array> arrays{state.x_cur,      state.x_ref,  state.y_cur,
                                  state.x_pdhg,     state.y_ref,  state.y_pdhg,
                                  state.dual_slack, state.x_init, state.y_init};
    mx::eval(arrays);
    mx::synchronize(state.stream);
    for (const auto &array : arrays) {
        require(array.dtype() == mx::float32 && array.flags().row_contiguous,
                "unexpected Metal state layout");
        const auto *values = array.data<float>();
        for (size_t i = 0; i < array.size(); ++i) {
            require(std::isfinite(values[i]), "non-finite live Metal state");
        }
        snapshot.state.insert(snapshot.state.end(), values, values + array.size());
    }
    for (auto [values, size] :
         {std::pair{result->primal_solution, p.n}, std::pair{result->dual_solution, p.m},
          std::pair{result->reduced_cost, p.n}}) {
        for (int i = 0; i < size; ++i)
            require(std::isfinite(values[i]), "non-finite solution");
        snapshot.solution.insert(snapshot.solution.end(), values, values + size);
    }
    return snapshot;
}

void compare(const Problem &p, int limit, int frequency) {
    const auto reference = solve(p, limit, frequency, false);
    const auto actual = solve(p, limit, frequency, true);
    std::printf("m=%d n=%d cap=%d checkpoint=%d A=%d AT=%d\n", p.m, p.n, limit, frequency,
                static_cast<int>(actual.a_strategy), static_cast<int>(actual.at_strategy));
    require(actual.state.size() == reference.state.size() &&
                std::memcmp(actual.state.data(), reference.state.data(),
                            actual.state.size() * sizeof(float)) == 0,
            "native batch changed live states, snapshots, or restart anchors");
    require(actual.solution.size() == reference.solution.size() &&
                std::memcmp(actual.solution.data(), reference.solution.data(),
                            actual.solution.size() * sizeof(double)) == 0,
            "native batch changed the returned certificate");
    require(actual.primal_step == reference.primal_step &&
                actual.dual_step == reference.dual_step &&
                actual.inner_count == reference.inner_count &&
                actual.total_count == reference.total_count && actual.reason == reference.reason,
            "native batch changed restart control or stopping behavior");
    if (limit > frequency) {
        require(actual.inner_count < actual.total_count,
                "fixture failed to exercise a restart boundary");
    }
}

} // namespace

int main() {
    try {
        if (!mx::is_available(mx::Device::gpu))
            return 77;
        const auto small = problem(65, 67, 3);
        for (int cap : {1, 2, 3, 15, 16, 17, 18, 31, 32, 33, 99, 100, 101, 201}) {
            compare(small, cap, 100);
        }
        for (int cap : {16, 17, 18, 35})
            compare(small, cap, 17);
        for (const auto &p : {problem(3, 2053, 3), problem(113, 127, 24), problem(521, 541, 100),
                              problem(131, 8209, 5, true), problem(9, 65537, 3)}) {
            for (int cap : {17, 100, 201})
                compare(p, cap, 100);
        }
        auto dense = problem(3, 5, 3);
        dense.expect_sparse = false;
        compare(dense, 201, 100);
        std::puts("Metal iteration batches preserve the fused solver trajectory");
    } catch (const std::exception &error) {
        std::fprintf(stderr, "%s\n", error.what());
        return 1;
    }
}
