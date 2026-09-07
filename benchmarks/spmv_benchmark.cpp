// Isolate the production Metal CSR kernels from PDHG and preprocessing.
#include "metal_spmv.h"
#include "benchmark_provenance.h"
#include "mlxPDLP/mps_loader.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

using namespace mlxpdlp;
using Clock = std::chrono::steady_clock;

namespace {
struct Matrix {
    int columns;
    std::vector<int32_t> row_ptr;
    std::vector<int32_t> col_ind;
    std::vector<float> values;
};

Matrix transpose(const Matrix &a) {
    Matrix out{static_cast<int>(a.row_ptr.size()) - 1,
               std::vector<int32_t>(a.columns + 1, 0),
               std::vector<int32_t>(a.values.size()),
               std::vector<float>(a.values.size())};
    for (int column : a.col_ind) ++out.row_ptr[column + 1];
    for (int row = 0; row < a.columns; ++row) out.row_ptr[row + 1] += out.row_ptr[row];
    auto next = out.row_ptr;
    for (int row = 0; row < out.columns; ++row) {
        for (int k = a.row_ptr[row]; k < a.row_ptr[row + 1]; ++k) {
            int slot = next[a.col_ind[k]]++;
            out.col_ind[slot] = row;
            out.values[slot] = a.values[k];
        }
    }
    return out;
}

int positive_integer(const char *text) {
    size_t end = 0;
    int value = std::stoi(text, &end);
    if (text[end] != '\0' || value < 1) throw std::invalid_argument("expected a positive integer");
    return value;
}

void benchmark(const Matrix &a, const char *orientation, int requested_repetitions,
               int trials, const std::string &selection, mx::Stream stream) {
    const int rows = static_cast<int>(a.row_ptr.size()) - 1;
    const int nonzeros = static_cast<int>(a.values.size());
    const int repetitions = static_cast<int>(std::min<int64_t>(requested_repetitions,
        std::max<int64_t>(1, (64LL * 1024 * 1024) /
                                (static_cast<int64_t>(std::max(rows, 1)) * sizeof(float)))));
    int maximum_length = 0;
    std::vector<float> input(a.columns);
    for (int column = 0; column < a.columns; ++column)
        input[column] = static_cast<float>((column % 257) * 17 % 257 - 128) / 129.7f;
    std::vector<double> expected(rows, 0.0), absolute_sum(rows, 0.0);
    for (int row = 0; row < rows; ++row) {
        maximum_length = std::max(maximum_length, a.row_ptr[row + 1] - a.row_ptr[row]);
        for (int k = a.row_ptr[row]; k < a.row_ptr[row + 1]; ++k) {
            double product = static_cast<double>(a.values[k]) * input[a.col_ind[k]];
            expected[row] += product;
            absolute_sum[row] += std::fabs(product);
        }
    }
    std::printf("# %s rows=%d columns=%d nonzeros=%d average_row=%.3f maximum_row=%d\n",
                orientation, rows, a.columns, nonzeros, double(nonzeros) / rows, maximum_length);
    auto work = detail::make_adaptive_metal_work(a.row_ptr, rows);
    mx::array rp(a.row_ptr.data(), {rows + 1});
    mx::array wide_columns(a.col_ind.data(), {nonzeros});
    auto compact_columns = detail::metal_column_indices(a.col_ind, a.columns);
    mx::array values(a.values.data(), {nonzeros});
    mx::array offsets(work.offsets.data(), {static_cast<int>(work.offsets.size())});
    mx::array descriptors(work.rows.data(), {static_cast<int>(work.rows.size())});
    mx::array x(input.data(), {a.columns});
    mx::eval(rp, wide_columns, compact_columns, values, offsets, descriptors, x);

    struct Strategy { const char *name; SparseMetalSpmvStrategy value; };
    const std::vector<Strategy> strategies{
        {"scalar", SparseMetalSpmvStrategy::scalar_rows},
        {"quad", SparseMetalSpmvStrategy::quad_rows},
        {"simd", SparseMetalSpmvStrategy::simdgroup_rows},
        {"adaptive", SparseMetalSpmvStrategy::adaptive}};
    for (const auto &strategy : strategies) {
        if (selection != "all" && selection != strategy.name) continue;
        for (bool compact : {false, true}) {
            if (compact && compact_columns.dtype() == mx::int32) continue;
            const auto &columns = compact ? compact_columns : wide_columns;
            auto run = [&]() {
                return detail::metal_spmv(rp, columns, values, offsets, descriptors, x,
                                          rows, work.item_count, strategy.value, stream);
            };
            auto result = run();
            mx::eval(result);
            const float *actual = result.data<float>();
            double max_error = 0.0;
            for (int row = 0; row < rows; ++row) {
                if (!std::isfinite(actual[row])) throw std::runtime_error("nonfinite SpMV result");
                max_error = std::max(max_error,
                    std::fabs(actual[row] - expected[row]) / (1.0 + absolute_sum[row]));
            }
            if (max_error > 5e-5) throw std::runtime_error("SpMV failed the independent FP64 check");

            // Retain every output so lazy evaluation cannot discard work.
            // Warm three batches, then report whole enqueue/evaluate/sync time.
            std::vector<double> samples;
            for (int trial = -3; trial < trials; ++trial) {
                std::vector<mx::array> outputs;
                outputs.reserve(repetitions);
                mx::synchronize(stream);
                auto start = Clock::now();
                for (int rep = 0; rep < repetitions; ++rep) outputs.push_back(run());
                mx::eval(outputs);
                mx::synchronize(stream);
                double us = std::chrono::duration<double, std::micro>(Clock::now() - start).count() /
                            repetitions;
                if (trial >= 0) samples.push_back(us);
            }
            std::sort(samples.begin(), samples.end());
            const size_t middle = samples.size() / 2;
            const double median = samples.size() % 2 ? samples[middle]
                                                     : (samples[middle - 1] + samples[middle]) / 2;
            std::printf("%s,%d,%s,%d,%.3f,%.3f,%.3f,%.9g\n", orientation,
                        compact ? 16 : 32, strategy.name, repetitions, median,
                        samples.front(), samples.back(), max_error);
            std::fflush(stdout);
        }
    }
}
} // namespace

int main(int argc, char **argv) {
    if (argc < 2 || argc > 5 || std::string(argv[1]) == "--help") {
        std::fprintf(stderr, "Usage: %s MPS_PATH [REPETITIONS=64] [TRIALS=7] "
                             "[all|scalar|quad|simd|adaptive]\n", argv[0]);
        return argc == 2 ? 0 : 2;
    }
    try {
        int repetitions = argc > 2 ? positive_integer(argv[2]) : 64;
        int trials = argc > 3 ? positive_integer(argv[3]) : 7;
        std::string selection = argc > 4 ? argv[4] : "all";
        if (selection != "all" && selection != "scalar" && selection != "quad" &&
            selection != "simd" && selection != "adaptive")
            throw std::invalid_argument("unknown SpMV strategy");
        if (!mx::is_available(mx::Device::gpu)) {
            std::fprintf(stderr, "Metal is unavailable\n");
            return 77;
        }
        std::unique_ptr<mlxpdlp_mps_problem_t, decltype(&mlxpdlp_mps_problem_free)> problem(
            mlxpdlp_mps_problem_load(argv[1]), mlxpdlp_mps_problem_free);
        if (!problem || problem->num_variables < 1 || problem->num_constraints < 1)
            throw std::runtime_error("benchmark requires an MPS matrix with positive dimensions");
        const auto &p = *problem;
        Matrix a{p.num_variables,
                 std::vector<int32_t>(p.row_ptr, p.row_ptr + p.num_constraints + 1),
                 std::vector<int32_t>(p.col_ind, p.col_ind + p.num_nonzeros),
                 std::vector<float>(p.values, p.values + p.num_nonzeros)};
        auto stream = mx::default_stream(mx::Device::gpu);
        mx::StreamContext context(stream);
        std::printf("# Production kernels; FP32 values; FP64 reference; unscaled MPS matrix\n");
        const auto &device_info = mx::gpu::device_info(0);
        std::printf("# device=%s\n", std::get<std::string>(device_info.at("device_name")).c_str());
        std::printf("# revision=%s dirty=%s source_sha256=%s mlx_revision=%s\n",
                    MLXPDLP_BENCHMARK_GIT_REVISION, MLXPDLP_BENCHMARK_GIT_DIRTY,
                    MLXPDLP_BENCHMARK_SOURCE_SHA256, MLXPDLP_BENCHMARK_MLX_REVISION);
        std::printf("orientation,index_bits,strategy,repetitions,median_us,min_us,max_us,max_scaled_error\n");
        benchmark(a, "A", repetitions, trials, selection, stream);
        benchmark(transpose(a), "AT", repetitions, trials, selection, stream);
    } catch (const std::exception &error) {
        std::fprintf(stderr, "%s\n", error.what());
        return 1;
    }
}
