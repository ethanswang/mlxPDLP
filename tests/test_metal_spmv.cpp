// Exercise the production CSR reductions directly against an independent FP64
// sum, including cancellation, duplicates, empty rows and partial work packs.
#include "metal_spmv.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>
#include <stdexcept>
#include <vector>

using namespace mlxpdlp;

namespace {
struct Matrix {
    int columns;
    std::vector<int32_t> row_ptr{0};
    std::vector<int32_t> col_ind;
    std::vector<float> values;
};

Matrix make_matrix(bool quad_heavy = false) {
    Matrix matrix{521, {0}, {}, {}};
    // Odd dimensions and lengths straddle every row-mapping boundary. Long
    // rows contain repeated columns, which must retain their additive meaning.
    const std::vector<int> lengths{0, 1, 3, 16, 17, 31, 32, 33, 63, 64, 65,
                                   127, 128, 129, 255, 256, 257, 4095, 4096,
                                   4097, 8193, 16385};
    std::mt19937 random(19);
    for (int row = 0; row < 263; ++row) {
        int length = row < static_cast<int>(lengths.size())
                         ? lengths[row]
                         : (quad_heavy ? 17 + row % 48 : row % 7);
        for (int k = 0; k < length; ++k) {
            matrix.col_ind.push_back(static_cast<int32_t>(random() % matrix.columns));
            matrix.values.push_back(static_cast<float>(static_cast<int>(random() % 2001) - 1000) /
                                    317.0f);
        }
        matrix.row_ptr.push_back(static_cast<int32_t>(matrix.values.size()));
    }
    return matrix;
}

Matrix transpose(const Matrix &a) {
    Matrix result{static_cast<int>(a.row_ptr.size()) - 1,
                  std::vector<int32_t>(a.columns + 1, 0),
                  std::vector<int32_t>(a.values.size()),
                  std::vector<float>(a.values.size())};
    for (int column : a.col_ind) ++result.row_ptr[column + 1];
    for (int row = 0; row < a.columns; ++row)
        result.row_ptr[row + 1] += result.row_ptr[row];
    auto next = result.row_ptr;
    for (int row = 0; row < result.columns; ++row) {
        for (int k = a.row_ptr[row]; k < a.row_ptr[row + 1]; ++k) {
            int index = next[a.col_ind[k]]++;
            result.col_ind[index] = row;
            result.values[index] = a.values[k];
        }
    }
    return result;
}

void check_matrix(const Matrix &a, const char *label, mx::Stream stream) {
    int rows = static_cast<int>(a.row_ptr.size()) - 1;
    std::vector<float> input(a.columns);
    for (int col = 0; col < a.columns; ++col)
        input[col] = static_cast<float>((col * 17) % 257 - 128) / 129.7f;
    std::vector<double> reference(rows, 0.0), absolute_sum(rows, 0.0);
    for (int row = 0; row < rows; ++row) {
        for (int k = a.row_ptr[row]; k < a.row_ptr[row + 1]; ++k) {
            double term = static_cast<double>(a.values[k]) * input[a.col_ind[k]];
            reference[row] += term;
            absolute_sum[row] += std::fabs(term);
        }
    }

    auto work = detail::make_adaptive_metal_work(a.row_ptr, rows);
    mx::array rp(a.row_ptr.data(), {rows + 1});
    mx::array ci(a.col_ind.data(), {static_cast<int>(a.col_ind.size())});
    auto compact_ci = detail::metal_column_indices(a.col_ind, a.columns);
    const auto expected_dtype = a.columns <= 65536 ? mx::uint16 : mx::int32;
    if (compact_ci.dtype() != expected_dtype)
        throw std::runtime_error("incorrect Metal column-index storage width");
    mx::array values(a.values.data(), {static_cast<int>(a.values.size())});
    mx::array offsets(work.offsets.data(), {static_cast<int>(work.offsets.size())});
    mx::array descriptors(work.rows.data(), {static_cast<int>(work.rows.size())});
    mx::array x(input.data(), {a.columns});
    for (auto strategy : {SparseMetalSpmvStrategy::scalar_rows,
                          SparseMetalSpmvStrategy::quad_rows,
                          SparseMetalSpmvStrategy::simdgroup_rows,
                          SparseMetalSpmvStrategy::adaptive}) {
        // Poison outputs so skipped tail rows cannot pass by reusing a
        // previously correct buffer from MLX's allocation cache.
        constexpr float unwritten = -1234567.0f;
        auto result = detail::metal_spmv(rp, ci, values, offsets, descriptors, x,
                                         rows, work.item_count, strategy, stream, unwritten);
        mx::eval(result);
        auto compact_result = detail::metal_spmv(rp, compact_ci, values, offsets,
                                                 descriptors, x, rows, work.item_count,
                                                 strategy, stream, unwritten);
        mx::eval(compact_result);
        const float *actual = result.data<float>();
        const float *compact_actual = compact_result.data<float>();
        for (int row = 0; row < rows; ++row) {
            const double tolerance = 5e-7 + 5e-6 * absolute_sum[row];
            if (!std::isfinite(actual[row]) || actual[row] != compact_actual[row] ||
                std::fabs(actual[row] - reference[row]) > tolerance) {
                std::fprintf(stderr, "%s strategy=%d row=%d actual=%.9g reference=%.17g\n",
                             label, static_cast<int>(strategy), row, actual[row], reference[row]);
                throw std::runtime_error("Metal SpMV disagrees with independent FP64 sum");
            }
        }
        std::printf("%s strategy=%d PASS\n", label, static_cast<int>(strategy));
    }
}
} // namespace

int main() {
    try {
        if (!mx::is_available(mx::Device::gpu)) return 77;
        auto stream = mx::default_stream(mx::Device::gpu);
        mx::StreamContext context(stream);
        auto matrix = make_matrix();
        check_matrix(matrix, "A", stream);
        check_matrix(transpose(matrix), "AT", stream);
        check_matrix(make_matrix(true), "quad-heavy", stream);
        Matrix empty{67, std::vector<int32_t>(260, 0), {}, {}};
        check_matrix(empty, "empty", stream);
        Matrix boundary{65536, {0, 0, 4, 5}, {0, 32767, 32768, 65535, 65535},
                        {0.3f, -1.7f, 0.9f, 1.1f, -0.7f}};
        check_matrix(boundary, "uint16-boundary", stream);
        boundary.columns = 65537;
        boundary.col_ind.back() = 65536;
        check_matrix(boundary, "int32-fallback", stream);
        mx::synchronize(stream);
    } catch (const std::exception &error) {
        std::fprintf(stderr, "%s\n", error.what());
        return 1;
    }
}
