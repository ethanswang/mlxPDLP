// Shared RAII ownership for Accelerate FP64 sparse products. Construction
// canonicalizes duplicate/unsorted columns; products leave inputs untouched.
#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <utility>
#include <vector>
#ifdef MLXPDLP_HAS_ACCELERATE_SPARSE
#include <Accelerate/Accelerate.h>
#endif

namespace mlxpdlp::detail {
#ifdef MLXPDLP_HAS_ACCELERATE_SPARSE
struct CpuSparseMatrix {
    explicit CpuSparseMatrix(int rows, int columns)
        : handle(sparse_matrix_create_double(rows, columns)) {
        if (!handle)
            throw std::runtime_error("Accelerate failed to create a sparse matrix");
    }
    ~CpuSparseMatrix() {
        sparse_matrix_destroy(handle);
    }
    CpuSparseMatrix(const CpuSparseMatrix &) = delete;
    CpuSparseMatrix &operator=(const CpuSparseMatrix &) = delete;

    void assign_csr(int rows, const int *row_ptr, const int *col_ind, const double *values) {
        std::vector<std::pair<int32_t, double>> entries;
        std::vector<double> row_values;
        std::vector<sparse_index> row_columns;
        double squared_frobenius_norm = 0.0;

        for (int row = 0; row < rows; ++row) {
            entries.clear();
            for (int32_t k = row_ptr[static_cast<size_t>(row)];
                 k < row_ptr[static_cast<size_t>(row) + 1]; ++k) {
                const double value = values[static_cast<size_t>(k)];
                if (!std::isfinite(value)) {
                    throw std::runtime_error("non-finite coefficient in sparse CPU matrix");
                }
                entries.emplace_back(col_ind[static_cast<size_t>(k)], value);
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
                const sparse_status status =
                    sparse_insert_row_double(handle, static_cast<sparse_index>(row),
                                             static_cast<sparse_dimension>(row_values.size()),
                                             row_values.data(), row_columns.data());
                if (status != SPARSE_SUCCESS) {
                    throw std::runtime_error("Accelerate failed to insert a sparse matrix row");
                }
                nonzeros += static_cast<int64_t>(row_values.size());
            }
        }
        if (sparse_commit(handle) != SPARSE_SUCCESS) {
            throw std::runtime_error("Accelerate failed to commit the sparse matrix");
        }

        frobenius_norm = std::sqrt(squared_frobenius_norm);
    }

    void multiply(bool transpose, const double *input, double *output, int output_size) const {
        std::fill_n(output, output_size, 0.0);
        const sparse_status status = sparse_matrix_vector_product_dense_double(
            transpose ? CblasTrans : CblasNoTrans, 1.0, handle, input, 1, output, 1);
        if (status != SPARSE_SUCCESS)
            throw std::runtime_error("Accelerate sparse matrix-vector product failed");
    }
    sparse_matrix_double handle = nullptr;
    int64_t nonzeros = 0;
    double frobenius_norm = 0.0;
};
#else
struct CpuSparseMatrix {};
#endif
} // namespace mlxpdlp::detail
