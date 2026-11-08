// Copyright 2026 Ethan Wang <ethanshurui.wang@gmail.com>
// SPDX-License-Identifier: Apache-2.0
#include "shared_matrix.h"
#include "cpu_sparse_matrix.h"

namespace mlxpdlp::detail {
std::vector<double> absolute_csr_values(const std::vector<int32_t> &row_ptr,
                                      const std::vector<int32_t> &col_ind,
                                      const std::vector<double> &values) {
    std::vector<double> result(values.size(), 0.0);
    std::vector<std::pair<int32_t, int32_t>> entries;
    for (size_t row = 1; row < row_ptr.size(); ++row) {
        entries.clear();
        for (int k = row_ptr[row - 1]; k < row_ptr[row]; ++k)
            entries.emplace_back(col_ind[k], k);
        std::sort(entries.begin(), entries.end());
        for (size_t begin = 0; begin < entries.size();) {
            size_t end = begin + 1;
            double value = values[entries[begin].second];
            while (end < entries.size() && entries[end].first == entries[begin].first)
                value += values[entries[end++].second];
            // Take abs after merging duplicates: cancellation must not inflate
            // the activity used to accept an infeasibility certificate.
            result[entries[begin].second] = std::abs(value);
            begin = end;
        }
    }
    return result;
}

size_t SolverMatrixStorage::resident_bytes() const {
    size_t total = sizeof(*this);
    auto count = [&](const auto &v) { total += v.capacity() * sizeof(typename std::decay_t<decltype(v)>::value_type); };
    count(original_row_ptr_); count(original_col_ind_); count(original_matrix_values_);
    count(working_row_ptr_); count(working_col_ind_); count(working_matrix_values_);
    count(sparse_a_row_ptr_host_); count(sparse_a_col_ind_host_); count(sparse_a_values_host_);
    count(sparse_at_row_ptr_host_); count(sparse_at_col_ind_host_); count(sparse_at_source_index_);
    count(sparse_con_rescale_host_); count(sparse_var_rescale_host_);
    for (const auto *a : {&sparse_a_row_ptr_, &sparse_a_col_ind_, &sparse_a_values_,
                         &sparse_a_work_offsets_, &sparse_a_work_rows_,
                         &sparse_at_row_ptr_, &sparse_at_col_ind_, &sparse_at_values_,
                         &sparse_at_work_offsets_, &sparse_at_work_rows_,
                         &absolute_a_, &absolute_at_}) total += a->nbytes();
    // Accelerate retains its own CSR storage and handle. Reserve a conservative
    // additional CSR+transpose allowance when that backend is present.
    if (sparse_cpu_matrix_) total += 32 * sparse_a_values_host_.size() +
                                    16 * (sparse_a_row_ptr_host_.size() + sparse_at_row_ptr_host_.size());
    if (sparse_cpu_absolute_matrix_) total += 32 * sparse_a_values_host_.size() +
                                    16 * (sparse_a_row_ptr_host_.size() + sparse_at_row_ptr_host_.size());
    return total;
}
} // namespace mlxpdlp::detail
