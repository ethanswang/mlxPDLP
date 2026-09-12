// Copyright 2026 Ethan Wang <ethanshurui.wang@gmail.com>
// SPDX-License-Identifier: Apache-2.0
#include "shared_matrix.h"
#include "cpu_sparse_matrix.h"

namespace mlxpdlp::detail {
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
                         &sparse_at_work_offsets_, &sparse_at_work_rows_}) total += a->nbytes();
    // Accelerate retains its own CSR storage and handle. Reserve a conservative
    // additional CSR+transpose allowance when that backend is present.
    if (sparse_cpu_matrix_) total += 32 * sparse_a_values_host_.size() +
                                    16 * (sparse_a_row_ptr_host_.size() + sparse_at_row_ptr_host_.size());
    return total;
}
} // namespace mlxpdlp::detail
