// Copyright 2026 Ethan Wang <ethanshurui.wang@gmail.com>
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "mlxPDLP/solver.h"

namespace mlxpdlp::detail {
// Mutable only during a standalone solver's preparation or plan construction.
// A prepared plan and all its solve states share this owner without copying CSR.
struct SolverMatrixStorage {
    std::vector<int> original_row_ptr_;
    std::vector<int> original_col_ind_;
    std::vector<double> original_matrix_values_;
    std::vector<int> working_row_ptr_;
    std::vector<int> working_col_ind_;
    std::vector<double> working_matrix_values_;
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
    mx::array sparse_at_row_ptr_ = _mlx_empty_array();
    mx::array sparse_at_col_ind_ = _mlx_empty_array();
    mx::array sparse_at_values_ = _mlx_empty_array();
    mx::array sparse_at_work_offsets_ = _mlx_empty_array();
    mx::array sparse_at_work_rows_ = _mlx_empty_array();
    int sparse_a_work_item_count_ = 0;
    int sparse_at_work_item_count_ = 0;
    double sparse_frobenius_norm_ = 0.0;
    std::shared_ptr<detail::CpuSparseMatrix> sparse_cpu_matrix_;
    size_t resident_bytes() const;
};
} // namespace mlxpdlp::detail
