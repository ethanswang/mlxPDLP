// Shared Metal CSR reduction bodies for standalone products and fused PDHG.
#pragma once
#include "mlxPDLP/solver.h"
#include <algorithm>
#include <stdexcept>
#include <string>
#include <utility>

namespace mlxpdlp::detail {
// The solver validates column bounds before this upload. Values and row
// pointers retain their existing precision; only exact integer indices shrink.
inline mx::array metal_column_indices(const std::vector<int32_t> &columns, int column_count) {
    const int nonzeros = static_cast<int>(columns.size());
    if (column_count <= 65536) {
        std::vector<uint16_t> compact(columns.begin(), columns.end());
        return mx::array(compact.data(), {nonzeros}, mx::uint16);
    }
    return mx::array(columns.data(), {nonzeros}, mx::int32);
}

struct MetalSpmvWork {
    std::vector<int32_t> offsets{0};
    std::vector<int32_t> rows;
    int item_count = 0;
};

// A work item packs rows with the same thread mapping. Quad rows use ~row
// descriptors; the other mappings use ordinary row indices and CSR lengths.
inline MetalSpmvWork make_adaptive_metal_work(const std::vector<int32_t> &row_ptr, int row_count) {
    constexpr int short_row_max_nonzeros = 64;
    constexpr int medium_row_max_nonzeros = 4096;
    constexpr size_t simdgroup_width = 32;
    constexpr size_t threadgroup_width = 256;
    constexpr size_t simdgroups_per_threadgroup =
        threadgroup_width / simdgroup_width;
    std::vector<int32_t> short_rows;
    std::vector<int32_t> quad_rows;
    std::vector<int32_t> medium_rows;
    std::vector<int32_t> long_rows;
    int64_t short_nonzeros = 0;
    short_rows.reserve(static_cast<size_t>(row_count));
    for (int row = 0; row < row_count; ++row) {
        const int length = row_ptr[static_cast<size_t>(row) + 1] -
                           row_ptr[static_cast<size_t>(row)];
        if (length <= short_row_max_nonzeros) {
            short_rows.push_back(row);
            short_nonzeros += length;
        } else if (length <= medium_row_max_nonzeros) {
            medium_rows.push_back(row);
        } else {
            long_rows.push_back(row);
        }
    }

    // Packing four lanes per row pays off when the short-row population has
    // enough arithmetic. A mostly tiny population with a few 17-64 entry
    // rows is faster in the scalar pack (notably SCPM1 and EX10's A).
    if (short_nonzeros >= 24LL * static_cast<int64_t>(short_rows.size())) {
        size_t scalar_count = 0;
        for (int32_t row : short_rows) {
            if (row_ptr[row + 1] - row_ptr[row] > 16)
                quad_rows.push_back(row);
            else
                short_rows[scalar_count++] = row;
        }
        short_rows.resize(scalar_count);
    }

    MetalSpmvWork work;
    work.rows.reserve(short_rows.size() + quad_rows.size() + medium_rows.size() + long_rows.size());
    work.offsets.reserve((short_rows.size() + threadgroup_width - 1) /
                             threadgroup_width +
                         (quad_rows.size() + 63) / 64 +
                         (medium_rows.size() + simdgroups_per_threadgroup - 1) /
                             simdgroups_per_threadgroup +
                         long_rows.size() + 1);
    for (size_t begin = 0; begin < short_rows.size(); begin += threadgroup_width) {
        const size_t end =
            std::min(begin + threadgroup_width, short_rows.size());
        work.rows.insert(work.rows.end(), short_rows.begin() + begin,
                         short_rows.begin() + end);
        work.offsets.push_back(static_cast<int32_t>(work.rows.size()));
    }
    for (size_t begin = 0; begin < quad_rows.size(); begin += 64) {
        const size_t end = std::min(begin + 64, quad_rows.size());
        for (size_t index = begin; index < end; ++index)
            work.rows.push_back(~quad_rows[index]);
        work.offsets.push_back(static_cast<int32_t>(work.rows.size()));
    }
    for (size_t begin = 0; begin < medium_rows.size();
         begin += simdgroups_per_threadgroup) {
        const size_t end =
            std::min(begin + simdgroups_per_threadgroup, medium_rows.size());
        for (size_t index = begin; index < end; ++index) {
            work.rows.push_back(medium_rows[index]);
        }
        work.offsets.push_back(static_cast<int32_t>(work.rows.size()));
    }
    for (int32_t row : long_rows) {
        work.rows.push_back(row);
        work.offsets.push_back(static_cast<int32_t>(work.rows.size()));
    }
    work.item_count = static_cast<int>(work.offsets.size()) - 1;
    return work;
}

inline std::string replace_all(std::string text, const std::string &from, const std::string &to) {
    size_t pos = 0;
    while ((pos = text.find(from, pos)) != std::string::npos) {
        text.replace(pos, from.size(), to);
        pos += to.size();
    }
    return text;
}

// Assemble the kernel body for one SpMV dispatch strategy. The RS/CI/VALS
// tokens name the CSR arrays, VEC the operand vector, WO/WR the adaptive work
// arrays, and UPD the output/update call using the register accumulator acc.
inline std::string metal_spmv_body(const char *rs, const char *ci, const char *vals,
                       const char *wo, const char *wr, const char *vec,
                       SparseMetalSpmvStrategy strategy,
                       const std::string &update_acc) {
    std::string body;
    switch (strategy) {
    case SparseMetalSpmvStrategy::scalar_rows:
        body = R"(
            uint row = thread_position_in_grid.x;
            float acc = 0.0f;
            int begin = $RS$[row];
            int end = $RS$[row + 1];
            for (int k = begin; k < end; ++k) {
                acc = fma($VALS$[k], $VEC$[$CI$[k]], acc);
            }
            $UPD$
        )";
        break;
    case SparseMetalSpmvStrategy::quad_rows:
        body = R"(
            uint row = thread_position_in_grid.x / 4;
            uint lane = thread_index_in_simdgroup & 3;
            float acc = 0.0f;
            int begin = $RS$[row];
            int end = $RS$[row + 1];
            for (int k = begin + int(lane); k < end; k += 4) {
                acc = fma($VALS$[k], $VEC$[$CI$[k]], acc);
            }
            acc += simd_shuffle_down(acc, ushort(2));
            acc += simd_shuffle_down(acc, ushort(1));
            if (lane == 0) {
                $UPD$
            }
        )";
        break;
    case SparseMetalSpmvStrategy::simdgroup_rows:
        body = R"(
            constexpr uint simdgroup_width = 32;
            uint lane = thread_index_in_simdgroup;
            uint row = thread_position_in_grid.x / simdgroup_width;
            float partial = 0.0f;
            int begin = $RS$[row];
            int end = $RS$[row + 1];
            for (int k = begin + int(lane); k < end; k += int(simdgroup_width)) {
                partial = fma($VALS$[k], $VEC$[$CI$[k]], partial);
            }
            float acc = simd_sum(partial);
            if (lane == 0) {
                $UPD$
            }
        )";
        break;
    case SparseMetalSpmvStrategy::adaptive:
        body = R"(
            uint local_thread = thread_index_in_threadgroup;
            uint threadgroup_width = threads_per_threadgroup.x;
            uint work_item = threadgroup_position_in_grid.x;
            int descriptor_begin = $WO$[work_item];
            int descriptor_end = $WO$[work_item + 1];
            int first_descriptor = $WR$[descriptor_begin];
            int first_row = first_descriptor < 0 ? ~first_descriptor : first_descriptor;
            int first_length = $RS$[first_row + 1] - $RS$[first_row];
            threadgroup float partials[8];
            if (first_length > 64 && first_length <= 4096) {
                uint simd_lane = local_thread & 31;
                uint simd_group = local_thread >> 5;
                int descriptor = descriptor_begin + int(simd_group);
                if (descriptor < descriptor_end) {
                    uint row = uint($WR$[descriptor]);
                    float acc = 0.0f;
                    int begin = $RS$[row];
                    int end = $RS$[row + 1];
                    for (int k = begin + int(simd_lane); k < end; k += 32) {
                        acc = fma($VALS$[k], $VEC$[$CI$[k]], acc);
                    }
                    acc = simd_sum(acc);
                    if (simd_lane == 0) {
                        $UPD$
                    }
                }
            } else if (first_length > 4096) {
                uint row = uint(first_row);
                float partial = 0.0f;
                int begin = $RS$[row];
                int end = $RS$[row + 1];
                for (int k = begin + int(local_thread); k < end;
                     k += int(threadgroup_width)) {
                    partial = fma($VALS$[k], $VEC$[$CI$[k]], partial);
                }
                uint lane = local_thread & 31;
                uint group = local_thread >> 5;
                float group_sum = simd_sum(partial);
                if (lane == 0) {
                    partials[group] = group_sum;
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                if (group == 0) {
                    float acc = simd_sum(lane < 8 ? partials[lane] : 0.0f);
                    if (lane == 0) {
                        $UPD$
                    }
                }
            } else if (first_descriptor < 0) {
                uint lane = local_thread & 3;
                int descriptor = descriptor_begin + int(local_thread >> 2);
                if (descriptor < descriptor_end) {
                    uint row = uint(~$WR$[descriptor]);
                    float acc = 0.0f;
                    int begin = $RS$[row];
                    int end = $RS$[row + 1];
                    for (int k = begin + int(lane); k < end; k += 4) {
                        acc = fma($VALS$[k], $VEC$[$CI$[k]], acc);
                    }
                    acc += simd_shuffle_down(acc, ushort(2));
                    acc += simd_shuffle_down(acc, ushort(1));
                    if (lane == 0) {
                        $UPD$
                    }
                }
            } else {
                int descriptor = descriptor_begin + int(local_thread);
                if (descriptor < descriptor_end) {
                    uint row = uint($WR$[descriptor]);
                    float acc = 0.0f;
                    int begin = $RS$[row];
                    int end = $RS$[row + 1];
                    for (int k = begin; k < end; ++k) {
                        acc = fma($VALS$[k], $VEC$[$CI$[k]], acc);
                    }
                    $UPD$
                }
            }
        )";
        break;
    }
    body = replace_all(std::move(body), "$RS$", rs);
    body = replace_all(std::move(body), "$CI$", ci);
    body = replace_all(std::move(body), "$VALS$", vals);
    body = replace_all(std::move(body), "$WO$", wo);
    body = replace_all(std::move(body), "$WR$", wr);
    body = replace_all(std::move(body), "$VEC$", vec);
    body = replace_all(std::move(body), "$UPD$", update_acc);
    return body;
}

inline mx::array metal_spmv(
    const mx::array &row_ptr, const mx::array &col_ind, const mx::array &values,
    const mx::array &work_offsets, const mx::array &work_rows, const mx::array &x,
    int rows, int work_item_count, SparseMetalSpmvStrategy strategy, mx::Stream stream,
    std::optional<float> initial_value = std::nullopt) {
    constexpr int threadgroup_width = 256;
    auto make_kernel = [](const char *name, SparseMetalSpmvStrategy kind) {
        std::vector<std::string> names{"row_starts", "column_indices", "nonzeros", "vector"};
        if (kind == SparseMetalSpmvStrategy::adaptive) {
            names = {"row_starts", "column_indices", "nonzeros", "work_offsets", "work_rows", "vector"};
        }
        return mx::fast::metal_kernel(name, names, {"output"},
            metal_spmv_body("row_starts", "column_indices", "nonzeros",
                            "work_offsets", "work_rows", "vector", kind,
                            "output[row] = acc;"));
    };
    const mx::fast::CustomKernelFunction *kernel = nullptr;
    int grid = rows;
    switch (strategy) {
    case SparseMetalSpmvStrategy::scalar_rows: {
        static const auto value = make_kernel("mlxpdlp_csr_spmv_scalar_rows", strategy);
        kernel = &value;
        break;
    }
    case SparseMetalSpmvStrategy::quad_rows: {
        static const auto value = make_kernel("mlxpdlp_csr_spmv_quad_rows", strategy);
        kernel = &value;
        grid *= 4;
        break;
    }
    case SparseMetalSpmvStrategy::simdgroup_rows: {
        static const auto value = make_kernel("mlxpdlp_csr_spmv_simdgroup_rows", strategy);
        kernel = &value;
        grid *= 32;
        break;
    }
    case SparseMetalSpmvStrategy::adaptive: {
        static const auto value = make_kernel("mlxpdlp_csr_spmv_adaptive", strategy);
        return value({row_ptr, col_ind, values, work_offsets, work_rows, x},
                     {{rows}}, {mx::float32}, {work_item_count * threadgroup_width, 1, 1},
                     {threadgroup_width, 1, 1}, {}, initial_value, false, stream)[0];
    }
    }
    if (!kernel) throw std::logic_error("unknown Metal SpMV strategy");
    return (*kernel)({row_ptr, col_ind, values, x}, {{rows}}, {mx::float32},
                     {grid, 1, 1}, {threadgroup_width, 1, 1}, {}, initial_value,
                     false, stream)[0];
}
} // namespace mlxpdlp::detail
