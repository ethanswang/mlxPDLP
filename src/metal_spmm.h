// Copyright 2026 Ethan Wang <ethanshurui.wang@gmail.com>
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "mlxPDLP/solver.h"
#include "mlx/fast.h"
#include <algorithm>
#include <map>
#include <mutex>
#include <tuple>

namespace mlxpdlp::detail {

struct MetalSpmmRowWork {
    // A negative descriptor (~offset) packs short rows; a nonnegative one
    // packs SIMD reductions. The final descriptor is the end sentinel.
    std::vector<int32_t> offsets;
    std::vector<int32_t> rows;
};

inline MetalSpmmRowWork make_metal_spmm_row_work(const std::vector<int32_t> &row_ptr,
                                               int tile) {
    MetalSpmmRowWork work;
    std::vector<int32_t> short_rows, long_rows;
    for (size_t r = 0; r + 1 < row_ptr.size(); ++r)
        (row_ptr[r + 1] - row_ptr[r] <= 16 ? short_rows : long_rows).push_back(int32_t(r));
    work.rows.reserve(short_rows.size() + long_rows.size());
    auto append = [&](const std::vector<int32_t> &rows, int capacity, bool scalar) {
        for (size_t begin = 0; begin < rows.size(); begin += capacity) {
            const int32_t offset = static_cast<int32_t>(work.rows.size());
            work.offsets.push_back(scalar ? ~offset : offset);
            const size_t end = std::min(begin + capacity, rows.size());
            work.rows.insert(work.rows.end(), rows.begin() + begin, rows.begin() + end);
        }
    };
    append(short_rows, 256 / tile, true);
    append(long_rows, 8, false);
    work.offsets.push_back(static_cast<int32_t>(work.rows.size()));
    return work;
}

struct MetalSpmmRowSchedule {
    mx::array offsets, rows;
    int work_items;
    explicit MetalSpmmRowSchedule(const MetalSpmmRowWork &work)
        : offsets(work.offsets.data(), {static_cast<int>(work.offsets.size())}, mx::int32),
          rows(work.rows.data(), {static_cast<int>(work.rows.size())}, mx::int32),
          work_items(static_cast<int>(work.offsets.size()) - 1) {}
};

// A SIMD group covers LP lanes and, for longer rows, several nonzeros.
// Each coefficient/index is loaded by one LP lane and broadcast to its tile.
// All lanes reach shuffles even when a member has finished. Only active lanes
// enter the arithmetic/projection, so NaNs and padded lanes cannot mix members.
inline const mx::fast::CustomKernelFunction &spmm_kernel(int tile, int reduction,
                                                        int kind, bool major) {
    using Key = std::tuple<int, int, int, bool>;
    static std::mutex mutex;
    static std::map<Key, mx::fast::CustomKernelFunction> kernels;
    std::lock_guard<std::mutex> guard(mutex);
    const Key key{tile, reduction, kind, major};
    auto found = kernels.find(key);
    if (found != kernels.end()) return found->second;
    const bool scheduled = reduction == 0;
    std::string body = "constexpr uint TILE = " + std::to_string(tile) + ";\n" +
                       "constexpr uint RED = " + std::to_string(scheduled ? 32 / tile : reduction) + ";\n";
    if (scheduled) {
        body += R"(
        uint local = thread_index_in_threadgroup;
        uint group = threadgroup_position_in_grid.x;
        int descriptor = work_offsets[group];
        int next_descriptor = work_offsets[group + 1];
        int begin = descriptor < 0 ? ~descriptor : descriptor;
        int end = next_descriptor < 0 ? ~next_descriptor : next_descriptor;
        uint slot = uint(begin) + (descriptor < 0 ? local / TILE : local / 32);
        // A partial pack drops whole LP tiles or whole SIMD groups. Every
        // surviving tile still participates in all of its coefficient shuffles.
        if (slot >= uint(end)) return;
        uint row = uint(work_rows[slot]);
        uint lane = local % TILE;
        uint part = descriptor < 0 ? 0 : (local / TILE) % RED;
        )";
    } else {
        body += R"(
        uint lane = thread_position_in_grid.x % TILE;
        uint part = (thread_position_in_grid.x / TILE) % RED;
        uint row = thread_position_in_grid.x / (TILE * RED);
        )";
    }
    body += R"(
        uint width = vector_shape[1];
        uint member = thread_position_in_grid.y * TILE + lane;
        uint index = row * width + member;
        bool run = active[member] != 0;
        float acc = 0.0f;
        uint source_lane = thread_index_in_simdgroup & ~(TILE - 1);
    )";
    if (scheduled) body += R"(
        if (descriptor < 0) {
            for (int k = starts[row]; k < starts[row + 1]; ++k) {
                float value = lane == 0 ? values[k] : 0.0f;
                uint column = lane == 0 ? uint(columns[k]) : 0;
                value = simd_shuffle(value, ushort(source_lane));
                column = simd_shuffle(column, ushort(source_lane));
                if (run) acc = fma(value, vector[column * width + member], acc);
            }
        } else {
    )";
    body += R"(
        for (int k = starts[row] + int(part); k < starts[row + 1]; k += int(RED)) {
            float value = lane == 0 ? values[k] : 0.0f;
            uint column = lane == 0 ? uint(columns[k]) : 0;
            value = simd_shuffle(value, ushort(source_lane));
            column = simd_shuffle(column, ushort(source_lane));
            if (run) acc = fma(value, vector[column * width + member], acc);
        }
        for (uint offset = RED / 2; offset > 0; offset /= 2)
            acc += simd_shuffle_down(acc, ushort(offset * TILE));
    )";
    if (scheduled) body += "}\n";
    body += R"(
        if (part == 0) {
    )";
    std::vector<std::string> inputs{"starts", "columns", "values", "vector", "active"};
    std::vector<std::string> outputs;
    if (kind == 0) {
        outputs = {"product"};
        body += "product[index] = acc;\n";
    } else {
        inputs.insert(inputs.end(), {"cur", "anchor", "ref", "lower", "upper", "objective", "coeff"});
        outputs = {"next", "reflection"};
        if (major) {
            inputs.push_back("old_pdhg");
            outputs.push_back("pdhg");
            if (kind == 1) {
                inputs.push_back("old_slack");
                outputs.push_back("slack");
            }
        }
        body += "if (!run) { next[index] = cur[index]; reflection[index] = ref[index];\n";
        if (major) body += "pdhg[index] = old_pdhg[index];\n";
        if (major && kind == 1) body += "slack[index] = old_slack[index];\n";
        body += "} else {\n";
        body += "float step = coeff[" + std::to_string(kind - 1) + " * width + member];\n";
        body += "float rc = coeff[2 * width + member];\nfloat weight = coeff[3 * width + member];\n";
        if (kind == 1) {
            body += "float temp = cur[index] - step * (objective[index] - acc);\n";
            body += "float candidate = min(max(temp, lower[index]), upper[index]);\n";
        } else {
            body += "float temp = cur[index] / step - acc;\n";
            body += "float projected = min(max(temp, -upper[index]), -lower[index]);\n";
            body += "float candidate = (temp - projected) * step;\n";
        }
        body += R"(
            float reflected = 2.0f * candidate - cur[index];
            reflection[index] = reflected;
            next[index] = weight * (rc * reflected + (1.0f - rc) * cur[index])
                          + (1.0f - weight) * anchor[index];
        )";
        if (major) body += "pdhg[index] = candidate;\n";
        if (major && kind == 1) body += "slack[index] = (candidate - temp) / step;\n";
        body += "}\n";
    }
    body += "}\n";
    if (scheduled) inputs.insert(inputs.end(), {"work_offsets", "work_rows"});
    const std::string name = "mlxpdlp_spmm_" + std::to_string(tile) + "_" +
        std::to_string(reduction) + "_" + std::to_string(kind) + (major ? "_major" : "_minor");
    return kernels.emplace(key, mx::fast::metal_kernel(name, inputs, outputs, body)).first->second;
}

inline std::vector<mx::array> metal_spmm_dispatch(const std::vector<mx::array> &inputs,
                                                int rows, int tile, int reduction,
                                                int kind, bool major, mx::Stream stream,
                                                const MetalSpmmRowSchedule *schedule = nullptr) {
    const int width = inputs[3].shape(1);
    const int outputs = kind == 0 ? 1 : (major ? (kind == 1 ? 4 : 3) : 2);
    std::vector<mx::Shape> shapes(outputs, {rows, width});
    if (rows == 0) return std::vector<mx::array>(outputs, mx::zeros({rows, width}, mx::float32, stream));
    if (schedule) {
        auto scheduled_inputs = inputs;
        scheduled_inputs.insert(scheduled_inputs.end(), {schedule->offsets, schedule->rows});
        return spmm_kernel(tile, 0, kind, major)(
            scheduled_inputs, shapes, std::vector<mx::Dtype>(outputs, mx::float32),
            {schedule->work_items * 256, width / tile, 1}, {256, 1, 1}, {}, std::nullopt, false, stream);
    }
    return spmm_kernel(tile, reduction, kind, major)(
        inputs, shapes, std::vector<mx::Dtype>(outputs, mx::float32),
        {rows * tile * reduction, width / tile, 1}, {256, 1, 1}, {}, std::nullopt, false, stream);
}

inline mx::array metal_spmm(const mx::array &row_ptr, const mx::array &columns,
                             const mx::array &values, const mx::array &vectors,
                             const mx::array &active, int rows, int tile,
                             int reduction, mx::Stream stream,
                             const MetalSpmmRowSchedule *schedule = nullptr) {
    return metal_spmm_dispatch({row_ptr, columns, values, vectors, active}, rows,
                               tile, reduction, 0, false, stream, schedule)[0];
}
} // namespace mlxpdlp::detail
