// Copyright 2026 Ethan Wang <ethanshurui.wang@gmail.com>
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "mlxPDLP/solver.h"
#include "mlx/fast.h"
#include <map>
#include <mutex>
#include <tuple>

namespace mlxpdlp::detail {

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
    std::string body = "constexpr uint TILE = " + std::to_string(tile) + ";\n" +
                       "constexpr uint RED = " + std::to_string(reduction) + ";\n";
    body += R"(
        uint lane = thread_position_in_grid.x % TILE;
        uint part = (thread_position_in_grid.x / TILE) % RED;
        uint row = thread_position_in_grid.x / (TILE * RED);
        uint width = vector_shape[1];
        uint member = thread_position_in_grid.y * TILE + lane;
        uint index = row * width + member;
        bool run = active[member] != 0;
        float acc = 0.0f;
        uint source_lane = thread_index_in_simdgroup & ~(TILE - 1);
        for (int k = starts[row] + int(part); k < starts[row + 1]; k += int(RED)) {
            float value = lane == 0 ? values[k] : 0.0f;
            uint column = lane == 0 ? uint(columns[k]) : 0;
            value = simd_shuffle(value, ushort(source_lane));
            column = simd_shuffle(column, ushort(source_lane));
            if (run) acc = fma(value, vector[column * width + member], acc);
        }
        for (uint offset = RED / 2; offset > 0; offset /= 2)
            acc += simd_shuffle_down(acc, ushort(offset * TILE));
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
    const std::string name = "mlxpdlp_spmm_" + std::to_string(tile) + "_" +
        std::to_string(reduction) + "_" + std::to_string(kind) + (major ? "_major" : "_minor");
    return kernels.emplace(key, mx::fast::metal_kernel(name, inputs, outputs, body)).first->second;
}

inline std::vector<mx::array> metal_spmm_dispatch(const std::vector<mx::array> &inputs,
                                                int rows, int tile, int reduction,
                                                int kind, bool major, mx::Stream stream) {
    const int width = inputs[3].shape(1);
    const int outputs = kind == 0 ? 1 : (major ? (kind == 1 ? 4 : 3) : 2);
    std::vector<mx::Shape> shapes(outputs, {rows, width});
    if (rows == 0) return std::vector<mx::array>(outputs, mx::zeros({rows, width}, mx::float32, stream));
    return spmm_kernel(tile, reduction, kind, major)(
        inputs, shapes, std::vector<mx::Dtype>(outputs, mx::float32),
        {rows * tile * reduction, width / tile, 1}, {256, 1, 1}, {}, std::nullopt, false, stream);
}

inline mx::array metal_spmm(const mx::array &row_ptr, const mx::array &columns,
                             const mx::array &values, const mx::array &vectors,
                             const mx::array &active, int rows, int tile,
                             int reduction, mx::Stream stream) {
    return metal_spmm_dispatch({row_ptr, columns, values, vectors, active}, rows,
                               tile, reduction, 0, false, stream)[0];
}
} // namespace mlxpdlp::detail
