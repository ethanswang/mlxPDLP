/*
Copyright 2026 Ethan Wang <ethanshurui.wang@gmail.com>

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/
#pragma once

#include "mlx/array.h"
#include "mlx/stream.h"

#include <array>
#include <vector>

namespace mlxpdlp::detail {

inline constexpr int max_metal_batch_iterations = 16;
using BatchScalars = std::array<std::array<float, 3>, 2>;

// Replay the two minor half-step kernels with owned intermediate buffers.
// Prototype outputs provide kernel metadata and input roles; they are not
// evaluated. The returned arrays are {x_cur, x_ref, y_cur} and remain lazy.
std::vector<mlx::core::array> metal_minor_batch(const std::vector<mlx::core::array> &primal,
                                                const std::vector<mlx::core::array> &dual,
                                                std::vector<BatchScalars> scalars,
                                                mlx::core::Stream stream);

} // namespace mlxpdlp::detail
