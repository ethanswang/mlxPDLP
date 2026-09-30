// Copyright 2026 Ethan Wang <ethanshurui.wang@gmail.com>
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "mlxPDLP/solver.h"

namespace mlxpdlp::detail {
// Shared by single solves, batch preparation and batch submission. Python
// reaches these same entry points and translates invalid_argument to ValueError.
void validate_parameters(const pdhg_parameters_t &parameters);
} // namespace mlxpdlp::detail
