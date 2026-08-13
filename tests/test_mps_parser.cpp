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

#include "mlxPDLP/mps_loader.h"

#include <cmath>
#include <cstdio>

namespace {

int failures = 0;

#define CHECK(condition, message)                                                                  \
    do {                                                                                           \
        if (!(condition)) {                                                                        \
            std::fprintf(stderr, "FAIL: %s (line %d)\n", message, __LINE__);                    \
            ++failures;                                                                            \
        }                                                                                          \
    } while (0)

void test_omitted_rhs_vector_name() {
    mlxpdlp_mps_problem_t *problem =
        mlxpdlp_mps_problem_load(OMITTED_RHS_NAME_MPS);
    CHECK(problem != nullptr, "fixture should parse");
    if (!problem)
        return;

    CHECK(problem->num_variables == 1, "variable count");
    CHECK(problem->num_constraints == 2, "constraint count");
    CHECK(std::fabs(problem->objective[0] - 2.0) <= 1e-15,
          "objective coefficient");
    CHECK(std::fabs(problem->constraint_lb[0] - 3.0) <= 1e-15,
          "unnamed RHS lower bound");
    CHECK(std::isinf(problem->constraint_ub[0]) &&
              problem->constraint_ub[0] > 0.0,
          "greater-than upper bound");
    CHECK(std::isinf(problem->constraint_lb[1]) &&
              problem->constraint_lb[1] < 0.0,
          "less-than lower bound");
    CHECK(std::fabs(problem->constraint_ub[1] - 4.0) <= 1e-15,
          "unnamed RHS upper bound");

    mlxpdlp_mps_problem_free(problem);
}

void test_fixed_column_names_with_spaces() {
    mlxpdlp_mps_problem_t *problem =
        mlxpdlp_mps_problem_load(FIXED_NAMES_WITH_SPACES_MPS);
    CHECK(problem != nullptr, "fixed-column fixture should parse");
    if (!problem)
        return;

    CHECK(problem->num_variables == 1, "fixed-column variable count");
    CHECK(problem->num_constraints == 1, "fixed-column constraint count");
    CHECK(problem->num_nonzeros == 1, "fixed-column nonzero count");
    CHECK(std::fabs(problem->objective[0] - 0.02466) <= 1e-15,
          "fixed-column objective coefficient");
    CHECK(std::fabs(problem->values[0] + 1.0) <= 1e-15,
          "fixed-column matrix coefficient");
    CHECK(std::fabs(problem->constraint_lb[0] - 3.0) <= 1e-15,
          "fixed-column RHS");

    mlxpdlp_mps_problem_free(problem);
}

void test_free_format_long_names_are_not_truncated() {
    mlxpdlp_mps_problem_t *problem =
        mlxpdlp_mps_problem_load(FREE_FORMAT_LONG_NAMES_MPS);
    CHECK(problem != nullptr, "free-format long-name fixture should parse");
    if (!problem)
        return;

    CHECK(problem->num_variables == 1, "long variable name should remain one column");
    CHECK(problem->num_constraints == 2,
          "long row names with a shared prefix must remain distinct");
    CHECK(problem->num_nonzeros == 2, "long-name matrix nonzero count");
    CHECK(std::fabs(problem->objective[0] - 3.0) <= 1e-15,
          "long-name objective coefficient");
    CHECK(problem->row_ptr[0] == 0 && problem->row_ptr[1] == 1 &&
              problem->row_ptr[2] == 2,
          "long-name rows should each retain their coefficient");
    CHECK(std::fabs(problem->values[0] - 1.0) <= 1e-15 &&
              std::fabs(problem->values[1] - 2.0) <= 1e-15,
          "long-name matrix coefficients");
    CHECK(std::fabs(problem->constraint_ub[0] - 4.0) <= 1e-15,
          "long-name less-than RHS");
    CHECK(std::fabs(problem->constraint_lb[1] - 5.0) <= 1e-15,
          "long-name greater-than RHS");

    mlxpdlp_mps_problem_free(problem);
}

void test_free_aligned_numeric_fields_use_cupdlpx_tokenization() {
    mlxpdlp_mps_problem_t *problem =
        mlxpdlp_mps_problem_load(FREE_ALIGNED_NUMERIC_MPS);
    CHECK(problem != nullptr, "free-aligned numeric fixture should parse");
    if (!problem)
        return;

    CHECK(problem->num_variables == 1, "free-aligned variable count");
    CHECK(problem->num_constraints == 1, "free-aligned constraint count");
    CHECK(problem->num_nonzeros == 1, "free-aligned nonzero count");
    CHECK(std::fabs(problem->objective[0] + 3.20005933333333) <= 1e-14,
          "free-aligned objective must retain the full token");
    CHECK(std::fabs(problem->values[0] - 1.0) <= 1e-15,
          "free-aligned matrix coefficient");
    CHECK(std::fabs(problem->constraint_lb[0] - 1.0) <= 1e-15 &&
              std::fabs(problem->constraint_ub[0] - 1.0) <= 1e-15,
          "free-aligned RHS");
    CHECK(std::fabs(problem->variable_lb[0]) <= 1e-15 &&
              std::fabs(problem->variable_ub[0] - 1.0) <= 1e-15,
          "free-aligned upper bound must not default to zero");

    mlxpdlp_mps_problem_free(problem);
}

} // namespace

int main() {
    test_omitted_rhs_vector_name();
    test_fixed_column_names_with_spaces();
    test_free_format_long_names_are_not_truncated();
    test_free_aligned_numeric_fields_use_cupdlpx_tokenization();
    if (failures == 0)
        std::printf("All MPS parser tests passed.\n");
    return failures == 0 ? 0 : 1;
}
