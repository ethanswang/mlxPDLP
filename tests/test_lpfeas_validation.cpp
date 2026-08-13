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

#include "lpfeas_support.h"

#include <cmath>
#include <cstdio>

using namespace mlxpdlp;
using namespace mlxpdlp::benchmark;

namespace {

int failures = 0;

#define CHECK(condition, message)                                                                  \
    do {                                                                                           \
        if (!(condition)) {                                                                        \
            std::fprintf(stderr, "FAIL: %s (line %d)\n", message, __LINE__);                     \
            ++failures;                                                                            \
        }                                                                                          \
    } while (0)

#define CHECK_CLOSE(actual, expected, tolerance, message)                                          \
    CHECK(std::fabs((actual) - (expected)) <= (tolerance), message)

struct Fixture {
    int row_ptr[2] = {0, 1};
    int col_ind[1] = {0};
    double values[1] = {1.0};
    double variable_lb[1] = {0.0};
    double variable_ub[1] = {INFINITY};
    double constraint_lb[1] = {1.0};
    double constraint_ub[1] = {INFINITY};
    double objective[1] = {1.0};
    double primal[1] = {1.0};
    double dual[1] = {1.0};
    double reduced_cost[1] = {0.0};
    mlxpdlp_mps_problem_t problem{};
    mlxpdlp_result_t result{};

    Fixture() {
        problem.num_variables = 1;
        problem.num_constraints = 1;
        problem.num_nonzeros = 1;
        problem.row_ptr = row_ptr;
        problem.col_ind = col_ind;
        problem.values = values;
        problem.variable_lb = variable_lb;
        problem.variable_ub = variable_ub;
        problem.constraint_lb = constraint_lb;
        problem.constraint_ub = constraint_ub;
        problem.objective = objective;
        result.num_variables = 1;
        result.num_constraints = 1;
        result.primal_solution = primal;
        result.dual_solution = dual;
        result.reduced_cost = reduced_cost;
    }
};

void test_exact_certificate() {
    Fixture fixture;
    ValidationMetrics metrics =
        validate_original_problem(fixture.problem, fixture.result, fixture.objective, 0.0);
    CHECK(metrics.satisfies(1e-12), "exact certificate should validate");
    CHECK_CLOSE(metrics.primal_objective, 1.0, 1e-15, "primal objective");
    CHECK_CLOSE(metrics.dual_objective, 1.0, 1e-15, "dual objective");
    CHECK_CLOSE(metrics.relative_objective_gap, 0.0, 1e-15, "zero gap");
}

void test_primal_infeasibility_is_detected() {
    Fixture fixture;
    fixture.primal[0] = 0.0;
    ValidationMetrics metrics =
        validate_original_problem(fixture.problem, fixture.result, fixture.objective, 0.0);
    CHECK_CLOSE(metrics.absolute_primal_residual, 1.0, 1e-15, "absolute primal residual");
    CHECK_CLOSE(metrics.relative_primal_residual, 0.5, 1e-15, "relative primal residual");
    CHECK(!metrics.satisfies(1e-6), "infeasible point must fail validation");
}

void test_cuda_gap_normalization() {
    Fixture fixture;
    fixture.primal[0] = 2.0;
    ValidationMetrics metrics =
        validate_original_problem(fixture.problem, fixture.result, fixture.objective, 0.0);
    CHECK_CLOSE(metrics.objective_gap, 1.0, 1e-15, "absolute objective gap");
    CHECK_CLOSE(metrics.relative_objective_gap, 0.25, 1e-15,
                "gap must use 1 + |primal| + |dual|");
}

void test_dual_infeasibility_is_detected() {
    Fixture fixture;
    fixture.dual[0] = 0.0;
    ValidationMetrics metrics =
        validate_original_problem(fixture.problem, fixture.result, fixture.objective, 0.0);
    CHECK_CLOSE(metrics.relative_dual_residual, 0.5, 1e-15, "relative dual residual");
    CHECK(!metrics.satisfies(1e-6), "dual-infeasible point must fail validation");
}

} // namespace

int main() {
    test_exact_certificate();
    test_primal_infeasibility_is_detected();
    test_cuda_gap_normalization();
    test_dual_infeasibility_is_detected();
    if (failures == 0)
        std::printf("All LPfeas validation tests passed.\n");
    return failures == 0 ? 0 : 1;
}
