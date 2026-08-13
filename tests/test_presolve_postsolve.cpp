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
#include "presolve_adapter.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <memory>
#include <sys/stat.h>
#include <vector>

namespace {

struct ProblemDeleter {
    void operator()(mlxpdlp_mps_problem_t *problem) const {
        mlxpdlp_mps_problem_free(problem);
    }
};

struct PresolveDeleter {
    void operator()(mlxpdlp::detail::PresolveContext *context) const {
        mlxpdlp::detail::destroy_presolve(context);
    }
};

double project_zero(double lower, double upper) {
    return std::max(lower, std::min(0.0, upper));
}

double max_abs(const std::vector<double> &values) {
    double result = 0.0;
    for (double value : values)
        result = std::max(result, std::fabs(value));
    return result;
}

} // namespace

int main() {
    // The irish-e fixture is an LPfeas instance and is not distributed with
    // this repository. Skip with CTest's skip code when the corpus has not
    // been downloaded (benchmarks/data/lpfeas/download.sh).
    struct stat fixture_info;
    if (stat(IRISH_E_MPS, &fixture_info) != 0) {
        std::printf("skipping: irish-e regression fixture not downloaded "
                    "(run benchmarks/data/lpfeas/download.sh)\n");
        return 77;
    }

    std::unique_ptr<mlxpdlp_mps_problem_t, ProblemDeleter> problem(
        mlxpdlp_mps_problem_load(IRISH_E_MPS));
    if (!problem) {
        std::fprintf(stderr, "failed to load irish-e regression fixture\n");
        return 1;
    }

    std::vector<double> objective(problem->objective,
                                  problem->objective + problem->num_variables);
    double objective_constant = problem->objective_constant;
    if (problem->maximize) {
        for (double &coefficient : objective)
            coefficient = -coefficient;
        objective_constant = -objective_constant;
    }

    mlxpdlp::detail::HostProblemView original{
        problem->num_variables,
        problem->num_constraints,
        problem->num_nonzeros,
        problem->row_ptr,
        problem->col_ind,
        problem->values,
        problem->variable_lb,
        problem->variable_ub,
        problem->constraint_lb,
        problem->constraint_ub,
        objective.data(),
        objective_constant,
    };
    mlxpdlp::detail::PresolveOptions options;
    options.primal_propagation = false;
    auto outcome = mlxpdlp::detail::run_presolve(original, 0.0, options, false);
    std::unique_ptr<mlxpdlp::detail::PresolveContext, PresolveDeleter> context(outcome.context);

    const auto &reduced = outcome.reduced_problem;
    std::vector<double> primal(static_cast<size_t>(reduced.num_variables));
    std::vector<double> dual(static_cast<size_t>(reduced.num_constraints), 1.0);
    std::vector<double> reduced_cost(static_cast<size_t>(reduced.num_variables), 0.0);
    for (int column = 0; column < reduced.num_variables; ++column) {
        primal[static_cast<size_t>(column)] =
            project_zero(reduced.variable_lower_bound[column],
                         reduced.variable_upper_bound[column]);
    }

    std::vector<double> original_lb(problem->variable_lb,
                                    problem->variable_lb + problem->num_variables);
    std::vector<double> original_ub(problem->variable_ub,
                                    problem->variable_ub + problem->num_variables);
    auto postsolved = mlxpdlp::detail::postsolve(context.get(), primal.data(), dual.data(),
                                                reduced_cost.data(), original_lb, original_ub);

    const double maximum_dual = max_abs(postsolved.dual);
    const double maximum_reduced_cost = max_abs(postsolved.reduced_cost);
    if (!std::isfinite(maximum_dual) || !std::isfinite(maximum_reduced_cost) ||
        maximum_dual > 1e12 || maximum_reduced_cost > 1e12) {
        std::fprintf(stderr,
                     "irish-e postsolve amplified a bounded reduced certificate: "
                     "max|y|=%.17g max|z|=%.17g\n",
                     maximum_dual, maximum_reduced_cost);
        return 1;
    }

    std::printf("irish-e postsolve stability regression passed: max|y|=%.6g max|z|=%.6g\n",
                maximum_dual, maximum_reduced_cost);
    return 0;
}
