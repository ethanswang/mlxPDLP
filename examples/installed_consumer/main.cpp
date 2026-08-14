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

#include <mlxPDLP/solver.h>

#ifdef MLXPDLP_CONSUMER_HAS_MPS
#include <mlxPDLP/mps_loader.h>
#endif

#include <cmath>
#include <cstdio>
#include <memory>

int main() {
    mlxpdlp::pdhg_parameters_t parameters;
    mlxpdlp::mlxpdlp_set_default_parameters(&parameters);
    parameters.verbose = false;
    parameters.presolve = false;
    parameters.feasibility_polishing = false;
    parameters.host_double_polishing = false;
    parameters.termination_evaluation_frequency = 20;
    parameters.termination_criteria.eps_optimal_relative = 1e-4;
    parameters.termination_criteria.eps_feasible_relative = 1e-4;

    // minimize x subject to x = 1 and x >= 0.
    int row_ptr[] = {0, 1};
    int col_ind[] = {0};
    double values[] = {1.0};
    double variable_lb[] = {0.0};
    double variable_ub[] = {INFINITY};
    double constraint_bound[] = {1.0};
    double objective[] = {1.0};

    const mlxpdlp::mx::Device device = mlxpdlp::mx::is_available(mlxpdlp::mx::Device::gpu)
                                           ? mlxpdlp::mx::Device::gpu
                                           : mlxpdlp::mx::Device::cpu;
    mlxpdlp::MlxPdlpSolver solver(1, 1, row_ptr, col_ind, values, variable_lb, variable_ub,
                                  constraint_bound, constraint_bound, objective, 0.0, &parameters,
                                  device);
    std::unique_ptr<mlxpdlp::mlxpdlp_result_t, decltype(&mlxpdlp::mlxpdlp_result_free)> result(
        solver.solve(), mlxpdlp::mlxpdlp_result_free);
    mlxpdlp::mx::synchronize(solver.state().stream);

    const bool valid = solver.state().stream.device == device &&
                       result->termination_reason == mlxpdlp::TERMINATION_REASON_OPTIMAL &&
                       std::fabs(result->primal_solution[0] - 1.0) <= 5e-3;
    std::printf("Installed mlxPDLP consumer: device=%s x=%.8f %s\n",
                device.type == mlxpdlp::mx::Device::gpu ? "Metal" : "CPU",
                result->primal_solution[0], valid ? "PASS" : "FAIL");

#ifdef MLXPDLP_CONSUMER_HAS_MPS
    mlxpdlp_mps_problem_free(nullptr);
#endif

    return valid ? 0 : 1;
}
