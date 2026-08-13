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

int main() {
    mlxpdlp::pdhg_parameters_t parameters;
    mlxpdlp::mlxpdlp_set_default_parameters(&parameters);

#ifdef MLXPDLP_CONSUMER_HAS_MPS
    mlxpdlp_mps_problem_free(nullptr);
#endif

    return parameters.termination_criteria.iteration_limit > 0 ? 0 : 1;
}
