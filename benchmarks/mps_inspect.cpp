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

#include <cstdio>

int main(int argc, char **argv) {
    if (argc < 2) {
        std::fprintf(stderr, "Usage: %s MPS_PATH [MPS_PATH ...]\n", argv[0]);
        return 2;
    }

    int failures = 0;
    std::printf("path\tconstraint_rows\tcolumns\tmatrix_nonzeros\tobjective_nonzeros\n");
    for (int i = 1; i < argc; ++i) {
        mlxpdlp_mps_problem_t *problem = mlxpdlp_mps_problem_load(argv[i]);
        if (!problem) {
            std::fprintf(stderr, "Failed to parse MPS file: %s\n", argv[i]);
            ++failures;
            continue;
        }
        int objective_nonzeros = 0;
        for (int column = 0; column < problem->num_variables; ++column) {
            if (problem->objective[column] != 0.0) {
                ++objective_nonzeros;
            }
        }
        std::printf("%s\t%d\t%d\t%d\t%d\n", argv[i], problem->num_constraints,
                    problem->num_variables, problem->num_nonzeros, objective_nonzeros);
        mlxpdlp_mps_problem_free(problem);
    }
    return failures == 0 ? 0 : 1;
}
