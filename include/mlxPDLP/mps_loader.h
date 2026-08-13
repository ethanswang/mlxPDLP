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

#ifdef __cplusplus
extern "C" {
#endif

// Parsed LP storage owned by the loader. Array pointers remain valid until
// mlxpdlp_mps_problem_free() is called.
typedef struct {
    int num_variables;
    int num_constraints;
    int num_nonzeros;

    const int *row_ptr;
    const int *col_ind;
    const double *values;
    const double *variable_lb;
    const double *variable_ub;
    const double *constraint_lb;
    const double *constraint_ub;
    const double *objective;
    double objective_constant;
    int maximize;

    void *internal;
} mlxpdlp_mps_problem_t;

// Returns NULL on I/O or parse failure.
mlxpdlp_mps_problem_t *mlxpdlp_mps_problem_load(const char *path);

// Releases the parsed problem and all arrays. Accepts NULL.
void mlxpdlp_mps_problem_free(mlxpdlp_mps_problem_t *problem);

#ifdef __cplusplus
}
#endif
