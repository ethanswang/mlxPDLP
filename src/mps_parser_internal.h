/*
Copyright 2025 Haihao Lu
Copyright 2026 Ethan Wang <ethanshurui.wang@gmail.com>

This file is derived from cuPDLPx (https://github.com/MIT-Lu-Lab/cuPDLPx),
ported from CUDA to Apple MLX/Metal and modified for the mlxPDLP Metal FP32
and CPU FP64 backends.

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

#include <stddef.h>

typedef enum { OBJECTIVE_SENSE_MINIMIZE = 0, OBJECTIVE_SENSE_MAXIMIZE = 1 } objective_sense_t;

typedef struct {
    int num_variables;
    int num_constraints;
    double *variable_lower_bound;
    double *variable_upper_bound;
    double *objective_vector;
    double objective_constant;
    objective_sense_t objective_sense;

    int *constraint_matrix_row_pointers;
    int *constraint_matrix_col_indices;
    double *constraint_matrix_values;
    int constraint_matrix_num_nonzeros;

    double *constraint_lower_bound;
    double *constraint_upper_bound;

    double *primal_start;
    double *dual_start;
} lp_problem_t;

void *safe_malloc(size_t size);
void *safe_calloc(size_t count, size_t size);
void *safe_realloc(void *pointer, size_t new_size);

lp_problem_t *read_mps_file(const char *filename);
void lp_problem_free(lp_problem_t *problem);
