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

#include "mps_parser_internal.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void *safe_malloc(size_t size) {
    void *pointer = malloc(size);
    if (!pointer && size != 0) {
        perror("Fatal error: malloc failed");
        exit(EXIT_FAILURE);
    }
    return pointer;
}

void *safe_calloc(size_t count, size_t size) {
    void *pointer = calloc(count, size);
    if (!pointer && count != 0 && size != 0) {
        perror("Fatal error: calloc failed");
        exit(EXIT_FAILURE);
    }
    return pointer;
}

void *safe_realloc(void *pointer, size_t new_size) {
    if (new_size == 0) {
        free(pointer);
        return NULL;
    }
    void *resized = realloc(pointer, new_size);
    if (!resized) {
        perror("Fatal error: realloc failed");
        exit(EXIT_FAILURE);
    }
    return resized;
}

void lp_problem_free(lp_problem_t *problem) {
    if (!problem) {
        return;
    }
    free(problem->constraint_matrix_row_pointers);
    free(problem->constraint_matrix_col_indices);
    free(problem->constraint_matrix_values);
    free(problem->variable_lower_bound);
    free(problem->variable_upper_bound);
    free(problem->objective_vector);
    free(problem->constraint_lower_bound);
    free(problem->constraint_upper_bound);
    free(problem->primal_start);
    free(problem->dual_start);
    memset(problem, 0, sizeof(*problem));
    free(problem);
}

mlxpdlp_mps_problem_t *mlxpdlp_mps_problem_load(const char *path) {
    lp_problem_t *parsed = read_mps_file(path);
    if (!parsed) {
        return NULL;
    }

    mlxpdlp_mps_problem_t *problem = calloc(1, sizeof(*problem));
    if (!problem) {
        lp_problem_free(parsed);
        return NULL;
    }

    problem->num_variables = parsed->num_variables;
    problem->num_constraints = parsed->num_constraints;
    problem->num_nonzeros = parsed->constraint_matrix_num_nonzeros;
    problem->row_ptr = parsed->constraint_matrix_row_pointers;
    problem->col_ind = parsed->constraint_matrix_col_indices;
    problem->values = parsed->constraint_matrix_values;
    problem->variable_lb = parsed->variable_lower_bound;
    problem->variable_ub = parsed->variable_upper_bound;
    problem->constraint_lb = parsed->constraint_lower_bound;
    problem->constraint_ub = parsed->constraint_upper_bound;
    problem->objective = parsed->objective_vector;
    problem->objective_constant = parsed->objective_constant;
    problem->maximize = parsed->objective_sense == OBJECTIVE_SENSE_MAXIMIZE;
    problem->internal = parsed;
    return problem;
}

void mlxpdlp_mps_problem_free(mlxpdlp_mps_problem_t *problem) {
    if (!problem) {
        return;
    }
    lp_problem_free((lp_problem_t *)problem->internal);
    free(problem);
}
