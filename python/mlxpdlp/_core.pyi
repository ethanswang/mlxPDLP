# Copyright 2026 Ethan Wang <ethanshurui.wang@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Type stubs for the mlxpdlp._core extension module."""

import enum
from typing import Optional

import numpy as np

__version__: str


class TerminationCriteria:
    eps_optimal_relative: float
    eps_feasible_relative: float
    eps_feas_polish_relative: float
    time_sec_limit: float
    iteration_limit: int

    def __init__(self) -> None: ...


class RestartParameters:
    artificial_restart_threshold: float
    sufficient_reduction_for_restart: float
    necessary_reduction_for_restart: float
    k_p: float
    k_i: float
    k_d: float
    i_smooth: float

    def __init__(self) -> None: ...


class NormType(enum.IntEnum):
    L2: int
    L_INF: int


class Parameters:
    curtis_reid_iterations: int
    l_inf_ruiz_iterations: int
    has_pock_chambolle_alpha: bool
    pock_chambolle_alpha: float
    bound_objective_rescaling: bool
    verbose: bool
    termination_evaluation_frequency: int
    sv_max_iter: int
    sv_tol: float
    termination_criteria: TerminationCriteria
    restart_params: RestartParameters
    restart_policy: int
    reflection_coefficient: float
    feasibility_polishing: bool
    host_double_polishing: bool
    host_double_early_handoff: bool
    host_double_polishing_iteration_limit: int
    host_double_polishing_time_sec_limit: float
    optimality_norm: NormType
    presolve: bool
    presolve_singleton_columns: bool
    presolve_doubleton_equations: bool
    presolve_parallel_rows: bool
    presolve_parallel_columns: bool
    presolve_dual_fix: bool
    presolve_finite_bound_tightening: bool
    presolve_primal_propagation: bool
    matrix_zero_tol: float
    tolerance: float
    time_limit_seconds: float
    iteration_limit: int

    def __init__(self) -> None: ...


class TerminationReason(enum.IntEnum):
    UNSPECIFIED: int
    OPTIMAL: int
    PRIMAL_INFEASIBLE: int
    DUAL_INFEASIBLE: int
    INFEASIBLE_OR_UNBOUNDED: int
    TIME_LIMIT: int
    ITERATION_LIMIT: int
    FEAS_POLISH_SUCCESS: int
    HOST_DOUBLE_HANDOFF: int


class SolveResult:
    primal_solution: np.ndarray
    dual_solution: np.ndarray
    reduced_cost: np.ndarray
    num_variables: int
    num_constraints: int
    num_nonzeros: int
    num_reduced_variables: int
    num_reduced_constraints: int
    num_reduced_nonzeros: int
    total_count: int
    rescaling_time_sec: float
    cumulative_time_sec: float
    presolve_time: float
    presolve_status: int
    absolute_primal_residual: float
    relative_primal_residual: float
    absolute_dual_residual: float
    relative_dual_residual: float
    primal_objective_value: float
    dual_objective_value: float
    objective_gap: float
    relative_objective_gap: float
    max_primal_ray_infeasibility: float
    max_dual_ray_infeasibility: float
    primal_ray_linear_objective: float
    dual_ray_objective: float
    termination_reason: int
    termination_reason_name: str
    feasibility_polishing_time: float
    feasibility_iteration: int
    host_double_polishing_time: float
    host_double_polishing_iteration: int
    host_double_handoff: bool


class Solver:
    def __init__(
        self,
        num_variables: int,
        num_constraints: int,
        row_ptr: np.ndarray,
        col_indices: np.ndarray,
        values: np.ndarray,
        variable_lower_bounds: Optional[np.ndarray] = None,
        variable_upper_bounds: Optional[np.ndarray] = None,
        constraint_lower_bounds: Optional[np.ndarray] = None,
        constraint_upper_bounds: Optional[np.ndarray] = None,
        objective: np.ndarray,
        objective_constant: float = 0.0,
        parameters: Optional[Parameters] = None,
        primal_start: Optional[np.ndarray] = None,
        dual_start: Optional[np.ndarray] = None,
        reduced_cost_start: Optional[np.ndarray] = None,
        device: str = "cpu",
    ) -> None: ...

    def solve(self) -> SolveResult: ...

    def expects_sparse_metal_backend(self) -> bool: ...

    def expects_sparse_cpu_backend(self) -> bool: ...


class MpsProblem:
    num_variables: int
    num_constraints: int
    num_nonzeros: int
    maximize: int
    objective_constant: float
    row_ptr: np.ndarray
    col_ind: np.ndarray
    values: np.ndarray
    variable_lb: np.ndarray
    variable_ub: np.ndarray
    constraint_lb: np.ndarray
    constraint_ub: np.ndarray
    objective: np.ndarray


def has_gpu() -> bool: ...


def version() -> str: ...


def load_mps(path: str) -> MpsProblem: ...
