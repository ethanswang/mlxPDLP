"""The Python entry points share the C++ parameter contract and diagnostics."""
import math

import numpy as np
import pytest

import mlxpdlp


def parameters():
    p = mlxpdlp.Parameters()
    p.verbose = False
    p.presolve = False
    return p


def matrix():
    return dict(num_variables=1, num_constraints=1,
                row_ptr=np.array([0, 1], dtype=np.int32),
                col_indices=np.array([0], dtype=np.int32), values=np.ones(1))


def single(p):
    return mlxpdlp.Solver(**matrix(), objective=np.ones(1),
                          variable_lower_bounds=np.zeros(1),
                          variable_upper_bounds=np.ones(1),
                          constraint_lower_bounds=np.array([0.5]),
                          constraint_upper_bounds=np.array([0.5]),
                          parameters=p, device="cpu")


def plan(p):
    return mlxpdlp.SharedMatrixPlan(**matrix(), parameters=p, device="cpu")


def set_parameter(p, path, value):
    parts = path.split(".")
    owner = p
    for name in parts[:-1]:
        owner = getattr(owner, name)
    setattr(owner, parts[-1], value)


@pytest.fixture(scope="module")
def prepared_plan():
    return plan(parameters())


@pytest.mark.parametrize("field,value", [
    ("termination_criteria.eps_optimal_relative", -1),
    ("termination_criteria.eps_optimal_relative", math.nan),
    ("termination_criteria.eps_optimal_relative", math.inf),
    ("termination_criteria.eps_feasible_relative", -1),
    ("termination_criteria.eps_feasible_relative", math.nan),
    ("termination_criteria.eps_feasible_relative", math.inf),
    ("termination_criteria.eps_feas_polish_relative", -1),
    ("termination_criteria.eps_feas_polish_relative", math.nan),
    ("termination_criteria.eps_infeasible_relative", 0),
    ("termination_criteria.iteration_limit", -1),
    ("termination_criteria.time_sec_limit", -1),
    ("termination_criteria.time_sec_limit", -math.inf),
    ("termination_criteria.time_sec_limit", math.nan),
    ("geometric_mean_iterations", -1),
    ("curtis_reid_iterations", -1),
    ("l_inf_ruiz_iterations", -1),
    ("pock_chambolle_alpha", math.nan),
    ("pock_chambolle_alpha", -0.1),
    ("pock_chambolle_alpha", 2.1),
    ("termination_evaluation_frequency", 0),
    ("sv_max_iter", -1),
    ("sv_tol", 0),
    ("sv_tol", math.nan),
    ("sv_tol", math.inf),
    ("restart_policy", 2),
    ("reflection_coefficient", 0),
    ("reflection_coefficient", 1.1),
    ("reflection_coefficient", math.nan),
    ("restart_params.k_p", math.nan),
    ("restart_params.k_i", math.inf),
    ("restart_params.k_d", -math.inf),
    ("restart_params.i_smooth", -0.1),
    ("restart_params.i_smooth", 1.1),
    ("restart_params.i_smooth", math.nan),
    ("restart_params.artificial_restart_threshold", math.inf),
    ("restart_params.sufficient_reduction_for_restart", math.nan),
    ("restart_params.necessary_reduction_for_restart", math.nan),
    ("matrix_zero_tol", -1),
    ("matrix_zero_tol", math.inf),
    ("matrix_zero_tol", math.nan),
    ("host_double_polishing_iteration_limit", -1),
    ("host_double_polishing_time_sec_limit", -1),
    ("host_double_polishing_time_sec_limit", math.nan),
])
def test_invalid_parameters_agree(prepared_plan, field, value):
    p = parameters()
    set_parameter(p, field, value)
    errors = []
    for action in (
        lambda: single(p),
        lambda: plan(p),
        lambda: prepared_plan.solve_batch(objective=np.empty((0, 1)), parameters=p),
    ):
        with pytest.raises(ValueError) as error:
            action()
        assert field in str(error.value)
        errors.append(str(error.value))
    assert errors[0] == errors[1] == errors[2]


@pytest.mark.parametrize("field,value", [
    ("termination_criteria.iteration_limit", 0),
    ("termination_criteria.time_sec_limit", 0),
    ("termination_criteria.time_sec_limit", math.inf),
    ("geometric_mean_iterations", 0),
    ("curtis_reid_iterations", 0),
    ("l_inf_ruiz_iterations", 0),
    ("pock_chambolle_alpha", 0),
    ("pock_chambolle_alpha", 2),
    ("termination_evaluation_frequency", 1),
    ("sv_max_iter", 0),
    ("restart_policy", 1),
    ("optimality_norm", mlxpdlp.NormType.L_INF),
    ("restart_params.i_smooth", 0),
    ("restart_params.i_smooth", 1),
    ("matrix_zero_tol", 0),
    ("host_double_polishing_iteration_limit", 0),
    ("host_double_polishing_time_sec_limit", 0),
    ("host_double_polishing_time_sec_limit", math.inf),
])
def test_valid_boundaries(field, value):
    p = parameters()
    set_parameter(p, field, value)
    single(p)
    prepared = plan(p)
    assert prepared.solve_batch(objective=np.empty((0, 1)), parameters=p).results == []


def test_zero_tolerance_alias_keeps_fixed_work_behavior():
    p = parameters()
    p.tolerance = 0
    p.iteration_limit = 7
    p.termination_evaluation_frequency = 3
    result = single(p).solve()
    prepared = plan(p)
    batch = prepared.solve_batch(objective=np.ones((1, 1)),
                                variable_lower_bounds=np.zeros(1),
                                variable_upper_bounds=np.ones(1),
                                constraint_lower_bounds=np.array([0.5]),
                                constraint_upper_bounds=np.array([0.5]))
    assert result.total_count == batch.results[0].total_count == 7
    assert result.termination_reason_name == batch.results[0].termination_reason_name == "ITERATION_LIMIT"
