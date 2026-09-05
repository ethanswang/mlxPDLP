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

"""Unit tests for the mlxpdlp Python binding."""

import os

import numpy as np
import pytest

import mlxpdlp


def small_lp_arrays():
    # minimize -x0 - x1 subject to x0 + x1 <= 1, x >= 0  -> objective -1.
    row_ptr = np.array([0, 2], dtype=np.int32)
    col_ind = np.array([0, 1], dtype=np.int32)
    values = np.array([1.0, 1.0])
    var_lb = np.zeros(2)
    var_ub = np.full(2, np.inf)
    con_lb = np.array([-np.inf])
    con_ub = np.array([1.0])
    objective = np.array([-1.0, -1.0])
    return row_ptr, col_ind, values, var_lb, var_ub, con_lb, con_ub, objective


def solve_small(device="cpu", **kwargs):
    arrays = small_lp_arrays()
    solver = mlxpdlp.Solver(2, 1, *arrays, **kwargs)
    return solver.solve()


def test_version():
    assert mlxpdlp.version() == "0.1.0"
    assert mlxpdlp.__version__ == "0.1.0"


def test_parameters_defaults_and_roundtrip():
    p = mlxpdlp.Parameters()
    assert p.presolve is True
    assert p.termination_criteria.eps_infeasible_relative == 1e-14
    assert mlxpdlp.TerminationCriteria().eps_infeasible_relative == 1e-14
    p.termination_criteria.eps_infeasible_relative = 1e-6
    assert p.geometric_mean_iterations == 12
    p.geometric_mean_iterations = 3
    assert p.geometric_mean_iterations == 3
    p.tolerance = 1e-5
    assert p.termination_criteria.eps_optimal_relative == 1e-5
    assert p.termination_criteria.eps_feasible_relative == 1e-5
    assert p.termination_criteria.eps_feas_polish_relative <= 1e-6
    assert p.termination_criteria.eps_infeasible_relative == 1e-6
    p.iteration_limit = 1234
    assert p.termination_criteria.iteration_limit == 1234
    p.time_limit_seconds = 12.5
    assert p.termination_criteria.time_sec_limit == 12.5
    p.restart_params.k_p = 0.25
    assert p.restart_params.k_p == 0.25
    assert p.conditional_termination_evaluation is True
    p.conditional_termination_evaluation = False
    assert p.conditional_termination_evaluation is False


def test_small_lp_cpu():
    result = solve_small("cpu")
    assert result.termination_reason == int(mlxpdlp.TerminationReason.OPTIMAL)
    assert result.termination_reason_name == "OPTIMAL"
    x = result.primal_solution
    assert x.shape == (2,)
    assert x.sum() == pytest.approx(1.0, abs=1e-6)
    assert (x >= -1e-9).all()
    assert result.primal_objective_value == pytest.approx(-1.0, abs=1e-6)
    assert result.relative_primal_residual < 1e-5
    assert result.relative_dual_residual < 1e-5
    assert result.relative_objective_gap < 1e-5


@pytest.mark.skipif(not mlxpdlp.has_gpu(), reason="no Metal GPU device")
def test_small_lp_gpu():
    result = solve_small("gpu")
    assert result.termination_reason_name == "OPTIMAL"
    assert result.primal_solution.sum() == pytest.approx(1.0, abs=1e-3)
    assert result.primal_objective_value == pytest.approx(-1.0, abs=1e-3)


def test_dtype_flexibility():
    # int64 indices and float32 values must be accepted (converted to the
    # solver's int32/double CSR storage).
    arrays = small_lp_arrays()
    row_ptr, col_ind, values, var_lb, var_ub, con_lb, con_ub, obj = arrays
    solver = mlxpdlp.Solver(
        2,
        1,
        row_ptr.astype(np.int64),
        col_ind.astype(np.int64),
        values.astype(np.float32),
        var_lb,
        var_ub,
        con_lb,
        con_ub,
        obj,
    )
    result = solver.solve()
    assert result.primal_objective_value == pytest.approx(-1.0, abs=1e-6)


def test_iteration_limit_termination():
    params = mlxpdlp.Parameters()
    params.presolve = False
    params.termination_evaluation_frequency = 1
    params.iteration_limit = 1
    arrays = small_lp_arrays()
    solver = mlxpdlp.Solver(2, 1, *arrays, parameters=params)
    result = solver.solve()
    assert result.termination_reason in (
        int(mlxpdlp.TerminationReason.ITERATION_LIMIT),
        int(mlxpdlp.TerminationReason.TIME_LIMIT),
    )


def test_warm_start_with_presolve_rejected():
    arrays = small_lp_arrays()
    row_ptr, col_ind, values, var_lb, var_ub, con_lb, con_ub, obj = arrays
    params = mlxpdlp.Parameters()
    params.presolve = True
    with pytest.raises(ValueError, match="presolve"):
        mlxpdlp.Solver(
            2, 1, row_ptr, col_ind, values, var_lb, var_ub, con_lb, con_ub,
            obj, parameters=params, primal_start=np.zeros(2),
        )


def test_warm_start_roundtrip():
    # A warm start near the optimum with presolve disabled must solve.
    arrays = small_lp_arrays()
    params = mlxpdlp.Parameters()
    params.presolve = False
    solver = mlxpdlp.Solver(
        2, 1, *arrays, parameters=params,
        primal_start=np.array([0.5, 0.5]), dual_start=np.array([-1.0]),
    )
    result = solver.solve()
    assert result.termination_reason_name == "OPTIMAL"
    assert result.primal_objective_value == pytest.approx(-1.0, abs=1e-6)


def test_invalid_inputs():
    arrays = small_lp_arrays()
    row_ptr, col_ind, values, var_lb, var_ub, con_lb, con_ub, obj = arrays
    with pytest.raises(ValueError):
        mlxpdlp.Solver(2, 1, row_ptr, col_ind, values, var_lb, var_ub,
                       con_lb, con_ub, obj[:-1])
    with pytest.raises(ValueError):
        mlxpdlp.Solver(2, 1, row_ptr, col_ind, values, var_lb, var_ub,
                       con_lb, con_ub, obj, device="tpu")
    with pytest.raises(ValueError):
        # row_ptr must have m + 1 entries
        mlxpdlp.Solver(2, 2, row_ptr, col_ind, values, var_lb, var_ub,
                       con_lb, con_ub, obj)
    parameters = mlxpdlp.Parameters()
    parameters.geometric_mean_iterations = -1
    with pytest.raises(ValueError, match="geometric_mean_iterations"):
        solve_small(parameters=parameters)


@pytest.mark.parametrize("tolerance", [0.0, -1e-6, np.nan, np.inf, -np.inf])
def test_invalid_infeasibility_tolerance(tolerance):
    parameters = mlxpdlp.Parameters()
    parameters.termination_criteria.eps_infeasible_relative = tolerance
    with pytest.raises(ValueError, match="eps_infeasible_relative"):
        solve_small(parameters=parameters)


ADLITTLE = os.path.join(
    os.path.dirname(__file__), "..", "..", "benchmarks", "data", "netlib",
    "adlittle.mps.gz")


@pytest.mark.skipif(not os.path.exists(ADLITTLE), reason="Netlib data not downloaded")
def test_adlittle_mps():
    problem = mlxpdlp.load_mps(ADLITTLE)
    assert (problem.num_variables, problem.num_constraints,
            problem.num_nonzeros) == (97, 56, 383)
    assert problem.row_ptr.shape == (57,)
    assert problem.row_ptr[-1] == 383
    parameters = mlxpdlp.Parameters()
    # The assertion below asks for 1e-5 objective accuracy, so request that
    # tolerance explicitly instead of relying on the default 1e-4 solve to
    # overshoot its stopping criteria on a particular scaling trajectory.
    parameters.tolerance = 1e-5
    parameters.verbose = False
    result = mlxpdlp.solve_mps(
        ADLITTLE, device="cpu", parameters=parameters)
    assert result.primal_objective_value == pytest.approx(
        225494.96316, rel=1e-5)
    assert result.termination_reason_name == "OPTIMAL"


@pytest.mark.skipif(not mlxpdlp.has_gpu(), reason="no Metal GPU device")
def test_backend_selection_flags():
    solver = mlxpdlp.Solver(2, 1, *small_lp_arrays(), device="gpu")
    assert solver.expects_sparse_cpu_backend() is False
    assert solver.expects_sparse_metal_backend() is False  # tiny dense LP


def test_mps_load_failure():
    with pytest.raises(RuntimeError, match="MPS"):
        mlxpdlp.load_mps("/nonexistent/model.mps")


def test_result_repr():
    result = solve_small("cpu")
    text = repr(result)
    assert "SolveResult" in text and "OPTIMAL" in text
