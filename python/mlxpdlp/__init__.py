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

"""mlxPDLP: a PDHG linear-programming solver on Apple MLX devices.

The solver minimizes ``c^T x + constant`` subject to
``constraint_lb <= A x <= constraint_ub`` and ``variable_lb <= x <=
variable_ub``. CPU execution is float64 throughout and serves as the
higher-accuracy fallback; Metal execution is float32 with optional
host-float64 polishing.

Typical usage::

    import mlxpdlp
    import numpy as np

    solver = mlxpdlp.Solver(
        num_variables=2,
        num_constraints=1,
        row_ptr=np.array([0, 2], dtype=np.int32),
        col_indices=np.array([0, 1], dtype=np.int32),
        values=np.array([1.0, 1.0]),
        variable_lower_bounds=np.zeros(2),
        variable_upper_bounds=np.full(2, np.inf),
        constraint_lower_bounds=np.array([1.0]),
        constraint_upper_bounds=np.array([np.inf]),
        objective=np.array([-1.0, -1.0]),
        device="gpu" if mlxpdlp.has_gpu() else "cpu",
    )
    result = solver.solve()
    print(result.primal_solution, result.primal_objective_value)
"""

from . import _core
from ._core import (
    MpsProblem,
    NormType,
    Parameters,
    RestartParameters,
    Solver,
    SolveResult,
    TerminationCriteria,
    TerminationReason,
    has_gpu,
    load_mps,
    version,
)

__version__ = _core.__version__

__all__ = [
    "MpsProblem",
    "NormType",
    "Parameters",
    "RestartParameters",
    "Solver",
    "SolveResult",
    "TerminationCriteria",
    "TerminationReason",
    "has_gpu",
    "load_mps",
    "solve_mps",
    "version",
    "__version__",
]


def solve_mps(path, *, device="cpu", parameters=None):
    """Load an MPS file and solve it on the requested device.

    Maximization models are converted to the solver's minimization form
    internally (the objective is negated), and the returned objective
    values are sign-corrected back to the model's original convention.

    Args:
        path: Path to a plain or gzip-compressed MPS file.
        device: ``"cpu"`` (float64) or ``"gpu"``/``"metal"`` (float32).
        parameters: Optional :class:`Parameters`; defaults are used when
            omitted.

    Returns:
        :class:`SolveResult` with objective values in the model's
        original (possibly maximize) convention.
    """
    problem = load_mps(str(path))
    objective = -problem.objective if problem.maximize else problem.objective
    objective_constant = (
        -problem.objective_constant
        if problem.maximize
        else problem.objective_constant
    )
    solver = Solver(
        num_variables=problem.num_variables,
        num_constraints=problem.num_constraints,
        row_ptr=problem.row_ptr,
        col_indices=problem.col_ind,
        values=problem.values,
        variable_lower_bounds=problem.variable_lb,
        variable_upper_bounds=problem.variable_ub,
        constraint_lower_bounds=problem.constraint_lb,
        constraint_upper_bounds=problem.constraint_ub,
        objective=objective,
        objective_constant=objective_constant,
        parameters=parameters,
        device=device,
    )
    result = solver.solve()
    if problem.maximize:
        result.primal_objective_value = -result.primal_objective_value
        result.dual_objective_value = -result.dual_objective_value
    return result
