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

"""Small Netlib regression through the Python binding.

Solves a fixed subset of the Netlib corpus on the CPU float64 backend
(and the Metal float32 backend when a GPU is available) and checks each
solution against the published optimal objective plus the solver's
relative residuals at a 1e-4 tolerance.

Data directory: set MLXPDLP_NETLIB_DATA or download the corpus with
``benchmarks/data/netlib/download.sh`` (the default location next to
this repository is used automatically).

Note: the objective agreement check mirrors the C++ Netlib suite's
published-objective gate; the residual check uses the solver's own
postsolved certificate rather than an independent float64 audit.
"""

import os
from pathlib import Path

import pytest

import mlxpdlp

CASES = ["afiro", "sc50b", "adlittle", "blend", "sc105", "scagr7", "sc50a"]

TOLERANCE = 1e-4
SOLVER_TOLERANCE = 5e-5  # internal stopping target, matching the C++ suite
DATA_DIR = os.environ.get(
    "MLXPDLP_NETLIB_DATA",
    str(Path(__file__).resolve().parents[2] / "benchmarks" / "data" / "netlib"),
)
REFERENCE_FILE = os.path.join(DATA_DIR, "reference_objectives.tsv")


def _reference_objectives():
    objectives = {}
    with open(REFERENCE_FILE) as handle:
        header = handle.readline().rstrip("\n").split("\t")
        assert header == ["name", "optimal_objective"]
        for line in handle:
            name, value = line.rstrip("\n").split("\t")
            objectives[name] = float(value)
    return objectives


_OBJECTIVES = _reference_objectives() if os.path.exists(REFERENCE_FILE) else {}

HAVE_DATA = all(
    os.path.exists(os.path.join(DATA_DIR, f"{name}.mps.gz")) for name in CASES
) and bool(_OBJECTIVES)

pytestmark = pytest.mark.skipif(
    not HAVE_DATA, reason="Netlib data not downloaded (run "
    "benchmarks/data/netlib/download.sh)")

_ACCEPTABLE_TERMINATIONS = {
    int(mlxpdlp.TerminationReason.OPTIMAL),
    int(mlxpdlp.TerminationReason.FEAS_POLISH_SUCCESS),
    int(mlxpdlp.TerminationReason.HOST_DOUBLE_HANDOFF),
    int(mlxpdlp.TerminationReason.ITERATION_LIMIT),
    int(mlxpdlp.TerminationReason.TIME_LIMIT),
}


def _passes(name, device, result):
    reference = _OBJECTIVES[name]
    if result.termination_reason not in _ACCEPTABLE_TERMINATIONS:
        return False
    relative_objective_error = abs(
        result.primal_objective_value - reference
    ) / max(1.0, abs(reference))
    return (
        relative_objective_error <= TOLERANCE
        and result.relative_primal_residual <= TOLERANCE
        and result.relative_dual_residual <= TOLERANCE
        and result.relative_objective_gap <= TOLERANCE
    )


def _solve_with_fallback(name, device):
    """Miniature version of the C++ suite's retry recipe.

    Aggressive PSLP can amplify an approximate dual certificate through
    its inverse map (especially on FP32 Metal), so this helper retries
    without presolve when the safe-PSLP certificate misses the 1e-4
    audit.
    """
    path = os.path.join(DATA_DIR, f"{name}.mps.gz")
    parameters = mlxpdlp.Parameters()
    parameters.tolerance = SOLVER_TOLERANCE
    parameters.time_limit_seconds = 300.0
    parameters.iteration_limit = 200000
    parameters.presolve_primal_propagation = False  # safe PSLP first
    result = mlxpdlp.solve_mps(path, device=device, parameters=parameters)
    if not _passes(name, device, result):
        parameters.presolve = False
        result = mlxpdlp.solve_mps(path, device=device, parameters=parameters)
    return result


def _check_case(name, device):
    reference = _OBJECTIVES[name]
    result = _solve_with_fallback(name, device)
    assert result.termination_reason in _ACCEPTABLE_TERMINATIONS, (
        f"{name}/{device}: unexpected termination "
        f"{result.termination_reason_name}")
    relative_objective_error = abs(
        result.primal_objective_value - reference
    ) / max(1.0, abs(reference))
    assert relative_objective_error <= TOLERANCE, (
        f"{name}/{device}: objective {result.primal_objective_value:.8e} "
        f"vs published {reference:.8e} "
        f"(rel err {relative_objective_error:.2e})")
    assert result.relative_primal_residual <= TOLERANCE, (
        f"{name}/{device}: relative primal residual "
        f"{result.relative_primal_residual:.2e}")
    assert result.relative_dual_residual <= TOLERANCE, (
        f"{name}/{device}: relative dual residual "
        f"{result.relative_dual_residual:.2e}")
    assert result.relative_objective_gap <= TOLERANCE, (
        f"{name}/{device}: relative objective gap "
        f"{result.relative_objective_gap:.2e}")


@pytest.mark.parametrize("name", CASES)
def test_netlib_case_cpu(name):
    _check_case(name, "cpu")


@pytest.mark.skipif(not mlxpdlp.has_gpu(), reason="no Metal GPU device")
@pytest.mark.parametrize("name", CASES)
def test_netlib_case_gpu(name):
    _check_case(name, "gpu")


def test_maximize_sign_convention(tmp_path):
    # maximize x1 subject to 1 <= x1 <= 2  ->  x1 = 2, objective 2.
    # solve_mps must return the objective in the model's original
    # (maximize) convention.
    path = tmp_path / "maxtest.mps"
    path.write_text(
        "NAME          MAXTEST\n"
        "OBJSENSE MAX\n"
        "ROWS\n"
        " N  COST\n"
        " G  R1\n"
        "COLUMNS\n"
        "    X1        COST      1.0   R1        1.0\n"
        "RHS\n"
        "    B         R1        1.0\n"
        "BOUNDS\n"
        " UP BND       X1        2.0\n"
        "ENDATA\n")
    problem = mlxpdlp.load_mps(str(path))
    assert problem.maximize == 1
    result = mlxpdlp.solve_mps(str(path), device="cpu")
    assert result.primal_objective_value == pytest.approx(2.0, abs=1e-6)
    assert result.primal_solution[0] == pytest.approx(2.0, abs=1e-6)
    assert result.dual_objective_value == pytest.approx(2.0, abs=1e-6)
