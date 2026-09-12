# Copyright 2026 Ethan Wang <ethanshurui.wang@gmail.com>
# SPDX-License-Identifier: Apache-2.0
"""Trace/capture support tests, not tests of the proposed public batch API."""

from copy import deepcopy
from pathlib import Path
import sys

import numpy as np
import pytest

import mlxpdlp

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "benchmarks"))
from batch_lp_capture import TraceRecorder, replay, restore_parameters, snapshot_parameters
from batch_lp_trace import TraceError, analyze, read_trace


def source():
    return {"kind": "synthetic", "application": "analytic capture test", "workload_id": "two_variables",
            "representative": False, "selection": "unit test only", "scheduling_unchanged": True,
            "complete_capture": True, "settings_complete": True, "revision": "test",
            "solver_revision": "test", "mlx_revision": "test", "hardware": "test",
            "build": "test", "command": "pytest python/tests/test_batch_trace_replay.py"}


def arguments():
    params = mlxpdlp.Parameters()
    params.verbose = False
    params.presolve = False
    params.tolerance = 1e-6
    params.time_limit_seconds = 10
    params.iteration_limit = 2000
    return dict(num_variables=2, num_constraints=1, row_ptr=np.array([0, 3], np.int32),
                col_indices=np.array([0, 0, 1], np.int32), values=np.array([0.5, 0.5, 1.0]),
                objective=np.array([1.0, 2.0]), objective_constant=3.0,
                variable_lower_bounds=np.zeros(2), variable_upper_bounds=np.ones(2),
                constraint_lower_bounds=np.ones(1), constraint_upper_bounds=np.ones(1),
                parameters=params, device="cpu")


def recorder():
    return TraceRecorder(source=source(), latency_limit_ns=10**10, memory_limit_bytes=2**28,
                         scheduling_policy="existing serial input order")


def test_parameters_round_trip_including_nested_settings():
    params = arguments()["parameters"]
    params.restart_params.k_p = 0.25
    params.optimality_norm = mlxpdlp.NormType.L_INF
    params.termination_criteria.time_sec_limit = float("inf")
    snapshot = snapshot_parameters(params)
    restored = restore_parameters(snapshot, mlxpdlp.Parameters())
    assert snapshot_parameters(restored) == snapshot
    params.restart_params.k_p = 0.8
    assert restored.restart_params.k_p == 0.25


def test_capture_owns_inputs_and_replay_keeps_order_and_offsets():
    capture = recorder()
    args = arguments()
    when = capture.now_ns()
    capture.ready("min", "bounds", args, ready_ns=when, depends_on=[])
    args["objective"][:] = [2.0, 1.0]
    args["objective_constant"] = -2.0
    capture.ready("other", "bounds", args, ready_ns=when, depends_on=[])
    # Mutating the caller's arrays/parameters must not change recorded LPs.
    args["objective"][:] = 100.0
    args["values"][:] = 10.0
    args["parameters"].iteration_limit = 1
    # Dispatch order can differ from input order; record both independently.
    second = capture.solve("other")
    first = capture.solve("min")
    trace = capture.finish()
    assert len(trace["matrices"]) == 1
    assert first.primal_objective_value == pytest.approx(4.0, abs=1e-5)
    assert second.primal_objective_value == pytest.approx(-1.0, abs=1e-5)
    repeated = replay(trace)
    assert [r["id"] for r in repeated["results"]] == ["min", "other"]
    assert [r["result"]["primal_objective_value"] for r in repeated["results"]] == pytest.approx([4, -1], abs=1e-5)
    assert all(r["execution_mode"] == "independent" for r in repeated["results"])
    assert not repeated["qualification_evidence"]
    assert analyze(trace)["decision"] == "defer"


def test_warm_start_snapshot_in_original_coordinates():
    capture = recorder()
    args = arguments()
    args.update(primal_start=np.array([1.0, 0.0]), dual_start=np.array([1.0]),
                reduced_cost_start=np.array([0.0, 1.0]))
    capture.ready("warm", "bounds", args, ready_ns=capture.now_ns(), depends_on=[])
    result = capture.solve("warm")
    assert result.primal_objective_value == pytest.approx(4.0, abs=1e-5)
    trace = capture.finish()
    assert trace["requests"][0]["data"]["reduced_cost_start"] == [0.0, 1.0]
    assert replay(trace)["results"][0]["result"]["primal_objective_value"] == pytest.approx(4, abs=1e-5)


def test_invalid_later_request_rejects_before_replay_launch(monkeypatch):
    capture = recorder()
    args = arguments()
    for rid in ("0", "1"):
        capture.ready(rid, "bounds", args, ready_ns=capture.now_ns(), depends_on=[])
        capture.solve(rid)
    trace = capture.finish()

    def should_not_construct(**kwargs):
        pytest.fail("replay launched a solver before validating the complete trace")

    monkeypatch.setattr(mlxpdlp, "Solver", should_not_construct)
    invalid = deepcopy(trace)
    invalid["requests"][1]["data"]["objective"] = [1.0]
    with pytest.raises(TraceError, match="objective"):
        replay(invalid)
    invalid = deepcopy(trace)
    invalid["requests"][1]["parameters"]["unrecognized_setting"] = True
    with pytest.raises(TraceError, match="binding settings differ"):
        replay(invalid)


def test_incomplete_capture_is_explicit():
    capture = recorder()
    capture.ready("queued", "bounds", arguments(), ready_ns=capture.now_ns(), depends_on=[])
    with pytest.raises(TraceError, match="unfinished"):
        capture.finish()


@pytest.mark.parametrize("device", ["cpu", "gpu"])
def test_shipped_fixture_replays_to_analytic_objectives(device):
    if device == "gpu" and not mlxpdlp.has_gpu():
        pytest.skip("Metal unavailable")
    path = Path(__file__).resolve().parents[2] / "tests/data/batch_lp/ready_and_dependent.json"
    trace = read_trace(path)
    report = replay(trace, device=device)
    assert not report["qualification_evidence"]
    for item in report["results"]:
        req = trace["requests"][item["input_index"]]
        expected = min(req["data"]["objective"]) + req["data"]["objective_constant"]
        result = item["result"]
        assert result["termination_reason_name"] == "OPTIMAL"
        assert result["primal_objective_value"] == pytest.approx(expected, abs=1e-5)
        assert sum(result["primal_solution"]) == pytest.approx(1, abs=1e-5)
