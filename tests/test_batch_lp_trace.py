#!/usr/bin/env python3
# Copyright 2026 Ethan Wang <ethanshurui.wang@gmail.com>
# SPDX-License-Identifier: Apache-2.0
"""Stage 0 gate tests. All observations here are deliberately synthetic."""

from copy import deepcopy
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))
from batch_lp_trace import MatrixRegistry, SCHEMA, TraceError, analyze, matrix_bytes, read_trace, validate_trace


def fixture(width=4):
    trace = {
        "schema": SCHEMA,
        "source": {"kind": "synthetic", "application": "gate unit test", "workload_id": "unit",
                   "representative": False, "selection": "handwritten times, not an application measurement",
                   "scheduling_unchanged": True, "complete_capture": True, "settings_complete": True,
                   "revision": "test", "solver_revision": "test", "mlx_revision": "test",
                   "hardware": "test", "build": "test", "command": "python3 tests/test_batch_lp_trace.py"},
        "application": {"wall_time_ns": max(100, width * 10 + 1), "latency_limit_ns": 1000,
                        "memory_limit_bytes": 1024 * 1024, "scheduling_policy": "serial FIFO"},
        "matrices": {"a": {"num_variables": 2, "num_constraints": 1,
                           "row_ptr": [0, 3], "col_indices": [0, 0, 1], "values": [0.5, 0.5, 1.0]}},
        "requests": [],
    }
    for i in range(width):
        trace["requests"].append({
            "id": str(i), "family": "objectives", "matrix_id": "a", "device": "cpu",
            "parameters": {"presolve": False, "termination_criteria": {
                "eps_optimal_relative": 1e-4, "eps_feasible_relative": 1e-4,
                "eps_infeasible_relative": 1e-8, "iteration_limit": 100, "time_sec_limit": 1.0}},
            "ready_ns": 0, "start_ns": i * 10, "end_ns": (i + 1) * 10, "depends_on": [],
            "construct_ns": 2, "solve_ns": 8, "rescale_ns": 1, "iterations": i + 1, "termination": "OPTIMAL",
            "data": {"objective": [1.0 + i, 2.0], "objective_constant": float(i),
                     "variable_lower_bounds": [0.0, 0.0], "variable_upper_bounds": ["+inf", "+inf"],
                     "constraint_lower_bounds": [1.0], "constraint_upper_bounds": ["+inf"],
                     "primal_start": None, "dual_start": None, "reduced_cost_start": None},
        })
    return trace


class MatrixIdentityTests(unittest.TestCase):
    def test_ownership_and_one_ulp_difference(self):
        registry = MatrixRegistry()
        matrix = fixture()["matrices"]["a"]
        identity = registry.add(matrix)
        matrix["values"][0] = math.nextafter(0.5, 1)
        self.assertNotEqual(identity, registry.add(matrix))
        self.assertEqual(registry.matrices[identity]["values"][0], 0.5)

    def test_hash_collision_requires_exact_comparison(self):
        class Collision:
            def hexdigest(self):
                return "forced-collision"
        registry = MatrixRegistry()
        matrix = fixture()["matrices"]["a"]
        with patch("batch_lp_trace.hashlib.sha256", return_value=Collision()):
            original = registry.add(matrix)
            self.assertEqual(original, registry.add(deepcopy(matrix)))
            matrix["values"][0] = 2.0
            self.assertNotEqual(original, registry.add(matrix))
        self.assertEqual(len(registry.matrices), 2)

    def test_duplicate_coordinates_and_order_are_preserved(self):
        original = fixture()["matrices"]["a"]
        coalesced = {**original, "row_ptr": [0, 2], "col_indices": [0, 1], "values": [1.0, 1.0]}
        self.assertNotEqual(matrix_bytes(original), matrix_bytes(coalesced))
        first = {**original, "values": [1e20, -1e20, 1.0]}
        second = {**first, "values": [-1e20, 1e20, 1.0]}
        self.assertNotEqual(matrix_bytes(first), matrix_bytes(second))

    def test_aliases_are_verified_from_data(self):
        trace = fixture()
        trace["matrices"]["alias"] = deepcopy(trace["matrices"]["a"])
        trace["requests"][2]["matrix_id"] = "alias"
        report = analyze(trace)
        self.assertEqual(report["distinct_exact_matrices"], 1)
        self.assertEqual(report["ready_groups"][0]["width"], 4)
        self.assertEqual(report["matrices"][0]["reuse_count"], 3)

    def test_invalid_csr_never_reaches_grouping(self):
        cases = ({"row_ptr": [1, 3]}, {"row_ptr": [0, 2]}, {"col_indices": [0, 0, 2]},
                 {"col_indices": [0, 0, 0.0]}, {"values": [math.nan, 1, 1]},
                 {"values": [math.inf, 1, 1]}, {"num_variables": 2**31},
                 {"row_ptr": [0, True]}, {"values": [1e1000, 1, 1]})
        for change in cases:
            with self.subTest(change=change), self.assertRaises(TraceError):
                matrix_bytes({**fixture()["matrices"]["a"], **change})

    def test_empty_rows_zero_nnz_and_index_boundaries(self):
        for n in (0, 1, 65536, 65537):
            matrix_bytes({"num_variables": n, "num_constraints": 0, "row_ptr": [0],
                          "col_indices": [], "values": []})
            matrix_bytes({"num_variables": n, "num_constraints": 2, "row_ptr": [0, 0, 0],
                          "col_indices": [], "values": []})
            if n:
                matrix_bytes({"num_variables": n, "num_constraints": 2, "row_ptr": [0, 0, 1],
                              "col_indices": [n - 1], "values": [1.0]})


class QualificationTests(unittest.TestCase):
    def test_explicit_derived_policy_and_unlimited_caller_limits(self):
        trace = fixture()
        trace["application"].update(latency_limit_ns="unlimited", memory_limit_bytes="unlimited")
        self.assertEqual(analyze(trace)["decision"], "defer")
        self.assertEqual(analyze(trace, allow_synthetic=True)["decision"], "defer")
        trace["source"]["timings_measured"] = True
        report = analyze(trace, allow_synthetic=True)
        self.assertEqual(report["decision"], "go")
        self.assertEqual(report["qualification_policy"], "measured-derived-families-allowed")
        trace["requests"][1]["ready_ns"] = 10
        self.assertEqual(analyze(trace, allow_synthetic=True)["decision"], "defer")

    def test_synthetic_only_cannot_pass(self):
        report = analyze(fixture())
        self.assertTrue(report["families"][0]["numerical_workload_gate_passed"])
        self.assertEqual(report["decision"], "defer")
        self.assertFalse(report["release_gate_evaluated"])

    def test_gate_thresholds_and_widths(self):
        for width in (0, 1, 2, 3, 4, 5, 8, 9, 16, 17):
            trace = fixture(width)
            # Simulate metadata solely to exercise the decision logic. This
            # test result is never shipped as a representative trace/report.
            trace["source"].update(kind="application", representative=True)
            with self.subTest(width=width):
                report = analyze(trace)
                self.assertEqual(report["decision"], "go" if width >= 4 else "defer")
                self.assertEqual(sum(g["width"] for g in report["ready_groups"]), width)

    def test_future_and_dependent_work_is_not_ready_early(self):
        trace = fixture()
        for i, req in enumerate(trace["requests"]):
            req["ready_ns"] = req["start_ns"]
            req["depends_on"] = [] if i == 0 else [str(i - 1)]
        report = analyze(trace)
        self.assertEqual([g["width"] for g in report["ready_groups"]], [1] * 4)
        self.assertEqual(report["families"][0]["compatible_lp_wall_fraction"], 0)
        trace["requests"][1]["ready_ns"] = 0
        with self.assertRaisesRegex(TraceError, "dependency 0"):
            analyze(trace)

    def test_unknown_dependencies_or_readiness_do_not_qualify(self):
        for name in ("ready_ns", "depends_on"):
            trace = fixture()
            for req in trace["requests"]:
                req[name] = None
            report = analyze(trace)
            self.assertEqual([g["width"] for g in report["ready_groups"]], [1] * 4)
            self.assertTrue(any(name in r for r in report["reasons"]))

    def test_latency_limit_does_not_allow_arbitrary_lookahead(self):
        trace = fixture()
        trace["application"]["latency_limit_ns"] = 5
        for req in trace["requests"]:
            req["start_ns"] += 5
            req["end_ns"] += 5
        report = analyze(trace)
        self.assertEqual([g["width"] for g in report["ready_groups"]], [1] * 4)

    def test_incompatible_parameters_and_matrices_split_groups(self):
        trace = fixture()
        trace["requests"][0]["parameters"]["presolve"] = True
        trace["matrices"]["changed"] = deepcopy(trace["matrices"]["a"])
        trace["matrices"]["changed"]["values"][0] = math.nextafter(0.5, 1)
        trace["requests"][1]["matrix_id"] = "changed"
        self.assertEqual([g["width"] for g in analyze(trace)["ready_groups"]], [1, 1, 2])

    def test_wall_time_is_not_summed_across_concurrent_lps(self):
        trace = fixture()
        for req in trace["requests"]:
            req.update(start_ns=0, end_ns=10)
        report = analyze(trace)
        self.assertEqual(report["lp_wall_union_ns"], 10)
        self.assertEqual(report["summed_lp_execution_ns"], 40)
        self.assertEqual(report["peak_concurrent_requests"], 4)
        self.assertEqual(report["lp_fraction_of_application_wall"], 0.1)
        self.assertEqual(report["families"][0]["compatible_lp_wall_fraction"], 1.0)

    def test_wall_fraction_threshold_includes_unrelated_lp_time(self):
        for last_end, passed in ((160, True), (161, False)):
            trace = fixture()
            other = deepcopy(trace["requests"][0])
            other.update(id="other", family="other", ready_ns=40, start_ns=40, end_ns=last_end)
            trace["requests"].append(other)
            trace["application"]["wall_time_ns"] = last_end
            report = analyze(trace)
            family = next(f for f in report["families"] if f["family"] == "objectives")
            self.assertEqual(family["numerical_workload_gate_passed"], passed)

    def test_missing_metadata_cannot_pass(self):
        for key in ("latency_limit_ns", "memory_limit_bytes"):
            trace = fixture()
            trace["application"][key] = None
            self.assertTrue(any(key in r for r in analyze(trace)["reasons"]))
        for key in ("construct_ns", "solve_ns", "rescale_ns", "iterations"):
            trace = fixture()
            trace["requests"][0][key] = None
            self.assertTrue(any(key in r for r in analyze(trace)["reasons"]))

    def test_reordering_trace_records_does_not_change_ready_sizes(self):
        trace = fixture()
        expected = analyze(trace)
        trace["requests"].reverse()
        actual = analyze(trace)
        self.assertEqual(actual["families"], expected["families"])
        indices = actual["ready_groups"][0]["input_indices"]
        self.assertEqual([trace["requests"][i]["id"] for i in indices], ["0", "1", "2", "3"])

    def test_invalid_inputs_are_attributed_to_member_and_field(self):
        trace = fixture()
        trace["requests"][2]["data"]["variable_lower_bounds"] = ["+inf", 0]
        with self.assertRaisesRegex(TraceError, r"requests\[2\].*variable_lower_bounds\[0\]"):
            analyze(trace)
        trace = fixture()
        trace["requests"][2]["data"]["objective"] = [1.0]
        with self.assertRaisesRegex(TraceError, "objective"):
            analyze(trace)

    def test_dependency_cycles_are_invalid_even_at_equal_timestamps(self):
        trace = fixture(2)
        for i, req in enumerate(trace["requests"]):
            req.update(start_ns=0, end_ns=0, construct_ns=0, solve_ns=0, rescale_ns=0,
                       depends_on=[str(1 - i)])
        with self.assertRaisesRegex(TraceError, "cycle"):
            analyze(trace)

    def test_timing_inconsistency_is_invalid(self):
        for changes in ({"ready_ns": 11}, {"rescale_ns": 9}, {"construct_ns": 3}, {"end_ns": 101}):
            trace = fixture()
            trace["requests"][0].update(changes)
            with self.subTest(changes=changes), self.assertRaises(TraceError):
                validate_trace(trace)

    def test_nonstandard_numbers_and_duplicate_json_keys_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.json"
            for payload in ('{"x": NaN}', '{"x": 1, "x": 2}'):
                path.write_text(payload)
                with self.assertRaises(TraceError):
                    read_trace(path)
            path.write_text(json.dumps(fixture()))
            self.assertEqual(len(read_trace(path)["requests"]), 4)


if __name__ == "__main__":
    unittest.main()
