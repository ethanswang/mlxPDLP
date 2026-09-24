# SPDX-License-Identifier: Apache-2.0
"""Optional checks; never registered in mlxPDLP's default test/build targets."""
import copy
import gzip
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

REFERENCE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REFERENCE))
from common import Problem, Solution, audit, RESIDUALS
from compare import comparisons


class AuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import numpy as np
            from scipy import sparse
        except ImportError:
            raise unittest.SkipTest("run with an optional reference environment")
        cls.np, cls.sparse = np, sparse

    def fixture(self):
        np = self.np
        vector = lambda x: np.array(x, dtype=float)
        p = Problem(self.sparse.csr_matrix([[1.0]]), vector([1]), vector([0]),
                    vector([np.inf]), vector([1]), vector([np.inf]))
        s = Solution(vector([1]), vector([1]), vector([0]), "OPTIMAL", 10, {})
        return p, s

    def test_exact_certificate_and_cpp_normalization(self):
        p, s = self.fixture()
        self.assertTrue(audit(p, s, 1e-10)["verified"])
        s.primal[0] = 0
        result = audit(p, s, 1e-6)
        self.assertFalse(result["verified"])
        self.assertEqual(result["relative_primal_residual"], 0.5)
        self.assertEqual(result["relative_objective_gap"], 0.5)

    def test_free_bound_dual_sign_and_stationarity(self):
        p, s = self.fixture()
        p.constraint_lower[0] = -self.np.inf
        p.constraint_upper[0] = 1
        # c-A'y-z=0 and zero gap alone must not certify this wrong-signed dual.
        s.reduced_cost[0] = 0
        result = audit(p, s, 1e-6)
        self.assertEqual(result["relative_dual_residual"], 0)
        self.assertEqual(result["relative_dual_bound_violation"], 0.5)
        self.assertFalse(result["verified"])

    def test_nonfinite_wrong_dimensions_and_rays_are_not_verified(self):
        for damage in ("nan", "shape", "ray"):
            p, s = self.fixture()
            if damage == "nan":
                s.primal[0] = self.np.nan
            elif damage == "shape":
                s.dual = self.np.array([])
            else:
                s.has_solution = False
            self.assertFalse(audit(p, s, 1e-6)["verified"])

    def test_maximization_offset_and_netlib_reference(self):
        p, s = self.fixture()
        p.objective_sign = -1
        p.offset = -7
        result = audit(p, s, 1e-8, reference_objective=-1)
        self.assertTrue(result["verified"])
        self.assertEqual(result["primal_objective"], 6)
        self.assertEqual(result["objective_without_constant"], -1)
        self.assertFalse(audit(p, s, 1e-8, reference_objective=6)["verified"])


class ComparisonTests(unittest.TestCase):
    def test_ratio_excludes_failure_mismatch_and_forged_verified_flag(self):
        good = {"name": "case", "verified": True, "termination": "OPTIMAL",
                "dimensions": {"rows": 1, "columns": 1, "nonzeros": 1},
                "timing_seconds": {"total": 2.0}, "input_sha256": "abc",
                "original_model": {key: 0.0 for key in RESIDUALS}}
        self.assertEqual(comparisons({"case": good}, {"case": good}, 1e-6, "total")[0]["reference_over_baseline"], 1)
        for damage in ("error", "hash", "residual", "time"):
            bad = copy.deepcopy(good)
            if damage == "error":
                bad = {"name": "case", "termination": "ERROR", "verified": False}
            elif damage == "hash":
                bad["input_sha256"] = "different"
            elif damage == "residual":
                bad["original_model"]["relative_primal_residual"] = 0.1
            else:
                bad["timing_seconds"]["total"] = None
            row = comparisons({"case": good}, {"case": bad}, 1e-6, "total")[0]
            self.assertIsNone(row["reference_over_baseline"])
            self.assertTrue(row["reason"])


# Maximization, objective constant, a ranged row, free x, and integer-marked y.
# LP relaxation: x=3, y=0.5, original objective=17; the integer model is infeasible.
MAX_MPS = """NAME          REFERENCE
OBJSENSE
 MAX
ROWS
 N  COST
 L  CAP
 E  FIX
 G  RANGE
COLUMNS
    X         COST      3            CAP       1
    X         RANGE     1
    MARK0     'MARKER'                 'INTORG'
    Y         COST      2            CAP       1
    Y         FIX       1
    MARK1     'MARKER'                 'INTEND'
RHS
    RHS1      COST      -7           CAP       4
    RHS1      FIX       0.5          RANGE     0
RANGES
    RNG1      RANGE     3
BOUNDS
 FR BND1      X
 LO BND1      Y         0
 UP BND1      Y         2
ENDATA
"""


class AdapterTests(unittest.TestCase):
    def exercise(self, name, solver=None):
        adapter = REFERENCE / name
        python = adapter / ".venv/bin/python"
        if not python.exists():
            self.skipTest(f"{name} is not installed")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            # The copied adapter has no sibling solver directory or installation.
            shared = root / "reference"
            shared.mkdir()
            shutil.copy(REFERENCE / "common.py", shared / "common.py")
            isolated = shared / name
            isolated.mkdir()
            for file in ("run.py", "requirements.txt"):
                shutil.copy(adapter / file, isolated / file)
            model = root / "max.mps.gz"
            with gzip.open(model, "wt") as stream:
                stream.write(MAX_MPS)
            output = root / "report"
            command = [str(python), str(isolated / "run.py"), str(model),
                       "--output-prefix", str(output), "--time-limit", "10", "--tolerance", "1e-6",
                       "--solver-tolerance", "1e-8", "--save-solutions", "--fail-on-validation"]
            if solver:
                command += ["--solver", solver]
            completed = subprocess.run(command, capture_output=True, text=True, timeout=60)
            self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
            report = json.loads(output.with_suffix(".json").read_text())
            row = report["results"][0]
            self.assertTrue(row["verified"], row)
            self.assertGreater(row["iterations"], 0)
            self.assertAlmostEqual(row["original_model"]["primal_objective"], 17, delta=1e-5)
            self.assertAlmostEqual(row["original_model"]["dual_objective"], 17, delta=1e-5)
            self.assertTrue(Path(row["solution_file"]).exists())
            self.assertEqual(report["protocol"]["integer_variables"], "relaxed")
            self.assertEqual(report["protocol"]["jobs"], 1)
            # Input errors must be recorded without preventing subsequent solves.
            malformed = root / "malformed.mps"
            malformed.write_text("this is not an MPS model\n")
            command.insert(2, str(malformed))
            command.insert(2, str(root / "missing.mps"))
            completed = subprocess.run(command, capture_output=True, text=True, timeout=60)
            self.assertEqual(completed.returncode, 1, completed.stdout + completed.stderr)
            rows = json.loads(output.with_suffix(".json").read_text())["results"]
            for row in rows[:2]:
                self.assertEqual(row["termination"], "ERROR")
                self.assertFalse(row["verified"])
            self.assertTrue(rows[2]["verified"])
            # A native budget exit must remain a failure of the common audit.
            del command[2:4]
            command += ["--iteration-limit", "1"]
            completed = subprocess.run(command, capture_output=True, text=True, timeout=60)
            self.assertEqual(completed.returncode, 1, completed.stdout + completed.stderr)
            row = json.loads(output.with_suffix(".json").read_text())["results"][0]
            self.assertNotEqual(row["termination"], "OPTIMAL")
            self.assertFalse(row["verified"])
            self.assertFalse(row["error"])

    def test_ortools_without_highs_directory(self):
        self.exercise("ortools_pdlp")

    def test_hipdlp_without_ortools_directory(self):
        self.exercise("highs", "hipdlp")

    def test_cupdlpc_cpu(self):
        self.exercise("highs", "pdlp")


if __name__ == "__main__":
    unittest.main()
