"""Presolve-terminal CPU results must bypass checks of unused backend arrays."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import gzip


with tempfile.TemporaryDirectory(prefix="mlxpdlp-presolve-") as directory:
    root = Path(directory)
    for name, rhs in (("optimal", 1), ("infeasible", -1)):
        with gzip.open(root / f"{name}.mps.gz", "wt") as stream:
            stream.write(f"NAME {name}\nROWS\n N OBJ\n E R\nCOLUMNS\n X OBJ 1\n X R 1\nRHS\n RHS1 R {rhs}\nENDATA\n")
    (root / "manifest.tsv").write_text(
        "name\trows\tcolumns\tnonzeros\tformat\turl\n"
        "optimal\t1\t1\t1\tmps\tlocal\ninfeasible\t1\t1\t1\tmps\tlocal\n"
    )
    subprocess.run([
        sys.argv[1], "--data", str(root), "--device", "cpu", "--jobs", "1",
        "--output-prefix", str(root / "results"), "--no-primal-propagation",
        "--time-limit", "5", "--no-propagation-retry", "--no-presolve-retry",
        "--no-warm-start-correction", "--no-scaling-retry", "--no-restart-policy-retry",
    ], check=True, timeout=30)
    rows = {row["name"]: row for row in json.loads((root / "results.json").read_text())["results"]}
    for name, status in (("optimal", "OPTIMAL"), ("infeasible", "PRIMAL_INFEASIBLE")):
        assert not rows[name]["error"], rows[name]
        assert rows[name]["termination"] == status, rows[name]
        assert rows[name]["iterations"] == 0, rows[name]
    assert rows["optimal"]["verified"], rows["optimal"]
