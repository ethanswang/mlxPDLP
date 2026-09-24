# Optional CPU reference solvers

OR-Tools PDLP and HiGHS are benchmark tools, with no dependency from the
mlxPDLP library, Python package, CMake build, or default tests. Installation is
explicit or happens when `make run` is invoked in a reference directory.

```text
reference/
  common.py            manifest selection, original-model audit, JSON/CSV reports
  compare.py           standard-library-only comparison with native reports
  environment.mk       on-demand environment setup
  ortools_pdlp/        adapter, pinned requirements, private .venv and results
  highs/               adapter, pinned requirements, private .venv and results
  tests/               optional protocol and adapter checks
```

Delete `ortools_pdlp/` or `highs/` to remove that adapter, its environment, pip
cache, and default results. The other adapter and all existing mlxPDLP
functionality remain usable. The entire `reference/` directory is removable.
Nothing registers these solvers in the core build or loads them automatically.

Each adapter runs in its own Python process and environment. Do not import
`highspy` and OR-Tools together: their bundled HiGHS libraries can conflict on
macOS. The shared utilities import neither solver. No solver sources are vendored.

## Install and run

Use native ARM64 Python 3.12–3.14 on Apple Silicon (Python 3.13 is tested).
Set `PYTHON` to the interpreter available on your machine. From the repository root:

```sh
make -C benchmarks/reference/ortools_pdlp setup PYTHON=python3.12
make -C benchmarks/reference/highs setup PYTHON=python3.12

benchmarks/reference/ortools_pdlp/.venv/bin/python benchmarks/reference/ortools_pdlp/run.py \
  --data-dir benchmarks/data/netlib --instance afiro --instance adlittle \
  --threads 1 --tolerance 1e-6 --solver-tolerance 1e-8 --fail-on-validation \
  --output-prefix benchmarks/reference/ortools_pdlp/results/netlib

benchmarks/reference/highs/.venv/bin/python benchmarks/reference/highs/run.py \
  --data-dir benchmarks/data/netlib --instance afiro --instance adlittle \
  --solver hipdlp --threads 1 --tolerance 1e-6 --solver-tolerance 1e-8 --fail-on-validation \
  --output-prefix benchmarks/reference/highs/results/netlib
```

The existing download scripts supply the datasets; these adapters never fetch
MPS files. Without positional files the runners use the existing TSV manifests.
Repeat `--instance` to select cases, or omit it for the whole manifest. A custom
`.mps` or `.mps.gz` file can be passed positionally. Relative paths are relative
to the caller's working directory; `make -C` changes that directory.

For an on-demand LPfeas run, this command installs the selected reference if
needed, uses the repository's LPfeas data, and saves results inside its directory:

```sh
make -C benchmarks/reference/ortools_pdlp run PYTHON=python3.12 ARGS="--instance qap15 --threads 1"
make -C benchmarks/reference/highs run PYTHON=python3.12 ARGS="--instance qap15 --solver hipdlp --threads 1"
```

Plain `make` and `python3 run.py --help` do not install anything. Solver package
versions and numerical dependencies are pinned per adapter; each JSON report
also records all installed versions, Python/host information, adapter digests,
the input SHA-256, and explicit native settings. To change Python versions,
recreate only the selected adapter's `.venv`.

## Comparison protocol

Defaults are CPU float64, one solver thread, one instance at a time, presolve
off, a 1,000-second native solve limit, and audit/stopping tolerances of `1e-6`.
Use `--presolve on` for an end-to-end presolve comparison. `--solver-tolerance`
sets the solver's stopping target independently of the common audit tolerance.
The implementations use different native stopping rules, so equal parameter
values do not alone establish equal accuracy. The Netlib examples request a
tighter native target to satisfy the independent `1e-6` audit; tighter targets
are still not a guarantee of passing on every instance. Integer annotations are relaxed;
quadratic objectives are rejected. This is an LP benchmark, including for MIPLIB
files. Solver-provided infeasibility rays are not treated as optimal solutions.

The independent NumPy/SciPy float64 audit implements the equations in
`benchmarks/lpfeas_support.cpp`: L2 primal and stationarity residuals, variable
and dual-bound violations, and gap divided by `1 + |primal| + |dual|`. Bound
normalization includes both finite endpoints. Floating-point summation order
can differ from the C++ audit. The audit uses the original model as parsed by
each reference, before scaling or presolve. Compare dimensions and input hashes
and investigate parser discrepancies. Both adapters normalize maximization to
minimization for solving, then restore the original objective for reporting.
Netlib's optional `reference_objectives.tsv` adds an objective check
excluding the MPS constant, matching the native benchmark's convention.

JSON preserves `termination` separately from `verified`; a solver's `OPTIMAL`
does not override an audit failure. CSV carries the same metrics in flat form.
Reports are updated after every instance. Missing/unreadable inputs produce
error rows and a nonzero exit; subsequent cases still run. `--fail-on-validation`
also returns nonzero for an unverified result. `--save-solutions` writes NPZ
arrays beside the report; dual/reduced-cost vectors use minimization signs,
with `objective_sign` recording the conversion to the original sense.

Timing separates parse/model conversion, setup, native solve/result extraction,
and audit. Total wall time includes all four, but excludes imports, provenance
hashing, and output writes. Hashing happens after the timed work to avoid
prewarming the input file. Native presolve/postsolve are inside solve time for
these references; mlxPDLP also does work in setup. Use **total** time for the
default comparison. Native limits cover the optimizer, not MPS parsing or audit.
There are no adapter retries, external polishing, or warm starts.

For a matching native report, use the same data, audit tolerance, declared
presolve policy, and `--jobs 1`. For example:

```sh
build/mlxpdlp_lpfeas_benchmark --device cpu --jobs 1 \
  --data benchmarks/data/netlib --instance afiro --instance adlittle \
  --tolerance 1e-6 --solver-tolerance 1e-8 --no-presolve \
  --output-prefix benchmarks/results/netlib-cpu

python3 benchmarks/reference/compare.py benchmarks/results/netlib-cpu.json \
  benchmarks/reference/ortools_pdlp/results/netlib.json
python3 benchmarks/reference/compare.py benchmarks/results/netlib-cpu.json \
  benchmarks/reference/highs/results/netlib.json
```

The comparator checks audit tolerance, host, serial scheduling, dimensions, and
input hashes when available. It reports speed ratios only for verified pairs;
failures/timeouts remain visible. Native schema 9 does not contain input hashes,
so confirm that both runs used identical files. It does not claim an aggregate
speedup from the subset both solvers solve. Use distinct output prefixes for
repeat runs, run solvers sequentially, and record thread counts. mlxPDLP's
polishing/retry policy remains part of its total time and is recorded in its
report; consult the native benchmark options when isolating algorithm choices.

## Optional validation

After setting up both adapters:

```sh
benchmarks/reference/ortools_pdlp/.venv/bin/python -m unittest discover \
  -s benchmarks/reference/tests -v
```

The tests use analytic certificates and temporary MPS models, check independent
audits and failure reporting, and exercise each adapter with its sibling absent.
Adapter checks skip if that adapter or its environment has been removed. These
tests are intentionally absent from the core CTest/Python test dependencies.
