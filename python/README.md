# mlxpdlp: Python bindings for mlxPDLP

Python bindings for [mlxPDLP](../README.md), the Primal-Dual Hybrid
Gradient (PDHG) linear-programming solver on Apple MLX devices.

- **CPU backend**: float64 arithmetic throughout — the higher-accuracy
  reference/fallback.
- **Metal backend**: float32 CSR SpMV with optional bounded host-float64
  polishing — several times faster than the CPU backend on large sparse
  models (see [benchmarks/README.md](../benchmarks/README.md)).

> Apple Silicon GPUs do not support FP64, so Metal solves run in FP32:
> the supported accuracy is a `1e-4` tolerance with a `5e-5` internal
> target. Use `device="cpu"` when tighter tolerances are required.
- CSR input with NumPy arrays, PSLP presolve, warm starts, MPS loading
  (plain and gzip), and full termination/residual diagnostics.

## Installation

Requirements: macOS, Python >= 3.9, NumPy, a C++20 toolchain, and a
built [MLX](https://github.com/ml-explore/mlx) C++ library (with
`MLX_BUILD_METAL=ON` for the Metal backend).

```sh
pip install . \
  -Ccmake.define.MLX_BUILD_DIR=/absolute/path/to/mlx/build \
  -Ccmake.define.MLXPDLP_ALLOW_DOWNLOADS=ON
# or via an environment variable:
CMAKE_ARGS=-DMLXPDLP_ALLOW_DOWNLOADS=ON \
  MLX_BUILD_DIR=/absolute/path/to/mlx/build pip install .
```

PSLP 0.0.8 is downloaded only when presolve support is enabled (default) and
`MLXPDLP_ALLOW_DOWNLOADS=ON` explicitly permits it. For an offline build, reuse
an existing checkout with `FETCHCONTENT_SOURCE_DIR_PSLP=/path/to/pslp-src`
alongside the MLX path shown above.

## Quick start

```python
import mlxpdlp
import numpy as np

# minimize -x0 - x1  subject to  x0 + x1 <= 1,  x >= 0
solver = mlxpdlp.Solver(
    num_variables=2,
    num_constraints=1,
    row_ptr=np.array([0, 2], dtype=np.int32),
    col_indices=np.array([0, 1], dtype=np.int32),
    values=np.array([1.0, 1.0]),
    variable_lower_bounds=np.zeros(2),
    variable_upper_bounds=np.full(2, np.inf),
    constraint_lower_bounds=np.array([-np.inf]),
    constraint_upper_bounds=np.array([1.0]),
    objective=np.array([-1.0, -1.0]),
    device="gpu" if mlxpdlp.has_gpu() else "cpu",
)
result = solver.solve()
print(result.primal_solution)          # array([1., 0.]) (or symmetric)
print(result.primal_objective_value)   # -1.0
print(result.termination_reason_name)  # OPTIMAL
```

Loading and solving an MPS model (maximization conventions are handled
automatically):

```python
result = mlxpdlp.solve_mps("model.mps.gz", device="cpu")
problem = mlxpdlp.load_mps("model.mps.gz")
print(problem.num_variables, problem.num_constraints, problem.num_nonzeros)
```

Device fallback pattern:

```python
device = "gpu" if mlxpdlp.has_gpu() else "cpu"
```

## Parameters

```python
params = mlxpdlp.Parameters()
params.tolerance = 1e-4          # optimality + feasibility tolerances
params.time_limit_seconds = 60.0
params.iteration_limit = 100000
params.verbose = True
params.presolve = True           # PSLP presolve (default); disable to use
                                 # warm starts
params.geometric_mean_iterations = 12  # default; set 0 to disable
solver = mlxpdlp.Solver(..., parameters=params, device="cpu")
```

Warm starts use the original, unscaled problem coordinates and require
`params.presolve = False`:

```python
solver = mlxpdlp.Solver(..., parameters=params,
                        primal_start=x0, dual_start=y0, device="cpu")
```

## API

| Symbol | Description |
|---|---|
| `mlxpdlp.Solver(...)` | CSR LP solver on `"cpu"` (float64) or `"gpu"`/`"metal"` (float32) |
| `mlxpdlp.Parameters` | Solver parameters (`tolerance`, `time_limit_seconds`, `iteration_limit`, presolve switches, ...) |
| `mlxpdlp.SolveResult` | Primal/dual solutions, objectives, residuals, termination diagnostics |
| `mlxpdlp.load_mps(path)` | Parse an MPS file into an `MpsProblem` |
| `mlxpdlp.solve_mps(path, *, device=..., parameters=...)` | Load + solve an MPS file, handling maximize conventions |
| `mlxpdlp.has_gpu()` | Whether the Metal GPU device is available |
| `mlxpdlp.version()` | Library version |

See [docs/python.md](../docs/python.md) for the full reference.

## Tests

```sh
pytest          # unit tests (fast)
pytest -m ""    # include the small Netlib regression (needs
                # benchmarks/data/netlib/download.sh)
```

## License

Apache-2.0. See [LICENSE](../LICENSE).
