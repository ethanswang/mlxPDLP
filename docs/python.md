# Python bindings

`mlxpdlp` exposes the mlxPDLP solver to Python through a nanobind
extension module built with scikit-build-core. The binding wraps the
public C++ API (CSR solver, PSLP presolve, warm starts, MPS loading)
and follows the layered design of the reference bindings
(cuPDLPx, HPR-LP-C): a thin compiled core (`mlxpdlp._core`) plus a
small Python layer with conveniences such as `solve_mps`.

See [python/README.md](../python/README.md) for installation and a
quick start.

## Backends

| Device | Arithmetic | Use |
|---|---|---|
| `"cpu"` | float64 throughout (dense MLX or Accelerate sparse SpMV) | default; higher-accuracy reference/fallback |
| `"gpu"` / `"metal"` | float32 CSR SpMV + optional bounded host-float64 polishing | primary; several times faster on large sparse models |

`mlxpdlp.has_gpu()` reports whether the linked MLX library exposes a
usable GPU device; the portable pattern is
`device = "gpu" if mlxpdlp.has_gpu() else "cpu"`.

Apple Silicon GPUs do not expose FP64 arithmetic, so the Metal backend
runs FP32 and its supported accuracy is a `1e-4` tolerance (`5e-5`
internal target). Choose the CPU backend (FP64 throughout) when tighter
tolerances are required.

## Core API

```python
solver = mlxpdlp.Solver(
    num_variables, num_constraints,
    row_ptr, col_indices, values,           # CSR (int32/int64, float32/float64 numpy arrays)
    variable_lower_bounds=None,             # None = -inf
    variable_upper_bounds=None,             # None = +inf
    constraint_lower_bounds=None,
    constraint_upper_bounds=None,
    objective, objective_constant=0.0,
    parameters=None,
    primal_start=None, dual_start=None, reduced_cost_start=None,
    device="cpu",
)
result = solver.solve()                     # SolveResult
```

- `parameters` is an optional `mlxpdlp.Parameters` instance; defaults
  come from `mlxpdlp_set_default_parameters`.
- Warm starts are given in original, unscaled problem coordinates and
  require `parameters.presolve = False` (PSLP does not expose a
  mapping for initial iterates; combining the two raises
  `ValueError`).
- Nullable bounds are passed as `None`; otherwise each must have
  exactly `num_variables` (variable bounds, objective, starts) or
  `num_constraints` (constraint bounds, dual start) entries.

### Parameters

```python
params = mlxpdlp.Parameters()
params.tolerance = 1e-4            # sets optimality + feasibility tolerances
params.time_limit_seconds = 60.0
params.iteration_limit = 100000
params.verbose = True
params.presolve = True
params.presolve_primal_propagation = False   # safe PSLP
params.curtis_reid_iterations = 20
params.termination_criteria.eps_optimal_relative = 1e-5   # fine-grained
params.restart_params.k_p = 0.25
```

`Parameters.restart_policy` selects the primal-weight restart rule:
`0` = cuPDLPx PID (default), `1` = HPR-LP-style sigma update,
`2` = frozen-weight diagnostic.

### SolveResult

Numpy arrays: `primal_solution`, `dual_solution`, `reduced_cost`.
Scalars: `primal_objective_value`, `dual_objective_value`,
`relative_primal_residual`, `relative_dual_residual`,
`relative_objective_gap`, `total_count`, `termination_reason` (int,
compare with `mlxpdlp.TerminationReason`), `termination_reason_name`
(str), `cumulative_time_sec`, `presolve_time`, reduced-model sizes,
feasibility-polishing and host-double-polishing diagnostics.

### MPS

```python
problem = mlxpdlp.load_mps("model.mps.gz")   # MpsProblem
problem.num_variables, problem.num_constraints, problem.num_nonzeros
problem.row_ptr, problem.col_ind, problem.values
problem.variable_lb, problem.variable_ub
problem.constraint_lb, problem.constraint_ub
problem.objective, problem.objective_constant, problem.maximize

result = mlxpdlp.solve_mps("model.mps.gz", device="cpu")   # convenience
```

`solve_mps` converts maximize models to the solver's minimization
form internally and sign-corrects the reported objective values.

## Testing

```sh
# unit tests (analytic LPs, parameters, warm starts, MPS)
pytest python/tests/test_bindings.py

# small Netlib regression through the binding (7 cases, both
# backends, 1e-4 audit against the published objectives)
# requires benchmarks/data/netlib/download.sh
pytest python/tests/test_netlib_regression.py
```

The Netlib regression mirrors the C++ suite's practical recipe in
miniature: a 5e-5 internal target, a safe-PSLP first attempt, and a
no-presolve retry when the postsolved certificate misses the 1e-4
audit (PSLP's inverse map can amplify an approximate dual certificate
on FP32 Metal).

## Building from source

```sh
# explicitly allow CMake to obtain PSLP:
CMAKE_ARGS=-DMLXPDLP_ALLOW_DOWNLOADS=ON \
  MLX_BUILD_DIR=/path/to/mlx/build pip install ./python
# or remain offline by reusing an existing PSLP checkout:
FETCHCONTENT_SOURCE_DIR_PSLP=/path/to/pslp-src \
  MLX_BUILD_DIR=/path/to/mlx/build pip install ./python
```

The CMake portion of a Python build does not download PSLP unless
`MLXPDLP_ALLOW_DOWNLOADS=ON` is supplied. This flag does not govern packages
that pip itself may obtain for its isolated build environment.

The wheel links MLX and PSLP statically, so the installed package is
self-contained (only macOS system frameworks are required at import
time).
