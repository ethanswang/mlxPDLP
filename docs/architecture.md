# mlxPDLP — Architecture and PDHG Walkthrough

## Overview

`mlxPDLP` is a standalone C++ package that implements the cuPDLPx
Primal-Dual Hybrid Gradient (PDHG) algorithm with MLX operations. It does not
compile, include, or link against the original CUDA cuPDLPx package.

The solver:

- accepts an LP in CSR form through a `double`-precision public API;
- keeps CPU numerical arrays in MLX `float64` and Metal arrays in `float32`;
- uses CSR plus transpose-CSR Metal kernels for sufficiently large sparse
  matrices, with a dense MLX fallback;
- runs the same algorithm on an explicitly selected MLX CPU or GPU device;
- supports MLX Metal when the external MLX library was built with
  `MLX_BUILD_METAL=ON`;
- optionally reduces the host problem with PSLP and reconstructs the original
  solution after PDHG;
- accepts primal and dual warm starts in original problem coordinates;
- includes a local MPS parser for plain or gzip-compressed MPS files;
- exports installable CMake targets for downstream projects.

The implementation originated as a port of cuPDLPx, and the bundled
`src/mps_parser.c` originated in that project. PSLP is an independent,
Apache-2.0 dependency that CMake either finds as version 0.0.8 or fetches from
its pinned `v0.0.8` tag.

This document describes the current implementation by symbol rather than by
line number so it remains useful as the source evolves.

## Current scope

| Capability | Status |
|---|---|
| MLX CPU execution | Implemented and tested |
| MLX Metal GPU execution | Implemented and tested |
| Explicit per-solver device selection | Implemented |
| Ruiz, Pock-Chambolle, and bound/objective scaling | Enabled |
| Halpern PDHG with adaptive restart | Implemented |
| CSR input | Public input format |
| Sparse Metal matrix-vector products | Implemented with custom MLX Metal kernels |
| Plain and gzip MPS loading | Implemented through bundled parser |
| CMake install and `find_package` consumption | Implemented |
| PSLP presolve and postsolve | Implemented and optional at build time |
| Guarded primal/dual and host-FP64 feasibility polishing | Implemented |
| Primal and dual warm starts | Implemented when presolve is disabled |
| Active infeasibility-certificate termination | Implemented (Farkas separation ray tests) |
| Fused single-kernel Metal PDHG half-steps | Implemented and enabled by default |
| Batched lazy evaluation of fused iterations | Implemented with a memory-bounded batch |

The C++ solver API lives in the `mlxpdlp` namespace. The C-compatible MPS
loader uses globally visible `mlxpdlp_`-prefixed names.

## Package layout

| Path | Purpose |
|---|---|
| `include/mlxPDLP/solver.h` | Public solver types, state, and `MlxPdlpSolver` API |
| `src/solver.cpp` | MLX-backed PDHG implementation |
| `src/presolve_adapter.*` | Private PSLP ownership, CSR normalization, and postsolve adapter |
| `include/mlxPDLP/mps_loader.h` | Public standalone MPS loader API |
| `src/mps_loader.c` | Ownership adapter from the parser to `mlxpdlp_mps_problem_t` |
| `src/mps_parser.c` | Bundled MPS parser |
| `src/mps_parser_internal.h` | Private parser types and allocation declarations |
| `examples/basic.cpp` | Explicit Metal trivial-LP solve and validation |
| `examples/tiny_convergence.cpp` | Exact-LP fixed-iteration CPU FP64/Metal FP32 sweep |
| `examples/netlib_convergence.cpp` | Netlib ADLITTLE convergence sweep against its published optimum |
| `examples/metal_acceleration.cpp` | Generated sparse fixed-work CPU/Metal comparison |
| `examples/installed_consumer` | Downstream `find_package` link-and-solve check |
| `tests/test_mlx_basic.cpp` | Basic MLX CPU diagnostic |
| `tests/test_solver.cpp` | Solver regression tests |
| `tests/test_device_comparison.cpp` | Analytic and sparse duplicate-coordinate CPU/GPU comparison |
| `tests/test_mps_device_comparison.cpp` | Netlib ADLITTLE CPU/GPU comparison |
| `tests/data/netlib` | Vendored ADLITTLE MPS benchmark and provenance |
| `benchmarks/mps_device_benchmark.cpp` | Fixed-work MPS CPU/Metal benchmark |
| `benchmarks/lpfeas_benchmark.cpp` | Manifest-driven LPfeas Metal protocol runner |
| `benchmarks/lpfeas_support.cpp` | Original-model float64 certificate verifier |
| `benchmarks/compare_lpfeas.py` | Cross-system comparison with published B200 GPU times |
| `benchmarks/data/netlib` | Larger PILOT87 benchmark and provenance |
| `cmake/FindMLX.cmake` | MLX source/build/install discovery |
| `cmake/mlxPDLPConfig.cmake.in` | Installed CMake package configuration |

## Standalone build and installation

### External dependencies

The package has no dependency on a parent cuPDLPx checkout. Its external
dependencies are:

- CMake 3.25 or newer;
- a C++20 compiler;
- an MLX C++ library, either supplied by the user or privately bootstrapped by
  the source-level Makefile after explicit dependency-download approval;
- PSLP 0.0.8 when `MLXPDLP_BUILD_PRESOLVE=ON`, supplied locally or obtained
  under that same approval;
- Zlib when the bundled MPS parser is enabled;
- Foundation, Metal, MetalKit, and Accelerate on macOS.

`FindMLX.cmake` accepts three independent hints:

- `MLX_BUILD_DIR`: directory containing `libmlx`;
- `MLX_SOURCE_DIR`: directory containing `mlx/mlx.h`;
- `MLX_ROOT`: common installation or source prefix.

When `MLX_BUILD_DIR` is supplied, its parent is also searched for MLX source
headers. The selected build's `CMakeCache.txt` is inspected to report whether
`MLX_BUILD_METAL=ON`.

The source-level `make` entry point probes those hints, an existing mlxPDLP
build cache, and normal CMake search paths before any network operation. It
attempts project configuration with downloads disabled to determine whether
PSLP is also available. Interactive builds ask once when a source dependency is
missing; noninteractive builds require `MLXPDLP_FETCH_DEPS=ON`. Consent covers
MLX, PSLP, and pinned third-party sources requested by MLX's CMake build. The
approved path checks out the MLX revision tested in CI, disables MLX
tests/examples/benchmarks/Python and unused file-format backends, builds the
static CPU/Metal library, and installs it under `_deps/mlx-install`. Source,
build, and install trees remain separate so the normal `FindMLX.cmake` path is
exercised.

`MLXPDLP_FETCH_DEPS=OFF` is passed to CMake on every invocation and guarantees
that the bootstrap performs no dependency download, even if the build cache or
`CMAKE_ARGS` previously enabled downloads. Direct CMake configuration has the
same safe default through `MLXPDLP_ALLOW_DOWNLOADS=OFF`. A previously populated
PSLP checkout is reused through `FETCHCONTENT_SOURCE_DIR_PSLP`, which disables
Git's FetchContent update step. The deprecated `MLXPDLP_FETCH_MLX` name remains
an alias for compatibility.

Managed MLX cloning uses a staging directory. If either the destination or a
staged download is incomplete or at the wrong revision, the bootstrap preserves
it with an `.incomplete[.N]` suffix and replaces it with a fresh, pinned
checkout. This makes interrupted first-run downloads recoverable without manual
filesystem work while retaining any unexpected user content.

### Configure, build, and test

```sh
make
make test
```

Equivalent direct CMake commands are:

```sh
cmake -S . -B build \
  -DMLX_BUILD_DIR=/absolute/path/to/mlx/build \
  -DMLXPDLP_ALLOW_DOWNLOADS=ON
cmake --build build --parallel
ctest --test-dir build --output-on-failure
```

Available build options:

| Option | Default | Effect |
|---|---:|---|
| `BUILD_TESTING` | `ON` | Builds and registers the available CTest targets |
| `MLXPDLP_BUILD_PRESOLVE` | `ON` | Enables PSLP 0.0.8 presolve/postsolve support |
| `MLXPDLP_BUILD_MPS` | `ON` | Builds the bundled MPS loader and requires Zlib |
| `MLXPDLP_BUILD_EXAMPLES` | `ON` | Builds Metal correctness, convergence, and acceleration examples |
| `MLXPDLP_BUILD_BENCHMARKS` | `OFF` | Builds fixed-work and LPfeas Metal benchmarks |
| `MLXPDLP_ENABLE_WARNINGS` | `ON` | Enables common compiler warnings |
| `MLXPDLP_ALLOW_DOWNLOADS` | `OFF` | Allows direct CMake to fetch missing PSLP source |

A minimal solver-only build can disable all optional targets:

```sh
cmake -S . -B build-minimal \
  -DMLX_BUILD_DIR=/absolute/path/to/mlx/build \
  -DBUILD_TESTING=OFF \
  -DMLXPDLP_BUILD_PRESOLVE=OFF \
  -DMLXPDLP_BUILD_MPS=OFF \
  -DMLXPDLP_BUILD_EXAMPLES=OFF
cmake --build build-minimal --parallel
```

### Install and consume

```sh
cmake --install build --prefix /absolute/install/prefix
```

The installation exports:

- `mlxPDLP::solver`;
- `mlxPDLP::mps` when MPS support is enabled.

A downstream project can use:

```cmake
find_package(mlxPDLP CONFIG REQUIRED)
target_link_libraries(my_solver PRIVATE mlxPDLP::solver)
```

The exported backend target propagates its C++20 requirement and MLX
dependency. The installed package includes `FindMLX.cmake`, so the downstream
configure step can use the same `MLX_BUILD_DIR`, `MLX_SOURCE_DIR`, or
`MLX_ROOT` hints. A presolve-enabled installation also installs or locates
PSLP and records it as a package dependency.

## LP model and public API

The core solver handles minimization problems of the form:

```text
minimize    cᵀx + objective_constant
subject to  constraint_lb ≤ A x ≤ constraint_ub
            variable_lb   ≤ x   ≤ variable_ub
```

The constructor accepts a CSR matrix:

```cpp
using namespace mlxpdlp;

MlxPdlpSolver(
    int num_vars,
    int num_cons,
    const int *csr_row_ptr,
    const int *csr_col_ind,
    const double *csr_vals,
    const double *var_lb,
    const double *var_ub,
    const double *con_lb,
    const double *con_ub,
    const double *objective,
    double objective_constant,
    const pdhg_parameters_t *params,
    mx::Device device = mx::Device::cpu);
```

An overload adds optional primal and dual initial points immediately before
the device argument:

```cpp
MlxPdlpSolver(/* LP data */, &params,
              primal_start, dual_start, mx::Device::cpu);
```

Each start pointer may be null. A non-null primal start has `num_vars` finite
values and a non-null dual start has `num_cons` finite values, both in the
original unscaled coordinates.

The final argument selects execution:

```cpp
MlxPdlpSolver cpu_solver(/* LP data */, &params, mx::Device::cpu);
MlxPdlpSolver gpu_solver(/* LP data */, &params, mx::Device::gpu);
```

Null lower-bound pointers mean `-∞`; null upper-bound pointers mean `+∞`.
A null objective pointer produces a zero objective.

The constructor validates dimensions, monotone row pointers, nonzero storage,
and column-index ranges. The bundled MPS parser creates valid CSR storage.

## Device and precision model

### Explicit stream ownership

`MlxPdlpState` stores an `mx::Stream`. The constructor resolves the default
stream for the requested device:

```cpp
s_.stream = mx::default_stream(device);
mx::StreamContext stream_context(s_.stream);
```

The scoped context ensures that array creation, preprocessing, and
constructor-time evaluation occur on the requested device.

`solve()` establishes the same `mx::StreamContext` for the entire solve path.
Most MLX expressions therefore do not need an explicit stream argument while
still executing on the solver's stored device.

The comparison tests verify:

```cpp
solver.state().stream.device == requested_device
```

This is stronger than merely checking whether MLX reports that a GPU exists.

### Device-specific internal precision

The public API accepts and returns `double`. CPU execution retains that
precision throughout PDHG: dense MLX arrays, bounds, iterates, scaling vectors,
reductions, and Accelerate sparse products are FP64. Metal uses FP32 because
Apple GPU kernels do not support FP64; host CSR preprocessing remains in
double and is rounded once when the final Metal buffers are materialized.

The algorithm and stopping logic are shared, but the arithmetic contract is
intentionally different. CPU is the reliable higher-accuracy fallback and
reference path; Metal is the high-throughput path with a practical `1e-4`
target. Because Apple Silicon GPUs do not expose FP64 arithmetic, that
`1e-4` tolerance (with a `5e-5` internal stopping target) is the portable
Metal accuracy ceiling — tighter targets are supported on the FP64 CPU
backend. Result extraction handles either dtype and returns public `double`
buffers.

### Lazy evaluation and synchronization

MLX builds expressions lazily. The implementation calls `mx::eval()` at
algorithmically significant points:

- after constructing dense fallback matrices or uploading sparse matrices;
- after scaling updates;
- after major primal/dual updates;
- before reading scalar reductions or host data.

Scalar `.item<float>()`/`.item<double>()` calls and result extraction are synchronization points
because they make device results visible to the host. The benchmark tests
also call `mx::synchronize(solver.state().stream)` before stopping their wall
clock.

## State initialization

MLX arrays have no default constructor. `_mlx_empty_array()` supplies a
shared zero-length `float32` CPU array so every array member of
`MlxPdlpState` can be initialized safely before a device is selected.
The placeholder carries no problem data.

The constructor then initializes:

- dimensions `m`, `n`, and CSR nonzero count `nnz`;
- either dense fallback `A` and `AT`, or sparse Metal host CSR plus transpose
  structure;
- bounds and objective;
- finite-safe bounds and infinity masks;
- identity rescaling factors;
- zero primal and dual iterates;
- step sizes, restart state, residuals, and timing fields;
- the requested MLX stream.

Primal and dual iterates start at zero unless the optional warm-start
constructor arguments provide finite initial points.

## Matrix representation and linear algebra

### Sparse backend preprocessing and dense fallback storage

The constructor captures the public CSR matrix as double values with int32
row pointers and column indices. It also constructs the row structure of
`Aᵀ` and records a source-index map from every transpose entry back to its
entry in `A`.

For a sparse Metal candidate, Ruiz and Pock-Chambolle scaling scan and update
the host CSR values directly. Each pass costs `O(nnz + m + n)` and retains
the cuPDLPx behavior that duplicate coordinates are distinct stored entries
during scaling. Bounds, objective coefficients, warm starts, and cumulative
scale factors receive their corresponding MLX vector transformations.

After scaling, `prepare_sparse_metal_backend()` rounds and mirrors the
already-scaled values into transpose order and uploads:

- CSR `A` for `A*x`;
- CSR `Aᵀ` for `Aᵀ*y`.

Storing the transpose explicitly avoids floating-point atomics. Metal sparse
selection requires nonzero input, at least 4096 logical dense elements, and
density at most 25%. On macOS, CPU sparse selection uses the same density cap
once the dense matrix would contain 16,777,216 elements; it commits the scaled
double CSR to an Accelerate sparse matrix and uses FP64 products in both
directions. A selected sparse problem never allocates dense `A` or `Aᵀ`; its
storage is `O(nnz + m + n)`. Small or dense problems retain the device-dtype
MLX dense fallback. Dense conversion sums duplicate coordinates so fallback
matrix-vector products preserve their additive semantics.

### Matrix-vector products

Three dispatch strategies are selected per matrix direction from the CSR row
profile:

- `scalar_rows`: one thread per row with a serial FMA loop (rows up to 16
  nonzeros);
- `simdgroup_rows`: one 32-lane SIMD group per row with a strided loop and
  `simd_sum` (uniform medium rows);
- `adaptive`: short rows (up to 64 nonzeros) packed 256 per threadgroup, long
  rows reduced cooperatively by a whole 256-thread group through shared
  memory.

The SIMD-group crossover threshold is device-tuned (the 8M-nonzero default was
measured on an M3 Max; wider or narrower GPU families shift it) and can be
overridden with `MLXPDLP_SPMV_SIMD_NNZ_THRESHOLD`. All accumulation loops use
explicit `fma`. The sparse CPU path wraps Accelerate's double-precision CSR
matrix-vector product in an MLX primitive.

### Fused iteration kernels

On the sparse Metal path, each PDHG half-step runs as one fused kernel: CSR
SpMV, scaled gradient step, bound projection, reflection, Halpern weighting,
and (on major iterations) the `x_pdhg`/`y_pdhg`/`dual_slack` snapshots in a
single dispatch. Six kernels exist (three dispatch strategies per direction);
the update bodies replicate the MLX expression sequences so the fused path is
numerically comparable to the unfused one. Minor iterations accumulate in the
lazy graph and are evaluated every `fused_eval_batch_size()` iterations (a
memory-bounded batch, at most 16), so one `mx::eval` materializes many
iterations instead of four per iteration. `metal_fused_kernels = false`
restores the unfused MLX-expression formulation for A/B comparison. This
design matches the fused bucket kernels of HPR-LP-C.

The fallback uses MLX `matmul`, reshaping vectors to two-dimensional operands:

```cpp
// A * x
auto x_2d = mx::reshape(x, {n, 1});
auto y = mx::reshape(mx::matmul(A, x_2d), {m});

// Aᵀ * y
auto y_2d = mx::reshape(y, {m, 1});
auto x = mx::reshape(mx::matmul(AT, y_2d), {n});
```

`mat_Ax()` and `mat_ATx()` evaluate their results before returning.

### Scalar reductions

The helper methods build and evaluate MLX expressions, then return host
scalars:

| Method | MLX operation | Host result |
|---|---|---|
| `mlx_dot(a, b)` | `sum(a * b)` | `double` extracted from the backend dtype |
| `mlx_norm2(v)` | `linalg::norm(v)` | `double` |
| `mlx_norm_inf(v)` | `max(abs(v))` | `double` |
| `mlx_sum(v)` | `sum(v)` | `double` |

## Bounds and finite-safe arrays

`preprocess_bounds()` constructs:

- `var_lb_finite`, `var_ub_finite`, `con_lb_finite`, and
  `con_ub_finite`, where infinities are replaced by zero;
- four backend-dtype masks marking infinite lower and upper bounds.

Finite-safe values prevent expressions such as `∞ × 0` from producing NaN.
The masks are used by the implemented infeasibility-ray helper.

After Ruiz scaling changes the bounds, `solve()` copies the scaled bounds to
the host and reruns `preprocess_bounds()` so these derived arrays reflect the
scaled problem.

## Default parameters

`mlxpdlp_set_default_parameters()` initializes:

| Parameter | Default |
|---|---:|
| `l_inf_ruiz_iterations` | `10` |
| `has_pock_chambolle_alpha` | `true` |
| `pock_chambolle_alpha` | `1.0` |
| `bound_objective_rescaling` | `true` |
| `verbose` | `true` |
| `termination_evaluation_frequency` | `200` |
| `reflection_coefficient` | `1.0` |
| `metal_fused_kernels` | `true` |
| `sv_max_iter` | `200` |
| `sv_tol` | `1e-4` |
| `eps_optimal_relative` | `1e-4` |
| `eps_feasible_relative` | `1e-4` |
| `time_sec_limit` | `3600` seconds |
| `iteration_limit` | `INT32_MAX` |
| `artificial_restart_threshold` | `0.36` |
| `sufficient_reduction_for_restart` | `0.2` |
| `necessary_reduction_for_restart` | `0.5` |
| `k_p`, `k_i`, `k_d` | `0.99`, `0.01`, `0.0` |
| `i_smooth` | `0.3` |
| `presolve` | `true` when built with PSLP, otherwise `false` |
| `presolve_primal_propagation` | `false` |
| `feasibility_polishing` | `false` |
| `host_double_polishing` | `false` |
| `host_double_polishing_iteration_limit` | `50000` |
| `host_double_polishing_time_sec_limit` | `30` seconds |
| `matrix_zero_tol` | `1e-9` |

Feasibility polishing and host-double correction are guarded, opt-in solve
phases. Host FP64 correction can reconstruct a certificate at fixed primal,
apply an objective-neutral descent step to a primal-only blocker, and continue
joint PDHG. Every route retains the incumbent unless the complete
primal/dual/gap KKT merit improves. L∞ optimality remains available through
`optimality_norm`.

## Presolve, postsolve, and warm starts

When `params.presolve` is true, the constructor runs PSLP on host data before
creating dense MLX arrays. The private adapter:

1. canonicalizes each CSR row by summing duplicate coordinates;
2. removes coefficients whose absolute value is at most
   `matrix_zero_tol`;
3. owns PSLP settings, presolver state, and reduced-problem storage;
4. passes the reduced LP to the normal scaling and PDHG path.

Primal propagation is disabled by the API default because PSLP 0.0.8 can
amplify an approximate reduced dual certificate during postsolve. The LPFeas
runner exposes it as an aggressive, float64-audited portfolio stage and can
fall back to safe presolve or the original model.

If PSLP removes every variable, the solver returns the reconstructed optimal
solution without entering PDHG. PSLP `INFEASIBLE` maps to
`TERMINATION_REASON_PRIMAL_INFEASIBLE`; its combined unbounded-or-infeasible
status maps to `TERMINATION_REASON_INFEASIBLE_OR_UNBOUNDED`. Otherwise,
`extract_result()` passes the reduced primal, dual, and reduced-cost vectors
through PSLP postsolve and reports original problem dimensions alongside the
reduced dimensions and presolve timing.

Presolve support is compiled out with `MLXPDLP_BUILD_PRESOLVE=OFF`. Requesting
presolve from such a build throws `std::runtime_error`.

Warm starts initialize `x_cur` and `y_cur` before preconditioning. Ruiz,
Pock-Chambolle, and bound/objective scaling transform them with the same
factors applied by cuPDLPx, after which the normal Halpern anchors are
initialized from the scaled starts. Non-finite entries are rejected.

Warm starts and PSLP presolve cannot currently be combined: PSLP 0.0.8 does
not expose mappings for projecting initial primal and dual iterates into the
reduced problem. The constructor rejects that combination with
`std::invalid_argument`; callers can set `params.presolve = false`.

## Preconditioning

Preconditioning is active by default and runs in this order:

1. L∞ Ruiz scaling;
2. Pock-Chambolle scaling;
3. bound/objective scaling.

### L∞ Ruiz scaling

For each configured iteration:

```text
row_scale     = sqrt(max(abs(A), axis=1))
col_scale     = sqrt(max(abs(A), axis=0))
inv_row_scale = 1 / row_scale
inv_col_scale = 1 / col_scale
A             = A * inv_row_scale[:, None] * inv_col_scale[None, :]
constraint bounds *= inv_row_scale
objective     *= inv_col_scale
variable bounds *= col_scale
```

Both maxima are measured from the same matrix state, matching cuPDLPx.
Near-zero row or column maxima use a scale of one. `con_rescale` and
`var_rescale` accumulate the non-inverted scale factors for final solution
recovery. The dense fallback rebuilds `AT` after scaling; sparse Metal builds
transpose values once after all preprocessing.

### Pock-Chambolle scaling

With exponent `alpha`, the implementation computes:

```text
row_powsum = sum(abs(A)^alpha, axis=1)
con_scale  = sqrt(row_powsum)

col_powsum = sum(abs(A)^(2 - alpha), axis=0)
var_scale  = sqrt(col_powsum)
```

Near-zero power sums again use one. The matrix is multiplied by the inverse
row and column scales, the objective and constraint bounds receive the
matching inverse scales, variable bounds receive `var_scale`, and the
accumulated rescaling arrays are updated. The dense fallback rebuilds `AT`;
sparse Metal updates CSR values in place.

### Bound/objective scaling

The implementation computes finite constraint-bound contributions and the
objective norm:

```text
constraint_scale = 1 / (constraint_bound_norm + 1)
objective_scale  = 1 / (objective_norm + 1)
```

Constraint bounds and variable bounds are multiplied by
`constraint_scale`; objective coefficients are multiplied by
`objective_scale`. These two global factors are retained in
`con_bound_rescale` and `obj_vec_rescale` for residual and result unscaling.

## Spectral norm and initial step size

`mlx_estimate_max_singular_value()` applies a deterministic power method to
`A Aᵀ`, starting from a fixed-seed random-normal vector:

```text
eigen       = eigen / norm(eigen)
y           = Aᵀ eigen
eigen_next  = A y
sigma_sq    = dot(eigen, eigen_next)
```

The Rayleigh quotient and `norm(eigen_next)` are evaluated in one batch, so
each iteration requires one host synchronization. Iteration stops when the
relative change in `sigma_sq` over a 10-iteration window falls below `sv_tol`,
or when `sv_max_iter` is reached. The solver then uses:

```text
step_size = 0.998 / sqrt(abs(sigma_sq))
```

If the deterministic start lies in `null(Aᵀ)` or the power method produces no
finite estimate, the solver uses the Frobenius norm of `A` as a conservative
spectral-norm bound. An empty or numerically zero matrix falls back to a step
size of one.

When bound/objective scaling is enabled, the initial primal weight is one.
Otherwise it is initialized from the original objective and constraint-bound
norms. The directional steps are:

```text
step_size_primal = step_size / primal_weight
step_size_dual   = step_size * primal_weight
```

## Halpern PDHG iteration

Iterations are grouped into blocks of
`termination_evaluation_frequency`, which defaults to 200.

Each block contains:

1. one major primal/dual iteration;
2. minor iterations `2 ... eval_freq - 1`;
3. one final major primal/dual iteration;
4. fixed-point and residual evaluation;
5. termination and restart checks.

Major iterations store `x_pdhg`, `y_pdhg`, and `dual_slack`. Minor
iterations advance the running `x_cur` and `y_cur` without replacing those
candidate snapshots.

### Primal update

For `k = inner_count + k_offset`:

```text
weight    = k / (k + 1)
ATy       = Aᵀ y_cur
temp      = x_cur - step_primal * (objective - ATy)
projected = clip(temp, variable_lb, variable_ub)
x_ref     = 2 * projected - x_cur
reflected = reflection_coefficient * x_ref
            + (1 - reflection_coefficient) * x_cur
x_cur     = weight * reflected + (1 - weight) * x_init
```

On a major iteration:

```text
x_pdhg     = projected
dual_slack = (x_pdhg - temp) / step_primal
```

### Dual update

The dual step uses the reflected primal point:

```text
Ax          = A x_ref
temp        = y_cur / step_dual - Ax
temp_proj   = clip(temp, -constraint_ub, -constraint_lb)
y_candidate = (temp - temp_proj) * step_dual
y_ref       = 2 * y_candidate - y_cur
reflected   = reflection_coefficient * y_ref
              + (1 - reflection_coefficient) * y_cur
y_cur       = weight * reflected + (1 - weight) * y_init
```

On a major iteration, `y_pdhg = y_candidate`.

## Fixed-point error

The current implementation defines:

```text
delta_x = x_ref - x_pdhg
delta_y = y_ref - y_pdhg

movement = primal_weight * ||delta_x||²
           + ||delta_y||² / primal_weight

interaction = 2 * step_size * dot(Aᵀ delta_y, delta_x)

fixed_point_error = sqrt(max(movement + interaction, 0))
```

The nonnegative guard prevents roundoff or a negative interaction term from
producing an invalid square root.

## Residuals and objectives

Residuals are evaluated from the current candidate:

```text
Ax          = A x_pdhg
primal_res  = Ax - clip(Ax, constraint_lb, constraint_ub)

ATy         = Aᵀ y_pdhg
dual_res    = objective - ATy - dual_slack
```

Absolute residuals undo the corresponding global scaling:

```text
absolute_primal_residual = ||primal_res||₂ / con_bound_rescale
absolute_dual_residual   = ||dual_res||₂ / obj_vec_rescale
```

Relative residuals use the original problem norms computed before
preconditioning:

```text
relative_primal_residual =
    absolute_primal_residual / (1 + constraint_bound_norm)

relative_dual_residual =
    absolute_dual_residual / (1 + objective_vector_norm)
```

The primal objective is:

```text
dot(scaled_objective, x_pdhg)
------------------------------------- + objective_constant
con_bound_rescale * obj_vec_rescale
```

The dual objective selects the lower constraint bound for positive dual
components and the upper bound otherwise, replacing infinite bounds with
zero before multiplication. It also includes the dual-slack contribution.

```text
objective_gap          = primal_objective - dual_objective
relative_objective_gap =
    objective_gap / max(abs(primal_objective), abs(dual_objective), 1)
```

The relative gap is currently signed, and the termination check compares it
directly with `eps_optimal_relative`.

## Adaptive restart and primal-weight update

`mlx_should_adaptive_restart()` uses the following rules:

1. the first completed evaluation block always restarts;
2. restart when fixed-point error is at most
   `sufficient_reduction_for_restart × initial_fixed_point_error`;
3. restart when it is below the necessary-reduction threshold but has
   increased since the previous trial;
4. restart when `inner_count` reaches
   `artificial_restart_threshold × total_count`.

The legacy `interaction` and `movement` output parameters remain in the
method signature but are not used by these current restart decisions.

`mlx_perform_restart()` first measures:

```text
delta_x = x_pdhg - x_init
delta_y = y_pdhg - y_init
primal_distance = ||delta_x||₂
dual_distance   = ||delta_y||₂
```

When the distances and residual ratio pass safety guards, the PID error is:

```text
error = log(dual_distance)
        - log(primal_distance)
        - log(primal_weight)
```

The integral term is exponentially smoothed by `i_smooth`, and the weight is
updated multiplicatively:

```text
primal_weight *= exp(k_p * error
                     + k_i * smoothed_error_sum
                     + k_d * delta_error)
```

Invalid distance or residual conditions restore the best previously observed
primal weight. The final weight is clamped to `[1e-12, 1e12]`.

Restart then:

- moves both Halpern anchors to the current candidate;
- moves running iterates to that candidate;
- resets `inner_count` to zero;
- recomputes primal and dual step sizes.

## Termination

`mlx_check_termination()` checks, in order:

1. L2 optimality (gap compared in absolute value);
2. infeasibility certification (Farkas separation ray tests);
3. the host-FP64 handoff gates when enabled;
4. iteration limit;
5. time limit.

Optimality requires:

```text
relative_primal_residual < eps_feasible_relative
relative_dual_residual   < eps_feasible_relative
abs(relative_objective_gap) < eps_optimal_relative
```

### Infeasibility certification

`mlx_compute_infeasibility_information()` builds Farkas separation
certificates from the sign-constrained fixed-point deltas, normalized to
unit inf-norm:

- A **primal ray** r (dual-infeasibility certificate) must lie in the
  recession cone of the variable box with A r in the recession cone of the
  constraint box, and needs cᵀr < 0. Violations of the recession-cone sign
  conditions form `max_primal_ray_infeasibility`.
- A **dual ray** y (primal-infeasibility certificate) needs the separation
  gap min_s yᵀs − max_x (Aᵀy)ᵀx > 0 over the constraint box s and variable
  box x, with finiteness sign violations forming
  `max_dual_ray_infeasibility`. Feasible problems satisfy the gap ≤ 0 for
  every y by weak duality, so this test cannot fire on them in exact
  arithmetic.

Termination fires when the certificate gap is significant relative to the
original problem data and the recession-cone residual is small relative to
the gap, with residual ratios compared in consistent working units. The
FP64 CPU path uses `eps_feasible_relative` for the ratio; the FP32 Metal
path uses at least 1e-3 because FP32 ray arithmetic floors the attainable
ratio. Both reference implementations are more conservative: cuPDLPx
computes ray metrics but never calls its criteria, and HPR-LP-C relies
entirely on PSLP presolve for infeasibility proofs.

Timing uses `clock()` and is stored in `cumulative_time_sec`.

## Result extraction and ownership

`extract_result()` reverses accumulated scaling:

```text
x_final = x_pdhg / var_rescale / con_bound_rescale
y_final = y_pdhg / con_rescale / obj_vec_rescale

reduced_cost =
    (scaled_objective - Aᵀ y_pdhg) * var_rescale / obj_vec_rescale
```

The three MLX arrays are evaluated, read as `float`, widened to `double`, and
stored in new host arrays. For a reduced PSLP problem, postsolve replaces
those buffers with reconstructed original-dimension primal, dual, and
reduced-cost arrays. The result retains reduced dimensions in its
`num_reduced_*` fields and exposes PSLP status and elapsed time.

`solve()` returns a heap-allocated `mlxpdlp_result_t`, including its three
solution buffers. Release the complete result with:

```cpp
mlxpdlp_result_free(result);
```

The helper accepts `nullptr`.

## Bundled MPS loader

`mlxpdlp_mps_problem_load(path)` parses plain `.mps` files and gzip-compressed
files whose name ends in `.gz`. The local parser handles the standard MPS
sections:

- `ROWS`;
- `COLUMNS`;
- `RHS`;
- `RANGES`;
- `BOUNDS`;
- `OBJSENSE`;
- `ENDATA`.

The parser first accumulates coordinate entries, then creates owned CSR
arrays. `src/mps_loader.c` exposes those arrays through `mlxpdlp_mps_problem_t` and
retains an opaque ownership pointer. Call:

```c
mlxpdlp_mps_problem_t *problem = mlxpdlp_mps_problem_load(path);
/* use problem */
mlxpdlp_mps_problem_free(problem);
```

The core solver is a minimizer. When an MPS problem has maximize sense, the
caller must negate the objective before solving and restore the objective
sign when reporting. `tests/test_mps_device_comparison.cpp` demonstrates this
normalization.

MPS support can be excluded entirely with
`MLXPDLP_BUILD_MPS=OFF`; the solver library then has no Zlib
dependency.

## Tests and CPU/GPU comparison

CTest registers:

| Test | Coverage |
|---|---|
| `bootstrap_dependency_policy` | Unified consent, offline enforcement, and interrupted-checkout recovery |
| `metal_trivial_example` | Explicit Metal device selection and trivial-LP validation |
| `metal_acceleration_example` | Generated sparse CPU/Metal fixed-work smoke comparison |
| `tiny_convergence_example` | Exact-LP fixed-iteration sweep (`regression;smoke;convergence`) |
| `netlib_convergence_example` | Netlib ADLITTLE convergence sweep and published-objective check |
| `mlx_basic` | Basic MLX CPU array operations |
| `solver` | Solver, warm-start, presolve, postsolve, termination, and FP64 Farkas infeasibility-certificate regressions |
| `device_comparison` | Analytic and sparse LPs on CPU and GPU, fused/unfused iteration agreement across all three SpMV strategies, SIMD-group threshold override, and the FP32 Metal infeasibility certificate |
| `mps_device_comparison` | Bundled Netlib ADLITTLE MPS on CPU and GPU |
| `netlib_regression_cpu` | Opt-in downloaded 40-case Netlib audit on CPU FP64 |
| `netlib_regression_metal` | Opt-in downloaded 40-case Netlib audit on Metal FP32 |

The two GPU comparisons:

- request `mx::Device::gpu` explicitly;
- verify the solver retained the requested stream device;
- synchronize before recording wall time;
- return skip code 77 if MLX exposes no GPU device.

Device-comparison tests also assert the intended precision split: CPU state
arrays are FP64 and Metal state arrays are FP32.

The long Netlib tests are enabled with
`MLXPDLP_ENABLE_NETLIB_REGRESSION=ON`. CTest keeps the backend sweeps serial,
while each Netlib runner seeds structurally expensive models first and
dynamically steals remaining cases across a backend-specific, memory-bounded
worker pool. Automatic LPfeas sweeps use one worker so hard cases retain clean
per-model convergence and timing diagnostics.

The ADLITTLE regression checks:

- 97 variables;
- 56 parsed constraints;
- 383 constraint-matrix nonzeros;
- published optimum `225494.96316`;
- solver optimal termination;
- CPU and GPU objective agreement within a `1e-4` relative tolerance;
- dense fallback selection on CPU and CSR-kernel selection on Metal.

The test uses `1e-5` feasibility and optimality tolerances. Timing is
diagnostic rather than a pass criterion.

ADLITTLE and the analytic comparison are both small for a GPU, so their
timings are not performance requirements. The larger fixed-work PILOT87
benchmark is the sustained sparse-path comparison.

## Verified standalone behavior

The package has been validated by:

1. copying `mlxPDLP` outside the original repository;
2. configuring and building that isolated source tree;
3. passing all registered tests with actual Metal execution;
4. installing to a separate prefix;
5. configuring, linking, and running
   `examples/installed_consumer` through `find_package(mlxPDLP)`;
6. building the minimal solver-only configuration without tests, examples,
   PSLP, MPS support, or Zlib.

No source, build, or installed-package artifact from that validation referred
back to the original cuPDLPx checkout.

## Current limitations and next engineering priorities

1. **Metal is single precision internally.** This limits attainable accuracy
   on poorly scaled LPs; CPU FP64 is the reliability fallback. FP32 ray
   arithmetic also limits infeasibility certification to cleaner rays on
   Metal (the FP64 CPU path certifies the pathological cases).
2. **Warm-start/presolve exclusivity.** PSLP 0.0.8 does not expose an initial
   iterate mapping, so callers must choose one feature per solve.
3. **Fused kernels cover the sparse Metal path only.** Dense-fallback and
   CPU problems still use the unfused MLX-expression formulation.
4. **Infeasibility certification uses L2-norm reduction reads.** The ray
   checks perform two extra SpMVs and several host-synchronizing reductions
   per evaluation block.
5. **Core minimization semantics.** Maximize sense is normalized by callers,
   including the MPS comparison.

The next performance priorities are moving sparse preprocessing to Metal if
host setup becomes material, and widening the fused kernels to cover the
residual and fixed-point SpMVs so block-level reductions run as single
dispatches.
