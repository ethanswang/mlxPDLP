# mlxPDLP

`mlxPDLP` is a standalone C++ implementation of the Primal-Dual Hybrid
Gradient algorithm for linear programming, powered by
[Apple MLX](https://github.com/ml-explore/mlx).

It runs the same PDHG algorithm on MLX CPU or Metal GPU devices,
accepts CSR linear programs through a small C++ API, and includes an optional
PSLP presolver and MPS loader. It has no build or runtime dependency on CUDA
or the original cuPDLPx source tree.

> [!IMPORTANT]
> Sparse Metal solves remain in CSR form through geometric-mean,
> Ruiz/Pock-Chambolle preprocessing, the power method, and PDHG iteration;
> they never allocate dense `A` or `Aᵀ`. On macOS, sufficiently large sparse
> CPU problems use an Accelerate SpMV primitive for the same reason. Small or
> dense problems retain the dense MLX path.
>
> CPU execution is deliberately FP64 throughout the numerical solve. Metal
> remains FP32 because Apple Silicon GPUs do not expose FP64 arithmetic.
> This makes CPU the higher-accuracy fallback while preserving Metal's
> throughput advantage. On well-conditioned LPs, FP32 Metal iteration plus
> scaling, FP64 original-model auditing, and bounded host-FP64 correction can
> reach an independently verified `1e-5` tolerance (`5e-6` internal target).
> The portable cross-suite guarantee remains the practical `1e-4` tolerance
> because ill-conditioned models can stagnate below that scale.

## Features

- Explicit MLX CPU and GPU device selection
- Metal execution when MLX is built with `MLX_BUILD_METAL=ON`
- CSR Metal matrix-vector products with a stored sparse transpose
- Accelerate sparse CPU matrix-vector products for large CSR models
- Fused single-kernel Metal PDHG half-steps with batched lazy evaluation
- Halpern PDHG with adaptive restart and primal-weight control
- Infeasibility and unboundedness certificates with active termination
- Geometric-mean, Ruiz, Pock-Chambolle, and bound/objective preconditioning
- Bounded host-FP64 correction with original-model residual auditing
- Optional PSLP presolve, early termination, and solution postsolve
- Primal and dual warm starts in original problem coordinates
- Public `double` API with FP64 CPU and FP32 Metal numerical backends
- Python bindings (nanobind) with NumPy CSR input and MPS helpers
- Plain and gzip-compressed MPS loading
- CMake install and `find_package(mlxPDLP)` support
- Analytic CPU/GPU tests plus an opt-in 40-case Netlib regression suite

## Requirements

- CMake 3.25 or newer
- A C++20 compiler
- An MLX C++ library (reused when installed or supplied, otherwise obtainable
  by the source-level `make` bootstrap)
- PSLP 0.0.8 when `MLXPDLP_BUILD_PRESOLVE=ON` (reused when available or
  downloaded under the same explicit build-dependency consent)
- Zlib when `MLXPDLP_BUILD_MPS=ON`
- macOS and Apple Silicon for the Metal path

## Quick start

From the mlxPDLP repository root:

```sh
make
./build/mlxpdlp_example
./build/mlxpdlp_tiny_convergence
./build/mlxpdlp_netlib_convergence
./build/mlxpdlp_metal_acceleration
make test
```

The examples separate one-shot Metal correctness, fixed-iteration convergence,
and acceleration. The convergence programs compare CPU FP64 with Metal FP32 on
an exact two-variable LP and the bundled Netlib ADLITTLE model. The acceleration
program generates a much larger sparse LP and times identical fixed work. A
tiny LP proves API correctness but cannot show GPU acceleration.

`make` checks the current build cache, `MLX_ROOT`, `MLX_SOURCE_DIR`, and
`MLX_BUILD_DIR`, followed by normal CMake search locations. It first attempts
the complete configure with network access disabled. If MLX or PSLP is missing,
it asks once before any download. The prompt covers the tested MLX revision,
PSLP 0.0.8, MLX's metal-cpp archive from `developer.apple.com`, and its pinned
JSON/fmt sources from `github.com`. Allow roughly 1 GB of free disk space and
several minutes for the first MLX compilation. Approved dependencies are built
privately under `_deps/` or the project build tree; no system prefix is
modified. The same `make` invocation then configures and builds mlxPDLP. Later
invocations reuse complete local dependencies without prompting.

Plain `make` defaults to three parallel jobs to avoid overcommitting memory
during MLX's Metal-kernel build. Local `make -j` detects the machine's logical
CPU count and passes it into both the MLX and mlxPDLP CMake builds.
`CMAKE_BUILD_PARALLEL_LEVEL=N` remains an explicit override; CI sets it to
three regardless of local Make flags.

On macOS, the bootstrap builds MLX and mlxPDLP with the same deployment target
(`14.0` by default), the oldest target supported by the pinned MLX revision.
This avoids linker warnings caused when MLX records the host macOS patch
release but mlxPDLP uses the SDK baseline.
Override both builds with `MLXPDLP_MACOS_DEPLOYMENT_TARGET=N`; an existing
`MACOSX_DEPLOYMENT_TARGET` environment value is also respected. Changing the
target automatically rebuilds a managed MLX install.

For a noninteractive build, the opt-in itself records approval:

```sh
make MLXPDLP_FETCH_DEPS=ON
```

To prohibit every build-dependency download, or to select an existing MLX build
explicitly:

```sh
make MLXPDLP_FETCH_DEPS=OFF \
  MLX_SOURCE_DIR=/absolute/path/to/mlx \
  MLX_BUILD_DIR=/absolute/path/to/mlx/build \
  CMAKE_ARGS='-DFETCHCONTENT_SOURCE_DIR_PSLP=/absolute/path/to/pslp-src'
```

With presolve enabled, an offline fresh build must provide an installed PSLP
package through `PSLP_DIR`/`CMAKE_PREFIX_PATH` in `CMAKE_ARGS`, or a source tree
as shown above. Alternatively, pass `-DMLXPDLP_BUILD_PRESOLVE=OFF`. An installed
MLX prefix can be selected with `MLX_ROOT=/absolute/prefix`.

`MLXPDLP_FETCH_MLX` remains accepted as a deprecated compatibility alias for
`MLXPDLP_FETCH_DEPS`. If both are supplied, their normalized values must agree.
Pass extra project options through `CMAKE_ARGS`, for example
`make CMAKE_ARGS='-DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF'`.

### C++ Metal examples

The essential device selection in the trivial example is explicit:

```cpp
#include <mlxPDLP/solver.h>

using namespace mlxpdlp;

pdhg_parameters_t parameters;
mlxpdlp_set_default_parameters(&parameters);
parameters.presolve = false; // ensure this tiny LP reaches PDHG

if (!mx::is_available(mx::Device::gpu)) {
    // MLX was not built with an available Metal backend.
}

MlxPdlpSolver solver(
    num_variables, num_constraints,
    row_ptr, col_ind, values,
    variable_lb, variable_ub,
    constraint_lb, constraint_ub,
    objective, 0.0, &parameters,
    mx::Device::gpu);

auto *result = solver.solve();
mx::synchronize(solver.state().stream);
mlxpdlp_result_free(result);
```

See [`examples/basic.cpp`](examples/basic.cpp) for the complete copyable LP,
backend checks, result ownership, and validation.

For convergence rather than timing, run:

```sh
./build/mlxpdlp_tiny_convergence
./build/mlxpdlp_netlib_convergence
```

[`examples/tiny_convergence.cpp`](examples/tiny_convergence.cpp) runs the exact
two-variable LP at 10, 50, 100, 200, 500, 1,000, and 5,000 iterations. It
reports `||x-x*||inf`, absolute primal and dual residuals, absolute duality gap,
and objective for CPU FP64 and Metal FP32.
[`examples/netlib_convergence.cpp`](examples/netlib_convergence.cpp) runs the
same diagnostic on the bundled 97-variable, 56-constraint Netlib ADLITTLE model
through 20,000 iterations and compares with its published objective. Presolve
and polishing are disabled so every row measures raw PDHG progress; neither
program reports a performance conclusion.

The exact sweep is also the labeled regression smoke test:

```sh
ctest --test-dir build -L smoke --output-on-failure
```

The self-contained
[`examples/metal_acceleration.cpp`](examples/metal_acceleration.cpp) uses a
generated 163,840-by-163,840 CSR matrix with 10,485,760 nonzeros and 1,000
PDHG iterations by default:

```sh
./build/mlxpdlp_metal_acceleration
# optional explicit size/iteration override:
./build/mlxpdlp_metal_acceleration 327680 2000
```

It reports solve-only and setup-inclusive speedups. The first invocation may
spend additional time compiling MLX Metal kernels; warmup is excluded from the
reported comparison. Results depend on the Mac model and power state, and the
comparison is equal PDHG work rather than an accuracy claim because CPU uses
FP64 while Metal uses FP32. See [`examples/README.md`](examples/README.md) for
the example map.

### Manual CMake workflow

The bootstrap is optional. To build MLX separately with Metal enabled:

```sh
cmake -S /path/to/mlx -B /path/to/mlx/build \
  -DCMAKE_OSX_DEPLOYMENT_TARGET=14.0 \
  -DMLX_BUILD_METAL=ON \
  -DMLX_BUILD_TESTS=OFF \
  -DMLX_BUILD_EXAMPLES=OFF
cmake --build /path/to/mlx/build --parallel
```

Then configure mlxPDLP directly:

```sh
cmake -S . -B build \
  -DCMAKE_OSX_DEPLOYMENT_TARGET=14.0 \
  -DMLX_BUILD_DIR=/absolute/path/to/mlx/build \
  -DMLXPDLP_ALLOW_DOWNLOADS=ON
cmake --build build --parallel
ctest --test-dir build --output-on-failure
```

`MLX_BUILD_DIR` should contain `libmlx`. Its parent is searched for MLX
headers. Separate layouts can specify:

```sh
cmake -S . -B build \
  -DMLX_SOURCE_DIR=/absolute/path/to/mlx \
  -DMLX_BUILD_DIR=/absolute/path/to/mlx/build
```

The GPU tests use CTest skip code 77 when MLX exposes no GPU device.

Direct CMake configuration never downloads a missing PSLP checkout by default.
Supply `PSLP_DIR` or `FETCHCONTENT_SOURCE_DIR_PSLP`, disable presolve, or pass
`-DMLXPDLP_ALLOW_DOWNLOADS=ON` as explicit consent. The source-level Makefile
sets this internal configure policy from the current invocation's
`MLXPDLP_FETCH_DEPS` decision, so a cached value cannot weaken a later offline
build.

The managed dependency revision and repository can be overridden for testing
with `MLXPDLP_MLX_REVISION` and `MLXPDLP_MLX_REPOSITORY`. Such overrides are
not part of the tested dependency combination.

### Build options

| Option | Default | Purpose |
|---|---:|---|
| `BUILD_TESTING` | `ON` | Build the regression and device tests |
| `MLXPDLP_BUILD_PRESOLVE` | `ON` | Build PSLP 0.0.8 presolve/postsolve support |
| `MLXPDLP_BUILD_MPS` | `ON` | Build the bundled MPS loader |
| `MLXPDLP_BUILD_EXAMPLES` | `ON` | Build Metal correctness, convergence, and acceleration examples |
| `MLXPDLP_BUILD_BENCHMARKS` | `OFF` | Build fixed-work and LPfeas Metal benchmarks |
| `MLXPDLP_ENABLE_NETLIB_REGRESSION` | `OFF` | Register the downloaded 40-case Netlib CPU/Metal regression suite |
| `MLXPDLP_ENABLE_WARNINGS` | `ON` | Enable common compiler warnings |
| `MLXPDLP_ALLOW_DOWNLOADS` | `OFF` | Allow direct CMake to obtain a missing PSLP source checkout |

For a minimal solver-only library:

```sh
cmake -S . -B build-minimal \
  -DMLX_BUILD_DIR=/absolute/path/to/mlx/build \
  -DBUILD_TESTING=OFF \
  -DMLXPDLP_BUILD_PRESOLVE=OFF \
  -DMLXPDLP_BUILD_MPS=OFF \
  -DMLXPDLP_BUILD_EXAMPLES=OFF
cmake --build build-minimal --parallel
```

## Python bindings

`mlxpdlp` exposes the solver to Python (NumPy CSR input, MPS loading,
PSLP presolve, warm starts, CPU float64 / Metal float32 device
selection):

```sh
CMAKE_ARGS=-DMLXPDLP_ALLOW_DOWNLOADS=ON \
  MLX_BUILD_DIR=/absolute/path/to/mlx/build pip install ./python
```

```python
import mlxpdlp
import numpy as np

solver = mlxpdlp.Solver(2, 1,
    np.array([0, 2], dtype=np.int32), np.array([0, 1], dtype=np.int32),
    np.array([1.0, 1.0]), np.zeros(2), np.full(2, np.inf),
    np.array([-np.inf]), np.array([1.0]), np.array([-1.0, -1.0]),
    device="gpu" if mlxpdlp.has_gpu() else "cpu")
result = solver.solve()
print(result.primal_solution, result.primal_objective_value)
```

See [python/README.md](python/README.md) and
[docs/python.md](docs/python.md) for the full reference, including the
small Netlib regression that runs through the Python binding.

## C++ API

The solver handles minimization problems:

```text
minimize    cᵀx + constant
subject to  constraint_lb ≤ A x ≤ constraint_ub
            variable_lb   ≤ x   ≤ variable_ub
```

```cpp
#include <mlxPDLP/solver.h>

using namespace mlxpdlp;

pdhg_parameters_t parameters;
mlxpdlp_set_default_parameters(&parameters);
parameters.verbose = false;

MlxPdlpSolver solver(
    num_variables,
    num_constraints,
    csr_row_ptr,
    csr_col_indices,
    csr_values,
    variable_lower_bounds,
    variable_upper_bounds,
    constraint_lower_bounds,
    constraint_upper_bounds,
    objective,
    objective_constant,
    &parameters,
    mx::Device::gpu);  // or mx::Device::cpu

mlxpdlp_result_t *result = solver.solve();
// Read result->primal_solution, result->primal_objective_value, ...
mlxpdlp_result_free(result);
```

Null lower-bound pointers represent `-∞`; null upper-bound pointers represent
`+∞`. The constructor validates CSR row pointers, nonzero storage, and column
indices; direct API callers remain responsible for compatible vector lengths.

The overload with `primal_start` and `dual_start` accepts either pointer as
null. Non-null starts must contain finite values in the original, unscaled
problem coordinates:

```cpp
MlxPdlpSolver solver(
    num_variables, num_constraints,
    csr_row_ptr, csr_col_indices, csr_values,
    variable_lower_bounds, variable_upper_bounds,
    constraint_lower_bounds, constraint_upper_bounds,
    objective, objective_constant, &parameters,
    primal_start, dual_start, mx::Device::cpu);
```

PSLP presolve is enabled by the default parameters when presolve support was
built. Set `parameters.presolve = false` to use warm starts or to run the
original problem unchanged. Combining warm starts with presolve currently
throws `std::invalid_argument`, because PSLP does not expose a mapping for
initial iterates.

`parameters.presolve_primal_propagation` is off by default. PSLP's aggressive
propagation can substantially reduce structured models, but its inverse map
may amplify an approximate dual certificate (especially on FP32 Metal). The LPFeas runner may try
it as an audit-guarded portfolio stage; direct API users must opt in.

## MPS loading

```cpp
#include <mlxPDLP/mps_loader.h>

mlxpdlp_mps_problem_t *problem =
    mlxpdlp_mps_problem_load("model.mps");
if (!problem) {
    // Parse or I/O failure.
}

// Construct MlxPdlpSolver from the exposed CSR arrays and bounds.

mlxpdlp_mps_problem_free(problem);
```

The core solver minimizes. Callers loading a maximize MPS model should negate
the objective before solving and restore its sign when reporting. See
[`tests/test_mps_device_comparison.cpp`](tests/test_mps_device_comparison.cpp).

## Install and consume

```sh
cmake --install build --prefix /absolute/install/prefix
```

The installation exports:

- `mlxPDLP::solver`
- `mlxPDLP::mps` when MPS support is enabled

Downstream CMake:

```cmake
find_package(mlxPDLP CONFIG REQUIRED)
target_link_libraries(my_solver PRIVATE mlxPDLP::solver)
```

Pass `MLX_BUILD_DIR`, `MLX_SOURCE_DIR`, or `MLX_ROOT` when configuring the
downstream project so the installed package can locate MLX. A complete link
and solve check (using Metal when available) is provided in
[`examples/installed_consumer`](examples/installed_consumer).

## Tests

| CTest name | Coverage |
|---|---|
| `metal_trivial_example` | Explicit Metal trivial-LP solve and backend validation |
| `metal_acceleration_example` | Generated fixed-work CPU/Metal smoke comparison |
| `tiny_convergence_example` | Seven-point exact-LP convergence sweep; labeled `regression`, `smoke`, and `convergence` |
| `netlib_convergence_example` | Netlib ADLITTLE convergence sweep against its published optimum |
| `mlx_basic` | Basic MLX CPU operations |
| `solver` | Analytic solver regressions |
| `device_comparison` | Analytic LP plus duplicate-coordinate sparse regression on CPU and GPU |
| `mps_device_comparison` | Netlib ADLITTLE on CPU and GPU |
| `netlib_regression_cpu` | Downloaded 40-case Netlib audit on CPU FP64 (opt-in, long) |
| `netlib_regression_metal` | Downloaded 40-case Netlib audit on Metal FP32 (opt-in, long) |

The ADLITTLE regression parses 97 variables, 56 constraints, and 383 matrix
nonzeros, then checks both devices against the published objective
`225494.96316`.

Timing output is diagnostic. ADLITTLE also verifies that a real sparse LP
selects the CSR Metal backend; the analytic comparison remains on the dense
fallback because it is tiny.

## Performance benchmark

### Metal acceleration on Apple Silicon

Fixed-work, equal-iteration head-to-head measurements (presolve off,
identical PDHG work, cold machine; Apple M3 Max, 16 cores, 64 GB;
mlxPDLP 0.1.0, 2026-08-30):

| LPfeas instance | Size (rows x cols / nonzeros) | CPU FP64 | Metal FP32 | Speedup |
|---|---:|---:|---:|---:|
| `s82` | 87,878 x 1,690,631 / 7.0M | 200.4 s | 32.6 s | **6.1x** |
| `dlr1` | 1,735,470 x 9,142,907 / 18.4M | 572.5 s | 26.8 s | **21.4x** |

> **Accuracy note:** Apple Silicon GPUs do not expose FP64 arithmetic,
> so Metal iteration runs in FP32. On normal, well-conditioned problems,
> equilibration plus bounded host-FP64 correction has reached an independently
> audited `1e-5` tolerance: NUG08-3RD and QAP15 measured maximum KKT errors of
> `9.696e-8` and `8.960e-7`, respectively. This is a stronger opt-in target,
> not a blanket FP32 guarantee. The portable supported accuracy remains
> `1e-4` (`5e-5` internal target), and the CPU backend remains the FP64
> reliability path for ill-conditioned models or tighter requirements.

CPU runs the Accelerate sparse FP64 SpMV backend; Metal runs the CSR
FP32 SpMV backend, so the advantage widens with model size (per
iteration Metal is 6.1x faster on `s82` and 21.4x faster on `dlr1`,
with preconditioning overhead negligible on both backends). At the
same `1e-4` audit, 42 of the 49 public LPfeas instances are verified
on Metal FP32.

The optional `mlxpdlp_mps_benchmark` executable runs identical fixed PDHG work
on CPU and Metal; the LPfeas measurements above are its sustained reference
comparison. The small Netlib PILOT87 model remains the quickly-downloadable
payload for smoke-testing the executable (Netlib data is otherwise reserved
for the regression suite). The instances are not distributed with this
repository; fetch them with the download scripts documented in
[benchmarks/data](benchmarks/README.md#data):

```sh
cmake -S . -B build \
  -DMLX_BUILD_DIR=/absolute/path/to/mlx/build \
  -DMLXPDLP_BUILD_BENCHMARKS=ON
cmake --build build --target \
  mlxpdlp_mps_benchmark mlxpdlp_lpfeas_benchmark \
  --parallel
./benchmarks/data/netlib/download.sh
./build/mlxpdlp_mps_benchmark \
  benchmarks/data/netlib/pilot87.mps.gz \
  150000 100 both
```

For the convergence-based LPfeas runner:

```sh
./build/mlxpdlp_lpfeas_benchmark \
  --instance qap15 \
  --tolerance 1e-5 \
  --solver-tolerance 5e-6 \
  --output-prefix benchmarks/results/qap15-metal-1e5 \
  --fail-on-validation
```

The same audited runner supports the 40-instance small-to-medium
Netlib progression:

```sh
./benchmarks/data/netlib/download.sh
./build/mlxpdlp_lpfeas_benchmark \
  --data benchmarks/data/netlib \
  --jobs auto \
  --tolerance 1e-4 \
  --output-prefix benchmarks/results/netlib-metal \
  --fail-on-validation
```

To turn that corpus into a persistent long-running CTest regression:

```sh
./benchmarks/data/netlib/download.sh
cmake -S . -B build \
  -DMLX_BUILD_DIR=/absolute/path/to/mlx/build \
  -DMLXPDLP_BUILD_BENCHMARKS=ON \
  -DMLXPDLP_ENABLE_NETLIB_REGRESSION=ON
cmake --build build --target mlxpdlp_netlib_regression
```

Netlib `--jobs auto` uses backend-specific concurrency, longest-first queue
seeding, and dynamic work stealing. LPfeas `auto` remains serial because those
hard cases are diagnosed one at a time; an explicit `--jobs N` is still
available for experiments. CPU and Metal regression tests are marked
`RUN_SERIAL` at the CTest level so their internal worker pools never compete.

See [the benchmark report](benchmarks/README.md) for the full LPfeas protocol,
float64 audit semantics, online NVIDIA B200 comparison, fixed-work methodology,
provenance, and the latest measured result.

## Documentation

- [Architecture and PDHG walkthrough](docs/architecture.md)
- [CPU/Metal performance benchmark](benchmarks/README.md)
- [Netlib benchmark provenance and audit results](benchmarks/data/netlib/README.md)
- [Contributing](CONTRIBUTING.md)
- [Security policy](SECURITY.md)
- [Changelog](CHANGELOG.md)

## Project status

mlxPDLP is an early standalone release. The highest-priority engineering work
is moving more sparse preprocessing and block-level array work onto Metal.
Active Farkas infeasibility termination, batched block reductions, and a
bounded sparse-Metal certificate cadence are wired into the iteration loop.

## Attribution

The PDHG implementation is derived from
[MIT-Lu-Lab/cuPDLPx](https://github.com/MIT-Lu-Lab/cuPDLPx). The bundled MPS
parser also originated there. mlxPDLP contains the required source locally
and does not depend on a cuPDLPx checkout.

The Curtis-Reid step-size initialization heuristic follows the approach of
[PolyU-IOR/HPR-LP-C](https://github.com/PolyU-IOR/HPR-LP-C) (MIT). No code is
copied from HPR-LP-C; the reference is an algorithmic acknowledgment.

Optional presolve and postsolve use
[PSLP 0.0.8](https://github.com/dance858/PSLP), also licensed under
Apache-2.0.

## Authors

- Ethan Wang <ethanshurui.wang@gmail.com>

## License

Apache-2.0. See [LICENSE](LICENSE).
