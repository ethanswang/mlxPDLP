# C++ examples

The examples separate API correctness from performance. A two-variable LP is
useful for reading and validating code, but it is too small to amortize GPU
launch overhead and must not be presented as an acceleration benchmark.

## Explicit Metal solve

[`basic.cpp`](basic.cpp) constructs a two-variable LP in CSR form, disables
presolve and host correction, passes `mx::Device::gpu` to `MlxPdlpSolver`, and
checks that MLX retained a Metal FP32 stream before validating the solution.

```sh
make
./build/mlxpdlp_example
```

Expected output identifies `dense-MLX-Metal-FP32`, reports a nonzero PDHG
iteration count, and ends with `Metal solve validation: PASS`. It returns skip
code 77 with an actionable message when MLX exposes no GPU device.

## Convergence diagnostics

[`tiny_convergence.cpp`](tiny_convergence.cpp) runs the exact two-variable LP
for 10, 50, 100, 200, 500, 1,000, and 5,000 iterations on CPU FP64 and Metal
FP32. Each table reports `||x-x*||inf`, absolute primal and dual residuals,
absolute duality gap, and objective.

[`netlib_convergence.cpp`](netlib_convergence.cpp) performs a seven-point sweep
through 20,000 iterations on the bundled Netlib ADLITTLE MPS model. It reports
relative residuals, relative gap, objective error against the published
optimum, and the backend used.

```sh
./build/mlxpdlp_tiny_convergence
./build/mlxpdlp_netlib_convergence
# Run the exact sweep as the regression smoke test:
ctest --test-dir build -L smoke --output-on-failure
```

Both disable presolve, feasibility polishing, host FP64 correction, and early
termination. Their purpose is to reveal iterative convergence and the FP32
numerical floor, not to compare speed. ADLITTLE requires
`MLXPDLP_BUILD_MPS=ON`, which is the default.

## Self-contained acceleration comparison

[`metal_acceleration.cpp`](metal_acceleration.cpp) generates a diagonally
dominant sparse equality LP in memory. No MPS or benchmark download is needed.
It warms CPU and Metal, then runs the same fixed number of PDHG iterations on
Accelerate CSR FP64 and CSR Metal FP32. The defaults are 163,840 rows and
columns, 10,485,760 nonzeros, and 1,000 iterations per device.

```sh
./build/mlxpdlp_metal_acceleration
# SIZE and ITERATIONS are optional; iterations must be a multiple of 100.
./build/mlxpdlp_metal_acceleration 327680 2000
```

The output reports both backend names, setup and synchronized solve times, and
CPU-time/Metal-time speedups. Speed is not asserted: it varies by Mac, thermal
and power state, and workload size. The comparison demonstrates equal-work
throughput, not equal precision or convergence time.

The first process may spend extra time compiling MLX Metal kernels. That warmup
is intentionally excluded from the displayed timings.

## Installed-package consumer

[`installed_consumer`](installed_consumer) is a separate CMake project using
`find_package(mlxPDLP CONFIG REQUIRED)`. It solves a one-variable LP on Metal
when available and otherwise uses CPU, proving that the installed target is
usable rather than merely linkable.
