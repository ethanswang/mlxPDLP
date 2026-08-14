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

## Self-contained acceleration comparison

[`metal_acceleration.cpp`](metal_acceleration.cpp) generates a diagonally
dominant sparse equality LP in memory. No MPS or benchmark download is needed.
It warms CPU and Metal, then runs the same fixed number of PDHG iterations on
Accelerate CSR FP64 and CSR Metal FP32.

```sh
./build/mlxpdlp_metal_acceleration
# SIZE and ITERATIONS are optional; iterations must be a multiple of 100.
./build/mlxpdlp_metal_acceleration 65536 1000
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
