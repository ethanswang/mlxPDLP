# Contributing to mlxPDLP

Thanks for helping improve mlxPDLP.

## Before opening a change

- Use a focused issue or pull request.
- Keep public API changes explicit in the pull-request description.
- Add or update tests for numerical behavior.
- Update `docs/architecture.md` when solver behavior or package structure
  changes.
- Add user-visible changes to `CHANGELOG.md`.

## Development setup

The source-level Makefile reuses existing MLX and PSLP dependencies or asks once
before downloading MLX, PSLP, and MLX's pinned transitive build dependencies:

```sh
make
make test
```

For a separately built MLX, provide its paths and prohibit all fallback
downloads. Supply a local PSLP checkout too, or disable presolve on a fresh
build:

```sh
make MLXPDLP_FETCH_DEPS=OFF \
  MLX_SOURCE_DIR=/absolute/path/to/mlx \
  MLX_BUILD_DIR=/absolute/path/to/mlx/build \
  CMAKE_ARGS='-DFETCHCONTENT_SOURCE_DIR_PSLP=/absolute/path/to/pslp-src'
make test MLXPDLP_FETCH_DEPS=OFF \
  MLX_SOURCE_DIR=/absolute/path/to/mlx \
  MLX_BUILD_DIR=/absolute/path/to/mlx/build
```

The direct CMake workflow and checked-in presets remain available:

```sh
export MLX_BUILD_DIR=/absolute/path/to/mlx/build
cmake --preset dev -DMLXPDLP_ALLOW_DOWNLOADS=ON
cmake --build --preset dev
ctest --preset dev
```

GPU comparison tests skip when the selected MLX build has no GPU backend.

## Numerical regression

Treat the 40-case Netlib corpus as the required convergence regression for a
substantive solver, presolve, parser, correction, or numerical-backend change:

```sh
./benchmarks/data/netlib/download.sh
cmake -S . -B build \
  -DMLX_BUILD_DIR=/absolute/path/to/mlx/build \
  -DMLXPDLP_ENABLE_NETLIB_REGRESSION=ON
cmake --build build --target mlxpdlp_netlib_regression
```

The CPU and Metal tests each audit all solutions in FP64 at `1e-4`. Netlib
uses backend-aware parallel work stealing; automatic LPfeas validation stays
serial because those harder cases should be diagnosed and timed one at a time.

## Code style

- C++ requires C++20.
- Follow the checked-in `.clang-format` and `.editorconfig`.
- Prefer explicit ownership and device selection.
- Keep MLX operations on the solver's stored stream.
- Preserve the public `double` API unless a versioned API change is intended.

Format changed C and C++ files with:

```sh
clang-format -i path/to/changed_file.cpp
```

## Pull-request checklist

- The project builds with warnings enabled.
- `ctest` passes on MLX CPU.
- The Netlib CPU and Metal regressions pass after substantive numerical
  changes.
- Metal-specific changes are tested on Apple Silicon when available.
- Install and downstream-consumer behavior remain valid for packaging changes.
- Documentation and changelog entries are current.

## Reporting bugs

Include:

- mlxPDLP and MLX revisions;
- macOS and Xcode versions;
- CPU or GPU device selection;
- CMake configuration output;
- a minimal LP or MPS reproducer when possible;
- complete solver termination metrics.
