# Changelog

All notable changes to mlxPDLP will be documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and the project intends to follow
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Independent `eps_infeasible_relative` in the C++ and Python termination
  criteria, defaulting to cuPDLPx's `1e-14` on CPU and Metal. The requested
  ray residual ratio is honored without Metal relaxation; the separate
  FP32 separation-gap significance guard remains in place.
- Convergence-aware, cuOpt Stable3-inspired conditional termination
  checkpoints (`conditional_termination_evaluation`, enabled by default): a
  Metal-adapted midpoint check is used only near tolerance and only for working
  models up to 262,144 nonzeros, while restart decisions retain their configured
  cadence. The Python API and LPfeas benchmark protocol expose the switch.
- Tomlin geometric-mean matrix scaling, with 12 alternating row/column passes
  enabled by default ahead of the existing Curtis-Reid, Ruiz, Pock-Chambolle,
  and bound/objective preconditioners; the iteration count is exposed through
  the C++ and Python parameter APIs and benchmark protocol.
- Fused single-kernel Metal PDHG half-steps: CSR SpMV, scaled gradient step,
  bound projection, reflection, Halpern weighting, and major-iteration
  snapshots run in one dispatch per half-step across all three SpMV
  strategies (`metal_fused_kernels`, enabled by default), with minor
  iterations batched into memory-bounded lazy evaluations.
- Active infeasibility/unboundedness termination through Farkas separation
  ray certificates (primal ray for dual infeasibility, dual ray for primal
  infeasibility), with significance floors and unit-consistent residual
  ratio tests.
- Absolute-value objective-gap comparisons in every optimality check so a
  numerically negative gap cannot pass one-sided comparisons.
- Device-tuned SpMV SIMD-group dispatch threshold (scaled by GPU family,
  overridable with `MLXPDLP_SPMV_SIMD_NNZ_THRESHOLD`) and explicit FMA
  accumulation in every Metal SpMV loop.
- `fused|unfused` mode argument for the Metal acceleration example for
  kernel-fusion A/B measurements.
- Regression tests for fused/unfused iteration agreement (scalar, adaptive,
  and SIMD-group dispatch), the SIMD-group threshold override, and the
  primal/dual infeasibility certificates on CPU and Metal.

### Changed

- Upgrade PSLP from 0.0.8 to 0.0.11 for parallel-row bound preservation,
  nonvertex reduced-cost reconstruction, objective-cancellation fixes, and
  early release of the presolve transpose. Build, bootstrap, installed-package
  requirements, and CI caches now use 0.0.11; old source caches are rejected.
- PDHG evaluation blocks now stop exactly at the configured iteration limit and
  use the cheaper minor fused kernel for their first iteration unless a restart
  needs a new fixed-point baseline, avoiding discarded major snapshots.
- Batch fixed-point, residual, objective, restart-distance, and
  infeasibility-ray reductions so each metric group needs one host
  synchronization instead of serial scalar round trips.
- Check sparse-Metal infeasibility certificates on the first block, about
  every 1,000 iterations, and at iteration/time limits; CPU and dense paths
  retain every-block checks.
- Split fused Metal major/minor kernels so minor iterations expose only their
  live state arrays and do not allocate three discarded snapshot outputs.
- Metal iteration performance: the fixed-work 16K-128K acceleration sweep
  improved from 1.2x-2.0x to 3.1x-6.5x versus CPU FP64 on an M3 Max by
  fusing iteration kernels and batching evaluations.
- The benchmark portfolio now falls back to the HPR restart policy when the
  cuPDLPx PID portfolio fails the original-model float64 audit, re-running
  the scaling families with a fresh warm-start correction gate. Netlib
  FORPLAN's FP32 Metal trajectory strands the PID dual certificate beyond
  the published-objective tolerance and now passes through the HPR family.

- The SpMV SIMD-group dispatch threshold now parses the Apple Silicon family
  generation from the device name instead of matching a fixed M1-M4 list,
  covering M5 and newer families automatically.
- Feasibility polishing phases now share the solve-level `time_sec_limit`
  budget: as in cuPDLPx's `check_feas_polishing_termination_criteria`, polish
  elapsed time is measured from the original solve start, so the main loop
  and both polish phases together stay within the requested limit instead of
  each phase receiving its own full copy (up to 3x the limit).
- Removed dead solver state: the cached finite-safe bound arrays
  (`var_lb_finite`, `var_ub_finite`, `con_lb_finite`, `con_ub_finite`) and
  the unused `primal_slack` buffer were written but never read; finite-safe
  bound values are computed fresh where needed, and only the infinite-bound
  masks are cached.

- An approval-gated, one-shot `make` workflow that reuses a local MLX package
  when available or downloads, builds, and privately installs the CI-tested
  MLX revision under `_deps/`.
- One download-consent policy for MLX, PSLP, and MLX's transitive CMake
  dependencies, with strict offline enforcement, direct-CMake opt-in, and
  automatic recovery from interrupted managed MLX checkouts.
- User-facing C++ Metal examples: an explicit trivial-LP solve, exact and
  Netlib CPU FP64/Metal FP32 convergence sweeps, and a self-contained generated
  sparse fixed-work acceleration comparison.
- Python bindings (nanobind + scikit-build-core) under `python/` with
  NumPy CSR input, PSLP presolve, warm starts, MPS loading, CPU
  float64 and Metal float32 device selection, and a small Netlib
  regression through the binding.
- Primal and dual warm starts in original, unscaled problem coordinates.
- Guarded primal/dual feasibility polishing with `FEAS_POLISH_SUCCESS`
  termination and per-phase timing.
- Optional PSLP 0.0.8 presolve, early termination, and solution postsolve,
  including installed-package dependency propagation.
- Presolve and warm-start regression coverage for full reduction, partial
  reduction, postsolve reconstruction, and infeasibility.
- A fixed-work MPS CPU/Metal benchmark and the larger Netlib PILOT87 model
  with reproducible provenance.
- A CSR Metal matrix-vector backend with an explicitly stored sparse
  transpose, dense fallback heuristics, and real-MPS backend-selection
  coverage.
- Sparse preprocessing regression coverage that verifies empty dense matrix
  storage and duplicate-coordinate solutions on Metal.
- An opt-in 40-case Netlib CTest regression for both CPU FP64 and Metal FP32.
- A first-class parallel Netlib regression sweep with backend-aware worker
  counts, longest-first queue seeding, dynamic work stealing, per-worker Metal
  warmup, atomic checkpoint reports, and schema-v7 scheduling metadata.

### Changed

- Propagate local `make -j` into managed MLX and mlxPDLP builds using the
  detected logical CPU count while retaining CI's explicit three-job limit.
- Keep automatic LPfeas validation serial for attributable hard-case
  diagnostics; explicit `--jobs N` remains available for experiments.

- Run the complete CPU PDHG path in FP64, including dense MLX arrays,
  reductions, scaling, iterates, and Accelerate sparse products. Metal remains
  FP32, making CPU the higher-accuracy fallback instead of an arithmetic-match
  backend.
- Select an adaptive 256-thread Metal CSR kernel layout that packs short rows
  while assigning long rows to full threadgroups.
- Keep sparse Metal candidates in host CSR during Ruiz and Pock-Chambolle
  preprocessing, avoiding dense matrix allocation and using
  `O(nnz + m + n)` matrix storage and preprocessing work.
- Use the cuPDLPx Ruiz definition on both sparse and dense paths by measuring
  row and column maxima before applying either scale.
- Keep matrix helpers lazy instead of eagerly evaluating their result before
  the primal or dual update immediately evaluates the dependent graph.
- Select two automatic CPU workers and up to 12 memory-bounded Metal workers,
  based on measured suite-throughput scaling on a 16-core M3 Max.

### Fixed

- Reserve half of the remaining bounded host-FP64 correction budget for joint
  primal-dual continuation when fixed-primal certificate reconstruction does
  not meet the full KKT target. This prevents a stationarity-only phase from
  exhausting the cap at a feasible but suboptimal point; Netlib BOEING2 now
  passes the opt-in `1e-5` original-model audit in its first presolved attempt.
- Build managed MLX and mlxPDLP with one macOS deployment target, defaulting to
  MLX's minimum supported version 14.0, and invalidate stale managed archives
  instead of mixing deployment targets at link time.
- Bound managed MLX compilation to a configurable number of jobs so cache-miss
  builds do not overload GitHub's hosted macOS runner and time out.
- Correct primal-only FP32 near misses with a safeguarded host-FP64 descent on
  the unscaled feasibility norm while preserving the incumbent objective; this
  brings LPfeas `cont11` below the independent `1e-4` audit threshold.

- Accumulate duplicate CSR coordinates during dense conversion instead of
  silently keeping only the final coefficient.
- Use a conservative Frobenius-norm fallback when the deterministic power
  iteration start cannot produce a singular-value estimate.
- Validate CSR row pointers, nonzero storage, and column-index ranges at
  construction.
- Pin a parallel sweep to one process-wide MLX backend so overlapping
  `StreamContext` restoration cannot move CPU FP64 work onto Metal.
- Detect Metal support from the selected MLX library directory when a cached
  `MLX_BUILD_DIR` hint is stale.

## [0.1.0] - 2026-07-26

### Added

- Standalone MLX CPU and Metal implementation of the PDLP algorithm.
- Explicit device selection through `MlxPdlpSolver`.
- Bundled plain and gzip MPS loader.
- Netlib ADLITTLE CPU/GPU regression coverage.
- Installable `mlxPDLP::solver` and `mlxPDLP::mps` CMake targets.
- Downstream `find_package(mlxPDLP)` example.
- macOS GitHub Actions build, test, install, and consumer validation.

### Known limitations

- Metal execution uses `float32` internally (Apple GPU kernels do not
  expose float64); the CPU backend runs `float64` throughout.
- Active infeasibility-certificate termination is not yet implemented;
  infeasibility is currently detected through PSLP presolve.
