# Plan: shared-matrix LP batching

Status: complete. Workload qualification passed; C++/Python batch APIs, shared
preparation, fused Metal SpMM, independent control/audits, optional native K
buffer reuse, release measurements, and full regression validation are complete.
The benchmark artifacts retain the detailed performance report and completion
audit. On 2026-09-10 the user authorized `../alpha-beta-CROWN` or families
synthesized from existing Netlib and LPfeas cases as the qualification workload,
with no caller per-LP latency or resident-memory limit. Synthetic measurements
must retain their provenance and do not establish whole-application speedups.
The qualification record and capture protocol are retained with the benchmark
artifacts. Baseline:
`b29fc6b64c881371a8a77e6cf8144d6a6b569e50` (2026-09-08 review).

Add a reusable matrix plan and a Metal solver that advances multiple independent
LPs together when they have identical constraint matrices. Proceed with the full
feature only if representative workloads show enough compatible, simultaneously
available LPs to improve application performance. The measured workload qualification is recorded above; the thresholds below
remain the acceptance criteria for the implementation.

The motivating example is a family of minimization problems indexed by `j`:

```text
minimize       c_j^T x_j + offset_j
subject to     constraint_lb_j <= A x_j <= constraint_ub_j
               variable_lb_j   <= x_j   <= variable_ub_j
```

All members share the dimensions, sparsity structure, and numerical values of
`A`. Objectives, objective offsets, both kinds of bounds, and warm starts may
differ. Sharing only the sparsity pattern is insufficient for this version.
The products become `A X` and `A^T Y`, where each column represents one LP.

This adds a second batching dimension: `B` denotes independent LPs, while `K`
denotes consecutive PDHG iterations between evaluations. The existing
`metal_iteration_batching` implementation handles `K`; it does not implement
`B`. Keep the settings, statistics, and benchmarks distinct.

**Workload qualification comes first.**

Capture representative application traces or replayable fixtures before changing
the solver. Record the following without changing the application’s scheduling:

- Exact matrix identity, dimensions, nonzeros, and matrix reuse count.
- Which fields vary, presolve and warm-start settings, tolerances, iteration
  counts, solve times, and time spent constructing and scaling each solver.
- How many compatible LPs are ready together, their dependencies, and the
  caller’s latency limit. A future LP whose data depends on an earlier result
  cannot be made available early by batching.
- Time spent in LP solving relative to the whole application, and the fraction
  attributable to compatible groups. Collect latency distributions, not just
  average batch size.

Use an explicit plan identity for production requests. Trace grouping may hash
validated CSR inputs, but must confirm exact equality before combining matrices;
do not group approximately equal coefficients or drop small entries. Existing
duplicate-coordinate semantics must remain intact.

Proposed qualification gate: at least one representative application family has
compatible groups accounting for 25% or more of current LP wall time, with a
median ready group of at least four LPs within its latency limit. If this is not
met, record the evidence and defer the SpMM solver. Reusing matrix preparation
for sequential solves can still be evaluated separately, without claiming a
batch-throughput benefit. The 2026-09-10 authorization above permits measured
derived families to establish this workload gate; reports must explicitly enable
and record that qualification policy.

**Architecture and ownership.**

Use a matrix plan shared by independent mutable solve states. Refactor the
existing numerical helpers only as required for this feature; preserve the
current single-LP API, defaults, result ownership, and fallback behavior.

| Component | Owns or does | Existing integration points |
|---|---|---|
| `SharedMatrixPlan` (new) | Validated original FP64 CSR; common diagonal scaling; scaled CSR and transpose; row-work metadata; device kernels; matrix norm estimate and certified bound | `capture_sparse_matrix`, scaling methods, `prepare_sparse_metal_backend` in [solver.cpp](src/solver.cpp); [metal_spmv.h](src/metal_spmv.h) |
| `BatchWorkspace` (new, private) | Packed per-LP vectors, bounds, objectives, snapshots, controller state, masks, and reusable scratch arrays | State definitions in [solver.h](include/mlxPDLP/solver.h); buffer ownership pattern in [metal_minor_batch.cpp](src/metal_minor_batch.cpp) |
| Batched PDHG driver (new) | Shared products and independent projection, Halpern, restart, stopping, and recovery decisions | Fused step helpers and `solve`; [pdhg_control.h](src/pdhg_control.h) |
| Certificate service (small extraction) | Unscaling and original-model checks using a shared matrix view and each LP’s own data | `copy_unscaled_certificate`, `recompute_original_certificate`, [host_double_polish.cpp](src/host_double_polish.cpp) |
| Public batch result (new) | Ordered owned results plus execution and timing metadata | `mlxpdlp_result_t`, existing result deleter, `SolveResult` in [module.cpp](python/src/module.cpp) |

The plan owns its inputs and is immutable after preparation. Mutating caller
arrays cannot change an existing plan. Matrix preparation parameters and device
selection are fixed at plan creation; incompatible changes require a new plan.
Every submission gets independent mutable state. Initially serialize submissions
through one plan; do not promise concurrent calls on a shared workspace. Results
must remain valid after the next submission and after plan destruction.

Proposed file additions are `include/mlxPDLP/batch_solver.h`,
`src/shared_matrix.{h,cpp}`, `src/batch_solver.cpp`, and `src/metal_spmm.h`.
Keep numerical formulas in shared helpers where practical instead of maintaining
two divergent implementations. Reuse [cpu_sparse_matrix.h](src/cpu_sparse_matrix.h)
for suitable host products. Production code must not depend on benchmark code;
[lpfeas_support.cpp](benchmarks/lpfeas_support.cpp) remains an independent verifier.

**Public API contract.**

Add C++ plan construction and `solve_batch` first, followed by Python bindings
and type stubs. The following is a proposed Python interface, not an existing API:

```python
parameters = mlxpdlp.Parameters()
parameters.presolve = False  # Explicit admission to the shared-matrix path.

plan = mlxpdlp.SharedMatrixPlan(
    num_variables=n,
    num_constraints=m,
    row_ptr=row_ptr,
    col_indices=col_indices,
    values=values,
    parameters=parameters,
    device="gpu",
)
batch = plan.solve_batch(
    objective=objectives,                         # [B, n]
    objective_constant=offsets,                    # [B] or scalar
    variable_lower_bounds=variable_lb,             # [B, n] or [n]
    variable_upper_bounds=variable_ub,             # [B, n] or [n]
    constraint_lower_bounds=constraint_lb,         # [B, m] or [m]
    constraint_upper_bounds=constraint_ub,         # [B, m] or [m]
)
results = batch.results  # B owned SolveResult objects in input order.
```

Accept FP64 problem data and int32 CSR through the same validation conventions
as the existing API. Specify broadcasting only for the documented common bounds
and scalar offset; reject accidental broadcasting or incompatible shapes.
Optional primal, dual, and reduced-cost warm starts use original coordinates,
with explicit member masks if only part of the batch supplies them. Copy/pack
request data before execution can outlive the caller’s buffers.

Use one common set of algorithm settings and requested tolerances per submission
in the initial version. State such as actual step sizes, weights, restart epochs,
and completion remains independent per member. Expose stable input indices,
per-member status and iteration count, certificate metrics, queue time, and
actual execution mode with a fallback reason. Preserve the distinction between
batch wall time and overlapping per-member elapsed times.

Add separate batch options for a request wall-time limit, a resident-memory
budget, and execution mode (`auto`, `shared`, or `independent`). `auto` reports
any fallback; forced `shared` mode rejects unsupported configurations so tests
and benchmarks cannot silently measure independent solves. Snapshot parameters
on submission and validate that matrix-preparation settings match the plan.

Return all member results. Do not introduce “stop at the first feasible member”
semantics in the initial release. Invalid input rejects the request before any
kernel launch, identifying the member and field. Numerical failure after valid
input is handled per member; an unrecoverable device failure fails the request
explicitly and must not fabricate successful member results. An empty batch
returns an empty result; `B=1` uses the existing single-LP route.

**Scaling and presolve must preserve the shared operator.**

Compute matrix-only geometric-mean, Curtis–Reid, Ruiz, and Pock–Chambolle factors
once using the existing order and selected parameters. Preserve host FP64
preparation and the final rounding into Metal storage. Cache norm information for
the actual scaled, stored operator, including the bound used by safeguards.
Each LP retains its own bound/objective normalization and original residual
denominators; objective offsets do not enter projections.

To make coordinate conversions reviewable, let `R` and `C` be the positive
diagonal matrices recorded by the existing row and variable rescaling, and let
`beta_j` and `alpha_j` be the per-LP bound and objective normalization factors:

```text
A_scaled            = R^-1 A C^-1                  # shared
constraint_bounds_j = beta_j R^-1 constraint_bounds_original_j
variable_bounds_j   = beta_j C variable_bounds_original_j
objective_j         = alpha_j C^-1 objective_original_j

x_original_j        = C^-1 x_scaled_j / beta_j
y_original_j        = R^-1 y_scaled_j / alpha_j
z_original_j        = C z_scaled_j / alpha_j
```

Apply the inverse mappings to warm starts and include each original offset when
auditing objectives. Test finite/infinite bounds without multiplying masks by
infinity. Common matrix scaling does not imply common primal weights or common
recovery step sizes.

The shared backend initially requires `presolve=False`. If presolve is requested,
use the existing independent solve path with that setting and report the
fallback. Never silently disable presolve or reuse one member’s PSLP reductions
or postsolve map for another member. A future common presolve may share only
transformations proven valid for every member. Compare the batch path against
the application’s actual presolved baseline as well as a matched no-presolve
baseline, since losing presolve can outweigh GPU savings.

**Metal execution and independent solver control.**

Expose batch-first arrays at the API boundary and initially pack device vectors
as `[component, B_padded]`, with LP lanes contiguous. Mask padded lanes. Benchmark
small compile-time LP tile widths, initially 4 and 8, and select row scheduling
independently for `A` and `A^T`. Do not assume the best SpMV row mapping is also
the best SpMM mapping.

Start with correct shared CSR products. Then fuse each product with its PDHG
half-step: `A^T Y` with primal projection/reflection/Halpern blending, followed
by `A X_ref` with the corresponding dual update. Avoid materializing intermediate
products in minor iterations. Preserve separate major variants for projected
`X_pdhg`, `Y_pdhg`, and projection slack `Z`; keep the current fixed-point delta
normalization. Checkpoint products and reductions are per LP, never reduced
across the LP axis.

Each member needs its own:

- Current iterate, projected candidate, reflection, restart anchors, and best
  complete certificate, including reduced costs.
- Iteration count, restart epoch count, Halpern weight, primal weight, directional
  steps, PID/HPR history, fixed-point baseline, and numerical recovery state.
- Feasibility and objective metrics, original norms, active/completion state,
  infeasibility checks, audit state, and polishing eligibility.

Initially make controller decisions on the host with the existing guarded FP64
helpers, transferring packed vectors of metrics and decisions at checkpoints.
Avoid a separate host synchronization for every LP. Preserve the existing
FP64-to-FP32 coefficient conversion when preparing Halpern weights for a block.
Use device masks to restart or freeze individual LPs without disturbing others.
Capture a fresh first-step baseline only for members that have restarted.

Start with common regular checkpoint times and independent decisions. Additional
conditional checks use the union of eligible members’ due times; only due
members update checkpoint-derived state. They must not change other members’
restart schedules or best-candidate histories. A completed member’s buffers stay
frozen while others continue. Guard inactive lanes before divisions and other
unsafe arithmetic so one non-finite state cannot pollute another LP’s reductions.
Keep threadgroup barriers uniform even when some LP lanes are inactive.

Only mark `OPTIMAL` after the member’s original-model FP64 audit passes primal
and variable-bound feasibility, dual feasibility and bound signs, and objective
gap checks. A failed audit resumes that member or enters its bounded correction
path. Preserve optional FP64 feedback, original-coordinate warm starts, and
host continuation through reusable helpers. Host corrections can run serially
initially, with their time included in batch measurements. Validate infeasibility
or unboundedness certificates per member; failure to converge is not such a proof.

An iteration limit applies to each LP, not the sum of work across the batch.
Clip blocks to the earliest due check or remaining member limit. A batch wall
deadline starts at request entry and includes packing, queueing, audits, and
correction. A member’s solver time allowance starts when it is admitted, bounded
by the batch deadline; report both queue and execution time. Check time before
dispatches and correction stages. Final extraction/auditing can incur bounded
overhead beyond a deadline, as with the existing solver; measure it explicitly.
Queued members that cannot start return a time-limit status with explicit
`has_solution=False` batch metadata rather than an invented certificate.

Bound memory by one shared matrix plan plus `O(B * (n + m))` member storage.
Account for bounds, snapshots, corrections, padded lanes, retained MLX arrays,
and allocator overhead when selecting a resident batch. Process larger requests
in bounded groups using the same plan. Keep stable result indices. Use masks
first; compaction or moving slow members to a separate group requires evidence
that its copy and synchronization costs pay off.

After the SpMM solver is correct, add optional reuse across `K` minor iterations.
The current native adapter assumes one-dimensional vectors and specific kernel
metadata; it cannot be reused unchanged for batched LP tensors. Extend it through
an explicit supported interface or add a separate adapter. Reuse scratch buffers
while retaining anchors and result snapshots. Keep an ordinary MLX-kernel path
when native encoder interfaces are unavailable. Never expand the matrix into
`B` copies or a block-diagonal representation on the production sparse path.

**Implementation sequence and exit gates.**

Each stage should be a reviewable change with its own evidence. Intermediate
reference paths do not constitute completion of the accelerated feature.

| Stage | Work | Exit evidence |
|---|---|---|
| 0. Qualify demand | Add a trace/replay protocol and representative fixtures; evaluate the workload gate above | Recorded family distribution, ready batch sizes, baseline time breakdown, and go/defer decision |
| 1. Extract reusable matrix preparation | Introduce the immutable plan; split matrix-only preparation from per-LP data; retain an independent single-LP solve using the plan | Existing tests pass; repeated submissions preserve results; preparation/upload occurs once; no extra matrix copies per member |
| 2. Establish the batch contract | Add C++ data/result types, validation, packing, CPU/reference execution, and explicit fallback modes | Shapes, broadcasting, ownership, limits, input order, and width-one behavior tested; no claim of GPU acceleration yet |
| 3. Implement shared CSR SpMM | Add both orientations, LP tiling, tail masks, and direct FP64 reference comparisons; then fused major/minor half-steps | Correct products and fixed-work iterates across row layouts and batch widths; allocation evidence confirms shared matrix storage |
| 4. Complete independent control | Add per-member restarts, conditional checks, stopping, numerical recovery, audits, and correction | Heterogeneous batches match independent certificate outcomes; one member’s restart/failure/exit cannot change another member’s state |
| 5. Add Python and package support | Bind the plan and owned results; update exports, stubs, installed headers, examples, and package tests | Python ownership and repeated-call tests pass; an installed C++ consumer solves a batch without private headers |
| 6. Optimize and measure | Tune LP tiles and resident sizes; add optional native iteration-buffer reuse; measure fallback dispatch | Matched and application-level benchmarks meet the performance gates; ordinary-kernel compatibility path remains covered |
| 7. Release behind explicit use | Document supported workloads, presolve fallback, memory/deadline semantics, and measured regimes | Full regression suite passes; reproducible results and API examples accompany the feature; default single-LP calls remain unchanged |

**Correctness coverage.**

Add `tests/test_shared_matrix.cpp`, `tests/test_metal_spmm.cpp`,
`tests/test_batch_solver.cpp`, and `python/tests/test_batch_bindings.py` as
appropriate to the stages. Register native checks in CTest and retain the
existing Metal-unavailable skip convention. Required cases include:

- Batch widths 0, 1, 2, 3, 4, 5, 8, 9, 16, and 17; partial tiles; zero-row or
  zero-nonzero models; empty rows; duplicate coordinates; mixed/long rows;
  cancellation; and both sides of the 65,536-column index-compression boundary.
- Exact common matrices with varying objectives, offsets, variable and
  constraint bounds; equality constraints; free variables; fixed variables;
  and original-coordinate warm starts with and without reduced costs.
- Different scaling normalizations and original residual denominators;
  presolve requested versus disabled; all optional numerical features either
  honored or explicitly routed to the independent fallback.
- A batch containing easy, slow, infeasible, unbounded, and precision-limited
  LPs; independent restart and step-recovery events; a failed audit; iteration
  and time limits; and preservation of already completed results.
- Reordered batch members, added inactive/padded lanes, repeated submissions,
  caller-buffer mutation, and results that outlive their plan. Under identical
  arithmetic and evaluation schedules, unrelated members must not alter a
  member’s trajectory.
- Native iteration reuse on/off and builds without compatible native batching
  interfaces. Compare complete snapshots and anchors where arithmetic matches.
  When SpMM changes reduction order, use justified numerical tolerances and
  independent original-model audits rather than demanding bitwise equality.

Use analytic small LPs, independent CPU FP64 solves, direct FP64 sparse products,
and `validate_original_problem` as separate checks. For nonunique optima compare
certificates and objectives, not equality of primal vectors. Corrupt one member’s
returned certificate deliberately in validator tests to prove that the audit
checks each supplied result. Infeasibility/ray checks need their own independent
host validation. Final release validation includes the existing CPU and Metal
Netlib regressions, Python bindings tests, and installed-consumer checks.

**Performance protocol and acceptance.**

Add a dedicated `benchmarks/batch_lp_benchmark.cpp` and registered benchmark
target. Replay real compatible families and synthetic common-matrix fixtures
with known feasible points. Include both similar and highly unequal iteration
counts, differing bounds/objectives, repeated submissions, and sequential
dependent traces to expose the limits of concurrency.

Compare three arms: existing independent solves with the application’s current
settings; independent solves reusing the new matrix plan; and native shared-matrix
SpMM solves. Also run a matched-settings no-presolve comparison to isolate
kernel/dispatch improvements. Compare any application concurrency that already
exists. A speedup against repeatedly rebuilding matrices alone is insufficient
evidence for SpMM batching.

Sweep `B=1,2,4,8,16,32,64`, odd tail widths, and selected `K` values within memory
limits. Separate plan creation, packing/upload, PDHG, checkpoints, audits,
correction, queueing, and total request wall time. Report both cold application
time and amortized repeated-plan time, completed-and-audited LPs/second, median
and p95 latency, per-member iterations, active-lane occupancy, fallback reasons,
and peak process/MLX memory. Never score failed or unaudited solves as wins.

Use serial benchmark processes, warmup, interleaved comparison order, and at
least five trials per measured cell. Record hardware, compiler/build flags,
solver and MLX revisions, data identity, random seeds, every solver setting,
and all fallback/correction time. Keep raw JSON/CSV reports. Re-run uncertain
cells when variability is comparable to the claimed benefit; a fixed-work
kernel gain must be confirmed in time to audited solutions.

Proposed release gate: at least 1.2x aggregate throughput versus independent
plan-reusing solves on the qualified real families, an improvement over the
actual application baseline including presolve, and compliance with the
application’s p95 latency and memory limits. Dispatch `B=1` and measured losing
regimes to the independent path, with no more than 5% added median latency for
that route. These are targets to evaluate, not performance claims. Verify on
the development Apple GPU and a second available Apple GPU family before
claiming a portable automatic dispatch rule; otherwise document the measured
hardware scope and keep use explicit.

Completion requires passing workload qualification, the public C++/Python
contract, a real shared-matrix Metal SpMM implementation, independent numerical
control and audited results, bounded storage, explicit fallback behavior, and
the measured release gates. The correctness scaffold, plan reuse alone, or
synthetic kernel timing does not satisfy the complete feature.

Background: [cuOpt applicability review](docs/audits/cuopt-pdlp-2026-09-08.md),
[current architecture](docs/architecture.md), and cuOpt’s
[extended internal PDLP batching](https://github.com/NVIDIA/cuopt/pull/1152).
The internal shared-matrix mechanism is the relevant reference; the deprecated
public cuOpt `BatchSolve` API is not the proposed integration surface.
