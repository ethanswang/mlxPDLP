# Shared-matrix LP batches

`SharedMatrixPlan` solves independently ready LPs that have **exactly the same
numeric CSR matrix**. Objectives, offsets, row/variable bounds and warm starts
may differ. It owns the original FP64 matrix and prepares scaling, sparse
transpose, row scheduling and norm information once. Submissions reuse that
immutable preparation. CPU execution uses independent FP64 solves; explicit
Metal shared execution uses CSR SpMM and independent PDHG controllers.

Use this when multiple complete requests are already available. A sequence in
which the next objective or bound depends on the preceding result has width one.
`B` denotes independent LPs; `iteration_batch_size` (`K`) denotes consecutive
iterations within each LP. There is no block-diagonal matrix or matrix copy per
member on the shared sparse path.

## Python

```python
import numpy as np
import mlxpdlp

p = mlxpdlp.Parameters()
p.presolve = False
p.verbose = False
p.tolerance = 1e-4
plan = mlxpdlp.SharedMatrixPlan(
    2, 1, np.array([0, 2], np.int32), np.array([0, 1], np.int32),
    np.array([1., 1.]), parameters=p, device="gpu",
)
batch = plan.solve_batch(
    objective=np.array([[1., 2.], [2., 1.], [1., 3.], [3., 1.]]),
    objective_constant=0.,
    variable_lower_bounds=np.zeros(2), variable_upper_bounds=np.ones(2),
    constraint_lower_bounds=np.ones(1), constraint_upper_bounds=np.ones(1),
    execution="shared",
)
for result in batch.results:
    print(result.input_index, result.termination_reason_name, result.primal_solution)
```

The runnable [example](../examples/batch_lp.py) also supports CPU. Numerical input
is copied as FP64; CSR indices use int32. `objective` has shape `[B,n]`, including
`[0,n]` for an empty request. Only bounds (`[n]`/`[m]` or `[B,n]`/`[B,m]`) and the
offset (scalar or `[B]`) broadcast. Omitted bounds are infinite in the appropriate
direction. Lower bounds may equal upper bounds. Shapes such as `[1,n]` do not
broadcast over a larger batch.

`primal_start`, `dual_start` and `reduced_cost_start` have shape `[B,n]`, `[B,m]`
and `[B,n]`, in original coordinates. Each accepts an optional corresponding
`*_mask` containing B booleans. False rows are ignored; true rows must contain
finite values. Starts cannot be combined with presolve. A submission accepts one
common `parameters` snapshot. Matrix-scaling settings, singular-value estimation
settings, conservative norm selection and device are fixed when creating a plan;
changing them requires another plan.

## C++ and ownership

Include `mlxPDLP/batch_solver.h` and link `mlxPDLP::solver`. Construct
`SharedMatrixPlan(n,m,row_ptr,col_indices,values,&parameters,device)`, fill a
`std::vector<BatchProblem>`, and call `plan.solve_batch(problems,options)`.
`BatchProblem` owns its vectors; an empty bound vector means unbounded, and
optional starts are `std::optional<std::vector<double>>`. Each `BatchMemberResult`
contains an owned `mlxpdlp_result_t` with its normal deleter, plus batch metadata.
The [installed consumer](../examples/installed_consumer/main.cpp) demonstrates a
complete C++ batch using only installed public headers.

Plans copy caller CSR buffers and parameters. Requests copy their member data
before waiting for the plan. Changing a caller array after submission cannot
change an active solve. Calls through one plan are serialized, including calls
from different Python threads; the GIL is released during execution. Each request
has fresh mutable solver state. Results retain their own certificates after
another call or destruction of the plan. Moving/destroying a plan concurrently
with a call remains the caller's responsibility.

## Execution and fallback

| Requested mode | Behavior |
|---|---|
| `auto` / `BatchExecution::automatic` | Independent execution with `fallback_reason`. Shared selection is kept explicit because only one Apple GPU family has been measured. |
| `shared` | For B > 1, requires GPU, `presolve=False` and `metal_fused_kernels=True`; otherwise rejects the request. |
| `independent` | Reuses matrix preparation when presolve is disabled; requested presolve uses the existing solver separately for each LP. |
| Any mode with B = 1 | Existing single-LP construction/solve route; reports independent execution. |
| Empty batch | Empty ordered result; no solve is dispatched. |

Presolve is never silently disabled, and no member shares another member's PSLP
reductions or postsolve map. Existing single-LP calls and defaults are unchanged.
All optional numerical controls remain available: PID/HPR restart, reflection,
conditional termination, guarded step recovery, host FP64 feedback, feasibility
polishing and bounded FP64 continuation. Metal iteration uses FP32 and can reach
a precision limit even though the original data is valid FP64.

The driver transfers packed metrics at checkpoints. Restarts, first-step
baselines, best complete certificates and termination belong to each member.
Conditional checks use the union of due times; only due members update their
controller history. Finished lanes freeze while other LPs continue. Original
FP64 primal/bound feasibility, dual feasibility/signs and objective gap must pass
before returning `OPTIMAL`. Failed audits resume or use the selected bounded
correction. Original-model ray audits precede shared-path infeasibility statuses;
their owned vectors are exposed as `primal_ray` or `dual_ray` (empty otherwise).

Shared kernels pack vectors as `[component,B_padded]`, with LP tile width 4 or 8.
Both orientations choose row reduction independently. Padding and inactive LPs
are masked. Optional native scratch reuse supports padded widths up to 256 and
blocks up to 16 minor iterations. `iteration_batch_size` accepts 1–64 and bounds
ordinary lazy evaluation; larger K is split around native, major and checkpoint
boundaries. Other widths, zero-dimensional models, disabled native reuse and
builds without compatible MLX encoder headers use ordinary fused MLX kernels.
`native_iteration_batching_active` reports whether native reuse actually ran.

## Scaling

With positive row/column factors R,C and member-specific bound/objective factors
beta,alpha, the cached operator is `R^-1 A C^-1`. Its norm bound covers the actual
stored coefficients, including FP32 rounding. Matrix-only geometric-mean,
Curtis–Reid, Ruiz and Pock–Chambolle scaling run once, in that order, in FP64.

```text
row bounds     = beta R^-1 original row bounds
variable bounds= beta C original variable bounds
objective      = alpha C^-1 original objective
x_original     = C^-1 x / beta
y_original     = R^-1 y / alpha
z_original     = C z / alpha
```

Warm starts use the inverse maps. Offsets enter objective audits and never enter
projections. Original residual denominators and alpha/beta remain per member.
Infinite bounds are handled without multiplying infinity by a zero mask.

## Deadlines, memory and results

`time_sec_limit` on a batch starts at request entry and includes copying,
queueing, construction, iteration, audit and correction. The solver parameter's
time allowance starts at member admission and is bounded by the batch deadline.
Time is checked before dispatch and correction stages and at bounded iteration
blocks. Final extraction/audit can overrun a deadline; `deadline_overrun_sec`
measures that overhead. It is not a real-time interrupt guarantee.

Every input gets a result in input order. A queued member that cannot start has
`has_solution=False`, `TIME_LIMIT`, and empty Python certificate arrays (null
C++ pointers). An admitted numerical failure has `has_solution=True` and its
own failure status. Invalid input rejects the whole request before dispatch and
identifies the member/field. An unrecoverable device exception fails the request.

`resident_memory_budget_bytes=0` means unconstrained. A positive budget selects
bounded resident groups using a conservative estimate of the shared plan,
copied request, all retained results, padded workspace, snapshots, bounded K
graphs, correction work and allocator headroom. Requests too small to hold one
workspace are rejected. Storage grows as one matrix plus `O(B*(n+m))`; previous
results and complete request data count toward the estimate even when execution
is grouped. This is a per-request allocation estimate, **not an operating-system
RSS cap**: other MLX users, process-wide caches and driver allocations are outside
its control. Inspect measured process/MLX peaks for a deployment memory limit.

`queue_time_sec` and `execution_time_sec` are per member; their sum is that
member's completion latency. Shared execution times overlap, so do not sum them
to obtain batch duration. `wall_time_sec` includes the complete request (and
Python result conversion). Batch phase timers expose construction,
initialization, packing/upload, PDHG, checkpoints and audits. Checkpoint time
includes audits/corrections done there; audit time includes final extraction and
correction. These nested timers are not additive. Each member also retains the
existing rescaling and polishing timings. `groups`, `max_active_width`, and
`estimated_peak_resident_bytes` explain memory grouping.

Batch members expose `step_size_reductions` and `original_audit_failures` for
diagnosing numerical recovery and shared-path certificate rejection. These
counters belong to the member and cannot be inferred from its neighbors.

## Validation and measured scope

The benchmark protocol and raw reports compare presolved existing solves, matched
no-presolve solves, independent plan reuse and shared SpMM. Qualification uses
measured Netlib/LPfeas-derived families
under the user's September 10 policy, with no caller latency or memory limits.
It does not establish a whole alpha-beta-CROWN application speedup. Measurements
cover Apple M3 Max; no portable automatic dispatch rule is claimed.

Release measurements include throughput, member latency, peak memory and losing
regimes. The SC50A derived family passes qualification and achieves 2.44× audited
throughput over independent plan reuse. Benchmark results do not select shared
execution for you; use it explicitly after checking your own ready workload.
