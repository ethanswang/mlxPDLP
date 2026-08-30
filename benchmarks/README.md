# Performance benchmarks

Two benchmark modes are available:

- `mlxpdlp_mps_benchmark` compares CPU and Metal using identical fixed work.
- `mlxpdlp_lpfeas_benchmark` runs provenance-tracked LPfeas or Netlib manifests
  to convergence and writes independently audited CSV and JSON reports.

Configure and build it with:

```sh
cmake -S . -B build \
  -DMLX_BUILD_DIR=/absolute/path/to/mlx/build \
  -DMLXPDLP_BUILD_BENCHMARKS=ON
cmake --build build --target \
  mlxpdlp_mps_benchmark mlxpdlp_lpfeas_benchmark \
  --parallel
```

## LPfeas protocol benchmark

Fetch the 49 publicly disclosed LPfeas instances, then run one instance or
the entire manifest:

```sh
benchmarks/data/lpfeas/download.sh

./build/mlxpdlp_lpfeas_benchmark \
  --instance qap15 \
  --output-prefix benchmarks/results/qap15-metal

./build/mlxpdlp_lpfeas_benchmark \
  --output-prefix benchmarks/results/lpfeas-metal
```

The default protocol uses Metal CSR in FP32, PSLP 0.0.8, L2 residuals,
12 geometric-mean iterations, 10 Ruiz iterations, Pock-Chambolle alpha 1,
bound/objective scaling, evaluation frequency 200, up to 5,000 power-method
iterations, a 1,000-second per-attempt limit, and a practical `1e-4`
convergence tolerance. The iteration limit is otherwise unlimited. Each worker
warms the sparse Metal path and both adaptive row-kernel branches before the
sweep clock starts; pass `--cold-start` to include first-use compilation.
The internal stopping target defaults to `0.5 * tolerance` (`5e-5`) to leave
rounding margin for the original-model audit; override it with
`--solver-tolerance` when reproducing an exact trajectory.
On macOS, CPU problems whose dense matrix would contain at least 16,777,216
elements and whose density is at most 25% use Accelerate FP64 sparse SpMV.
Smaller CPU problems keep an FP64 MLX dense path. Metal remains FP32 in both
its sparse and dense paths. Schema-v7 reports expose `arithmetic_precision`,
`cpu_double_precision`, worker/completion order, scheduling policy, and sweep
wall time in addition to `sparse_cpu` and `sparse_metal`.
The optional primal/dual feasibility phases use a tighter internal `1e-6`
target to leave rounding margin for the original-model audit; pass/fail is
still determined by the requested `1e-4` tolerance.

The runner also enables a bounded host-double correction (50,000 iterations,
30 seconds by default). It runs after original-model certificate
recomputation, accepts only a lower full-KKT merit, and reports its time and
iterations separately. A primal-only near miss first follows an FP64 descent
direction for the actual unscaled feasibility norm, projected orthogonally to
the objective so correction does not sacrifice the existing gap certificate.
`--no-feasibility-polishing` disables the backend primal/dual phases without
disabling this host correction; use
`--no-host-double-polishing` to disable the latter explicitly. Its independent
caps can be changed with `--host-double-iteration-limit` and
`--host-double-time-limit`.

Cold Metal FP32 trajectories may hand a saved checkpoint to host double before the
main time limit, but only after all of these conditions hold:

- primal residual is at most `max(5e-3, 50 * eps_feasible)`;
- dual residual is finite and at most one;
- objective gap is at most half the internal optimality tolerance;
- the host budget can complete at least one evaluation block; and
- no twofold KKT improvement has occurred for 25 evaluation blocks.

The admissible checkpoint is retained across later FP32 oscillation. A
successful correction rewrites the intermediate `HOST_DOUBLE_HANDOFF` status
to `OPTIMAL`; otherwise the distinct status and `host_double_handoff` report
field remain visible. Disable the mechanism with
`--no-host-double-handoff`. Warm-start correction attempts do not use early
handoff: they run to their own stopping criterion.

If the independent audit rejects an early presolved handoff, the runner first
matures that same presolved trajectory with handoff disabled. When the best
postsolved dual residual or gap is a gross miss (above
`max(1e-2, 100 * audit tolerance)`), safe PSLP with propagation disabled is
tried before reusing that point as an original-model warm start. Otherwise the
warm correction remains first, followed by safe PSLP and finally no presolve.
This keeps badly mapped points such as FINNIS/Metal out of a long correction
trajectory without penalizing near-target cases such as FORPLAN. The warm
correction is capped independently at
200,000 iterations / 300 seconds. The runner stops at the first audited
solution meeting the requested tolerance; if none passes, it reports the
candidate with the lowest audited KKT merit. Disable stages with
`--no-warm-start-correction`, `--no-propagation-retry`,
`--no-presolve-retry`, or `--no-scaling-retry` when reproducing one exact
trajectory.

Every returned solution is also rechecked against the original, unscaled MPS
model on the CPU in float64. This audit recomputes `Ax`, `A^T y`, primal and
dual objectives, variable-bound violations, and the CUDA-normalized gap. It
is deliberately report-only by default. `OPTIMAL` means the solver's own
original-model certificate met its requested tolerances, while `verified`
records whether the benchmark runner's independent float64 metrics met the
requested audit tolerance. Because the solver target is stricter by default,
an audited `1e-4` solution can legitimately have
`verified: true` with an internal `TIME_LIMIT` or `ITERATION_LIMIT` status.
Pass
`--fail-on-validation` when CI should return nonzero for an audit failure.

`--jobs auto` deliberately uses two suite policies. LPfeas runs serially so
each hard model has attributable convergence and timing diagnostics. Netlib is
the first-class parallel regression corpus: work is seeded in descending
structural-cost order, distributed across per-worker deques, and stolen from
the back of another queue whenever a worker becomes idle. Reports remain in
manifest order while `worker_id` and `completion_order` preserve the actual
execution history. Netlib automatic concurrency is memory-bounded; it retains
two host cores and permits up to 12 Metal streams, while CPU is capped at two
workers because MLX and Accelerate already parallelize each FP64 solve. Use
`--jobs N` for an explicit experiment on either corpus and `--jobs 1` for a
sequential reproduction.
On the 12-case calibration subset, Metal sweep times for 1/2/4/8/12 workers
were 11.67/5.82/4.32/2.83/2.36 seconds; CPU times for 1/2/4/8 workers were
2.25/1.68/2.01/2.77 seconds. Every calibration audit passed.

Parsing, setup/presolve, rescaling, feasibility polishing, host-double
correction, solve, final verification, and total wall times are reported
separately. The JSON report also records the host, solver version, complete
protocol, selected worker count, and suite wall time. CSV and JSON are
atomically republished after each completion, so a long suite run retains
valid reports for completed rows if it is interrupted.

## Staged LPfeas sweeps and result comparison

`run_lpfeas_staged.sh` runs the 23 LPfeas instances that have recorded
pre-geometric Metal/FP32 baselines in three serial stages, ordered by the last
recorded wall time:

```sh
benchmarks/run_lpfeas_staged.sh easy    # ~22 min serial
benchmarks/run_lpfeas_staged.sh medium  # ~2 h serial
benchmarks/run_lpfeas_staged.sh hard    # ~21 h serial
```

Each instance runs with `--jobs 1` under the current protocol (1e-4 audit
tolerance, 5e-5 solver target, 1,200 s per-attempt limit, geometric-mean 12,
Curtis-Reid 20, 5,000 power-method iterations, host-double correction at
180 s / 300,000 iterations) and writes
`results/lpfeas-staged-geomean12/<name>.csv/json`. Keeping the geometric-mean
results in a distinct default directory prevents old `geo=0` files from being
mistaken for completed `geo=12` runs. Existing
results are skipped, so an interrupted stage resumes where it left off. The
default 120 s cooldown between instances (`COOLDOWN_SECONDS`) matters:
back-to-back serial sweeps thermally throttle laptop GPUs — degme measured
1073 s hot versus 422 s cool on identical code — and throttled wall times
cannot be compared with recorded ones.

Compare a staged run against the recorded baselines with:

```sh
python3 benchmarks/compare_lpfeas_results.py \
  results/lpfeas-staged-geomean12
```

It pairs each result with the newest recorded `lpfeas-<name>-metal-fp32.json`
and prints verified status, termination, iteration counts, and solve times
with percentage deltas. Iteration counts and verified status are the robust
regression axes; treat solve-time deltas below ~30% as environmental noise
unless a cooled re-run reproduces them.

Compare measured solve times to one of the four published NVIDIA B200 GPU
columns with:

```sh
python3 benchmarks/compare_lpfeas.py \
  benchmarks/results/lpfeas-metal.csv \
  --reference cupdlpx
```

The bundled reference table is dated 2026-07-20. Its values are whole-second
published times (`t`, `f`, and `m` are preserved as statuses), so ratios near
one second are low precision. Hardware is also fundamentally different: the
online GPU runs use an NVIDIA B200 with 192 GiB, whereas mlxPDLP uses Apple
Metal and unified memory. Treat the comparison as cross-system context, not a
same-hardware speedup.

The `1e-4` default is intentional: Apple Silicon GPUs do not expose FP64
arithmetic, so the Metal backend runs FP32 and the portable success
criterion is capped at the practical `1e-4` tolerance (`5e-5` internal
target). FP32 PDHG is reliable at low-to-moderate accuracy, while some
ill-conditioned models stagnate or take a different adaptive-restart
trajectory below that scale. CPU FP64 is available as the
reliability/reference backend and may support tighter targets, but uses the
same practical default for suite comparisons. `--tolerance 1e-5` remains a
useful stronger opt-in target and is independently audited, but is not the
portable success criterion for the suite.

## Netlib convergence regression

After downloading the corpus, register the full CPU and Metal suites with:

```sh
cmake -S . -B build \
  -DMLX_BUILD_DIR=/absolute/path/to/mlx/build \
  -DMLXPDLP_BUILD_BENCHMARKS=ON \
  -DMLXPDLP_ENABLE_NETLIB_REGRESSION=ON
cmake --build build --target mlxpdlp_netlib_regression
```

This opt-in target runs all 40 cases at an independently audited `1e-4`, with
a `5e-5` backend target, 500,000-iteration main cap, bounded retry/correction
portfolio, and `--fail-on-validation`. CTest serializes the two backend-level
tests; each benchmark process performs its own dynamic scheduling.

Measured on 2026-08-09 on the M3 Max host: all 40 cases verify at `1e-4`
on both backends (Metal FP32 and CPU FP64). Runtime comparisons are
intentionally omitted here — Netlib is the regression corpus for
accuracy, not a performance showcase (see the LPfeas fixed-work
measurements above for Metal-vs-CPU throughput). Machine-readable
reports are regenerable local artifacts under `benchmarks/results/`
and are not distributed with this repository.

## LPfeas verification status (2026-08-10)

Metal FP32 at an independently audited `1e-4` is the primary success criterion
(the Metal-first policy: CPU FP64 is run only to trace Metal failures). As of
2026-08-12, **42 of the 49 public LPfeas instances are verified on Metal
FP32** at `1e-4` (5 failed on Metal and are recorded as such; `Linf_520c` is
retired and `psched3-3` is deferred). Verbatim machine-readable reports are regenerable local
artifacts under `benchmarks/results/` and are not distributed; the table below
summarizes the current state. `attempts` counts the audited retry portfolio
(warm-start correction, safe-PSLP propagation retry, host-double correction).

| Instance | Termination | Attempts | Metal iterations |
|---|---:|---:|---:|
| qap15 | OPTIMAL | 1 | 3,200 |
| nug08-3rd | OPTIMAL | 1 | 1,200 |
| fome13 | OPTIMAL | 1 | 6,600 |
| cont1 | OPTIMAL | 2 | 275,600 |
| cont11 | HOST_DOUBLE_HANDOFF | 1 | 5,600 |
| irish-e | HOST_DOUBLE_HANDOFF | 2 | 149,600 |
| support10 | OPTIMAL | 2 | 3,000 |
| stp3d | OPTIMAL | 1 | 8,800 |
| rail02 | OPTIMAL | 1 | 64,000 |
| rmine15 | OPTIMAL | 1 | 14,800 |
| shs1023 | OPTIMAL | 4 | 43,000 |
| neos | ITERATION_LIMIT | 2 | 200,000 |
| ex10 | OPTIMAL | 1 | 600 |
| graph40-40 | OPTIMAL | 1 | 400 |
| pds-100 | OPTIMAL | 2 | 1,200 |
| datt256 | OPTIMAL | 1 | 400 |
| neos3 | OPTIMAL | 1 | 3,000 |
| s250r10 | OPTIMAL | 1 | 21,600 |
| savsched1 | OPTIMAL | 1 | 1,000 |
| neos-5251015 | OPTIMAL | 1 | 3,200 |
| s100 | OPTIMAL | 1 | 29,400 |
| woodlands09 | OPTIMAL | 1 | 800 |
| stat96v2 | OPTIMAL | 1 | 55,400 |
| stormG2_1000 | OPTIMAL | 2 | 3,000 |
| L1_sixm250obs | OPTIMAL | 1 | 470,200 |
| tpl-tub-ws16 | OPTIMAL | 1 | 19,600 |
| neos5052403 | OPTIMAL | 1 | 9,800 |
| Primal2_1000 | OPTIMAL | 1 | 12,000 |
| scpm1 | OPTIMAL | 1 | 11,600 |
| s82 | OPTIMAL | 2 | 217,600 |
| degme | OPTIMAL | 2 | 14,800 |
| neos-3025225 | OPTIMAL | 1 | 33,400 |
| square41 | OPTIMAL | 2 | 255,600 |
| a2864 | OPTIMAL | 1 | 1,600 |
| fhnw-bin1 | OPTIMAL | 5 | 151,400 |
| rail4284 | OPTIMAL | 2 | 24,800 |
| set-cover | OPTIMAL | 1 | 18,600 |
| thk_63 | OPTIMAL | 2 | 14,000 |
| L1_sixm1000obs | OPTIMAL | 3 | 165,800 |
| L2CTA3D | OPTIMAL | 1 | 600 |
| thk_48 | OPTIMAL | 1 | 20,800 |
| Dual2_5000 | OPTIMAL | 1 | 5,400 |

`HOST_DOUBLE_HANDOFF` and `ITERATION_LIMIT` rows are still counted as verified
because the independent float64 original-model audit met `1e-4`. Failed on
Metal FP32 at `1e-4`: `bdry2` (6 attempts), `dlr1` (8 attempts; dual-side
blocked), `dlr2` (6 attempts; dual-side blocked), `ns1687037` (7
attempts; gap stalled near `9e-4`), `ns1688926` (8 attempts; dual residual
large). `bdry2`, `ns1687037`, and `ns1688926` are also unsolved by cuPDLPx on
the B200 reference (timeout), so their Metal failure is expected; `dlr1` and
`dlr2` are labeled failed for now and are candidates for future solver work
on dual-side convergence.

## Fixed-work CPU/Metal benchmark

Current equal-work reference measurements on two large sparse LPfeas
instances (cold machine, presolve off, identical PDHG iteration
counts, mlxPDLP 0.1.0, M3 Max 16-core/64 GB, 2026-08-30):

| Instance | Iterations | CPU FP64 solve | Metal FP32 solve | Solve speedup | Per-iteration CPU / Metal |
|---|---:|---:|---:|---:|---:|
| `s82` (7.0M nnz) | 20,000 | 200.4 s | 32.6 s | 6.15x | 10.02 ms / 1.63 ms |
| `dlr1` (18.4M nnz) | 10,000 | 572.5 s | 26.8 s | 21.39x | 57.25 ms / 2.68 ms |

Both backends stay in CSR for preprocessing and iteration (the dense
`A + A^T` payload would be 1,108 GiB for `s82` and 118 TiB for `dlr1`;
CSR + transpose CSR is 114 MiB / 322 MiB). Setup plus preconditioning
(geometric mean, Ruiz, Pock-Chambolle, 100 power-method iterations) costs <= 2.5 s on
either backend, so the entire gap lives in the iteration loop. The
Metal advantage combines half the traffic (FP32 vs FP64) with higher
memory bandwidth: effective SpMV bandwidth is 34.5 vs 11.2 GB/s on
`s82` and 55.0 vs 5.1 GB/s on `dlr1`. Both backends execute the same
fixed iteration count, so this measures equal-work throughput rather
than convergence timing.

Run the PILOT87 workload (fetch the instance data first — it is not
distributed in this repository):

```sh
./build/mlxpdlp_mps_benchmark \
  benchmarks/data/netlib/pilot87.mps.gz \
  150000 100 both
```

Arguments after the path are the iteration limit, termination-evaluation
frequency, and device selection (`cpu`, `gpu`/`metal`, or `both`).

The harness disables presolve and sets convergence tolerances to zero. Both
devices therefore perform the same fixed iteration count and terminate with
`ITERATION_LIMIT` by design. CPU uses FP64 while Metal uses FP32, so this is an
equal-work comparison of the production backends rather than bit-identical
arithmetic. Wall time includes preconditioning, singular
value estimation, PDHG iterations, residual extraction, and synchronization.
MPS parsing occurs before the timed device sections.

## Data

The LPfeas public suite (about 3.8 GB expanded) and the 40-instance Netlib
progression are fetched and expanded by provenance-tracked scripts:

```sh
benchmarks/data/lpfeas/download.sh
benchmarks/data/netlib/download.sh
```

The Netlib progression includes PILOT87; its checksums and a manual reproduction
recipe are in [data/netlib/README.md](data/netlib/README.md).

## Reference result

The fixed-work reference measurements are the LPfeas table at the top of
this section (`s82` 6.15x, `dlr1` 21.39x solve speedup, 2026-08-30).
The small Netlib corpus remains available for smoke-testing the
benchmark executable, but Netlib measurements are reserved for the
regression suite rather than performance claims.

Remaining optimization opportunities include donated custom-kernel output
buffers and reducing periodic fixed-point evaluation work.
