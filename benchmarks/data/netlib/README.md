# Netlib benchmark data

`download.sh` fetches and expands a 40-instance progression of small-to-medium
Netlib LPs in `manifest.tsv`. The compact sources
come directly from Netlib and are expanded with Netlib's EMPS utility. Published
optimal objectives are recorded separately in `reference_objectives.tsv`.

Run the practical-accuracy audit suite with, for example:

```sh
./benchmarks/data/netlib/download.sh
./build/mlxpdlp_lpfeas_benchmark \
  --data benchmarks/data/netlib \
  --manifest benchmarks/data/netlib/manifest.tsv \
  --jobs auto \
  --tolerance 1e-4 \
  --output-prefix benchmarks/results/netlib-metal \
  --fail-on-validation
```

The benchmark runner requires both its independent float64 original-model KKT
certificate and agreement with the published objective at the requested
tolerance. This catches model-import errors that could otherwise produce a
self-consistent certificate for the wrong LP.

## Current audited result

On 2026-08-09, mlxPDLP 0.1.0 on an Apple M3 Max (`Mac15,9`, 16 logical CPUs)
verified all 40 instances at `1e-4` on both backends. The runner used a `5e-5`
internal target, safe PSLP defaults, bounded host-double correction, and
the cold-trajectory stagnation handoff described in
[the benchmark protocol](../../README.md#lpfeas-protocol-benchmark).

Both backends ran the complete manifest through the dynamic work-stealing
runner (12 Metal workers, two CPU workers). The table reports worst-case
audited accuracy across all 40 instances; runtimes are intentionally
omitted — Netlib is the regression corpus, not a performance showcase.

| Backend | Verified | Max primal | Max dual | Max gap | Max published-objective error |
|---|---:|---:|---:|---:|---:|
| MLX/Metal FP32 | 40/40 | `4.348e-5` | `4.034e-5` | `2.836e-5` | `6.256e-5` |
| MLX/CPU FP64 | 40/40 | `9.910e-5` | `9.437e-5` | `4.935e-5` | `6.504e-5` |

Complete machine-readable reports are regenerable local artifacts under
`benchmarks/results/` and are not distributed with this repository.

The expanded `boeing1` file contains 352 ROWS records (one objective plus 351
constraints), although Netlib's summary table lists 351 rows. Its column and
nonzero totals match the catalog; `manifest.tsv` records the expanded file's
actual row count so import audits remain reproducible.

## PILOT87

`pilot87.mps.gz` is the Netlib PILOT87 linear-programming benchmark. It was
expanded from Netlib's compact MPS representation with Netlib's `emps.c`
utility, then reproducibly compressed with `gzip -9 -n`.

- Compact source: https://www.netlib.org/lp/data/pilot87
- Expander: https://www.netlib.org/lp/data/emps.c
- Netlib metadata: https://www.netlib.org/lp/data/readme
- Published dimensions: 2,031 MPS rows, 4,883 columns, 73,804 nonzeros
- Published optimal objective: `3.0171072827E+02`
- Compact source SHA-256:
  `b7c9fa4a97f9503c7b44faf96445d296eddb18ed5fc1eddac8ca47fb1f21489f`
- `emps.c` SHA-256:
  `fee41f544f6873a5e12bc598947828dc9964ef0676162e4df55e915760e2be22`
- Expanded MPS SHA-256:
  `88f84ca1ffae4a0b604cea5b6aa29dbe9512e9a32d4d6c70431d55fa14f25e01`
- Checked-in gzip SHA-256:
  `12e78dd4e0ec554ee8b5ed309c9e077305e17c25ede487366f5f4629e8583c69`

Netlib's row and nonzero counts include the objective row. The mlxPDLP parser
exposes 2,030 constraints and 73,152 constraint-matrix nonzeros.

With the same `1e-4` audit / `5e-5` internal-target protocol used above,
PILOT87 passed on both backends. Concurrent per-case times include contention
from the other active suite workers:

| Backend | Solve time | Primal | Dual | Gap | Published-objective error |
|---|---:|---:|---:|---:|---:|
| MLX/Metal FP32 | 82.85 s | `9.283e-6` | `9.878e-7` | `3.693e-6` | `1.549e-5` |
| MLX/CPU FP64 | 94.20 s | `1.602e-6` | `1.281e-6` | `6.599e-7` | `9.953e-6` |

Reproduction:

```sh
curl -fsSLo emps.c https://www.netlib.org/lp/data/emps.c
curl -fsSLo pilot87 https://www.netlib.org/lp/data/pilot87
cc -O2 -o emps emps.c
./emps < pilot87 > pilot87.mps
gzip -9 -n pilot87.mps
```
