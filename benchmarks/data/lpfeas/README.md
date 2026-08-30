# LPfeas public benchmark data

This directory contains the 49 publicly disclosed instances from Hans
Mittelmann's LPfeas benchmark dated 2026-07-20. The benchmark lists 65
instances, but its final 16 are undisclosed.

`manifest.tsv` records the published dimensions, source representation, and
download URL. Run:

```sh
bash benchmarks/data/lpfeas/download.sh
bash benchmarks/data/lpfeas/verify.sh
```

The downloader preserves source archives under `sources/` and streams
conversion to runnable `.mps.gz` files. Sources ending in `.mps.bz2` contain
ordinary MPS; sources without the `.mps` suffix use the Netlib compressed-MPS
encoding and are expanded with EMPS. Plain MPS is never materialized because
some instances contain tens of millions of nonzeros.

`verify.sh` checks each file with mlxPDLP's MPS parser. The published table
mixes dimension conventions independently: a row count may include the
objective row, and a nonzero count may include the objective coefficients.
`dimensions.tsv` records both parts and accepts those conventions.

`online_gpu_reference.csv` transcribes the four GPU columns from the
2026-07-20 [LPfeas table](https://plato.asu.edu/ftp/lpfeas.html) for the 49
public rows. Whole-second values and the source table's `t`, `f`, and `m`
status markers are preserved verbatim for `compare_lpfeas.py`.

The current source archives for `dlr1`, `support10`, and `tpl-tub-ws16` have
dimensions that differ from the benchmark table. A separate streaming scan of
their raw MPS sections confirms the parser's counts. The confirmed differences
are allowlisted in `known_differences.tsv` rather than changing the published
dimensions in `manifest.tsv`.

Run the LPfeas protocol benchmark with practical fp32 tolerance `1e-4` and independent
float64 original-model audit metrics:

```sh
./build/mlxpdlp_lpfeas_benchmark \
  --instance qap15 \
  --output-prefix benchmarks/results/qap15-metal
```

Omit `--instance` to run all 49 public rows. See
[`benchmarks/README.md`](../../README.md) for the exact protocol, report
semantics, and online B200 comparison command.

## Current practical-accuracy bridge

On 2026-08-08, the four smallest progression cases selected for the renewed
LPFeas pass verified at `1e-4` on both backends. Runs used a `5e-5` internal
target, bounded host-double correction, and the audited retry portfolio.

| Instance | MLX/Metal | Sparse CPU | Max audited KKT, Metal | Max audited KKT, CPU |
|---|---:|---:|---:|---:|
| `nug08-3rd` | 0.86 s, `OPTIMAL` | 0.70 s, `OPTIMAL` | `1.002e-7` | `9.919e-8` |
| `qap15` | 4.87 s, `OPTIMAL` | 3.89 s, `OPTIMAL` | `9.928e-7` | `9.920e-7` |
| `fome13` | 15.49 s, `OPTIMAL` | 15.98 s, `OPTIMAL` | `8.314e-7` | `8.806e-7` |
| `cont1` | 508.89 s, audited at time limit | 426.95 s, audited at iteration limit | `7.100e-5` | `7.691e-5` |

`cont1` meets the requested external audit even though it does not cross the
stricter `5e-5` internal stopping target. `Linf_520c` remains intentionally
retired after its requested final attempt, and `psched3-3` remains deferred at
the documented PSLP postsolve-certificate boundary while this smaller bridge
is expanded.

### Geometric-mean default check

On 2026-08-30, a two-tier fixed-protocol A/B selected 12 geometric-mean passes
as the library default. The current easy/well-conditioned accuracy gate ran
`nug08-3rd` and `qap15` at a `1e-5` float64 audit and `5e-6` solver target.
Changing only `geometric_mean_iterations` from 0 to 12 produced identical
iteration counts and audited metrics on both Metal/FP32 and CPU/FP64: 1,200
and 4,400 iterations with the library-default `curtis_reid_iterations = 0`,
or 1,200 and 5,200 with the LPfeas-tuned value 20. All 16 case/configuration
results verified in one attempt, so geometric mean does not regress the
stronger accuracy gate.

The hard-case stress tier used all four bridge cases at a `1e-4` audit and
`5e-5` solver target, with 300 seconds / 500,000 iterations per attempt, 5,000
power-method iterations, and `curtis_reid_iterations = 0`. Every case again
verified on both backends.

- `fome13` dropped from 6,000 to 4,400 Metal iterations and from 6,000 to
  4,600 CPU iterations, although extra host-double cleanup made its total time
  roughly neutral.
- `cont1` retained two audited Metal attempts but reduced reported aggregate
  solve time from 299.2 to 157.0 seconds. On CPU it improved from a timed-out
  first attempt plus a 200,000-iteration warm correction to one verified
  193,600-iteration attempt.

The existing LPfeas-tuned `curtis_reid_iterations = 20` portfolio also
verified all four cases with geometric mean enabled; `cont1` was neutral
(146.1 versus 146.8 seconds). On the default presolved path for `a2864`
(20,078,717 source nonzeros, 1,378,816 reduced nonzeros), geometric mean added
only 0.039 seconds to rescaling. These checks support enabling 12 passes by
default while retaining 0 as the explicit opt-out.

For the stronger normal-case accuracy check, `nug08-3rd` and `qap15` are the
selected well-shaped LPfeas fixtures. On 2026-08-23, Metal FP32 plus the
bounded host-FP64 correction passed both at an independent `1e-5`
original-model audit (`5e-6` internal target):

| Instance | Solve time | Max audited KKT |
|---|---:|---:|
| `nug08-3rd` | 0.17 s | `9.696e-8` |
| `qap15` | 2.89 s | `8.960e-7` |

Reproduce that focused gate with:

```sh
./build/mlxpdlp_lpfeas_benchmark \
  --instance nug08-3rd \
  --instance qap15 \
  --tolerance 1e-5 \
  --solver-tolerance 5e-6 \
  --fail-on-validation
```

Run a fixed-work Metal benchmark with:

```sh
./build/mlxpdlp_mps_benchmark \
  benchmarks/data/lpfeas/qap15.mps.gz \
  50000 100 metal
```

Sources:

- <https://plato.asu.edu/ftp/lpfeas.html>
- <https://plato.asu.edu/ftp/lptestset/>
- <https://miplib.zib.de/>
- <https://miplib2010.zib.de/>
- <https://old.sztaki.hu/~meszaros/public_ftp/lptestset/>
- <https://www.netlib.org/lp/data/emps.c>
