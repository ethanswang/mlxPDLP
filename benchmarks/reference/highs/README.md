# HiGHS reference

HiGHS 1.15.1 through `highspy`, using its native macOS ARM64 CPU wheel.
No CUDA, OR-Tools, or mlxPDLP build is needed.

```sh
make setup PYTHON=python3.12
make run ARGS="--instance qap15 --solver hipdlp --threads 1 --tolerance 1e-6"
```

`--solver hipdlp` (default) selects native HiPDLP. `--solver pdlp` selects
bundled cuPDLP-C, which also runs on CPU in this build. `simplex` and `ipm`
are available as additional accuracy references and are labeled separately in
reports. The runner never selects the automatic solver. Crossover is disabled.
`--presolve on` enables HiGHS presolve. Native PDLP and KKT tolerances are set
explicitly; the shared original-model audit determines `verified`.
Maximization costs and offsets are negated before solving, working around
HiPDLP 1.15.1's incorrect handling of the native maximization sense. Reported
audit objectives retain the input sense; native solver details use minimization.

See the [shared protocol and comparison commands](../README.md).
[Upstream solver documentation](https://ergo-code.github.io/HiGHS/stable/solvers/)
describes the algorithms. HiGHS is MIT licensed (see upstream notices for
bundled components). Delete this directory to remove this reference.
