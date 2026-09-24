# OR-Tools PDLP reference

Google's CPU PDHG implementation, pinned to OR-Tools 9.15.6755. Native macOS
ARM64 wheels are available; no CUDA or mlxPDLP build is needed.

```sh
make setup PYTHON=python3.12
make run ARGS="--instance qap15 --threads 1 --tolerance 1e-6"
```

Use `--presolve on` for GLOP presolve. Native feasibility polishing is opt-in
with `--feasibility-polishing`. The runner uses the direct PDLP API with L2
absolute/relative stopping tolerances and records its solve log. MPS parsing
uses the checked ModelBuilder reader; solving always calls PDLP.

See the [shared protocol and comparison commands](../README.md).
[Upstream PDLP](https://github.com/google/or-tools/tree/v9.15/ortools/pdlp)
is Apache-2.0 licensed. Delete this directory to remove this reference.
