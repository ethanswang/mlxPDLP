"""Solve independent objectives with one immutable CSR matrix plan."""
import numpy as np
import mlxpdlp

parameters = mlxpdlp.Parameters()
parameters.presolve = False
parameters.verbose = False
parameters.tolerance = 1e-4
device = "gpu" if mlxpdlp.has_gpu() else "cpu"
plan = mlxpdlp.SharedMatrixPlan(
    2, 1, np.array([0, 2], dtype=np.int32), np.array([0, 1], dtype=np.int32),
    np.array([1.0, 1.0]), parameters=parameters, device=device,
)
batch = plan.solve_batch(
    objective=np.array([[1., 2.], [2., 1.], [1., 3.], [3., 1.]]),
    variable_lower_bounds=np.zeros(2), variable_upper_bounds=np.ones(2),
    constraint_lower_bounds=np.ones(1), constraint_upper_bounds=np.ones(1),
    execution="shared" if device == "gpu" else "independent",
)
for result in batch.results:
    print(result.input_index, result.termination_reason_name, result.primal_solution)
print(batch.execution, batch.wall_time_sec, "seconds")
