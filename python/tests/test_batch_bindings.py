"""Public batch contract, independently known optima, and owned certificates."""
import gc
import numpy as np
import pytest
import mlxpdlp


def parameters():
    p = mlxpdlp.Parameters()
    p.presolve = False
    p.verbose = False
    p.tolerance = 1e-4
    p.iteration_limit = 5000
    p.time_limit_seconds = 10
    p.termination_evaluation_frequency = 50
    return p


def plan(device="gpu", p=None):
    return mlxpdlp.SharedMatrixPlan(2, 1, np.array([0, 3], np.int32),
        np.array([0, 0, 1], np.int32), np.array([0.5, 0.5, 1.0]),
        parameters=p or parameters(), device=device)


def request(width):
    return dict(objective=np.tile([1.0, 2.0], (width, 1)),
                objective_constant=np.arange(width, dtype=float) / 4,
                variable_lower_bounds=np.zeros(2), variable_upper_bounds=np.ones(2),
                constraint_lower_bounds=np.ones(1), constraint_upper_bounds=np.ones(1))


@pytest.mark.parametrize("device", ["cpu", "gpu"])
@pytest.mark.parametrize("width", [0, 1, 2, 3, 4, 5, 8, 9, 16, 17])
def test_widths_order_and_independent_certificates(device, width):
    result = plan(device).solve_batch(**request(width), execution="shared" if device == "gpu" else "independent")
    assert len(result.results) == width
    for j, member in enumerate(result.results):
        assert member.input_index == j and member.has_solution
        assert member.termination_reason_name == "OPTIMAL"
        np.testing.assert_allclose(member.primal_solution, [1, 0], atol=5e-4)
        assert abs(member.primal_objective_value - 1 - j/4) < 5e-4
        assert abs(member.primal_solution.sum() - 1) < 5e-4
        assert np.linalg.norm(np.array([1., 2.]) - member.dual_solution[0] - member.reduced_cost) < 1e-3


def test_ownership_permutation_repeated_submissions_and_partial_warm_starts():
    p = parameters()
    values = np.array([1., 1.])
    shared = mlxpdlp.SharedMatrixPlan(2, 1, np.array([0, 2], np.int32), np.array([0, 1], np.int32), values,
                                    parameters=p, device="gpu")
    values[:] = 100
    p.presolve = True
    args = request(5)
    args["objective"][1::2] = [3, 1]
    warm = np.tile([0.5, 0.5], (5, 1)); warm[1::2] = np.nan
    result = shared.solve_batch(**args, primal_start=warm, primal_start_mask=[True, False, True, False, True], execution="shared")
    saved = [r.primal_solution.copy() for r in result.results]
    order = [4, 2, 3, 0, 1]
    again = shared.solve_batch(**{**args, "objective": args["objective"][order], "objective_constant": args["objective_constant"][order]}, execution="shared", lp_tile_width=8, iteration_batch_size=1)
    for i, j in enumerate(order): np.testing.assert_allclose(again.results[i].primal_solution, saved[j], atol=5e-4)
    args["objective"][:] = np.nan
    del shared; gc.collect()
    for i, r in enumerate(result.results): np.testing.assert_array_equal(r.primal_solution, saved[i])


def test_validation_deadlines_empty_and_fallback():
    shared = plan()
    args = request(5)
    for field, value in [("objective", np.ones(2)), ("variable_lower_bounds", np.zeros((1, 2))),
                         ("constraint_upper_bounds", np.ones((5, 2))), ("objective_constant", np.zeros((5, 1)))]:
        with pytest.raises(ValueError, match=field): shared.solve_batch(**{**args, field: value})
    invalid = args["objective"].copy(); invalid[-1, 0] = np.nan
    with pytest.raises(ValueError, match=r"member\[4\].objective"):
        shared.solve_batch(**{**args, "objective": invalid}, execution="shared")
    expired = shared.solve_batch(**args, time_sec_limit=0, execution="shared")
    for r in expired.results:
        assert not r.has_solution and r.termination_reason_name == "TIME_LIMIT"
        assert r.primal_solution.size == r.dual_solution.size == r.reduced_cost.size == 0
    with pytest.raises((ValueError, RuntimeError), match="resident_memory_budget_bytes"):
        shared.solve_batch(**args, resident_memory_budget_bytes=1)
    p = parameters(); p.presolve = True
    with pytest.raises(ValueError, match="presolve"):
        shared.solve_batch(**args, parameters=p, execution="shared")
    fallback = shared.solve_batch(**args, parameters=p)
    assert fallback.execution == "independent" and "presolve" in fallback.fallback_reason
    p = parameters(); p.l_inf_ruiz_iterations += 1
    with pytest.raises(ValueError, match="preparation"):
        shared.solve_batch(**args, parameters=p)


def test_mixed_infeasible_unbounded_and_finished_members():
    p = parameters(); p.termination_criteria.eps_infeasible_relative = 1e-7
    shared = mlxpdlp.SharedMatrixPlan(1, 1, np.array([0, 1], np.int32), np.array([0], np.int32),
                                    np.ones(1), parameters=p, device="gpu")
    result = shared.solve_batch(objective=np.array([[1.], [1.], [-1.]]),
        variable_lower_bounds=np.zeros((3, 1)), variable_upper_bounds=np.array([[1.], [1.], [np.inf]]),
        constraint_lower_bounds=np.array([[0.], [2.], [0.]]), constraint_upper_bounds=np.full((3, 1), np.inf),
        execution="shared")
    assert [r.termination_reason_name for r in result.results] == ["OPTIMAL", "PRIMAL_INFEASIBLE", "DUAL_INFEASIBLE"]
    assert result.results[0].primal_solution[0] == 0
    # Independent original-coordinate Farkas/recession checks.
    assert result.results[1].dual_ray[0] > 0
    assert 2*result.results[1].dual_ray[0] - result.results[1].dual_ray[0] > 0
    assert result.results[2].primal_ray[0] > 0
    assert -result.results[2].primal_ray[0] < 0


@pytest.mark.parametrize("rows", [0, 1])
def test_zero_operator(rows):
    shared = mlxpdlp.SharedMatrixPlan(2, rows, np.zeros(rows+1, np.int32), np.empty(0, np.int32),
                                    np.empty(0), parameters=parameters(), device="gpu")
    result = shared.solve_batch(objective=np.array([[1., -1.], [-1., 1.]]),
                               variable_lower_bounds=np.zeros(2), variable_upper_bounds=np.ones(2),
                               execution="shared")
    assert all(r.termination_reason_name == "OPTIMAL" for r in result.results)
    np.testing.assert_allclose(result.results[0].primal_solution, [0, 1], atol=1e-5)


def test_native_buffer_reuse_matches_ordinary_kernels_and_memory_groups():
    shared = plan()
    args = request(9)
    args["objective"][1::2] = [3, 1]
    p = parameters(); p.metal_iteration_batching = False
    ordinary = shared.solve_batch(**args, parameters=p, execution="shared", iteration_batch_size=8)
    p.metal_iteration_batching = True
    native = shared.solve_batch(**args, parameters=p, execution="shared", iteration_batch_size=8)
    assert not ordinary.native_iteration_batching_active
    for left, right in zip(ordinary.results, native.results):
        assert left.total_count == right.total_count
        np.testing.assert_allclose(left.primal_solution, right.primal_solution, atol=1e-7)
        np.testing.assert_allclose(left.dual_solution, right.dual_solution, atol=1e-7)
        np.testing.assert_allclose(left.reduced_cost, right.reduced_cost, atol=1e-7)
    budget = shared.resident_bytes + 35000
    grouped = shared.solve_batch(**args, execution="shared", resident_memory_budget_bytes=budget, iteration_batch_size=1)
    assert grouped.groups > 1 and grouped.max_active_width <= 8
    assert grouped.estimated_peak_resident_bytes <= budget
    for j, r in enumerate(grouped.results):
        assert r.input_index == j and r.termination_reason_name == "OPTIMAL"


@pytest.mark.parametrize("policy", [0, 1])
def test_scaled_warm_certificates_bounds_and_controller_options(policy):
    p = parameters(); p.restart_policy = policy
    p.curtis_reid_iterations = 5
    p.bound_objective_rescaling = True
    p.host_double_residual_evaluation = True
    p.host_double_polishing = True
    # Diagonal A and varying fixed coordinates give exact x/y/z certificates.
    diagonal = np.array([1e-5, 1e5, 3.])
    shared = mlxpdlp.SharedMatrixPlan(3, 3, np.arange(4, dtype=np.int32), np.arange(3, dtype=np.int32),
                                    diagonal, parameters=p, device="gpu")
    x = np.array([[1., 2., 3.], [4., 5., 6.], [0., 1., -1.], [-2., 0., 2.]])
    objectives = np.array([[1., 2., 3.], [-1., 1., 2.], [0., -1., 1.], [2., 3., -1.]])
    rhs = x * diagonal
    result = shared.solve_batch(objective=objectives, constraint_lower_bounds=rhs, constraint_upper_bounds=rhs,
        primal_start=x, dual_start=objectives/diagonal, reduced_cost_start=np.zeros_like(x), execution="shared")
    for j, r in enumerate(result.results):
        assert r.termination_reason_name == "OPTIMAL"
        np.testing.assert_allclose(r.primal_solution, x[j], atol=3e-3)
        assert abs(r.primal_objective_value - objectives[j] @ x[j]) < 3e-3


def test_zero_variables_infeasibility_and_no_row_unboundedness():
    p = parameters(); p.termination_criteria.eps_infeasible_relative = 1e-7
    empty = mlxpdlp.SharedMatrixPlan(0, 1, np.zeros(2, np.int32), np.empty(0, np.int32), np.empty(0), parameters=p, device="gpu")
    results = empty.solve_batch(objective=np.empty((2, 0)), constraint_lower_bounds=np.array([[0.], [1.]]),
                               execution="shared").results
    assert [r.termination_reason_name for r in results] == ["OPTIMAL", "PRIMAL_INFEASIBLE"]
    assert results[1].dual_ray[0] > 0
    no_rows = mlxpdlp.SharedMatrixPlan(1, 0, np.zeros(1, np.int32), np.empty(0, np.int32), np.empty(0), parameters=p, device="gpu")
    results = no_rows.solve_batch(objective=np.array([[-1.], [1.]]), variable_lower_bounds=np.zeros(1), execution="shared").results
    assert [r.termination_reason_name for r in results] == ["DUAL_INFEASIBLE", "OPTIMAL"]


def test_serialized_submissions_and_owned_results_from_threads():
    from concurrent.futures import ThreadPoolExecutor
    shared = plan()
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(shared.solve_batch, **request(5), execution="shared") for _ in range(2)]
        results = [f.result() for f in futures]
    for batch in results:
        assert all(r.termination_reason_name == "OPTIMAL" for r in batch.results)
    np.testing.assert_array_equal(results[0].results[0].primal_solution, results[1].results[0].primal_solution)


def test_nonrepresentable_member_does_not_poison_finished_neighbor():
    p = parameters(); p.iteration_limit = 100
    shared = mlxpdlp.SharedMatrixPlan(1, 1, np.array([0, 1], np.int32), np.array([0], np.int32),
                                    np.ones(1), parameters=p, device="gpu")
    result = shared.solve_batch(objective=np.array([[1.], [1e40]]), variable_lower_bounds=np.zeros(1),
                               variable_upper_bounds=np.ones(1), execution="shared")
    assert result.results[0].termination_reason_name == "OPTIMAL"
    assert result.results[0].primal_solution[0] == 0
    assert result.results[1].termination_reason_name == "NUMERICAL_ERROR"


def test_mixed_difficulty_precision_limit_and_trajectory_independence():
    p = parameters(); p.iteration_limit = 1000
    p.geometric_mean_iterations = p.curtis_reid_iterations = p.l_inf_ruiz_iterations = 0
    p.has_pock_chambolle_alpha = False
    p.bound_objective_rescaling = False
    p.conditional_termination_evaluation = True
    p.termination_criteria.eps_infeasible_relative = 1e-7
    shared = mlxpdlp.SharedMatrixPlan(2, 2, np.array([0, 2, 3], np.int32),
        np.array([0, 0, 1], np.int32), np.array([100000001., -100000000., 1.]),
        parameters=p, device="gpu")
    args = dict(objective=np.array([[0., 1.], [0., 1.], [0., 1.], [0., -1.], [1., 0.]]),
        variable_lower_bounds=np.zeros((5, 2)),
        variable_upper_bounds=np.array([[0., 1.], [0., np.inf], [0., 1.], [0., np.inf], [np.inf, 0.]]),
        constraint_lower_bounds=np.array([[0., 0.], [0., 1.], [0., 2.], [0., 0.], [1., 0.]]),
        constraint_upper_bounds=np.array([[0., np.inf], [0., np.inf], [0., np.inf], [0., np.inf], [1., np.inf]]))
    batch = shared.solve_batch(**args, execution="shared", iteration_batch_size=8)
    assert [r.termination_reason_name for r in batch.results] == [
        "OPTIMAL", "OPTIMAL", "PRIMAL_INFEASIBLE", "DUAL_INFEASIBLE", "ITERATION_LIMIT"]
    assert batch.results[0].total_count < batch.results[1].total_count < batch.results[4].total_count
    # Removing the precision-limited/ray members and moving LP lanes must
    # preserve the easy and slow member's complete certificate and count.
    reduced = shared.solve_batch(**{k: v[[1, 0]] for k, v in args.items()},
                                 execution="shared", iteration_batch_size=8)
    for j, original in enumerate([1, 0]):
        left, right = batch.results[original], reduced.results[j]
        assert left.total_count == right.total_count
        for field in ("primal_solution", "dual_solution", "reduced_cost"):
            np.testing.assert_array_equal(getattr(left, field), getattr(right, field))


def test_optional_numerical_features_and_fused_fallback():
    p = parameters()
    p.conservative_step_size = True
    p.feasibility_polishing = True
    p.host_double_polishing = p.host_double_early_handoff = True
    p.conditional_termination_evaluation = True
    p.reflection_coefficient = 0.5
    p.optimality_norm = mlxpdlp.NormType.L_INF
    shared = plan(p=p)
    result = shared.solve_batch(**request(4), execution="shared")
    assert all(r.termination_reason_name == "OPTIMAL" for r in result.results)
    p.metal_fused_kernels = False
    with pytest.raises(ValueError, match="metal_fused_kernels"):
        shared.solve_batch(**request(4), parameters=p, execution="shared")
    fallback = shared.solve_batch(**request(4), parameters=p)
    assert fallback.execution == "independent"
    assert all(r.termination_reason_name == "OPTIMAL" for r in fallback.results)


def test_request_deadline_expires_while_waiting_for_plan():
    from concurrent.futures import ThreadPoolExecutor
    from threading import Event
    import time
    p = parameters(); p.tolerance = 1e-20
    p.iteration_limit = 1000000; p.time_limit_seconds = 0.25
    shared = mlxpdlp.SharedMatrixPlan(1, 1, np.array([0, 1], np.int32), np.array([0], np.int32),
        np.array([1.0000001]), parameters=p, device="gpu")
    args = dict(objective=np.ones((2, 1)), constraint_lower_bounds=np.ones(1), constraint_upper_bounds=np.ones(1),
                execution="shared")
    entered = Event()
    def occupying_request():
        entered.set()
        return shared.solve_batch(**args)
    with ThreadPoolExecutor(max_workers=1) as pool:
        occupying = pool.submit(occupying_request)
        assert entered.wait(1)
        time.sleep(0.03)
        assert not occupying.done()
        queued = shared.solve_batch(**args, time_sec_limit=0.001)
        assert all(not r.has_solution and r.termination_reason_name == "TIME_LIMIT" for r in queued.results)
        assert all(r.primal_solution.size == 0 for r in queued.results)
        assert queued.wall_time_sec >= 0.001
        assert all(r.has_solution for r in occupying.result().results)


def test_conditional_midpoint_does_not_update_an_unrelated_member():
    p = parameters(); p.tolerance = 1e-3
    p.termination_evaluation_frequency = 200; p.iteration_limit = 400
    p.bound_objective_rescaling = False
    p.geometric_mean_iterations = p.curtis_reid_iterations = p.l_inf_ruiz_iterations = 0
    p.has_pock_chambolle_alpha = False
    shared = mlxpdlp.SharedMatrixPlan(2, 2, np.array([0, 1, 2], np.int32), np.array([0, 1], np.int32),
                                    np.array([1., 1e-5]), parameters=p, device="gpu")
    args = dict(objective=np.array([[1., 0.], [0., 1.], [0., 0.]]), variable_lower_bounds=np.zeros(2),
                constraint_lower_bounds=np.array([[1., 0.], [0., 1.], [0., 0.]]),
                constraint_upper_bounds=np.array([[1., 0.], [0., 1.], [0., 0.]]), execution="shared")
    p.conditional_termination_evaluation = False
    fixed = shared.solve_batch(**args, parameters=p)
    p.conditional_termination_evaluation = True
    conditional = shared.solve_batch(**args, parameters=p)
    assert conditional.results[0].total_count == 300 < fixed.results[0].total_count
    assert conditional.results[1].total_count == fixed.results[1].total_count == 400
    assert conditional.results[1].termination_reason_name == "ITERATION_LIMIT"
    for field in ("primal_solution", "dual_solution", "reduced_cost"):
        np.testing.assert_array_equal(getattr(conditional.results[1], field), getattr(fixed.results[1], field))
