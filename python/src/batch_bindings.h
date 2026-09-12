// Copyright 2026 Ethan Wang <ethanshurui.wang@gmail.com>
// SPDX-License-Identifier: Apache-2.0
// Included by module.cpp after the shared SolveResult/numpy conversion helpers.
#pragma once
using F64BatchArr = nb::ndarray<const double, nb::c_contig>;
struct PythonBatchResult {
    std::vector<SolveResult> results;
    std::string execution, fallback_reason;
    double wall_time_sec, packing_time_sec, pdhg_time_sec, checkpoint_time_sec, audit_time_sec, deadline_overrun_sec;
    double construction_time_sec, initialization_time_sec;
    size_t estimated_peak_resident_bytes, groups, max_active_width;
    int lp_tile_width, iteration_batch_size;
    bool native_iteration_batching_active;
};
static void bind_batch(nb::module_ &m) {
    nb::class_<PythonBatchResult>(m, "BatchResult")
        .def_ro("results", &PythonBatchResult::results)
        .def_ro("execution", &PythonBatchResult::execution)
        .def_ro("fallback_reason", &PythonBatchResult::fallback_reason)
        .def_ro("wall_time_sec", &PythonBatchResult::wall_time_sec)
        .def_ro("packing_time_sec", &PythonBatchResult::packing_time_sec)
        .def_ro("construction_time_sec", &PythonBatchResult::construction_time_sec)
        .def_ro("initialization_time_sec", &PythonBatchResult::initialization_time_sec)
        .def_ro("pdhg_time_sec", &PythonBatchResult::pdhg_time_sec)
        .def_ro("checkpoint_time_sec", &PythonBatchResult::checkpoint_time_sec)
        .def_ro("audit_time_sec", &PythonBatchResult::audit_time_sec)
        .def_ro("deadline_overrun_sec", &PythonBatchResult::deadline_overrun_sec)
        .def_ro("estimated_peak_resident_bytes", &PythonBatchResult::estimated_peak_resident_bytes)
        .def_ro("groups", &PythonBatchResult::groups)
        .def_ro("max_active_width", &PythonBatchResult::max_active_width)
        .def_ro("lp_tile_width", &PythonBatchResult::lp_tile_width)
        .def_ro("iteration_batch_size", &PythonBatchResult::iteration_batch_size)
        .def_ro("native_iteration_batching_active", &PythonBatchResult::native_iteration_batching_active);
    nb::class_<SharedMatrixPlan>(m, "SharedMatrixPlan")
        .def("__init__", [](SharedMatrixPlan *self, int n, int rows, const I32Arr &rp,
                            const I32Arr &ci, const F64Arr &values, pdhg_parameters_t *params,
                            const std::string &device) {
            if (n < 0 || rows < 0 || rp.size() != size_t(rows)+1)
                throw nb::value_error("matrix.row_ptr: expected num_constraints + 1 entries");
            if (rp(rows) < 0 || ci.size() != size_t(rp(rows)) || values.size() != ci.size())
                throw nb::value_error("matrix: CSR values and column indices must match row_ptr[-1]");
            new (self) SharedMatrixPlan(n, rows, rp.data(), ci.data(), values.data(), params, parse_device(device));
        }, nb::arg("num_variables"), nb::arg("num_constraints"), nb::arg("row_ptr"),
           nb::arg("col_indices"), nb::arg("values"), nb::arg("parameters") = nb::none(), nb::arg("device") = "cpu")
        .def_prop_ro("num_variables", &SharedMatrixPlan::num_variables)
        .def_prop_ro("num_constraints", &SharedMatrixPlan::num_constraints)
        .def_prop_ro("num_nonzeros", &SharedMatrixPlan::num_nonzeros)
        .def_prop_ro("resident_bytes", &SharedMatrixPlan::resident_bytes)
        .def_prop_ro("preparation_time_sec", &SharedMatrixPlan::preparation_time_sec)
        .def_prop_ro("operator_norm_upper_bound", &SharedMatrixPlan::operator_norm_upper_bound)
        .def("solve_batch", [](SharedMatrixPlan &self, const F64BatchArr &objective,
              nb::object offsets, const std::optional<F64BatchArr> &vl,
              const std::optional<F64BatchArr> &vu, const std::optional<F64BatchArr> &cl,
              const std::optional<F64BatchArr> &cu, const std::optional<F64BatchArr> &ps,
              const std::optional<F64BatchArr> &ds, const std::optional<F64BatchArr> &zs,
              const std::optional<std::vector<bool>> &pm, const std::optional<std::vector<bool>> &dm,
              const std::optional<std::vector<bool>> &zm, pdhg_parameters_t *parameters,
              const std::string &execution, double time_limit, size_t memory_budget,
              int tile, int iterations) {
            const auto entry = std::chrono::steady_clock::now();
            const size_t n = self.num_variables(), rows = self.num_constraints();
            if (objective.ndim() != 2 || objective.shape(1) != n)
                throw nb::value_error("objective: expected shape [B, num_variables]");
            const size_t count = objective.shape(0);
            std::vector<BatchProblem> problems(count);
            for (size_t j = 0; j < count; ++j)
                problems[j].objective.assign(objective.data()+j*n, objective.data()+(j+1)*n);
            auto fill_bounds = [&](const auto &array, size_t length, const char *field, auto member) {
                if (!array) return;
                const bool common = array->ndim() == 1 && array->shape(0) == length;
                const bool batch = array->ndim() == 2 && array->shape(0) == count && array->shape(1) == length;
                if (!common && !batch) throw nb::value_error((std::string(field)+": expected [components] or [B, components]").c_str());
                for (size_t j = 0; j < count; ++j) {
                    const auto *p = array->data() + (common ? 0 : j*length);
                    (problems[j].*member).assign(p,p+length);
                }
            };
            fill_bounds(vl,n,"variable_lower_bounds",&BatchProblem::variable_lower_bounds);
            fill_bounds(vu,n,"variable_upper_bounds",&BatchProblem::variable_upper_bounds);
            fill_bounds(cl,rows,"constraint_lower_bounds",&BatchProblem::constraint_lower_bounds);
            fill_bounds(cu,rows,"constraint_upper_bounds",&BatchProblem::constraint_upper_bounds);
            auto fill_starts = [&](const auto &array, const auto &mask, size_t length, const char *field, auto member) {
                if (mask && (!array || mask->size() != count))
                    throw nb::value_error((std::string(field)+"_mask: requires a start array and B booleans").c_str());
                if (!array) return;
                if (array->ndim() != 2 || array->shape(0) != count || array->shape(1) != length)
                    throw nb::value_error((std::string(field)+": expected [B, components]; no broadcasting").c_str());
                for (size_t j = 0; j < count; ++j) if (!mask || (*mask)[j])
                    problems[j].*member = std::vector<double>(array->data()+j*length,array->data()+(j+1)*length);
            };
            fill_starts(ps,pm,n,"primal_start",&BatchProblem::primal_start);
            fill_starts(ds,dm,rows,"dual_start",&BatchProblem::dual_start);
            fill_starts(zs,zm,n,"reduced_cost_start",&BatchProblem::reduced_cost_start);
            nb::object offset_array = nb::module_::import_("numpy").attr("asarray")(offsets, nb::arg("dtype")="float64");
            auto offset_view = nb::cast<F64BatchArr>(offset_array);
            if (offset_view.ndim() != 0 && !(offset_view.ndim() == 1 && offset_view.size() == count))
                throw nb::value_error("objective_constant: expected scalar or [B]");
            for (size_t j = 0; j < count; ++j) problems[j].objective_constant = offset_view.data()[offset_view.ndim() == 0 ? 0 : j];
            BatchOptions options;
            if (execution == "auto") options.execution = BatchExecution::automatic;
            else if (execution == "shared") options.execution = BatchExecution::shared;
            else if (execution == "independent") options.execution = BatchExecution::independent;
            else throw nb::value_error("execution: expected auto, shared, or independent");
            if (std::isnan(time_limit) || time_limit < 0) throw nb::value_error("time_sec_limit: must be nonnegative");
            const double packing = std::chrono::duration<double>(std::chrono::steady_clock::now()-entry).count();
            options.time_sec_limit = std::max(0.0,time_limit-packing);
            options.resident_memory_budget_bytes = memory_budget;
            options.lp_tile_width = tile; options.iteration_batch_size = iterations;
            std::optional<pdhg_parameters_t> settings;
            if (parameters) settings = *parameters;
            BatchResult batch;
            {
                nb::gil_scoped_release release;
                batch = self.solve_batch(problems, options, settings ? &*settings : nullptr);
            }
            PythonBatchResult out;
            for (auto &member : batch.results) {
                auto value = SolveResult::from_result(member.result.get());
                value.input_index = member.input_index; value.has_solution = member.has_solution;
                value.step_size_reductions = member.step_size_reductions;
                value.original_audit_failures = member.original_audit_failures;
                value.primal_ray = to_numpy_f64(member.primal_ray.data(), member.primal_ray.size());
                value.dual_ray = to_numpy_f64(member.dual_ray.data(), member.dual_ray.size());
                value.queue_time_sec = member.queue_time_sec + packing;
                value.execution_time_sec = member.execution_time_sec;
                out.results.push_back(std::move(value));
            }
            out.execution = batch.execution == BatchExecution::shared ? "shared" : "independent";
            out.fallback_reason = batch.fallback_reason;
            out.wall_time_sec = std::chrono::duration<double>(std::chrono::steady_clock::now()-entry).count();
            out.packing_time_sec = packing + batch.packing_time_sec;
            out.construction_time_sec = batch.construction_time_sec;
            out.initialization_time_sec = batch.initialization_time_sec;
            out.pdhg_time_sec = batch.pdhg_time_sec; out.checkpoint_time_sec = batch.checkpoint_time_sec;
            out.audit_time_sec = batch.audit_time_sec;
            out.deadline_overrun_sec = std::max(0.0,out.wall_time_sec-time_limit);
            out.estimated_peak_resident_bytes = batch.estimated_peak_resident_bytes;
            out.groups = batch.groups; out.max_active_width = batch.max_active_width;
            out.lp_tile_width = batch.lp_tile_width; out.iteration_batch_size = batch.iteration_batch_size;
            out.native_iteration_batching_active = batch.native_iteration_batching_active;
            return out;
        }, nb::arg("objective"), nb::arg("objective_constant")=0.0,
           nb::arg("variable_lower_bounds")=nb::none(), nb::arg("variable_upper_bounds")=nb::none(),
           nb::arg("constraint_lower_bounds")=nb::none(), nb::arg("constraint_upper_bounds")=nb::none(),
           nb::arg("primal_start")=nb::none(), nb::arg("dual_start")=nb::none(), nb::arg("reduced_cost_start")=nb::none(),
           nb::arg("primal_start_mask")=nb::none(), nb::arg("dual_start_mask")=nb::none(), nb::arg("reduced_cost_start_mask")=nb::none(),
           nb::arg("parameters")=nb::none(), nb::arg("execution")="auto", nb::arg("time_sec_limit")=INFINITY,
           nb::arg("resident_memory_budget_bytes")=0, nb::arg("lp_tile_width")=4, nb::arg("iteration_batch_size")=16);
}
