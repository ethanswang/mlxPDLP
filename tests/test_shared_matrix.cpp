// Copyright 2026 Ethan Wang <ethanshurui.wang@gmail.com>
// SPDX-License-Identifier: Apache-2.0
#include "mlxPDLP/batch_solver.h"
#include <iostream>
#include <stdexcept>

using namespace mlxpdlp;
int main() {
    try {
        if (!mx::is_available(mx::Device::gpu)) return 77;
        pdhg_parameters_t p; mlxpdlp_set_default_parameters(&p);
        p.presolve=false;p.verbose=false;p.geometric_mean_iterations=0;
        p.curtis_reid_iterations=0;p.l_inf_ruiz_iterations=0;
        p.has_pock_chambolle_alpha=false;p.bound_objective_rescaling=false;
        p.conservative_step_size=false;p.termination_evaluation_frequency=10;
        p.termination_criteria.iteration_limit=100;
        p.termination_criteria.eps_feasible_relative=1e-4;
        p.termination_criteria.eps_optimal_relative=1e-4;
        int rp[]={0,2},ci[]={0,0};double values[]={100000001.,-100000000.};
        // Exact original A=1; after separate FP32 rounding, stored A=0.
        // The norm bound must describe the stored operator while certificates
        // must describe the original model. A GPU Farkas ray is invalid here.
        SharedMatrixPlan gpu(1,1,rp,ci,values,&p,mx::Device::gpu);
        SharedMatrixPlan cpu(1,1,rp,ci,values,&p,mx::Device::cpu);
        // The absolute-sum bound can be loose for duplicate cancellation.
        // A separate single-entry operator distinguishes FP32 rounding from
        // the original FP64 coefficient without relying on bound tightness.
        int single_rp[]={0,1},single_ci[]={0};
        double rounded_value=1.0+0.75*std::numeric_limits<float>::epsilon();
        SharedMatrixPlan rounded(1,1,single_rp,single_ci,&rounded_value,&p,mx::Device::gpu);
        const double stored=static_cast<float>(rounded_value);
        if (rounded.operator_norm_upper_bound()<stored ||
            rounded.operator_norm_upper_bound()>stored*(1+1e-12))
            throw std::runtime_error("norm bound does not describe the actual stored operator");
        const size_t resident=gpu.resident_bytes();
        std::vector<BatchProblem> members(4);
        for (auto &m : members) {
            m.objective={1};m.variable_lower_bounds={0};
            m.constraint_lower_bounds=m.constraint_upper_bounds={1};
        }
        BatchOptions shared;shared.execution=BatchExecution::shared;
        auto limited=gpu.solve_batch(members,shared);
        for (const auto &m : limited.results) {
            if (!m.has_solution || m.result->termination_reason!=TERMINATION_REASON_ITERATION_LIMIT)
                throw std::runtime_error("rounded matrix produced a false original-model certificate");
            if (m.original_audit_failures==0)
                throw std::runtime_error("adversary did not exercise original-model audit rejection");
        }
        if (gpu.resident_bytes()!=resident) throw std::runtime_error("submission mutated matrix preparation");
        auto reference=cpu.solve_batch(members);
        for (const auto &m : reference.results)
            if (m.result->termination_reason!=TERMINATION_REASON_OPTIMAL || std::abs(m.result->primal_solution[0]-1)>1e-3)
                throw std::runtime_error("original FP64 model should be feasible and optimal");
        values[0]=0;ci[0]=999;
        auto again=cpu.solve_batch(members);
        for (const auto &m : again.results)
            if (m.result->termination_reason!=TERMINATION_REASON_OPTIMAL)
                throw std::runtime_error("plan borrowed caller CSR buffers");
        std::cout<<"shared matrix ownership, stored norm, and original certificate separation passed\n";
    } catch (const std::exception &e) {std::cerr<<e.what()<<'\n';return 1;}
}
