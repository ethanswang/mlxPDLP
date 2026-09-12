// Copyright 2026 Ethan Wang <ethanshurui.wang@gmail.com>
// SPDX-License-Identifier: Apache-2.0
#include "mlxPDLP/batch_solver.h"
#include "lpfeas_support.h"
#include "benchmark_provenance.h"
#include "mlx/memory.h"
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>
#include <sys/resource.h>

using namespace mlxpdlp;
using Clock = std::chrono::steady_clock;
static double elapsed(Clock::time_point t) { return std::chrono::duration<double>(Clock::now()-t).count(); }
static double percentile(std::vector<double> values, double p) {
    if (values.empty()) return 0;
    std::sort(values.begin(),values.end());
    return values[std::min(values.size()-1,size_t(std::ceil(p*values.size())-1))];
}
static void parameter_metadata(const pdhg_parameters_t &p) {
    std::cerr << "@parameters {";
#define PARAM(field) std::cerr << "\"" #field "\":" << p.field << ','
    PARAM(geometric_mean_iterations); PARAM(curtis_reid_iterations); PARAM(l_inf_ruiz_iterations);
    PARAM(has_pock_chambolle_alpha); PARAM(pock_chambolle_alpha); PARAM(bound_objective_rescaling);
    PARAM(verbose); PARAM(termination_evaluation_frequency); PARAM(sv_max_iter); PARAM(sv_tol);
    PARAM(termination_criteria.eps_optimal_relative); PARAM(termination_criteria.eps_feasible_relative);
    PARAM(termination_criteria.eps_feas_polish_relative); PARAM(termination_criteria.eps_infeasible_relative);
    PARAM(termination_criteria.time_sec_limit); PARAM(termination_criteria.iteration_limit);
    PARAM(restart_params.artificial_restart_threshold); PARAM(restart_params.sufficient_reduction_for_restart);
    PARAM(restart_params.necessary_reduction_for_restart); PARAM(restart_params.k_p); PARAM(restart_params.k_i);
    PARAM(restart_params.k_d); PARAM(restart_params.i_smooth); PARAM(restart_policy); PARAM(reflection_coefficient);
    PARAM(feasibility_polishing); PARAM(host_double_polishing); PARAM(host_double_early_handoff);
    PARAM(host_double_polishing_iteration_limit); PARAM(host_double_polishing_time_sec_limit); PARAM(optimality_norm);
    PARAM(presolve); PARAM(presolve_singleton_columns); PARAM(presolve_doubleton_equations); PARAM(presolve_parallel_rows);
    PARAM(presolve_parallel_columns); PARAM(presolve_dual_fix); PARAM(presolve_finite_bound_tightening);
    PARAM(presolve_primal_propagation); PARAM(matrix_zero_tol); PARAM(metal_fused_kernels); PARAM(metal_iteration_batching);
    PARAM(conditional_termination_evaluation); PARAM(conservative_step_size);
#undef PARAM
    std::cerr << "\"host_double_residual_evaluation\":" << p.host_double_residual_evaluation << "}\n";
}
static void json_number(std::ostream &out, double x) {
    if (std::isfinite(x)) out << x; else out << "null";
}
int main(int argc, char **argv) {
    try {
        if (argc < 4) {
            std::cerr << "usage: mlxpdlp_batch_lp_benchmark case.mps[.gz] old|old-no-presolve|reused|shared|auto B [K=16] [tile=4] [trials=5] [heterogeneous=0] [dependent=0] [vary_bounds=0] [memory_budget_bytes=0]\n";
            return 2;
        }
        const std::string arm=argv[2];
        const int width=std::stoi(argv[3]), k=argc>4?std::stoi(argv[4]):16;
        const int tile=argc>5?std::stoi(argv[5]):4, trials=argc>6?std::stoi(argv[6]):5;
        const bool heterogeneous=argc>7&&std::stoi(argv[7])!=0;
        const bool dependent=argc>8&&std::stoi(argv[8])!=0;
        const bool vary_bounds=argc>9&&std::stoi(argv[9])!=0;
        const size_t memory_budget=argc>10?std::stoull(argv[10]):0;
        if (width<1||trials<1||(arm!="old"&&arm!="old-no-presolve"&&arm!="reused"&&arm!="shared"&&arm!="auto"))
            throw std::invalid_argument("invalid benchmark arguments");
        const auto application_entry=Clock::now();
        std::unique_ptr<mlxpdlp_mps_problem_t,decltype(&mlxpdlp_mps_problem_free)> problem(
            mlxpdlp_mps_problem_load(argv[1]),mlxpdlp_mps_problem_free);
        if (!problem) throw std::runtime_error("cannot load MPS");
        const int n=problem->num_variables,m=problem->num_constraints;
        pdhg_parameters_t params; mlxpdlp_set_default_parameters(&params);
        params.presolve=arm=="old"; params.verbose=false;
        params.termination_criteria.eps_optimal_relative=1e-4;
        params.termination_criteria.eps_feasible_relative=1e-4;
        params.termination_criteria.iteration_limit=50000;
        params.termination_criteria.time_sec_limit=5;
        std::cerr << std::setprecision(17);
        parameter_metadata(params);
        // Legacy K is the solver's native default, not the requested shared K.
        std::vector<BatchProblem> members(width);
        for (int j=0;j<width;++j) {
            auto &p=members[j];
            p.objective.assign(problem->objective,problem->objective+n);
            const double factor=(problem->maximize?-1:1)*(1.0+double(j)/width);
            for (int i=0;i<n;++i) {
                p.objective[i]*=factor;
                if (heterogeneous && j%3==1) p.objective[i]+=(i%7-3)*1e-3;
                if (heterogeneous && j%3==2) p.objective[i]=0;
            }
            p.objective_constant=(problem->maximize?-1:1)*problem->objective_constant+j/8.0;
            p.variable_lower_bounds.assign(problem->variable_lb,problem->variable_lb+n);
            p.variable_upper_bounds.assign(problem->variable_ub,problem->variable_ub+n);
            p.constraint_lower_bounds.assign(problem->constraint_lb,problem->constraint_lb+m);
            p.constraint_upper_bounds.assign(problem->constraint_ub,problem->constraint_ub+m);
            if (vary_bounds && j%3==1) {
                // Relax finite bounds; every feasible point of the source LP
                // remains feasible in the derived member. A stays identical.
                for (auto *v : {&p.variable_lower_bounds,&p.constraint_lower_bounds})
                    for (double &b : *v) if (std::isfinite(b)) b-=1e-4*(1+std::abs(b));
                for (auto *v : {&p.variable_upper_bounds,&p.constraint_upper_bounds})
                    for (double &b : *v) if (std::isfinite(b)) b+=1e-4*(1+std::abs(b));
            }
        }
        auto validate_member=[&](int j,const mlxpdlp_result_t &result) {
            auto model=*problem;
            model.variable_lb=members[j].variable_lower_bounds.data();
            model.variable_ub=members[j].variable_upper_bounds.data();
            model.constraint_lb=members[j].constraint_lower_bounds.data();
            model.constraint_ub=members[j].constraint_upper_bounds.data();
            return benchmark::validate_original_problem(model,result,members[j].objective.data(),members[j].objective_constant);
        };
        std::unique_ptr<SharedMatrixPlan> plan;
        const double input_setup_sec=elapsed(application_entry);
        double plan_sec=0;
        if (arm=="reused"||arm=="shared"||arm=="auto") {
            const auto start=Clock::now();
            plan=std::make_unique<SharedMatrixPlan>(n,m,problem->row_ptr,problem->col_ind,problem->values,&params,mx::Device::gpu);
            plan_sec=elapsed(start);
        }
        auto run=[&]() {
            BatchResult result;
            const auto start=Clock::now();
            if (plan) {
                BatchOptions options;
                options.execution=arm=="shared"?BatchExecution::shared:
                    (arm=="auto"?BatchExecution::automatic:BatchExecution::independent);
                options.iteration_batch_size=k;options.lp_tile_width=tile;
                options.resident_memory_budget_bytes=memory_budget;
                if (!dependent) result=plan->solve_batch(members,options);
                else {
                    // Synthetic control dependency: admit the next request
                    // only after consuming the previous result. Never combine
                    // these requests into an artificially ready B-wide solve.
                    for (int j=0;j<width;++j) {
                        auto one=plan->solve_batch({members[j]},options);
                        auto member=std::move(one.results[0]);member.input_index=j;
                        result.results.push_back(std::move(member));
                        result.construction_time_sec+=one.construction_time_sec;
                        result.initialization_time_sec+=one.initialization_time_sec;
                        result.packing_time_sec+=one.packing_time_sec;
                        result.audit_time_sec+=one.audit_time_sec;
                    }
                    result.wall_time_sec=elapsed(start);
                    result.fallback_reason="sequential control dependencies admit B=1";
                }
            } else {
                for (int j=0;j<width;++j) {
                    const auto admitted=Clock::now();const auto &p=members[j];
                    MlxPdlpSolver solver(n,m,problem->row_ptr,problem->col_ind,problem->values,
                        p.variable_lower_bounds.data(),p.variable_upper_bounds.data(),p.constraint_lower_bounds.data(),
                        p.constraint_upper_bounds.data(),p.objective.data(),p.objective_constant,&params,mx::Device::gpu);
                    result.construction_time_sec+=elapsed(admitted);
                    BatchMemberResult member;
                    member.input_index=j;member.has_solution=true;
                    member.result.reset(solver.solve());
                    member.queue_time_sec=dependent?0:std::chrono::duration<double>(admitted-start).count();
                    member.execution_time_sec=elapsed(admitted);
                    result.results.push_back(std::move(member));
                }
                result.wall_time_sec=elapsed(start);
            }
            return result;
        };
        auto warmup=run(); // kernel compilation; measured separately as cold time
        int cold_audited=0;
        for (int j=0;j<width;++j)
            cold_audited += warmup.results[j].result->termination_reason==TERMINATION_REASON_OPTIMAL &&
                validate_member(j,*warmup.results[j].result).satisfies(1e-4);
        const double cold_sec=elapsed(application_entry);
        warmup.results.clear();
        std::cout << "arm,B,K,tile,trial,presolve,heterogeneous,dependent,vary_bounds,memory_budget_bytes,groups,max_active_width,n,m,nnz,input_setup_sec,plan_sec,cold_wall_sec,cold_audited,wall_sec,packing_sec,construction_sec,initialization_sec,member_rescaling_sec,single_solver_sec,pdhg_sec,checkpoint_sec,audit_sec,correction_sec,audited,iterations,max_iterations,occupancy,median_queue_sec,p95_queue_sec,median_execution_sec,p95_execution_sec,median_latency_sec,p95_latency_sec,audited_per_sec,peak_rss_bytes,peak_mlx_bytes,mlx_cache_bytes,estimated_bytes,execution,native_active,fallback_reason\n";
        for (int trial=0;trial<trials;++trial) {
            mx::reset_peak_memory();
            auto result=run();
            int audited=0,maximum=0;long long iterations=0;
            double correction=0,rescaling=0,single_solver=0;
            std::vector<double> execution,latency,queue;
            std::ostringstream member_log;
            member_log << std::setprecision(17);
            const auto audit_start=Clock::now();
            for (int j=0;j<width;++j) {
                const auto &r=*result.results[j].result;
                const auto metrics=validate_member(j,r);
                const bool passed=r.termination_reason==TERMINATION_REASON_OPTIMAL&&metrics.satisfies(1e-4);
                if (passed) ++audited;
                iterations+=r.total_count;maximum=std::max(maximum,r.total_count);
                correction+=r.feasibility_polishing_time+r.host_double_polishing_time;
                rescaling+=r.rescaling_time_sec;single_solver+=r.cumulative_time_sec;
                queue.push_back(result.results[j].queue_time_sec);
                execution.push_back(result.results[j].execution_time_sec);
                latency.push_back(result.results[j].queue_time_sec+result.results[j].execution_time_sec);
                member_log << "@member {\"trial\":" << trial << ",\"index\":" << j
                          << ",\"status\":" << r.termination_reason << ",\"audited\":" << passed
                          << ",\"iterations\":" << r.total_count << ",\"queue_sec\":" << queue.back()
                          << ",\"execution_sec\":" << execution.back() << ",\"primal_residual\":";
                json_number(member_log,metrics.relative_primal_residual);
                member_log << ",\"dual_residual\":";json_number(member_log,metrics.relative_dual_residual);
                member_log << ",\"gap\":";json_number(member_log,metrics.relative_objective_gap);
                member_log << ",\"correction_sec\":" << r.feasibility_polishing_time+r.host_double_polishing_time << "}\n";
            }
            const double independent_audit=elapsed(audit_start);
            std::cerr << member_log.str();
            // Throughput includes the same external audit in all arms.
            const double total=result.wall_time_sec+independent_audit;
            struct rusage usage{};getrusage(RUSAGE_SELF,&usage);
            std::cout << std::setprecision(12) << arm << ',' << width << ',' << (arm=="shared"?k:0) << ',' << (arm=="shared"?tile:0)
                << ',' << trial << ',' << params.presolve << ',' << heterogeneous << ',' << dependent << ',' << vary_bounds
                << ',' << memory_budget << ',' << result.groups << ',' << result.max_active_width << ',' << n << ',' << m << ',' << problem->num_nonzeros
                << ',' << input_setup_sec << ',' << plan_sec << ',' << cold_sec << ',' << cold_audited << ',' << total << ',' << result.packing_time_sec
                << ',' << result.construction_time_sec << ',' << result.initialization_time_sec << ',' << rescaling << ',' << single_solver
                << ',' << result.pdhg_time_sec << ',' << result.checkpoint_time_sec
                << ',' << result.audit_time_sec+independent_audit << ',' << correction << ',' << audited << ',' << iterations << ',' << maximum
                << ',' << (maximum?double(iterations)/(width*double(maximum)):1) << ',' << percentile(queue,0.5) << ',' << percentile(queue,0.95)
                << ',' << percentile(execution,0.5) << ',' << percentile(execution,0.95)
                << ',' << percentile(latency,0.5) << ',' << percentile(latency,0.95) << ',' << audited/total << ',' << usage.ru_maxrss
                << ',' << mx::get_peak_memory() << ',' << mx::get_cache_memory() << ',' << result.estimated_peak_resident_bytes
                << ',' << (result.execution==BatchExecution::shared?"shared":"independent")
                << ',' << result.native_iteration_batching_active << ",\"" << result.fallback_reason << "\"\n";
            std::cout.flush();
        }
        std::cerr << "solver=" << MLXPDLP_BENCHMARK_GIT_REVISION << " source=" << MLXPDLP_BENCHMARK_SOURCE_SHA256
                  << " mlx=" << MLXPDLP_BENCHMARK_MLX_REVISION << '\n';
    } catch (const std::exception &e) {std::cerr<<e.what()<<'\n';return 1;}
}
