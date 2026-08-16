/*
Copyright 2026 Ethan Wang <ethanshurui.wang@gmail.com>

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

/*
Run the public LPfeas instances with the published cuPDLPx-style protocol and
independently validate returned solutions on the original model in float64.
*/
#include "lpfeas_support.h"

#include "mlxPDLP/mps_loader.h"
#include "mlxPDLP/solver.h"
#include "mlxPDLP/version.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cerrno>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <latch>
#include <memory>
#include <map>
#include <mutex>
#include <optional>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <ctime>
#include <vector>

#include <sys/utsname.h>
#include <unistd.h>

#ifdef __APPLE__
#include <sys/sysctl.h>
#endif

namespace fs = std::filesystem;
using namespace mlxpdlp;
using mlxpdlp::benchmark::ValidationMetrics;

namespace {

using Clock = std::chrono::steady_clock;

std::mutex progress_mutex;

template <typename... Args>
void progress_printf(const char *format, Args... args) {
    std::lock_guard<std::mutex> lock(progress_mutex);
    std::printf(format, args...);
    std::fflush(stdout);
}

struct ProblemDeleter {
    void operator()(mlxpdlp_mps_problem_t *problem) const {
        mlxpdlp_mps_problem_free(problem);
    }
};
using ProblemPtr = std::unique_ptr<mlxpdlp_mps_problem_t, ProblemDeleter>;
using ResultPtr = std::unique_ptr<mlxpdlp_result_t, decltype(&mlxpdlp_result_free)>;

struct ManifestEntry {
    std::string name;
    int expected_rows = 0;
    int expected_columns = 0;
    int expected_nonzeros = 0;
    std::string format;
    std::string url;
    std::optional<double> reference_objective;
};

enum class DeviceSelection { metal, cpu };

struct Options {
    fs::path data_directory = "benchmarks/data/lpfeas";
    std::optional<fs::path> manifest;
    std::optional<fs::path> reference_objectives;
    fs::path output_prefix = "benchmarks/results/lpfeas-metal";
    std::set<std::string> instances;
    DeviceSelection device = DeviceSelection::metal;
    double tolerance = 1e-4;
    std::optional<double> solver_tolerance;
    std::optional<fs::path> load_warm_start;
    std::optional<fs::path> save_solution;
    double time_limit_seconds = 1000.0;
    int iteration_limit = INT_MAX;
    int warm_start_correction_iteration_limit = 200000;
    double warm_start_correction_time_limit_seconds = 300.0;
    int host_double_polishing_iteration_limit = 50000;
    double host_double_polishing_time_limit_seconds = 30.0;
    int evaluation_frequency = 200;
    int singular_value_iterations = 200;
    int curtis_reid_iterations = 20;
    int restart_policy = 0;
    int jobs = 0;
    bool jobs_auto = true;
    bool feasibility_polishing = true;
    bool host_double_polishing = true;
    bool host_double_early_handoff = true;
    bool presolve = true;
    bool presolve_singleton_columns = false;
    bool presolve_doubleton_equations = true;
    bool presolve_parallel_rows = true;
    bool presolve_parallel_columns = true;
    bool presolve_dual_fix = true;
    bool presolve_finite_bound_tightening = true;
    bool presolve_primal_propagation = true;
    bool retry_without_primal_propagation = true;
    bool warm_start_correction = true;
    bool retry_without_presolve = true;
    bool retry_without_curtis_reid = true;
    bool verbose = false;
    bool warm_up = true;
    bool fail_on_validation = false;
};

struct WarmStartData {
    std::vector<double> primal;
    std::vector<double> dual;
    std::vector<double> reduced_cost;
};

constexpr uint64_t warm_start_magic_v1 = UINT64_C(0x4d4c5850444c5031);
constexpr uint64_t warm_start_magic_v2 = UINT64_C(0x4d4c5850444c5032);

WarmStartData read_warm_start(const fs::path &path, int expected_columns,
                              int expected_rows) {
    std::ifstream input(path, std::ios::binary);
    if (!input)
        throw std::runtime_error("cannot read warm-start checkpoint: " + path.string());
    uint64_t magic = 0;
    int64_t columns = 0;
    int64_t rows = 0;
    input.read(reinterpret_cast<char *>(&magic), sizeof(magic));
    input.read(reinterpret_cast<char *>(&columns), sizeof(columns));
    input.read(reinterpret_cast<char *>(&rows), sizeof(rows));
    if (!input ||
        (magic != warm_start_magic_v1 && magic != warm_start_magic_v2) ||
        columns != expected_columns ||
        rows != expected_rows) {
        throw std::runtime_error("warm-start checkpoint does not match the LP dimensions");
    }
    WarmStartData data;
    data.primal.resize(static_cast<size_t>(columns));
    data.dual.resize(static_cast<size_t>(rows));
    input.read(reinterpret_cast<char *>(data.primal.data()),
               static_cast<std::streamsize>(data.primal.size() * sizeof(double)));
    input.read(reinterpret_cast<char *>(data.dual.data()),
               static_cast<std::streamsize>(data.dual.size() * sizeof(double)));
    if (magic == warm_start_magic_v2) {
        data.reduced_cost.resize(static_cast<size_t>(columns));
        input.read(reinterpret_cast<char *>(data.reduced_cost.data()),
                   static_cast<std::streamsize>(data.reduced_cost.size() *
                                                sizeof(double)));
    }
    if (!input)
        throw std::runtime_error("warm-start checkpoint is truncated: " + path.string());
    return data;
}

void write_warm_start(const fs::path &path, const std::vector<double> &primal,
                      const std::vector<double> &dual,
                      const std::vector<double> &reduced_cost) {
    if (reduced_cost.size() != primal.size())
        throw std::invalid_argument(
            "complete solution checkpoint requires one reduced cost per column");
    if (!path.parent_path().empty())
        fs::create_directories(path.parent_path());
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    if (!output)
        throw std::runtime_error("cannot write solution checkpoint: " + path.string());
    const int64_t columns = static_cast<int64_t>(primal.size());
    const int64_t rows = static_cast<int64_t>(dual.size());
    output.write(reinterpret_cast<const char *>(&warm_start_magic_v2),
                 sizeof(warm_start_magic_v2));
    output.write(reinterpret_cast<const char *>(&columns), sizeof(columns));
    output.write(reinterpret_cast<const char *>(&rows), sizeof(rows));
    output.write(reinterpret_cast<const char *>(primal.data()),
                 static_cast<std::streamsize>(primal.size() * sizeof(double)));
    output.write(reinterpret_cast<const char *>(dual.data()),
                 static_cast<std::streamsize>(dual.size() * sizeof(double)));
    output.write(reinterpret_cast<const char *>(reduced_cost.data()),
                 static_cast<std::streamsize>(reduced_cost.size() * sizeof(double)));
    if (!output)
        throw std::runtime_error("failed while writing solution checkpoint: " + path.string());
}

double effective_solver_tolerance(const Options &options) {
    // Leave enough room between backend termination and the independently audited
    // original-model target.  Netlib FORPLAN can satisfy an 0.8x reduced KKT
    // threshold while missing the published objective just beyond 1e-4; 0.5x
    // reliably crosses that boundary on both CPU and Metal.
    return options.solver_tolerance.value_or(0.5 * options.tolerance);
}

struct RunRecord {
    ManifestEntry manifest;
    std::string path;
    std::string error;
    std::string validation_warning;
    std::string termination = "NOT_RUN";
    bool completed = false;
    bool verified = false;
    bool sparse_metal = false;
    bool sparse_cpu = false;
    bool cpu_double_precision = false;
    int worker_id = 0;
    int completion_order = 0;
    bool selected_presolve = false;
    bool selected_primal_propagation = false;
    bool selected_warm_start = false;
    bool fallback_attempted = false;
    int attempts = 0;
    std::string fallback_reason;
    bool propagation_fallback_attempted = false;
    std::string propagation_fallback_reason;
    bool warm_start_correction_attempted = false;
    std::string warm_start_correction_reason;
    bool host_handoff_maturity_retry_attempted = false;
    std::string host_handoff_maturity_retry_reason;
    bool scaling_fallback_attempted = false;
    std::string scaling_fallback_reason;
    bool restart_policy_fallback_attempted = false;
    std::string restart_policy_fallback_reason;
    int selected_curtis_reid_iterations = 0;
    int rows = 0;
    int columns = 0;
    int nonzeros = 0;
    int reduced_rows = 0;
    int reduced_columns = 0;
    int reduced_nonzeros = 0;
    int iterations = 0;
    int feasibility_iterations = 0;
    int host_double_iterations = 0;
    bool host_double_handoff = false;
    double parse_seconds = 0.0;
    double setup_seconds = 0.0;
    double solve_seconds = 0.0;
    double verification_seconds = 0.0;
    double total_seconds = 0.0;
    double presolve_seconds = 0.0;
    double rescaling_seconds = 0.0;
    double feasibility_polishing_seconds = 0.0;
    double host_double_polishing_seconds = 0.0;
    double solver_reported_seconds = 0.0;
    double original_primal_objective = 0.0;
    double original_dual_objective = 0.0;
    double objective_without_constant = 0.0;
    double dual_objective_without_constant = 0.0;
    bool reference_objective_available = false;
    double reference_objective = 0.0;
    double reference_objective_relative_error = 0.0;
    double solver_relative_primal_residual = 0.0;
    double solver_relative_dual_residual = 0.0;
    double solver_relative_objective_gap = 0.0;
    ValidationMetrics validation;
    // Retained only while constructing the in-process audit portfolio.
    std::vector<double> primal_solution;
    std::vector<double> dual_solution;
    std::vector<double> reduced_cost;
};

double seconds_since(Clock::time_point start) {
    return std::chrono::duration<double>(Clock::now() - start).count();
}

std::string json_escape(const std::string &value) {
    std::ostringstream output;
    for (unsigned char character : value) {
        switch (character) {
        case '\"':
            output << "\\\"";
            break;
        case '\\':
            output << "\\\\";
            break;
        case '\b':
            output << "\\b";
            break;
        case '\f':
            output << "\\f";
            break;
        case '\n':
            output << "\\n";
            break;
        case '\r':
            output << "\\r";
            break;
        case '\t':
            output << "\\t";
            break;
        default:
            if (character < 0x20)
                output << "\\u" << std::hex << std::setw(4) << std::setfill('0')
                       << static_cast<int>(character) << std::dec << std::setfill(' ');
            else
                output << character;
        }
    }
    return output.str();
}

std::string csv_escape(const std::string &value) {
    if (value.find_first_of(",\"\n\r") == std::string::npos)
        return value;
    std::string escaped = "\"";
    for (char character : value) {
        if (character == '\"')
            escaped += '\"';
        escaped += character;
    }
    escaped += '\"';
    return escaped;
}

void write_json_number(std::ostream &output, double value) {
    if (std::isfinite(value))
        output << std::setprecision(17) << value;
    else
        output << "null";
}

std::vector<std::string> split_tabs(const std::string &line) {
    std::vector<std::string> fields;
    size_t begin = 0;
    while (true) {
        size_t end = line.find('\t', begin);
        fields.push_back(line.substr(begin, end - begin));
        if (end == std::string::npos)
            return fields;
        begin = end + 1;
    }
}

int parse_positive_int(const std::string &value, const char *name) {
    size_t consumed = 0;
    long parsed = std::stol(value, &consumed);
    if (consumed != value.size() || parsed <= 0 || parsed > INT_MAX)
        throw std::invalid_argument(std::string(name) + " must be a positive integer");
    return static_cast<int>(parsed);
}

int parse_nonnegative_int(const std::string &value, const char *name) {
    size_t consumed = 0;
    long parsed = std::stol(value, &consumed);
    if (consumed != value.size() || parsed < 0 || parsed > INT_MAX)
        throw std::invalid_argument(std::string(name) + " must be a nonnegative integer");
    return static_cast<int>(parsed);
}

double parse_positive_double(const std::string &value, const char *name) {
    size_t consumed = 0;
    double parsed = std::stod(value, &consumed);
    if (consumed != value.size() || !std::isfinite(parsed) || parsed <= 0.0)
        throw std::invalid_argument(std::string(name) + " must be a positive finite number");
    return parsed;
}

DeviceSelection parse_device(const std::string &value) {
    if (value == "metal" || value == "gpu")
        return DeviceSelection::metal;
    if (value == "cpu")
        return DeviceSelection::cpu;
    throw std::invalid_argument("device must be metal, gpu, or cpu");
}

mx::Device mlx_device(DeviceSelection selection) {
    return selection == DeviceSelection::metal ? mx::Device::gpu : mx::Device::cpu;
}

const char *device_name(DeviceSelection selection) {
    return selection == DeviceSelection::metal ? "Metal" : "CPU";
}

const char *arithmetic_precision_name(DeviceSelection selection) {
    return selection == DeviceSelection::cpu ? "FP64" : "FP32";
}

void print_usage(const char *program) {
    std::printf(
        "Usage: %s [options]\n"
        "  --data DIR                 LPfeas .mps.gz directory\n"
        "  --manifest FILE            Manifest TSV (default: DIR/manifest.tsv)\n"
        "  --reference-objectives FILE  Optional name/objective TSV (auto-detected in DIR)\n"
        "  --instance NAME            Run one instance; repeat to select several\n"
        "  --output-prefix PATH        Write PATH.csv and PATH.json\n"
        "  --device DEVICE             Solver device: metal/gpu or cpu (default metal)\n"
        "  --jobs N|auto               Workers (auto: Netlib parallel, LPfeas serial)\n"
        "  --tolerance VALUE           Feasibility/optimality tolerance (default 1e-4)\n"
        "  --solver-tolerance VALUE    Internal backend target (default 0.5 * tolerance)\n"
        "  --load-warm-start FILE      Resume original model from an x/y/z checkpoint\n"
        "  --save-solution FILE        Save selected x/y/z certificate for reuse\n"
        "  --time-limit SECONDS        Per-attempt PDHG limit (default 1000)\n"
        "  --iteration-limit N         Per-attempt PDHG limit; 0 skips device PDHG\n"
        "  --correction-iteration-limit N  Warm-start correction limit (default 200000)\n"
        "  --correction-time-limit SECONDS Warm-start correction limit (default 300)\n"
        "  --evaluation-frequency N    Termination cadence (default 200)\n"
        "  --sv-max-iterations N       Power-method limit (default 200)\n"
        "  --curtis-reid-iterations N  Curtis-Reid passes; 0 disables (default 20)\n"
        "  --restart-policy N        0=cuPDLPx PID weight restart (default),\n"
        "                             1=HPR-LP movement-ratio sigma update\n"
        "  --feasibility-polishing     Run a separate primal-feasibility phase (default)\n"
        "  --no-feasibility-polishing  Disable the guarded feasibility phase\n"
        "  --host-double-polishing     Run bounded reduced/original fp64 correction (default)\n"
        "  --no-host-double-polishing  Disable the host fp64 correction\n"
        "  --host-double-handoff       Hand stalled Metal FP32 checkpoints to host FP64 (default)\n"
        "  --no-host-double-handoff    Keep Metal FP32 running until another stopping condition\n"
        "  --host-double-iteration-limit N  Host fp64 correction limit (default 50000)\n"
        "  --host-double-time-limit SECONDS Host fp64 correction limit (default 30)\n"
        "  --no-presolve               Disable PSLP (non-official diagnostic run)\n"
        "  --singleton-columns         Enable PSLP singleton-column reductions (off by default)\n"
        "  --no-singleton-columns      Disable PSLP singleton-column reductions\n"
        "  --no-doubleton-equations    Disable PSLP doubleton-equation reductions\n"
        "  --no-parallel-rows          Disable PSLP parallel-row reductions\n"
        "  --no-parallel-columns       Disable PSLP parallel-column reductions\n"
        "  --no-presolve-dual-fix      Disable PSLP simple dual fixing\n"
        "  --no-bound-tightening       Disable PSLP finite-bound tightening\n"
        "  --no-primal-propagation     Disable PSLP's aggressive propagation pass\n"
        "  --no-propagation-retry      Do not retry an aggressive PSLP audit safely\n"
        "  --no-warm-start-correction  Do not reuse an aggressively presolved primal point\n"
        "  --no-presolve-retry         Do not retry a failed postsolve audit without PSLP\n"
        "  --no-scaling-retry          Do not retry a failed audit without Curtis-Reid\n"
        "  --cold-start                Include first Metal kernel compilation\n"
        "  --fail-on-validation        Exit nonzero if float64 verification fails\n"
        "  --verbose                   Enable solver iteration logs\n"
        "  --help                      Show this help\n",
        program);
}

Options parse_options(int argc, char **argv) {
    Options options;
    for (int index = 1; index < argc; ++index) {
        const std::string argument = argv[index];
        auto take_value = [&]() -> std::string {
            if (++index >= argc)
                throw std::invalid_argument(argument + " requires a value");
            return argv[index];
        };
        if (argument == "--data")
            options.data_directory = take_value();
        else if (argument == "--manifest")
            options.manifest = fs::path(take_value());
        else if (argument == "--reference-objectives")
            options.reference_objectives = fs::path(take_value());
        else if (argument == "--instance")
            options.instances.insert(take_value());
        else if (argument == "--output-prefix")
            options.output_prefix = take_value();
        else if (argument == "--device")
            options.device = parse_device(take_value());
        else if (argument == "--jobs") {
            const std::string value = take_value();
            if (value == "auto") {
                options.jobs = 0;
                options.jobs_auto = true;
            } else {
                options.jobs = parse_positive_int(value, "jobs");
                options.jobs_auto = false;
            }
        }
        else if (argument == "--tolerance")
            options.tolerance = parse_positive_double(take_value(), "tolerance");
        else if (argument == "--solver-tolerance")
            options.solver_tolerance =
                parse_positive_double(take_value(), "solver tolerance");
        else if (argument == "--load-warm-start")
            options.load_warm_start = fs::path(take_value());
        else if (argument == "--save-solution")
            options.save_solution = fs::path(take_value());
        else if (argument == "--time-limit")
            options.time_limit_seconds = parse_positive_double(take_value(), "time limit");
        else if (argument == "--iteration-limit")
            options.iteration_limit =
                parse_nonnegative_int(take_value(), "iteration limit");
        else if (argument == "--correction-iteration-limit")
            options.warm_start_correction_iteration_limit =
                parse_positive_int(take_value(), "correction iteration limit");
        else if (argument == "--correction-time-limit")
            options.warm_start_correction_time_limit_seconds =
                parse_positive_double(take_value(), "correction time limit");
        else if (argument == "--evaluation-frequency")
            options.evaluation_frequency =
                parse_positive_int(take_value(), "evaluation frequency");
        else if (argument == "--sv-max-iterations")
            options.singular_value_iterations =
                parse_positive_int(take_value(), "singular value iterations");
        else if (argument == "--curtis-reid-iterations")
            options.curtis_reid_iterations =
                parse_nonnegative_int(take_value(), "Curtis-Reid iterations");
        else if (argument == "--restart-policy")
            options.restart_policy =
                parse_nonnegative_int(take_value(), "restart policy");
        else if (argument == "--feasibility-polishing")
            options.feasibility_polishing = true;
        else if (argument == "--no-feasibility-polishing")
            options.feasibility_polishing = false;
        else if (argument == "--host-double-polishing")
            options.host_double_polishing = true;
        else if (argument == "--no-host-double-polishing")
            options.host_double_polishing = false;
        else if (argument == "--host-double-handoff")
            options.host_double_early_handoff = true;
        else if (argument == "--no-host-double-handoff")
            options.host_double_early_handoff = false;
        else if (argument == "--host-double-iteration-limit")
            options.host_double_polishing_iteration_limit =
                parse_positive_int(take_value(), "host-double iteration limit");
        else if (argument == "--host-double-time-limit")
            options.host_double_polishing_time_limit_seconds =
                parse_positive_double(take_value(), "host-double time limit");
        else if (argument == "--no-presolve")
            options.presolve = false;
        else if (argument == "--singleton-columns")
            options.presolve_singleton_columns = true;
        else if (argument == "--no-singleton-columns")
            options.presolve_singleton_columns = false;
        else if (argument == "--no-doubleton-equations")
            options.presolve_doubleton_equations = false;
        else if (argument == "--no-parallel-rows")
            options.presolve_parallel_rows = false;
        else if (argument == "--no-parallel-columns")
            options.presolve_parallel_columns = false;
        else if (argument == "--no-presolve-dual-fix")
            options.presolve_dual_fix = false;
        else if (argument == "--no-bound-tightening")
            options.presolve_finite_bound_tightening = false;
        else if (argument == "--no-primal-propagation")
            options.presolve_primal_propagation = false;
        else if (argument == "--no-propagation-retry")
            options.retry_without_primal_propagation = false;
        else if (argument == "--no-warm-start-correction")
            options.warm_start_correction = false;
        else if (argument == "--no-presolve-retry")
            options.retry_without_presolve = false;
        else if (argument == "--no-scaling-retry")
            options.retry_without_curtis_reid = false;
        else if (argument == "--cold-start")
            options.warm_up = false;
        else if (argument == "--fail-on-validation")
            options.fail_on_validation = true;
        else if (argument == "--verbose")
            options.verbose = true;
        else if (argument == "--help") {
            print_usage(argv[0]);
            std::exit(0);
        } else {
            throw std::invalid_argument("unknown option: " + argument);
        }
    }
    if (options.evaluation_frequency < 2)
        throw std::invalid_argument("evaluation frequency must be at least 2");
    if (!options.manifest)
        options.manifest = options.data_directory / "manifest.tsv";
    if (!options.reference_objectives) {
        const fs::path candidate =
            options.data_directory / "reference_objectives.tsv";
        if (fs::exists(candidate))
            options.reference_objectives = candidate;
    }
    return options;
}

std::vector<ManifestEntry> read_manifest(const fs::path &path,
                                         const std::set<std::string> &selection) {
    std::ifstream input(path);
    if (!input)
        throw std::runtime_error("cannot open manifest: " + path.string());
    std::vector<ManifestEntry> entries;
    std::set<std::string> found;
    std::string line;
    while (std::getline(input, line)) {
        if (line.empty())
            continue;
        std::vector<std::string> fields = split_tabs(line);
        if (fields[0] == "name")
            continue;
        if (fields.size() != 6)
            throw std::runtime_error("malformed manifest row: " + line);
        if (!selection.empty() && !selection.contains(fields[0]))
            continue;
        entries.push_back(ManifestEntry{fields[0], parse_positive_int(fields[1], "rows"),
                                        parse_positive_int(fields[2], "columns"),
                                        parse_positive_int(fields[3], "nonzeros"), fields[4],
                                        fields[5], std::nullopt});
        found.insert(fields[0]);
    }
    for (const std::string &name : selection) {
        if (!found.contains(name))
            throw std::runtime_error("instance is absent from manifest: " + name);
    }
    if (entries.empty())
        throw std::runtime_error("manifest selection is empty");
    return entries;
}

void apply_reference_objectives(const fs::path &path,
                                std::vector<ManifestEntry> &entries) {
    std::ifstream input(path);
    if (!input)
        throw std::runtime_error("cannot open reference objectives: " + path.string());
    std::map<std::string, double> references;
    std::string line;
    while (std::getline(input, line)) {
        if (line.empty())
            continue;
        const std::vector<std::string> fields = split_tabs(line);
        if (fields[0] == "name")
            continue;
        if (fields.size() != 2)
            throw std::runtime_error("malformed reference objective row: " + line);
        size_t consumed = 0;
        const double objective = std::stod(fields[1], &consumed);
        if (consumed != fields[1].size() || !std::isfinite(objective))
            throw std::runtime_error("invalid reference objective row: " + line);
        references[fields[0]] = objective;
    }
    for (ManifestEntry &entry : entries) {
        const auto found = references.find(entry.name);
        if (found == references.end()) {
            throw std::runtime_error("reference objective is absent for instance: " +
                                     entry.name);
        }
        entry.reference_objective = found->second;
    }
}

std::string host_name() {
    char buffer[256] = {};
    return gethostname(buffer, sizeof(buffer) - 1) == 0 ? buffer : "unknown";
}

std::string os_description() {
    struct utsname info {};
    if (uname(&info) != 0)
        return "unknown";
    return std::string(info.sysname) + " " + info.release + " " + info.machine;
}

std::string hardware_model() {
#ifdef __APPLE__
    size_t size = 0;
    if (sysctlbyname("hw.model", nullptr, &size, nullptr, 0) == 0 && size > 1) {
        std::vector<char> value(size);
        if (sysctlbyname("hw.model", value.data(), &size, nullptr, 0) == 0)
            return value.data();
    }
#endif
    return "unknown";
}

uint64_t physical_memory_bytes() {
#ifdef __APPLE__
    uint64_t bytes = 0;
    size_t size = sizeof(bytes);
    if (sysctlbyname("hw.memsize", &bytes, &size, nullptr, 0) == 0)
        return bytes;
#endif
    const long pages = sysconf(_SC_PHYS_PAGES);
    const long page_size = sysconf(_SC_PAGESIZE);
    if (pages > 0 && page_size > 0)
        return static_cast<uint64_t>(pages) * static_cast<uint64_t>(page_size);
    return 0;
}

bool is_netlib_suite(const Options &options) {
    auto directory_name = [](fs::path path) {
        path = path.lexically_normal();
        fs::path name = path.filename();
        if (name.empty())
            name = path.parent_path().filename();
        return name;
    };
    if (directory_name(options.data_directory) == "netlib")
        return true;
    return options.manifest &&
           directory_name(options.manifest->parent_path()) == "netlib";
}

const char *benchmark_suite_name(const Options &options) {
    return is_netlib_suite(options) ? "Netlib" : "LPfeas";
}

int resolve_worker_count(const Options &options,
                         const std::vector<ManifestEntry> &entries) {
    if (entries.size() <= 1)
        return 1;
    if (!options.jobs_auto)
        return std::min(options.jobs, static_cast<int>(entries.size()));

    // LPfeas models are intentionally diagnosed one at a time. Their retry
    // portfolios are long, heterogeneous experiments, and concurrent Metal
    // command queues obscure both convergence and per-case timing. Netlib is
    // the first-class parallel regression corpus; an explicit --jobs N still
    // permits deliberate concurrency experiments on any manifest.
    if (!is_netlib_suite(options))
        return 1;

    const unsigned logical_cpus =
        std::max(1u, std::thread::hardware_concurrency());
    // CPU MLX/Accelerate kernels already fan out across the machine, so two
    // independent solves provide useful overlap without oversubscribing their
    // internal pools. Metal spends substantial time in host dispatch and
    // convergence checks; more independent command streams fill those gaps.
    // Twelve was the throughput knee on a 16-core M3 Max. Retain two host
    // cores for parsing, PSLP, and float64 audits on larger machines.
    unsigned compute_cap =
        options.device == DeviceSelection::metal
            ? std::min(12u, std::max(1u, logical_cpus > 2 ? logical_cpus - 2 : 1u))
            : std::min(2u, std::max(1u, logical_cpus / 4));

    long double largest_estimated_bytes = 1.0;
    for (const ManifestEntry &entry : entries) {
        const long double matrix_bytes =
            static_cast<long double>(entry.expected_nonzeros) *
            (options.device == DeviceSelection::cpu ? 128.0L : 96.0L);
        const long double vector_bytes =
            static_cast<long double>(entry.expected_rows + entry.expected_columns) *
            (options.device == DeviceSelection::cpu ? 224.0L : 192.0L);
        largest_estimated_bytes =
            std::max(largest_estimated_bytes, matrix_bytes + vector_bytes);
    }
    if (options.device == DeviceSelection::metal) {
        // Small Netlib cases benefit from many independent streams, but a
        // dozen million-nonzero LPFeas trajectories can leave hours of queued
        // GPU work behind nominal 300-second solver limits. Reduce stream
        // concurrency as the largest resident solver state grows. The byte
        // estimate is intentionally conservative and is also used below for
        // the physical-memory bound.
        constexpr long double mib = 1024.0L * 1024.0L;
        unsigned workload_cap = 12;
        if (largest_estimated_bytes > 1024.0L * mib)
            workload_cap = 1;
        else if (largest_estimated_bytes > 256.0L * mib)
            workload_cap = 2;
        else if (largest_estimated_bytes > 64.0L * mib)
            workload_cap = 4;
        compute_cap = std::min(compute_cap, workload_cap);
    }
    unsigned memory_cap = compute_cap;
    const uint64_t physical_bytes = physical_memory_bytes();
    if (physical_bytes > 0) {
        const long double budget = 0.5L * physical_bytes;
        memory_cap = std::max(
            1u, static_cast<unsigned>(std::min<long double>(
                    compute_cap, std::floor(budget / largest_estimated_bytes))));
    }
    return static_cast<int>(
        std::min<size_t>(entries.size(), std::min(compute_cap, memory_cap)));
}

std::string utc_timestamp() {
    const std::time_t now = std::time(nullptr);
    std::tm utc {};
    gmtime_r(&now, &utc);
    char buffer[32] = {};
    std::strftime(buffer, sizeof(buffer), "%Y-%m-%dT%H:%M:%SZ", &utc);
    return buffer;
}

void warm_up_metal() {
    // Cross the sparse-selection threshold and include both adaptive-kernel
    // branches: row zero is long, while the remaining rows are packed short
    // rows. Each worker calls this on its own thread-local Metal stream before
    // timed instances begin.
    constexpr int rows = 65;
    constexpr int columns = 128;
    std::vector<int> row_ptr(static_cast<size_t>(rows) + 1, 0);
    std::vector<int> col_ind;
    std::vector<double> values;
    col_ind.reserve(129);
    values.reserve(129);
    for (int column = 0; column < 65; ++column) {
        col_ind.push_back(column);
        values.push_back(1.0);
    }
    row_ptr[1] = static_cast<int>(col_ind.size());
    for (int row = 1; row < rows; ++row) {
        col_ind.push_back(63 + row);
        values.push_back(1.0);
        row_ptr[static_cast<size_t>(row) + 1] =
            static_cast<int>(col_ind.size());
    }
    std::vector<double> variable_lb(columns, 0.0);
    std::vector<double> variable_ub(columns, 1.0);
    std::vector<double> constraint_lb(rows, 0.0);
    std::vector<double> constraint_ub(rows, 1.0);
    std::vector<double> objective(columns, 1.0);
    pdhg_parameters_t parameters;
    mlxpdlp_set_default_parameters(&parameters);
    parameters.verbose = false;
    parameters.presolve = false;
    parameters.termination_evaluation_frequency = 2;
    parameters.termination_criteria.eps_optimal_relative = 0.0;
    parameters.termination_criteria.eps_feasible_relative = 0.0;
    parameters.termination_criteria.iteration_limit = 2;
    parameters.termination_criteria.time_sec_limit = 60.0;
    parameters.sv_max_iter = 2;
    MlxPdlpSolver solver(columns, rows, row_ptr.data(), col_ind.data(), values.data(),
                         variable_lb.data(), variable_ub.data(), constraint_lb.data(),
                         constraint_ub.data(), objective.data(), 0.0, &parameters,
                         mx::Device::gpu);
    if (!solver.expects_sparse_metal_backend())
        throw std::runtime_error("Metal warmup did not select the sparse backend");
    ResultPtr result(solver.solve(), mlxpdlp_result_free);
    mx::synchronize(solver.state().stream);
    if (!solver.state().sparse_metal_active || !result)
        throw std::runtime_error("Metal sparse warmup failed");
}

double validation_merit(const RunRecord &record) {
    if (!record.completed || !record.validation.dimensions_match || !record.validation.finite)
        return std::numeric_limits<double>::infinity();
    double merit = std::max({record.validation.relative_primal_residual,
                             record.validation.relative_dual_residual,
                             record.validation.relative_objective_gap,
                             record.validation.relative_variable_bound_violation,
                             record.validation.relative_dual_bound_violation});
    if (record.reference_objective_available)
        merit = std::max(merit, record.reference_objective_relative_error);
    return merit;
}

RunRecord run_attempt(const ManifestEntry &entry, const fs::path &instance_path,
                      const mlxpdlp_mps_problem_t &problem,
                      const std::vector<double> &objective, double objective_constant,
                      const Options &options, bool presolve, bool primal_propagation,
                      const double *primal_start, const double *dual_start,
                      const double *reduced_cost_start,
                      double time_limit_seconds,
                      int iteration_limit, int curtis_reid_iterations,
                      bool allow_early_handoff) {
    RunRecord record;
    record.manifest = entry;
    record.path = instance_path.string();
    record.rows = problem.num_constraints;
    record.columns = problem.num_variables;
    record.nonzeros = problem.num_nonzeros;
    record.selected_presolve = presolve;
    record.selected_primal_propagation = presolve && primal_propagation;
    record.selected_warm_start = primal_start != nullptr || dual_start != nullptr ||
                                 reduced_cost_start != nullptr;
    record.selected_curtis_reid_iterations = curtis_reid_iterations;
    record.attempts = 1;

    pdhg_parameters_t parameters;
    mlxpdlp_set_default_parameters(&parameters);
    parameters.verbose = options.verbose;
    parameters.presolve = presolve;
    parameters.presolve_singleton_columns = options.presolve_singleton_columns;
    parameters.presolve_doubleton_equations = options.presolve_doubleton_equations;
    parameters.presolve_parallel_rows = options.presolve_parallel_rows;
    parameters.presolve_parallel_columns = options.presolve_parallel_columns;
    parameters.presolve_dual_fix = options.presolve_dual_fix;
    parameters.presolve_finite_bound_tightening =
        options.presolve_finite_bound_tightening;
    parameters.presolve_primal_propagation = presolve && primal_propagation;
    parameters.curtis_reid_iterations = curtis_reid_iterations;
    parameters.restart_policy = options.restart_policy;
    parameters.l_inf_ruiz_iterations = 10;
    parameters.has_pock_chambolle_alpha = true;
    parameters.pock_chambolle_alpha = 1.0;
    parameters.bound_objective_rescaling = true;
    parameters.feasibility_polishing = options.feasibility_polishing;
    parameters.host_double_polishing = options.host_double_polishing;
    // A warm attempt is already the audit-driven correction path. Let it run
    // to its own stopping criterion instead of repeatedly selecting the same
    // merely polishable checkpoint that caused the preceding audit failure.
    parameters.host_double_early_handoff =
        options.host_double_early_handoff && allow_early_handoff &&
        primal_start == nullptr &&
        dual_start == nullptr;
    parameters.host_double_polishing_iteration_limit =
        options.host_double_polishing_iteration_limit;
    parameters.host_double_polishing_time_sec_limit =
        options.host_double_polishing_time_limit_seconds;
    parameters.termination_evaluation_frequency = options.evaluation_frequency;
    parameters.sv_max_iter = options.singular_value_iterations;
    parameters.sv_tol = 1e-4;
    parameters.optimality_norm = NORM_TYPE_L2;
    const double solver_tolerance = effective_solver_tolerance(options);
    parameters.termination_criteria.eps_optimal_relative = solver_tolerance;
    parameters.termination_criteria.eps_feasible_relative = solver_tolerance;
    parameters.termination_criteria.eps_feas_polish_relative =
        std::min(solver_tolerance, 1e-6);
    parameters.termination_criteria.time_sec_limit = time_limit_seconds;
    parameters.termination_criteria.iteration_limit = iteration_limit;

    const mx::Device device = mlx_device(options.device);
    const auto setup_start = Clock::now();
    MlxPdlpSolver solver(problem.num_variables, problem.num_constraints, problem.row_ptr,
                         problem.col_ind, problem.values, problem.variable_lb,
                         problem.variable_ub, problem.constraint_lb,
                         problem.constraint_ub, objective.data(), objective_constant,
                         &parameters, primal_start, dual_start, reduced_cost_start,
                         device);
    if (solver.state().stream.device != device)
        throw std::runtime_error("solver did not retain the requested device");
    const bool expect_cpu_double = options.device == DeviceSelection::cpu;
    if (solver.state().cpu_double_precision_active != expect_cpu_double ||
        solver.state().obj.dtype() != (expect_cpu_double ? mx::float64 : mx::float32)) {
        throw std::runtime_error("solver arithmetic precision does not match the requested device");
    }
    mx::synchronize(solver.state().stream);
    record.setup_seconds = seconds_since(setup_start);

    const auto solve_start = Clock::now();
    ResultPtr result(solver.solve(), mlxpdlp_result_free);
    mx::synchronize(solver.state().stream);
    record.solve_seconds = seconds_since(solve_start);
    if (!result)
        throw std::runtime_error("solver returned no result");

    record.completed = true;
    record.termination = benchmark::termination_reason_name(result->termination_reason);
    record.sparse_metal = solver.state().sparse_metal_active;
    record.sparse_cpu = solver.state().sparse_cpu_active;
    record.cpu_double_precision = solver.state().cpu_double_precision_active;
    record.reduced_rows = presolve ? result->num_reduced_constraints : result->num_constraints;
    record.reduced_columns = presolve ? result->num_reduced_variables : result->num_variables;
    record.reduced_nonzeros = presolve ? result->num_reduced_nonzeros : result->num_nonzeros;
    record.iterations = result->total_count;
    record.feasibility_iterations = result->feasibility_iteration;
    record.host_double_iterations = result->host_double_polishing_iteration;
    record.host_double_handoff = result->host_double_handoff;
    record.presolve_seconds = result->presolve_time;
    record.rescaling_seconds = result->rescaling_time_sec;
    record.feasibility_polishing_seconds = result->feasibility_polishing_time;
    record.host_double_polishing_seconds = result->host_double_polishing_time;
    record.solver_reported_seconds = result->cumulative_time_sec;
    record.solver_relative_primal_residual = result->relative_primal_residual;
    record.solver_relative_dual_residual = result->relative_dual_residual;
    record.solver_relative_objective_gap = result->relative_objective_gap;
    record.primal_solution.assign(result->primal_solution,
                                  result->primal_solution + result->num_variables);
    record.dual_solution.assign(result->dual_solution,
                                result->dual_solution + result->num_constraints);
    record.reduced_cost.assign(result->reduced_cost,
                               result->reduced_cost + result->num_variables);

    const auto verification_start = Clock::now();
    record.validation = benchmark::validate_original_problem(problem, *result, objective.data(),
                                                             objective_constant);
    record.verification_seconds = seconds_since(verification_start);
    record.original_primal_objective =
        problem.maximize ? -record.validation.primal_objective : record.validation.primal_objective;
    record.original_dual_objective =
        problem.maximize ? -record.validation.dual_objective : record.validation.dual_objective;
    // Netlib's published optima use the variable-dependent objective and do
    // not include RHS entries on the objective row (E226 has a +7.113 MPS
    // reporting offset).  Preserve the standards-compliant solver objective
    // above while comparing the published benchmark on the same convention.
    record.objective_without_constant =
        record.original_primal_objective - problem.objective_constant;
    record.dual_objective_without_constant =
        record.original_dual_objective - problem.objective_constant;

    if (entry.reference_objective) {
        record.reference_objective_available = true;
        record.reference_objective = *entry.reference_objective;
        const double denominator = 1.0 + std::fabs(record.reference_objective);
        record.reference_objective_relative_error =
            std::max(std::fabs(record.objective_without_constant -
                               record.reference_objective),
                     std::fabs(record.dual_objective_without_constant -
                               record.reference_objective)) /
            denominator;
    }

    if (options.device == DeviceSelection::metal &&
        solver.expects_sparse_metal_backend() && !record.sparse_metal)
        record.error = "Metal sparse backend was not active";
    if (options.device == DeviceSelection::cpu &&
        solver.expects_sparse_cpu_backend() && !record.sparse_cpu)
        record.error = "CPU sparse backend was not active";
    const bool reference_matches =
        !record.reference_objective_available ||
        (std::isfinite(record.reference_objective_relative_error) &&
         record.reference_objective_relative_error <= options.tolerance);
    record.verified = record.error.empty() &&
                      record.validation.satisfies(options.tolerance) &&
                      reference_matches;
    return record;
}

RunRecord run_instance(const ManifestEntry &entry, Options options) {
    RunRecord record;
    record.manifest = entry;
    const fs::path instance_path = options.data_directory / (entry.name + ".mps.gz");
    record.path = instance_path.string();
    const auto total_start = Clock::now();
    try {
        const auto parse_start = Clock::now();
        ProblemPtr problem(mlxpdlp_mps_problem_load(instance_path.c_str()));
        const double parse_seconds = seconds_since(parse_start);
        if (!problem)
            throw std::runtime_error("failed to load MPS");

        std::vector<double> objective(problem->objective,
                                      problem->objective + problem->num_variables);
        if (entry.reference_objective) {
            const int objective_nonzeros = static_cast<int>(
                std::count_if(objective.begin(), objective.end(),
                              [](double value) { return value != 0.0; }));
            const int imported_rows = problem->num_constraints + 1;
            const int imported_nonzeros = problem->num_nonzeros + objective_nonzeros;
            if (problem->num_variables != entry.expected_columns ||
                imported_rows != entry.expected_rows ||
                imported_nonzeros != entry.expected_nonzeros) {
                std::ostringstream message;
                message << "imported dimensions differ from manifest: got rows="
                        << imported_rows << " columns=" << problem->num_variables
                        << " nonzeros=" << imported_nonzeros << ", expected rows="
                        << entry.expected_rows << " columns=" << entry.expected_columns
                        << " nonzeros=" << entry.expected_nonzeros;
                throw std::runtime_error(message.str());
            }
        }
        double objective_constant = problem->objective_constant;
        if (problem->maximize) {
            for (double &coefficient : objective)
                coefficient = -coefficient;
            objective_constant = -objective_constant;
        }

        std::optional<WarmStartData> loaded_warm_start;
        if (options.load_warm_start) {
            loaded_warm_start = read_warm_start(
                *options.load_warm_start, problem->num_variables,
                problem->num_constraints);
        }

        double aggregate_setup = 0.0;
        double aggregate_solve = 0.0;
        double aggregate_verification = 0.0;
        double aggregate_presolve = 0.0;
        double aggregate_rescaling = 0.0;
        double aggregate_polishing = 0.0;
        double aggregate_host_double_polishing = 0.0;
        double aggregate_solver_time = 0.0;
        int attempts = 0;
        bool have_candidate = false;
        bool last_attempt_handoff = false;
        bool propagation_fallback_attempted = false;
        bool warm_start_correction_attempted = false;
        bool host_handoff_maturity_retry_attempted = false;
        bool presolve_fallback_attempted = false;
        bool scaling_fallback_attempted = false;
        bool restart_policy_fallback_attempted = false;
        std::vector<std::string> attempt_errors;
        std::vector<double> correction_primal;
        std::vector<double> correction_dual;
        std::vector<double> correction_reduced_cost;
        double correction_seed_merit = std::numeric_limits<double>::infinity();
        bool correction_dual_reusable = false;
        // Gross mapped misses should bypass a propagation-only retry, but a
        // finite dual may still provide a useful direction for original-model
        // correction. On neos CPU, the ~0.83 mapped dual nearly converges in
        // one correction budget while a zero seed is materially worse. Metal's
        // ~1.26 mapped dual remains too damaged and is discarded.
        constexpr double direct_correction_dual_threshold = 0.5;
        constexpr double mapped_dual_reuse_limit = 1.0;

        auto attempt_label = [&](bool presolve, bool primal_propagation,
                                 int curtis_reid_iterations) {
            const std::string presolve_label =
                presolve ? (primal_propagation ? "presolve/propagate" : "presolve/safe")
                          : "no-presolve";
            std::string label = presolve_label + ", CR=" +
                               std::to_string(curtis_reid_iterations);
            if (options.restart_policy == 1) {
                label += ", HPR";
            }
            return label;
        };
        auto execute_attempt = [&](bool presolve, bool primal_propagation,
                                   int curtis_reid_iterations,
                                   const std::vector<double> *primal_start,
                                   const std::vector<double> *dual_start,
                                   const std::vector<double> *reduced_cost_start,
                                   bool use_correction_limits,
                                   bool allow_early_handoff) {
            ++attempts;
            last_attempt_handoff = false;
            try {
                RunRecord candidate =
                    run_attempt(entry, instance_path, *problem, objective, objective_constant,
                                options, presolve, primal_propagation,
                                primal_start ? primal_start->data() : nullptr,
                                dual_start ? dual_start->data() : nullptr,
                                reduced_cost_start && !reduced_cost_start->empty()
                                    ? reduced_cost_start->data()
                                    : nullptr,
                                use_correction_limits
                                    ? options.warm_start_correction_time_limit_seconds
                                    : options.time_limit_seconds,
                                use_correction_limits
                                    ? options.warm_start_correction_iteration_limit
                                    : options.iteration_limit,
                                curtis_reid_iterations, allow_early_handoff);
                last_attempt_handoff = candidate.host_double_handoff;
                progress_printf(
                    "    [%s] attempt %-22s status=%s iter=%d host64=%d "
                    "solve=%.1fs verify=%s audit=(%.3e, %.3e, %.3e) "
                    "solver=(%.3e, %.3e, %.3e)%s\n",
                    entry.name.c_str(),
                    attempt_label(presolve, primal_propagation,
                                  curtis_reid_iterations).c_str(),
                    candidate.termination.c_str(), candidate.iterations,
                    candidate.host_double_iterations, candidate.solve_seconds,
                    candidate.verified ? "PASS" : "FAIL",
                    candidate.validation.relative_primal_residual,
                    candidate.validation.relative_dual_residual,
                    candidate.validation.relative_objective_gap,
                    candidate.solver_relative_primal_residual,
                    candidate.solver_relative_dual_residual,
                    candidate.solver_relative_objective_gap,
                    primal_start ? " warm" : "");
                aggregate_setup += candidate.setup_seconds;
                aggregate_solve += candidate.solve_seconds;
                aggregate_verification += candidate.verification_seconds;
                aggregate_presolve += candidate.presolve_seconds;
                aggregate_rescaling += candidate.rescaling_seconds;
                aggregate_polishing += candidate.feasibility_polishing_seconds;
                aggregate_host_double_polishing +=
                    candidate.host_double_polishing_seconds;
                aggregate_solver_time += candidate.solver_reported_seconds;

                // Preserve the best mapped seed admissible for the solver's
                // host-double handoff. A reusable bounded dual is preferred;
                // within the same dual class, rank the complete audited KKT
                // merit rather than primal residual alone. Using the former
                // 1e-3 gate discarded useful points such as neos at 1.37e-3
                // before the original-model warm correction could run.
                if (presolve && primal_propagation &&
                    candidate.validation.dimensions_match &&
                    candidate.validation.finite &&
                    candidate.validation.relative_primal_residual <=
                        std::max(5e-3, 50.0 * options.tolerance) &&
                    candidate.validation.relative_variable_bound_violation <=
                        std::max(1e-6, 10.0 * options.tolerance)) {
                    const bool bounded_dual =
                        candidate.validation.relative_dual_residual <=
                            mapped_dual_reuse_limit &&
                        std::all_of(candidate.dual_solution.begin(),
                                    candidate.dual_solution.end(), [](double value) {
                                        return std::isfinite(value) &&
                                               std::fabs(value) <= 1e12;
                                    });
                    const bool better_correction_seed =
                        correction_primal.empty() ||
                        (bounded_dual && !correction_dual_reusable) ||
                        (bounded_dual == correction_dual_reusable &&
                         validation_merit(candidate) < correction_seed_merit);
                    if (better_correction_seed) {
                        correction_primal = candidate.primal_solution;
                        correction_dual = bounded_dual
                                              ? candidate.dual_solution
                                              : std::vector<double>(
                                                    static_cast<size_t>(
                                                        problem->num_constraints),
                                                    0.0);
                        correction_reduced_cost =
                            bounded_dual ? candidate.reduced_cost
                                         : std::vector<double>();
                        correction_seed_merit = validation_merit(candidate);
                        correction_dual_reusable = bounded_dual;
                    }
                }

                const bool select =
                    !have_candidate || (candidate.verified && !record.verified) ||
                    (candidate.verified == record.verified &&
                     validation_merit(candidate) < validation_merit(record));
                if (select)
                    record = std::move(candidate);
                have_candidate = true;
            } catch (const std::exception &error) {
                attempt_errors.push_back(attempt_label(presolve, primal_propagation,
                                                       curtis_reid_iterations) +
                                         ": " + error.what());
            }
        };

        auto execute_scaling_family = [&](int curtis_reid_iterations) {
            bool safe_presolve_attempted = false;
            execute_attempt(options.presolve, options.presolve_primal_propagation,
                            curtis_reid_iterations, nullptr, nullptr, nullptr,
                            false, true);
            const bool initial_attempt_handoff = last_attempt_handoff;
            const double material_audit_miss =
                std::max(1e-2, 100.0 * options.tolerance);
            const bool initial_handoff_has_objective_only_miss =
                initial_attempt_handoff && record.validation.finite &&
                record.validation.relative_dual_residual <= options.tolerance &&
                record.validation.relative_objective_gap > material_audit_miss;
            if (!record.verified && initial_handoff_has_objective_only_miss &&
                options.presolve && options.presolve_primal_propagation &&
                options.warm_start_correction && !correction_primal.empty()) {
                // The reduced trajectory has already reached the host-double
                // admission region, but postsolve materially damaged its
                // original-model certificate. Correct that mapped point on the
                // original model before paying for another cold trajectory.
                // pds-100 is the representative case: Metal SpMV is much faster
                // than CPU, while repeating presolve erased that advantage.
                warm_start_correction_attempted = true;
                execute_attempt(false, false, curtis_reid_iterations,
                                &correction_primal, &correction_dual,
                                &correction_reduced_cost, true, false);
            }
            if (!record.verified && initial_attempt_handoff && options.presolve &&
                options.presolve_primal_propagation) {
                // Postsolve rejected a point that was otherwise safe for host
                // correction. Mature the same reduced trajectory before using
                // it as the original-model warm-start seed.
                host_handoff_maturity_retry_attempted = true;
                execute_attempt(options.presolve,
                                options.presolve_primal_propagation,
                                curtis_reid_iterations, nullptr, nullptr,
                                nullptr, false, false);
            }
            const bool prefer_safe_presolve =
                record.validation.finite &&
                (record.validation.relative_dual_residual > material_audit_miss ||
                 record.validation.relative_objective_gap > material_audit_miss);
            const bool mapped_dual_requires_direct_correction =
                record.validation.finite &&
                record.validation.relative_dual_residual >
                    direct_correction_dual_threshold;
            if (!record.verified && mapped_dual_requires_direct_correction &&
                options.presolve &&
                options.presolve_primal_propagation &&
                options.warm_start_correction && !correction_primal.empty()) {
                // Disabling propagation cannot repair an unusable certificate
                // produced by another inverse transform (notably PSLP's
                // parallel-row map on neos). The admissible aggressive primal
                // point is still valuable; retain only a bounded finite dual
                // direction and continue directly on the original model. Less
                // severe mapped misses retain the safe-PSLP-first path used by
                // FINNIS and irish-e.
                warm_start_correction_attempted = true;
                execute_attempt(false, false, curtis_reid_iterations,
                                &correction_primal, &correction_dual,
                                &correction_reduced_cost, true, false);
            }
            if (!record.verified && options.presolve &&
                options.presolve_primal_propagation &&
                options.retry_without_primal_propagation &&
                prefer_safe_presolve) {
                // A badly mapped dual/gap is a poor original-model warm start.
                // Try the less aggressive inverse map first; FINNIS on Metal is
                // both more reliable and orders of magnitude faster this way.
                propagation_fallback_attempted = true;
                safe_presolve_attempted = true;
                execute_attempt(true, false, curtis_reid_iterations, nullptr,
                                nullptr, nullptr, false, true);
            }
            if (!record.verified && options.presolve &&
                options.presolve_primal_propagation &&
                options.warm_start_correction && !warm_start_correction_attempted &&
                !correction_primal.empty()) {
                warm_start_correction_attempted = true;
                execute_attempt(false, false, curtis_reid_iterations,
                                &correction_primal, &correction_dual,
                                &correction_reduced_cost, true, false);
            }
            if (!record.verified && options.presolve &&
                options.presolve_primal_propagation &&
                options.retry_without_primal_propagation &&
                !safe_presolve_attempted) {
                propagation_fallback_attempted = true;
                execute_attempt(true, false, curtis_reid_iterations, nullptr,
                                nullptr, nullptr, false, true);
            }
            if (!record.verified && options.presolve &&
                options.retry_without_presolve) {
                presolve_fallback_attempted = true;
                execute_attempt(false, false, curtis_reid_iterations, nullptr,
                                nullptr, nullptr, false, true);
            }
        };

        // Audit-driven portfolio. Each scaling family tries aggressive PSLP,
        // safe PSLP, then no presolve, stopping as soon as an original-model
        // float64 audit passes.
        if (loaded_warm_start) {
            execute_attempt(false, false, options.curtis_reid_iterations,
                            &loaded_warm_start->primal,
                            &loaded_warm_start->dual,
                            &loaded_warm_start->reduced_cost, false, false);
        } else {
            execute_scaling_family(options.curtis_reid_iterations);
            if (!record.verified && options.retry_without_curtis_reid &&
                options.curtis_reid_iterations > 0) {
                scaling_fallback_attempted = true;
                execute_scaling_family(0);
            }
        }

        // A PID-restarted FP32 trajectory can strand the dual certificate in
        // a poor basin (Netlib FORPLAN on Metal). The HPR movement-ratio
        // sigma update explores a different restart path; re-run the
        // portfolio with it when the cuPDLPx PID portfolio fails the
        // original-model float64 audit.
        if (!record.verified && options.restart_policy == 0) {
            restart_policy_fallback_attempted = true;
            // The warm-start correction gate is per-family: the PID family
            // may have consumed it, but the HPR family benefits from the
            // same audited primal seed (FORPLAN's passing HPR attempt is a
            // warm original-model solve).
            warm_start_correction_attempted = false;
            options.restart_policy = 1;
            execute_scaling_family(options.curtis_reid_iterations);
            if (!record.verified && options.retry_without_curtis_reid &&
                options.curtis_reid_iterations > 0) {
                scaling_fallback_attempted = true;
                execute_scaling_family(0);
            }
            options.restart_policy = 0;
        }

        if (!have_candidate) {
            throw std::runtime_error(attempt_errors.empty()
                                         ? "no benchmark attempt completed"
                                         : "all benchmark attempts failed; " + attempt_errors.front());
        }

        record.fallback_attempted = presolve_fallback_attempted;
        if (presolve_fallback_attempted) {
            record.fallback_reason =
                "alternate presolve state evaluated after float64 audit failure";
        }
        record.propagation_fallback_attempted = propagation_fallback_attempted;
        if (propagation_fallback_attempted) {
            record.propagation_fallback_reason =
                "safe PSLP evaluated after aggressive postsolve audit failure";
        }
        record.warm_start_correction_attempted = warm_start_correction_attempted;
        if (warm_start_correction_attempted) {
            record.warm_start_correction_reason =
                "original model warm-started from audited aggressive PSLP primal point";
        }
        record.host_handoff_maturity_retry_attempted =
            host_handoff_maturity_retry_attempted;
        if (host_handoff_maturity_retry_attempted) {
            record.host_handoff_maturity_retry_reason =
                "same presolved trajectory matured after an early host-handoff audit failure";
        }
        record.scaling_fallback_attempted = scaling_fallback_attempted;
        if (scaling_fallback_attempted) {
            record.scaling_fallback_reason =
                "CR=0 configurations evaluated after float64 audit failure";
        }
        record.restart_policy_fallback_attempted =
            restart_policy_fallback_attempted;
        if (restart_policy_fallback_attempted) {
            record.restart_policy_fallback_reason =
                "HPR restart policy evaluated after PID portfolio audit failure";
        }

        record.manifest = entry;
        record.path = instance_path.string();
        record.parse_seconds = parse_seconds;
        record.setup_seconds = aggregate_setup;
        record.solve_seconds = aggregate_solve;
        record.verification_seconds = aggregate_verification;
        record.presolve_seconds = aggregate_presolve;
        record.rescaling_seconds = aggregate_rescaling;
        record.feasibility_polishing_seconds = aggregate_polishing;
        record.host_double_polishing_seconds = aggregate_host_double_polishing;
        record.solver_reported_seconds = aggregate_solver_time;
        record.attempts = attempts;

        if (!attempt_errors.empty()) {
            record.validation_warning = "portfolio attempt failed: " + attempt_errors.front();
        }

        if (!record.verified) {
            std::string validation_warning;
            if (record.reference_objective_available &&
                record.reference_objective_relative_error > options.tolerance) {
                validation_warning =
                    "audited objective does not match the published reference";
            } else {
                validation_warning =
                    record.termination == "OPTIMAL"
                        ? "solver met its internal stopping criterion but float64 original-model validation failed"
                        : "float64 original-model validation did not meet the requested tolerance";
            }
            if (!record.validation_warning.empty())
                record.validation_warning += "; ";
            record.validation_warning += validation_warning;
            if (options.fail_on_validation) {
                record.error = record.error.empty() ? record.validation_warning
                                                    : record.error + "; " + record.validation_warning;
            }
        }
        if (options.save_solution) {
            write_warm_start(*options.save_solution, record.primal_solution,
                             record.dual_solution, record.reduced_cost);
        }
        record.primal_solution.clear();
        record.dual_solution.clear();
        record.reduced_cost.clear();
    } catch (const std::exception &error) {
        record.error = error.what();
        record.termination = "ERROR";
    }
    record.total_seconds = seconds_since(total_start);
    return record;
}

void write_csv_header(std::ostream &output) {
    output << "name,worker_id,completion_order,termination,verified,attempts,selected_presolve,selected_primal_propagation,selected_warm_start,selected_cr_iterations,"
              "fallback_attempted,fallback_reason,propagation_fallback_attempted,"
              "propagation_fallback_reason,warm_start_correction_attempted,"
              "warm_start_correction_reason,host_handoff_maturity_retry_attempted,"
              "host_handoff_maturity_retry_reason,scaling_fallback_attempted,"
              "scaling_fallback_reason,restart_policy_fallback_attempted,"
              "restart_policy_fallback_reason,error,validation_warning,rows,columns,nonzeros,reduced_rows,"
              "reduced_columns,reduced_nonzeros,iterations,feasibility_iterations,host_double_iterations,host_double_handoff,sparse_metal,sparse_cpu,cpu_double_precision,"
              "parse_seconds,setup_seconds,presolve_seconds,rescaling_seconds,"
              "feasibility_polishing_seconds,host_double_polishing_seconds,solve_seconds,"
              "solver_reported_seconds,verification_seconds,total_seconds,"
              "primal_objective,dual_objective,objective_without_constant,"
              "dual_objective_without_constant,verified_rel_primal,verified_rel_dual,"
              "verified_rel_gap,verified_rel_variable_bounds,verified_rel_dual_bounds,"
              "reference_objective,reference_objective_relative_error,"
              "solver_rel_primal,solver_rel_dual,solver_rel_gap\n";
}

void write_csv_record(std::ostream &output, const RunRecord &record) {
    output << csv_escape(record.manifest.name) << ',' << record.worker_id << ','
           << record.completion_order << ',' << record.termination << ','
           << (record.verified ? "true" : "false") << ',' << record.attempts << ','
           << (record.selected_presolve ? "true" : "false") << ','
           << (record.selected_primal_propagation ? "true" : "false") << ','
           << (record.selected_warm_start ? "true" : "false") << ','
           << record.selected_curtis_reid_iterations << ','
           << (record.fallback_attempted ? "true" : "false") << ','
           << csv_escape(record.fallback_reason) << ','
           << (record.propagation_fallback_attempted ? "true" : "false") << ','
           << csv_escape(record.propagation_fallback_reason) << ','
           << (record.warm_start_correction_attempted ? "true" : "false") << ','
           << csv_escape(record.warm_start_correction_reason) << ','
           << (record.host_handoff_maturity_retry_attempted ? "true" : "false") << ','
           << csv_escape(record.host_handoff_maturity_retry_reason) << ','
           << (record.scaling_fallback_attempted ? "true" : "false") << ','
           << csv_escape(record.scaling_fallback_reason) << ','
           << (record.restart_policy_fallback_attempted ? "true" : "false") << ','
           << csv_escape(record.restart_policy_fallback_reason) << ','
           << csv_escape(record.error) << ','
           << csv_escape(record.validation_warning) << ',' << record.rows << ',' << record.columns
           << ',' << record.nonzeros << ','
           << record.reduced_rows << ',' << record.reduced_columns << ','
           << record.reduced_nonzeros << ',' << record.iterations << ','
           << record.feasibility_iterations << ',' << record.host_double_iterations << ','
           << (record.host_double_handoff ? "true" : "false") << ','
           << (record.sparse_metal ? "true" : "false") << ','
           << (record.sparse_cpu ? "true" : "false") << ','
           << (record.cpu_double_precision ? "true" : "false") << ','
           << std::setprecision(17)
           << record.parse_seconds << ',' << record.setup_seconds << ','
           << record.presolve_seconds << ',' << record.rescaling_seconds << ','
           << record.feasibility_polishing_seconds << ','
           << record.host_double_polishing_seconds << ','
           << record.solve_seconds << ',' << record.solver_reported_seconds << ','
           << record.verification_seconds << ',' << record.total_seconds << ','
           << record.original_primal_objective << ',' << record.original_dual_objective << ','
           << record.objective_without_constant << ','
           << record.dual_objective_without_constant << ','
           << record.validation.relative_primal_residual << ','
           << record.validation.relative_dual_residual << ','
           << record.validation.relative_objective_gap << ','
           << record.validation.relative_variable_bound_violation << ','
           << record.validation.relative_dual_bound_violation << ',';
    if (record.reference_objective_available) {
        output << record.reference_objective << ','
               << record.reference_objective_relative_error << ',';
    } else {
        output << ",,";
    }
    output << record.solver_relative_primal_residual << ','
           << record.solver_relative_dual_residual << ','
           << record.solver_relative_objective_gap << '\n';
}

void write_csv(const fs::path &path, const std::vector<RunRecord> &records) {
    fs::path temporary = path;
    temporary += ".tmp";
    std::ofstream output(temporary, std::ios::trunc);
    if (!output)
        throw std::runtime_error("cannot write CSV report: " + temporary.string());
    write_csv_header(output);
    for (const RunRecord &record : records)
        write_csv_record(output, record);
    output.close();
    if (!output)
        throw std::runtime_error("failed while writing CSV report: " + temporary.string());
    if (std::rename(temporary.c_str(), path.c_str()) != 0)
        throw std::runtime_error("cannot publish CSV report: " + path.string() + ": " +
                                 std::strerror(errno));
}

void write_json(const fs::path &path, const Options &options,
                const std::vector<RunRecord> &records, const std::string &generated_at,
                double sweep_wall_seconds) {
    fs::path temporary = path;
    temporary += ".tmp";
    std::ofstream output(temporary, std::ios::trunc);
    if (!output)
        throw std::runtime_error("cannot write JSON report: " + temporary.string());
    output << "{\n  \"schema_version\": 8,\n"
           << "  \"generated_at_utc\": \"" << json_escape(generated_at) << "\",\n"
           << "  \"solver\": \"mlxPDLP " << MLXPDLP_VERSION_STRING << "\",\n"
           << "  \"suite\": \""
           << (is_netlib_suite(options) ? "netlib" : "lpfeas") << "\",\n"
           << "  \"host\": {\"name\": \"" << json_escape(host_name())
           << "\", \"os\": \"" << json_escape(os_description())
           << "\", \"hardware_model\": \"" << json_escape(hardware_model())
           << "\", \"logical_cpus\": " << std::thread::hardware_concurrency() << "},\n"
           << "  \"protocol\": {\"device\": \"" << device_name(options.device)
           << "\", \"arithmetic_precision\": \""
           << arithmetic_precision_name(options.device)
           << "\", \"jobs\": " << options.jobs
           << ", \"job_selection\": \"" << (options.jobs_auto ? "auto" : "explicit")
           << "\", \"auto_suite_policy\": \""
           << (is_netlib_suite(options) ? "netlib_parallel" : "lpfeas_serial")
           << "\", \"scheduling\": \""
           << (options.jobs > 1 ? "work_stealing_lpt" : "manifest_order") << "\""
           << ", \"presolve\": "
           << (options.presolve ? "true" : "false")
           << ", \"presolve_primal_propagation\": "
           << (options.presolve_primal_propagation ? "true" : "false")
           << ", \"presolve_singleton_columns\": "
           << (options.presolve_singleton_columns ? "true" : "false")
           << ", \"presolve_doubleton_equations\": "
           << (options.presolve_doubleton_equations ? "true" : "false")
           << ", \"presolve_parallel_rows\": "
           << (options.presolve_parallel_rows ? "true" : "false")
           << ", \"presolve_parallel_columns\": "
           << (options.presolve_parallel_columns ? "true" : "false")
           << ", \"presolve_dual_fix\": "
           << (options.presolve_dual_fix ? "true" : "false")
           << ", \"presolve_finite_bound_tightening\": "
           << (options.presolve_finite_bound_tightening ? "true" : "false")
           << ", \"tolerance\": ";
    write_json_number(output, options.tolerance);
    output << ", \"solver_tolerance\": ";
    write_json_number(output, effective_solver_tolerance(options));
    output << ", \"feasibility_polishing_tolerance\": ";
    write_json_number(output, std::min(effective_solver_tolerance(options), 1e-6));
    output << ", \"time_limit_seconds\": ";
    write_json_number(output, options.time_limit_seconds);
    output << ", \"warm_start_correction\": "
           << (options.warm_start_correction ? "true" : "false")
           << ", \"correction_time_limit_seconds\": ";
    write_json_number(output, options.warm_start_correction_time_limit_seconds);
    output << ", \"correction_iteration_limit\": "
           << options.warm_start_correction_iteration_limit;
    output << ", \"time_limit_scope\": \"per_attempt\", \"iteration_limit\": "
           << options.iteration_limit
           << ", \"evaluation_frequency\": " << options.evaluation_frequency
           << ", \"sv_max_iterations\": " << options.singular_value_iterations
           << ", \"sparse_cpu_dense_element_threshold\": 16777216"
           << ", \"curtis_reid_iterations\": " << options.curtis_reid_iterations
           << ", \"ruiz_iterations\": 10, "
              "\"pock_chambolle_alpha\": 1.0, "
              "\"bound_objective_rescaling\": true, \"optimality_norm\": \"L2\", "
              "\"feasibility_polishing\": "
           << (options.feasibility_polishing ? "true" : "false")
           << ", \"host_double_polishing\": "
           << (options.host_double_polishing ? "true" : "false")
           << ", \"host_double_early_handoff\": "
           << (options.host_double_early_handoff ? "true" : "false")
           << ", \"host_double_early_handoff_effective\": "
           << (options.device == DeviceSelection::metal && options.host_double_early_handoff
                   ? "true"
                   : "false")
           << ", \"host_double_early_handoff_on_warm_starts\": false"
           << ", \"host_double_time_limit_seconds\": ";
    write_json_number(output, options.host_double_polishing_time_limit_seconds);
    output << ", \"host_double_iteration_limit\": "
           << options.host_double_polishing_iteration_limit
           << ", \"presolve_retry\": "
           << (options.retry_without_presolve ? "true" : "false")
           << ", \"propagation_retry\": "
           << (options.retry_without_primal_propagation ? "true" : "false")
           << ", \"scaling_retry_without_curtis_reid\": "
           << (options.retry_without_curtis_reid ? "true" : "false")
           << ", \"metal_warmup\": "
           << (options.device == DeviceSelection::metal && options.warm_up ? "true" : "false")
           << ", \"fail_on_float64_validation\": "
           << (options.fail_on_validation ? "true" : "false")
           << "},\n  \"sweep\": {\"completed_instances\": " << records.size()
           << ", \"wall_seconds\": ";
    write_json_number(output, sweep_wall_seconds);
    output << "},\n  \"results\": [\n";
    for (size_t index = 0; index < records.size(); ++index) {
        const RunRecord &record = records[index];
        output << "    {\"name\": \"" << json_escape(record.manifest.name)
               << "\", \"path\": \"" << json_escape(record.path)
               << "\", \"worker_id\": " << record.worker_id
               << ", \"completion_order\": " << record.completion_order
               << ", \"termination\": \"" << record.termination
               << "\", \"verified\": " << (record.verified ? "true" : "false")
               << ", \"attempts\": " << record.attempts
               << ", \"selected_presolve\": "
               << (record.selected_presolve ? "true" : "false")
               << ", \"selected_primal_propagation\": "
               << (record.selected_primal_propagation ? "true" : "false")
               << ", \"selected_warm_start\": "
               << (record.selected_warm_start ? "true" : "false")
               << ", \"selected_curtis_reid_iterations\": "
               << record.selected_curtis_reid_iterations
               << ", \"fallback_attempted\": "
               << (record.fallback_attempted ? "true" : "false")
               << ", \"fallback_reason\": \"" << json_escape(record.fallback_reason) << "\""
               << ", \"propagation_fallback_attempted\": "
               << (record.propagation_fallback_attempted ? "true" : "false")
               << ", \"propagation_fallback_reason\": \""
               << json_escape(record.propagation_fallback_reason) << "\""
               << ", \"warm_start_correction_attempted\": "
               << (record.warm_start_correction_attempted ? "true" : "false")
               << ", \"warm_start_correction_reason\": \""
               << json_escape(record.warm_start_correction_reason) << "\""
               << ", \"host_handoff_maturity_retry_attempted\": "
               << (record.host_handoff_maturity_retry_attempted ? "true" : "false")
               << ", \"host_handoff_maturity_retry_reason\": \""
               << json_escape(record.host_handoff_maturity_retry_reason) << "\""
               << ", \"scaling_fallback_attempted\": "
               << (record.scaling_fallback_attempted ? "true" : "false")
               << ", \"scaling_fallback_reason\": \""
               << json_escape(record.scaling_fallback_reason) << "\""
               << ", \"restart_policy_fallback_attempted\": "
               << (record.restart_policy_fallback_attempted ? "true" : "false")
               << ", \"restart_policy_fallback_reason\": \""
               << json_escape(record.restart_policy_fallback_reason) << "\""
               << ", \"error\": \"" << json_escape(record.error)
               << "\", \"validation_warning\": \""
               << json_escape(record.validation_warning) << "\", "
               << "\"dimensions\": {\"rows\": " << record.rows
               << ", \"columns\": " << record.columns << ", \"nonzeros\": "
               << record.nonzeros << ", \"reduced_rows\": " << record.reduced_rows
               << ", \"reduced_columns\": " << record.reduced_columns
               << ", \"reduced_nonzeros\": " << record.reduced_nonzeros << "}, "
               << "\"iterations\": " << record.iterations
               << ", \"feasibility_iterations\": " << record.feasibility_iterations
               << ", \"host_double_iterations\": " << record.host_double_iterations
               << ", \"host_double_handoff\": "
               << (record.host_double_handoff ? "true" : "false")
               << ", \"sparse_metal\": " << (record.sparse_metal ? "true" : "false")
               << ", \"sparse_cpu\": " << (record.sparse_cpu ? "true" : "false")
               << ", \"cpu_double_precision\": "
               << (record.cpu_double_precision ? "true" : "false")
               << ", \"timing_seconds\": {\"parse\": ";
        write_json_number(output, record.parse_seconds);
        output << ", \"setup\": ";
        write_json_number(output, record.setup_seconds);
        output << ", \"presolve\": ";
        write_json_number(output, record.presolve_seconds);
        output << ", \"rescaling\": ";
        write_json_number(output, record.rescaling_seconds);
        output << ", \"feasibility_polishing\": ";
        write_json_number(output, record.feasibility_polishing_seconds);
        output << ", \"host_double_polishing\": ";
        write_json_number(output, record.host_double_polishing_seconds);
        output << ", \"solve\": ";
        write_json_number(output, record.solve_seconds);
        output << ", \"solver_reported\": ";
        write_json_number(output, record.solver_reported_seconds);
        output << ", \"verification\": ";
        write_json_number(output, record.verification_seconds);
        output << ", \"total\": ";
        write_json_number(output, record.total_seconds);
        output << "}, \"original_model\": {\"primal_objective\": ";
        write_json_number(output, record.original_primal_objective);
        output << ", \"dual_objective\": ";
        write_json_number(output, record.original_dual_objective);
        output << ", \"objective_without_constant\": ";
        write_json_number(output, record.objective_without_constant);
        output << ", \"dual_objective_without_constant\": ";
        write_json_number(output, record.dual_objective_without_constant);
        output << ", \"relative_primal_residual\": ";
        write_json_number(output, record.validation.relative_primal_residual);
        output << ", \"relative_dual_residual\": ";
        write_json_number(output, record.validation.relative_dual_residual);
        output << ", \"relative_objective_gap\": ";
        write_json_number(output, record.validation.relative_objective_gap);
        output << ", \"relative_variable_bound_violation\": ";
        write_json_number(output, record.validation.relative_variable_bound_violation);
        output << ", \"relative_dual_bound_violation\": ";
        write_json_number(output, record.validation.relative_dual_bound_violation);
        output << ", \"reference_objective\": ";
        if (record.reference_objective_available)
            write_json_number(output, record.reference_objective);
        else
            output << "null";
        output << ", \"reference_objective_relative_error\": ";
        if (record.reference_objective_available)
            write_json_number(output, record.reference_objective_relative_error);
        else
            output << "null";
        output << "}, \"solver_model\": {\"relative_primal_residual\": ";
        write_json_number(output, record.solver_relative_primal_residual);
        output << ", \"relative_dual_residual\": ";
        write_json_number(output, record.solver_relative_dual_residual);
        output << ", \"relative_objective_gap\": ";
        write_json_number(output, record.solver_relative_objective_gap);
        output << "}}" << (index + 1 == records.size() ? "\n" : ",\n");
    }
    output << "  ]\n}\n";
    output.close();
    if (!output)
        throw std::runtime_error("failed while writing JSON report: " + temporary.string());
    if (std::rename(temporary.c_str(), path.c_str()) != 0)
        throw std::runtime_error("cannot publish JSON report: " + path.string() + ": " +
                                 std::strerror(errno));
}

} // namespace

int main(int argc, char **argv) {
    try {
        Options options = parse_options(argc, argv);
#ifndef MLXPDLP_LPFEAS_HAS_PRESOLVE
        if (options.presolve)
            throw std::runtime_error(
                "official LPfeas mode requires a build configured with MLXPDLP_BUILD_PRESOLVE=ON");
#endif
        if (options.device == DeviceSelection::metal && !mx::is_available(mx::Device::gpu))
            throw std::runtime_error("MLX Metal device is not available");

        // MLX initializes the per-device default-stream table lazily. That
        // table queries every compiled backend, so allowing several fresh
        // worker threads to initialize it concurrently can race Metal device
        // discovery even during a CPU-only sweep. Prime the selected runtime
        // once on the main thread; workers still receive independent
        // thread-local streams.
        const mx::Device selected_device = mlx_device(options.device);
        // MLX's default device is process-global (the streams themselves are
        // thread-local). Keep a backend-homogeneous sweep pinned for its full
        // lifetime so overlapping StreamContext destructors cannot restore a
        // different device underneath another worker.
        mx::set_default_device(selected_device);
        (void)mx::default_stream(selected_device);

        std::vector<ManifestEntry> entries = read_manifest(*options.manifest, options.instances);
        if (options.reference_objectives)
            apply_reference_objectives(*options.reference_objectives, entries);
        if ((options.load_warm_start || options.save_solution) && entries.size() != 1) {
            throw std::runtime_error(
                "solution checkpoints require exactly one selected instance");
        }
        options.jobs = resolve_worker_count(options, entries);
        if (options.verbose && options.jobs > 1) {
            throw std::runtime_error(
                "--verbose requires --jobs 1 so solver logs remain attributable");
        }
        fs::path csv_path = options.output_prefix;
        csv_path += ".csv";
        fs::path json_path = options.output_prefix;
        json_path += ".json";
        if (!csv_path.parent_path().empty())
            fs::create_directories(csv_path.parent_path());
        if (!json_path.parent_path().empty())
            fs::create_directories(json_path.parent_path());

        std::printf("mlxPDLP %s %s/%s benchmark (%zu instance%s)\n",
                    benchmark_suite_name(options), device_name(options.device),
                    arithmetic_precision_name(options.device), entries.size(),
                    entries.size() == 1 ? "" : "s");
        std::printf("Protocol: PSLP=%s/%s, singleton=%s, retry=%s/%s, warm-correct=%s/%d, "
                    "scale-retry=%s, audit-tol=%.1e, solver-tol=%.1e, time=%.0fs, "
                    "eval=%d, sv-max=%d, CR=%d, polish=%s, host64=%s/%s/%d/%.0fs, "
                    "jobs=%d (%s/%s)\n",
                    options.presolve ? "on" : "off",
                    options.presolve_primal_propagation ? "propagate" : "safe",
                    options.presolve_singleton_columns ? "on" : "off",
                    options.retry_without_primal_propagation ? "safe" : "off",
                    options.retry_without_presolve ? "on" : "off",
                    options.warm_start_correction ? "on" : "off",
                    options.warm_start_correction_iteration_limit,
                    options.retry_without_curtis_reid ? "on" : "off", options.tolerance,
                    effective_solver_tolerance(options),
                    options.time_limit_seconds, options.evaluation_frequency,
                    options.singular_value_iterations, options.curtis_reid_iterations,
                    options.feasibility_polishing ? "on" : "off",
                    options.host_double_polishing ? "on" : "off",
                    options.device == DeviceSelection::metal &&
                            options.host_double_early_handoff
                        ? "handoff"
                        : "no-handoff",
                    options.host_double_polishing_iteration_limit,
                    options.host_double_polishing_time_limit_seconds,
                    options.jobs, options.jobs_auto ? "auto" : "explicit",
                    options.jobs > 1 ? "work-stealing" : "manifest-order");
        if (options.device == DeviceSelection::metal && options.warm_up) {
            std::printf("Warming %d Metal worker stream%s (excluded from instance timings)...\n",
                        options.jobs, options.jobs == 1 ? "" : "s");
        }
        std::fflush(stdout);

        const std::string generated_at = utc_timestamp();
        write_csv(csv_path, {});
        write_json(json_path, options, {}, generated_at, 0.0);

        std::vector<size_t> work_order(entries.size());
        for (size_t index = 0; index < work_order.size(); ++index)
            work_order[index] = index;
        if (options.jobs > 1) {
            std::stable_sort(work_order.begin(), work_order.end(),
                             [&](size_t lhs, size_t rhs) {
                                 const auto structural_work = [&](size_t index) {
                                     const ManifestEntry &entry = entries[index];
                                     return static_cast<int64_t>(entry.expected_nonzeros) * 8 +
                                            entry.expected_rows + entry.expected_columns;
                                 };
                                 return structural_work(lhs) > structural_work(rhs);
                             });
        }

        std::vector<std::deque<size_t>> worker_queues(
            static_cast<size_t>(options.jobs));
        std::vector<std::mutex> queue_mutexes(static_cast<size_t>(options.jobs));
        for (size_t position = 0; position < work_order.size(); ++position) {
            worker_queues[position % static_cast<size_t>(options.jobs)].push_back(
                work_order[position]);
        }

        std::mutex report_mutex;
        std::mutex exception_mutex;
        std::vector<std::optional<RunRecord>> result_slots(entries.size());
        std::vector<RunRecord> completed_records;
        std::atomic<size_t> queued_work{entries.size()};
        std::atomic<int> completion_count{0};
        std::atomic<bool> validation_error{false};
        std::atomic<bool> abort_workers{false};
        std::exception_ptr worker_exception;
        std::latch workers_ready(options.jobs);
        std::latch start_workers(1);
        Clock::time_point sweep_start;

        auto capture_worker_exception = [&](std::exception_ptr error) {
            abort_workers.store(true, std::memory_order_relaxed);
            std::lock_guard<std::mutex> lock(exception_mutex);
            if (!worker_exception)
                worker_exception = std::move(error);
        };

        auto take_work = [&](int worker_index) -> std::optional<size_t> {
            {
                std::lock_guard<std::mutex> lock(
                    queue_mutexes[static_cast<size_t>(worker_index)]);
                auto &own_queue = worker_queues[static_cast<size_t>(worker_index)];
                if (!own_queue.empty()) {
                    const size_t index = own_queue.front();
                    own_queue.pop_front();
                    queued_work.fetch_sub(1, std::memory_order_release);
                    return index;
                }
            }
            // A failed try_lock only means another thief briefly owns that
            // queue, not that the sweep is empty. Retry until every item has
            // actually been claimed; this prevents a transient lock collision
            // from retiring workers early.
            while (queued_work.load(std::memory_order_acquire) > 0) {
                for (int offset = 1; offset < options.jobs; ++offset) {
                    const size_t victim = static_cast<size_t>(
                        (worker_index + offset) % options.jobs);
                    std::unique_lock<std::mutex> lock(queue_mutexes[victim],
                                                      std::try_to_lock);
                    if (lock.owns_lock() && !worker_queues[victim].empty()) {
                        const size_t index = worker_queues[victim].back();
                        worker_queues[victim].pop_back();
                        queued_work.fetch_sub(1, std::memory_order_release);
                        return index;
                    }
                }
                std::this_thread::yield();
            }
            return std::nullopt;
        };

        auto publish_result = [&](size_t index, const RunRecord &record) {
            std::lock_guard<std::mutex> lock(report_mutex);
            result_slots[index] = record;
            completed_records.clear();
            completed_records.reserve(entries.size());
            for (const auto &slot : result_slots) {
                if (slot)
                    completed_records.push_back(*slot);
            }
            write_csv(csv_path, completed_records);
            write_json(json_path, options, completed_records, generated_at,
                       seconds_since(sweep_start));
        };

        auto print_result = [&](const RunRecord &record) {
            const char *selection =
                record.selected_warm_start
                    ? "original-warm"
                    : (record.selected_presolve
                           ? (record.selected_primal_propagation
                                  ? "presolve-propagate"
                                  : "presolve-safe")
                           : "original");
            // A verified fallback-family attempt necessarily belongs to the
            // HPR restart policy: the fallback only runs after every PID
            // attempt failed the audit.
            std::string selection_with_policy = selection;
            if (record.restart_policy_fallback_attempted && record.verified) {
                selection_with_policy += "/HPR";
            }
            progress_printf(
                "[done %d/%zu worker=%d] %s %-15s iter=%d polish=%d host64=%d "
                "solve=%.3fs verify=%s rel=(%.3e, %.3e, %.3e) "
                "attempts=%d selected=%s/CR%d%s%s%s%s\n",
                record.completion_order, entries.size(), record.worker_id,
                record.manifest.name.c_str(), record.termination.c_str(),
                record.iterations, record.feasibility_iterations,
                record.host_double_iterations, record.solve_seconds,
                record.verified ? "PASS" : "FAIL",
                record.validation.relative_primal_residual,
                record.validation.relative_dual_residual,
                record.validation.relative_objective_gap, record.attempts,
                selection_with_policy.c_str(), record.selected_curtis_reid_iterations,
                record.error.empty() ? "" : " error=",
                record.error.empty() ? "" : record.error.c_str(),
                record.validation_warning.empty() || !record.error.empty() ? ""
                                                                           : " warning=",
                record.validation_warning.empty() || !record.error.empty()
                    ? ""
                    : record.validation_warning.c_str());
        };

        std::vector<std::jthread> workers;
        workers.reserve(static_cast<size_t>(options.jobs));
        for (int worker_index = 0; worker_index < options.jobs; ++worker_index) {
            workers.emplace_back([&, worker_index] {
                bool warmup_ok = true;
                try {
                    if (options.device == DeviceSelection::metal && options.warm_up)
                        warm_up_metal();
                } catch (...) {
                    warmup_ok = false;
                    capture_worker_exception(std::current_exception());
                }
                workers_ready.count_down();
                start_workers.wait();
                if (!warmup_ok || abort_workers.load(std::memory_order_relaxed))
                    return;

                while (!abort_workers.load(std::memory_order_relaxed)) {
                    const std::optional<size_t> work = take_work(worker_index);
                    if (!work)
                        return;
                    const size_t index = *work;
                    progress_printf("[start worker=%d manifest=%zu/%zu] %s\n",
                                    worker_index + 1, index + 1, entries.size(),
                                    entries[index].name.c_str());
                    RunRecord record = run_instance(entries[index], options);
                    record.worker_id = worker_index + 1;
                    record.completion_order =
                        completion_count.fetch_add(1, std::memory_order_relaxed) + 1;
                    if (!record.error.empty())
                        validation_error.store(true, std::memory_order_relaxed);
                    try {
                        publish_result(index, record);
                    } catch (...) {
                        capture_worker_exception(std::current_exception());
                        return;
                    }
                    print_result(record);
                }
            });
        }

        workers_ready.wait();
        sweep_start = Clock::now();
        start_workers.count_down();
        for (std::jthread &worker : workers)
            worker.join();
        if (worker_exception)
            std::rethrow_exception(worker_exception);

        const double sweep_wall_seconds = seconds_since(sweep_start);
        {
            std::lock_guard<std::mutex> lock(report_mutex);
            write_csv(csv_path, completed_records);
            write_json(json_path, options, completed_records, generated_at,
                       sweep_wall_seconds);
        }
        std::printf("Sweep: %zu instances in %.3fs with %d worker%s\n",
                    completed_records.size(), sweep_wall_seconds, options.jobs,
                    options.jobs == 1 ? "" : "s");
        std::printf("Reports: %s and %s\n", csv_path.c_str(), json_path.c_str());
        return validation_error.load(std::memory_order_relaxed) ? 1 : 0;
    } catch (const std::exception &error) {
        std::fprintf(stderr, "LPfeas benchmark failed: %s\n", error.what());
        return 2;
    }
}
