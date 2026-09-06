# Run at build time so rebuilding after local edits cannot reuse an old SHA.
find_package(Git QUIET)
set(revision "unknown")
set(mlx_revision "unknown")
set(mlx_revision_status 1)
set(dirty "null")
if(GIT_FOUND)
    execute_process(COMMAND "${GIT_EXECUTABLE}" -C "${SOURCE_DIR}" rev-parse HEAD
        OUTPUT_VARIABLE revision OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET
        RESULT_VARIABLE revision_status)
    execute_process(COMMAND "${GIT_EXECUTABLE}" -C "${SOURCE_DIR}" status --porcelain
        OUTPUT_VARIABLE status ERROR_QUIET RESULT_VARIABLE status_result)
    if(status_result EQUAL 0)
        if(status STREQUAL "")
            set(dirty "false")
        else()
            set(dirty "true")
        endif()
    endif()
    # Installed headers may sit inside the consumer's repository. Git would
    # walk upward and report that unrelated HEAD. Only inspect a checkout
    # rooted at the MLX source directory; .git may be a worktree file.
    foreach(mlx_checkout IN ITEMS "${MLX_SOURCE_DIR}" "${MLX_INCLUDE_DIR}")
        if(NOT EXISTS "${mlx_checkout}/.git" OR
           NOT EXISTS "${mlx_checkout}/mlx/mlx.h")
            continue()
        endif()
        execute_process(COMMAND "${GIT_EXECUTABLE}" -C "${mlx_checkout}" rev-parse HEAD
            OUTPUT_VARIABLE mlx_revision OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET
            RESULT_VARIABLE mlx_revision_status)
        if(mlx_revision_status EQUAL 0 AND NOT mlx_revision STREQUAL "")
            break()
        endif()
    endforeach()
endif()
foreach(name revision mlx_revision)
    if(NOT ${name}_status EQUAL 0 OR "${${name}}" STREQUAL "")
        set(${name} "unknown")
    endif()
endforeach()
# Include uncommitted/new source files in a reproducible source digest. Logs,
# reports and build artifacts are deliberately outside this source manifest.
file(GLOB_RECURSE sources RELATIVE "${SOURCE_DIR}"
    "${SOURCE_DIR}/src/*.cpp" "${SOURCE_DIR}/src/*.c" "${SOURCE_DIR}/src/*.h"
    "${SOURCE_DIR}/include/*.h" "${SOURCE_DIR}/benchmarks/*.cpp"
    "${SOURCE_DIR}/benchmarks/*.h" "${SOURCE_DIR}/cmake/*.cmake")
list(APPEND sources CMakeLists.txt)
list(SORT sources)
set(manifest "")
foreach(source IN LISTS sources)
    file(SHA256 "${SOURCE_DIR}/${source}" digest)
    string(APPEND manifest "${source}:${digest}\n")
endforeach()
string(SHA256 source_digest "${manifest}")
set(content "#pragma once\n#define MLXPDLP_BENCHMARK_GIT_REVISION \"${revision}\"\n#define MLXPDLP_BENCHMARK_GIT_DIRTY \"${dirty}\"\n#define MLXPDLP_BENCHMARK_SOURCE_SHA256 \"${source_digest}\"\n#define MLXPDLP_BENCHMARK_MLX_REVISION \"${mlx_revision}\"\n")
set(previous "")
if(EXISTS "${OUTPUT}")
    file(READ "${OUTPUT}" previous)
endif()
if(NOT content STREQUAL previous)
    file(WRITE "${OUTPUT}" "${content}")
endif()
