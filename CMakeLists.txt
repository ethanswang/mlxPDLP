# Copyright 2026 Ethan Wang <ethanshurui.wang@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

cmake_minimum_required(VERSION 3.25)

project(
    mlxPDLP
    VERSION 0.1.0
    DESCRIPTION "A PDLP solver for Apple MLX CPU and Metal devices"
    LANGUAGES C CXX
)

include(CMakePackageConfigHelpers)
include(CTest)
include(GNUInstallDirs)

option(MLXPDLP_BUILD_MPS "Build the bundled MPS loader (requires Zlib)" ON)
option(MLXPDLP_BUILD_PRESOLVE "Build PSLP presolve and postsolve support" ON)
option(MLXPDLP_BUILD_EXAMPLES "Build mlxPDLP examples" ON)
option(MLXPDLP_BUILD_BENCHMARKS "Build performance benchmark executables" OFF)
option(MLXPDLP_ENABLE_NETLIB_REGRESSION
    "Register the downloaded 40-case Netlib CPU/Metal regression suite" OFF)
option(MLXPDLP_ENABLE_WARNINGS "Enable compiler warnings for mlxPDLP targets" ON)

if(MLXPDLP_ENABLE_NETLIB_REGRESSION AND
   (NOT BUILD_TESTING OR NOT MLXPDLP_BUILD_BENCHMARKS OR
    NOT MLXPDLP_BUILD_MPS OR NOT MLXPDLP_BUILD_PRESOLVE))
    message(FATAL_ERROR
        "MLXPDLP_ENABLE_NETLIB_REGRESSION requires BUILD_TESTING=ON, "
        "MLXPDLP_BUILD_BENCHMARKS=ON, MLXPDLP_BUILD_MPS=ON, and "
        "MLXPDLP_BUILD_PRESOLVE=ON")
endif()

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

list(PREPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
find_package(MLX REQUIRED MODULE)

if(MLXPDLP_BUILD_PRESOLVE)
    find_package(PSLP 0.0.8 CONFIG QUIET)
    if(NOT TARGET PSLP::PSLP)
        include(FetchContent)
        set(PSLP_BUILD_TESTING OFF CACHE BOOL "Build PSLP tests" FORCE)
        FetchContent_Declare(
            pslp
            GIT_REPOSITORY https://github.com/dance858/PSLP.git
            GIT_TAG v0.0.8
            GIT_SHALLOW TRUE
        )
        FetchContent_MakeAvailable(pslp)
        if(TARGET PSLP AND NOT TARGET PSLP::PSLP)
            add_library(PSLP::PSLP ALIAS PSLP)
        endif()
    endif()
    if(NOT TARGET PSLP::PSLP)
        message(FATAL_ERROR "MLXPDLP_BUILD_PRESOLVE=ON but no PSLP target is available")
    endif()
endif()

if(MLX_BUILD_METAL_ENABLED)
    message(STATUS "MLX Metal backend: enabled")
else()
    message(WARNING
        "The selected MLX library was not confirmed to have Metal enabled. "
        "Configure MLX with -DMLX_BUILD_METAL=ON for GPU comparisons.")
endif()

function(mlxpdlp_enable_warnings target)
    if(NOT MLXPDLP_ENABLE_WARNINGS)
        return()
    endif()
    if(CMAKE_CXX_COMPILER_ID MATCHES "AppleClang|Clang|GNU" OR
       CMAKE_C_COMPILER_ID MATCHES "AppleClang|Clang|GNU")
        target_compile_options("${target}" PRIVATE -Wall -Wextra -Wpedantic)
    endif()
endfunction()

# Solver library
add_library(mlxpdlp_solver STATIC src/solver.cpp src/host_double_polish.cpp)
add_library(mlxPDLP::solver ALIAS mlxpdlp_solver)
set_target_properties(mlxpdlp_solver PROPERTIES
    EXPORT_NAME solver
    POSITION_INDEPENDENT_CODE ON
)
target_compile_features(mlxpdlp_solver PUBLIC cxx_std_20)
target_include_directories(mlxpdlp_solver PUBLIC
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
    "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>"
)
target_link_libraries(mlxpdlp_solver PUBLIC MLX::MLX)
if(APPLE)
    find_library(MLXPDLP_ACCELERATE_FRAMEWORK Accelerate REQUIRED)
    target_compile_definitions(mlxpdlp_solver PRIVATE MLXPDLP_HAS_ACCELERATE_SPARSE=1)
    target_link_libraries(mlxpdlp_solver PRIVATE "${MLXPDLP_ACCELERATE_FRAMEWORK}")
endif()
if(MLXPDLP_BUILD_PRESOLVE)
    target_sources(mlxpdlp_solver PRIVATE src/presolve_adapter.cpp)
    target_compile_definitions(mlxpdlp_solver PRIVATE MLXPDLP_HAS_PRESOLVE=1)
    target_link_libraries(mlxpdlp_solver PRIVATE PSLP::PSLP)
endif()
mlxpdlp_enable_warnings(mlxpdlp_solver)

set(MLXPDLP_INSTALL_TARGETS mlxpdlp_solver)

# Optional standalone MPS loader
if(MLXPDLP_BUILD_MPS)
    find_package(ZLIB REQUIRED)
    add_library(mlxpdlp_mps STATIC
        src/mps_loader.c
        src/mps_parser.c
    )
    add_library(mlxPDLP::mps ALIAS mlxpdlp_mps)
    set_target_properties(mlxpdlp_mps PROPERTIES
        EXPORT_NAME mps
        POSITION_INDEPENDENT_CODE ON
    )
    target_include_directories(mlxpdlp_mps PUBLIC
        "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
        "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>"
    )
    target_include_directories(mlxpdlp_mps PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/src"
    )
    target_link_libraries(mlxpdlp_mps PRIVATE ZLIB::ZLIB)
    mlxpdlp_enable_warnings(mlxpdlp_mps)
    list(APPEND MLXPDLP_INSTALL_TARGETS mlxpdlp_mps)
endif()

# Example
if(MLXPDLP_BUILD_EXAMPLES)
    add_executable(mlxpdlp_example examples/basic.cpp)
    target_link_libraries(mlxpdlp_example PRIVATE mlxPDLP::solver)
    mlxpdlp_enable_warnings(mlxpdlp_example)
endif()

# Original-model float64 validation shared by the LPfeas runner and its tests.
if(MLXPDLP_BUILD_MPS AND (MLXPDLP_BUILD_BENCHMARKS OR BUILD_TESTING))
    add_library(mlxpdlp_lpfeas_support STATIC
        benchmarks/lpfeas_support.cpp
    )
    target_include_directories(mlxpdlp_lpfeas_support PUBLIC
        "${CMAKE_CURRENT_SOURCE_DIR}/benchmarks"
    )
    target_link_libraries(mlxpdlp_lpfeas_support PUBLIC
        mlxPDLP::solver
        mlxPDLP::mps
    )
    mlxpdlp_enable_warnings(mlxpdlp_lpfeas_support)
endif()

# Benchmarks
if(MLXPDLP_BUILD_BENCHMARKS)
    if(NOT MLXPDLP_BUILD_MPS)
        message(FATAL_ERROR "MLXPDLP_BUILD_BENCHMARKS requires MLXPDLP_BUILD_MPS=ON")
    endif()
    add_executable(mlxpdlp_mps_benchmark benchmarks/mps_device_benchmark.cpp)
    target_link_libraries(mlxpdlp_mps_benchmark PRIVATE
        mlxPDLP::solver
        mlxPDLP::mps
    )
    mlxpdlp_enable_warnings(mlxpdlp_mps_benchmark)

    add_executable(mlxpdlp_lpfeas_benchmark benchmarks/lpfeas_benchmark.cpp)
    target_link_libraries(mlxpdlp_lpfeas_benchmark PRIVATE mlxpdlp_lpfeas_support)
    if(MLXPDLP_BUILD_PRESOLVE)
        target_compile_definitions(mlxpdlp_lpfeas_benchmark PRIVATE
            MLXPDLP_LPFEAS_HAS_PRESOLVE=1
        )
    endif()
    mlxpdlp_enable_warnings(mlxpdlp_lpfeas_benchmark)

    add_executable(mlxpdlp_mps_inspect benchmarks/mps_inspect.cpp)
    target_link_libraries(mlxpdlp_mps_inspect PRIVATE mlxPDLP::mps)
    mlxpdlp_enable_warnings(mlxpdlp_mps_inspect)
endif()

# Tests
if(BUILD_TESTING)
    add_executable(mlxpdlp_mlx_basic_test tests/test_mlx_basic.cpp)
    target_link_libraries(mlxpdlp_mlx_basic_test PRIVATE MLX::MLX)
    mlxpdlp_enable_warnings(mlxpdlp_mlx_basic_test)

    add_executable(mlxpdlp_solver_test tests/test_solver.cpp)
    target_link_libraries(mlxpdlp_solver_test PRIVATE mlxPDLP::solver)
    if(MLXPDLP_BUILD_PRESOLVE)
        target_compile_definitions(mlxpdlp_solver_test PRIVATE MLXPDLP_TEST_HAS_PRESOLVE=1)
    endif()
    mlxpdlp_enable_warnings(mlxpdlp_solver_test)

    add_executable(mlxpdlp_device_comparison
        tests/test_device_comparison.cpp
    )
    target_link_libraries(mlxpdlp_device_comparison PRIVATE mlxPDLP::solver)
    mlxpdlp_enable_warnings(mlxpdlp_device_comparison)

    add_test(NAME mlx_basic COMMAND mlxpdlp_mlx_basic_test)
    add_test(NAME solver COMMAND mlxpdlp_solver_test)
    add_test(NAME device_comparison COMMAND mlxpdlp_device_comparison)
    set_tests_properties(device_comparison PROPERTIES SKIP_RETURN_CODE 77)

    if(MLXPDLP_BUILD_MPS)
        add_executable(mlxpdlp_mps_parser_test
            tests/test_mps_parser.cpp
        )
        target_compile_definitions(mlxpdlp_mps_parser_test PRIVATE
            OMITTED_RHS_NAME_MPS="${CMAKE_CURRENT_SOURCE_DIR}/tests/data/mps/omitted_rhs_name.mps"
            FIXED_NAMES_WITH_SPACES_MPS="${CMAKE_CURRENT_SOURCE_DIR}/tests/data/mps/fixed_names_with_spaces.mps"
            FREE_FORMAT_LONG_NAMES_MPS="${CMAKE_CURRENT_SOURCE_DIR}/tests/data/mps/free_format_long_names.mps"
            FREE_ALIGNED_NUMERIC_MPS="${CMAKE_CURRENT_SOURCE_DIR}/tests/data/mps/free_aligned_numeric.mps"
        )
        target_link_libraries(mlxpdlp_mps_parser_test PRIVATE mlxPDLP::mps)
        mlxpdlp_enable_warnings(mlxpdlp_mps_parser_test)
        add_test(NAME mps_parser COMMAND mlxpdlp_mps_parser_test)

        add_executable(mlxpdlp_lpfeas_validation_test
            tests/test_lpfeas_validation.cpp
        )
        target_link_libraries(mlxpdlp_lpfeas_validation_test PRIVATE
            mlxpdlp_lpfeas_support
        )
        mlxpdlp_enable_warnings(mlxpdlp_lpfeas_validation_test)
        add_test(NAME lpfeas_validation COMMAND mlxpdlp_lpfeas_validation_test)

        if(MLXPDLP_BUILD_PRESOLVE)
            add_executable(mlxpdlp_presolve_postsolve_test
                tests/test_presolve_postsolve.cpp
            )
            target_include_directories(mlxpdlp_presolve_postsolve_test PRIVATE
                "${CMAKE_CURRENT_SOURCE_DIR}/src"
            )
            target_compile_definitions(mlxpdlp_presolve_postsolve_test PRIVATE
                IRISH_E_MPS="${CMAKE_CURRENT_SOURCE_DIR}/benchmarks/data/lpfeas/irish-e.mps.gz"
            )
            target_link_libraries(mlxpdlp_presolve_postsolve_test PRIVATE
                mlxPDLP::solver
                mlxPDLP::mps
            )
            mlxpdlp_enable_warnings(mlxpdlp_presolve_postsolve_test)
            add_test(NAME presolve_postsolve COMMAND mlxpdlp_presolve_postsolve_test)
        endif()

        add_executable(mlxpdlp_mps_device_comparison
            tests/test_mps_device_comparison.cpp
        )
        target_compile_definitions(mlxpdlp_mps_device_comparison PRIVATE
            NETLIB_ADLITTLE_MPS="${CMAKE_CURRENT_SOURCE_DIR}/tests/data/netlib/adlittle.mps"
        )
        target_link_libraries(mlxpdlp_mps_device_comparison PRIVATE
            mlxPDLP::solver
            mlxPDLP::mps
        )
        mlxpdlp_enable_warnings(mlxpdlp_mps_device_comparison)
        add_test(NAME mps_device_comparison
                 COMMAND mlxpdlp_mps_device_comparison)
        set_tests_properties(mps_device_comparison
            PROPERTIES SKIP_RETURN_CODE 77
        )
    endif()
endif()

# Downloaded full-corpus regression. The benchmark owns its worker pool so
# CTest runs the CPU and Metal sweeps serially instead of oversubscribing the
# host or unified memory with two independent pools.
if(MLXPDLP_ENABLE_NETLIB_REGRESSION)
    set(MLXPDLP_NETLIB_DATA_DIR "${CMAKE_CURRENT_SOURCE_DIR}/benchmarks/data/netlib")
    if(NOT EXISTS "${MLXPDLP_NETLIB_DATA_DIR}/manifest.tsv" OR
       NOT EXISTS "${MLXPDLP_NETLIB_DATA_DIR}/pilot87.mps.gz")
        message(FATAL_ERROR
            "The Netlib corpus is incomplete. Run "
            "benchmarks/data/netlib/download.sh before enabling its regression suite.")
    endif()

    set(MLXPDLP_NETLIB_REGRESSION_ARGS
        --data "${MLXPDLP_NETLIB_DATA_DIR}"
        --jobs auto
        --tolerance 1e-4
        --solver-tolerance 5e-5
        --time-limit 300
        --iteration-limit 500000
        --correction-time-limit 300
        --correction-iteration-limit 200000
        --host-double-polishing
        --host-double-time-limit 30
        --host-double-iteration-limit 50000
        --curtis-reid-iterations 20
        --fail-on-validation
    )

    add_test(NAME netlib_regression_cpu
        COMMAND mlxpdlp_lpfeas_benchmark
            ${MLXPDLP_NETLIB_REGRESSION_ARGS}
            --device cpu
            --output-prefix "${CMAKE_CURRENT_BINARY_DIR}/netlib-regression-cpu"
    )
    set_tests_properties(netlib_regression_cpu PROPERTIES
        LABELS "regression;netlib;cpu;long"
        RUN_SERIAL TRUE
        TIMEOUT 86400
    )

    if(MLX_BUILD_METAL_ENABLED)
        add_test(NAME netlib_regression_metal
            COMMAND mlxpdlp_lpfeas_benchmark
                ${MLXPDLP_NETLIB_REGRESSION_ARGS}
                --device metal
                --output-prefix "${CMAKE_CURRENT_BINARY_DIR}/netlib-regression-metal"
        )
        set_tests_properties(netlib_regression_metal PROPERTIES
            LABELS "regression;netlib;metal;long"
            RUN_SERIAL TRUE
            TIMEOUT 86400
        )
    else()
        message(STATUS
            "Netlib Metal regression omitted because the selected MLX build has no Metal backend")
    endif()

    add_custom_target(mlxpdlp_netlib_regression
        COMMAND "${CMAKE_CTEST_COMMAND}" --output-on-failure
            --tests-regex "^netlib_regression_"
        DEPENDS mlxpdlp_lpfeas_benchmark
        USES_TERMINAL
        COMMENT "Running the 40-case Netlib regression suite"
    )
endif()

# Installable CMake package
install(TARGETS ${MLXPDLP_INSTALL_TARGETS}
    EXPORT mlxPDLPTargets
    ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
    LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
    RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}"
)
install(FILES
    include/mlxPDLP/solver.h
    include/mlxPDLP/version.h
    DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/mlxPDLP"
)
if(MLXPDLP_BUILD_MPS)
    install(FILES include/mlxPDLP/mps_loader.h
        DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/mlxPDLP"
    )
endif()

set(MLXPDLP_INSTALL_CMAKEDIR
    "${CMAKE_INSTALL_LIBDIR}/cmake/mlxPDLP"
    CACHE STRING "CMake package installation directory"
)

configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/mlxPDLPConfig.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/mlxPDLPConfig.cmake"
    INSTALL_DESTINATION "${MLXPDLP_INSTALL_CMAKEDIR}"
)
write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/mlxPDLPConfigVersion.cmake"
    VERSION "${PROJECT_VERSION}"
    COMPATIBILITY SameMajorVersion
)
install(EXPORT mlxPDLPTargets
    FILE mlxPDLPTargets.cmake
    NAMESPACE mlxPDLP::
    DESTINATION "${MLXPDLP_INSTALL_CMAKEDIR}"
)
install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/mlxPDLPConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/mlxPDLPConfigVersion.cmake"
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/FindMLX.cmake"
    DESTINATION "${MLXPDLP_INSTALL_CMAKEDIR}"
)
install(FILES README.md LICENSE CHANGELOG.md
    DESTINATION "${CMAKE_INSTALL_DATADIR}/doc/mlxPDLP"
)
install(FILES docs/architecture.md
    DESTINATION "${CMAKE_INSTALL_DATADIR}/doc/mlxPDLP/docs"
)
