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
project(mlxpdlp-python LANGUAGES C CXX)

# Environment hints for the mlxPDLP core build.
if(NOT DEFINED MLX_BUILD_DIR AND DEFINED ENV{MLX_BUILD_DIR})
    set(MLX_BUILD_DIR "$ENV{MLX_BUILD_DIR}" CACHE PATH
        "Directory containing libmlx and generated mlx/version.h")
endif()
if(NOT DEFINED MLX_SOURCE_DIR AND DEFINED ENV{MLX_SOURCE_DIR})
    set(MLX_SOURCE_DIR "$ENV{MLX_SOURCE_DIR}" CACHE PATH
        "Directory containing mlx/mlx.h")
endif()
if(NOT DEFINED FETCHCONTENT_SOURCE_DIR_PSLP AND
   DEFINED ENV{FETCHCONTENT_SOURCE_DIR_PSLP})
    set(FETCHCONTENT_SOURCE_DIR_PSLP "$ENV{FETCHCONTENT_SOURCE_DIR_PSLP}"
        CACHE PATH "Reuse an existing PSLP checkout")
endif()

# Build only the core libraries from the in-tree sources: no tests,
# examples, benchmarks, or install/export machinery.
set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
# PSLP must be statically linked into the extension module so the wheel
# stays self-contained (no @rpath dependency on a fetched dylib).
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(MLXPDLP_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(MLXPDLP_BUILD_BENCHMARKS OFF CACHE BOOL "" FORCE)
set(MLXPDLP_BUILD_MPS ON CACHE BOOL "" FORCE)
set(MLXPDLP_ENABLE_NETLIB_REGRESSION OFF CACHE BOOL "" FORCE)
add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/.." mlxpdlp-core)

# Python extension module built with nanobind.
find_package(Python 3.9 COMPONENTS Interpreter Development.Module REQUIRED)
execute_process(
    COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
    OUTPUT_VARIABLE nanobind_ROOT
    OUTPUT_STRIP_TRAILING_WHITESPACE)
find_package(nanobind CONFIG REQUIRED)

nanobind_add_module(_core src/module.cpp)
target_link_libraries(_core PRIVATE mlxPDLP::solver mlxPDLP::mps)
target_compile_features(_core PRIVATE cxx_std_20)

install(TARGETS _core LIBRARY DESTINATION .)
