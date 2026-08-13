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

# Locate an MLX C++ source/build pair or installation.
#
# Input hints:
#   MLX_ROOT        Common installation or source prefix.
#   MLX_SOURCE_DIR  Directory containing mlx/mlx.h.
#   MLX_BUILD_DIR   Directory containing libmlx and generated mlx/version.h.
#
# Result:
#   MLX::MLX
#   MLX_FOUND
#   MLX_BUILD_METAL_ENABLED

include(FindPackageHandleStandardArgs)

set(MLX_ROOT "" CACHE PATH "MLX installation or source prefix")
set(MLX_SOURCE_DIR "" CACHE PATH "MLX source directory")
set(MLX_BUILD_DIR "" CACHE PATH "MLX build directory")

set(_MLX_SOURCE_HINTS)
set(_MLX_BUILD_HINTS)
if(MLX_ROOT)
    list(APPEND _MLX_SOURCE_HINTS
        "${MLX_ROOT}"
        "${MLX_ROOT}/include"
    )
    list(APPEND _MLX_BUILD_HINTS
        "${MLX_ROOT}"
        "${MLX_ROOT}/lib"
        "${MLX_ROOT}/lib64"
    )
endif()
if(MLX_SOURCE_DIR)
    list(APPEND _MLX_SOURCE_HINTS "${MLX_SOURCE_DIR}")
endif()
if(MLX_BUILD_DIR)
    list(APPEND _MLX_SOURCE_HINTS "${MLX_BUILD_DIR}/..")
    list(APPEND _MLX_BUILD_HINTS "${MLX_BUILD_DIR}")
endif()

find_path(MLX_INCLUDE_DIR
    NAMES mlx/mlx.h
    HINTS ${_MLX_SOURCE_HINTS}
)
find_path(MLX_GENERATED_INCLUDE_DIR
    NAMES mlx/version.h
    HINTS
        ${_MLX_SOURCE_HINTS}
        ${_MLX_BUILD_HINTS}
        "${MLX_ROOT}/include"
)
find_library(MLX_LIBRARY
    NAMES mlx
    HINTS ${_MLX_BUILD_HINTS}
)
find_path(MLX_JSON_INCLUDE_DIR
    NAMES nlohmann/json.hpp
    HINTS
        "${MLX_BUILD_DIR}/_deps/json-src/include"
        "${MLX_ROOT}/include"
)
find_path(MLX_FMT_INCLUDE_DIR
    NAMES fmt/format.h
    HINTS
        "${MLX_BUILD_DIR}/_deps/fmt-src/include"
        "${MLX_ROOT}/include"
)

set(_MLX_REQUIRED_VARS
    MLX_LIBRARY
    MLX_INCLUDE_DIR
    MLX_GENERATED_INCLUDE_DIR
    MLX_JSON_INCLUDE_DIR
    MLX_FMT_INCLUDE_DIR
)
set(_MLX_PLATFORM_LIBRARIES)
if(APPLE)
    find_library(MLX_FOUNDATION_FRAMEWORK Foundation)
    find_library(MLX_METAL_FRAMEWORK Metal)
    find_library(MLX_METALKIT_FRAMEWORK MetalKit)
    find_library(MLX_ACCELERATE_FRAMEWORK Accelerate)
    list(APPEND _MLX_REQUIRED_VARS
        MLX_FOUNDATION_FRAMEWORK
        MLX_METAL_FRAMEWORK
        MLX_METALKIT_FRAMEWORK
        MLX_ACCELERATE_FRAMEWORK
    )
    list(APPEND _MLX_PLATFORM_LIBRARIES
        "${MLX_FOUNDATION_FRAMEWORK}"
        "${MLX_METAL_FRAMEWORK}"
        "${MLX_METALKIT_FRAMEWORK}"
        "${MLX_ACCELERATE_FRAMEWORK}"
    )
endif()

find_package_handle_standard_args(MLX
    REQUIRED_VARS ${_MLX_REQUIRED_VARS}
)

set(MLX_BUILD_METAL_ENABLED FALSE)
set(_MLX_BUILD_METADATA_DIRS)
if(MLX_BUILD_DIR)
    list(APPEND _MLX_BUILD_METADATA_DIRS "${MLX_BUILD_DIR}")
endif()
# A cached MLX_BUILD_DIR can become stale while find_library still resolves a
# valid library from another prefix. Inspect the selected library's own build
# directory as the authoritative fallback.
if(MLX_LIBRARY)
    get_filename_component(_MLX_SELECTED_LIBRARY_DIR "${MLX_LIBRARY}" DIRECTORY)
    list(APPEND _MLX_BUILD_METADATA_DIRS "${_MLX_SELECTED_LIBRARY_DIR}")
endif()
list(REMOVE_DUPLICATES _MLX_BUILD_METADATA_DIRS)
foreach(_MLX_BUILD_METADATA_DIR IN LISTS _MLX_BUILD_METADATA_DIRS)
    if(EXISTS "${_MLX_BUILD_METADATA_DIR}/CMakeCache.txt")
        file(STRINGS "${_MLX_BUILD_METADATA_DIR}/CMakeCache.txt"
             _MLX_METAL_CACHE_ENTRY REGEX "^MLX_BUILD_METAL:BOOL=")
        if(_MLX_METAL_CACHE_ENTRY STREQUAL "MLX_BUILD_METAL:BOOL=ON")
            set(MLX_BUILD_METAL_ENABLED TRUE)
            break()
        endif()
    endif()
endforeach()

if(MLX_FOUND AND NOT TARGET MLX::MLX)
    add_library(MLX::MLX UNKNOWN IMPORTED)
    set_target_properties(MLX::MLX PROPERTIES
        IMPORTED_LOCATION "${MLX_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES
            "${MLX_INCLUDE_DIR};${MLX_GENERATED_INCLUDE_DIR};${MLX_JSON_INCLUDE_DIR};${MLX_FMT_INCLUDE_DIR}"
        INTERFACE_LINK_LIBRARIES "${_MLX_PLATFORM_LIBRARIES}"
    )
endif()

mark_as_advanced(
    MLX_LIBRARY
    MLX_INCLUDE_DIR
    MLX_GENERATED_INCLUDE_DIR
    MLX_JSON_INCLUDE_DIR
    MLX_FMT_INCLUDE_DIR
    MLX_FOUNDATION_FRAMEWORK
    MLX_METAL_FRAMEWORK
    MLX_METALKIT_FRAMEWORK
    MLX_ACCELERATE_FRAMEWORK
)
