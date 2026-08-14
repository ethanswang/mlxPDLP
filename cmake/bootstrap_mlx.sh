#!/bin/sh

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

set -eu

fail() {
    printf '\nmlxPDLP: %s\n' "$*" >&2
    exit 2
}

cmake_command=${CMAKE:-cmake}
source_dir=${MLXPDLP_SOURCE_DIR:-}
build_dir=${MLXPDLP_BUILD_DIR:-}
deps_dir=${MLXPDLP_DEPS_DIR:-}
fetch_mode=${MLXPDLP_FETCH_MLX:-ask}
mlx_repository=${MLXPDLP_MLX_REPOSITORY:-https://github.com/ml-explore/mlx.git}
mlx_revision=${MLXPDLP_MLX_REVISION:-25616a0a6acf78a6e23379a0ffcdc3296775a468}

[ -n "$source_dir" ] || fail "MLXPDLP_SOURCE_DIR is not set"
[ -n "$build_dir" ] || build_dir="$source_dir/build"
[ -n "$deps_dir" ] || deps_dir="$source_dir/_deps"

case "$build_dir" in
    /*) ;;
    *) build_dir="$source_dir/$build_dir" ;;
esac
case "$deps_dir" in
    /*) ;;
    *) deps_dir="$source_dir/$deps_dir" ;;
esac

command -v "$cmake_command" >/dev/null 2>&1 ||
    fail "CMake was not found (CMAKE=$cmake_command)"

mlx_root=${MLX_ROOT:-}
mlx_source_dir=${MLX_SOURCE_DIR:-}
mlx_build_dir=${MLX_BUILD_DIR:-}
explicit_hints=no
if [ -n "$mlx_root$mlx_source_dir$mlx_build_dir" ]; then
    explicit_hints=yes
fi

mlx_available() {
    "$cmake_command" \
        "-DMLX_ROOT:PATH=$1" \
        "-DMLX_SOURCE_DIR:PATH=$2" \
        "-DMLX_BUILD_DIR:PATH=$3" \
        -P "$source_dir/cmake/CheckMLX.cmake" >/dev/null 2>&1
}

cache_value() {
    sed -n "s/^$1:[^=]*=//p" "$build_dir/CMakeCache.txt"
}

have_mlx=no
reuse_cache=no
if [ "$explicit_hints" = no ] && [ -f "$build_dir/CMakeCache.txt" ]; then
    cached_library=$(cache_value MLX_LIBRARY)
    cached_include=$(cache_value MLX_INCLUDE_DIR)
    cached_generated_include=$(cache_value MLX_GENERATED_INCLUDE_DIR)
    if [ -f "$cached_library" ] &&
       [ -f "$cached_include/mlx/mlx.h" ] &&
       [ -f "$cached_generated_include/mlx/version.h" ]; then
        have_mlx=yes
        reuse_cache=yes
    fi
fi

if [ "$have_mlx" = no ] &&
   mlx_available "$mlx_root" "$mlx_source_dir" "$mlx_build_dir"; then
    have_mlx=yes
fi

managed_source_dir="$deps_dir/mlx"
managed_build_dir="$deps_dir/mlx-build"
managed_root="$deps_dir/mlx-install"
managed_metadata="$managed_root/.mlxpdlp-managed-mlx"
managed_selected=no

if [ "$have_mlx" = no ] && [ -f "$managed_metadata" ]; then
    managed_repository=$(sed -n '1p' "$managed_metadata")
    managed_revision=$(sed -n '2p' "$managed_metadata")
    if [ "$managed_repository" = "$mlx_repository" ] &&
       [ "$managed_revision" = "$mlx_revision" ] &&
       mlx_available "$managed_root" "" ""; then
        mlx_root=$managed_root
        mlx_source_dir=$managed_source_dir
        mlx_build_dir=$managed_build_dir
        managed_selected=yes
        have_mlx=yes
    fi
fi

if [ "$have_mlx" = no ]; then
    case "$fetch_mode" in
        1|ON|on|YES|yes|TRUE|true)
            approved=yes
            ;;
        0|OFF|off|NO|no|FALSE|false)
            approved=no
            ;;
        ask|ASK|Ask|"")
            printf '%s\n' "mlxPDLP could not find a usable MLX C++ library."
            printf 'Download MLX revision %.12s from %s, build it, and install it under %s? [y/N] ' \
                "$mlx_revision" "$mlx_repository" "$deps_dir"
            if IFS= read -r answer; then
                case "$answer" in
                    y|Y|yes|YES|Yes) approved=yes ;;
                    *) approved=no ;;
                esac
            else
                approved=no
            fi
            ;;
        *)
            fail "MLXPDLP_FETCH_MLX must be ON, OFF, or ask (got '$fetch_mode')"
            ;;
    esac

    if [ "$approved" != yes ]; then
        if [ "$explicit_hints" = yes ]; then
            printf '%s\n' "The supplied MLX path hints did not identify a usable library." >&2
        fi
        fail "MLX is required. Provide MLX_ROOT/MLX_SOURCE_DIR/MLX_BUILD_DIR, or approve the managed dependency with 'make MLXPDLP_FETCH_MLX=ON'."
    fi

    command -v git >/dev/null 2>&1 ||
        fail "Git is required to download MLX"
    mkdir -p "$deps_dir"

    if [ ! -e "$managed_source_dir" ]; then
        printf 'mlxPDLP: downloading MLX revision %s from %s\n' \
            "$mlx_revision" "$mlx_repository"
        git clone --filter=blob:none --no-checkout \
            "$mlx_repository" "$managed_source_dir"
        git -C "$managed_source_dir" checkout --detach "$mlx_revision"
    elif [ ! -f "$managed_source_dir/CMakeLists.txt" ] ||
         [ ! -f "$managed_source_dir/mlx/mlx.h" ]; then
        fail "$managed_source_dir exists but is not a usable MLX source tree; move it aside and retry"
    else
        managed_head=$(git -C "$managed_source_dir" rev-parse HEAD 2>/dev/null) ||
            fail "$managed_source_dir is not the requested Git checkout; move it aside and retry"
        managed_requested=$(git -C "$managed_source_dir" \
            rev-parse "$mlx_revision^{commit}" 2>/dev/null) ||
            fail "$managed_source_dir does not contain requested revision $mlx_revision; move it aside and retry"
        if [ "$managed_head" != "$managed_requested" ]; then
            fail "$managed_source_dir is at revision $managed_head, not requested revision $managed_requested; move it aside and retry"
        fi
        printf 'mlxPDLP: reusing MLX source in %s\n' "$managed_source_dir"
    fi

    printf 'mlxPDLP: configuring MLX in %s\n' "$managed_build_dir"
    set -- "$cmake_command" -S "$managed_source_dir" -B "$managed_build_dir" \
        -DCMAKE_BUILD_TYPE=Release \
        -DBUILD_SHARED_LIBS=OFF \
        -DMLX_BUILD_METAL=ON \
        -DMLX_BUILD_TESTS=OFF \
        -DMLX_BUILD_EXAMPLES=OFF \
        -DMLX_BUILD_BENCHMARKS=OFF \
        -DMLX_BUILD_PYTHON_BINDINGS=OFF \
        -DMLX_BUILD_GGUF=OFF \
        -DMLX_BUILD_SAFETENSORS=OFF
    if [ -n "${MLXPDLP_MLX_CMAKE_ARGS:-}" ]; then
        set -f
        # shellcheck disable=SC2086
        set -- "$@" $MLXPDLP_MLX_CMAKE_ARGS
        set +f
    fi
    "$@"

    printf 'mlxPDLP: building MLX\n'
    "$cmake_command" --build "$managed_build_dir" --parallel

    printf 'mlxPDLP: installing MLX in %s\n' "$managed_root"
    "$cmake_command" --install "$managed_build_dir" \
        --prefix "$managed_root" --config Release

    if ! mlx_available "$managed_root" "" ""; then
        fail "the managed MLX build completed but could not be discovered"
    fi

    {
        printf '%s\n' "$mlx_repository"
        printf '%s\n' "$mlx_revision"
    } > "$managed_metadata"

    mlx_root=$managed_root
    mlx_source_dir=$managed_source_dir
    mlx_build_dir=$managed_build_dir
    managed_selected=yes
fi

if [ "$reuse_cache" = yes ]; then
    printf 'mlxPDLP: reusing MLX recorded in %s\n' "$build_dir"
elif [ "$managed_selected" = yes ]; then
    printf 'mlxPDLP: using the managed MLX dependency in %s\n' "$managed_root"
elif [ -n "$mlx_root$mlx_source_dir$mlx_build_dir" ]; then
    printf 'mlxPDLP: using an existing MLX dependency\n'
else
    printf 'mlxPDLP: using a discoverable MLX installation\n'
fi

set -- "$cmake_command" -S "$source_dir" -B "$build_dir"
[ -z "$mlx_root" ] || set -- "$@" "-DMLX_ROOT:PATH=$mlx_root"
[ -z "$mlx_source_dir" ] || set -- "$@" "-DMLX_SOURCE_DIR:PATH=$mlx_source_dir"
[ -z "$mlx_build_dir" ] || set -- "$@" "-DMLX_BUILD_DIR:PATH=$mlx_build_dir"

# CMAKE_ARGS is intentionally shell-word-split so callers can pass multiple
# ordinary CMake options, for example CMAKE_ARGS='-DCMAKE_BUILD_TYPE=Release
# -DBUILD_TESTING=OFF'. Paths with spaces should use the dedicated variables
# above or a CMake preset instead.
if [ -n "${CMAKE_ARGS:-}" ]; then
    set -f
    # shellcheck disable=SC2086
    set -- "$@" $CMAKE_ARGS
    set +f
fi

printf 'mlxPDLP: configuring the project in %s\n' "$build_dir"
"$@"
