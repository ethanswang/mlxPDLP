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
mlx_build_parallel_level=${CMAKE_BUILD_PARALLEL_LEVEL:-${MLXPDLP_MAKE_PARALLEL_LEVEL:-3}}
source_dir=${MLXPDLP_SOURCE_DIR:-}
build_dir=${MLXPDLP_BUILD_DIR:-}
deps_dir=${MLXPDLP_DEPS_DIR:-}
fetch_deps_mode=${MLXPDLP_FETCH_DEPS:-}
legacy_fetch_mode=${MLXPDLP_FETCH_MLX:-}
mlx_repository=${MLXPDLP_MLX_REPOSITORY:-https://github.com/ml-explore/mlx.git}
mlx_revision=${MLXPDLP_MLX_REVISION:-25616a0a6acf78a6e23379a0ffcdc3296775a468}
pslp_repository=https://github.com/dance858/PSLP.git
pslp_revision=v0.0.11
macos_deployment_target=
managed_macos_deployment_target=
if [ "$(uname -s)" = Darwin ]; then
    macos_deployment_target=${MLXPDLP_MACOS_DEPLOYMENT_TARGET:-${MACOSX_DEPLOYMENT_TARGET:-}}
    managed_macos_deployment_target=${macos_deployment_target:-14.0}
fi

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

normalize_fetch_mode() {
    case "$1" in
        1|ON|on|YES|yes|TRUE|true) printf '%s\n' yes ;;
        0|OFF|off|NO|no|FALSE|false) printf '%s\n' no ;;
        ask|ASK|Ask|"") printf '%s\n' ask ;;
        *) fail "$2 must be ON, OFF, or ask (got '$1')" ;;
    esac
}

if [ -n "$fetch_deps_mode" ]; then
    fetch_mode=$(normalize_fetch_mode "$fetch_deps_mode" MLXPDLP_FETCH_DEPS)
else
    fetch_mode=ask
fi
if [ -n "$legacy_fetch_mode" ]; then
    normalized_legacy_fetch_mode=$(
        normalize_fetch_mode "$legacy_fetch_mode" MLXPDLP_FETCH_MLX
    )
    if [ -n "$fetch_deps_mode" ] &&
       [ "$normalized_legacy_fetch_mode" != "$fetch_mode" ]; then
        fail "MLXPDLP_FETCH_DEPS and MLXPDLP_FETCH_MLX disagree"
    fi
    fetch_mode=$normalized_legacy_fetch_mode
fi
approval=$fetch_mode

request_download_approval() {
    case "$approval" in
        yes) return 0 ;;
        no) return 1 ;;
    esac

    printf '%s\n' "mlxPDLP needs one or more source dependencies that are not available locally."
    printf '%s\n' "Approval allows this make invocation to access the network for:"
    printf '  - MLX revision %.12s from %s (when MLX is missing)\n' \
        "$mlx_revision" "$mlx_repository"
    printf '  - PSLP %s from %s (when presolve is enabled and PSLP is missing)\n' \
        "$pslp_revision" "$pslp_repository"
    printf '%s\n' "  - MLX's pinned metal-cpp source from developer.apple.com"
    printf '%s\n' "  - MLX's pinned JSON/fmt sources from github.com"
    printf 'Managed files stay under %s and %s/_deps; no system prefix is modified.\n' \
        "$deps_dir" "$build_dir"
    printf '%s\n' "Allow roughly 1 GB of free disk space; the first build may take several minutes."
    printf '%s' "Download and build the missing dependencies? [y/N] "
    if IFS= read -r answer; then
        case "$answer" in
            y|Y|yes|YES|Yes) approval=yes; return 0 ;;
        esac
    fi
    approval=no
    return 1
}

command -v "$cmake_command" >/dev/null 2>&1 ||
    fail "CMake was not found (CMAKE=$cmake_command)"

case "$mlx_build_parallel_level" in
    0|*[!0-9]*)
        fail "CMAKE_BUILD_PARALLEL_LEVEL must be a positive integer"
        ;;
esac

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

managed_source_dir="$deps_dir/mlx"
managed_build_dir="$deps_dir/mlx-build"
managed_root="$deps_dir/mlx-install"
managed_metadata="$managed_root/.mlxpdlp-managed-mlx"
managed_selected=no

managed_metadata_matches() {
    [ -f "$managed_metadata" ] &&
        [ "$(sed -n '1p' "$managed_metadata")" = "$mlx_repository" ] &&
        [ "$(sed -n '2p' "$managed_metadata")" = "$mlx_revision" ] &&
        [ "$(sed -n '3p' "$managed_metadata")" = "$managed_macos_deployment_target" ]
}

have_mlx=no
reuse_cache=no
if [ "$explicit_hints" = no ] && [ -f "$build_dir/CMakeCache.txt" ]; then
    cached_library=$(cache_value MLX_LIBRARY)
    cached_include=$(cache_value MLX_INCLUDE_DIR)
    cached_generated_include=$(cache_value MLX_GENERATED_INCLUDE_DIR)
    cached_managed_metadata_valid=yes
    case "$cached_library" in
        "$managed_root"/*)
            if ! managed_metadata_matches; then
                cached_managed_metadata_valid=no
            fi
            ;;
    esac
    if [ "$cached_managed_metadata_valid" = yes ] &&
       [ -f "$cached_library" ] &&
       [ -f "$cached_include/mlx/mlx.h" ] &&
       [ -f "$cached_generated_include/mlx/version.h" ]; then
        have_mlx=yes
        reuse_cache=yes
        case "$cached_library" in
            "$managed_root"/*) managed_selected=yes ;;
        esac
    fi
fi

if [ "$have_mlx" = no ] &&
   mlx_available "$mlx_root" "$mlx_source_dir" "$mlx_build_dir"; then
    have_mlx=yes
fi

if [ "$have_mlx" = no ] && managed_metadata_matches; then
    if mlx_available "$managed_root" "" ""; then
        mlx_root=$managed_root
        mlx_source_dir=$managed_source_dir
        mlx_build_dir=$managed_build_dir
        managed_selected=yes
        have_mlx=yes
    fi
fi

if [ "$have_mlx" = no ]; then
    if ! request_download_approval; then
        if [ "$explicit_hints" = yes ]; then
            printf '%s\n' "The supplied MLX path hints did not identify a usable library." >&2
        fi
        fail "MLX is required. Provide MLX_ROOT/MLX_SOURCE_DIR/MLX_BUILD_DIR," \
            "or approve managed dependencies with 'make MLXPDLP_FETCH_DEPS=ON'."
    fi

    command -v git >/dev/null 2>&1 ||
        fail "Git is required to download MLX"
    mkdir -p "$deps_dir"

    move_incomplete_checkout_aside() {
        incomplete_path=$1
        backup_path="$incomplete_path.incomplete"
        backup_index=1
        while [ -e "$backup_path" ]; do
            backup_path="$incomplete_path.incomplete.$backup_index"
            backup_index=$((backup_index + 1))
        done
        mv "$incomplete_path" "$backup_path"
        printf 'mlxPDLP: preserved an incomplete managed checkout as %s\n' \
            "$backup_path"
    }

    managed_source_usable=no
    if [ -f "$managed_source_dir/CMakeLists.txt" ] &&
       [ -f "$managed_source_dir/mlx/mlx.h" ]; then
        managed_head=$(git -C "$managed_source_dir" rev-parse HEAD 2>/dev/null || true)
        managed_requested=$(git -C "$managed_source_dir" \
            rev-parse "$mlx_revision^{commit}" 2>/dev/null || true)
        if [ -n "$managed_head" ] &&
           [ "$managed_head" = "$managed_requested" ]; then
            managed_source_usable=yes
        fi
    fi

    if [ -e "$managed_source_dir" ] && [ "$managed_source_usable" = no ]; then
        printf 'mlxPDLP: recovering the managed MLX source directory in %s\n' \
            "$managed_source_dir"
        move_incomplete_checkout_aside "$managed_source_dir"
    fi

    if [ "$managed_source_usable" = no ]; then
        managed_download_dir="$managed_source_dir.download"
        if [ -e "$managed_download_dir" ]; then
            printf 'mlxPDLP: recovering an interrupted MLX download in %s\n' \
                "$managed_download_dir"
            move_incomplete_checkout_aside "$managed_download_dir"
        fi
        printf 'mlxPDLP: downloading MLX revision %s from %s\n' \
            "$mlx_revision" "$mlx_repository"
        git clone --filter=blob:none --no-checkout \
            "$mlx_repository" "$managed_download_dir"
        git -C "$managed_download_dir" checkout --detach "$mlx_revision"
        if [ ! -f "$managed_download_dir/CMakeLists.txt" ] ||
           [ ! -f "$managed_download_dir/mlx/mlx.h" ]; then
            fail "the downloaded MLX checkout is incomplete; retry to recover automatically"
        fi
        mv "$managed_download_dir" "$managed_source_dir"
    else
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
    if [ -n "$managed_macos_deployment_target" ]; then
        set -- "$@" \
            "-DCMAKE_OSX_DEPLOYMENT_TARGET:STRING=$managed_macos_deployment_target"
    fi
    "$@"

    printf 'mlxPDLP: building MLX with %s parallel jobs\n' \
        "$mlx_build_parallel_level"
    MAKEFLAGS= "$cmake_command" --build "$managed_build_dir" \
        --parallel "$mlx_build_parallel_level"

    printf 'mlxPDLP: installing MLX in %s\n' "$managed_root"
    "$cmake_command" --install "$managed_build_dir" \
        --prefix "$managed_root" --config Release

    if ! mlx_available "$managed_root" "" ""; then
        fail "the managed MLX build completed but could not be discovered"
    fi

    {
        printf '%s\n' "$mlx_repository"
        printf '%s\n' "$mlx_revision"
        printf '%s\n' "$managed_macos_deployment_target"
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

configure_project() {
    allow_downloads=$1
    project_macos_deployment_target=$macos_deployment_target
    if [ "$managed_selected" = yes ]; then
        project_macos_deployment_target=$managed_macos_deployment_target
    fi
    set -- "$cmake_command" -S "$source_dir" -B "$build_dir"
    [ -z "$mlx_root" ] || set -- "$@" "-DMLX_ROOT:PATH=$mlx_root"
    [ -z "$mlx_source_dir" ] || set -- "$@" "-DMLX_SOURCE_DIR:PATH=$mlx_source_dir"
    [ -z "$mlx_build_dir" ] || set -- "$@" "-DMLX_BUILD_DIR:PATH=$mlx_build_dir"

    # CMAKE_ARGS is intentionally shell-word-split so callers can pass
    # multiple ordinary CMake options. Paths with spaces should use the
    # dedicated variables above or a CMake preset instead.
    if [ -n "${CMAKE_ARGS:-}" ]; then
        set -f
        # shellcheck disable=SC2086
        set -- "$@" $CMAKE_ARGS
        set +f
    fi

    if [ -n "$project_macos_deployment_target" ]; then
        set -- "$@" \
            "-DCMAKE_OSX_DEPLOYMENT_TARGET:STRING=$project_macos_deployment_target"
    fi

    # Append the consent decision after CMAKE_ARGS and pass it every time. A
    # cached ON value or a conflicting CMAKE_ARGS entry must not weaken an
    # explicit offline invocation.
    set -- "$@" "-DMLXPDLP_ALLOW_DOWNLOADS:BOOL=$allow_downloads"
    printf 'mlxPDLP: configuring the project in %s\n' "$build_dir"
    "$@"
}

case "$approval" in
    yes)
        configure_project ON
        ;;
    no)
        configure_project OFF
        ;;
    ask)
        # Probe without network access first. If PSLP is the only missing
        # dependency, CMake leaves a private marker and we can ask once before
        # retrying. Any unrelated configure error is reproduced verbatim.
        mkdir -p "$build_dir"
        configure_log="$build_dir/.mlxpdlp-configure-probe.log"
        missing_pslp_marker="$build_dir/.mlxpdlp-missing-pslp"
        rm -f "$configure_log" "$missing_pslp_marker"
        if configure_project OFF >"$configure_log" 2>&1; then
            cat "$configure_log"
            rm -f "$configure_log"
        else
            configure_status=$?
            if [ -f "$missing_pslp_marker" ]; then
                rm -f "$configure_log" "$missing_pslp_marker"
                if ! request_download_approval; then
                    fail "PSLP is required for presolve. Provide" \
                        "PSLP_DIR/FETCHCONTENT_SOURCE_DIR_PSLP, disable presolve," \
                        "or approve managed dependencies with" \
                        "'make MLXPDLP_FETCH_DEPS=ON'."
                fi
                configure_project ON
            else
                cat "$configure_log" >&2
                rm -f "$configure_log"
                exit "$configure_status"
            fi
        fi
        ;;
esac
