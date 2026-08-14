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

source_dir=${1:?usage: test_bootstrap.sh SOURCE_DIR}
bootstrap="$source_dir/cmake/bootstrap_mlx.sh"
fake_bin="$source_dir/tests/bootstrap/fake-bin"
test_root=$(mktemp -d "${TMPDIR:-/tmp}/mlxpdlp-bootstrap.XXXXXX")
revision=0123456789abcdef0123456789abcdef01234567

cleanup() {
    rm -rf "$test_root"
}
trap cleanup EXIT HUP INT TERM

fail() {
    printf 'bootstrap test: %s\n' "$*" >&2
    exit 1
}

assert_contains() {
    grep -F -- "$2" "$1" >/dev/null 2>&1 ||
        fail "$1 does not contain: $2"
}

assert_not_contains() {
    if grep -F -- "$2" "$1" >/dev/null 2>&1; then
        fail "$1 unexpectedly contains: $2"
    fi
}

new_case() {
    case_dir="$test_root/$1"
    mkdir -p "$case_dir"
    cmake_log="$case_dir/cmake.log"
    git_log="$case_dir/git.log"
    policy_log="$case_dir/policy.log"
    : > "$cmake_log"
    : > "$git_log"
    : > "$policy_log"
}

new_case decline
printf '%s\n' n > "$case_dir/answer"
if env \
    PATH="$fake_bin:/usr/bin:/bin" \
    FAKE_CMAKE_LOG="$cmake_log" \
    FAKE_GIT_LOG="$git_log" \
    FAKE_MLX_AVAILABLE=yes \
    FAKE_MLX_REVISION="$revision" \
    FAKE_POLICY_LOG="$policy_log" \
    FAKE_PSLP_MISSING=yes \
    FAKE_SOURCE_DIR="$source_dir" \
    MLXPDLP_BUILD_DIR="$case_dir/build" \
    MLXPDLP_DEPS_DIR="$case_dir/deps" \
    MLXPDLP_FETCH_DEPS=ask \
    MLXPDLP_MLX_REVISION="$revision" \
    MLXPDLP_SOURCE_DIR="$source_dir" \
    sh "$bootstrap" < "$case_dir/answer" > "$case_dir/output" 2>&1; then
    fail "declining a required PSLP download unexpectedly succeeded"
fi
assert_contains "$case_dir/output" "Approval allows this make invocation"
assert_contains "$case_dir/output" "PSLP v0.0.8"
assert_contains "$case_dir/output" "metal-cpp source from developer.apple.com"
assert_contains "$case_dir/output" "JSON/fmt sources from github.com"
assert_contains "$case_dir/output" "PSLP is required for presolve"

new_case accept
printf '%s\n' y > "$case_dir/answer"
env \
    PATH="$fake_bin:/usr/bin:/bin" \
    FAKE_CMAKE_LOG="$cmake_log" \
    FAKE_GIT_LOG="$git_log" \
    FAKE_MLX_AVAILABLE=yes \
    FAKE_MLX_REVISION="$revision" \
    FAKE_POLICY_LOG="$policy_log" \
    FAKE_PSLP_MISSING=yes \
    FAKE_SOURCE_DIR="$source_dir" \
    MLXPDLP_BUILD_DIR="$case_dir/build" \
    MLXPDLP_DEPS_DIR="$case_dir/deps" \
    MLXPDLP_FETCH_DEPS=ask \
    MLXPDLP_MLX_REVISION="$revision" \
    MLXPDLP_SOURCE_DIR="$source_dir" \
    sh "$bootstrap" < "$case_dir/answer" > "$case_dir/output" 2>&1
[ "$(sed -n '1p' "$policy_log")" = OFF ] ||
    fail "approval probe did not begin offline"
[ "$(sed -n '2p' "$policy_log")" = ON ] ||
    fail "approved retry did not enable downloads"
[ "$(wc -l < "$policy_log" | tr -d ' ')" = 2 ] ||
    fail "approval flow configured the project an unexpected number of times"

new_case offline
if env \
    PATH="$fake_bin:/usr/bin:/bin" \
    CMAKE_ARGS=-DMLXPDLP_ALLOW_DOWNLOADS:BOOL=ON \
    FAKE_CMAKE_LOG="$cmake_log" \
    FAKE_GIT_LOG="$git_log" \
    FAKE_MLX_AVAILABLE=yes \
    FAKE_MLX_REVISION="$revision" \
    FAKE_POLICY_LOG="$policy_log" \
    FAKE_PSLP_MISSING=yes \
    FAKE_SOURCE_DIR="$source_dir" \
    MLXPDLP_BUILD_DIR="$case_dir/build" \
    MLXPDLP_DEPS_DIR="$case_dir/deps" \
    MLXPDLP_FETCH_DEPS=OFF \
    MLXPDLP_MLX_REVISION="$revision" \
    MLXPDLP_SOURCE_DIR="$source_dir" \
    sh "$bootstrap" > "$case_dir/output" 2>&1; then
    fail "offline mode unexpectedly succeeded without PSLP"
fi
[ "$(sed -n '1p' "$policy_log")" = OFF ] ||
    fail "CMAKE_ARGS or cached consent weakened offline mode"
assert_not_contains "$case_dir/output" "Download and build"
[ ! -s "$git_log" ] || fail "offline mode invoked Git"

new_case local
env \
    PATH="$fake_bin:/usr/bin:/bin" \
    FAKE_CMAKE_LOG="$cmake_log" \
    FAKE_GIT_LOG="$git_log" \
    FAKE_MLX_AVAILABLE=yes \
    FAKE_MLX_REVISION="$revision" \
    FAKE_POLICY_LOG="$policy_log" \
    FAKE_PSLP_MISSING=no \
    FAKE_SOURCE_DIR="$source_dir" \
    MLXPDLP_BUILD_DIR="$case_dir/build" \
    MLXPDLP_DEPS_DIR="$case_dir/deps" \
    MLXPDLP_FETCH_DEPS=ask \
    MLXPDLP_MLX_REVISION="$revision" \
    MLXPDLP_SOURCE_DIR="$source_dir" \
    sh "$bootstrap" > "$case_dir/output" 2>&1
[ "$(sed -n '1p' "$policy_log")" = OFF ] ||
    fail "fully local configure did not remain offline"
assert_not_contains "$case_dir/output" "Download and build"

new_case make_j
env \
    PATH="$fake_bin:/usr/bin:/bin" \
    CMAKE=cmake \
    FAKE_CMAKE_LOG="$cmake_log" \
    FAKE_GIT_LOG="$git_log" \
    FAKE_MLX_AVAILABLE=yes \
    FAKE_MLX_REVISION="$revision" \
    FAKE_POLICY_LOG="$policy_log" \
    FAKE_PSLP_MISSING=no \
    FAKE_SOURCE_DIR="$source_dir" \
    MLXPDLP_DEPS_DIR="$case_dir/deps" \
    MLXPDLP_FETCH_DEPS=OFF \
    MLXPDLP_LOCAL_CPU_COUNT=12 \
    make -j -C "$source_dir" BUILD_DIR="$case_dir/build" build \
    > "$case_dir/output" 2>&1
assert_contains "$cmake_log" "$case_dir/build --parallel 12"

new_case make_j_override
env \
    PATH="$fake_bin:/usr/bin:/bin" \
    CMAKE=cmake \
    CMAKE_BUILD_PARALLEL_LEVEL=3 \
    FAKE_CMAKE_LOG="$cmake_log" \
    FAKE_GIT_LOG="$git_log" \
    FAKE_MLX_AVAILABLE=yes \
    FAKE_MLX_REVISION="$revision" \
    FAKE_POLICY_LOG="$policy_log" \
    FAKE_PSLP_MISSING=no \
    FAKE_SOURCE_DIR="$source_dir" \
    MLXPDLP_DEPS_DIR="$case_dir/deps" \
    MLXPDLP_FETCH_DEPS=OFF \
    MLXPDLP_LOCAL_CPU_COUNT=12 \
    make -j -C "$source_dir" BUILD_DIR="$case_dir/build" build \
    > "$case_dir/output" 2>&1
assert_contains "$cmake_log" "$case_dir/build --parallel 3"
assert_not_contains "$cmake_log" "$case_dir/build --parallel 12"

new_case recovery
mkdir -p "$case_dir/deps/mlx" "$case_dir/deps/mlx.download"
: > "$case_dir/deps/mlx/partial-checkout"
: > "$case_dir/deps/mlx.download/partial-download"
env \
    PATH="$fake_bin:/usr/bin:/bin" \
    CMAKE_BUILD_PARALLEL_LEVEL=3 \
    FAKE_CMAKE_LOG="$cmake_log" \
    FAKE_GIT_LOG="$git_log" \
    FAKE_MLX_AVAILABLE=no \
    FAKE_MLX_REVISION="$revision" \
    FAKE_POLICY_LOG="$policy_log" \
    FAKE_PSLP_MISSING=yes \
    FAKE_SOURCE_DIR="$source_dir" \
    MLXPDLP_BUILD_DIR="$case_dir/build" \
    MLXPDLP_DEPS_DIR="$case_dir/deps" \
    MLXPDLP_FETCH_DEPS=ON \
    MLXPDLP_MLX_REVISION="$revision" \
    MLXPDLP_SOURCE_DIR="$source_dir" \
    sh "$bootstrap" > "$case_dir/output" 2>&1
[ -f "$case_dir/deps/mlx.incomplete/partial-checkout" ] ||
    fail "incomplete managed checkout was not preserved"
[ -f "$case_dir/deps/mlx.download.incomplete/partial-download" ] ||
    fail "interrupted staged download was not preserved"
[ -f "$case_dir/deps/mlx/CMakeLists.txt" ] ||
    fail "managed MLX source was not recovered"
[ -f "$case_dir/deps/mlx-install/.fake-installed" ] ||
    fail "recovered MLX was not installed"
assert_contains "$case_dir/output" "recovering the managed MLX source directory"
assert_contains "$case_dir/output" "recovering an interrupted MLX download"
assert_contains "$cmake_log" "$case_dir/deps/mlx-build --parallel 3"
assert_contains "$cmake_log" "-DCMAKE_OSX_DEPLOYMENT_TARGET:STRING=26.2"

new_case stale_target
mkdir -p \
    "$case_dir/build" \
    "$case_dir/deps/mlx/mlx" \
    "$case_dir/deps/mlx-install/include/mlx" \
    "$case_dir/deps/mlx-install/lib"
: > "$case_dir/deps/mlx/CMakeLists.txt"
: > "$case_dir/deps/mlx/mlx/mlx.h"
printf '%s\n' "$revision" > "$case_dir/deps/mlx/.fake-head"
: > "$case_dir/deps/mlx-install/.fake-installed"
: > "$case_dir/deps/mlx-install/include/mlx/mlx.h"
: > "$case_dir/deps/mlx-install/include/mlx/version.h"
: > "$case_dir/deps/mlx-install/lib/libmlx.a"
{
    printf '%s\n' "https://github.com/ml-explore/mlx.git"
    printf '%s\n' "$revision"
} > "$case_dir/deps/mlx-install/.mlxpdlp-managed-mlx"
{
    printf 'MLX_LIBRARY:FILEPATH=%s\n' \
        "$case_dir/deps/mlx-install/lib/libmlx.a"
    printf 'MLX_INCLUDE_DIR:PATH=%s\n' \
        "$case_dir/deps/mlx-install/include"
    printf 'MLX_GENERATED_INCLUDE_DIR:PATH=%s\n' \
        "$case_dir/deps/mlx-install/include"
} > "$case_dir/build/CMakeCache.txt"
env \
    PATH="$fake_bin:/usr/bin:/bin" \
    FAKE_CMAKE_LOG="$cmake_log" \
    FAKE_GIT_LOG="$git_log" \
    FAKE_MLX_AVAILABLE=no \
    FAKE_MLX_REVISION="$revision" \
    FAKE_POLICY_LOG="$policy_log" \
    FAKE_PSLP_MISSING=no \
    FAKE_SOURCE_DIR="$source_dir" \
    MLXPDLP_BUILD_DIR="$case_dir/build" \
    MLXPDLP_DEPS_DIR="$case_dir/deps" \
    MLXPDLP_FETCH_DEPS=ON \
    MLXPDLP_MLX_REVISION="$revision" \
    MLXPDLP_SOURCE_DIR="$source_dir" \
    sh "$bootstrap" > "$case_dir/output" 2>&1
assert_contains "$case_dir/output" "configuring MLX"
[ "$(sed -n '3p' "$case_dir/deps/mlx-install/.mlxpdlp-managed-mlx")" = 26.2 ] ||
    fail "managed MLX metadata did not record the deployment target"

new_case make_parallel
env \
    PATH="$fake_bin:/usr/bin:/bin" \
    FAKE_CMAKE_LOG="$cmake_log" \
    FAKE_GIT_LOG="$git_log" \
    FAKE_MLX_AVAILABLE=no \
    FAKE_MLX_REVISION="$revision" \
    FAKE_POLICY_LOG="$policy_log" \
    FAKE_PSLP_MISSING=no \
    FAKE_SOURCE_DIR="$source_dir" \
    MLXPDLP_BUILD_DIR="$case_dir/build" \
    MLXPDLP_DEPS_DIR="$case_dir/deps" \
    MLXPDLP_FETCH_DEPS=ON \
    MLXPDLP_MACOS_DEPLOYMENT_TARGET=14.0 \
    MLXPDLP_MAKE_PARALLEL_LEVEL=12 \
    MLXPDLP_MLX_REVISION="$revision" \
    MLXPDLP_SOURCE_DIR="$source_dir" \
    sh "$bootstrap" > "$case_dir/output" 2>&1
assert_contains "$case_dir/output" "building MLX with 12 parallel jobs"
assert_contains "$cmake_log" "$case_dir/deps/mlx-build --parallel 12"
assert_contains "$cmake_log" "-DCMAKE_OSX_DEPLOYMENT_TARGET:STRING=14.0"
assert_not_contains "$cmake_log" "-DCMAKE_OSX_DEPLOYMENT_TARGET:STRING=26.2"

new_case conflict
if env \
    PATH="$fake_bin:/usr/bin:/bin" \
    FAKE_CMAKE_LOG="$cmake_log" \
    FAKE_GIT_LOG="$git_log" \
    FAKE_MLX_AVAILABLE=yes \
    FAKE_MLX_REVISION="$revision" \
    FAKE_POLICY_LOG="$policy_log" \
    FAKE_SOURCE_DIR="$source_dir" \
    MLXPDLP_BUILD_DIR="$case_dir/build" \
    MLXPDLP_DEPS_DIR="$case_dir/deps" \
    MLXPDLP_FETCH_DEPS=OFF \
    MLXPDLP_FETCH_MLX=ON \
    MLXPDLP_MLX_REVISION="$revision" \
    MLXPDLP_SOURCE_DIR="$source_dir" \
    sh "$bootstrap" > "$case_dir/output" 2>&1; then
    fail "conflicting current and legacy consent unexpectedly succeeded"
fi
assert_contains "$case_dir/output" "MLXPDLP_FETCH_DEPS and MLXPDLP_FETCH_MLX disagree"

printf '%s\n' "bootstrap dependency policy tests passed"
