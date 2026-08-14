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

.DEFAULT_GOAL := build

CMAKE ?= cmake
CTEST ?= ctest
BUILD_DIR ?= build
CMAKE_ARGS ?=
MLXPDLP_MACOS_DEPLOYMENT_TARGET ?=
# GNU Make 3.81 finishes populating MAKEFLAGS only after reading this file, so
# these variables stay recursively expanded until a recipe is launched.
MLXPDLP_MAKE_PARALLEL = $(strip \
    $(filter -j% --jobs=% --jobserver-fds=% --jobserver-auth=%,$(MAKEFLAGS)) \
    $(findstring j,$(firstword $(MAKEFLAGS))))
MLXPDLP_DEFAULT_PARALLEL_LEVEL := 3
MLXPDLP_LOCAL_CPU_COUNT ?= $(strip $(shell getconf _NPROCESSORS_ONLN 2>/dev/null))
MLXPDLP_DETECTED_PARALLEL_LEVEL = $(if $(MLXPDLP_LOCAL_CPU_COUNT),$(MLXPDLP_LOCAL_CPU_COUNT),$(MLXPDLP_DEFAULT_PARALLEL_LEVEL))
MLXPDLP_MAKE_PARALLEL_LEVEL = $(if $(MLXPDLP_MAKE_PARALLEL),$(MLXPDLP_DETECTED_PARALLEL_LEVEL),)
CMAKE_BUILD_PARALLEL_LEVEL ?=
MLXPDLP_REQUESTED_PARALLEL_LEVEL = $(if $(CMAKE_BUILD_PARALLEL_LEVEL),$(CMAKE_BUILD_PARALLEL_LEVEL),$(MLXPDLP_MAKE_PARALLEL_LEVEL))
MLXPDLP_EFFECTIVE_PARALLEL_LEVEL = $(if $(MLXPDLP_REQUESTED_PARALLEL_LEVEL),$(MLXPDLP_REQUESTED_PARALLEL_LEVEL),$(MLXPDLP_DEFAULT_PARALLEL_LEVEL))
CMAKE_BUILD_ARGS ?= --parallel $(MLXPDLP_EFFECTIVE_PARALLEL_LEVEL)
CMAKE_TEST_ARGS ?= --output-on-failure
CMAKE_INSTALL_ARGS ?=
# MLXPDLP_FETCH_MLX is retained as a compatibility alias. Leave both
# variables empty here so the bootstrap can distinguish an explicit value;
# it applies the default of "ask" itself.
MLXPDLP_FETCH_DEPS ?=
MLXPDLP_FETCH_MLX ?=
MLXPDLP_MLX_CMAKE_ARGS ?=
MLXPDLP_DEPS_DIR ?= $(MLXPDLP_SOURCE_DIR)/_deps

MLXPDLP_SOURCE_DIR := $(patsubst %/,%,$(dir $(abspath $(lastword $(MAKEFILE_LIST)))))
ifeq ($(filter /%,$(BUILD_DIR)),)
MLXPDLP_BUILD_DIR := $(MLXPDLP_SOURCE_DIR)/$(BUILD_DIR)
else
MLXPDLP_BUILD_DIR := $(BUILD_DIR)
endif

export CMAKE
export CMAKE_ARGS
export CMAKE_BUILD_PARALLEL_LEVEL
export MLX_BUILD_DIR
export MLX_ROOT
export MLX_SOURCE_DIR
export MLXPDLP_BUILD_DIR
export MLXPDLP_DEPS_DIR
export MLXPDLP_FETCH_DEPS
export MLXPDLP_FETCH_MLX
export MLXPDLP_MAKE_PARALLEL_LEVEL
export MLXPDLP_MACOS_DEPLOYMENT_TARGET
export MLXPDLP_MLX_CMAKE_ARGS
export MLXPDLP_MLX_REPOSITORY
export MLXPDLP_MLX_REVISION
export MLXPDLP_SOURCE_DIR

.PHONY: build configure help install test

configure:
	@sh "$(MLXPDLP_SOURCE_DIR)/cmake/bootstrap_mlx.sh"

build: configure
	@MAKEFLAGS= $(CMAKE) --build "$(MLXPDLP_BUILD_DIR)" $(CMAKE_BUILD_ARGS)

test: build
	@$(CTEST) --test-dir "$(MLXPDLP_BUILD_DIR)" $(CMAKE_TEST_ARGS)

install: build
	@$(CMAKE) --install "$(MLXPDLP_BUILD_DIR)" $(CMAKE_INSTALL_ARGS)

help:
	@echo "mlxPDLP source build"
	@echo
	@echo "  make              Find or obtain MLX, then configure and build mlxPDLP"
	@echo "  make -j           Use all detected logical CPUs for compilation"
	@echo "  make test         Build and run CTest"
	@echo "  make install      Build and run cmake --install"
	@echo
	@echo "After make:"
	@echo "  ./build/mlxpdlp_example             Solve a trivial LP on Metal"
	@echo "  ./build/mlxpdlp_tiny_convergence    Sweep an exact LP on CPU/Metal"
	@echo "  ./build/mlxpdlp_netlib_convergence  Sweep Netlib ADLITTLE on CPU/Metal"
	@echo "  ./build/mlxpdlp_metal_acceleration  Compare generated CPU/Metal work"
	@echo
	@echo "Useful variables:"
	@echo "  MLX_ROOT=/path                Installed MLX prefix"
	@echo "  MLX_SOURCE_DIR=/path          MLX source tree"
	@echo "  MLX_BUILD_DIR=/path           Existing MLX build tree"
	@echo "  MLXPDLP_FETCH_DEPS=ON|OFF|ask Download consent for all dependencies"
	@echo "                                  (default: ask)"
	@echo "  MLXPDLP_FETCH_MLX=...          Deprecated alias for FETCH_DEPS"
	@echo "  MLXPDLP_MLX_CMAKE_ARGS='-D...' Extra managed-MLX configure arguments"
	@echo "  MLXPDLP_MACOS_DEPLOYMENT_TARGET=N"
	@echo "                                  macOS target for managed MLX and mlxPDLP"
	@echo "                                  (managed-build default: 14.0)"
	@echo "  BUILD_DIR=/path               mlxPDLP build directory (default: build)"
	@echo "  CMAKE_ARGS='-D...'            Extra mlxPDLP configure arguments"
	@echo "  CMAKE_BUILD_PARALLEL_LEVEL=N  Explicit compile-job override"
	@echo "                                  (plain make default: 3)"
