# Copyright (C) Codeplay Software Limited

#===--- CMakeLists.txt -----------------------------------------------------===#
#
# MLIR-SYCL is under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
#===------------------------------------------------------------------------===#

cmake_minimum_required(VERSION 3.13.4)

# CMP0116: Ninja generators transform `DEPFILE`s from `add_custom_command()`
# New in CMake 3.20. https://cmake.org/cmake/help/latest/policy/CMP0116.html
if(POLICY CMP0116)
  cmake_policy(SET CMP0116 OLD)
endif()

set(LLVM_RUNTIME_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/bin)
set(LLVM_LIBRARY_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/lib)
set(MLIR_BINARY_DIR ${CMAKE_BINARY_DIR})

if (CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  project(sycl-dialect LANGUAGES CXX C)

  set(CMAKE_CXX_STANDARD 17 CACHE STRING "C++ standard to conform to")

  find_package(MLIR REQUIRED CONFIG)

  message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
else ()
  set(LLVM_SOURCE_DIR ${LLVM_MAIN_SRC_DIR})
  set(MLIR_MAIN_SRC_DIR ${LLVM_MAIN_SRC_DIR}/../mlir)
  set(MLIR_INCLUDE_DIRS ${MLIR_MAIN_SRC_DIR}/include)
  set(MLIR_CMAKE_DIR ${MLIR_MAIN_SRC_DIR}/cmake/modules)
  set(MLIR_TABLEGEN_EXE $<TARGET_FILE:mlir-tblgen>)
  set(PROJECT_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
  set(MLIR_TABLEGEN_OUTPUT_DIR ${LLVM_BINARY_DIR}/tools/mlir/include)
  set(SYCL_TABLEGEN_OUTPUT_DIR ${LLVM_BINARY_DIR}/tools/mlir-sycl/include)

  include_directories(${MLIR_TABLEGEN_OUTPUT_DIR})
  include_directories(${SYCL_TABLEGEN_OUTPUT_DIR})
endif ()

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}")

include(TableGen)
include(AddLLVM)
include(AddMLIR)

include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_BINARY_DIR}/include)

link_directories(${LLVM_BUILD_LIBRARY_DIR})

add_definitions(${LLVM_DEFINITIONS})

set(MLIR_SYCL_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(MLIR_SYCL_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR})
set(MLIR_SYCL_INCLUDE_DIR ${CMAKE_CURRENT_BINARY_DIR}/include )
set(MLIR_SYCL_TOOLS_DIR ${CMAKE_BINARY_DIR}/bin)

set(LLVM_LIT_ARGS "-sv" CACHE STRING "lit default options")
list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake/modules")

add_subdirectory(include/mlir)
add_subdirectory(lib)
add_subdirectory(test)
add_subdirectory(tools)

install(DIRECTORY include/
  DESTINATION include
  COMPONENT sycl-mlir-headers
  FILES_MATCHING
  PATTERN "*.def"
  PATTERN "*.h"
  PATTERN "*.inc"
  PATTERN "*.td"
  PATTERN "*.sv"
  PATTERN "LICENSE.TXT"
  )

install(DIRECTORY ${PROJECT_BINARY_DIR}/include/
  DESTINATION include
  COMPONENT sycl-mlir-headers
  FILES_MATCHING
  PATTERN "*.def"
  PATTERN "*.h"
  PATTERN "*.gen"
  PATTERN "*.inc"
  PATTERN "*.td"
  PATTERN "CMakeFiles" EXCLUDE
  PATTERN "config.h" EXCLUDE
  )
