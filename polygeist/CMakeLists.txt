cmake_minimum_required(VERSION 3.13.4)

if(POLICY CMP0068)
  cmake_policy(SET CMP0068 NEW)
  set(CMAKE_BUILD_WITH_INSTALL_NAME_DIR ON)
endif()

if(POLICY CMP0075)
  cmake_policy(SET CMP0075 NEW)
endif()

if(POLICY CMP0077)
  cmake_policy(SET CMP0077 NEW)
endif()

option(LLVM_INCLUDE_TOOLS "Generate build targets for the LLVM tools." ON)
option(LLVM_BUILD_TOOLS "Build the LLVM tools. If OFF, just generate build targets." ON)

set(LLVM_RUNTIME_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/bin)
set(LLVM_LIBRARY_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/lib)
set(MLIR_BINARY_DIR ${CMAKE_BINARY_DIR})

if (CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  project(polygeist LANGUAGES CXX C)

  set(CMAKE_CXX_STANDARD 14 CACHE STRING "C++ standard to conform to")

  message(STATUS "Searching for MLIRConfig.cmake in: ${MLIR_DIR}")
  find_package(MLIR REQUIRED CONFIG)

  set(Clang_DIR ${CLANG_DIR})
  message(STATUS "Searching for ClangConfig.cmake in: ${Clang_DIR}")
  find_package(Clang REQUIRED CONFIG)

  # This is exported if we are building against a build area.  If
  # building against an install area, then assume we're using the
  # submodule.
  if(NOT LLVM_BUILD_MAIN_SRC_DIR)
    set(LLVM_BUILD_MAIN_SRC_DIR ${CMAKE_SOURCE_DIR}/llvm-project/llvm)
  endif()
  set(LLVM_SOURCE_DIR ${LLVM_BUILD_MAIN_SRC_DIR} CACHE STRING "Location of LLVM source")

  message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
  message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")
  message(STATUS "Using ClangConfig.cmake in: ${CLANG_DIR}")
else ()
  set(LLVM_SOURCE_DIR ${LLVM_MAIN_SRC_DIR})
  set(MLIR_MAIN_SRC_DIR ${LLVM_MAIN_SRC_DIR}/../mlir)
  set(MLIR_INCLUDE_DIRS ${MLIR_MAIN_SRC_DIR}/include)
  set(MLIR_CMAKE_DIR ${MLIR_MAIN_SRC_DIR}/cmake/modules)
  set(MLIR_TABLEGEN_EXE $<TARGET_FILE:mlir-tblgen>)
  set(MLIR_TABLEGEN_OUTPUT_DIR ${LLVM_BINARY_DIR}/tools/mlir/include)
  set(PROJECT_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
  set(POLYGEIST_TABLEGEN_OUTPUT_DIR ${LLVM_BINARY_DIR}/tools/polygeist/include)
  set(MLIR_SYCL_MAIN_SRC_DIR ${LLVM_MAIN_SRC_DIR}/../mlir-sycl)
  set(MLIR_SYCL_INCLUDE_DIRS ${MLIR_SYCL_MAIN_SRC_DIR}/include)
  set(MLIR_SYCL_CMAKE_DIR ${MLIR_SYCL_MAIN_SRC_DIR}/cmake/modules)
  set(MLIR_SYCL_TABLEGEN_OUTPUT_DIR ${LLVM_BINARY_DIR}/tools/mlir-sycl/include)

  include_directories(${MLIR_TABLEGEN_OUTPUT_DIR})
  include_directories(${POLYGEIST_TABLEGEN_OUTPUT_DIR})
  include_directories(${MLIR_SYCL_TABLEGEN_OUTPUT_DIR})
endif()

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${MLIR_SYCL_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${CLANG_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}")
include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(AddClang)
include(HandleLLVMOptions)

set(POLYGEIST_TOOLS_DIR ${CMAKE_BINARY_DIR})

include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${CLANG_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${MLIR_SYCL_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_BINARY_DIR}/include)
link_directories(${LLVM_BUILD_LIBRARY_DIR})
add_definitions(${LLVM_DEFINITIONS})

set(POLYGEIST_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(POLYGEIST_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR})
set(POLYGEIST_INCLUDE_DIR ${CMAKE_CURRENT_BINARY_DIR}/include )
set(POLYGEIST_TOOLS_DIR ${CMAKE_BINARY_DIR}/bin)
set(POLYGEIST_UTILS_DIR ${CMAKE_CURRENT_SOURCE_DIR}/utils)

set(LLVM_LIT_ARGS "-sv" CACHE STRING "lit default options")
list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake/modules")
include(sanitizers)

include(AddPolygeist)

add_subdirectory(include)
add_subdirectory(lib)
if (MLIR_INCLUDE_TESTS)
  add_definitions(-DMLIR_INCLUDE_TESTS)
  add_custom_target(MLIRPolygeistUnitTests)
  if (EXISTS ${LLVM_THIRD_PARTY_DIR}/unittest/googletest/include/gtest/gtest.h)
    add_subdirectory(unittests)
  else()
    message(WARNING "gtest not found, unittests will not be available")
  endif()
  add_subdirectory(test)
endif()
add_subdirectory(tools)
