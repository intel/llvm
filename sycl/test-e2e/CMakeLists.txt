cmake_minimum_required(VERSION 3.20.0)

message("Configuring SYCL End-to-End Tests")

option(SYCL_E2E_LIT_ALLOW_UNKNOWN_ARCH
  "Allow unknown architectures when configuring e2e tests" Off)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  project(sycl-e2e-test-suite CXX)
  set(SYCL_TEST_E2E_STANDALONE TRUE)
endif()

if(SYCL_TEST_E2E_STANDALONE)
  if( NOT OpenCL_LIBRARY )
    find_package(OpenCL)
  endif()
  if(NOT DEFINED LEVEL_ZERO_INCLUDE)
    find_path(LEVEL_ZERO_PATH level_zero/ze_api.h PATH_SUFFIXES include)
    if(LEVEL_ZERO_PATH)
      set(LEVEL_ZERO_INCLUDE "${LEVEL_ZERO_PATH}")
    endif()
  endif()
  if(NOT DEFINED LEVEL_ZERO_LIBS_DIR)
    find_library(LEVEL_ZERO_LIBS_PATH ze_loader)
    if(LEVEL_ZERO_LIBS_PATH)
      # We have the path of the lib, get the containing directory
      # so we can include it.
      cmake_path(GET LEVEL_ZERO_LIBS_PATH PARENT_PATH LEVEL_ZERO_LIBS_PATH)
      set(LEVEL_ZERO_LIBS_DIR "${LEVEL_ZERO_LIBS_PATH}")
    endif()
  endif()
else()
  if( NOT OpenCL_LIBRARY )
    set(OpenCL_LIBRARY "${LLVM_BINARY_DIR}/lib")
  endif()
  set(LEVEL_ZERO_INCLUDE "${LLVM_BINARY_DIR}/_deps/level-zero-loader-src/include")
  set(LEVEL_ZERO_LIBS_DIR "${LLVM_BINARY_DIR}/lib")
endif() # Standalone.

if(SYCL_TEST_E2E_STANDALONE)
  set(SYCL_CXX_COMPILER ${CMAKE_CXX_COMPILER})
else()
  set(SYCL_CXX_COMPILER "${LLVM_BINARY_DIR}/bin/clang++")
  # Don't want options used for building sycl-toolchain.
  unset(CMAKE_CXX_FLAGS)
endif() # Standalone.

find_package(Threads REQUIRED)
set(SYCL_THREADS_LIB ${CMAKE_THREAD_LIBS_INIT})

find_package(Vulkan)

if(NOT LLVM_LIT)
  find_program(LLVM_LIT
               NAMES llvm-lit.py llvm-lit lit.py lit
               PATHS "${LLVM_MAIN_SRC_DIR}/utils/lit"
               DOC "Path to lit.py")
endif()

set(SYCL_E2E_TESTS_LIT_FLAGS "-sv" CACHE STRING "Flags used when running lit")

find_package(Python3 REQUIRED COMPONENTS Interpreter)

if(NOT SYCL_TEST_E2E_TARGETS)
  set(SYCL_TEST_E2E_TARGETS "all")
endif()

if(MSVC AND NOT SYCL_TEST_E2E_STANDALONE)
  # We're trying to pass MSVC flags to Clang, which doesn't work by default
  separate_arguments(cxx_flags NATIVE_COMMAND "${CMAKE_CXX_FLAGS}")
  foreach(flag IN ITEMS ${cxx_flags})
    # Skip certain flags that only exists for MSVC
    if("${flag}" STREQUAL "/EHsc")
      continue()
    endif()
    # Change the way compiler definitions are passed in
    string(REGEX REPLACE "^/D" "-D" clang_flag "${flag}")
    list(APPEND SYCL_E2E_CLANG_CXX_FLAGS ${clang_flag})
  endforeach()
  string (REPLACE ";" " " SYCL_E2E_CLANG_CXX_FLAGS "${SYCL_E2E_CLANG_CXX_FLAGS}")
else()
  set(SYCL_E2E_CLANG_CXX_FLAGS ${CMAKE_CXX_FLAGS})
endif()

set(SYCL_E2E_CLANG_CXX_FLAGS "${SYCL_E2E_CLANG_CXX_FLAGS} -Werror")

if(NOT DEFINED CUDA_LIBS_DIR AND NOT DEFINED CUDA_INCLUDE)
  find_package(CUDAToolkit)

  if(CUDAToolkit_FOUND)
    set(CUDA_LIBS_DIR "${CUDAToolkit_LIBRARY_DIR}")
    set(CUDA_INCLUDE "${CUDAToolkit_INCLUDE_DIRS}")
  endif()
endif()

find_package(hip HINTS ENV ROCM_PATH "/opt/rocm")
if(hip_FOUND)
  set(HIP_LIBS_DIR "${hip_LIB_INSTALL_DIR}")
  set(HIP_INCLUDE "${hip_INCLUDE_DIRS}")
endif()

if(SYCL_TEST_E2E_STANDALONE)
  configure_file("${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in"
                 "${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py")
else()
  configure_lit_site_cfg(
    ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
    ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
    MAIN_CONFIG
    ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
    )
endif() # Standalone.

if(NOT SYCL_TEST_E2E_STANDALONE)
  list(APPEND SYCL_E2E_TEST_DEPS
    sycl-toolchain
    FileCheck
    not
  )
endif() # Standalone.

add_custom_target(check-sycl-e2e
  COMMAND ${Python3_EXECUTABLE} ${LLVM_LIT} ${SYCL_E2E_TESTS_LIT_FLAGS} .
  COMMENT "Running SYCL End-to-End tests"
  DEPENDS ${SYCL_E2E_TEST_DEPS}
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  USES_TERMINAL
)

add_subdirectory(External)
add_subdirectory(ExtraTests)
