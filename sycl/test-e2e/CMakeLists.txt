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
