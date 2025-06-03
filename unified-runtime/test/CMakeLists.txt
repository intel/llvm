# Copyright (C) 2022-2023 Intel Corporation
# Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
# See LICENSE.TXT
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

include(FetchContent)
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG        v1.13.0
)

include(FindLit)

set(UR_TEST_DEVICES_COUNT 1 CACHE STRING "Count of devices on which conformance and adapters tests will be run")
set(UR_TEST_PLATFORMS_COUNT 1 CACHE STRING "Count of platforms on which conformance and adapters tests will be run")
set(UR_TEST_FUZZTESTS ON CACHE BOOL "Run fuzz tests if using clang and UR_DPCXX is specified")
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang" AND UR_DPCXX AND UR_TEST_FUZZTESTS)
    set(UR_FUZZTESTING_ENABLED ON)
endif()

set(INSTALL_GTEST OFF)
FetchContent_MakeAvailable(googletest)
enable_testing()

# At the time of writing this comment, this is only used for level_zero adapter testing.
# It is found here for use in `lit.site.cfg.py.in`, which is inherited by said testing.
find_program(VALGRIND valgrind)

function(configure_ur_lit_site_cfg input output)
  if(UR_STANDALONE_BUILD)
    configure_file(${input} ${output} @ONLY)
  else()
    configure_lit_site_cfg(${input} ${output} ${ARGN})
  endif()
endfunction()

# Set up the root `check-unified-runtime` target
add_custom_target(deps_check-unified-runtime)
configure_ur_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
  MAIN_CONFIG
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py)
if(UR_STANDALONE_BUILD)
  add_custom_target(check-unified-runtime
    COMMAND "${URLIT_LIT_BINARY}" "${CMAKE_CURRENT_BINARY_DIR}" -sv
    DEPENDS deps_check-unified-runtime
    USES_TERMINAL
  )
else()
  # Use the LLVM method to add the test suite - this also registers it under `check-all`
  add_lit_testsuite(check-unified-runtime "Running Unified Runtime ${suite} tests"
    ${CMAKE_CURRENT_BINARY_DIR}
    DEPENDS FileCheck deps_check-unified-runtime
  )
endif()
if(UR_STANDALONE_BUILD)
  add_test(NAME lit COMMAND ${URLIT_LIT_BINARY} "${CMAKE_CURRENT_BINARY_DIR}")
endif()

# Create a new test suite that can be ran with `check-unified-runtime-${suite}`
# Lit will execute tests in `${CMAKE_CURRENT_BINARY_DIR}`, and any `add_gtest_test`s will be added as dependencies of
# this test suite
function(add_ur_lit_testsuite suite)
  cmake_parse_arguments(PARSE_ARGV 0 arg
    "" "" "DEPENDS"
  )
  set(TARGET "check-unified-runtime-${suite}")

  if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in)
    configure_ur_lit_site_cfg(
      ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
      ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
      MAIN_CONFIG
      ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py)
  endif()

  if(UR_STANDALONE_BUILD)
    add_custom_target(${TARGET}
      COMMAND "${URLIT_LIT_BINARY}" "${CMAKE_CURRENT_BINARY_DIR}" -sv
      USES_TERMINAL
    )
  else()
    add_lit_testsuite(${TARGET} "Running Unified Runtime ${suite} tests"
      ${CMAKE_CURRENT_BINARY_DIR}
      DEPENDS FileCheck
    )
  endif()

  add_custom_target(deps_${TARGET} DEPENDS "${arg_DEPENDS}")
  add_dependencies(${TARGET} deps_${TARGET})
  add_dependencies(deps_check-unified-runtime deps_${TARGET})
  set(TEST_CURRENT_SUITE "${TARGET}" PARENT_SCOPE)
endfunction()

function(add_testing_binary target)
    add_ur_executable(${target} ${ARGN})
    target_link_libraries(${target}
        PRIVATE
        ${PROJECT_NAME}::loader
        ${PROJECT_NAME}::headers
        ${PROJECT_NAME}::testing
        ${PROJECT_NAME}::mock
        ${PROJECT_NAME}::common)
    set_target_properties(${target}
        PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
    )
    add_dependencies(deps_${TEST_CURRENT_SUITE} ${target})
endfunction()

# Add a new gtest test; it will be added to the dependency list of the current `add_ur_lit_testsuite`
function(add_gtest_test name)
    set(TEST_TARGET_NAME ${name}-test)
    add_testing_binary(${TEST_TARGET_NAME} ${ARGN})
    target_link_libraries(${TEST_TARGET_NAME}
        PRIVATE
        gmock
        GTest::gtest_main)
endfunction()

# Conformance defines the generate_device_binaries target which should be
# imported first
add_subdirectory(conformance)

add_subdirectory(loader)
add_subdirectory(adapters)
add_subdirectory(usm)
add_subdirectory(layers)
add_subdirectory(logger)
add_subdirectory(unit)
if(UR_BUILD_TOOLS)
  add_subdirectory(tools)
endif()
if(UR_FUZZTESTING_ENABLED)
    add_subdirectory(fuzz)
endif()
