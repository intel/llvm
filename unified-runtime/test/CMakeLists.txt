# Copyright (C) 2022-2023 Intel Corporation
# Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
# See LICENSE.TXT
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

include(FetchContent)
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG        release-1.12.1
)

include(FindLit)
function(add_ur_lit_testsuite target)
  configure_file(lit.site.cfg.py.in lit.site.cfg.py)

  if(UR_STANDALONE_BUILD)
    add_custom_target(check-${target}
      COMMAND "${URLIT_LIT_BINARY}" "${CMAKE_CURRENT_BINARY_DIR}" -v
    )
    add_test(NAME lit COMMAND ${URLIT_LIT_BINARY} "${CMAKE_CURRENT_BINARY_DIR}")
  else()
    add_lit_testsuite(check-${target} "Running Unified Runtime tests"
      ${CMAKE_CURRENT_BINARY_DIR}
      DEPENDS FileCheck
    )
  endif()
endfunction()

add_ur_lit_testsuite(unified-runtime)
add_dependencies(check-unified-runtime ur_adapter_mock)

set(UR_TEST_DEVICES_COUNT 1 CACHE STRING "Count of devices on which conformance and adapters tests will be run")
set(UR_TEST_PLATFORMS_COUNT 1 CACHE STRING "Count of platforms on which conformance and adapters tests will be run")
set(UR_TEST_FUZZTESTS ON CACHE BOOL "Run fuzz tests if using clang and UR_DPCXX is specified")
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
set(INSTALL_GTEST OFF)
FetchContent_MakeAvailable(googletest)
enable_testing()

# Conformance defines the generate_device_binaries target which should be
# imported first
add_subdirectory(conformance)

add_subdirectory(loader)
add_subdirectory(adapters)
add_subdirectory(usm)
add_subdirectory(layers)
add_subdirectory(unit)
add_subdirectory(mock)
if(UR_BUILD_TOOLS)
  add_subdirectory(tools)
endif()
if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang" AND UR_DPCXX AND UR_TEST_FUZZTESTS)
    add_subdirectory(fuzz)
endif()
