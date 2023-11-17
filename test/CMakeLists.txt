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

set(UR_TEST_DEVICES_COUNT 1 CACHE STRING "Count of devices on which conformance and adapters tests will be run")
set(UR_TEST_PLATFORMS_COUNT 1 CACHE STRING "Count of platforms on which conformance and adapters tests will be run")
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)
enable_testing()

add_subdirectory(loader)
add_subdirectory(adapters)
add_subdirectory(conformance)
add_subdirectory(usm)
add_subdirectory(layers)
add_subdirectory(unit)
if(UR_BUILD_TOOLS)
  add_subdirectory(tools)
endif()
if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang" AND UR_DPCXX)
    add_subdirectory(fuzz)
endif()
