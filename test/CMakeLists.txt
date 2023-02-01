# Copyright (C) 2022-2023 Intel Corporation
# SPDX-License-Identifier: MIT

include(FetchContent)
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG        release-1.12.1
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)
enable_testing()

add_subdirectory(python)
add_subdirectory(loader)
add_subdirectory(conformance)
add_subdirectory(unified_memory_allocation)
