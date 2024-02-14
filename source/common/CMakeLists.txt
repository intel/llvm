# Copyright (C) 2022-2023 Intel Corporation
# Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
# See LICENSE.TXT
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

add_ur_library(ur_common STATIC
    umf_helpers.hpp
    umf_pools/disjoint_pool_config_parser.cpp
    ur_pool_manager.hpp
    ur_util.cpp
    ur_util.hpp
    $<$<PLATFORM_ID:Windows>:windows/ur_lib_loader.cpp>
    $<$<PLATFORM_ID:Linux,Darwin>:linux/ur_lib_loader.cpp>
)
add_library(${PROJECT_NAME}::common ALIAS ur_common)

target_include_directories(ur_common PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/include
)

message(STATUS "Download Unified Memory Framework from github.com")

set(UMF_REPO "https://github.com/oneapi-src/unified-memory-framework.git")

# Merge pull request #119 from ldorau/Fix_arena_extent_split_and_arena_extent_merge
set(UMF_TAG 9bf7a0dc4dff76844e10edbb5c6e9d917536ef6d)

message(STATUS "Will fetch Unified Memory Framework from ${UMF_REPO}")

include(FetchContent)
FetchContent_Declare(unified-memory-framework
    GIT_REPOSITORY    ${UMF_REPO}
    GIT_TAG           ${UMF_TAG}
)

set(UMF_BUILD_LIBUMF_POOL_DISJOINT ON CACHE INTERNAL "Build Disjoint Pool")
set(UMF_BUILD_OS_MEMORY_PROVIDER OFF CACHE INTERNAL "Build OS Provider")

FetchContent_MakeAvailable(unified-memory-framework)
FetchContent_GetProperties(unified-memory-framework)

target_link_libraries(ur_common PUBLIC
    unified-memory-framework::umf
    unified-memory-framework::headers
    unified-memory-framework::disjoint_pool
    ${CMAKE_DL_LIBS}
    ${PROJECT_NAME}::headers
)
