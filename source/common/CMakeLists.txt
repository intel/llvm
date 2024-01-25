# Copyright (C) 2022-2023 Intel Corporation
# Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
# See LICENSE.TXT
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

add_subdirectory(unified_malloc_framework)
add_subdirectory(umf_pools)

add_ur_library(ur_common STATIC
    umf_helpers.hpp
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

target_link_libraries(ur_common PUBLIC
    unified_malloc_framework
    disjoint_pool
    ${CMAKE_DL_LIBS}
    ${PROJECT_NAME}::headers
)
