# Copyright (C) 2022-2023 Intel Corporation
# Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
# See LICENSE.TXT
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

add_library(ur_common INTERFACE)
add_library(${PROJECT_NAME}::common ALIAS ur_common)

target_include_directories(ur_common INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/include
)

add_subdirectory(unified_memory_allocation)
add_subdirectory(uma_pools)
target_link_libraries(ur_common INTERFACE unified_memory_allocation disjoint_pool ${CMAKE_DL_LIBS} ${PROJECT_NAME}::headers)

if(WIN32)
    target_sources(ur_common
        INTERFACE
            ${CMAKE_CURRENT_SOURCE_DIR}/windows/ur_lib_loader.cpp
            uma_helpers.hpp ur_pool_manager.hpp
    )
else()
    target_sources(ur_common
        INTERFACE
            ${CMAKE_CURRENT_SOURCE_DIR}/linux/ur_lib_loader.cpp
            uma_helpers.hpp ur_pool_manager.hpp
    )
endif()
