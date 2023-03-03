# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: MIT

set(UMA_SOURCES
    src/memory_pool.c
    src/memory_provider.c
    src/memory_tracker.cpp
)

if(UMA_BUILD_SHARED_LIBRARY)
    message(WARNING "Unified Memory Allocation is still an early work in progress."
                     "There are no API/ABI backward compatibility guarantees. There will be breakages."
                     "Do not use the shared library in production software.")
    add_library(unified_memory_allocation SHARED
        ${UMA_SOURCES})
else()
    add_library(unified_memory_allocation STATIC
        ${UMA_SOURCES})
endif()

add_library(${PROJECT_NAME}::unified_memory_allocation ALIAS unified_memory_allocation)

target_include_directories(unified_memory_allocation PUBLIC include)
