# Copyright (C) 2023 Intel Corporation
# Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
# See LICENSE.TXT
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

set(UMF_SOURCES
    src/memory_pool.c
    src/memory_provider.c
    src/memory_tracker.c
    src/memory_provider_get_last_failed.cpp
    src/critnib/critnib.c
)

if(MSVC)
    set(UMF_SOURCES ${UMF_SOURCES} src/utils/utils_windows.cpp src/memory_tracker_windows.cpp)
else()
    set(UMF_SOURCES ${UMF_SOURCES} src/utils/utils_posix.c)
endif()

if(UMF_BUILD_SHARED_LIBRARY)
    message(WARNING "Unified Malloc Framework is still an early work in progress."
                     "There are no API/ABI backward compatibility guarantees. There will be breakages."
                     "Do not use the shared library in production software.")
    add_ur_library(unified_malloc_framework SHARED
        ${UMF_SOURCES})
    target_compile_definitions(unified_malloc_framework PUBLIC UMF_SHARED_LIBRARY)
else()
    add_ur_library(unified_malloc_framework STATIC
        ${UMF_SOURCES})
endif()

if (UMF_ENABLE_POOL_TRACKING)
    target_sources(unified_malloc_framework PRIVATE src/memory_pool_tracking.c)
else()
    target_sources(unified_malloc_framework PRIVATE src/memory_pool_default.c)
endif()

add_library(${PROJECT_NAME}::unified_malloc_framework ALIAS unified_malloc_framework)

target_include_directories(unified_malloc_framework PUBLIC include)
