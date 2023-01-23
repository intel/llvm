# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: MIT

add_library(unified_memory_allocation STATIC
    src/memory_pool.c
    src/memory_provider.c
)

add_library(${PROJECT_NAME}::unified_memory_allocation ALIAS unified_memory_allocation)

target_include_directories(unified_memory_allocation PUBLIC include)
