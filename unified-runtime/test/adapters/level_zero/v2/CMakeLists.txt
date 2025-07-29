# Copyright (C) 2024 Intel Corporation
# Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
# See LICENSE.TXT
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

function(add_l0_v2_devices_test test)
    add_conformance_devices_test(${test} ${ARGN})

    target_link_libraries(${test}-test PRIVATE
        ${PROJECT_NAME}::umf
        LevelZeroLoader
        LevelZeroLoader-Headers
        ComputeRuntimeLevelZero-Headers
    )

    target_include_directories(${test}-test PRIVATE
        ${PROJECT_SOURCE_DIR}/source
        ${PROJECT_SOURCE_DIR}/source/adapters
        ${PROJECT_SOURCE_DIR}/source/adapters/level_zero/v2
        LevelZeroLoader-Headers
    )
endfunction()

function(add_l0_v2_kernels_test test)
    add_conformance_kernels_test(${test} ${ARGN})

    target_link_libraries(${test}-test PRIVATE
        ${PROJECT_NAME}::umf
        LevelZeroLoader
        LevelZeroLoader-Headers
        ComputeRuntimeLevelZero-Headers
    )

    target_include_directories(${test}-test PRIVATE
        ${PROJECT_SOURCE_DIR}/source
        ${PROJECT_SOURCE_DIR}/source/adapters
        ${PROJECT_SOURCE_DIR}/source/adapters/level_zero/v2
        LevelZeroLoader-Headers
    )
endfunction()

add_l0_v2_devices_test(command_list_cache
    command_list_cache_test.cpp
    ${PROJECT_SOURCE_DIR}/source/adapters/level_zero/common.cpp
    ${PROJECT_SOURCE_DIR}/source/adapters/level_zero/ur_level_zero.cpp
    ${PROJECT_SOURCE_DIR}/source/adapters/level_zero/v2/command_list_cache.cpp
)

add_l0_v2_devices_test(event_pool
    event_pool_test.cpp
    ${PROJECT_SOURCE_DIR}/source/adapters/level_zero/adapter.cpp
    ${PROJECT_SOURCE_DIR}/source/adapters/level_zero/common.cpp
    ${PROJECT_SOURCE_DIR}/source/adapters/level_zero/device.cpp
    ${PROJECT_SOURCE_DIR}/source/adapters/level_zero/platform.cpp
    ${PROJECT_SOURCE_DIR}/source/adapters/level_zero/ur_level_zero.cpp
    ${PROJECT_SOURCE_DIR}/source/adapters/level_zero/v2/event_pool_cache.cpp
    ${PROJECT_SOURCE_DIR}/source/adapters/level_zero/v2/event_pool.cpp
    ${PROJECT_SOURCE_DIR}/source/adapters/level_zero/v2/event_provider_counter.cpp
    ${PROJECT_SOURCE_DIR}/source/adapters/level_zero/v2/event_provider_normal.cpp
    ${PROJECT_SOURCE_DIR}/source/adapters/level_zero/v2/event.cpp
    ${PROJECT_SOURCE_DIR}/source/ur/ur.cpp
)

add_l0_v2_devices_test(memory_residency
    memory_residency.cpp
    ${PROJECT_SOURCE_DIR}/source/adapters/level_zero/common.cpp
    ${PROJECT_SOURCE_DIR}/source/adapters/level_zero/ur_level_zero.cpp
)

add_l0_v2_kernels_test(deferred_kernel
    deferred_kernel.cpp
    ${PROJECT_SOURCE_DIR}/source/adapters/level_zero/common.cpp
    ${PROJECT_SOURCE_DIR}/source/adapters/level_zero/ur_level_zero.cpp
)
