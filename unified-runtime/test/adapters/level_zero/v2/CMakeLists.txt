# Copyright (C) 2024 Intel Corporation
# Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
# See LICENSE.TXT
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

function(add_unittest name)
    set(target test-adapter-${name})
    add_adapter_test(${name}
        FIXTURE DEVICES
        ENVIRONMENT
            "UR_ADAPTERS_FORCE_LOAD=\"$<TARGET_FILE:ur_adapter_level_zero_v2>\""
        SOURCES
            ${PROJECT_SOURCE_DIR}/source/adapters/level_zero/common.cpp
            ${PROJECT_SOURCE_DIR}/source/adapters/level_zero/ur_level_zero.cpp
            ${ARGN})

    target_include_directories(${target} PUBLIC
        ${PROJECT_SOURCE_DIR}/source
        ${PROJECT_SOURCE_DIR}/source/adapters
        ${PROJECT_SOURCE_DIR}/source/adapters/level_zero/v2
        LevelZeroLoader-Headers)

    target_link_libraries(${target} PRIVATE
        ${PROJECT_NAME}::common
        ${PROJECT_NAME}::umf
        LevelZeroLoader
        LevelZeroLoader-Headers
        ComputeRuntimeLevelZero-Headers
    )
endfunction()

add_unittest(level_zero_command_list_cache
        command_list_cache_test.cpp
        ${PROJECT_SOURCE_DIR}/source/adapters/level_zero/v2/command_list_cache.cpp
)

if(CXX_HAS_CFI_SANITIZE)
    message(WARNING "Level Zero V2 Event Pool tests are disabled when using CFI sanitizer")
    message(NOTE "See https://github.com/oneapi-src/unified-runtime/issues/2324")
else()
    add_unittest(level_zero_event_pool
            event_pool_test.cpp
            ${PROJECT_SOURCE_DIR}/source/ur/ur.cpp
            ${PROJECT_SOURCE_DIR}/source/adapters/level_zero/adapter.cpp
            ${PROJECT_SOURCE_DIR}/source/adapters/level_zero/device.cpp
            ${PROJECT_SOURCE_DIR}/source/adapters/level_zero/platform.cpp
            ${PROJECT_SOURCE_DIR}/source/adapters/level_zero/v2/event_pool.cpp
            ${PROJECT_SOURCE_DIR}/source/adapters/level_zero/v2/event_pool_cache.cpp
            ${PROJECT_SOURCE_DIR}/source/adapters/level_zero/v2/event_provider_normal.cpp
            ${PROJECT_SOURCE_DIR}/source/adapters/level_zero/v2/event_provider_counter.cpp
            ${PROJECT_SOURCE_DIR}/source/adapters/level_zero/v2/event.cpp
            ${PROJECT_SOURCE_DIR}/source/adapters/level_zero/v2/queue_api.cpp
    )
endif()

add_adapter_test(level_zero_memory_residency
    FIXTURE DEVICES
    SOURCES
        memory_residency.cpp
    ENVIRONMENT
        "UR_ADAPTERS_FORCE_LOAD=\"$<TARGET_FILE:ur_adapter_level_zero_v2>\""
        "ZES_ENABLE_SYSMAN=1"
)

if(NOT WIN32)
    if (NOT UR_DPCXX)
        # Tests that require kernels can't be used if we aren't generating
        # device binaries
        message(WARNING
            "UR_DPCXX is not defined, skipping some tests for level_zero_v2 adapter")
    else()
        add_adapter_test(level_zero_deferred_kernel
            FIXTURE KERNELS
            SOURCES
                deferred_kernel.cpp
            ENVIRONMENT
                "UR_ADAPTERS_FORCE_LOAD=\"$<TARGET_FILE:ur_adapter_level_zero_v2>\""
        )

        set(backend level_zero)
        add_adapter_memcheck_test(level_zero_deferred_kernel
            FIXTURE KERNELS
            SOURCES
                deferred_kernel.cpp
            ENVIRONMENT
                "UR_ADAPTERS_FORCE_LOAD=\"$<TARGET_FILE:ur_adapter_level_zero_v2>\""
        )

        target_link_libraries(test-adapter-level_zero_deferred_kernel PRIVATE
            LevelZeroLoader
            LevelZeroLoader-Headers
        )
    endif()
endif()
